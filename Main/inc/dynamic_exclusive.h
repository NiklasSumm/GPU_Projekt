#include <cub/cub.cuh>
#include <block_prefix_callback_op.h>
#include <encodingBase.h>
#include <auxiliary_functions.h>


template <int blockSize>
__global__ void
setupKernel1(int numElements, long *input)
{
    int iterations = (1023 + blockDim.x) / blockDim.x;

    using BlockScan = cub::BlockScan<unsigned int, blockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockPrefixCallbackOp prefix_op(0);

    unsigned int elementId;
    unsigned int original_data;
    unsigned int thread_data;

    for (int i = 0; i < iterations; i++) {
        elementId = blockIdx.x * 1024 + i * blockDim.x + threadIdx.x;

        // Load 64 bit bitmask section and count bits
        if (elementId < numElements)
            original_data = __popcll(input[elementId]);
        else
            original_data = 0;

        // Collectively compute the block-wide exclusive sum
        BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, prefix_op);

        // First thread of each warp writes in layer 1
        if (((threadIdx.x & 31) == 0) && (elementId < numElements)) {
            reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/32] = static_cast<unsigned short>(thread_data);
        }
    }

    // Last thread of each full block writes into layer 2
    if ((threadIdx.x == blockDim.x - 1) && (elementId < numElements)) {
        int offset = numElements*2 + ((numElements+31)/32 + 1)/2;
        reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = thread_data + original_data;
    }
}

template <int blockSize>
__global__ void
setupKernel2(int numElements, unsigned int *input, bool next=true, bool nextButOne=true)
{
    int iterations = (1023 + blockDim.x) / blockDim.x;

    using BlockScan = cub::BlockScan<unsigned int, blockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockPrefixCallbackOp prefix_op(0);

    unsigned int elementId;
    unsigned int original_data;
    unsigned int thread_data;

    for (int i = 0; i < iterations; i++) {
        elementId = blockIdx.x * 1024 + i * blockDim.x + threadIdx.x;

        // Load prepared values from previous kernel
        if (elementId < numElements)
            original_data = input[elementId];
        else
            original_data = 0;

        // Collectively compute the block-wide exclusive sum over the prepared values
        BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, prefix_op);

        // The value in thread_data stored by the first thread in a warp needs to be subtracted
        // from the values each thread in the warp has, to get the correct values for the next value.
        // Doing it this way allows us to simultaneously prepare the next but one layer.
        unsigned int correction = thread_data;
        correction = __shfl_sync(0xffffffff, correction, 0);

        // Every thread needs to rewrite own value, to correct current layer
        if (elementId < numElements)
            input[elementId] = thread_data - correction;

        // First thread of each warp writes in next layer. These values are already fully correct.
        if (next && ((threadIdx.x & 31)) == 0 && (elementId < numElements)) {
            input[numElements+(elementId/32)] = thread_data;
        }
    }

    // Last thread of each full block writes into next but one layer. These values need to be corrected.
    if (nextButOne && (threadIdx.x == blockDim.x - 1) && (elementId < numElements)) {
        int offset = numElements + (numElements+31)/32;
        input[offset+blockIdx.x] = thread_data + original_data;
    }
}

__global__ void
applyDynamicExclusive(int numPacked, int *dst, int *src, int bitmaskSize, TreeStructure structure, bool unpack)
{
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elementIdx < numPacked) {
        uint32_t bitsToFind = elementIdx+1;

        int nextLayerOffset = 0;
        for (int layerIdx = 4; layerIdx > 1; layerIdx--) {
            int layerSize = structure.layerSizes[layerIdx] - nextLayerOffset;
            if (layerSize > 1) {
                layerSize = min(layerSize, 32);
                uint32_t *layer = &structure.layers[layerIdx][nextLayerOffset];

                int nextPowOf2 = getNextLargestPowerOf2(layerSize);

				int searchIndex = 0;
				int searchStep = nextPowOf2;

				while (searchStep > 1){
					searchStep >>= 1;
					int testIndex = min(searchIndex + searchStep, layerSize - 1);
					searchIndex += (static_cast<uint32_t>(layer[testIndex]) < bitsToFind) * searchStep;
				}
				searchIndex = min(searchIndex, layerSize - 1);
				uint32_t layerSum = static_cast<uint32_t>(layer[searchIndex]);
				
				if (layerSum < bitsToFind) {
					bitsToFind -= layerSum;
					nextLayerOffset += searchIndex;
				}
                nextLayerOffset *= 32;
            }
        }

        // Handle layer 1
        int layerSize = structure.layerSizes[1] - nextLayerOffset;
        if (layerSize > 1) {
            layerSize = min(layerSize, 32);
            uint16_t *layer = &reinterpret_cast<uint16_t *>(structure.layers[1])[nextLayerOffset];

            int nextPowOf2 = getNextLargestPowerOf2(layerSize);

			int searchIndex = 0;
			int searchStep = nextPowOf2; //cuda::std::bit_ceil<int>(layerSize);

			while (searchStep > 1){
				searchStep >>= 1;
				int testIndex = min(searchIndex + searchStep, layerSize - 1);
				searchIndex += (static_cast<uint32_t>(layer[testIndex]) < bitsToFind) * searchStep;
			}
			searchIndex = min(searchIndex, layerSize - 1);
			uint32_t layerSum = static_cast<uint32_t>(layer[searchIndex]);

			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
			}
            nextLayerOffset *= 32;
        }

        // Handle virtual layer 0 (before bitmask)
        uint64_t bitmaskSection;
        layerSize = structure.layerSizes[0]/2 - nextLayerOffset;
        layerSize = min(layerSize, 32);
        uint64_t *bitLayer = &reinterpret_cast<uint64_t *>(structure.layers[0])[nextLayerOffset];
        for (int i = 0; i < layerSize; i++) {
            bitmaskSection = bitLayer[i];
            int sectionSum = __popcll(bitmaskSection);
            if (bitsToFind <= sectionSum) break;
            bitsToFind -= sectionSum;
            nextLayerOffset++;
        }

        // Handle bitmask
        int expandedIndex = nextLayerOffset * 64;
        for (int k = sizeof(long)*8-1; k >= 0; k--) {
            if ((bitmaskSection >> k) & 1) {
                if (--bitsToFind == 0) {
                    expandedIndex += ((sizeof(long)*8-1)-k);
                    break;
                }
            }
        }
        if (src) {
            if (unpack) {
                dst[expandedIndex] = src[elementIdx]; // Unpack: Load from packed, write to expanded
            } else {
                dst[elementIdx] = src[expandedIndex]; // Pack: Load from expanded, write to packed
            }
        } else {
            dst[elementIdx] = expandedIndex; // Write permutation (used in apply)
        }
    }
}

__device__
void warpScan(unsigned mask, int tid, unsigned &val) {
    unsigned tmp;
    for (int d=1; d<32; d=2*d) {
        tmp = __shfl_up_sync(mask, val, d);
        if (tid%32 >= d) val += tmp;
    }
}

__device__
auto collabDescend(uint32_t ownSectionOffset, uint32_t ownSectionSize, int bitsToFind, int warpIdx, int sizeToCover) { // Returns {lane_id, ownBitsToFind}
    // Ballot vote
    unsigned ballotResult = __ballot_sync(0xffffffff, bitsToFind <= ownSectionOffset);

    // Determine warp lane that has correct bitmask section and fetch this section and the offset
    int warpLane = 31-__popc(ballotResult);
    uint32_t sectionOffset = __shfl_sync(0xffffffff, ownSectionOffset, warpLane);
    bitsToFind -= sectionOffset;

    // Determine if own value is in same section
    unsigned laneSize = __shfl_sync(0xffffffff, ownSectionSize, warpLane);
    int ownBitsToFind = bitsToFind+warpIdx;

    unsigned bitsCovered = laneSize - bitsToFind;
    int ownWarpLane = warpLane;
    bool diverged = false;
    while (bitsCovered <= sizeToCover) {
        diverged = true;
        warpLane++;
        if (bitsCovered < warpIdx) {
            ownBitsToFind -= laneSize;
            ownWarpLane = warpLane;
        }
        if (warpLane >= 31) break;

        laneSize = __shfl_sync(0xffffffff, ownSectionSize, warpLane);
        bitsCovered += laneSize;
    }
    struct result {int ownWarpLane; int ownBitsToFind; bool diverged;};
    return result {ownWarpLane, ownBitsToFind, diverged};
}

template <typename T>
__device__
bool traverseLayerCollab(uint32_t &bitsToFind, int &nextLayerOffset, T *layer, int layerSize, int sizeToCover, int warpIdx) {
    // Load section
    unsigned ownSectionOffset;
    if (layerSize > warpIdx) {
        ownSectionOffset = static_cast<uint32_t>(layer[warpIdx]);
    } else {
        ownSectionOffset = 4294967295; // Max uint32
    }
    unsigned ownSectionSize = __shfl_down_sync(0xffffffff, ownSectionOffset, 1) - ownSectionOffset;

    // Collab descend
    auto [ownWarpLane, ownBitsToFind, diverged] = collabDescend(ownSectionOffset, ownSectionSize, bitsToFind, warpIdx, sizeToCover);

    bitsToFind = ownBitsToFind-warpIdx;
    nextLayerOffset += ownWarpLane;
    return !diverged;
}

template <typename T>
__device__
void traverseLayerSolo(uint32_t &bitsToFind, int &nextLayerOffset, T *layer, int layerSize) {
    // Index and step for binary search
    int searchIndex = layerSize / 2;
    int searchStep = (layerSize + 1) / 2;

    uint32_t layerSum = static_cast<uint32_t>(layer[searchIndex]);

    while (searchStep > 1){
        searchStep = (searchStep + 1) / 2;
        searchIndex = (layerSum < bitsToFind) ? searchIndex + searchStep : searchIndex - searchStep;
        searchIndex = (searchIndex < 0) ? 0 : ((searchIndex < layerSize) ? searchIndex : layerSize - 1);
        layerSum = static_cast<uint32_t>(layer[searchIndex]);
    }
    // After binary search we either landed on the correct value or the one above
    // So we have to check if the result is correct and if not go to the value below
    if ((layerSum >= bitsToFind) && (searchIndex > 0)){
        searchIndex--;
        layerSum = static_cast<uint32_t>(layer[searchIndex]);
    }

    if (layerSum < bitsToFind) {
        bitsToFind -= layerSum;
        nextLayerOffset += searchIndex;
    }
}

__global__ void
applyDynamicExclusive_warpShuffle(int numPacked, int *dst, int *src, int bitmaskSize, TreeStructure structure, bool unpack)
{
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpStartIdx = blockIdx.x * blockDim.x + (threadIdx.x & 992);
    int warpIdx = threadIdx.x & 31;
    int sizeToCover = min(numPacked - warpStartIdx, 32);
    bool collab = true;

    if (warpStartIdx < numPacked) {
        uint32_t bitsToFind = warpStartIdx+1;

        int nextLayerOffset = 0;
        for (int layerIdx = 4; layerIdx > 1; layerIdx--) {
            int layerSize = structure.layerSizes[layerIdx] - nextLayerOffset;
            if (layerSize > 1) {
                layerSize = min(layerSize, 32);
                uint32_t *layer = &structure.layers[layerIdx][nextLayerOffset];

                if (collab) {
                    collab = traverseLayerCollab<uint32_t>(bitsToFind, nextLayerOffset, layer, layerSize, sizeToCover, warpIdx);
                    if (!collab) {
                        if (elementIdx >= numPacked) return; // Disable thread for solo descend
                        bitsToFind += warpIdx;
                    }
                } else {
                    traverseLayerSolo<uint32_t>(bitsToFind, nextLayerOffset, layer, layerSize);
                }
            }
            nextLayerOffset *= 32;
        }

        // Handle layer 1
        int layerSize = structure.layerSizes[1] - nextLayerOffset;
        if (layerSize > 1) {
            layerSize = min(layerSize, 32);
            uint16_t *layer = &reinterpret_cast<uint16_t *>(structure.layers[1])[nextLayerOffset];

            if (collab) {
                collab = traverseLayerCollab<uint16_t>(bitsToFind, nextLayerOffset, layer, layerSize, sizeToCover, warpIdx);
                if (!collab) {
                    if (elementIdx >= numPacked) return; // Disable thread for solo descend
                    bitsToFind += warpIdx;
                }
            } else {
                traverseLayerSolo<uint16_t>(bitsToFind, nextLayerOffset, layer, layerSize);
            }
        }
        nextLayerOffset *= 32;

        // Handle virtual layer 0 (before bitmask)
        layerSize = structure.layerSizes[0]/2 - nextLayerOffset;
        layerSize = min(layerSize, 32);
        uint64_t *bitLayer = &reinterpret_cast<uint64_t *>(structure.layers[0])[nextLayerOffset];

        // Collab descend
        uint64_t bitmaskSection;
        if (collab) {
            // Reduction with collaborative descend
            uint64_t ownBitmaskSection = bitLayer[warpIdx];
            unsigned ownSectionSize = __popcll(ownBitmaskSection);

            // Exclusive warp scan
            unsigned ownSectionOffset = ownSectionSize;
            warpScan(0xffffffff, warpIdx, ownSectionOffset); // Inclusive scan
            ownSectionOffset -= ownSectionSize; // Make exclusive

            int sizeToCover = min(numPacked - warpStartIdx, 32);
            auto [ownWarpLane, ownBitsToFind, diverged] = collabDescend(ownSectionOffset, ownSectionSize, bitsToFind, warpIdx, sizeToCover);

            // Switch to individual descend
            bitsToFind = ownBitsToFind;
            nextLayerOffset += ownWarpLane;
            bitmaskSection = __shfl_sync(0xffffffff, ownBitmaskSection, ownWarpLane);

            if (elementIdx >= numPacked) return; // Disable thread, since we now switch to solo descend
        } else {
            for (int i = 0; i < layerSize; i++) {
                bitmaskSection = bitLayer[i];
                int sectionSum = __popcll(bitmaskSection);
                if (bitsToFind <= sectionSum) break;
                bitsToFind -= sectionSum;
                nextLayerOffset++;
            }
        }

        // Handle bitmask
        int expandedIndex = nextLayerOffset * 64;
        for (int k = sizeof(long)*8-1; k >= 0; k--) {
            if ((bitmaskSection >> k) & 1) {
                if (--bitsToFind == 0) {
                    expandedIndex += ((sizeof(long)*8-1)-k);
                    break;
                }
            }
        }
        if (src) {
            if (unpack) {
                dst[expandedIndex] = src[elementIdx]; // Unpack: Load from packed, write to expanded
            } else {
                dst[elementIdx] = src[expandedIndex]; // Pack: Load from expanded, write to packed
            }
        } else {
            dst[elementIdx] = expandedIndex; // Write permutation (used in apply)
        }
    }
}

// layerSize calculates the amount of elements per layer, regardless of the actual size on disk.
// For the bitmask, we return the amount of integers.
int layerSize(int layer, int bitmaskSize) {
    int size = bitmaskSize * 2; // Bitmask size is size in long

    int factor = 1;
    for (int i = layer; i > 0; i--) {
        factor *= 32;
    }

    if (layer > 0) {
        size = (bitmaskSize+factor-1)/factor;
    }

    return size;
}

int layerOffsetInt(int layer, int bitmaskSize) {
    if (layer == 0) return 0;

    int offset = 0;
    for (int i = 0; i < layer; i++) {
        int size = layerSize(i, bitmaskSize);
        if (i == 1) size = (size+1)/2;
        offset += size;
    }
    return offset;
}

template <int blockSize>
class DynamicExclusive : public EncodingBase {
    private:
        uint64_t *d_bitmask;
        int n;
        bool collab;

        void applyImplementation(int *dst, int *src, int packedSize, bool unpack) {
            TreeStructure ts;

            uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
            for (int layer = 0; layer < 5; layer++) {
                ts.layers[layer] = &d_bitmask_int[layerOffsetInt(layer, n)];
                ts.layerSizes[layer] = layerSize(layer, n);
            }

            if (collab) {
                applyDynamicExclusive_warpShuffle<<<(packedSize+127)/128, 128>>>(packedSize, dst, src, n, ts, unpack);
            } else {
                applyDynamicExclusive<<<(packedSize+127)/128, 128>>>(packedSize, dst, src, n, ts, unpack);
            }
        }

    public:
        DynamicExclusive() : collab(true) {}
        DynamicExclusive(bool collab) : collab(collab) {}

        void setup(uint64_t *d_bitmask, int n) {
            //Calculationg layer sizes
            int size_layer2 = layerSize(2, n);
            int size_layer4 = layerSize(4, n);

            setupKernel1<blockSize><<<(n+1023)/1024, blockSize>>>(n, reinterpret_cast<long*>(d_bitmask));
            cudaDeviceSynchronize();

            if (layerSize(2, n) > 0) {
                int offset = layerOffsetInt(2, n);
                setupKernel2<blockSize><<<(size_layer2+1023)/1024, blockSize>>>(size_layer2, &reinterpret_cast<uint32_t*>(d_bitmask)[offset]);
                cudaDeviceSynchronize();
            }

            if (layerSize(4, n) > 0) {
                int offset = layerOffsetInt(4, n);
                setupKernel2<blockSize><<<(size_layer4+1023)/1024, blockSize>>>(size_layer4, &reinterpret_cast<uint32_t*>(d_bitmask)[offset], false, false);
                cudaDeviceSynchronize();
            }

            this->d_bitmask = d_bitmask;
            this->n = n;
        };

        void apply(int *permutation, int packedSize) {
            this->applyImplementation(permutation, nullptr, packedSize, false);
        };

        void pack(int *src, int *dst, int packedSize) {
            this->applyImplementation(dst, src, packedSize, false);
        };

        void unpack(int *src, int *dst, int packedSize) {
            this->applyImplementation(dst, src, packedSize, true);
        };

        void print(uint64_t *h_bitmask) {
            // Print bitmask
            if (n < 100) {
                std::cout << "bitmask: ";
                for (int i = 0; i < n; i++) {
                    uint64_t v = h_bitmask[i];
                    for (int k = sizeof(long)*8-1; k >= 0; k--) {
                        std::cout << ((v >> k) & 1) << "";
                    }
                    std::cout << " ";
                }
                std::cout << std::endl;
            }

            // Print first layer (short)
            if (layerSize(1, n) < 500) {
                int offset = layerOffsetInt(1, n) * 2;
                std::cout << "layer 1: ";
                for (int i = offset; i < offset+layerSize(1, n); i++) {
                    std::cout << reinterpret_cast<uint16_t*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }

            // Print second layer layer (int)
            uint32_t *intTree = reinterpret_cast<uint32_t*>(h_bitmask);
            if (layerSize(2, n) < 500) {
                int offset = layerOffsetInt(2, n);
                std::cout << "layer 2: ";
                for (int i = offset; i < offset+layerSize(2, n); i++) {
                    std::cout << intTree[i] << " ";
                }
                std::cout << std::endl;
            }

            // Print third layer layer (int)
            if (layerSize(3, n) < 500) {
                int offset = layerOffsetInt(3, n);
                std::cout << "layer 3: ";
                for (int i = offset; i < offset+layerSize(3, n); i++) {
                    std::cout << intTree[i] << " ";
                }
                std::cout << std::endl;
            }

            // Print fourth layer layer (int)
            int offset = layerOffsetInt(4, n);
            std::cout << "layer 4: ";
            for (int i = offset; i < offset+layerSize(4, n); i++) {
                std::cout << intTree[i] << " ";
            }
            std::cout << std::endl;
        };
};
