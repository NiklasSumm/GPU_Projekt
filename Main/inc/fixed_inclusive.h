#include <cub/cub.cuh>
#include <encodingBase.h>

template <int blockSize, int layer1Size, int layer2Size>
__global__ void
setupKernelFixedInclusive(int numElements, uint64_t *input)
{	
    int iterations = (((1 << (layer1Size - 6)) * (1 << layer2Size)) + blockDim.x - 1) / blockDim.x;

    using BlockScan = cub::BlockScan<unsigned int, blockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockPrefixCallbackOp prefix_op(0);

    unsigned int elementId;
    unsigned int thread_data;
    if (threadIdx.x < (1 << (layer1Size + layer2Size - 6))){
        for (int i = 0; i < iterations; i++) {
            elementId = blockIdx.x * ((1 << (layer1Size - 6)) * (1 << layer2Size)) + i * blockDim.x + threadIdx.x;

            // Load 64 bit bitmask section and count bits
            if (elementId < numElements)
                thread_data = __popcll(input[elementId]);
            else
                thread_data = 0;

            // Collectively compute the block-wide inclusive sum
            BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, prefix_op);
            __syncthreads();

            // Every second thread writes value in first layer
            if ((((threadIdx.x + 1) & ((1 << (layer1Size - 6)) - 1)) == 0) && (elementId < numElements)) {
                reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/(1 << (layer1Size - 6))] = static_cast<unsigned short>(thread_data);
            }
        }

        // Last thread of each full block writes into layer 2
        if (((threadIdx.x == blockDim.x - 1) || (threadIdx.x == ((1 << (layer1Size + layer2Size - 6))- 1))) && (elementId < numElements)) {
            int offset = numElements*2 + ((numElements+(1 << (layer1Size - 6))-1)/(1 << (layer1Size - 6)) + 1)/2;
            reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = thread_data;
        }
    }
}

template <int layer1Size, int layer2Size>
__global__ void
applyFixedInclusive(int numPacked, int *dst, int *src, int bitmaskSize, TreeStructure structure, bool unpack)
{	
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elementIdx < numPacked) {
        uint32_t bitsToFind = elementIdx+1;

        int nextLayerOffset = 0;
        int layerSize = structure.layerSizes[2];
        if (layerSize > 1) {
            uint32_t *layer2 = &structure.layers[2][0];

            int nextPowOf2 = getNextLargestPowerOf2(layerSize);

			int searchIndex = nextPowOf2;
			int searchStep = nextPowOf2;

			while (searchStep > 1){
				searchStep >>= 1;
				int testIndex = min(searchIndex - searchStep, layerSize - 1);
				searchIndex -= (static_cast<uint32_t>(layer2[testIndex]) >= bitsToFind) * searchStep;
			}
			searchIndex = min(searchIndex, layerSize - 1);
			
            uint32_t previousLayerSum = searchIndex > 0 ? static_cast<uint32_t>(layer2[searchIndex-1]) : 0;
			if (previousLayerSum < bitsToFind) {
				bitsToFind -= previousLayerSum;
				nextLayerOffset += searchIndex;
			}
            nextLayerOffset *= (1 << layer2Size);
        }

        // Handle layer 1
        layerSize = (structure.layerSizes[1] + structure.layerSizes[2] - 1) / structure.layerSizes[2];
        if (layerSize > 1) {
            layerSize = min(layerSize, structure.layerSizes[1] - nextLayerOffset);
            uint16_t *layer1 = &reinterpret_cast<uint16_t *>(structure.layers[1])[nextLayerOffset];

            int nextPowOf2 = getNextLargestPowerOf2(layerSize);

			int searchIndex = nextPowOf2;
			int searchStep = nextPowOf2;

			while (searchStep > 1){
				searchStep >>= 1;
				int testIndex = min(searchIndex - searchStep, layerSize - 1);
				searchIndex -= (static_cast<uint32_t>(layer1[testIndex]) >= bitsToFind) * searchStep;
			}
			searchIndex = min(searchIndex, layerSize - 1);
			
            uint32_t previousLayerSum = searchIndex > 0 ? static_cast<uint32_t>(layer1[searchIndex-1]) : 0;
			if (previousLayerSum < bitsToFind) {
				bitsToFind -= previousLayerSum;
				nextLayerOffset += searchIndex;
			}
            nextLayerOffset *= (1 << (layer1Size - 6));
        }

        // Handle virtual layer 0 (before bitmask)
        uint64_t bitmaskSection;
        layerSize = structure.layerSizes[0]/2 - nextLayerOffset;
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

// layerSize calculates the amount of elements per layer, regardless of the actual size on disk.
// For the bitmask, we return the amount of integers.
template <int layer1Size, int layer2Size>
int layerSizeFixedInclusive(int layer, int bitmaskSize) {
    int size = bitmaskSize * 2; // Bitmask size is size in long

    if (layer == 1){
        size = (bitmaskSize+ (1 << (layer1Size - 6)) - 1) / (1 << (layer1Size - 6));
    }
    if (layer == 2){
        size = (bitmaskSize+(((1 << (layer1Size - 6)) * (1 << layer2Size)) - 1)) / ((1 << (layer1Size - 6)) * (1 << layer2Size));
    }

    return size;
}

template <int layer1Size, int layer2Size>
int layerOffsetIntFixedInclusive(int layer, int bitmaskSize) {
    if (layer == 0) return 0;

    int offset = 0;
    for (int i = 0; i < layer; i++) {
        int size = layerSizeFixedInclusive<layer1Size,layer2Size>(i, bitmaskSize);
        if (i == 1) size = (size+1)/2;
        offset += size;
    }
    return offset;
}

template <int blockSize, int layer1Size, int layer2Size>
class FixedInclusive : public EncodingBase {
    private:
        uint64_t *d_bitmask;
        int n;

    void applyImplementation(int *dst, int *src, int packedSize, bool unpack) {
        TreeStructure ts;

        uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
        for (int layer = 0; layer < 3; layer++) {
            ts.layers[layer] = &d_bitmask_int[layerOffsetIntFixedInclusive<layer1Size,layer2Size>(layer, n)];
            ts.layerSizes[layer] = layerSizeFixedInclusive<layer1Size,layer2Size>(layer, n);
        }

        applyFixedInclusive<layer1Size,layer2Size><<<(packedSize+blockSize-1)/blockSize, blockSize>>>(packedSize, dst, src, n, ts, unpack);
    }

    public:
        void setup(uint64_t *d_bitmask, int n) {
            int gridSize = (n + ((1 << (layer1Size - 6)) * (1 << layer2Size)) - 1) / ((1 << (layer1Size - 6)) * (1 << layer2Size));

            setupKernelFixedInclusive<blockSize,layer1Size,layer2Size><<<gridSize, blockSize>>>(n, d_bitmask);

            int offset = n*2 + ((n+(1 << (layer1Size - 6))-1)/(1 << (layer1Size - 6)) + 1)/2;
            int size = gridSize;
            unsigned int *startPtr = &reinterpret_cast<unsigned int*>(d_bitmask)[offset];

            // Determine temporary device storage requirements
            void *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            // Run exclusive prefix sum
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

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
                    long v = h_bitmask[i];
                    for (int k = sizeof(long)*8-1; k >= 0; k--) {
                        std::cout << ((v >> k) & 1) << "";
                    }
                }
                std::cout << std::endl;
            }

            // Print first layer (short)
            int offset = n*4; // 4 shorts in one long
            int size = (n + (1 << (layer1Size - 6)) - 1) / (1 << (layer1Size - 6));
            if (size < 500) {
                std::cout << "layer 1: ";
                for (int i = offset; i < offset+size; i++) {
                    std::cout << reinterpret_cast<uint16_t*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }

            // Print second layer layer (int)
            offset = offset / 2 + (size+1) / 2;
            size = (n + (1 << (layer1Size - 6)) * (1 << layer2Size) - 1) / ((1 << (layer1Size - 6)) * (1 << layer2Size));
            if (size < 500) {
                std::cout << "layer 2: ";
                for (int i = offset; i < offset+size; i++) {
                    std::cout << reinterpret_cast<uint32_t*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }
        };
};
