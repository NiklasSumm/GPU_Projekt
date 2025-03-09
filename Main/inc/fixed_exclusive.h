#include <cub/cub.cuh>
#include <encodingBase.h>


//layer1- and layer2Size as power of 2 => layer1Size = 8 means one value of layer 1 accumulates 2^8 bits from the bit mask
template <int blockSize, int longsPerLayer1Value, int longsPerLayer2Value>
__global__ void
setupKernelFixedExclusive(int numElements, uint64_t *input)
{
    // Number of longs per layer two entry devided by block size
    // Shift operators are used to get power of two
    // layer1Size - 6  since one long already contains 2^6 bit
	int iterations = (longsPerLayer2Value + blockDim.x - 1) / blockDim.x;

    using BlockScan = cub::BlockScan<unsigned int, blockSize>;
    __shared__ typename BlockScan::TempStorage temp_storage;

    BlockPrefixCallbackOp prefix_op(0);

    unsigned int elementId;
    unsigned int original_data;
    unsigned int thread_data;

    if (threadIdx.x < longsPerLayer2Value) //if there are more threads than needed, some wont perform any calculations
    {
	    for (int i = 0; i < iterations; i++) {
	    	elementId = blockIdx.x * longsPerLayer2Value + i * blockDim.x + threadIdx.x;

            // Load 64 bit bitmask section and count bits
            if (elementId < numElements)
                original_data = __popcll(input[elementId]);
            else
                original_data = 0;

            // Collectively compute the block-wide inclusive sum
            BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, prefix_op);
            __syncthreads();

	    	// Depending on layer size, every n-th thread writes to layer 1
	    	if (((threadIdx.x & (longsPerLayer1Value - 1)) == 0) && (elementId < numElements)) {
	    		reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/(longsPerLayer1Value)] = static_cast<unsigned short>(thread_data);
	    	}
        }

	    // Last active thread of each full block writes into layer 2
	    if (((threadIdx.x == blockDim.x - 1) || (threadIdx.x == (longsPerLayer2Value - 1))) && (elementId < numElements)) {
	    	int offset = numElements*2 + ((numElements+longsPerLayer1Value-1)/longsPerLayer1Value + 1)/2;
	    	reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = thread_data + original_data;
	    }
    }
}

template <int longsPerLayer1Value, int longsPerLayer2Value>
__global__ void
applyFixedExclusive(int numPacked, int *dst, int *src, int bitmaskSize, TreeStructure structure, bool unpack)
{
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elementIdx < numPacked) {
        uint32_t bitsToFind = elementIdx+1;

        // descend layer 2 using binary search
        int nextLayerOffset = 0;
        int layerSize = structure.layerSizes[2];
        if (layerSize > 1) {
            uint32_t *layer2 = &structure.layers[2][0];

            // to perform binary search on arbitrary layer size we assume size to be next largest power of 2
            // this way we only have to check if the search index exceeds the upper bound of the range,
            // while the lower bound (0) can't be exceeded
            int nextPowOf2 = getNextLargestPowerOf2(layerSize);

			int searchIndex = 0;
			int searchStep = nextPowOf2;

			while (searchStep > 1){
				searchStep >>= 1;
				int testIndex = min(searchIndex + searchStep, layerSize - 1);
				searchIndex += (static_cast<uint32_t>(layer2[testIndex]) < bitsToFind) * searchStep;
			}
			searchIndex = min(searchIndex, layerSize - 1);
			uint32_t layerSum = static_cast<uint32_t>(layer2[searchIndex]);
				
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
			}
			nextLayerOffset *= (longsPerLayer2Value / longsPerLayer1Value);
		}

        // Handle layer 1
        // descend layer 1 using binary search
        layerSize = (structure.layerSizes[1] + structure.layerSizes[2] - 1) / structure.layerSizes[2];

        if (layerSize > 1) {
            layerSize = min(layerSize, structure.layerSizes[1] - nextLayerOffset);
            uint16_t *layer1 = &reinterpret_cast<uint16_t *>(structure.layers[1])[nextLayerOffset];

            // again for the binary search we asume size to be next largest power of 2 and then check if upper bound is exceeded
            int nextPowOf2 = getNextLargestPowerOf2(layerSize);

			int searchIndex = 0;
			int searchStep = nextPowOf2;

			while (searchStep > 1){
				searchStep >>= 1;
				int testIndex = min(searchIndex + searchStep, layerSize - 1);
				searchIndex += (static_cast<uint32_t>(layer1[testIndex]) < bitsToFind) * searchStep;
			}
			searchIndex = min(searchIndex, layerSize - 1);
			uint32_t layerSum = static_cast<uint32_t>(layer1[searchIndex]);
				
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
			}
			nextLayerOffset *= longsPerLayer1Value;;
		}

        // Handle virtual layer 0 (before bitmask)
        uint64_t bitmaskSection;
        layerSize = structure.layerSizes[0]/2 - nextLayerOffset;
        //layerSize = min(layerSize, 32);
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
template <int longsPerLayer1Value, int longsPerLayer2Value>
int layerSizeFixedExclusive(int layer, int bitmaskSize) {
    int size = bitmaskSize * 2; // Bitmask size is size in long

	if (layer == 1){
		size = (bitmaskSize+ longsPerLayer1Value - 1) / longsPerLayer1Value;
	}
	if (layer == 2){
		size = (bitmaskSize+(longsPerLayer2Value - 1)) / (longsPerLayer2Value);
	}

    return size;
}

template <int longsPerLayer1Value, int longsPerLayer2Value>
int layerOffsetIntFixedExclusive(int layer, int bitmaskSize) {
    if (layer == 0) return 0;

    int offset = 0;
    for (int i = 0; i < layer; i++) {
        int size = layerSizeFixedExclusive<longsPerLayer1Value, longsPerLayer2Value>(i, bitmaskSize);
        if (i == 1) size = (size+1)/2;
        offset += size;
    }
    return offset;
}

template <int blockSize, int layer1Size, int layer2Size>
class FixedExclusive : public EncodingBase {
    const int longsPerLayer2Value = 1 << (layer2Size + layer1Size - 6);
    const int longsPerLayer1Value = 1 << (layer1Size - 6);

	private:
		uint64_t *d_bitmask;
		int n;

        // Used for setup, cached between multiple calls
        bool init = false; // Track if we initialised already
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
	
	public:
		void setup(uint64_t *d_bitmask, int n) {
            // gridSize = n devided by number of longs each block handles
            int gridSize = (n + longsPerLayer2Value - 1) / longsPerLayer2Value;

            setupKernelFixedExclusive<blockSize,longsPerLayer1Value,longsPerLayer2Value><<<gridSize, blockSize>>>(n, d_bitmask);

            // offset for layer 2 in number of ints => bitmask size + layer 1 size
            // bitmarks size n * 2 since n is number of longs
            // + 
            // n devided by number of longs per layer 1 entry devided by 2 (-> since layer 1 is in shorts)
	        int offset = n*2 + ((n+longsPerLayer1Value-1)/longsPerLayer1Value + 1)/2;
            int size = gridSize;
            uint32_t *startPtr = &reinterpret_cast<uint32_t*>(d_bitmask)[offset];

            // Determine temporary device storage requirements
            if (!init) {
                cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

                // Allocate temporary storage
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                init = true;
            }

            // Run exclusive prefix sum
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

            this->d_bitmask = d_bitmask;
            this->n = n;
        };

        void apply(int *permutation, int packedSize) {
            TreeStructure ts;

            uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
            for (int layer = 0; layer < 3; layer++) {
                ts.layers[layer] = &d_bitmask_int[layerOffsetIntFixedExclusive<longsPerLayer1Value,longsPerLayer2Value>(layer, n)];
                ts.layerSizes[layer] = layerSizeFixedExclusive<longsPerLayer1Value,longsPerLayer2Value>(layer, n);
            }

            applyFixedExclusive<longsPerLayer1Value, longsPerLayer2Value><<<(packedSize+blockSize-1)/blockSize, blockSize>>>(packedSize, permutation, nullptr, n, ts, false);
        };

        void pack(int *src, int *dst, int packedSize) {
            TreeStructure ts;

            uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
            for (int layer = 0; layer < 3; layer++) {
                ts.layers[layer] = &d_bitmask_int[layerOffsetIntFixedExclusive<longsPerLayer1Value, longsPerLayer2Value>(layer, n)];
                ts.layerSizes[layer] = layerSizeFixedExclusive<longsPerLayer1Value, longsPerLayer2Value>(layer, n);
            }

            applyFixedExclusive<longsPerLayer1Value, longsPerLayer2Value><<<(packedSize+blockSize-1)/blockSize, blockSize>>>(packedSize, dst, src, n, ts, false);
        };

        void unpack(int *src, int *dst, int packedSize) {
            TreeStructure ts;

            uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
            for (int layer = 0; layer < 3; layer++) {
                ts.layers[layer] = &d_bitmask_int[layerOffsetIntFixedExclusive<longsPerLayer1Value, longsPerLayer2Value>(layer, n)];
                ts.layerSizes[layer] = layerSizeFixedExclusive<longsPerLayer1Value, longsPerLayer2Value>(layer, n);
            }

            applyFixedExclusive<longsPerLayer1Value, longsPerLayer2Value><<<(packedSize+blockSize-1)/blockSize, blockSize>>>(packedSize, dst, src, n, ts, true);
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
            int size = (n + longsPerLayer1Value - 1) / longsPerLayer1Value;
            if (size < 500) {
                std::cout << "layer 1: ";
                for (int i = offset; i < offset+size; i++) {
                    std::cout << reinterpret_cast<unsigned short*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }

            // Print second layer layer (int)
            offset = offset / 2 + (size+1) / 2;
            size = (n + longsPerLayer2Value - 1) / longsPerLayer2Value;
            if (size < 500) {
                std::cout << "layer 2: ";
                for (int i = offset; i < offset+size; i++) {
                    std::cout << reinterpret_cast<unsigned int*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }
        };

        ~FixedExclusive() {
            if (init) {
                // Free temporary storage
                cudaFree(d_temp_storage);
            }
        }
};
