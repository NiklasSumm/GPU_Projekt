#include <cub/cub.cuh>

#include <encodingBase.h>

template <int blockSize, int layer1Size, int layer2Size>
__global__ void
setupKernel88(int numElements, uint64_t *input)
{
    int print_id = 10;

	int iterations = ((int)(pow(2, layer1Size - 6) * pow(2, layer2Size)) + blockDim.x - 1) / blockDim.x;

	unsigned int aggregateSum = 0;
	unsigned int aggregate = 0;

	using BlockScan = cub::BlockScan<unsigned int, blockSize>;
	__shared__ typename BlockScan::TempStorage temp_storage;

	unsigned int elementId = 0;

	for (int i = 0; i < iterations; i++) {
		elementId = blockIdx.x * (int)(pow(2, layer1Size - 6) * pow(2, layer2Size)) + i * blockDim.x + threadIdx.x;

		// Load 64 bit bitmask section and count bits
		unsigned int original_data = 0;
		if (elementId < numElements) 
			original_data = __popcll(input[elementId]);
		unsigned int thread_data;

		// Collectively compute the block-wide inclusive sum
		BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, aggregate);

		// Every fourth thread writes value in first layer
		if ((threadIdx.x & (pow(2, layer1Size - 6) - 1) == 0) && (elementId < numElements)) {
			reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/(int)(pow(2, layer1Size - 6))] = static_cast<unsigned short>(thread_data + aggregateSum);
		}

		// Accumulate the aggregate for the next iteration of the loop 
		aggregateSum += aggregate;
	}

	// Last thread of each full block writes into layer 2
	if ((threadIdx.x == blockDim.x - 1) && (elementId < numElements)) {
		int offset = numElements*2 + ((numElements+3)/4 + 1)/2;
		reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = aggregateSum;
	}
}

__global__ void
apply88(int numPacked, int *permutation, int bitmaskSize, TreeStructure structure)
{
	int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementIdx < numPacked) {
		uint32_t bitsToFind = elementIdx+1;

		int nextLayerOffset = 0;
		int layerSize = structure.layerSizes[2];
		if (layerSize > 1) {
			uint32_t *layer2 = &structure.layers[2][0];

			// Index and step for binary search
			int searchIndex = layerSize / 2;
			int searchStep = (layerSize + 1) / 2;

			uint32_t layerSum = static_cast<uint32_t>(layer2[searchIndex]);

			while (searchStep > 1){
				searchStep = (searchStep + 1) / 2;
				searchIndex = (layerSum < bitsToFind) ? searchIndex + searchStep : searchIndex - searchStep;
				searchIndex = (searchIndex < 0) ? 0 : ((searchIndex < layerSize) ? searchIndex : layerSize - 1);
				layerSum = static_cast<uint32_t>(layer2[searchIndex]);
			}
			// After binary search we either landed on the correct value or the one above
			// So we have to check if the result is correct and if not go to the value below
			if ((layerSum >= bitsToFind) && (searchIndex > 0)){
				searchIndex--;
				layerSum = static_cast<uint32_t>(layer2[searchIndex]);
			}
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
			}
			nextLayerOffset *= 256;
		}

		// Handle layer 1
		layerSize = (structure.layerSizes[1] + structure.layerSizes[2] - 1) / structure.layerSizes[2];
		
		if (layerSize > 1) {
			layerSize = min(layerSize, structure.layerSizes[1] - nextLayerOffset);
			uint16_t *layer1 = &reinterpret_cast<uint16_t *>(structure.layers[1])[nextLayerOffset];

			// Index and step for binary search
			int searchIndex = layerSize / 2;
			int searchStep = (layerSize + 1) / 2;

			uint32_t layerSum = static_cast<uint32_t>(layer1[searchIndex]);

			while (searchStep > 1){
				searchStep = (searchStep + 1) / 2;
				searchIndex = (layerSum < bitsToFind) ? searchIndex + searchStep : searchIndex - searchStep;
				searchIndex = (searchIndex < 0) ? 0 : ((searchIndex < layerSize) ? searchIndex : layerSize - 1);
				layerSum = static_cast<uint32_t>(layer1[searchIndex]);
			}
			// After binary search we either landed on the correct value or the one above
			// So we have to check if the result is correct and if not go to the value below
			if ((layerSum >= bitsToFind) && (searchIndex > 0)){
				searchIndex--;
				layerSum = static_cast<uint32_t>(layer1[searchIndex]);
			}
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
			}
			nextLayerOffset *= 4;
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
		permutation[elementIdx] = expandedIndex;
	}
}

// layerSize calculates the amount of elements per layer, regardless of the actual size on disk.
// For the bitmask, we return the amount of integers.
template <int layer1Size, int layer2Size>
int layerSize88(int layer, int bitmaskSize) {
	int size = bitmaskSize * 2; // Bitmask size is size in long

	if (layer == 1){
		size = (bitmaskSize+ (int)pow(2, layer1Size - 6) - 1) / (int)pow(2, layer1Size - 6);
	}
	if (layer == 2){
		size = (bitmaskSize+((int)(pow(2, layer1Size - 6) * pow(2, layer2Size)) - 1)) / (int)(pow(2, layer1Size - 6) * pow(2, layer2Size));
	}

	return size;
}

template <int layer1Size, int layer2Size>
int layerOffsetInt88(int layer, int bitmaskSize) {
	if (layer == 0) return 0;

	int offset = 0;
	for (int i = 0; i < layer; i++) {
		int size = layerSize88<layer1Size, layer2Size>(i, bitmaskSize);
		if (i == 1) size = (size+1)/2;
		offset += size;
	}
	return offset;
}

template <int blockSize>
class Tree88 : public EncodingBase {
	private:
		uint64_t *d_bitmask;
		int n;
	
	public:
		void setup(uint64_t *d_bitmask, int n) {
			setupKernel88<blockSize,8,8><<<(n+1023)/1024, blockSize>>>(n, d_bitmask);

            int offset = n*2 + ((n+3)/4 + 1)/2;
            int size = (n+1023)/1024;
            uint32_t *startPtr = &reinterpret_cast<uint32_t*>(d_bitmask)[offset];

            // Determine temporary device storage requirements
            void *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            // Run exclusive prefix sum
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

			this->d_bitmask = d_bitmask;
			this->n = n;
		};

		void apply(int *permutation, int packedSize) {
			// TODO

			TreeStructure ts;

			uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
			for (int layer = 0; layer < 3; layer++) {
				ts.layers[layer] = &d_bitmask_int[layerOffsetInt88<8,8>(layer, n)];
				ts.layerSizes[layer] = layerSize88<8,8>(layer, n);
			}

			apply88<<<(packedSize+127)/128, 128>>>(packedSize, permutation, n, ts);
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
            int size = (n+3)/4;
            if (size < 500) {
                std::cout << "layer 1: ";
                for (int i = offset; i < offset+size; i++) {
                    std::cout << reinterpret_cast<unsigned short*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }

            // Print second layer layer (int)
            offset = offset / 2 + (size+1) / 2;
            size = (n+1023) / 1024;
            if (size < 500) {
                std::cout << "layer 2: ";
                for (int i = offset; i < offset+size; i++) {
                    std::cout << reinterpret_cast<unsigned int*>(h_bitmask)[i] << " ";
                }
                std::cout << std::endl;
            }
		};
};
