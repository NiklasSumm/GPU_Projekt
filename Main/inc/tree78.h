#include <cub/cub.cuh>

#include <encodingBase.h>

template <int blockSize>
__global__ void
setupKernel78(int numElements, uint64_t *input)
{	
	int iterations = (511 + blockDim.x) / blockDim.x;

	unsigned int aggregateSum = 0;
	unsigned int aggregate = 0;

	using BlockScan = cub::BlockScan<unsigned int, blockSize>;
	__shared__ typename BlockScan::TempStorage temp_storage;

	unsigned int elementId = 0;

	for (int i = 0; i < iterations; i++) {
		elementId = blockIdx.x * 512 + i * blockDim.x + threadIdx.x;

		// Load 64 bit bitmask section and count bits
		unsigned int thread_data = 0;
		if (elementId < numElements)
			thread_data = __popcll(input[elementId]);

		// Collectively compute the block-wide inclusive sum
		BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, aggregate);

		// Every second thread writes value in first layer
		if (((threadIdx.x+1 & 1) == 0) && (elementId < numElements)) {
			reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/2] = static_cast<unsigned short>(thread_data + aggregateSum);
		}

		// Accumulate the aggregate for the next iteration of the loop 
		aggregateSum += aggregate;
	}

	// Last thread of each full block writes into layer 2
	if ((threadIdx.x == blockDim.x - 1) && (elementId < numElements)) {
		int offset = numElements*2 + ((numElements+1)/2 + 1)/2;
		reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = aggregateSum;
	}
}

__global__ void
apply78(int numPacked, int *permutation, int bitmaskSize, TreeStructure structure)
{
	int print_thread = 127;
	
	int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementIdx < numPacked) {
		uint32_t bitsToFind = elementIdx+1;

		int nextLayerOffset = 0;
		int layerSize = structure.layerSizes[2];
		if (layerSize > 1) {
			//layerSize = min(layerSize, 32);
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
			if ((layerSum > bitsToFind) && (searchIndex > 0)){
				searchIndex--;
				layerSum = static_cast<uint32_t>(layer2[searchIndex]);
			}
			
			if (layerSum <= bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
			}
			if (elementIdx == print_thread){
				printf("%i - %i - %i\n", searchIndex, layerSum, bitsToFind);
			}
			nextLayerOffset *= 32;
		}

		// Handle layer 1
		layerSize = structure.layerSizes[1] - nextLayerOffset;
		if (layerSize > 1) {
			//layerSize = min(layerSize, 32);
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
			if (elementIdx == print_thread){
				printf("%i - %i - %i\n", searchIndex, layerSum, bitsToFind);
			}
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				nextLayerOffset += searchIndex;
				if (elementIdx == print_thread) printf("layersum subtracted\n");
			}
			if (elementIdx == print_thread){
				printf("%i - %i - %i\n", searchIndex, layerSum, bitsToFind);
			}
			nextLayerOffset *= 32;
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
			if (elementIdx == print_thread){
				printf("iteration %i ---\n", i);
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
		permutation[elementIdx] = expandedIndex;
	}
}

// layerSize calculates the amount of elements per layer, regardless of the actual size on disk.
// For the bitmask, we return the amount of integers.
int layerSize78(int layer, int bitmaskSize) {
	int size = bitmaskSize * 2; // Bitmask size is size in long

	//int factor = 1;

	if (layer == 1){
		size = (bitmaskSize+1)/2;
	}
	if (layer == 2){
		size = (bitmaskSize+511)/512;
	}
	//for (int i = layer; i > 0; i--) {
	//	factor *= 32;
	//}
//
	//if (layer > 0) {
	//	size = (bitmaskSize+factor-1)/factor;
	//}

	return size;
}

int layerOffsetInt78(int layer, int bitmaskSize) {
	if (layer == 0) return 0;

	int offset = 0;
	for (int i = 0; i < layer; i++) {
		int size = layerSize78(i, bitmaskSize);
		if (i == 1) size = (size+1)/2;
		offset += size;
	}
	return offset;
}

class Tree78 : public EncodingBase {
	private:
		uint64_t *d_bitmask;
		int n;
	
	public:
		void setup(uint64_t *d_bitmask, int n) {
			setupKernel78<512><<<(n+511)/512, 512>>>(n, d_bitmask);

			int offset = n*2 + ((n+1)/2 + 1)/2;
			int size = (n+511)/512;
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
			// TODO

			TreeStructure ts;

			uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
			for (int layer = 0; layer < 3; layer++) {
				ts.layers[layer] = &d_bitmask_int[layerOffsetInt78(layer, n)];
				ts.layerSizes[layer] = layerSize78(layer, n);
			}

			apply78<<<(packedSize+127)/128, 128>>>(packedSize, permutation, n, ts);
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
			int size = (n+1)/2;
			if (size < 500) {
				std::cout << "layer 1: ";
				for (int i = offset; i < offset+size; i++) {
					std::cout << reinterpret_cast<uint16_t*>(h_bitmask)[i] << " ";
				}
				std::cout << std::endl;
			}

			// Print second layer layer (int)
			offset = offset / 2 + (size+1) / 2;
			size = (n+511) / 512;
			if (size < 500) {
				std::cout << "layer 2: ";
				for (int i = offset; i < offset+size; i++) {
					std::cout << reinterpret_cast<uint32_t*>(h_bitmask)[i] << " ";
				}
				std::cout << std::endl;
			}
		};
};
