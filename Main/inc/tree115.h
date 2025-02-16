#include <cub/cub.cuh>
#include <chTimer.hpp>

#include <encodingBase.h>


class TreeStructure {
	public:
		uint32_t *layers[5];
		int layerSizes[5];
};


template <int blockSize>
__global__ void
setupKernel1(int numElements, long *input)
{
	int iterations = (1023 + blockDim.x) / blockDim.x;

	unsigned int aggregateSum = 0;
	unsigned int aggregate = 0;

	using BlockScan = cub::BlockScan<unsigned int, blockSize>;
	__shared__ typename BlockScan::TempStorage temp_storage;

	unsigned int elementId = 0;
	for (int i = 0; i < iterations; i++) {
		elementId = blockIdx.x * 1024 + i * blockDim.x + threadIdx.x;

		// Load 64 bit bitmask section and count bits
		unsigned int original_data = 0;
		if (elementId < numElements) {
			original_data = __popcll(input[elementId]);
		}
		unsigned int thread_data;

		// Collectively compute the block-wide exclusive sum
		BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, aggregate);

		// First thread of each warp writes in layer 1
		if (((threadIdx.x & 31) == 0) && (elementId < numElements)) {
			reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/32] = static_cast<unsigned short>(thread_data + aggregateSum);
		}

		// Accumulate the aggregate for the next iteration of the loop 
		aggregateSum += aggregate;
	}

	// Last thread of each full block writes into layer 2
	if ((threadIdx.x == blockDim.x - 1) && (elementId < numElements)) {
		int offset = numElements*2 + ((numElements+31)/32 + 1)/2;
		reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = aggregateSum;
	}
}

template <int blockSize>
__global__ void
setupKernel2(int numElements, unsigned int *input, bool next=true, bool nextButOne=true)
{
	int iterations = (1023 + blockDim.x) / blockDim.x;

	unsigned int aggregateSum = 0;
	unsigned int aggregate = 0;

	using BlockScan = cub::BlockScan<unsigned int, blockSize>;
	__shared__ typename BlockScan::TempStorage temp_storage;

	unsigned int elementId = 0;
	for (int i = 0; i < iterations; i++) {
		elementId = blockIdx.x * 1024 + i * blockDim.x + threadIdx.x;

		// Load prepared values from previous kernel
		unsigned int original_data = 0;
		if (elementId < numElements) {
			original_data = input[elementId];
		}
		unsigned int thread_data;

		// Collectively compute the block-wide exclusive sum over the prepared values
		BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, aggregate);

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
			input[numElements+(elementId/32)] = thread_data + aggregateSum;
		}

		// Accumulate the aggregate for the next iteration of the loop 
		aggregateSum += aggregate;
	}

	// Last thread of each full block writes into next but one layer. These values need to be corrected.
	if (nextButOne && (threadIdx.x == blockDim.x - 1) && (elementId < numElements)) {
		int offset = numElements + (numElements+31)/32;
		input[offset+blockIdx.x] = aggregateSum;
	}
}

__global__ void
improvedApply(int numPacked, int *permutation, int bitmaskSize, TreeStructure structure)
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
				nextLayerOffset *= 32;
			}
		}

		// Handle layer 1
		int layerSize = structure.layerSizes[1] - nextLayerOffset;
		if (layerSize > 1) {
			layerSize = min(layerSize, 32);
			uint16_t *layer = &reinterpret_cast<uint16_t *>(structure.layers[1])[nextLayerOffset];

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
		permutation[elementIdx] = expandedIndex;
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

class Tree115 : public EncodingBase {
	private:
		uint64_t *d_bitmask;
		int n;
	
	public:
		void setup(uint64_t *d_bitmask, int n) {
			//Calculationg layer sizes
			int size_layer1 = layerSize(1, n);
			int size_layer2 = layerSize(2, n);
			int size_layer3 = layerSize(3, n);
			int size_layer4 = layerSize(4, n);

			printf("running first kernel...\n");
			ChTimer setupTimer;

			setupTimer.start();
			setupKernel1<1024><<<(n+1023)/1024, 1024>>>(n, reinterpret_cast<long*>(d_bitmask));
			setupTimer.stop();

			printf("first kernel ran for %f ms\n", 1e3 * setupTimer.getTime());
			printf("with a bandwidth of %f GB/s\n", 1e-9 * setupTimer.getBandwidth(n * sizeof(long) + size_layer1 * sizeof(short) + size_layer2 * sizeof(int)));

			if (layerSize(2, n) > 0) {
				printf("running second kernel...\n");
				int offset = layerOffsetInt(2, n);
				setupTimer.start();
				setupKernel2<1024><<<(size+1023)/1024, 1024>>>(size_layer2, &reinterpret_cast<uint32_t*>(d_bitmask)[offset]);
				setupTimer.stop();
				printf("second kernel ran for %f ms\n", 1e3 * setupTimer.getTime());
				printf("with a bandwidth of %f GB/s\n", 1e-9 * setupTimer.getBandwidth(size_layer3 * sizeof(int) + size_layer2 * sizeof(int)));
			}

			if (layerSize(4, n) > 0) {
				printf("running third kernel...\n");
				int offset = layerOffsetInt(4, n);
				setupTimer.start();
				setupKernel2<1024><<<(size+1023)/1024, 1024>>>(size_layer4, &reinterpret_cast<uint32_t*>(d_bitmask)[offset], false, false);
				setupTimer.stop();
				printf("third kernel ran for %f ms\n", 1e3 * setupTimer.getTime());
				printf("with a bandwidth of %f GB/s\n", 1e-9 * setupTimer.getBandwidth(size_layer4 * sizeof(int) + size_layer3 * sizeof(int)));
			}

			this->d_bitmask = d_bitmask;
			this->n = n;
		};

		void apply(int *permutation, int packedSize) {
			TreeStructure ts;

			uint32_t *d_bitmask_int = reinterpret_cast<uint32_t*>(d_bitmask);
			for (int layer = 0; layer < 5; layer++) {
				ts.layers[layer] = &d_bitmask_int[layerOffsetInt(layer, n)];
				ts.layerSizes[layer] = layerSize(layer, n);
			}

			improvedApply<<<(packedSize+127)/128, 128>>>(packedSize, permutation, n, ts);
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
