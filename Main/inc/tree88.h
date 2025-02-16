#include <cub/cub.cuh>

#include <encodingBase.h>

template <int blockSize>
__global__ void
setupKernel88(int numElements, uint64_t *input)
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
		if (elementId < numElements) 
			original_data = __popcll(input[elementId]);
		unsigned int thread_data;

		// Collectively compute the block-wide inclusive sum
		BlockScan(temp_storage).ExclusiveSum(original_data, thread_data, aggregate);

		// Every fourth thread writes value in first layer
		if (((threadIdx.x & 3) == 0) && (elementId < numElements)) {
			reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/4] = static_cast<unsigned short>(thread_data + aggregateSum);
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


class Tree88 : public EncodingBase {
	private:
		uint64_t *d_bitmask;
		int n;
	
	public:
		void setup(uint64_t *d_bitmask, int n) {
			setupKernel88<1024><<<(n+1023)/1024, 1024>>>(n, d_bitmask);

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

