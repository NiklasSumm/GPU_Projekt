/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Gruppe : TODO
 *
 *                   File : main.cu
 *
 *                Purpose : n-Body Computation
 *
 ******************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <bits/stl_numeric.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <encodingBase.h>

#include <tree115.h>

const static int DEFAULT_NUM_ELEMENTS = 10;

//
// Function Prototypes
//
void printHelp(char *);
void printTree78(int, long*);
void printTree88(int, long*);


template <int blockSize>
__global__ void
setupKernel78(int numElements, long *input)
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

template <int blockSize>
__global__ void
setupKernel88(int numElements, long *input)
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


template<class T>
int popcount(T val) {
        int bits = 0;
        for (int i = 0; i < static_cast<int>(sizeof(val))*8; i++) {
                bits += ((val >> i) & 1);
        }
        return bits;
}

int* packedPermutation(int packedSize, int expandedSize, long* h_bitmask) {
	int *sequence = (int*) malloc(expandedSize * sizeof(int));
	std::iota(sequence, sequence + std::ptrdiff_t(expandedSize), 0);

	int* permutation = (int*) malloc(packedSize * sizeof(int));
	std::copy_if(sequence, sequence + std::ptrdiff_t(expandedSize), permutation, [h_bitmask](int i) { return (h_bitmask[i/64] >> (63-i%64)) & 1; });
	return permutation;
}

void setup78(int numElements, long *d_bitmask) {
	setupKernel78<512><<<(numElements+511)/512, 512>>>(numElements, d_bitmask);

	int offset = numElements*2 + ((numElements+1)/2 + 1)/2;
	int size = (numElements+511)/512;
	unsigned int *startPtr = &reinterpret_cast<unsigned int*>(d_bitmask)[offset];

	// Determine temporary device storage requirements
	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Run exclusive prefix sum
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);
}

void setup88(int numElements, long *d_bitmask) {
	setupKernel88<1024><<<(numElements+1023)/1024, 1024>>>(numElements, d_bitmask);

	int offset = numElements*2 + ((numElements+3)/4 + 1)/2;
	int size = (numElements+1023)/1024;
	unsigned int *startPtr = &reinterpret_cast<unsigned int*>(d_bitmask)[offset];

	// Determine temporary device storage requirements
	void *d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Run exclusive prefix sum
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, startPtr, startPtr, size);
}

//
// Main
//
int main(int argc, char *argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ? numElements : DEFAULT_NUM_ELEMENTS;

	int treeSize = numElements * sizeof(long) * 2; // TODO: Replace with correct calculation

	long *h_bitmask;
	cudaMallocHost(&h_bitmask, static_cast<size_t>(treeSize));
	
	// Initialize bitmask with random data
	srand(0); // Always the same random numbers
	int packedSize = 0;
	for (int i = 0; i < numElements*2; i++) {
		// unsigned int element = static_cast<unsigned int>(rand());
		uint32_t element = std::numeric_limits<uint32_t>::max();
		packedSize += popcount(element);
		reinterpret_cast<uint32_t*>(h_bitmask)[i] = element;
		// reinterpret_cast<unsigned int*>(h_bitmask)[i] = std::numeric_limits<unsigned int>::max();
	}


	//
	// Copy Data to the Device
	//
	long *d_bitmask;
	cudaMalloc(&d_bitmask, static_cast<size_t>(treeSize));
	cudaMemcpy(d_bitmask, h_bitmask, static_cast<size_t>(numElements * sizeof(*d_bitmask)), cudaMemcpyHostToDevice); // Only copy bitmask
	cudaDeviceSynchronize();

	Tree115 tree = Tree115{};
	EncodingBase* implementation = &tree;


	if (chCommandLineGetBool("115", argc, argv)) {
		implementation->setup(reinterpret_cast<uint64_t*>(d_bitmask), numElements);
	} else if (chCommandLineGetBool("78", argc, argv)) {
		setup78(numElements, d_bitmask);
	} else if (chCommandLineGetBool("88", argc, argv)) {
		setup88(numElements, d_bitmask);
	}
	

	// Synchronize
	cudaDeviceSynchronize();

	cudaMemcpy(h_bitmask, d_bitmask, static_cast<size_t>(treeSize), cudaMemcpyDeviceToHost); // Copy full tree back

	if (chCommandLineGetBool("115", argc, argv)) {
		implementation->print(reinterpret_cast<uint64_t*>(h_bitmask));
	} else if (chCommandLineGetBool("78", argc, argv)) {
		printTree78(numElements, h_bitmask);
	} else if (chCommandLineGetBool("88", argc, argv)) {
		printTree88(numElements, h_bitmask);
	}

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	if (!chCommandLineGetBool("115", argc, argv)) {
		exit(0);
	}

	// Apply
	int* d_input;
	cudaMalloc(&d_input, static_cast<size_t>(packedSize*sizeof(int)));

	implementation->apply(d_input, packedSize);
	cudaDeviceSynchronize();

	int* h_input;
	cudaMallocHost(&h_input, static_cast<size_t>(packedSize*sizeof(int)));
	cudaMemcpy(h_input, d_input, static_cast<size_t>(packedSize*sizeof(int)), cudaMemcpyDeviceToHost); // Copy input back

	int* permutation = packedPermutation(packedSize, numElements*64, h_bitmask);
	for (int i = 0; i < packedSize; i++) {
		if (h_input[i] != permutation[i]) {
			printf("%d: %d (ref: %d)\n", i, h_input[i], permutation[i]);
			break;
		}
	} 

	// Check for Errors
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	return 0;
}

void printHelp(char *argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  << std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "    Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "    Number of elements (particles)" << std::endl
			  << "" << std::endl
			  << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
			  << "    Number of iterations" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" << std::endl
			  << "    The number of threads per block" << std::endl
			  << "" << std::endl
			  << "  --silent" << std::endl
			  << "    Suppress print output during iterations (useful for benchmarking)" << std::endl
			  << "" << std::endl
			  << "  --soa" << std::endl
			  << "    Use simple implementation with SOA optimization" << std::endl
			  << "" << std::endl
			  << "  --tiling" << std::endl
			  << "    Use optimized implementation using shared memory tiling" << std::endl
			  << "" << std::endl
			  << "  --stream" << std::endl
			  << "    Use stream implementation with aritifical 4 MB limit" << std::endl
			  << "" << std::endl;
}

void printTree78(int numElements, long* tree) {
	// Print bitmask
	if (numElements < 100) {
		std::cout << "bitmask: ";
		for (int i = 0; i < numElements; i++) {
			long v = tree[i];
			for (int k = sizeof(long)*8-1; k >= 0; k--) {
				std::cout << ((v >> k) & 1) << "";
			}
		}
		std::cout << std::endl;
	}

	// Print first layer (short)
	int offset = numElements*4; // 4 shorts in one long
	int size = (numElements+1)/2;
	if (size < 500) {
		std::cout << "layer 1: ";
		for (int i = offset; i < offset+size; i++) {
			std::cout << reinterpret_cast<unsigned short*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}

	// Print second layer layer (int)
	offset = offset / 2 + (size+1) / 2;
	size = (numElements+511) / 512;
	if (size < 500) {
		std::cout << "layer 2: ";
		for (int i = offset; i < offset+size; i++) {
			std::cout << reinterpret_cast<unsigned int*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}
}

void printTree88(int numElements, long* tree) {
	// Print bitmask
	if (numElements < 100) {
		std::cout << "bitmask: ";
		for (int i = 0; i < numElements; i++) {
			long v = tree[i];
			for (int k = sizeof(long)*8-1; k >= 0; k--) {
				std::cout << ((v >> k) & 1) << "";
			}
		}
		std::cout << std::endl;
	}

	// Print first layer (short)
	int offset = numElements*4; // 4 shorts in one long
	int size = (numElements+3)/4;
	if (size < 500) {
		std::cout << "layer 1: ";
		for (int i = offset; i < offset+size; i++) {
			std::cout << reinterpret_cast<unsigned short*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}

	// Print second layer layer (int)
	offset = offset / 2 + (size+1) / 2;
	size = (numElements+1023) / 1024;
	if (size < 500) {
		std::cout << "layer 2: ";
		for (int i = offset; i < offset+size; i++) {
			std::cout << reinterpret_cast<unsigned int*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}
}
