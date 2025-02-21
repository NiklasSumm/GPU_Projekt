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
#include <tree78.h>
#include <tree88.h>

const static int DEFAULT_NUM_ELEMENTS = 10;

//
// Function Prototypes
//
void printHelp(char *);


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

	Tree115<256> tree115 = Tree115<256>{};
	Tree78<128> tree78 = Tree78<128>{};
	Tree88<256,9,6> tree88 = Tree88<256,9,6>{};

	// Select implementation based on command line parameters
	EncodingBase* implementation;
	if (chCommandLineGetBool("115", argc, argv)) {
		implementation = &tree115;
	} else if (chCommandLineGetBool("78", argc, argv)) {
		implementation = &tree78;
	} else if (chCommandLineGetBool("88", argc, argv)) {
		implementation = &tree88;
	} else {
		exit(1);
	}

	// Setup implementation
	implementation->setup(reinterpret_cast<uint64_t*>(d_bitmask), numElements);
	cudaDeviceSynchronize();

	// Copy full tree back and print structure
	cudaMemcpy(h_bitmask, d_bitmask, static_cast<size_t>(treeSize), cudaMemcpyDeviceToHost);
	implementation->print(reinterpret_cast<uint64_t*>(h_bitmask));
	
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

	// Apply implementation, producing a full permutation
	int* d_permutation;
	cudaMalloc(&d_permutation, static_cast<size_t>(packedSize*sizeof(int)));

	implementation->apply(d_permutation, packedSize);
	cudaDeviceSynchronize();

	int* h_permutation;
	cudaMallocHost(&h_permutation, static_cast<size_t>(packedSize*sizeof(int)));
	cudaMemcpy(h_permutation, d_permutation, static_cast<size_t>(packedSize*sizeof(int)), cudaMemcpyDeviceToHost); // Copy input back

	// Compare permutation to expected permutation
	int* permutation = packedPermutation(packedSize, numElements*64, h_bitmask);
	for (int i = 0; i < packedSize; i++) {
		if (h_permutation[i] != permutation[i]) {
			printf("%d: %d (ref: %d)\n", i, h_permutation[i], permutation[i]);
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
