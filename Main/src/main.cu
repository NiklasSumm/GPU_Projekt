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

const static int DEFAULT_NUM_ELEMENTS = 10;

//
// Function Prototypes
//
void printHelp(char *);
void printTree(int, long*);


__global__ void
setupKernel1(int numElements, long *input)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements) {
		using BlockScan = cub::BlockScan<unsigned int, 1024>;
		__shared__ typename BlockScan::TempStorage temp_storage;

		// Load 64 bit bitmask section and count bits
		unsigned int original_data = __popcll(input[elementId]);
		unsigned int thread_data;

		// Collectively compute the block-wide exclusive sum
		BlockScan(temp_storage).ExclusiveSum(original_data, thread_data);

		// First thread of each warp writes in layer 1
		if ((threadIdx.x & 31) == 0) {
			reinterpret_cast<unsigned short*>(input)[numElements*4+elementId/32] = static_cast<unsigned short>(thread_data);
		}

		// Last thread of each full block writes into layer 2
		if (threadIdx.x == 1023) {
			int offset = numElements*2 + ((numElements+31)/32 + 1)/2;
			reinterpret_cast<unsigned int*>(input)[offset+blockIdx.x] = thread_data + original_data;
		}
	}
}

__global__ void
setupKernel2(int numElements, unsigned int *input, bool next=true, bool nextButOne=true)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements) {
		using BlockScan = cub::BlockScan<unsigned int, 1024>;
		__shared__ typename BlockScan::TempStorage temp_storage;

		// Load prepared values from previous kernel
		unsigned int original_data = input[elementId];
		unsigned int thread_data;

		// Collectively compute the block-wide exclusive sum over the prepared values
		BlockScan(temp_storage).ExclusiveSum(original_data, thread_data);

		// The value in thread_data stored by the first thread in a warp needs to be subtracted
		// from the values each thread in the warp has, to get the correct values for the next value.
		// Doing it this way allows us to simultaneously prepare the next but one layer.
		unsigned int correction = thread_data;
		correction = __shfl_sync(0xffffffff, correction, 0);

		// Every thread needs to rewrite own value, to correct current layer
		input[elementId] = thread_data - correction;

		// First thread of each warp writes in next layer. These values are already fully correct.
		if (next && (threadIdx.x & 31) == 0) {
			input[numElements+(elementId/32)] = thread_data;
		}

		// Last thread of each full block writes into next but one layer. These values need to be corrected.
		if (nextButOne && threadIdx.x == 1023) {
			int offset = numElements + (numElements+31)/32;
			input[offset+blockIdx.x] = thread_data + original_data;
		}
	}
}


__global__ void
simpleApply(int numPacked, int *permutation, int bitmaskSize, long *tree)
{
	int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementIdx < numPacked) {
		int bitsToFind = elementIdx+1;

		// Handle layer 4
		int layer4Size = (bitmaskSize+1024*1024-1) / (1024*1024);
		int layer4Offset = bitmaskSize*2 + ((bitmaskSize+31)/32 + 1)/2 + (bitmaskSize+1023) / 1024 + (bitmaskSize+1024*32-1) / (1024*32);
		int offsetLayer3 = 0; // Offset inside layer 3 due to layer 4 selection
		for (int i = layer4Size-1; i > 0; i--) {
			int layerSum = reinterpret_cast<unsigned int*>(tree)[layer4Offset+i];
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				offsetLayer3 = i;
				break;
			}
		}
		int bitmaskOffset = offsetLayer3 * 32;

		// Handle layer 3
		int layer3Size = (bitmaskSize+1024*32-1) / (1024*32) - offsetLayer3 * 32;
		if (layer3Size > 32) layer3Size = 32;
		int layer3Offset = bitmaskSize*2 + ((bitmaskSize+31)/32 + 1)/2 + (bitmaskSize+1023) / 1024 + offsetLayer3 * 32;
		int offsetLayer2 = 0; // Offset inside layer 2 due to layer 3 selection
		for (int i = layer3Size-1; i > 0; i--) {
			int layerSum = reinterpret_cast<unsigned int*>(tree)[layer3Offset+i];
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				bitmaskOffset += i;
				break;
			}
		}
		bitmaskOffset *= 32;

		// Handle layer 2
		int layer2Size = (bitmaskSize+1023) / 1024 - offsetLayer2 * 32;
		if (layer2Size > 32) layer2Size = 32;
		int layer2Offset = bitmaskSize*2 + ((bitmaskSize+31)/32 + 1)/2 + offsetLayer2 * 32;
		int offsetLayer1 = 0; // Offset inside layer 1 due to layer 2 selection
		for (int i = layer2Size-1; i > 0; i--) {
			int layerSum = reinterpret_cast<unsigned int*>(tree)[layer2Offset+i];
			if (layerSum < bitsToFind) {
				bitsToFind -= layerSum;
				bitmaskOffset += i;
				break;
			}
		}
		bitmaskOffset *= 32;

		// Handle layer 1
		int layer1Size = (bitmaskSize+31) / 32 - offsetLayer1 * 32;
		if (layer1Size > 32) layer1Size = 32;
		int layer1Offset = bitmaskSize*4 + offsetLayer1 * 32; // 4 shorts in one long
		for (int i = layer1Size-1; i > 0; i--) {
			int layerSum = static_cast<int>(reinterpret_cast<unsigned short*>(tree)[layer1Offset+i]);
			if (layerSum < bitsToFind) {
				bitmaskOffset += i;
				bitsToFind -= layerSum;
				break;
			}
		}
		bitmaskOffset *= 32;

		// Handle virtual layer 0 (before bitmask)
		int layerSum;
		long bitmaskSection;
		int vLayerOffset = 0;
		for (; vLayerOffset < 32; vLayerOffset++) {
			bitmaskSection = tree[bitmaskOffset+vLayerOffset];
			layerSum = __popcll(bitmaskSection);
			bitsToFind -= layerSum;
			if (bitsToFind <= 0) break;
		}
		bitsToFind += layerSum;

		// Handle bitmask
		int expandedIndex = bitmaskOffset * 64 + vLayerOffset * 64;
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

/*
__global__ void
simpleApply2(int numPacked, int *input, int numExpanded, int *output, int bitmaskSize, long *tree)
{
	int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementIdx < numPacked) {
		int expandedIndex;

		int layer1Size = bitmaskSize / 32;
		int layer1Offset = bitmaskSize*2;
		int bitmaskOffset = 0;
		int start;
		for (int i = 1; i < layer1Size; i++) {
			int newStart = reinterpret_cast<int*>(tree)[layer1Offset+i];
			if (newStart < elementIdx) {
				bitmaskOffset = i;
				start = newStart;
			}
			else break;
		}
		bitmaskOffset *= 32;

		int add;
		long bitmaskSection;
		for (int i = 0; i < 32; i++) {
			bitmaskSection = tree[bitmaskOffset+i];
			add = __popcll(bitmaskSection);
			start += add;
			if (start >= elementIdx) break;
		}
		start -= add;

		

		for (int k = sizeof(long)-1; k >= 0; k--) {
			std::cout << ((v >> k) & 1) << "";
		}
	}
}

__global__ void
simpleApply3(int numPacked, int *input, int numExpanded, int *output, int* layers[2])
{
	int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementIdx < numPacked) {
		int expandedIndex;

		if (layers[0]) {
			// TODO: layer1 handling
			int layer1Size = numExpanded / (32*64);
		}

		if (layers[1]) {
			// TODO: bitmask handling
		}

		int layer1Size = bitmaskSize / 32;
		int layer1Offset = bitmaskSize*2;
		int bitmaskOffset = 0;
		int start;
		for (int i = 1; i < layer1Size; i++) {
			int newStart = reinterpret_cast<int*>(tree)[layer1Offset+i];
			if (newStart < elementIdx) {
				bitmaskOffset = i;
				start = newStart;
			}
			else break;
		}
		bitmaskOffset *= 32;

		int add;
		long bitmaskSection;
		for (int i = 0; i < 32; i++) {
			bitmaskSection = tree[bitmaskOffset+i];
			add = __popcll(bitmaskSection);
			start += add;
			if (start >= elementIdx) break;
		}
		start -= add;

		

		for (int k = sizeof(long)-1; k >= 0; k--) {
			std::cout << ((v >> k) & 1) << "";
		}
	}
}
*/



/*
__constant__ int topLayer[32];

__global__ void
secondKernel(int numElements, int *input)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements) {
		// Load own element
		int ownElement = input[elementId];

		// Find first top layer splitting point
		int start;
#pragma unroll 31
		for (int i = 1; i < 31; i++) {
			if (__all_sync(0xffffffff, ownElement > topLayer[i]) == 0) {
				start = i-1;
				break;
			}
		}

		// TODO: Traverse tree with only first element in mind
		// TODO: Once first element is done, try to re-use as much as possible with second, third, ... element


		// Specialize BlockScan for a 1D block of 128 threads of type int
		using BlockScan = cub::BlockScan<int, 128>;

		// Allocate shared memory for BlockScan
		__shared__ typename BlockScan::TempStorage temp_storage;

		// Obtain input item for each thread
		short thread_data = static_cast<short>(__popc(input[elementId]));

		printf("pre: %d - %d\n", elementId, thread_data);

		// Collectively compute the block-wide exclusive prefix max scan
		BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

		printf("post: %d - %d\n", elementId, thread_data);
	}
}
*/


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

int layerSize(int layer, int bitmaskSize) {
	int size = bitmaskSize * 2; // Bitmask size is size in long

	int factor = 1;
	for (int i = layer; i > 0; i--) {
		factor *= 32;
	}

	if (layer > 0) {
		size = (bitmaskSize+factor-1)/factor;

		// Correct for layer 1 using short instead of int
		if (layer == 1) {
			size = (size+1)/2;
		}
	}

	return size;
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

	int treeSize = 0;
	for (int layer = 0; layer < 5; layer++) {
		int size = layerSize(layer, numElements);
		treeSize += size;
		printf("layerSize: %d - %d\n", layer, size);
	}
	treeSize *= sizeof(int);
	printf("treeSize: %d Bytes\n", treeSize);

	long *h_bitmask;
	cudaMallocHost(&h_bitmask, static_cast<size_t>(treeSize));
	
	// Initialize bitmask with random data
	srand(0); // Always the same random numbers
	int packedSize = 0;
	for (int i = 0; i < numElements*2; i++) {
		// unsigned int element = static_cast<unsigned int>(rand());
		unsigned int element = std::numeric_limits<unsigned int>::max();
		packedSize += popcount(element);
		reinterpret_cast<int*>(h_bitmask)[i] = element;
		// reinterpret_cast<unsigned int*>(h_bitmask)[i] = std::numeric_limits<unsigned int>::max();
	}


	//
	// Copy Data to the Device
	//
	long *d_bitmask;
	cudaMalloc(&d_bitmask, static_cast<size_t>(treeSize));

	ChTimer copy_timer;
	copy_timer.start();

	cudaMemcpy(d_bitmask, h_bitmask, static_cast<size_t>(numElements * sizeof(*d_bitmask)), cudaMemcpyHostToDevice); // Only copy bitmask

	copy_timer.stop();
	printf("Time to copy from host to device = %f", copy_timer.getTime());


	cudaDeviceSynchronize();

	ChTimer setup_timer;
	ChTimer setup1_timer;
	ChTimer setup2_timer;
	ChTimer setup2_timer;

	setup_timer.start();
	setup1_timer.start();
	setupKernel1<<<(numElements+1023)/1024, 1024>>>(numElements, d_bitmask);
	setup1_timer.stop();

	int offset;
	if (layerSize(2, numElements) > 0) {
		printf("running second kernel...\n");
		offset = layerSize(0, numElements) + layerSize(1, numElements);
		int size = layerSize(2, numElements);
		setup2_timer.start();
		setupKernel2<<<(size+1023)/1024, 1024>>>(size, &reinterpret_cast<unsigned int*>(d_bitmask)[offset]);
		setup2_timer.stop();
	}

	if (layerSize(4, numElements) > 0) {
		printf("running third kernel...\n");
		offset += layerSize(2, numElements) + layerSize(3, numElements);
		int size = layerSize(4, numElements);
		setup3_timer.start();
		setupKernel2<<<(size+1023)/1024, 1024>>>(size, &reinterpret_cast<unsigned int*>(d_bitmask)[offset], false, false);
		setup3_timer.stop();
	}

	setup_timer.stop();

	printf("Overall setup time = %f", setup_timer.getTime());
	printf("Setup kernel 1 time = %f", setup1_timer.getTime());
	printf("Setup kernel 2 time = %f", setup2_timer.getTime());
	printf("Setup kernel 3 time = %f", setup3_timer.getTime());

	// Synchronize
	cudaDeviceSynchronize();

	copy_timer.start();

	cudaMemcpy(h_bitmask, d_bitmask, static_cast<size_t>(treeSize), cudaMemcpyDeviceToHost); // Copy full tree back

	copy_timer.stop();
	printf("Time to copy from device to host = %f", copy_timer.getTime());

	printTree(numElements, h_bitmask);

	// Apply
	int* d_input;
	cudaMalloc(&d_input, static_cast<size_t>(packedSize*sizeof(int)));

	simpleApply<<<(packedSize+127)/128, 128>>>(packedSize, d_input, numElements, d_bitmask);
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
	cudaError_t cudaError = cudaGetLastError();
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

void printTree(int numElements, long* tree) {
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
	if (layerSize(1, numElements)*2 < 500) {
		std::cout << "layer 1: ";
		for (int i = offset; i < offset+((numElements+31)/32); i++) {
			std::cout << reinterpret_cast<unsigned short*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}

	// Print second layer layer (int)
	offset = layerSize(0, numElements) + layerSize(1, numElements);
	if (layerSize(2, numElements) < 500) {
		std::cout << "layer 2: ";
		for (int i = offset; i < offset+layerSize(2, numElements); i++) {
			std::cout << reinterpret_cast<unsigned int*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}

	// Print third layer layer (int)
	offset += layerSize(2, numElements);
	if (layerSize(3, numElements) < 500) {
		std::cout << "layer 3: ";
		for (int i = offset; i < offset+layerSize(3, numElements); i++) {
			std::cout << reinterpret_cast<unsigned int*>(tree)[i] << " ";
		}
		std::cout << std::endl;
	}

	// Print fourth layer layer (int)
	offset += layerSize(3, numElements);
	std::cout << "layer 4: ";
	for (int i = offset; i < offset+layerSize(4, numElements); i++) {
		std::cout << reinterpret_cast<unsigned int*>(tree)[i] << " ";
	}
	std::cout << std::endl;
}
