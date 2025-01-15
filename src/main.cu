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

const static int DEFAULT_NUM_ELEMENTS = 10;
const static int DEFAULT_BLOCK_DIM = 128;

//
// Function Prototypes
//
void printHelp(char *);


__global__ void
simpleKernel(int numElements, int *input)
{
    int elementId = blockIdx.x * blockDim.x + threadIdx.x;

    if (elementId < numElements) {
        // Specialize BlockScan for a 1D block of 128 threads of type int
        using BlockScan = cub::BlockScan<int, 128>;

        // Allocate shared memory for BlockScan
        __shared__ typename BlockScan::TempStorage temp_storage;

        // Obtain input item for each thread
        long long thread_data = static_cast<long long>(__popc(input[elementId]));

        printf("pre: %d - %d\n", elementId, thread_data);

        // Collectively compute the block-wide exclusive prefix max scan
       	//BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

        printf("post: %d - %d\n", elementId, thread_data);
    }
}

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
        long long thread_data = static_cast<long long>(__popc(input[elementId]));

        printf("pre: %d - %d\n", elementId, thread_data);
				
		// Collectively compute the block-wide exclusive prefix max scan
        //BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

        printf("post: %d - %d\n", elementId, thread_data);
    }
}

__global__ void
setupKernel(int length, int *bitMask){
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	long* bitMask_long = reinterpret_cast<long*>(bitMask);

	if (elementId < length / (8 * sizeof(long))){
		printf("% d - %d - %d\n",bitMask[2 * elementId], bitMask[2 * elementId + 1], bitMask_long[elementId]);

		int threadSum = __popc(bitMask_long[elementId]);

		using BlockScan = cub::BlockScan<int, 128>;

		__shared__ typename BlockScan::TempStorage temp_storage;

		BlockScan(temp_storage).ExclusiveSum(threadSum, threadSum);

		//printf("post: %d - %d\n", elementId, threadSum);
	}

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

    int *h_bitmask;
    cudaMallocHost(&h_bitmask, static_cast<size_t>(numElements * sizeof(*h_bitmask)));
    h_bitmask[1] = 2;
    h_bitmask[3] = 5;

    //
    // Copy Data to the Device
    //
    int *d_bitmask;
    cudaMalloc(&d_bitmask, static_cast<size_t>(numElements * sizeof(*d_bitmask)));
    cudaMemcpy(d_bitmask, h_bitmask, static_cast<size_t>(numElements * sizeof(*d_bitmask)), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    setupKernel<<<1, 128>>>(numElements * 8 * sizeof(int), d_bitmask);

    // Synchronize
    cudaDeviceSynchronize();

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
