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
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include <encodingBase.h>

#include <dynamic_exclusive.h>
#include <fixed_inclusive.h>
#include <fixed_exclusive.h>
#include <baseline.h>

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

__global__ void
populateBitmask(int numElements, uint32_t *bitmask, float sparsity)
{
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elementIdx < numElements) {

        curandState state;
        curand_init(1234, elementIdx, 0, &state);

        uint32_t element = 0xffffffff;
        for (uint32_t k = 0; k < sizeof(element) * 8; k++) {
            if (curand_uniform(&state) < sparsity) {
                // element |= 1 << k;
                element &= ~(1 << k);
            }
        }
        bitmask[elementIdx] = element;
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
              << "***" << std::endl << std::endl;

    // Print device information
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "Device Memory: " << prop.totalGlobalMem / 1048576 << " MB" << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl << std::endl;

    int iterations = 1;
    chCommandLineGet<int>(&iterations, "i", argc, argv);
    chCommandLineGet<int>(&iterations, "num-iterations", argc, argv);

    bool benchmark = chCommandLineGetBool("benchmark", argc, argv);
    bool validate = chCommandLineGetBool("validate", argc, argv);

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

    float sparsity = 0.0;
    chCommandLineGet<float>(&sparsity, "sparsity", argc, argv);

    // Generate bitmask
    long *d_bitmask;
    cudaMalloc(&d_bitmask, static_cast<size_t>(treeSize));
    populateBitmask<<<(numElements*2 + 1023)/1024, 1024>>>(numElements*2, reinterpret_cast<uint32_t*>(d_bitmask), sparsity);
    cudaDeviceSynchronize();

    // Copy bitmask back to host and count packed size
    cudaMemcpy(h_bitmask, d_bitmask, static_cast<size_t>(numElements * sizeof(*d_bitmask)), cudaMemcpyDeviceToHost); // Only copy bitmask
    cudaDeviceSynchronize();

    int packedSize = 0;
    for (int i = 0; i < numElements*2; i++) {
        packedSize += popcount(reinterpret_cast<uint32_t*>(h_bitmask)[i]);
    }

    // Alloc result array. For now we produce a full permutation, which can easily be extended to map data.
    int* d_permutation;
    cudaMalloc(&d_permutation, static_cast<size_t>(packedSize*sizeof(int)));

    // All implementations
    DynamicExclusive<1024> dynamicExclusive = DynamicExclusive<1024>();
    DynamicExclusive<1024> dynamicExclusiveSolo = DynamicExclusive<1024>(false);
    FixedInclusive<512,7,8> fixedInclusive = FixedInclusive<512,7,8>{};
    //FixedExclusive<1024,8,8> fixedExclusive = FixedExclusive<1024,8,8>{};
    ThrustBaseline baseline = ThrustBaseline(packedSize);
    ThrustBaseline baselineSetupLess = ThrustBaseline();

    // Select implementation based on command line parameters
    EncodingBase* implementation;
    if (chCommandLineGetBool("dynamicExclusive", argc, argv)) {
        //DynamicExclusive<1024> dynamicExclusive = DynamicExclusive<1024>();
        implementation = &dynamicExclusive;
    } else if (chCommandLineGetBool("dynamicExclusiveSolo", argc, argv)) {
        //DynamicExclusive<1024> dynamicExclusiveSolo = DynamicExclusive<1024>(false);
        implementation = &dynamicExclusiveSolo;
    } else if (chCommandLineGetBool("fixedInclusive", argc, argv)) {
        //FixedInclusive<512,7,8> fixedInclusive = FixedInclusive<512,7,8>{};
        implementation = &fixedInclusive;
    } else if (chCommandLineGetBool("fixedExclusive", argc, argv)) {
        //FixedExclusive<1024,8,8> fixedExclusive = FixedExclusive<1024,8,8>{};
        implementation = std::make_unique<FixedExclusive<1024,8,8>>();
    } else if (chCommandLineGetBool("baseline", argc, argv)) {
        //ThrustBaseline baseline = ThrustBaseline(packedSize);
        implementation = &baseline;
    } else if (chCommandLineGetBool("baselineSetupLess", argc, argv)) {
        //ThrustBaseline baselineSetupLess = ThrustBaseline();
        implementation = &baselineSetupLess;
    } else {
        exit(1);
    }

    // Setup implementation
    if (benchmark) implementation->setup(reinterpret_cast<uint64_t*>(d_bitmask), numElements); // Warmup
    ChTimer setupTimer;
    setupTimer.start();
    for (int i = 0; i < iterations; i++) {
        implementation->setup(reinterpret_cast<uint64_t*>(d_bitmask), numElements);
    }
    cudaDeviceSynchronize();
    setupTimer.stop();

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
    if (benchmark) implementation->apply(d_permutation, packedSize);
    ChTimer applyTimer;
    applyTimer.start();
    for (int i = 0; i < iterations; i++) {
        implementation->apply(d_permutation, packedSize);
    }
    cudaDeviceSynchronize();
    applyTimer.stop();

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

    // Validate the implementation
    if (validate) {
        int* h_permutation;
        cudaMallocHost(&h_permutation, static_cast<size_t>(packedSize*sizeof(int)));
        cudaMemcpy(h_permutation, d_permutation, static_cast<size_t>(packedSize*sizeof(int)), cudaMemcpyDeviceToHost); // Copy input back

        // Compare permutation to expected permutation
        int* permutation = packedPermutation(packedSize, numElements*64, h_bitmask);
        for (int i = 0; i < packedSize; i++) {
            if (h_permutation[i] != permutation[i]) {
                printf("%d: %d (ref: %d)\n", i, h_permutation[i], permutation[i]);
                exit(1);
            }
        }
    }

    // Calculate and print benchmark results
    if (benchmark) {
        printf("Setup Time: %f ms\n", 1e3 * setupTimer.getTime() / iterations);
        printf("Setup Bandwidth: %f GB/s\n", 1e-9 * setupTimer.getBandwidth(numElements * sizeof(long)) * iterations);

        printf("Apply Time: %f ms\n", 1e3 * applyTimer.getTime() / iterations);
        printf("Apply Bandwidth: %f GB/s\n", 1e-9 * applyTimer.getBandwidth(numElements * sizeof(long)) * iterations);

        ChTimer totalTimer = setupTimer + applyTimer;
        printf("Total Time: %f ms\n", 1e3 * totalTimer.getTime() / iterations);
        printf("Total Bandwidth: %f GB/s\n", 1e-9 * totalTimer.getBandwidth(numElements * sizeof(long)) * iterations);
    }

    return 0;
}

void printHelp(char *argv)
{
    std::cout << "Help:" << std::endl
              << "  Usage: " << std::endl
              << "  " << argv << " [-s <bitmask-size>] [implementation]"
              << std::endl
              << "" << std::endl
              << "  -s <bitmask-size>|--size <bitmask-size>" << std::endl
              << "    Size of bitmask in steps of uint64_t datatype (64 bits)" << std::endl
              << "" << std::endl
              << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
              << "    Number of iterations for benchmarking" << std::endl
              << "" << std::endl
              << "  --benchmark" << std::endl
              << "    Calculate and print benchmarking metrics (this will implicitely run warmup kernels)" << std::endl
              << "" << std::endl
              << "  --validate" << std::endl
              << "    Validate a specific implementation by producing a full permutation and comparing it against a CPU version" << std::endl
              << "" << std::endl
              << "  --sparsity <fraction>" << std::endl
              << "    Control the fraction of how many bits in the mask are 0 (default: dense bitmask, sparsity 0.0)" << std::endl
              << "" << std::endl
              << "  --dynamicExclusive" << std::endl
              << "    Use tree implementation with dynamic layer count and fixed steps of 2^5 between layers, allowing for collaborative descend" << std::endl
              << "" << std::endl
              << "  --dynamicExclusiveSolo" << std::endl
              << "    Use tree implementation with dynamic layer count and fixed steps of 2^5 between layers, without collaborative descend" << std::endl
              << "" << std::endl
              << "  --fixedInclusive" << std::endl
              << "    Use tree implementation with 2 layers and steps of 2^7 / 2^8 between first 2 layers (based on inclusive scan)" << std::endl
              << "" << std::endl
              << "  --fixedExclusive" << std::endl
              << "    Use tree implementation with 2 layers and steps of 2^8 / 2^8 between first 2 layers (based on exclusive scan)" << std::endl
              << "" << std::endl
              << "  --baseline" << std::endl
              << "    Use baseline implementation based on thrust using an inverse permutation" << std::endl
              << "" << std::endl
              << "  --baselineSetupLess" << std::endl
              << "    Use baseline implementation based on on-the-fly thrust copy_if and fancy iterators" << std::endl
              << "" << std::endl;
}
