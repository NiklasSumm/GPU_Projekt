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
populateBitmask(int numElements, uint32_t *bitmask, float sparsity, int groupSize)
{
    int elementsPerGroup = max(1, groupSize / 32); // Wie viele 32-bit-Werte pro Gruppe
    int groupIdx = (blockIdx.x * blockDim.x + threadIdx.x) / elementsPerGroup; // Gruppenindex
    int localIdx = (blockIdx.x * blockDim.x + threadIdx.x) % elementsPerGroup; // Position innerhalb der Gruppe

    if (groupIdx * elementsPerGroup >= numElements) return;

    curandState state;
    curand_init(1234, groupIdx, 0, &state); // Jede Gruppe bekommt ihre eigene PRNG-Sequenz

    if (groupSize >= 32) {
        // Fall: groupSize >= 32 (über mehrere uint32_t)
        bool setZero = (curand_uniform(&state) < sparsity);

        if (groupIdx * elementsPerGroup + localIdx < numElements) {
            bitmask[groupIdx * elementsPerGroup + localIdx] = setZero ? 0x00000000 : 0xffffffff;
        }
    } else {
        // Fall: groupSize < 32 (innerhalb eines einzigen uint32_t)
        int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (elementIdx >= numElements) return;

        uint32_t element = 0xffffffff; // Standard: Alle Bits auf 1

        for (uint32_t k = 0; k < 32; k += groupSize) {
            uint32_t mask = ((1u << groupSize) - 1) << k; // Sichere Maske für kleinere Gruppen
            if (curand_uniform(&state) < sparsity) {
                element &= ~mask; // Setze groupSize Bits auf 0
            }
        }
        bitmask[elementIdx] = element;
    }

    //int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
//
    //if (elementIdx < numElements) {
//
    //    curandState state;
    //    //int offset = (elementIdx / (groupSize / 32)) * 32;
    //    curand_init(1234, elementIdx, 0, &state);
//
    //    uint32_t element = 0xffffffff;
    //    for (uint32_t k = 0; k < sizeof(element) * 8; k += groupSize) {
    //        if (curand_uniform(&state) < sparsity) {
    //            // element |= 1 << k;
    //            element &= ~(((1 << groupSize) - 1) << k);
    //        }
    //    }
    //    bitmask[elementIdx] = element;
    //}
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

    int blockSize = 256;
    chCommandLineGet<int>(&blockSize, "blockSize", argc, argv);

    int layer1Size = 8;
    if (chCommandLineGetBool("fixedInclusive", argc, argv)) {
        layer1Size = 7;
    }
    chCommandLineGet<int>(&layer1Size, "layer1Size", argc, argv);

    int layer2Size = 8;
    chCommandLineGet<int>(&layer2Size, "layer2Size", argc, argv);

    int groupSize = 1;
    chCommandLineGet<int>(&groupSize, "groupSize", argc, argv);

    // Generate bitmask
    long *d_bitmask;
    cudaMalloc(&d_bitmask, static_cast<size_t>(treeSize));
    populateBitmask<<<(numElements*2 + 1023)/1024, 1024>>>(numElements*2, reinterpret_cast<uint32_t*>(d_bitmask), sparsity, groupSize);
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

    // Select implementation based on command line parameters
    std::unique_ptr<EncodingBase> implementation;
    if (chCommandLineGetBool("dynamicExclusive", argc, argv)) {
        switch (blockSize){
            case 32:
                implementation = std::make_unique<DynamicExclusive<32>>(); break;
            case 64:
                implementation = std::make_unique<DynamicExclusive<64>>(); break;
            case 128:
                implementation = std::make_unique<DynamicExclusive<128>>(); break;
            case 256:
                implementation = std::make_unique<DynamicExclusive<256>>(); break;
            case 512:
                implementation = std::make_unique<DynamicExclusive<512>>(); break;
            case 1024:
                implementation = std::make_unique<DynamicExclusive<1024>>(); break;
            default:
                throw std::invalid_argument( "Block size is not allowed! Allowed block sizes are: 32, 64, 128, 256, 512, 1024" );
        }
    } else if (chCommandLineGetBool("dynamicExclusiveSolo", argc, argv)) {
        switch (blockSize){
            case 32:
                implementation = std::make_unique<DynamicExclusive<32>>(false); break;
            case 64:
                implementation = std::make_unique<DynamicExclusive<64>>(false); break;
            case 128:
                implementation = std::make_unique<DynamicExclusive<128>>(false); break;
            case 256:
                implementation = std::make_unique<DynamicExclusive<256>>(false); break;
            case 512:
                implementation = std::make_unique<DynamicExclusive<512>>(false); break;
            case 1024:
                implementation = std::make_unique<DynamicExclusive<1024>>(false); break;
            default:
                throw std::invalid_argument( "Block size is not allowed! Allowed block sizes are: 32, 64, 128, 256, 512, 1024" );
        }
    } else if (chCommandLineGetBool("fixedInclusive", argc, argv)) {
        switch (blockSize){
            case 32:
                implementation = std::make_unique<FixedInclusive<32,8,8>>(); break;
            case 64:
                implementation = std::make_unique<FixedInclusive<64,8,8>>(); break;
            case 128:
                switch (layer2Size){
                    case 0:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,0>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,0>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,0>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,0>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedInclusive<128,10,0>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedInclusive<128,11,0>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedInclusive<128,12,0>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedInclusive<128,13,0>>(); break;
                            case 14:
                                implementation = std::make_unique<FixedInclusive<128,14,0>>(); break;
                            case 15:
                                implementation = std::make_unique<FixedInclusive<128,15,0>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 0 allowed layer1 sizes are: 6 - 15" );
                        } break;
                    case 1:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,1>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,1>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,1>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,1>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedInclusive<128,10,1>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedInclusive<128,11,1>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedInclusive<128,12,1>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedInclusive<128,13,1>>(); break;
                            case 14:
                                implementation = std::make_unique<FixedInclusive<128,14,1>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 1 allowed layer1 sizes are: 6 - 14" );
                        } break;
                    case 2:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,2>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,2>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,2>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,2>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedInclusive<128,10,2>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedInclusive<128,11,2>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedInclusive<128,12,2>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedInclusive<128,13,2>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 2 allowed layer1 sizes are: 6 - 13" );
                        } break;
                    case 3:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,3>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,3>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,3>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,3>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedInclusive<128,10,3>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedInclusive<128,11,3>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedInclusive<128,12,3>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 3 allowed layer1 sizes are: 6 - 12" );
                        } break;
                    case 4:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,4>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,4>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,4>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,4>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedInclusive<128,10,4>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedInclusive<128,11,4>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 4 allowed layer1 sizes are: 6 - 11" );
                        } break;
                    case 5:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,5>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,5>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,5>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,5>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedInclusive<128,10,5>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 5 allowed layer1 sizes are: 6 - 10" );
                        } break;
                    case 6:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,6>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,6>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,6>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedInclusive<128,9,6>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 6 allowed layer1 sizes are: 6 - 9" );
                        } break;
                    case 7:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,7>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,7>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedInclusive<128,8,7>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 7 allowed layer1 sizes are: 6 - 8" );
                        } break;
                    case 8:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,8>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedInclusive<128,7,8>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 8 allowed layer1 sizes are: 6 - 7" );
                        } break;
                    case 9:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedInclusive<128,6,9>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 9 allowed layer1 sizes are: 6 - 6" );
                        } break;
                    default:
                        throw std::invalid_argument( "Size for layer 2 is not allowed! Allowed sizes are: 0 - 10" );
                } break;
            case 256:
                implementation = std::make_unique<FixedInclusive<256,8,8>>(); break;
            case 512:
                implementation = std::make_unique<FixedInclusive<512,8,8>>(); break;
            default:
                throw std::invalid_argument( "Block size is not allowed! Allowed block sizes are: 32, 64, 128, 256, 512, 1024" );
        }
    } else if (chCommandLineGetBool("fixedExclusive", argc, argv)) {
        switch (blockSize){
            case 32:
                implementation = std::make_unique<FixedExclusive<32,8,8>>(); break;
            case 64:
                implementation = std::make_unique<FixedExclusive<64,8,8>>(); break;
            case 128:
                switch (layer2Size){
                    case 0:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,0>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,0>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,0>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,0>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,0>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedExclusive<128,11,0>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedExclusive<128,12,0>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedExclusive<128,13,0>>(); break;
                            case 14:
                                implementation = std::make_unique<FixedExclusive<128,14,0>>(); break;
                            case 15:
                                implementation = std::make_unique<FixedExclusive<128,15,0>>(); break;
                            case 16:
                                implementation = std::make_unique<FixedExclusive<128,16,0>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 0 allowed layer1 sizes are: 6 - 16" );
                        } break;
                    case 1:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,1>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,1>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,1>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,1>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,1>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedExclusive<128,11,1>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedExclusive<128,12,1>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedExclusive<128,13,1>>(); break;
                            case 14:
                                implementation = std::make_unique<FixedExclusive<128,14,1>>(); break;
                            case 15:
                                implementation = std::make_unique<FixedExclusive<128,15,1>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 1 allowed layer1 sizes are: 6 - 15" );
                        } break;
                    case 2:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,2>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,2>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,2>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,2>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,2>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedExclusive<128,11,2>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedExclusive<128,12,2>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedExclusive<128,13,2>>(); break;
                            case 14:
                                implementation = std::make_unique<FixedExclusive<128,14,2>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 2 allowed layer1 sizes are: 6 - 14" );
                        } break;
                    case 3:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,3>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,3>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,3>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,3>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,3>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedExclusive<128,11,3>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedExclusive<128,12,3>>(); break;
                            case 13:
                                implementation = std::make_unique<FixedExclusive<128,13,3>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 3 allowed layer1 sizes are: 6 - 13" );
                        } break;
                    case 4:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,4>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,4>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,4>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,4>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,4>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedExclusive<128,11,4>>(); break;
                            case 12:
                                implementation = std::make_unique<FixedExclusive<128,12,4>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 4 allowed layer1 sizes are: 6 - 12" );
                        } break;
                    case 5:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,5>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,5>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,5>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,5>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,5>>(); break;
                            case 11:
                                implementation = std::make_unique<FixedExclusive<128,11,5>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 5 allowed layer1 sizes are: 6 - 11" );
                        } break;
                    case 6:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,6>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,6>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,6>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,6>>(); break;
                            case 10:
                                implementation = std::make_unique<FixedExclusive<128,10,6>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 6 allowed layer1 sizes are: 6 - 10" );
                        } break;
                    case 7:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,7>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,7>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,7>>(); break;
                            case 9:
                                implementation = std::make_unique<FixedExclusive<128,9,7>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 7 allowed layer1 sizes are: 6 - 9" );
                        } break;
                    case 8:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,8>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,8>>(); break;
                            case 8:
                                implementation = std::make_unique<FixedExclusive<128,8,8>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 8 allowed layer1 sizes are: 6 - 8" );
                        } break;
                    case 9:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,9>>(); break;
                            case 7:
                                implementation = std::make_unique<FixedExclusive<128,7,9>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 9 allowed layer1 sizes are: 6 - 7" );
                        } break;
                    case 10:
                        switch (layer1Size){
                            case 6:
                                implementation = std::make_unique<FixedExclusive<128,6,10>>(); break;
                            default:
                                throw std::invalid_argument( "Size for layer 1 is not allowed! For layer 2 size of 10 allowed layer1 sizes are: 6 - 6" );
                        } break;
                    default:
                        throw std::invalid_argument( "Size for layer 2 is not allowed! Allowed sizes are: 0 - 10" );
                } break;
            case 256:
                implementation = std::make_unique<FixedExclusive<256,8,8>>(); break;
            case 512:
                implementation = std::make_unique<FixedExclusive<512,8,8>>(); break;
            case 1024:
                implementation = std::make_unique<FixedExclusive<1024,8,8>>(); break;
            default:
                throw std::invalid_argument( "Block size is not allowed! Allowed block sizes are: 32, 64, 128, 256, 512, 1024" );
        }
    } else if (chCommandLineGetBool("baseline", argc, argv)) {
        implementation = std::make_unique<ThrustBaseline>(packedSize);
    } else if (chCommandLineGetBool("baselineSetupLess", argc, argv)) {
        implementation = std::make_unique<ThrustBaseline>();
    } else {
        exit(1);
    }

    // Allocate a buffer larger than the L2 cache, so that we can clear L2 between runs
    size_t bufferSize = 128 * 1024 * 1024;
    void* d_l2_buffer;
    cudaMalloc(&d_l2_buffer, bufferSize);

    // Setup implementation
    if (benchmark) implementation->setup(reinterpret_cast<uint64_t*>(d_bitmask), numElements); // Warmup
    double setupTimeTotal = 0;
    double setupBandwidthTotal = 0;
    for (int i = 0; i < iterations; i++) {
        // Ensure we don't have caching effects from previous runs
        cudaMemset(d_l2_buffer, 0, bufferSize);
        cudaDeviceSynchronize();

        ChTimer setupTimer;
        setupTimer.start();
        implementation->setup(reinterpret_cast<uint64_t*>(d_bitmask), numElements);
        cudaDeviceSynchronize();
        setupTimer.stop();
        setupTimeTotal += setupTimer.getTime();
        setupBandwidthTotal += 1e-9 * setupTimer.getBandwidth(numElements * sizeof(long));
    }

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

    int expandedSize = numElements * sizeof(long) * CHAR_BIT;
    int *d_src;
    cudaMalloc(&d_src, static_cast<size_t>(expandedSize * sizeof(int)));
    int *d_dst;
    cudaMalloc(&d_dst, static_cast<size_t>(expandedSize * sizeof(int)));
    printf("Expanded Size: %d\n", expandedSize);
    bool pack = chCommandLineGetBool("pack", argc, argv);
    bool unpack = chCommandLineGetBool("unpack", argc, argv);

    // Apply implementation, producing a full permutation
    if (benchmark) {
        if (pack) {
            implementation->pack(d_src, d_dst, packedSize);
        } else if (unpack) {
            implementation->unpack(d_src, d_dst, packedSize);
        } else {
            implementation->apply(d_permutation, packedSize);
        }
    }
    double applyTimeTotal = 0;
    double applyBandwidthTotal = 0;
    size_t bandwidthBaseSize = numElements * sizeof(long); // Bitmask size as reference
    if (pack || unpack) bandwidthBaseSize = static_cast<size_t>(expandedSize) * sizeof(int); // Expanded size as reference
    for (int i = 0; i < iterations; i++) {
        // Ensure we don't have caching effects from previous runs
        cudaMemset(d_l2_buffer, 0, bufferSize);
        cudaDeviceSynchronize();

        ChTimer applyTimer;
        applyTimer.start();
        if (pack) {
            implementation->pack(d_src, d_dst, packedSize);
        } else if (unpack) {
            implementation->unpack(d_src, d_dst, packedSize);
        } else {
            implementation->apply(d_permutation, packedSize);
        }
        cudaDeviceSynchronize();
        applyTimer.stop();
        applyTimeTotal += applyTimer.getTime();
        applyBandwidthTotal += 1e-9 * applyTimer.getBandwidth(bandwidthBaseSize);
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
        cudaFreeHost(h_permutation);
    }

    // Calculate and print benchmark results
    if (benchmark) {
        printf("Setup Time: %f ms\n", 1e3 * setupTimeTotal / iterations);
        printf("Setup Bandwidth: %f GB/s\n", setupBandwidthTotal / iterations);

        printf("Apply Time: %f ms\n", 1e3 * applyTimeTotal / iterations);
        printf("Apply Bandwidth: %f GB/s\n", applyBandwidthTotal / iterations);
    }

    // Free device memory
    cudaFree(d_bitmask);
    cudaFree(d_permutation);
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_l2_buffer);

    // Free host pinned memory
    cudaFreeHost(h_bitmask);

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
              << "  --pack/unpack" << std::endl
              << "    Apply the implementation to pack or unpack an array of int instead of producing a permutation for validation." << std::endl
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
