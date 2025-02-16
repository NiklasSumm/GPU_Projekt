#ifndef __ENCODINGIMPL_BASECLASS__
#define __ENCODINGIMPL_BASECLASS__

// An abstract class for all encoding implementations
class EncodingBase {
    public:
        virtual void setup(uint64_t *bitmask, int n) = 0; // Setup internal tree structure given a bitmask with length of n * 64 bits (long array with size n)
        virtual void apply(int *permutation, int packedSize) = 0; // Write permutation into provided array, mapping k (packed) to i (expanded)
        virtual void print(uint64_t *bitmask) = 0; // Print the tree structure to std::cout
};

class TreeStructure {
	public:
		uint32_t *layers[5];
		int layerSizes[5];
};

#endif
