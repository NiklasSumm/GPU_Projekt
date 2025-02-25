#ifndef __ENCODINGIMPL_BASECLASS__
#define __ENCODINGIMPL_BASECLASS__

// An abstract class for all encoding implementations
class EncodingBase {
    public:
        virtual void setup(uint64_t *bitmask, int n) = 0; // Setup internal tree structure given a bitmask with length of n * 64 bits (long array with size n)
        virtual void apply(int *permutation, int packedSize) = 0; // Write permutation into provided array, mapping k (packed) to i (expanded)
        virtual void pack(int *src, int *dst, int packedSize) = 0; // Write elements for which bitmask is 1 from src to dst. src has size n*64 (see setup), dst has size packedSize.
        virtual void unpack(int *src, int *dst, int packedSize) = 0; // Write elements from src (with size packedSize) to dst, based on pattern in bitmask (see setup).
        virtual void print(uint64_t *bitmask) = 0; // Print the tree structure to std::cout
};

class TreeStructure {
    public:
        uint32_t *layers[5];
        int layerSizes[5];
};

int getNextLargestPowerOf2(int num){
    if (num & (num - 1)){
		num |= num >> 1;
    	num |= num >> 2;
    	num |= num >> 4;
    	num |= num >> 8;
    	num |= num >> 16;

		num = (num ^ (num << 1)) - 1;
	}
    return num;
}

#define UNUSED(x) (void)(x) // Disables compiler warnings

#endif
