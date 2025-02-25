//returns the next largest power of two that is not lower than the input number
__device__ int getNextLargestPowerOf2(int num){
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