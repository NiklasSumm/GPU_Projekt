#include <cub/cub.cuh>

// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp
{
    // Running prefix
    int running_total;

    // Constructor
    __device__ BlockPrefixCallbackOp(int running_total) : running_total(running_total) {}

    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __device__ int operator()(int block_aggregate)
    {
        int old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};