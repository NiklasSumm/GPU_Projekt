#include <cuda/std/cstdint>
#include <cuda/std/iterator>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/detail/caching_allocator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <encodingBase.h>


using IndexT = cuda::std::int32_t;

template <typename MaskT>
constexpr auto num_bits() -> cuda::std::size_t {
    return sizeof(MaskT) * CHAR_BIT;
}

// Helper functor
template <typename MaskIter>
class Mask2Bool {
    public:
    Mask2Bool(MaskIter mask_iter) : mask_iter_{mask_iter} {}

    __host__ __device__ auto operator()(IndexT idx) -> bool {
        // As bits_per_mask is constexpr, the compiler should optimize these into bitshifts and masks.
        const auto mask_idx = idx / bits_per_mask;
        const auto mask_bit = bits_per_mask - idx % bits_per_mask - 1;
        return (mask_iter_[mask_idx] >> mask_bit) & MaskT{1};
    }

    private:
    using MaskT = typename cuda::std::iterator_traits<MaskIter>::value_type;
    static constexpr auto bits_per_mask = static_cast<IndexT>(num_bits<MaskT>());
    MaskIter mask_iter_;
};

template <typename MaskIter>
auto make_mask2bool_iterator(MaskIter mask_iter) {
    return thrust::make_transform_iterator(
            thrust::make_counting_iterator(IndexT{0}),
            Mask2Bool{mask_iter});
}

struct fetch_write
{
  int *dst;
  int *src;

  __host__ __device__
  void operator()(int packedIdx, int expandedIdx) const
  {
    dst[expandedIdx] = src[packedIdx];
  }
};

class ThrustBaseline : public EncodingBase {
    private:
        int packedSize;
        uint32_t *d_inverse_permutation;
        uint64_t *d_bitmask;
        int n;
        bool setupLess = false;
        thrust::mr::allocator<char, thrust::mr::disjoint_unsynchronized_pool_resource<thrust::device_memory_resource, thrust::mr::new_delete_resource>> allocator = thrust::detail::single_device_tls_caching_allocator();

    public:
        ThrustBaseline(int packedSize) {
            this->packedSize = packedSize;
            cudaMalloc(&d_inverse_permutation, static_cast<size_t>(packedSize*sizeof(uint32_t)));
        }
        ThrustBaseline(): setupLess(true) {} // Implicitely switches to setup-less implementation

        void setup(uint64_t *d_bitmask, int n) {
            this->d_bitmask = d_bitmask;
            this->n = n;

            if (!setupLess) {
                const auto bool_iter = make_mask2bool_iterator(thrust::device_ptr<uint64_t>(d_bitmask));
                const auto num_bools = num_bits<uint64_t>() * n;

                thrust::counting_iterator<int> iter(0);
                thrust::copy_if(thrust::cuda::par(allocator), iter, iter + num_bools, bool_iter, thrust::device_ptr<uint32_t>(d_inverse_permutation), cuda::std::identity{});
            }
        };

        void apply(int *permutation, int packedSize) {
            if (!setupLess) {
                cudaMemcpy(permutation, d_inverse_permutation, static_cast<size_t>(packedSize * sizeof(uint32_t)), cudaMemcpyDeviceToDevice);
            } else {
                // Setup-less
                const auto bool_iter = make_mask2bool_iterator(d_bitmask);
                const auto num_bools = num_bits<uint64_t>() * n;

                thrust::counting_iterator<int> iter(0);
                thrust::copy_if(thrust::cuda::par(allocator), iter, iter + num_bools, bool_iter, static_cast<thrust::device_ptr<int>>(permutation), cuda::std::identity{});
            }
        };

        void pack(int *src, int *dst, int packedSize) {
            if (!setupLess) {
                thrust::gather(thrust::cuda::par(allocator), d_inverse_permutation, d_inverse_permutation + packedSize, src, dst);
            } else {
                const auto bool_iter = make_mask2bool_iterator(d_bitmask);
                const auto num_bools = num_bits<uint64_t>() * n;

                thrust::copy_if(thrust::cuda::par(allocator), src, src + num_bools, bool_iter, dst, cuda::std::identity{});
            }
        };

        void unpack(int *src, int *dst, int packedSize) {
            if (!setupLess) {
                thrust::scatter(thrust::cuda::par(allocator), src, src + packedSize, d_inverse_permutation, dst);
            } else {
                const auto bool_iter = make_mask2bool_iterator(d_bitmask);
                const auto num_bools = num_bits<uint64_t>() * n;

                auto tabulate_it = thrust::make_tabulate_output_iterator(fetch_write{dst, src});

                thrust::counting_iterator<int> iter(0);
                thrust::copy_if(thrust::cuda::par(allocator), iter, iter + num_bools, bool_iter, tabulate_it, cuda::std::identity{});
            }
        };

        void print(uint64_t *h_bitmask) {
            UNUSED(h_bitmask);
            // Nothing to print
        };
};

