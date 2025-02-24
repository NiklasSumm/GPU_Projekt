#include <cuda/std/cstdint>
#include <cuda/std/iterator>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

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
		uint64_t *d_bitmask;
		int n;
	
	public:
		void setup(uint64_t *d_bitmask, int n) {
			this->d_bitmask = d_bitmask;
			this->n = n;
		};

		void apply(int *permutation, int packedSize) {
            UNUSED(packedSize);

            const auto bool_iter = make_mask2bool_iterator(d_bitmask);
            const auto num_bools = num_bits<uint64_t>() * n;

            thrust::counting_iterator<int> iter(0);
            thrust::copy_if(thrust::device, iter, iter + num_bools, bool_iter, permutation, cuda::std::identity{});
		};

        void pack(int *src, int *dst, int packedSize) {
            UNUSED(packedSize);

            const auto bool_iter = make_mask2bool_iterator(d_bitmask);
            const auto num_bools = num_bits<uint64_t>() * n;

            thrust::copy_if(thrust::device, src, src + num_bools, bool_iter, dst, cuda::std::identity{});
        };

        void unpack(int *src, int *dst, int packedSize) {
            UNUSED(packedSize);

            const auto bool_iter = make_mask2bool_iterator(d_bitmask);
            const auto num_bools = num_bits<uint64_t>() * n;

            auto tabulate_it = thrust::make_tabulate_output_iterator(fetch_write{dst, src});

            thrust::counting_iterator<int> iter(0);
            thrust::copy_if(thrust::device, iter, iter + num_bools, bool_iter, tabulate_it, cuda::std::identity{});
        };
	
		void print(uint64_t *h_bitmask) {
            UNUSED(h_bitmask);
			// Nothing to print
		};
};
