
#include "../../common_test_header.hpp"
#include "../test_utils_assertions.hpp"
#include "../test_utils_data_generation.hpp"

#include "../../common/utils_device_ptr.hpp"

#include <rocprim/block/block_store_func.hpp>
#include <rocprim/detail/merge_path.hpp>

template<int IPT, class T, class Op>
__global__
void merge_kernel(T* shared, rocprim::detail::range_t<> range, Op compare_function)
{
    T outputs[IPT];

    rocprim::detail::serial_merge(shared, outputs, range, compare_function);

    rocprim::block_store_direct_blocked(0, shared, outputs, range.end2);
}

template<int IPT, int N, class T, class OpT>
void serial_merge(std::vector<T>& input,
                  std::vector<T>& output,
                  unsigned int    mid,
                  OpT             compare_function)
{
    static_assert(IPT >= N, "Kernel must be launched such that all items can be processed!");

    common::device_ptr<T> device_data(input);

    HIP_CHECK_LAUNCH_SYNC(
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(merge_kernel<IPT>),
            dim3(1),
            dim3(1),
            0,
            0,
            device_data.get(),
            rocprim::detail::range_t<>{0, mid, mid, N},
            compare_function
        )
    );

    output = device_data.load();
}

TEST(RocprimInternalMergePathTests, Basic)
{
    using T   = int;
    using OpT = rocprim::less<T>;

    constexpr int n   = 512;
    constexpr int m   = n / 3;
    constexpr int ipt = 2 * n;

    std::vector<T> x = test_utils::get_random_data<T>(n,
                                                      std::numeric_limits<T>::min(),
                                                      std::numeric_limits<T>::max(),
                                                      0);
    std::vector<T> y(n);

    std::sort(x.begin(), x.begin() + m);
    std::sort(x.begin() + m, x.end());

    serial_merge<ipt, n>(x, y, m, OpT{});

    std::sort(x.begin(), x.end());

    test_utils::assert_eq(x, y);
}
