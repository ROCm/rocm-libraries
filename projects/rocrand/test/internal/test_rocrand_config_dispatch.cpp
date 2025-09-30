// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "rng/common.hpp"
#include "rng/config_types.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>

__global__
void write_target_arch(rocrand_impl::host::target_arch* dest_arch)
{
    constexpr auto arch = rocrand_impl::host::get_device_arch();
    *dest_arch          = arch;
}

static constexpr rocrand_rng_type dummy_rng_type = rocrand_rng_type(0);

namespace rocrand_impl::host
{

template<class T>
struct generator_config_defaults<dummy_rng_type, T>
{
    static constexpr inline unsigned int threads = 256;
    static constexpr inline unsigned int blocks  = 1;
};

template<>
struct generator_config_defaults<dummy_rng_type, double>
{
    static constexpr inline unsigned int threads = 512;
    static constexpr inline unsigned int blocks  = 2;
};

template<>
struct generator_config_defaults<dummy_rng_type, half>
{
    static constexpr inline unsigned int threads = 512;
    static constexpr inline unsigned int blocks  = 7;
};

} // namespace rocrand_impl::host

template<class T,
         unsigned int BlockSize = rocrand_impl::host::default_config_provider<
                                      dummy_rng_type>::template device_config<T>(true)
                                      .threads>
__global__ __launch_bounds__(BlockSize)
void write_config(unsigned int* block_size, unsigned int* grid_size)
{
    if(blockIdx.x == 0 && threadIdx.x == 0 && BlockSize == blockDim.x)
    {
        *block_size = blockDim.x;
        *grid_size  = gridDim.x;
    }
}

TEST(rocrand_config_dispatch_tests, host_matches_device)
{
    const hipStream_t stream = 0;

    rocrand_impl::host::target_arch host_arch;
    HIP_CHECK(rocrand_impl::host::get_device_arch(stream, host_arch));

    rocrand_impl::host::target_arch* device_arch_ptr;
    HIP_CHECK(hipMallocHelper(&device_arch_ptr, sizeof(*device_arch_ptr)));

    hipLaunchKernelGGL(write_target_arch, dim3(1), dim3(1), 0, stream, device_arch_ptr);
    HIP_CHECK(hipGetLastError());

    rocrand_impl::host::target_arch device_arch;
    HIP_CHECK(hipMemcpy(&device_arch, device_arch_ptr, sizeof(device_arch), hipMemcpyDeviceToHost));

    ASSERT_NE(host_arch, rocrand_impl::host::target_arch::invalid);
    ASSERT_EQ(host_arch, device_arch);

    HIP_CHECK(hipFree(device_arch_ptr));
}

TEST(rocrand_config_dispatch_tests, parse_common_architectures)
{
    using rocrand_impl::host::parse_gcn_arch;
    using rocrand_impl::host::target_arch;

    ASSERT_EQ(parse_gcn_arch(""), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("not a gfx arch"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch(":"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("g:"), target_arch::unknown);

    ASSERT_EQ(parse_gcn_arch("gfx900:sramecc+"), target_arch::gfx900);
    ASSERT_EQ(parse_gcn_arch("gfx906:::"), target_arch::gfx906);
    ASSERT_EQ(parse_gcn_arch("gfx908:"), target_arch::gfx908);
    ASSERT_EQ(parse_gcn_arch("gfx90a:sramecc+:xnack-"), target_arch::gfx90a);
}

TEST(rocrand_config_dispatch_tests, get_config_on_host_and_device)
{
    using T = unsigned int;

    const hipStream_t stream = 0;

    unsigned int *d_block_size{}, *d_grid_size{};
    HIP_CHECK(hipMallocHelper(&d_block_size, sizeof(*d_block_size)));
    HIP_CHECK(hipMallocHelper(&d_grid_size, sizeof(*d_grid_size)));

    rocrand_impl::host::generator_config config{};
    const hipError_t err = rocrand_impl::host::get_generator_config<dummy_rng_type, T>(
        stream,
        ROCRAND_ORDERING_PSEUDO_DEFAULT,
        config);
    HIP_CHECK(err);

    hipLaunchKernelGGL(write_config<T>,
                       dim3(config.blocks),
                       dim3(config.threads),
                       0,
                       stream,
                       d_block_size,
                       d_grid_size);
    HIP_CHECK(hipGetLastError());

    unsigned int block_size{}, grid_size{};
    HIP_CHECK(hipMemcpy(&block_size, d_block_size, sizeof(block_size), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&grid_size, d_grid_size, sizeof(grid_size), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_block_size));
    HIP_CHECK(hipFree(d_grid_size));

    ASSERT_EQ(block_size, config.threads);
    ASSERT_EQ(grid_size, config.blocks);
}

#if USE_DEVICE_DISPATCH
TEST(rocrand_config_dispatch_tests, device_id_from_stream)
{
    using rocrand_impl::host::get_device_from_stream;
    hipDevice_t device_id;
    HIP_CHECK(hipGetDevice(&device_id));

    int                          result;
    static constexpr hipStream_t default_stream = 0;
    HIP_CHECK(get_device_from_stream(default_stream, result));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(get_device_from_stream(hipStreamPerThread, result));
    ASSERT_EQ(result, device_id);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);
}

template<class ConfigProvider>
__global__
void least_common_grid_size_kernel(unsigned int* least_common_grid_size, rocrand_ordering order)
{
    *least_common_grid_size = rocrand_impl::host::get_least_common_grid_size<ConfigProvider>(
        rocrand_impl::host::is_ordering_dynamic(order));
}

TEST(rocrand_config_dispatch_tests, default_config_provider)
{
    using config_provider = rocrand_impl::host::default_config_provider<dummy_rng_type>;
    static constexpr hipStream_t      default_stream = 0;
    static constexpr rocrand_ordering ordering       = ROCRAND_ORDERING_PSEUDO_DEFAULT;

    rocrand_impl::host::generator_config config{};
    ASSERT_EQ(config_provider::host_config<unsigned int>(default_stream, ordering, config),
              hipSuccess);
    ASSERT_EQ(config.blocks, 1);
    ASSERT_EQ(config.threads, 256);

    config = {};
    ASSERT_EQ(config_provider::host_config<double>(default_stream, ordering, config), hipSuccess);
    ASSERT_EQ(config.blocks, 2);
    ASSERT_EQ(config.threads, 512);

    unsigned int least_common_grid_size{};
    ASSERT_EQ(
        rocrand_impl::host::get_least_common_grid_size<config_provider>(default_stream,
                                                                        ordering,
                                                                        least_common_grid_size),
        hipSuccess);
    ASSERT_EQ(least_common_grid_size, 512 * 2 * 7);

    unsigned int* d_least_common_grid_size{};
    HIP_CHECK(hipMalloc(&d_least_common_grid_size, sizeof(*d_least_common_grid_size)));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(least_common_grid_size_kernel<config_provider>),
                       1,
                       1,
                       0,
                       default_stream,
                       d_least_common_grid_size,
                       ordering);
    unsigned int h_least_common_grid_size{};
    HIP_CHECK(hipMemcpy(&h_least_common_grid_size,
                        d_least_common_grid_size,
                        sizeof(h_least_common_grid_size),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_least_common_grid_size));

    ASSERT_EQ(least_common_grid_size, h_least_common_grid_size);
}

template<class ConfigProvider>
__global__
void config_selector_kernel(unsigned int* output)
{
    if(threadIdx.x == 0 && blockIdx.x == 0)
    {
        output[0] = ConfigProvider::template device_config<unsigned short>(true).blocks;
        output[1] = ConfigProvider::template device_config<unsigned short>(true).threads;
    }
}

namespace rocrand_impl::host
{

template<>
struct generator_config_selector<dummy_rng_type, unsigned short>
{
    __host__ __device__
    static constexpr unsigned int get_threads(const target_arch arch)
    {
        if(arch == target_arch::gfx906)
            return 64;
        return generator_config_defaults<dummy_rng_type, unsigned short>::threads;
    }

    __host__ __device__
    static constexpr unsigned int get_blocks(const target_arch /*arch*/)
    {
        return generator_config_defaults<dummy_rng_type, unsigned short>::blocks;
    }
};

} // namespace rocrand_impl::host

TEST(rocrand_config_dispatch_tests, config_selection)
{
    unsigned int*         d_output{};
    constexpr std::size_t size = 2;
    HIP_CHECK(hipMallocHelper(&d_output, size * sizeof(*d_output)));

    using config_provider_t = rocrand_impl::host::default_config_provider<dummy_rng_type>;
    config_provider_t                    config_provider{};
    rocrand_impl::host::generator_config config{};

    static constexpr hipStream_t      default_stream = 0;
    static constexpr rocrand_ordering ordering       = ROCRAND_ORDERING_PSEUDO_DYNAMIC;
    HIP_CHECK(config_provider.host_config<unsigned short>(default_stream, ordering, config));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(config_selector_kernel<config_provider_t>),
                       config.blocks,
                       config.threads,
                       0,
                       default_stream,
                       d_output);
    HIP_CHECK(hipGetLastError());

    std::array<unsigned int, 2> h_output{};
    HIP_CHECK(
        hipMemcpy(h_output.data(), d_output, size * sizeof(*d_output), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_output));

    ASSERT_EQ(config.blocks, h_output[0]);
    ASSERT_EQ(config.threads, h_output[1]);
}
#endif // USE_DEVICE_DISPATCH
