/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

// Tests of the HIP backend functions behind the backend manager, in the
// initialized process of rocalution-unit-test

#include "unit_test_common.hpp"
#include "unit_test_types.hpp"

#include "base/hip/backend_hip.hpp"
#include "base/hip/hip_allocate_free.hpp"
#include "utils/allocate_free.hpp"

using namespace rocalution_unit;
using rocalution::_get_backend_descriptor;

namespace
{
    template <typename T>
    std::vector<T> device_values(const T* ptr, int64_t n)
    {
        std::unique_ptr<T[]> host(new T[n]);
        EXPECT_EQ(hipMemcpy(host.get(), ptr, sizeof(T) * n, hipMemcpyDeviceToHost), hipSuccess);
        return std::vector<T>(host.get(), host.get() + n);
    }

    // Whether ptr is page-locked host memory of the HIP runtime
    bool is_pinned(const void* ptr)
    {
        hipPointerAttribute_t attributes{};
        const hipError_t      status = hipPointerGetAttributes(&attributes, ptr);
        // A failed query must not reach the next CHECK_HIP_ERROR of the library
        static_cast<void>(hipGetLastError());
        return status == hipSuccess && attributes.type == hipMemoryTypeHost;
    }

    template <typename T>
    void expect_pinned_round_trip(bool pinned)
    {
        const int64_t n   = 5;
        T*            ptr = nullptr;
        rocalution::allocate_pinned(n, &ptr);
        ASSERT_NE(ptr, nullptr);
        EXPECT_EQ(is_pinned(ptr), pinned);

        std::vector<T> values = from_ints<T>({3, -1, 4, 1, -5});
        std::copy(values.begin(), values.end(), ptr);
        T* device = nullptr;
        rocalution::allocate_hip(n, &device);
        rocalution::copy_h2d(n, ptr, device);
        std::fill(ptr, ptr + n, T(0));
        rocalution::copy_d2h(n, device, ptr);
        rocalution::free_hip(&device);
        EXPECT_EQ(std::vector<T>(ptr, ptr + n), values);

        rocalution::free_pinned(&ptr);
        EXPECT_EQ(ptr, nullptr);

        rocalution::allocate_pinned(0, &ptr);
        EXPECT_EQ(ptr, nullptr);
        rocalution::free_pinned(&ptr);
        EXPECT_EQ(ptr, nullptr);
    }
}

// utils/def.hpp enables assert with NDEBUG, but the HIP sources include ROCm
// headers afterwards that disable it again, so release builds have no asserts there
#ifdef NDEBUG
#define REQUIRE_HIP_ASSERTS()                                                            \
    KNOWN_BUG("the HIP sources include ROCm headers after utils/def.hpp, which disable " \
              "assert again with NDEBUG")
#else
#define REQUIRE_HIP_ASSERTS()
#endif

class backend_hip : public testing::Test
{
protected:
    void SetUp() override
    {
        REQUIRE_BACKEND(Backend::Accelerator);
    }

private:
    DescriptorGuard guard_;
};

TEST_F(backend_hip, info_of_selected_device)
{
    testing::internal::CaptureStdout();
    rocalution::rocalution_info_hip(descriptor());
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("rocBLAS ver"), std::string::npos) << out;
    EXPECT_NE(out.find("rocSPARSE ver"), std::string::npos) << out;
    EXPECT_NE(out.find("Selected HIP device: " + std::to_string(descriptor().HIP_dev)),
              std::string::npos)
        << out;
}

TEST_F(backend_hip, info_without_selected_device)
{
    _get_backend_descriptor()->HIP_dev = -1;

    testing::internal::CaptureStdout();
    rocalution::rocalution_info_hip(descriptor());
    std::string out = testing::internal::GetCapturedStdout();

    EXPECT_NE(out.find("No HIP device is selected!"), std::string::npos) << out;
}

TEST_F(backend_hip, pinned_memory_is_page_locked_with_the_accelerator)
{
    for_each_type<FloatingTypes>([](auto type) {
        using T = typename decltype(type)::type;
        expect_pinned_round_trip<T>(true);
    });
}

TEST_F(backend_hip, pinned_memory_is_ordinary_host_memory_without_the_accelerator)
{
    _get_backend_descriptor()->accelerator = false;

    for_each_type<FloatingTypes>([](auto type) {
        using T = typename decltype(type)::type;
        expect_pinned_round_trip<T>(false);
    });
}

TEST_F(backend_hip, set_to_one_and_set_to_value_on_a_stream)
{
    const int   block_size = descriptor().HIP_block_size;
    hipStream_t stream     = *static_cast<hipStream_t*>(descriptor().HIP_stream_current);

    for_each_type<AllValueTypes>([&](auto type) {
        using T = typename decltype(type)::type;

        // More than one block, and a partial last block
        const int64_t n   = 2 * block_size + 3;
        T*            ptr = nullptr;
        rocalution::allocate_hip(n, &ptr);

        // Each step changes every value, also for bool
        auto zero = [&] {
            EXPECT_EQ(hipMemset(ptr, 0, n * sizeof(T)), hipSuccess);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        };

        zero();
        rocalution::set_to_value_hip(block_size, n, ptr, from_int<T>(7), true, stream);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(device_values(ptr, n), std::vector<T>(n, from_int<T>(7)));

        zero();
        rocalution::set_to_one_hip(block_size, n, ptr, true, stream);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(device_values(ptr, n), std::vector<T>(n, T(1)));

        rocalution::set_to_value_hip(block_size, 0, ptr, from_int<T>(6));
        rocalution::set_to_one_hip(block_size, 0, ptr);
        rocalution::set_to_value_hip(block_size, n - 1, ptr, from_int<T>(6));
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
        std::vector<T> expected(n, from_int<T>(6));
        expected.back() = T(1);
        EXPECT_EQ(device_values(ptr, n), expected);

        rocalution::free_hip(&ptr);
    });
}

// rocalution_init_hip() runs once per init_rocalution(), after stop_rocalution()
// has released the handles and streams of the previous initialization
TEST(backend_hip_DeathTest, init_requires_released_handles_and_streams)
{
    REQUIRE_BACKEND(Backend::Accelerator);
    REQUIRE_HIP_ASSERTS();

    int  dummy = 0;
    auto d     = _get_backend_descriptor();

    EXPECT_ASSERT_DEATH(rocalution::rocalution_init_hip(), "ROC_blas_handle == NULL");
    EXPECT_ASSERT_DEATH(
        {
            d->ROC_blas_handle = nullptr;
            rocalution::rocalution_init_hip();
        },
        "ROC_sparse_handle == NULL");
    EXPECT_ASSERT_DEATH(
        {
            d->ROC_blas_handle   = nullptr;
            d->ROC_sparse_handle = nullptr;
            rocalution::rocalution_init_hip();
        },
        "HIP_stream_default == NULL");
    EXPECT_ASSERT_DEATH(
        {
            d->ROC_blas_handle     = nullptr;
            d->ROC_sparse_handle   = nullptr;
            d->HIP_stream_default  = nullptr;
            d->HIP_stream_interior = &dummy;
            rocalution::rocalution_init_hip();
        },
        "HIP_stream_interior == NULL");
    EXPECT_ASSERT_DEATH(
        {
            d->ROC_blas_handle     = nullptr;
            d->ROC_sparse_handle   = nullptr;
            d->HIP_stream_default  = nullptr;
            d->HIP_stream_interior = nullptr;
            d->HIP_stream_ghost    = &dummy;
            rocalution::rocalution_init_hip();
        },
        "HIP_stream_ghost == NULL");
    EXPECT_ASSERT_DEATH(
        {
            d->ROC_blas_handle     = nullptr;
            d->ROC_sparse_handle   = nullptr;
            d->HIP_stream_default  = nullptr;
            d->HIP_stream_interior = nullptr;
            d->HIP_stream_ghost    = nullptr;
            d->HIP_stream_current  = &dummy;
            rocalution::rocalution_init_hip();
        },
        "HIP_stream_current == NULL");
}

TEST(backend_hip_DeathTest, allocation_beyond_the_device_memory_ends_the_program)
{
    REQUIRE_BACKEND(Backend::Accelerator);

    EXPECT_LOGGED_EXIT(
        {
            char* ptr = nullptr;
            rocalution::allocate_hip(int64_t(1) << 60, &ptr);
        },
        "HIP error: out of memory.*File: [^;]*hip_allocate_free\\.cpp;");
}

TEST(backend_hip_DeathTest, objects_require_the_hip_backend)
{
    REQUIRE_HIP_ASSERTS();

    rocalution::Rocalution_Backend_Descriptor none = descriptor();
    none.backend                                   = rocalution::None;

    EXPECT_ASSERT_DEATH(rocalution::_rocalution_init_base_hip_vector<double>(none),
                        "backend == HIP");
    EXPECT_ASSERT_DEATH(
        rocalution::_rocalution_init_base_hip_matrix<double>(none, rocalution::CSR, 1),
        "backend == HIP");
}

TEST(backend_hip_DeathTest, block_dimension_is_only_for_bcsr)
{
    REQUIRE_HIP_ASSERTS();

    EXPECT_ASSERT_DEATH(
        rocalution::_rocalution_init_base_hip_matrix<double>(descriptor(), rocalution::CSR, 2),
        "blockdim == 1");
}
