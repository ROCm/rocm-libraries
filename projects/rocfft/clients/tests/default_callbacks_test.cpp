// Copyright (C) 2022 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <memory>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include "../../shared/fftw_transform.h"
#include "../../shared/hip_object_wrapper.h"
#include "../../shared/params_gen.h"
#include "../../shared/rocfft_params.h"
#include "rocfft/rocfft.h"

// -------------------------------------
// default load callback definitions
// -------------------------------------

template <typename T>
__device__ T load_cb(T* data, size_t offset, void* cbdata, void* sharedMem)
{
    return data[offset];
}

__device__ auto load_cb_complex_double = load_cb<rocfft_complex<double>>;
__device__ auto load_cb_double         = load_cb<double>;
__device__ auto load_cb_complex_float  = load_cb<rocfft_complex<float>>;
__device__ auto load_cb_float          = load_cb<float>;

// -------------------------------------
// default store callback definitions
// -------------------------------------

template <typename T>
__device__ void store_cb(T* data, size_t offset, T element, void* cbdata, void* sharedMem)
{
    data[offset] = element;
}

__device__ auto store_cb_complex_double = store_cb<rocfft_complex<double>>;
__device__ auto store_cb_double         = store_cb<double>;
__device__ auto store_cb_complex_float  = store_cb<rocfft_complex<float>>;
__device__ auto store_cb_float          = store_cb<float>;

// -------------------------------------
// type traits definitions
// -------------------------------------

template <typename T>
struct is_hip_complex
{
    static const bool value = false;
};

template <>
struct is_hip_complex<rocfft_complex<float>>
{
    static const bool value = true;
};

template <>
struct is_hip_complex<rocfft_complex<double>>
{
    static const bool value = true;
};

// ASAN introduces some problems with mixing library and client
// callbacks, so skip these tests if it's enabled
#ifdef ADDRESS_SANITIZER
#define TEST_CALLBACK_CHECK_ASAN \
    GTEST_SKIP() << "mixed library/client callbacks not supported for ASAN";
#else
#define TEST_CALLBACK_CHECK_ASAN
#endif

// -------------------------------------
// test callbacks struct
// -------------------------------------

enum struct DefaultCallbackType
{
    LOAD,
    STORE,
};
struct Test_Callback
{
    Test_Callback(size_t                  _N,
                  size_t                  _dim,
                  rocfft_transform_type_e _frwd_transf_type,
                  rocfft_precision_e      _frwd_transf_precision,
                  DefaultCallbackType     _cb_type,
                  uint32_t                _seed)
        : N(_N)
        , dim(_dim)
        , fwrd_transf_type(_frwd_transf_type)
        , frwd_transf_precision(_frwd_transf_precision)
        , cb_type(_cb_type)
        , seed(_seed)
    {

        float  low_bound_f = -1.0f, up_bound_f = 1.0f;
        double low_bound_d = -1.0, up_bound_d = 1.0;

        std::vector<rocfft_complex<float>>  h_mem_out_f2, h_mem_out_no_cb_f2;
        std::vector<rocfft_complex<double>> h_mem_out_d2, h_mem_out_no_cb_d2;

        switch(fwrd_transf_type)
        {
        case rocfft_transform_type_complex_forward:
        {
            std::vector<rocfft_complex<float>>  h_mem_in_f2;
            std::vector<rocfft_complex<double>> h_mem_in_d2;

            (frwd_transf_precision == rocfft_precision_single)
                ? run(low_bound_f, up_bound_f, h_mem_in_f2, h_mem_out_f2, h_mem_out_no_cb_f2)
                : run(low_bound_d, up_bound_d, h_mem_in_d2, h_mem_out_d2, h_mem_out_no_cb_d2);

            break;
        }
        case rocfft_transform_type_real_forward:
        {
            std::vector<float>  h_mem_in_f;
            std::vector<double> h_mem_in_d;

            (frwd_transf_precision == rocfft_precision_single)
                ? run(low_bound_f, up_bound_f, h_mem_in_f, h_mem_out_f2, h_mem_out_no_cb_f2)
                : run(low_bound_d, up_bound_d, h_mem_in_d, h_mem_out_d2, h_mem_out_no_cb_d2);

            break;
        }
        default:
            break;
        }
    }

    size_t get_data_size()
    {
        // compute total data size
        size_t data_size = 1;
        for(size_t i = 0; i < dim; ++i)
        {
            data_size *= N;
        }

        return data_size;
    }

    template <typename Tin, typename Tout, typename Tbound>
    void run(Tbound             low_bound,
             Tbound             up_bound,
             std::vector<Tin>&  host_mem_in,
             std::vector<Tout>& host_mem_out,
             std::vector<Tout>& host_mem_out_no_cb)
    {
        auto data_sz = get_data_size();

        if(cb_type == DefaultCallbackType::LOAD)
            set_load_callback<Tin>();
        else if(cb_type == DefaultCallbackType::STORE)
            set_store_callback<Tout>();

        host_mem_in.resize(data_sz);

        if constexpr(!is_hip_complex<Tin>::value)
            init_data(low_bound, up_bound, host_mem_in);
        else
            init_data_complex(low_bound, up_bound, host_mem_in);

        if constexpr(!is_hip_complex<Tin>::value)
            data_sz = (data_sz / 2) + 1;

        host_mem_out.resize(data_sz);
        forward_transform(true, host_mem_in, host_mem_out);

        host_mem_out_no_cb.resize(data_sz);
        forward_transform(false, host_mem_in, host_mem_out_no_cb);

        validate_test<Tout, Tbound>(host_mem_out, host_mem_out_no_cb);
    }

    template <typename Tdata, typename Tbound>
    void init_data(const Tbound low_bound, const Tbound up_bound, std::vector<Tdata>& host_mem)
    {
        std::minstd_rand                       gen(seed);
        std::uniform_real_distribution<Tbound> dist(low_bound, up_bound);

        for(size_t i = 0; i < host_mem.size(); i++)
        {
            host_mem[i] = dist(gen);
        }
    }

    template <typename Tdata, typename Tbound>
    void init_data_complex(const Tbound        low_bound,
                           const Tbound        up_bound,
                           std::vector<Tdata>& host_mem)
    {
        std::minstd_rand                       gen(seed);
        std::uniform_real_distribution<Tbound> dist(low_bound, up_bound);

        for(size_t i = 0; i < host_mem.size(); i++)
        {
            host_mem[i].x = dist(gen);
            host_mem[i].y = dist(gen);
        }
    }

    template <typename Tin, typename Tout>
    void forward_transform(bool                    apply_callback,
                           const std::vector<Tin>& host_mem_in,
                           std::vector<Tout>&      host_mem_out)
    {
        rocfft_plan plan = nullptr;

        std::vector<size_t> lengths(dim, N);
        ASSERT_EQ(rocfft_plan_create(&plan,
                                     rocfft_placement_notinplace,
                                     fwrd_transf_type,
                                     frwd_transf_precision,
                                     dim,
                                     lengths.data(),
                                     1,
                                     nullptr),
                  rocfft_status_success);

        size_t work_buffer_size = 0;
        void*  work_buffer      = nullptr;

        ASSERT_EQ(rocfft_plan_get_work_buffer_size(plan, &work_buffer_size), rocfft_status_success);
        if(work_buffer_size)
        {
            ASSERT_EQ(hipMalloc(&work_buffer, work_buffer_size), hipSuccess);
        }

        hipStream_wrapper_t stream;
        stream.alloc();

        rocfft_execution_info info;
        ASSERT_EQ(rocfft_execution_info_create(&info), rocfft_status_success);
        ASSERT_EQ(rocfft_execution_info_set_stream(info, stream), rocfft_status_success);

        if(apply_callback)
        {
            if(cb_type == DefaultCallbackType::LOAD)
            {
                ASSERT_EQ(rocfft_execution_info_set_load_callback(info, &load_cb_host, nullptr, 0),
                          rocfft_status_success);
            }
            else if(cb_type == DefaultCallbackType::STORE)
            {
                ASSERT_EQ(
                    rocfft_execution_info_set_store_callback(info, &store_cb_host, nullptr, 0),
                    rocfft_status_success);
            }
        }

        gpubuf device_mem_in;
        size_t NbytesIn = host_mem_in.size() * sizeof(Tin);
        ASSERT_EQ(device_mem_in.alloc(NbytesIn), hipSuccess);

        EXPECT_EQ(
            hipMemcpy(device_mem_in.data(), host_mem_in.data(), NbytesIn, hipMemcpyHostToDevice),
            hipSuccess);

        gpubuf device_mem_out;
        size_t NbytesOut = host_mem_out.size() * sizeof(Tout);
        ASSERT_EQ(device_mem_out.alloc(NbytesOut), hipSuccess);

        void* in_ptr  = device_mem_in.data();
        void* out_ptr = device_mem_out.data();
        ASSERT_EQ(rocfft_execute(plan, &in_ptr, &out_ptr, info), rocfft_status_success);

        ASSERT_EQ(hipMemcpy(host_mem_out.data(), out_ptr, NbytesOut, hipMemcpyDeviceToHost),
                  hipSuccess);

        ASSERT_EQ(rocfft_execution_info_destroy(info), rocfft_status_success);

        ASSERT_EQ(rocfft_plan_destroy(plan), rocfft_status_success);

        ASSERT_EQ(hipFree(work_buffer), hipSuccess);
    }

    template <typename Tout, typename Tbound>
    void validate_test(const std::vector<Tout>& host_mem_out,
                       const std::vector<Tout>& host_mem_out_no_cb)
    {
        auto diff = distance_1to1_complex(
            reinterpret_cast<const rocfft_complex<Tbound>*>(host_mem_out.data()),
            reinterpret_cast<const rocfft_complex<Tbound>*>(host_mem_out_no_cb.data()),
            host_mem_out.size(),
            1,
            1,
            host_mem_out.size(),
            1,
            host_mem_out_no_cb.size(),
            nullptr,
            type_epsilon<Tbound>(),
            {0},
            {0});

        EXPECT_LT(diff.l_inf, type_epsilon<Tbound>());
    }

    // ------------------------------------------------
    // set_load_callback template specializations
    // ------------------------------------------------

    template <typename T>
    void set_load_callback(){};

    template <>
    void set_load_callback<rocfft_complex<double>>()
    {
        EXPECT_EQ(
            hipMemcpyFromSymbol(&load_cb_host, HIP_SYMBOL(load_cb_complex_double), sizeof(void*)),
            hipSuccess);
    };

    template <>
    void set_load_callback<double>()
    {
        EXPECT_EQ(hipMemcpyFromSymbol(&load_cb_host, HIP_SYMBOL(load_cb_double), sizeof(void*)),
                  hipSuccess);
    };

    template <>
    void set_load_callback<rocfft_complex<float>>()
    {
        EXPECT_EQ(
            hipMemcpyFromSymbol(&load_cb_host, HIP_SYMBOL(load_cb_complex_float), sizeof(void*)),
            hipSuccess);
    };

    template <>
    void set_load_callback<float>()
    {
        EXPECT_EQ(hipMemcpyFromSymbol(&load_cb_host, HIP_SYMBOL(load_cb_float), sizeof(void*)),
                  hipSuccess);
    };

    // ------------------------------------------------
    // set_store_callback template specializations
    // ------------------------------------------------

    template <typename T>
    void set_store_callback(){};

    template <>
    void set_store_callback<rocfft_complex<double>>()
    {
        EXPECT_EQ(
            hipMemcpyFromSymbol(&store_cb_host, HIP_SYMBOL(store_cb_complex_double), sizeof(void*)),
            hipSuccess);
    };

    template <>
    void set_store_callback<double>()
    {
        EXPECT_EQ(hipMemcpyFromSymbol(&store_cb_host, HIP_SYMBOL(store_cb_double), sizeof(void*)),
                  hipSuccess);
    };

    template <>
    void set_store_callback<rocfft_complex<float>>()
    {
        EXPECT_EQ(
            hipMemcpyFromSymbol(&store_cb_host, HIP_SYMBOL(store_cb_complex_float), sizeof(void*)),
            hipSuccess);
    };

    template <>
    void set_store_callback<float>()
    {
        EXPECT_EQ(hipMemcpyFromSymbol(&store_cb_host, HIP_SYMBOL(store_cb_float), sizeof(void*)),
                  hipSuccess);
    };

    size_t                  N   = 0;
    size_t                  dim = 0;
    rocfft_transform_type_e fwrd_transf_type;
    rocfft_precision_e      frwd_transf_precision;
    DefaultCallbackType     cb_type;
    uint32_t                seed          = 0;
    void*                   store_cb_host = nullptr;
    void*                   load_cb_host  = nullptr;
};

// -------------------------------------------------------------------
// Test forward transforms in single/double precision with real and
// complex data inputs and having only a load callback set.
// -------------------------------------------------------------------

#ifndef WIN32
TEST(rocfft_UnitTest, default_load_callback_complex_single)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(256,
                       1,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       DefaultCallbackType::LOAD,
                       1);
}

TEST(rocfft_UnitTest, default_load_callback_complex_double)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(512,
                       1,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_double,
                       DefaultCallbackType::LOAD,
                       2);
}

TEST(rocfft_UnitTest, default_load_callback_real_single)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(1024,
                       1,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_single,
                       DefaultCallbackType::LOAD,
                       3);
}

TEST(rocfft_UnitTest, default_load_callback_real_double)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(2048,
                       1,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_double,
                       DefaultCallbackType::LOAD,
                       4);
}

// -------------------------------------------------------------------
// Test forward transforms in single/double precision with real and
// complex data inputs and having only a store callback set.
// -------------------------------------------------------------------

TEST(rocfft_UnitTest, default_store_callback_complex_single)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(256,
                       1,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_single,
                       DefaultCallbackType::STORE,
                       5);
}

TEST(rocfft_UnitTest, default_store_callback_complex_double)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(512,
                       1,
                       rocfft_transform_type_complex_forward,
                       rocfft_precision_double,
                       DefaultCallbackType::STORE,
                       6);
}

TEST(rocfft_UnitTest, default_store_callback_real_single)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(1024,
                       1,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_single,
                       DefaultCallbackType::STORE,
                       7);
}

TEST(rocfft_UnitTest, default_store_callback_real_double)
{
    if(hash_prob(random_seed, ::testing::UnitTest::GetInstance()->current_test_info()->name())
       > unittest_prob)
    {
        GTEST_SKIP();
    }

    TEST_CALLBACK_CHECK_ASAN;
    Test_Callback test(2048,
                       1,
                       rocfft_transform_type_real_forward,
                       rocfft_precision_double,
                       DefaultCallbackType::STORE,
                       8);
}
#endif
