// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_rocrand_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/config_types.hpp>
#include <rng/mt19937.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace rocrand_impl::host;

// Generator API tests
using mt19937_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<mt19937_generator, ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(mt19937_generator,
                               generator_prng_tests,
                               mt19937_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(mt19937_generator,
                               generator_prng_continuity_tests,
                               mt19937_generator_prng_tests_types);

#ifdef CODE_COVERAGE_ENABLED
#include "test_rocrand_host_prng.hpp"

using rocrand_impl::host::mt19937_generator_host;
using mt19937_generator_prng_host_tests_types
    = ::testing::Types<generator_prng_host_tests_params<mt19937_generator_host<true>,
                                                        ROCRAND_ORDERING_PSEUDO_DEFAULT>>;

INSTANTIATE_TYPED_TEST_SUITE_P(mt19937_host_generator,
                               generator_prng_host_tests,
                               mt19937_generator_prng_host_tests_types);

//TODO: Figure out why this is causing compilation errors
// INSTANTIATE_TYPED_TEST_SUITE_P(mt19937_host_generator,
//                             generator_prng_continuity_host_tests,
//                             mt19937_generator_prng_host_tests_types);
#endif //CODE_COVERAGE_ENABLED

// mt19937-specific generator API tests
template<class Params>
struct mt19937_generator_prng_tests : public ::testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

TYPED_TEST_SUITE(mt19937_generator_prng_tests, mt19937_generator_prng_tests_types);

// Check that that heads and tails are generated correctly for misaligned pointers or sizes.
template<typename T, typename Generator, class ConfigProvider, typename GenerateFunc>
void head_and_tail_test(GenerateFunc generate_func, rocrand_ordering ordering, unsigned int divisor)
{
    generator_config config;
    HIP_CHECK(ConfigProvider::template host_config<T>(0, ordering, config));

    const unsigned int generator_count
        = config.threads * config.blocks / mt19937_octo_engine::threads_per_generator;

    const size_t stride = mt19937_constants::n * generator_count * divisor;

    // Large sizes are used for triggering all code paths in the kernels.
    std::vector<size_t>
        sizes{stride, 1, stride * 2 + 45651, 5, stride * 3 + 123, 6, 45, stride - 12};

    const size_t max_size             = *std::max_element(sizes.cbegin(), sizes.cend());
    const size_t canary_size          = 16;
    const size_t max_size_with_canary = max_size + canary_size * 2;

    const T canary = std::numeric_limits<T>::max();

    Generator g;
    g.set_order(ordering);

    std::vector<T> host_data(max_size_with_canary);
    T*             data;
    HIP_CHECK(hipMalloc(&data, sizeof(T) * max_size_with_canary));

    for(size_t offset : {0, 1, 2, 3})
    {
        for(size_t s : sizes)
        {
            const size_t s_with_canary = s + canary_size * 2;
            for(size_t i = 0; i < s_with_canary; i++)
            {
                host_data[i] = canary;
            }
            HIP_CHECK(
                hipMemcpy(data, host_data.data(), sizeof(T) * s_with_canary, hipMemcpyDefault));

            generate_func(g, data + canary_size + offset, s);

            HIP_CHECK(
                hipMemcpy(host_data.data(), data, sizeof(T) * s_with_canary, hipMemcpyDefault));

            // Check that the generator does not write more values than needed for head and tail
            // (so canary areas, or memory before and after data passed to generate(), are intact)
            for(size_t i = 0; i < canary_size + offset; i++)
            {
                ASSERT_EQ(host_data[i], canary);
            }
            for(size_t i = s_with_canary - (canary_size - offset); i < s_with_canary; i++)
            {
                ASSERT_EQ(host_data[i], canary);
            }

            // Check if head and tail are generated (canary value, used as an initial value,
            // can not be generated because it is not in the range of the distribution)
            size_t incorrect = 0;
            for(size_t i = canary_size + offset; i < s_with_canary - (canary_size - offset); i++)
            {
                if(host_data[i] == canary)
                {
                    incorrect++;
                }
            }
            ASSERT_EQ(incorrect, 0);
        }
    }
    HIP_CHECK(hipFree(data));
}

TYPED_TEST(mt19937_generator_prng_tests, head_and_tail_normal_float_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    head_and_tail_test<float, generator_t, ConfigProvider>(
        [](mt19937_generator& g, float* data, size_t s) { g.generate_normal(data, s, 0.0f, 1.0f); },
        ordering,
        2);
}

TYPED_TEST(mt19937_generator_prng_tests, head_and_tail_normal_double_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    head_and_tail_test<double, generator_t, ConfigProvider>(
        [](mt19937_generator& g, double* data, size_t s) { g.generate_normal(data, s, 0.0, 1.0); },
        ordering,
        2);
}

TYPED_TEST(mt19937_generator_prng_tests, head_and_tail_log_normal_float_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    head_and_tail_test<float, generator_t, ConfigProvider>(
        [](mt19937_generator& g, float* data, size_t s)
        { g.generate_log_normal(data, s, 0.0f, 1.0f); },
        ordering,
        2);
}

TYPED_TEST(mt19937_generator_prng_tests, head_and_tail_log_normal_double_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    head_and_tail_test<double, generator_t, ConfigProvider>(
        [](mt19937_generator& g, double* data, size_t s)
        { g.generate_log_normal(data, s, 0.0, 1.0); },
        ordering,
        2);
}

// Check if changing distribution sets m_start_input correctly
template<typename T0,
         typename T1,
         typename Generator,
         class ConfigProvider,
         typename GenerateFunc0,
         typename GenerateFunc1>
void change_distribution_test(GenerateFunc0    generate_func0,
                              GenerateFunc1    generate_func1,
                              size_t           size0,
                              size_t           start1,
                              rocrand_ordering ordering)
{
    SCOPED_TRACE(testing::Message() << "size0 = " << size0 << " start1 = " << start1);

    generator_config config;
    // Configs for mt19937 are independent of type, so just use T0
    HIP_CHECK(ConfigProvider::template host_config<T0>(0, ordering, config));

    const size_t size1 = config.threads * config.blocks * 3;

    T0* data0;
    T1* data10;
    T1* data11;
    HIP_CHECK(hipMalloc(&data0, sizeof(T0) * size0));
    HIP_CHECK(hipMalloc(&data10, sizeof(T1) * size1));
    HIP_CHECK(hipMalloc(&data11, sizeof(T1) * (start1 + size1)));

    Generator g0;
    // Generate the first distribution
    generate_func0(g0, data0, size0);
    // Change distribution to the second
    generate_func1(g0, data10, size1);

    Generator g1;
    // Generate the second distribution considering that first `start1` values correspond to
    // `size0` values of the first distribution and some discarded values
    generate_func1(g1, data11, start1 + size1);

    std::vector<T1> host_data10(size1);
    std::vector<T1> host_data11(size1);
    HIP_CHECK(hipMemcpy(host_data10.data(), data10, sizeof(T1) * size1, hipMemcpyDefault));
    // Ignore `start1` values
    HIP_CHECK(hipMemcpy(host_data11.data(), data11 + start1, sizeof(T1) * size1, hipMemcpyDefault));

    for(size_t i = 0; i < size1; i++)
    {
        ASSERT_EQ(host_data10[i], host_data11[i]);
    }

    HIP_CHECK(hipFree(data0));
    HIP_CHECK(hipFree(data10));
    HIP_CHECK(hipFree(data11));
}

TYPED_TEST(mt19937_generator_prng_tests, change_distribution0_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    generator_config config;
    // Configs for mt19937 are independent, just use void
    HIP_CHECK(ConfigProvider::template host_config<void>(0, ordering, config));

    const size_t s = config.threads * config.blocks;

    // Larger type (normal float) to smaller type (uniform uint)
    std::vector<std::pair<size_t, size_t>> test_cases{
        {         (s + 4) * 2, s * 4},
        {(s * 2 + s - 10) * 2, s * 6},
        {         (s * 3) * 2, s * 6},
        {         (s * 4) * 2, s * 8},
    };
    for(auto test_case : test_cases)
    {
        change_distribution_test<float, unsigned int, generator_t, ConfigProvider>(
            [](mt19937_generator& g, float* data, size_t s)
            { g.generate_normal(data, s, 0.0f, 1.0f); },
            [](mt19937_generator& g, unsigned int* data, size_t s) { g.generate(data, s); },
            test_case.first,
            test_case.second,
            ordering);
    }
}

TYPED_TEST(mt19937_generator_prng_tests, change_distribution1_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    generator_config config;
    // Configs for mt19937 are independent, just use void
    HIP_CHECK(ConfigProvider::template host_config<void>(0, ordering, config));

    const size_t s = config.threads * config.blocks;

    // Smaller type (uniform float) to larger type (normal double)
    std::vector<std::pair<size_t, size_t>> test_cases{
        {s * 2 + 100,  (s * 1) * 2},
        { s * 4 + 10,  (s * 2) * 2},
        {      s * 2,  (s * 1) * 2},
        {      s * 8,  (s * 2) * 2},
        {     s * 77, (s * 19) * 2}
    };
    for(auto test_case : test_cases)
    {
        change_distribution_test<float, double, generator_t, ConfigProvider>(
            [](mt19937_generator& g, float* data, size_t s) { g.generate_uniform(data, s); },
            [](mt19937_generator& g, double* data, size_t s)
            { g.generate_normal(data, s, 0.0, 1.0); },
            test_case.first,
            test_case.second,
            ordering);
    }
}

TYPED_TEST(mt19937_generator_prng_tests, change_distribution2_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    generator_config config;
    // Configs for mt19937 are independent, just use void
    HIP_CHECK(ConfigProvider::template host_config<void>(0, ordering, config));

    const size_t s = config.threads * config.blocks;

    // Smaller type (uniform double) to larger type (normal double)
    std::vector<std::pair<size_t, size_t>> test_cases{
        {s * 2 + 400, (s * 2) * 2},
        { s * 5 + 10, (s * 3) * 2},
        {      s * 3, (s * 2) * 2},
        {      s * 4, (s * 2) * 2},
    };
    for(auto test_case : test_cases)
    {
        change_distribution_test<double, double, generator_t, ConfigProvider>(
            [](mt19937_generator& g, double* data, size_t s) { g.generate_uniform(data, s); },
            [](mt19937_generator& g, double* data, size_t s)
            { g.generate_normal(data, s, 0.0, 1.0); },
            test_case.first,
            test_case.second,
            ordering);
    }
}

TYPED_TEST(mt19937_generator_prng_tests, change_distribution3_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using ConfigProvider                = default_config_provider<generator_t::type()>;
    generator_config config;
    // Configs for mt19937 are independent, just use void
    HIP_CHECK(ConfigProvider::template host_config<void>(0, ordering, config));

    const size_t s = config.threads * config.blocks;

    // Larger type (normal double) to smaller type (uniform ushort)
    std::vector<std::pair<size_t, size_t>> test_cases{
        {     100 * 2,  s * 8},
        {(s + 10) * 2, s * 16},
        { (s * 2) * 2, s * 16},
        { (s * 3) * 2, s * 24},
    };
    for(auto test_case : test_cases)
    {
        change_distribution_test<double, unsigned short, generator_t, ConfigProvider>(
            [](mt19937_generator& g, double* data, size_t s)
            { g.generate_normal(data, s, 0.0, 1.0); },
            [](mt19937_generator& g, unsigned short* data, size_t s) { g.generate(data, s); },
            test_case.first,
            test_case.second,
            ordering);
    }
}

// mt19937-specific generator API continuity tests
template<class Params>
struct mt19937_generator_prng_continuity_tests : public ::testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;
};

TYPED_TEST_SUITE(mt19937_generator_prng_continuity_tests, mt19937_generator_prng_tests_types);

// Check that subsequent generations of different sizes produce one
// sequence without gaps, no matter how many values are generated per call.
template<typename T,
         typename Generator,
         typename GenerateFunc,
         std::enable_if_t<std::is_same<Generator, mt19937_generator>::value, bool> = true>
void continuity_test(GenerateFunc     generate_func,
                     rocrand_ordering ordering,
                     unsigned int     divisor = 1)
{
    using ConfigProvider = default_config_provider<mt19937_generator::type()>;

    generator_config config;
    HIP_CHECK(ConfigProvider::template host_config<T>(0, ordering, config));

    const unsigned int generator_count
        = config.threads * config.blocks / mt19937_octo_engine::threads_per_generator;

    const size_t stride = mt19937_constants::n * generator_count * divisor;

    // Large sizes are used for triggering all code paths in the kernels (generating of middle,
    // start and end sequences).
    std::vector<size_t> sizes0{stride,
                               2,
                               stride,
                               100,
                               1,
                               24783,
                               stride / 2,
                               3 * stride + 704400,
                               2,
                               stride + 776543,
                               44176};
    std::vector<size_t> sizes1{2 * stride,
                               1024,
                               55,
                               65536,
                               stride / 2,
                               stride + 623456,
                               3 * stride - 300000,
                               1048576,
                               111331};

    // Round by the distribution's granularity (2 for normals, 2 for short and half, 4 for uchar).
    // Sizes not divisible by the granularity or pointers not aligned by it work but without strict
    // continuity.
    if(divisor > 1)
    {
        for(size_t& s : sizes0)
            s = (s + divisor - 1) & ~static_cast<size_t>(divisor - 1);
        for(size_t& s : sizes1)
            s = (s + divisor - 1) & ~static_cast<size_t>(divisor - 1);
    }

    const size_t size0 = std::accumulate(sizes0.cbegin(), sizes0.cend(), std::size_t{0});
    const size_t size1 = std::accumulate(sizes1.cbegin(), sizes1.cend(), std::size_t{0});
    const size_t size2 = std::min(size0, size1);

    mt19937_generator g0;
    g0.set_order(ordering);
    mt19937_generator g1;
    g1.set_order(ordering);
    mt19937_generator g2;
    g2.set_order(ordering);

    std::vector<T> host_data0(size0);
    std::vector<T> host_data1(size1);
    std::vector<T> host_data2(size2);

    size_t current0 = 0;
    for(size_t s : sizes0)
    {
        T* data0;
        HIP_CHECK(hipMalloc(&data0, sizeof(T) * s));
        HIP_CHECK(hipMemset(data0, -1, sizeof(T) * s));
        generate_func(g0, data0, s);
        HIP_CHECK(hipMemcpy(host_data0.data() + current0, data0, sizeof(T) * s, hipMemcpyDefault));
        current0 += s;
        HIP_CHECK(hipFree(data0));
    }
    size_t current1 = 0;
    for(size_t s : sizes1)
    {
        T* data1;
        HIP_CHECK(hipMalloc(&data1, sizeof(T) * s));
        HIP_CHECK(hipMemset(data1, -1, sizeof(T) * s));
        generate_func(g1, data1, s);
        HIP_CHECK(hipMemcpy(host_data1.data() + current1, data1, sizeof(T) * s, hipMemcpyDefault));
        current1 += s;
        HIP_CHECK(hipFree(data1));
    }
    T* data2;
    HIP_CHECK(hipMalloc(&data2, sizeof(T) * size2));
    HIP_CHECK(hipMemset(data2, -1, sizeof(T) * size2));
    generate_func(g2, data2, size2);
    HIP_CHECK(hipMemcpy(host_data2.data(), data2, sizeof(T) * size2, hipMemcpyDefault));
    HIP_CHECK(hipFree(data2));

    size_t incorrect = 0;
    for(size_t i = 0; i < size2; i++)
    {
        if constexpr(std::is_same<T, __half>::value)
        {
            if(__half2float(host_data0[i]) != __half2float(host_data1[i])
               || __half2float(host_data0[i]) != __half2float(host_data2[i]))
            {
                incorrect++;
            }
        }
        else
        {
            if(host_data0[i] != host_data1[i] || host_data0[i] != host_data2[i])
            {
                incorrect++;
            }
        }
    }
    ASSERT_EQ(incorrect, 0);
}

TYPED_TEST(mt19937_generator_prng_continuity_tests, continuity_uniform_short_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;

    typedef unsigned short output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering,
                                           2);
}

TYPED_TEST(mt19937_generator_prng_continuity_tests, continuity_uniform_half_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;

    typedef __half output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering,
                                           2);
}

// Engine API tests
template<class Generator>
struct mt19937_generator_engine_tests : public ::testing::Test
{
    using generator_t = Generator;
};

using mt19937_generator_engine_tests_types = ::testing::Types<mt19937_generator>;

TYPED_TEST_SUITE(mt19937_generator_engine_tests, mt19937_generator_engine_tests_types);

/// Initialize the octo engines for both generators. Skip \p subsequence_size for the first generator.
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void init_engines_kernel(mt19937_octo_engine* octo_engines,
                         const unsigned int*  engines,
                         unsigned int         subsequence_size)
{
    constexpr unsigned int n         = mt19937_constants::n;
    const unsigned int     thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int           engine_id = thread_id / mt19937_octo_engine::threads_per_generator;
    mt19937_octo_engine    engine    = octo_engines[thread_id];
    engine.gather(&engines[engine_id * n], threadIdx);

    engine.gen_next_n();
    if(engine_id == 0)
    {
        for(unsigned int i = 0; i < subsequence_size / n; i++)
        {
            engine.gen_next_n();
        }
    }

    octo_engines[thread_id] = engine;
}

/// Each generator produces \p n elements in its own \p data section.
__global__ __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE)
void generate_kernel(mt19937_octo_engine* engines,
                     unsigned int*        data,
                     unsigned int         elements_per_generator,
                     unsigned int         subsequence_size)
{
    constexpr unsigned int n                     = mt19937_constants::n;
    constexpr unsigned int threads_per_generator = mt19937_octo_engine::threads_per_generator;
    const unsigned int     local_thread_id       = threadIdx.x & 7U;
    const unsigned int     thread_id             = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int           engine_id             = thread_id / threads_per_generator;
    unsigned int*          ptr                   = data + engine_id * elements_per_generator;

    mt19937_octo_engine engine = engines[thread_id];

    unsigned int mti = engine_id == 0 ? subsequence_size % n : 0;

    for(size_t index = local_thread_id; index < elements_per_generator;
        index += threads_per_generator)
    {
        if(mti == n)
        {
            engine.gen_next_n();
            mti = 0;
        }
        ptr[index] = engine.get(mti / threads_per_generator);
        mti += threads_per_generator;
    }

    engines[thread_id] = engine;
}

TYPED_TEST(mt19937_generator_engine_tests, subsequence_test)
{
    using generator_t                                 = typename TestFixture::generator_t;
    using octo_engine_type                            = mt19937_octo_engine;
    constexpr unsigned int threads_per_generator      = mt19937_octo_engine::threads_per_generator;
    constexpr unsigned long long int seed             = 1ULL;
    constexpr unsigned int           subsequence_size = 552552U;
    // The size of the subsequence must be a multiple of threads per generator
    // otherwise the non-jumping generator cannot align due to generating
    // threads_per_generator per call
    static_assert(subsequence_size % threads_per_generator == 0,
                  "size of subsequence must be multiple of eight");
    constexpr unsigned int generator_count = 2U;
    constexpr unsigned int state_size      = mt19937_constants::n;

    // Constants to skip subsequence_size states.
    // Generated with tools/mt19937_precomputed_generator.cpp
    // clang-format off
    static constexpr unsigned int jump[mt19937_p_size] = {
          16836620U, 1241597017U, 2267910466U, 3327083746U, 1938175615U, 2308764182U, 3494318232U,  233458728U,
        3431172561U, 4034658304U,  540715081U, 2223083524U, 1290021546U,    4615588U, 2755265028U, 1487667568U,
         817793568U,  688998677U, 4177128730U, 1327584593U, 3083575336U, 3130198036U, 1730866499U,  236199859U,
        1319739361U, 1227620352U, 2030748367U,  338426818U, 3401904732U, 2068477099U,    9314332U,       1394U,
        1631647232U, 3360850049U, 3947386387U,  169910306U, 3403351184U, 2759828497U, 3936398567U, 3981649994U,
        3545643632U, 1211945956U, 4214442729U, 1516073261U, 1973528206U, 3127605291U, 1657881179U, 2639065177U,
        1695629247U, 1483473424U, 1922960899U,  147373172U, 2671913376U, 3824567940U,  719742235U, 3137653860U,
         464968244U, 1223024043U,     130661U, 1203785820U, 4020201862U, 2505398286U, 3526255407U,  419792716U,
         523023476U,  371258051U, 3673049403U, 1836542343U, 2302995407U,   89010016U,  345597150U,  215916134U,
         302835354U, 1549276007U,      78962U, 2610524389U, 3144872708U, 1810683989U, 3751221315U, 1590451824U,
        3344450054U,  700934502U,  110016935U, 2156795150U, 3785730224U, 2631375440U, 1974637886U, 3292329605U,
        3142139957U, 3701811334U, 1549078486U,  129980226U, 1391930951U, 2556241742U, 2185354446U,  887051003U,
        3413484806U, 1342283353U, 1424278535U, 2917569624U, 3429457066U,  924053705U, 4113066678U, 3805305864U,
        3627143398U, 4011722511U,  479136705U, 2075096001U, 1721089795U, 3074239461U, 4254620365U, 3246832812U,
        2600113446U, 2754943943U, 3388450324U, 1677024071U, 2500806419U,  158791876U, 3463832935U, 2458673960U,
        2747463520U, 3548197763U, 3182431084U,   17380539U, 1557533732U, 2107756592U,   46491733U, 1796341916U,
        1240657450U, 3670925904U,  546492343U, 4211712370U, 3978286571U, 2301647531U, 4277260054U, 1195041504U,
        3554107626U, 3536404767U, 3391935859U, 2250749215U,  882048618U, 2827024245U,  569173904U, 2149235115U,
          60945640U, 3833593866U,  956451456U, 1021006574U, 4236771596U, 2265058560U, 2215696731U,  778465250U,
        1751590318U,  741942625U, 3458004120U,  248619343U, 3843017115U,  839840654U, 1828044965U, 2617355055U,
        3779651646U,  525835946U,  537395281U,  992151708U, 1321781591U,  478234930U, 3884143138U, 2864985020U,
        2015099986U, 2965894308U, 1428387075U, 1310499846U, 1109267475U, 1075643877U, 2258267789U, 1069867669U,
        4205025922U, 2634970836U, 3132427367U, 2853906496U, 2678425777U, 4276765991U, 2575927964U, 2671947668U,
        1121017226U, 2080611588U, 2727225830U,  503316657U, 4042256386U, 1112199335U,  464744163U, 3572201075U,
         636055961U, 3899908992U,  892046540U,  896896758U,  393156791U,   49352486U, 3161122923U, 2585141935U,
        4006516250U, 2805665689U, 1866241881U,  462229762U, 1563106787U,  551098178U, 2128817785U, 3135100052U,
        4260031810U, 2726884032U,  892574702U, 3308689842U, 2326295075U, 3099353849U, 1166714571U, 1734201378U,
        3476395591U, 4061325047U, 2522521477U,  357030804U, 3726519752U, 2630348914U, 2394970464U,  539386556U,
         500917831U, 2573533705U, 3575402563U, 1930818072U, 2100596270U, 1470081741U, 1740257674U, 3408964191U,
        3883874908U, 3501867562U, 3979810829U,  491029434U, 3873155198U,  767047573U, 3512288254U, 3973630301U,
         918919925U,  505569179U, 2622866775U,  476301621U, 1653555785U,  588982683U, 2892810634U, 1481364624U,
        2900983412U, 1237365144U,  654858224U, 3888079105U,  880084185U, 2319840370U, 4149829702U, 2598788559U,
        1795697197U, 1141358839U, 1488115545U, 2023012969U, 1700767613U,  318307034U, 3741324886U, 2723508372U,
        2877869759U, 1089131774U, 3746529157U, 2032861327U, 1916908398U, 3979971761U, 1104333434U,  164965263U,
           2861111U, 4061197830U, 1227752574U, 1557686937U, 1372189256U, 1320514631U, 3006463383U,  995374202U,
        2475599921U, 2062684198U,  273818196U,  230102535U,  846172554U,  549643817U, 1291845833U,  313868405U,
        1684607830U, 3007524804U, 2696481972U, 1496449755U, 2336358181U, 2907909674U, 2046381710U, 1616425720U,
        2784933411U, 3488376037U, 4237079453U,  239405390U, 2260619063U, 4248573305U, 2538035035U, 3162243525U,
        1688422919U, 4214141536U, 2822423904U, 3616596572U, 1124198315U,  561546231U, 1450818809U, 1395937417U,
        2624134660U, 1180897824U, 1879143185U, 2716824526U,  269138413U,  147307449U, 2713808648U, 1397164069U,
        3082484694U, 3996518249U, 3822948465U, 2600900044U, 2073322101U, 3405864806U, 1465988883U, 2263895854U,
        1193632090U,  878563614U, 3633274523U,  832840620U, 1918686864U,   52700353U, 2164193894U, 4099319060U,
        1271654821U, 2934121786U, 4003740838U, 2025639926U, 1964764541U, 3711204924U, 1811665390U, 2651395047U,
        2574986913U, 2946806898U, 3374950428U,  407088658U, 3098475549U, 3678329103U, 1442862364U, 1149148015U,
        3829525455U, 1129287955U, 1691241488U, 2948237333U, 4111137958U, 3121299510U, 3228891983U, 1422674833U,
        1046218249U, 4255146817U, 1789888035U, 2790533305U, 2766564283U,  882036014U, 3138447493U, 1705216849U,
        4086442903U, 2084466724U, 1452448031U, 2518232572U, 2826320536U, 1155360986U, 1652202635U, 3192309572U,
        4278103747U, 4207611316U, 3471787642U, 3949425339U, 2428040116U, 3916643455U, 1047874548U, 3522678507U,
         679503438U, 3367670533U, 3305388393U, 1545534614U,  365979387U, 3813719383U, 2293866348U, 2311446870U,
        1294235417U, 1239874267U, 3149790803U, 3459617970U, 3553443070U, 4294547149U,  677815218U, 2480790846U,
          42062043U, 2099057395U,  608270600U,  555980767U, 3040155855U, 4220769542U,  442527978U, 4264098298U,
         748043208U, 1510266257U,  605080615U, 1870700985U, 2815631381U, 3546827936U,  389630240U,  530771376U,
        1023017605U,  570366154U,  943073743U, 2345589488U, 3331719336U, 3393668839U, 2125550217U, 3316487643U,
        2926149375U, 1563725876U, 2234336736U, 2691239566U, 4276736289U, 1534407008U,  634560074U,  537405234U,
        1007405586U, 3435919387U, 3526571791U, 1251978681U, 2423612524U, 4240584245U, 3077336530U, 1394628396U,
         667872456U, 3838386423U, 1264339781U, 2977905273U, 2493098225U, 1408656392U,  458665826U, 1193671488U,
        3192651130U, 3408345436U, 3416261569U, 2099306740U, 1457701667U, 1798196661U, 4078467644U, 1823475953U,
        3683947560U, 3421638713U, 1333246224U, 2253616524U, 3468957120U, 2028012393U, 3950669141U, 3379074497U,
         264072228U, 1478066089U, 3810691100U,  105177718U, 3239667112U,  548511418U,  462182385U, 1880566371U,
        3449712347U, 3945393228U, 1722247886U,  834860904U, 2023403204U, 1764955460U, 2314861647U, 2541662287U,
         174235623U, 1025135151U, 4183459969U,  140612992U, 2179351739U,  838789589U, 2321249467U, 3580648575U,
        1937895840U, 4082767776U, 4208618097U, 1150513792U, 3159007105U,  957333854U, 3121611318U, 1468699888U,
        2600870933U, 1887140383U, 1158880479U, 2021223243U, 4216226924U, 3362953576U,  148281745U,  743454457U,
        2055203028U, 1388596477U, 3430662102U,   14508704U, 3293390847U,  868061153U, 2803772691U,  167322815U,
         702986243U,  970691543U, 3211870732U,  132211178U, 1878239493U,  493828035U,  578005384U, 2083193988U,
         618003366U, 3516838252U, 1628942083U,  771893287U, 1783055259U, 2664147198U, 1123182254U, 3428230259U,
        4246995520U,  540075809U, 3006764017U, 2048903824U, 4018761289U, 1959828143U, 1058168427U, 2139631077U,
        2641577125U, 1819365340U,  391416789U,  408543984U, 2374873865U, 2638340220U, 2279187081U, 3509490132U,
         109546995U, 2943006029U,  301017297U, 2159298247U, 1201337642U, 3955051167U, 1131485785U, 4026925695U,
        2288659668U, 3259615238U,  610986470U, 1262822694U, 3447317355U, 3261746329U,  900784498U, 4163475604U,
        3571695718U,   95546624U,  765597843U, 1239045105U, 1375372467U, 2689038155U,  246401506U, 1717907899U,
        2072005013U, 2562942296U, 1328060883U, 3345146601U, 2369611890U, 1541866911U, 3503192374U, 1705689374U,
         444830279U,  308855830U, 1960063476U, 1642058452U, 1044063781U,  239034752U, 2929660102U, 2476585518U,
        3525477572U, 4104693897U, 2573076031U, 4190865194U, 2395897238U, 2400843904U, 1695065775U, 4178846862U,
         826627422U,  914883664U, 2172966192U,  375087119U, 1284236820U, 2458751356U, 2286795808U,  648305751U,
        2336236161U, 3238612623U, 3320228067U,     769191U,  430179840U, 2186883080U, 1430612668U,  973149413U,
        1121709821U,   90179392U,  411379749U,  552994832U,         10U, 4016980240U, 2433283182U, 1182819972U,
        2993305185U, 1410353515U, 2105574608U,      38722U, 1668746496U, 2299044730U, 4019202397U,          0U
    };
    // clang-format on

    // Starting from state zero or from a jump,
    // the generated number with index i's original position is permutation_table[i]
    constexpr unsigned int permutation_table[state_size] = {
        // clang-format off
        // First, eight special values
          0U, 113U, 170U, 283U, 340U, 397U, 510U, 567U,
        // Then, the regular pattern
          1U,   8U,  15U,  22U,  29U,  36U,  43U,  50U,   2U,   9U,  16U,  23U,  30U,  37U,  44U,  51U,   3U,  10U,  17U,  24U,  31U,  38U,  45U,  52U,   4U,  11U,  18U,  25U,  32U,  39U,  46U,  53U,   5U,  12U,  19U,  26U,  33U,  40U,  47U,  54U,   6U,  13U,  20U,  27U,  34U,  41U,  48U,  55U,   7U,  14U,  21U,  28U,  35U,  42U,  49U,  56U,
         57U,  64U,  71U,  78U,  85U,  92U,  99U, 106U,  58U,  65U,  72U,  79U,  86U,  93U, 100U, 107U,  59U,  66U,  73U,  80U,  87U,  94U, 101U, 108U,  60U,  67U,  74U,  81U,  88U,  95U, 102U, 109U,  61U,  68U,  75U,  82U,  89U,  96U, 103U, 110U,  62U,  69U,  76U,  83U,  90U,  97U, 104U, 111U,  63U,  70U,  77U,  84U,  91U,  98U, 105U, 112U,
        114U, 121U, 128U, 135U, 142U, 149U, 156U, 163U, 115U, 122U, 129U, 136U, 143U, 150U, 157U, 164U, 116U, 123U, 130U, 137U, 144U, 151U, 158U, 165U, 117U, 124U, 131U, 138U, 145U, 152U, 159U, 166U, 118U, 125U, 132U, 139U, 146U, 153U, 160U, 167U, 119U, 126U, 133U, 140U, 147U, 154U, 161U, 168U, 120U, 127U, 134U, 141U, 148U, 155U, 162U, 169U,
        171U, 178U, 185U, 192U, 199U, 206U, 213U, 220U, 172U, 179U, 186U, 193U, 200U, 207U, 214U, 221U, 173U, 180U, 187U, 194U, 201U, 208U, 215U, 222U, 174U, 181U, 188U, 195U, 202U, 209U, 216U, 223U, 175U, 182U, 189U, 196U, 203U, 210U, 217U, 224U, 176U, 183U, 190U, 197U, 204U, 211U, 218U, 225U, 177U, 184U, 191U, 198U, 205U, 212U, 219U, 226U,
        227U, 234U, 241U, 248U, 255U, 262U, 269U, 276U, 228U, 235U, 242U, 249U, 256U, 263U, 270U, 277U, 229U, 236U, 243U, 250U, 257U, 264U, 271U, 278U, 230U, 237U, 244U, 251U, 258U, 265U, 272U, 279U, 231U, 238U, 245U, 252U, 259U, 266U, 273U, 280U, 232U, 239U, 246U, 253U, 260U, 267U, 274U, 281U, 233U, 240U, 247U, 254U, 261U, 268U, 275U, 282U,
        284U, 291U, 298U, 305U, 312U, 319U, 326U, 333U, 285U, 292U, 299U, 306U, 313U, 320U, 327U, 334U, 286U, 293U, 300U, 307U, 314U, 321U, 328U, 335U, 287U, 294U, 301U, 308U, 315U, 322U, 329U, 336U, 288U, 295U, 302U, 309U, 316U, 323U, 330U, 337U, 289U, 296U, 303U, 310U, 317U, 324U, 331U, 338U, 290U, 297U, 304U, 311U, 318U, 325U, 332U, 339U,
        341U, 348U, 355U, 362U, 369U, 376U, 383U, 390U, 342U, 349U, 356U, 363U, 370U, 377U, 384U, 391U, 343U, 350U, 357U, 364U, 371U, 378U, 385U, 392U, 344U, 351U, 358U, 365U, 372U, 379U, 386U, 393U, 345U, 352U, 359U, 366U, 373U, 380U, 387U, 394U, 346U, 353U, 360U, 367U, 374U, 381U, 388U, 395U, 347U, 354U, 361U, 368U, 375U, 382U, 389U, 396U,
        398U, 405U, 412U, 419U, 426U, 433U, 440U, 447U, 399U, 406U, 413U, 420U, 427U, 434U, 441U, 448U, 400U, 407U, 414U, 421U, 428U, 435U, 442U, 449U, 401U, 408U, 415U, 422U, 429U, 436U, 443U, 450U, 402U, 409U, 416U, 423U, 430U, 437U, 444U, 451U, 403U, 410U, 417U, 424U, 431U, 438U, 445U, 452U, 404U, 411U, 418U, 425U, 432U, 439U, 446U, 453U,
        454U, 461U, 468U, 475U, 482U, 489U, 496U, 503U, 455U, 462U, 469U, 476U, 483U, 490U, 497U, 504U, 456U, 463U, 470U, 477U, 484U, 491U, 498U, 505U, 457U, 464U, 471U, 478U, 485U, 492U, 499U, 506U, 458U, 465U, 472U, 479U, 486U, 493U, 500U, 507U, 459U, 466U, 473U, 480U, 487U, 494U, 501U, 508U, 460U, 467U, 474U, 481U, 488U, 495U, 502U, 509U,
        511U, 518U, 525U, 532U, 539U, 546U, 553U, 560U, 512U, 519U, 526U, 533U, 540U, 547U, 554U, 561U, 513U, 520U, 527U, 534U, 541U, 548U, 555U, 562U, 514U, 521U, 528U, 535U, 542U, 549U, 556U, 563U, 515U, 522U, 529U, 536U, 543U, 550U, 557U, 564U, 516U, 523U, 530U, 537U, 544U, 551U, 558U, 565U, 517U, 524U, 531U, 538U, 545U, 552U, 559U, 566U,
        568U, 575U, 582U, 589U, 596U, 603U, 610U, 617U, 569U, 576U, 583U, 590U, 597U, 604U, 611U, 618U, 570U, 577U, 584U, 591U, 598U, 605U, 612U, 619U, 571U, 578U, 585U, 592U, 599U, 606U, 613U, 620U, 572U, 579U, 586U, 593U, 600U, 607U, 614U, 621U, 573U, 580U, 587U, 594U, 601U, 608U, 615U, 622U, 574U, 581U, 588U, 595U, 602U, 609U, 616U, 623U,
        // clang-format on
    };

    constexpr unsigned int rev_permutation_table[state_size] = {
        // clang-format off
          0U,   8U,  16U,  24U,  32U,  40U,  48U,  56U,
          9U,  17U,  25U,  33U,  41U,  49U,  57U,  10U,  18U,  26U,  34U,  42U,  50U,  58U,  11U,  19U,  27U,  35U,  43U,  51U,  59U,  12U,  20U,  28U,  36U,  44U,  52U,  60U,  13U,  21U,  29U,  37U,  45U,  53U,  61U,  14U,  22U,  30U,  38U,  46U,  54U,  62U,  15U,  23U,  31U,  39U,  47U,  55U,  63U,  64U,  72U,  80U,  88U,  96U, 104U, 112U,
         65U,  73U,  81U,  89U,  97U, 105U, 113U,  66U,  74U,  82U,  90U,  98U, 106U, 114U,  67U,  75U,  83U,  91U,  99U, 107U, 115U,  68U,  76U,  84U,  92U, 100U, 108U, 116U,  69U,  77U,  85U,  93U, 101U, 109U, 117U,  70U,  78U,  86U,  94U, 102U, 110U, 118U,  71U,  79U,  87U,  95U, 103U, 111U, 119U,   1U, 120U, 128U, 136U, 144U, 152U, 160U,
        168U, 121U, 129U, 137U, 145U, 153U, 161U, 169U, 122U, 130U, 138U, 146U, 154U, 162U, 170U, 123U, 131U, 139U, 147U, 155U, 163U, 171U, 124U, 132U, 140U, 148U, 156U, 164U, 172U, 125U, 133U, 141U, 149U, 157U, 165U, 173U, 126U, 134U, 142U, 150U, 158U, 166U, 174U, 127U, 135U, 143U, 151U, 159U, 167U, 175U,   2U, 176U, 184U, 192U, 200U, 208U,
        216U, 224U, 177U, 185U, 193U, 201U, 209U, 217U, 225U, 178U, 186U, 194U, 202U, 210U, 218U, 226U, 179U, 187U, 195U, 203U, 211U, 219U, 227U, 180U, 188U, 196U, 204U, 212U, 220U, 228U, 181U, 189U, 197U, 205U, 213U, 221U, 229U, 182U, 190U, 198U, 206U, 214U, 222U, 230U, 183U, 191U, 199U, 207U, 215U, 223U, 231U, 232U, 240U, 248U, 256U, 264U,
        272U, 280U, 233U, 241U, 249U, 257U, 265U, 273U, 281U, 234U, 242U, 250U, 258U, 266U, 274U, 282U, 235U, 243U, 251U, 259U, 267U, 275U, 283U, 236U, 244U, 252U, 260U, 268U, 276U, 284U, 237U, 245U, 253U, 261U, 269U, 277U, 285U, 238U, 246U, 254U, 262U, 270U, 278U, 286U, 239U, 247U, 255U, 263U, 271U, 279U, 287U,   3U, 288U, 296U, 304U, 312U,
        320U, 328U, 336U, 289U, 297U, 305U, 313U, 321U, 329U, 337U, 290U, 298U, 306U, 314U, 322U, 330U, 338U, 291U, 299U, 307U, 315U, 323U, 331U, 339U, 292U, 300U, 308U, 316U, 324U, 332U, 340U, 293U, 301U, 309U, 317U, 325U, 333U, 341U, 294U, 302U, 310U, 318U, 326U, 334U, 342U, 295U, 303U, 311U, 319U, 327U, 335U, 343U,   4U, 344U, 352U, 360U,
        368U, 376U, 384U, 392U, 345U, 353U, 361U, 369U, 377U, 385U, 393U, 346U, 354U, 362U, 370U, 378U, 386U, 394U, 347U, 355U, 363U, 371U, 379U, 387U, 395U, 348U, 356U, 364U, 372U, 380U, 388U, 396U, 349U, 357U, 365U, 373U, 381U, 389U, 397U, 350U, 358U, 366U, 374U, 382U, 390U, 398U, 351U, 359U, 367U, 375U, 383U, 391U, 399U,   5U, 400U, 408U,
        416U, 424U, 432U, 440U, 448U, 401U, 409U, 417U, 425U, 433U, 441U, 449U, 402U, 410U, 418U, 426U, 434U, 442U, 450U, 403U, 411U, 419U, 427U, 435U, 443U, 451U, 404U, 412U, 420U, 428U, 436U, 444U, 452U, 405U, 413U, 421U, 429U, 437U, 445U, 453U, 406U, 414U, 422U, 430U, 438U, 446U, 454U, 407U, 415U, 423U, 431U, 439U, 447U, 455U, 456U, 464U,
        472U, 480U, 488U, 496U, 504U, 457U, 465U, 473U, 481U, 489U, 497U, 505U, 458U, 466U, 474U, 482U, 490U, 498U, 506U, 459U, 467U, 475U, 483U, 491U, 499U, 507U, 460U, 468U, 476U, 484U, 492U, 500U, 508U, 461U, 469U, 477U, 485U, 493U, 501U, 509U, 462U, 470U, 478U, 486U, 494U, 502U, 510U, 463U, 471U, 479U, 487U, 495U, 503U, 511U,   6U, 512U,
        520U, 528U, 536U, 544U, 552U, 560U, 513U, 521U, 529U, 537U, 545U, 553U, 561U, 514U, 522U, 530U, 538U, 546U, 554U, 562U, 515U, 523U, 531U, 539U, 547U, 555U, 563U, 516U, 524U, 532U, 540U, 548U, 556U, 564U, 517U, 525U, 533U, 541U, 549U, 557U, 565U, 518U, 526U, 534U, 542U, 550U, 558U, 566U, 519U, 527U, 535U, 543U, 551U, 559U, 567U,   7U,
        568U, 576U, 584U, 592U, 600U, 608U, 616U, 569U, 577U, 585U, 593U, 601U, 609U, 617U, 570U, 578U, 586U, 594U, 602U, 610U, 618U, 571U, 579U, 587U, 595U, 603U, 611U, 619U, 572U, 580U, 588U, 596U, 604U, 612U, 620U, 573U, 581U, 589U, 597U, 605U, 613U, 621U, 574U, 582U, 590U, 598U, 606U, 614U, 622U, 575U, 583U, 591U, 599U, 607U, 615U, 623U,
        // clang-format on
    };

    unsigned int* d_mt19937_jump{};
    HIP_CHECK(hipMalloc(&d_mt19937_jump, sizeof(jump)));
    HIP_CHECK(hipMemcpy(d_mt19937_jump, jump, sizeof(jump), hipMemcpyHostToDevice));

    unsigned int* d_engines{};
    HIP_CHECK(hipMalloc(&d_engines, generator_count * state_size * sizeof(unsigned int)));

    // dummy config provider, kernel just needs to verify the amount of generators for the actual call
    using ConfigProvider = default_config_provider<ROCRAND_RNG_PSEUDO_MT19937>;

    rocrand_status status = rocrand_impl::system::device_system::template launch<
        rocrand_impl::host::
            jump_ahead_mt19937<generator_t::jump_ahead_thread_count, ConfigProvider, false>,
        rocrand_impl::host::static_block_size_config_provider<
            generator_t::jump_ahead_thread_count>>(dim3(generator_count),
                                                   dim3(generator_t::jump_ahead_thread_count),
                                                   0,
                                                   0,
                                                   d_engines,
                                                   seed,
                                                   d_mt19937_jump);
    ASSERT_EQ(status, ROCRAND_STATUS_SUCCESS);

    octo_engine_type* d_octo_engines{};
    HIP_CHECK(hipMalloc(&d_octo_engines,
                        threads_per_generator * generator_count * sizeof(octo_engine_type)));

    // Initialize the octo engines from the two engines.
    // Generate subsequence_size elements from the first engine.
    hipLaunchKernelGGL(init_engines_kernel,
                       dim3(1),
                       dim3(threads_per_generator * 2),
                       0,
                       0,
                       d_octo_engines,
                       d_engines,
                       subsequence_size);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_mt19937_jump));
    HIP_CHECK(hipFree(d_engines));

    // Device side data
    constexpr unsigned int elements_per_generator = (12345789U / state_size) * state_size;
    constexpr size_t       bytes_per_generator    = elements_per_generator * sizeof(unsigned int);
    unsigned int*          d_data;
    HIP_CHECK(hipMallocHelper(&d_data, 2 * bytes_per_generator));

    // Generate the data
    hipLaunchKernelGGL(generate_kernel,
                       dim3(1),
                       dim3(threads_per_generator * 2),
                       0,
                       0,
                       d_octo_engines,
                       d_data,
                       elements_per_generator,
                       subsequence_size);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(d_octo_engines));

    unsigned int* h_data = static_cast<unsigned int*>(malloc(2 * bytes_per_generator));
    HIP_CHECK(hipMemcpy(h_data, d_data, 2 * bytes_per_generator, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_data));

    // Output of the two generators
    unsigned int* data0 = h_data;
    unsigned int* data1 = h_data + elements_per_generator;

    // The absolute index for each generator, from which point dataxsize number of full state_size blocks are produced
    unsigned int data0min = ((subsequence_size + state_size - 1) / state_size) * state_size;
    unsigned int data1min = subsequence_size;

    // The number of full state_size blocks
    unsigned int data0size
        = (subsequence_size * 2) / state_size - (subsequence_size + state_size - 1) / state_size;
    unsigned int data1size = subsequence_size / state_size;

    unsigned int checks0to1 = 0;
    unsigned int checks1to0 = 0;
    for(unsigned int i = 0; i < elements_per_generator / state_size; i++)
    {
        for(unsigned int j = 0; j < state_size; j++)
        {
            unsigned int idx = state_size * i + j;
            SCOPED_TRACE(testing::Message() << "idx = " << idx);

            // For idx, get the absolute index in the sequence produced by the generators
            unsigned int idx_rel = subsequence_size + idx;
            unsigned int idx0
                = (idx_rel / state_size) * state_size + permutation_table[idx_rel % state_size];
            unsigned int idx1 = subsequence_size + (idx / state_size) * state_size
                                + permutation_table[idx % state_size];

            // If absolute index idx0 is within the range of full
            // state_size blocks produced by generator 1
            if(data1min <= idx0 && idx0 < data1min + data1size * state_size)
            {
                EXPECT_TRUE(idx0 >= subsequence_size);
                unsigned int tmp = idx0 - subsequence_size;
                unsigned int idx1rel
                    = (tmp / state_size) * state_size + rev_permutation_table[tmp % state_size];
                ASSERT_EQ(data0[idx], data1[idx1rel]);
                checks0to1++;
            }

            // If absolute index idx1 is within the range of full
            // state_size blocks produced by generator 0
            if(data0min <= idx1 && idx1 < data0min + data0size * state_size)
            {
                unsigned int tmp     = ((idx1 / state_size) * state_size) - subsequence_size;
                unsigned int idx0rel = tmp + rev_permutation_table[idx1 % state_size];
                ASSERT_EQ(data0[idx0rel], data1[idx]);
                checks1to0++;
            }
        }
    }

    ASSERT_TRUE(checks0to1 > 0);
    ASSERT_TRUE(checks1to0 > 0);

    free(h_data);
}

struct mt19937_engine
{
    static constexpr inline unsigned int m          = mt19937_constants::m;
    static constexpr inline unsigned int mexp       = mt19937_constants::mexp;
    static constexpr inline unsigned int matrix_a   = mt19937_constants::matrix_a;
    static constexpr inline unsigned int upper_mask = mt19937_constants::upper_mask;
    static constexpr inline unsigned int lower_mask = mt19937_constants::lower_mask;

    // Jumping constants.
    static constexpr inline unsigned int qq = 7;
    static constexpr inline unsigned int ll = 1U << qq;

    static constexpr inline unsigned int n = mt19937_constants::n;

    struct mt19937_state
    {
        unsigned int mt[n];
        // index of the next value to be calculated
        unsigned int ptr;
    };

    mt19937_state m_state;

    mt19937_engine(unsigned long long seed)
    {
        const unsigned int seedu = (seed >> 32) ^ seed;
        m_state.mt[0]            = seedu;
        for(unsigned int i = 1; i < n; i++)
        {
            m_state.mt[i] = 1812433253 * (m_state.mt[i - 1] ^ (m_state.mt[i - 1] >> 30)) + i;
        }
        m_state.ptr = 0;
    }

    /// Advances the internal state to skip a single subsequence, which is <tt>2 ^ 1000</tt> states long.
    void discard_subsequence()
    {
        // First n values of rocrand_h_mt19937_jump contain polynomial for a jump by 2 ^ 1000
        m_state = discard_subsequence_impl(rocrand_h_mt19937_jump, m_state);
    }

    // Generates the next state.
    static void gen_next(mt19937_state& state)
    {
        /// mag01[x] = x * matrix_a for x in [0, 1]
        constexpr unsigned int mag01[2] = {0x0U, matrix_a};

        if(state.ptr + m < n)
        {
            unsigned int y
                = (state.mt[state.ptr] & upper_mask) | (state.mt[state.ptr + 1] & lower_mask);
            state.mt[state.ptr] = state.mt[state.ptr + m] ^ (y >> 1) ^ mag01[y & 0x1U];
            state.ptr++;
        }
        else if(state.ptr < n - 1)
        {
            unsigned int y
                = (state.mt[state.ptr] & upper_mask) | (state.mt[state.ptr + 1] & lower_mask);
            state.mt[state.ptr] = state.mt[state.ptr - (n - m)] ^ (y >> 1) ^ mag01[y & 0x1U];
            state.ptr++;
        }
        else // state.ptr == n - 1
        {
            unsigned int y  = (state.mt[n - 1] & upper_mask) | (state.mt[0] & lower_mask);
            state.mt[n - 1] = state.mt[m - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            state.ptr       = 0;
        }
    }

    /// Return coefficient \p deg of the polynomial <tt>pf</tt>.
    static unsigned int get_coef(const unsigned int pf[mt19937_p_size], unsigned int deg)
    {
        constexpr unsigned int log_w_size  = 5;
        constexpr unsigned int w_size_mask = (1U << log_w_size) - 1;
        return (pf[deg >> log_w_size] & (1U << (deg & w_size_mask))) != 0;
    }

    /// Copy state \p ss into state <tt>ts</tt>.
    static void copy_state(mt19937_state& ts, const mt19937_state& ss)
    {
        for(unsigned int i = 0; i < n; i++)
        {
            ts.mt[i] = ss.mt[i];
        }

        ts.ptr = ss.ptr;
    }

    /// Add state \p s2 to state <tt>s1</tt>.
    static void add_state(mt19937_state& s1, const mt19937_state& s2)
    {
        if(s2.ptr >= s1.ptr)
        {
            unsigned int i = 0;
            for(; i < n - s2.ptr; i++)
            {
                s1.mt[i + s1.ptr] ^= s2.mt[i + s2.ptr];
            }
            for(; i < n - s1.ptr; i++)
            {
                s1.mt[i + s1.ptr] ^= s2.mt[i - (n - s2.ptr)];
            }
            for(; i < n; i++)
            {
                s1.mt[i - (n - s1.ptr)] ^= s2.mt[i - (n - s2.ptr)];
            }
        }
        else
        {
            unsigned int i = 0;
            for(; i < n - s1.ptr; i++)
            {
                s1.mt[i + s1.ptr] ^= s2.mt[i + s2.ptr];
            }
            for(; i < n - s2.ptr; i++)
            {
                s1.mt[i - (n - s1.ptr)] ^= s2.mt[i + s2.ptr];
            }
            for(; i < n; i++)
            {
                s1.mt[i - (n - s1.ptr)] ^= s2.mt[i - (n - s2.ptr)];
            }
        }
    }

    /// Generate Gray code.
    static void gray_code(unsigned int h[ll])
    {
        h[0U] = 0U;

        unsigned int l    = 1;
        unsigned int term = ll;
        unsigned int j    = 1;
        for(unsigned int i = 1; i <= qq; i++)
        {
            l    = (l << 1);
            term = (term >> 1);
            for(; j < l; j++)
            {
                h[j] = h[l - j - 1] ^ term;
            }
        }
    }

    /// Compute \p h(f)ss where \p h(t) are exact <tt>q</tt>-degree polynomials,
    /// \p f is the transition function, and \p ss is the initial state
    /// the results are stored in <tt>vec_h[0] , ... , vec_h[ll - 1]</tt>.
    static void gen_vec_h(const mt19937_state& ss, mt19937_state vec_h[ll])
    {
        mt19937_state v{};
        unsigned int  h[ll];

        gray_code(h);

        copy_state(vec_h[0], ss);

        for(unsigned int i = 0; i < qq; i++)
        {
            gen_next(vec_h[0]);
        }

        for(unsigned int i = 1; i < ll; i++)
        {
            copy_state(v, ss);
            unsigned int g = h[i] ^ h[i - 1];
            for(unsigned int k = 1; k < g; k = (k << 1))
            {
                gen_next(v);
            }
            copy_state(vec_h[h[i]], vec_h[h[i - 1]]);
            add_state(vec_h[h[i]], v);
        }
    }

    /// Compute pf(ss) using Sliding window algorithm.
    static mt19937_state calc_state(const unsigned int   pf[mt19937_p_size],
                                    const mt19937_state& ss,
                                    const mt19937_state  vec_h[ll])
    {
        mt19937_state tmp{};
        int           i = mexp - 1;

        while(get_coef(pf, i) == 0)
        {
            i--;
        }

        for(; i >= static_cast<int>(qq); i--)
        {
            if(get_coef(pf, i) != 0)
            {
                for(unsigned int j = 0; j < qq + 1; j++)
                {
                    gen_next(tmp);
                }
                unsigned int digit = 0;
                for(int j = 0; j < static_cast<int>(qq); j++)
                {
                    digit = (digit << 1) ^ get_coef(pf, i - j - 1);
                }
                add_state(tmp, vec_h[digit]);
                i -= qq;
            }
            else
            {
                gen_next(tmp);
            }
        }

        for(; i > -1; i--)
        {
            gen_next(tmp);
            if(get_coef(pf, i) == 1)
            {
                add_state(tmp, ss);
            }
        }

        return tmp;
    }

    /// Computes jumping ahead with Sliding window algorithm.
    static mt19937_state discard_subsequence_impl(const unsigned int   pf[mt19937_p_size],
                                                  const mt19937_state& ss)
    {
        // skip state
        mt19937_state vec_h[ll];
        gen_vec_h(ss, vec_h);
        mt19937_state new_state = calc_state(pf, ss, vec_h);

        // rotate the array to align ptr with the array boundary
        if(new_state.ptr != 0)
        {
            unsigned int tmp[n];
            for(unsigned int i = 0; i < n; i++)
            {
                tmp[i] = new_state.mt[(i + new_state.ptr) % n];
            }

            for(unsigned int i = 0; i < n; i++)
            {
                new_state.mt[i] = tmp[i];
            }
        }

        // set to 0, which is the index of the next number to be calculated
        new_state.ptr = 0;

        return new_state;
    }
};

TYPED_TEST(mt19937_generator_engine_tests, jump_ahead_test)
{
    // Compare states of all engines
    // * computed consecutively on host using Sliding window algorithm
    //   (each engine jumps 2 ^ 1000 ahead from the previous one);
    // * computed in parallel on device using standard Horner algorithm
    //   with precomputed jumps of i * 2 ^ 1000 and mt19937_jumps_radix * i * 2 ^ 1000 values
    //   where i is in range [1; mt19937_jumps_radix).

    using generator_t = typename TestFixture::generator_t;

    const unsigned long long seed = 12345678;
    constexpr unsigned int   n    = mt19937_constants::n;

    // Test for default config
    using ConfigProvider = default_config_provider<ROCRAND_RNG_PSEUDO_MT19937>;
    generator_config config;
    HIP_CHECK(
        ConfigProvider::host_config<unsigned int>(0, ROCRAND_ORDERING_PSEUDO_DEFAULT, config));

    const unsigned int generator_count
        = config.threads * config.blocks / mt19937_octo_engine::threads_per_generator;

    // Initialize the engines on host using Sliding window algorithm
    std::vector<mt19937_engine> h_engines0;
    h_engines0.reserve(generator_count);
    // initialize the first engine with the seed and no skips
    h_engines0.emplace_back(seed);
    for(size_t i = 1; i < generator_count; i++)
    {
        // every consecutive engine is one subsequence away from the previous
        h_engines0.push_back(h_engines0.back());
        h_engines0[i].discard_subsequence();
    }

    // Initialize the engines on device using Horner algorithm

    unsigned int* d_mt19937_jump{};
    HIP_CHECK(hipMalloc(&d_mt19937_jump, sizeof(rocrand_h_mt19937_jump)));
    HIP_CHECK(hipMemcpy(d_mt19937_jump,
                        rocrand_h_mt19937_jump,
                        sizeof(rocrand_h_mt19937_jump),
                        hipMemcpyHostToDevice));

    unsigned int* d_engines1{};
    HIP_CHECK(hipMalloc(&d_engines1, generator_count * n * sizeof(unsigned int)));

    rocrand_impl::host::dynamic_dispatch(
        ROCRAND_ORDERING_PSEUDO_DEFAULT,
        [&](auto is_dynamic)
        {
            rocrand_status status = rocrand_impl::system::device_system::template launch<
                rocrand_impl::host::jump_ahead_mt19937<generator_t::jump_ahead_thread_count,
                                                       ConfigProvider,
                                                       is_dynamic>,
                rocrand_impl::host::static_block_size_config_provider<
                    generator_t::jump_ahead_thread_count>>(
                dim3(generator_count),
                dim3(generator_t::jump_ahead_thread_count),
                0,
                0,
                d_engines1,
                seed,
                d_mt19937_jump);
            ASSERT_EQ(status, ROCRAND_STATUS_SUCCESS);
        });

    std::vector<unsigned int> h_engines1(generator_count * n);
    HIP_CHECK(hipMemcpy(h_engines1.data(),
                        d_engines1,
                        generator_count * n * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));

    for(unsigned int gi = 0; gi < generator_count; gi++)
    {
        for(unsigned int i = 0; i < n; i++)
        {
            unsigned int a = h_engines0[gi].m_state.mt[i];
            unsigned int b = h_engines1[gi * n + i];
            if(i == 0)
            {
                // 31 bits of the first value contain garbage, only the last bit (19937 % 32 == 1)
                // matters
                a &= 0x80000000U;
                b &= 0x80000000U;
            }
            ASSERT_EQ(a, b);
        }
    }

    HIP_CHECK(hipFree(d_mt19937_jump));
    HIP_CHECK(hipFree(d_engines1));
}

/*
    HOST SIDE TESTS
*/

TYPED_TEST(mt19937_generator_engine_tests, jump_ahead_host_test)
{
    // Compare states of all engines
    // computed consecutively on host using Sliding window and Horner algorithm

    using generator_t = typename TestFixture::generator_t;

    const unsigned long long seed = 12345678;
    constexpr unsigned int   n    = mt19937_constants::n;

    // Test for default config
    using ConfigProvider = default_config_provider<ROCRAND_RNG_PSEUDO_MT19937>;
    generator_config config;
    HIP_CHECK(
        ConfigProvider::host_config<unsigned int>(0, ROCRAND_ORDERING_PSEUDO_DEFAULT, config));

    const unsigned int generator_count
        = config.threads * config.blocks / mt19937_octo_engine::threads_per_generator;

    // Initialize the engines on host using Sliding window algorithm
    std::vector<mt19937_engine> h_engines0;
    h_engines0.reserve(generator_count);
    // initialize the first engine with the seed and no skips
    h_engines0.emplace_back(seed);
    for(size_t i = 1; i < generator_count; i++)
    {
        // every consecutive engine is one subsequence away from the previous
        h_engines0.push_back(h_engines0.back());
        h_engines0[i].discard_subsequence();
    }

    std::vector<unsigned int> h_engines1(generator_count * n);

    for(unsigned int engine_id = 0; engine_id < generator_count; ++engine_id)
    {
        jump_ahead_mt19937<generator_t::jump_ahead_thread_count, ConfigProvider, false>(
            dim3(engine_id),
            dim3(0),
            dim3(generator_count),
            dim3(generator_t::jump_ahead_thread_count),
            h_engines1.data(),
            seed,
            rocrand_h_mt19937_jump);
    }
    for(unsigned int gi = 0; gi < generator_count; gi++)
    {
        for(unsigned int i = 0; i < n; i++)
        {
            unsigned int a = h_engines0[gi].m_state.mt[i];
            unsigned int b = h_engines1[gi * n + i];
            if(i == 0)
            {
                // 31 bits of the first value contain garbage, only the last bit (19937 % 32 == 1)
                // matters
                a &= 0x80000000U;
                b &= 0x80000000U;
            }
            ASSERT_EQ(a, b);
        }
    }
}
