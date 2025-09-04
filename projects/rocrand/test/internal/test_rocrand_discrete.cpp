// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <stdio.h>

#include <numeric>
#include <random>

#include <rocrand/rocrand_discrete.h>
#include <rocrand/rocrand_mtgp32_11213.h>
#include <vector>

#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                                         \
    do                                                                                         \
    {                                                                                          \
        auto error = (cmd);                                                                    \
        if(error != hipSuccess)                                                                \
        {                                                                                      \
            std::cerr << "Encountered HIP error (" << hipGetErrorString(error) << ") at line " \
                      << __LINE__ << " in file " << __FILE__ << "\n";                          \
            exit(-1);                                                                          \
        }                                                                                      \
    }                                                                                          \
    while(0)

#define ROCRAND_CHECK(cmd)                                                                \
    do                                                                                    \
    {                                                                                     \
        auto status = cmd;                                                                \
        if(status != 0)                                                                   \
        {                                                                                 \
            std::cerr << "Encountered ROCRAND error: " << status << "at line" << __LINE__ \
                      << " in file " << __FILE__ << "\n";                                 \
            exit(-1);                                                                     \
        }                                                                                 \
    }                                                                                     \
    while(0)

struct GlobalSizes
{
    static constexpr size_t items_per_thread = 256;
    static constexpr size_t block_size       = 32;
    static constexpr size_t items_per_block  = items_per_thread * block_size;
    static constexpr size_t grid_size        = 1234;
    static constexpr size_t size             = grid_size * items_per_block;
};

using DiscreteDataType = ::testing::Types<double, unsigned int, unsigned long, unsigned long long int>;

template <typename DT>
class InternalDiscreteDistributionTests : public ::testing::Test{
    public:
        using T = DT;
};

TYPED_TEST_SUITE(InternalDiscreteDistributionTests, DiscreteDataType);

template<typename T, class DiscreteWrapper>
__global__
void internal_discrete_kernel(T*                                device_input,
                              unsigned int*                     device_output,
                              rocrand_discrete_distribution_st& dis)
{
    const size_t items_per_block = GlobalSizes::items_per_thread * GlobalSizes::block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (GlobalSizes::items_per_thread * threadIdx.x);

    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++){
        device_output[offset + i] = DiscreteWrapper{}(device_input[offset + i], dis);
    }
}

template<typename T, class DiscreteWrapper>
void run_internal_discrete_tests()
{
    std::vector<std::vector<double>> all_distributions = {
        {10, 10, 10, 10},
        {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
        {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
        {1, 2, 8, 4, 3, 2, 1}
    };

    std::random_device rd;
    std::mt19937 gen(rd());

    T * host_input = new T[GlobalSizes::size];
    unsigned int * host_output = new unsigned int[GlobalSizes::size];

    // Check for different types of data input and generate the input data
    if constexpr (std::is_same_v<T, double>){
        std::uniform_real_distribution<double> dis(0, 1);
        for(size_t i = 0; i < GlobalSizes::size; i++) host_input[i] = dis(gen);
    }
    else if constexpr(std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned int>){
        std::uniform_int_distribution<T> dis(0, std::numeric_limits<unsigned int>::max());
        for(size_t i = 0; i < GlobalSizes::size; i++) host_input[i] = dis(gen);
    }
    else{
        std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
        for(size_t i = 0; i < GlobalSizes::size; i++) host_input[i] = dis(gen);
    }

    T * device_input;
    unsigned int * device_output;

    HIP_CHECK(hipMalloc(&device_input, sizeof(T) * GlobalSizes::size));
    HIP_CHECK(hipMalloc(&device_output, sizeof(unsigned int) * GlobalSizes::size));

    HIP_CHECK(hipMemcpy(device_input, host_input, sizeof(T) * GlobalSizes::size, hipMemcpyHostToDevice));

    // Generate different discrete distributions and check them against expected
    for(std::vector<double> distribution : all_distributions){
        // Getting expected Results
        double sum = std::accumulate(distribution.begin(), distribution.end(), 0);
        std::vector<double> expected_prob(distribution.size());
        for(size_t i = 0; i < distribution.size(); i++)
            expected_prob[i] = distribution[i] / sum;

        // Creating the discrete distribution
        rocrand_discrete_distribution discrete_distribution;
        ROCRAND_CHECK(rocrand_create_discrete_distribution(expected_prob.data(), expected_prob.size(), 0, &discrete_distribution));

        internal_discrete_kernel<T, DiscreteWrapper>
            <<<dim3(GlobalSizes::grid_size), dim3(GlobalSizes::block_size), 0, 0>>>(
                device_input,
                device_output,
                *discrete_distribution);

        HIP_CHECK(hipMemcpy(host_output, device_output, sizeof(unsigned int) * GlobalSizes::size, hipMemcpyDeviceToHost));

        std::vector<double> histogram(distribution.size());

        // Calculating the actual results
        for(size_t i = 0; i < GlobalSizes::size; i++)
            histogram[host_output[i]]++;

        std::vector<double> actual_prob(distribution.size());
        for(size_t i = 0; i < actual_prob.size(); i++)
            actual_prob[i] = histogram[i] / static_cast<double>(GlobalSizes::size);

        // If the original probability is bigger than 5% then expected should be within 1% difference.
        // Otherwise it should be within 0.01
        for(size_t i = 0; i < expected_prob.size(); i++){
            double eps = expected_prob[i] > 0.05 ? expected_prob[i] * 0.01 : 0.01;
            ASSERT_NEAR(expected_prob[i], actual_prob[i], eps);
        }

        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    delete [] host_input;
    delete [] host_output;

    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}

template<typename T>
struct internal_discrete_alias
{
    __device__
    auto operator()(T val, rocrand_discrete_distribution_st& dis)
    {
        return rocrand_device::detail::discrete_alias(val, dis);
    }
};

template<typename T>
struct internal_discrete_cdf
{
    __device__
    auto operator()(T val, rocrand_discrete_distribution_st& dis)
    {
        return rocrand_device::detail::discrete_cdf(val, dis);
    }
};

TYPED_TEST(InternalDiscreteDistributionTests, InternalDiscreteAliasTest){
    using T = typename TestFixture::T;

    run_internal_discrete_tests<T, internal_discrete_alias<T>>();
}

TYPED_TEST(InternalDiscreteDistributionTests, InternalDiscreteCDFTest){
    using T = typename TestFixture::T;

    run_internal_discrete_tests<T, internal_discrete_cdf<T>>();
}

template<class RocRandPrngType>
__global__ void block_wide_external_discrete_kernel(
    RocRandPrngType * states,
    unsigned int * device_output,
    rocrand_discrete_distribution_st & dis,
    size_t items_per_thread,
    size_t block_size
){
    const size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    for(size_t i = 0; i < items_per_thread; i++){
        __shared__ RocRandPrngType state;
        if(threadIdx.x == 0)
            state = states[blockIdx.x];
        __syncthreads();

        device_output[offset + i] = rocrand_discrete(&state, &dis);

        if(threadIdx.x == 0)
            states[blockIdx.x] = state;
        __syncthreads();
    }
}

template<class RocRandPrngType>
__global__ void external_discrete_kernel(
    RocRandPrngType * states,
    unsigned int * device_output,
    rocrand_discrete_distribution_st & dis,
    size_t items_per_thread,
    size_t block_size
){
    const size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    for(size_t i = 0; i < items_per_thread; i++){
        auto local_state = states[offset + i];
        device_output[offset + i] = rocrand_discrete(&local_state, &dis);
        states[offset + i] = local_state;
    }
}

template<bool block_wide, class PrngState>
void run_external_discrete_tests(
    PrngState & device_states,
    size_t items_per_thread = GlobalSizes::items_per_thread,
    size_t block_size = GlobalSizes::block_size,
    size_t grid_size = GlobalSizes::grid_size,
    size_t size = GlobalSizes::size
){

    std::vector<std::vector<double>> all_distributions = {
        {10, 10, 10, 10},
        {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
        {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
        {1, 2, 8, 4, 3, 2, 1}
    };

    unsigned int * host_output = new unsigned int[size];
    unsigned int * device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(unsigned int) * size));

    for(std::vector<double> distribution : all_distributions){

        // Getting expected Results
        double sum = std::accumulate(distribution.begin(), distribution.end(), 0);
        std::vector<double> expected_prob(distribution.size());
        for(size_t i = 0; i < distribution.size(); i++)
            expected_prob[i] = distribution[i] / sum;

        // Creating the discrete distribution
        rocrand_discrete_distribution discrete_distribution;
        ROCRAND_CHECK(rocrand_create_discrete_distribution(expected_prob.data(), expected_prob.size(), 0, &discrete_distribution));

        if constexpr(block_wide){
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(block_wide_external_discrete_kernel),
                dim3(grid_size), dim3(block_size), 0, 0,
                device_states, device_output, *discrete_distribution, items_per_thread, block_size
            );
        }
        else{
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(external_discrete_kernel),
                dim3(grid_size), dim3(block_size), 0, 0,
                device_states, device_output, *discrete_distribution, items_per_thread, block_size
            );
        }

        HIP_CHECK(hipMemcpy(host_output, device_output, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        std::vector<double> histogram(distribution.size());

        // Calculating the actual results
        for(size_t i = 0; i < size; i++)
            histogram[host_output[i]]++;

        std::vector<double> actual_prob(distribution.size());
        for(size_t i = 0; i < actual_prob.size(); i++)
            actual_prob[i] = histogram[i] / static_cast<double>(size);

        // If the original probability is bigger than 5% then expected should be within 1% difference.
        // Otherwise it should be within 0.01
        for(size_t i = 0; i < expected_prob.size(); i++){
            double eps = expected_prob[i] > 0.05 ? expected_prob[i] * 0.01 : 0.01;
            ASSERT_NEAR(expected_prob[i], actual_prob[i], eps);
        }

        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    delete [] host_output;
    HIP_CHECK(hipFree(device_output));
}

template<class RocRandPrngType>
__global__
void init_rocrand_states_kernel(RocRandPrngType* states)
{
    constexpr size_t items_per_block = GlobalSizes::items_per_thread * GlobalSizes::block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (GlobalSizes::items_per_thread * threadIdx.x);

    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++)
    {
        rocrand_init(static_cast<unsigned int>(123456 ^ i), offset + i, 0, &states[offset + i]);
    }
}

template<class RocRandPrngType>
__global__
void init_rocrand_states_kernel4(RocRandPrngType* states)
{
    constexpr size_t items_per_block = GlobalSizes::items_per_thread * GlobalSizes::block_size;
    const size_t     offset
        = (items_per_block * blockIdx.x) + (GlobalSizes::items_per_thread * threadIdx.x);

    for(size_t i = 0; i < GlobalSizes::items_per_thread; i++)
    {
        rocrand_init(uint4{static_cast<unsigned int>((123456 ^ i)),
                           static_cast<unsigned int>((123456 ^ i) << 1),
                           static_cast<unsigned int>((123456 ^ i) << 2),
                           static_cast<unsigned int>((123456 ^ i) << 3)},
                     offset + i,
                     0,
                     &states[offset + i]);
    }
}

TEST(ExternalDiscreteDistributionTests, Philox4x32_10Test){
    // Initialize the prng state
    rocrand_state_philox4x32_10 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_philox4x32_10) * GlobalSizes::size));

    struct f
    {
        __device__
        static void init(size_t index, size_t offset, rocrand_state_philox4x32_10* state)
        {
            rocrand_init((123456 ^ index), offset + index, 0, state);
        }
    };

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Mrg31k3pTest){
    // Initialize the prng state
    rocrand_state_mrg31k3p * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_mrg31k3p) * GlobalSizes::size));

    struct f
    {
        __device__
        static void init(size_t index, size_t offset, rocrand_state_mrg31k3p* state)
        {
            rocrand_init((123456 ^ index), offset + index, 0, state);
        }
    };

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Mrg32k3aTest){
    // Initialize the prng state
    rocrand_state_mrg32k3a * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_mrg32k3a) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, XorwowTest){
    // Initialize the prng state
    rocrand_state_xorwow * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_xorwow) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Mtgp32Test){
    constexpr size_t items_per_thread = 1024;
    constexpr size_t block_size = 256;
    constexpr size_t grid_size = 12;
    constexpr size_t items_per_block = block_size * items_per_thread;

    constexpr size_t test_size = items_per_block * grid_size;
    rocrand_state_mtgp32 * states;

    HIP_CHECK(hipMalloc(&states, sizeof(rocrand_state_mtgp32) * grid_size));
    rocrand_make_state_mtgp32(states,  mtgp32dc_params_fast_11213, grid_size, 123456);
    HIP_CHECK(hipDeviceSynchronize());

    run_external_discrete_tests<true>(
        states,
        items_per_thread,
        block_size,
        grid_size,
        test_size
    );
    HIP_CHECK(hipFree(states));
}

TEST(ExternalDiscreteDistributionTests, Lfsr113Test){
    // Initialize the prng state
    rocrand_state_lfsr113 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_lfsr113) * GlobalSizes::size));

    init_rocrand_states_kernel4<<<dim3(GlobalSizes::grid_size),
                                  dim3(GlobalSizes::block_size),
                                  0,
                                  0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Sobol32Test){
    // Initialize the prng state
    rocrand_state_sobol32 * host_states = new rocrand_state_sobol32[GlobalSizes::size];
    const unsigned int* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

    // 640000 is the size of directions. This is to prevent overflow stuff
    for(size_t i = 0; i < GlobalSizes::size; i++)
        rocrand_init(directions, i % 640000, host_states + i);

    rocrand_state_sobol32 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_sobol32) * GlobalSizes::size));
    HIP_CHECK(hipMemcpy(device_states, host_states, sizeof(rocrand_state_sobol32) * GlobalSizes::size, hipMemcpyHostToDevice));

    run_external_discrete_tests<false>(device_states);

    delete [] host_states;
    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, ScrambledSobol32Test){
    // Initialize the prng state
    rocrand_state_scrambled_sobol32 * host_states = new rocrand_state_scrambled_sobol32[GlobalSizes::size];
    const unsigned int* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));

    // 640000 is the size of directions. This is to prevent overflow stuff
    for(size_t i = 0; i < GlobalSizes::size; i++)
        rocrand_init(directions, 123456 ^ i, i % 640000, host_states + i);

    rocrand_state_scrambled_sobol32 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_scrambled_sobol32) * GlobalSizes::size));
    HIP_CHECK(hipMemcpy(device_states, host_states, sizeof(rocrand_state_scrambled_sobol32) * GlobalSizes::size, hipMemcpyHostToDevice));

    run_external_discrete_tests<false>(device_states);

    delete [] host_states;
    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Sobol64Test){
    // Initialize the prng state
    rocrand_state_sobol64 * host_states = new rocrand_state_sobol64[GlobalSizes::size];
    const unsigned long long* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));

    // 1280000 is the size of directions. This is to prevent overflow stuff
    for(size_t i = 0; i < GlobalSizes::size; i++)
        rocrand_init(directions, i % 1280000, host_states + i);

    rocrand_state_sobol64 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_sobol64) * GlobalSizes::size));
    HIP_CHECK(hipMemcpy(device_states, host_states, sizeof(rocrand_state_sobol64) * GlobalSizes::size, hipMemcpyHostToDevice));

    run_external_discrete_tests<false>(device_states);

    delete [] host_states;
    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, ScrambledSobol64Test){
    // Initialize the prng state
    rocrand_state_scrambled_sobol64 * host_states = new rocrand_state_scrambled_sobol64[GlobalSizes::size];
    const unsigned long long* directions;
    ROCRAND_CHECK(rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));

    // 1280000 is the size of directions. This is to prevent overflow stuff
    for(size_t i = 0; i < GlobalSizes::size; i++)
        rocrand_init(directions, 123456 ^ i, i % 1280000, host_states + i);

    rocrand_state_scrambled_sobol64 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_scrambled_sobol64) * GlobalSizes::size));
    HIP_CHECK(hipMemcpy(device_states, host_states, sizeof(rocrand_state_scrambled_sobol64) * GlobalSizes::size, hipMemcpyHostToDevice));

    run_external_discrete_tests<false>(device_states);

    delete [] host_states;
    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Threefry2x32_20Test){
    // Initialize the prng state
    rocrand_state_threefry2x32_20 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_threefry2x32_20) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Threefry2x64_20Test){
    // Initialize the prng state
    rocrand_state_threefry2x64_20 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_threefry2x64_20) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Threefry4x32_20Test){
    // Initialize the prng state
    rocrand_state_threefry4x32_20 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_threefry4x32_20) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

TEST(ExternalDiscreteDistributionTests, Threefry4x64_20Test){
    // Initialize the prng state
    rocrand_state_threefry4x64_20 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_threefry4x64_20) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    run_external_discrete_tests<false>(device_states);

    HIP_CHECK(hipFree(device_states));
}

template<size_t items_per_thread, size_t block_size>
__global__ void uint4_kernel(rocrand_state_philox4x32_10 * states, uint4 * device_output, rocrand_discrete_distribution_st & dis){
    const size_t items_per_block = items_per_thread * block_size;
    const size_t offset = (items_per_block * blockIdx.x) + (items_per_thread * threadIdx.x);

    for(size_t i = 0; i < items_per_thread; i++){
        auto local_state = states[offset + i];
        device_output[offset + i] = rocrand_discrete4(&local_state, &dis);
        states[offset + i] = local_state;
    }
}

TEST(ExternalDiscreteDistributionTests, Philox4x32_10WithUIN4OutputTest)
{
    // Initialize the prng state
    rocrand_state_philox4x32_10 * device_states;
    HIP_CHECK(hipMalloc(&device_states, sizeof(rocrand_state_philox4x32_10) * GlobalSizes::size));

    init_rocrand_states_kernel<<<dim3(GlobalSizes::grid_size),
                                 dim3(GlobalSizes::block_size),
                                 0,
                                 0>>>(device_states);

    std::vector<std::vector<double>> all_distributions = {
        {10, 10, 10, 10},
        {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
        {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
        {1, 2, 8, 4, 3, 2, 1}
    };

    uint4 * host_output = new uint4[GlobalSizes::size];
    uint4 * device_output;
    HIP_CHECK(hipMalloc(&device_output, sizeof(uint4) * GlobalSizes::size));

    for(std::vector<double> distribution : all_distributions){

        // Getting expected Results
        double sum = std::accumulate(distribution.begin(), distribution.end(), 0);
        std::vector<double> expected_prob(distribution.size());
        for(size_t i = 0; i < distribution.size(); i++)
            expected_prob[i] = distribution[i] / sum;

        // Creating the discrete distribution
        rocrand_discrete_distribution discrete_distribution;
        ROCRAND_CHECK(rocrand_create_discrete_distribution(expected_prob.data(), expected_prob.size(), 0, &discrete_distribution));

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(uint4_kernel<GlobalSizes::items_per_thread, GlobalSizes::block_size>),
            dim3(GlobalSizes::grid_size), dim3(GlobalSizes::block_size), 0, 0,
            device_states, device_output, * discrete_distribution
        );

        HIP_CHECK(hipMemcpy(host_output, device_output, sizeof(uint4) * GlobalSizes::size, hipMemcpyDeviceToHost));
        std::vector<double> histogram(distribution.size());

        // Calculating the actual results
        for(size_t i = 0; i < GlobalSizes::size; i++){
            histogram[host_output[i].w]++;
            histogram[host_output[i].x]++;
            histogram[host_output[i].y]++;
            histogram[host_output[i].z]++;
        }

        std::vector<double> actual_prob(distribution.size());
        for(size_t i = 0; i < actual_prob.size(); i++)
            actual_prob[i] = histogram[i] / static_cast<double>(GlobalSizes::size * 4);

        // If the original probability is bigger than 5% then expected should be within 1% difference.
        // Otherwise it should be within 0.01
        for(size_t i = 0; i < expected_prob.size(); i++){
            double eps = expected_prob[i] > 0.05 ? expected_prob[i] * 0.01 : 0.01;
            ASSERT_NEAR(expected_prob[i], actual_prob[i], eps);
        }

        ROCRAND_CHECK(rocrand_destroy_discrete_distribution(discrete_distribution));
    }

    delete [] host_output;
    HIP_CHECK(hipFree(device_output));
}

/* #################################################

                TEST HOST SIDE

   ###############################################*/

#include "rng/distribution/discrete.hpp"

template<typename T, class DiscreteFunc>
void run_internal_host_test(const DiscreteFunc& df)
{
    constexpr size_t test_size = 100000;

    std::vector<std::vector<double>> all_distributions = {
        {10, 10, 10, 10},
        {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
        {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
        {1, 2, 8, 4, 3, 2, 1}
    };

    std::random_device rd;
    std::mt19937       gen(rd());

    std::vector<T>            input(test_size);
    std::vector<unsigned int> histogram(test_size, 0);

    // Check for different types of data input and generate the input data
    if constexpr(std::is_same_v<T, double>)
    {
        std::uniform_real_distribution<double> dis(0, 1);
        for(size_t i = 0; i < test_size; i++)
            input[i] = dis(gen);
    }
    else if constexpr(std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned int>)
    {
        std::uniform_int_distribution<T> dis(0, std::numeric_limits<unsigned int>::max());
        for(size_t i = 0; i < test_size; i++)
            input[i] = dis(gen);
    }
    else
    {
        std::uniform_int_distribution<T> dis(0, std::numeric_limits<T>::max());
        for(size_t i = 0; i < test_size; i++)
            input[i] = dis(gen);
    }

    for(std::vector<double> distribution : all_distributions)
    {
        // Getting expected Results
        double              sum = std::accumulate(distribution.begin(), distribution.end(), 0);
        std::vector<double> expected_prob(distribution.size());
        for(size_t i = 0; i < distribution.size(); i++)
            expected_prob[i] = distribution[i] / sum;

        // Creating the discrete distribution
        rocrand_discrete_distribution_st discrete_dis;

        using namespace rocrand_impl::host;

        rocrand_status rocrand_err
            = discrete_distribution_factory<discrete_method::DISCRETE_METHOD_UNIVERSAL,
                                            true>::create(distribution,
                                                          distribution.size(),
                                                          0,
                                                          discrete_dis);

        ROCRAND_CHECK(rocrand_err);
        std::vector<double> histogram(distribution.size());
        for(size_t i = 0; i < test_size; i++)
        {
            histogram[df(input[i], discrete_dis)]++;
        }

        std::vector<double> actual_prob(distribution.size());
        for(size_t i = 0; i < actual_prob.size(); i++)
            actual_prob[i] = histogram[i] / static_cast<double>(test_size);

        // If the original probability is bigger than 5% then expected should be within 1% difference.
        // Otherwise it should be within 0.01
        for(size_t i = 0; i < expected_prob.size(); i++)
        {
            double eps = expected_prob[i] > 0.05 ? expected_prob[i] * 0.05 : 0.01;
            ASSERT_NEAR(expected_prob[i], actual_prob[i], eps);
        }
    }
}

template<typename InType, bool UseDiscreteAlias>
struct InternalHostParams
{
    using T                   = InType;
    static constexpr bool uda = UseDiscreteAlias;
};

using InternalDiscreteHostParams
    = ::testing::Types<InternalHostParams<double, true>,
                       InternalHostParams<unsigned int, true>,
                       InternalHostParams<unsigned long, true>,
                       InternalHostParams<unsigned long long int, true>,
                       InternalHostParams<double, false>,
                       InternalHostParams<unsigned int, false>,
                       InternalHostParams<unsigned long, false>,
                       InternalHostParams<unsigned long long int, false>>;

template<class InternalHostParams>
class InternalDiscreteHostTest : public ::testing::Test
{
public:
    using input_type                         = typename InternalHostParams::T;
    static constexpr bool use_discrete_alias = InternalHostParams::uda;
};

TYPED_TEST_SUITE(InternalDiscreteHostTest, InternalDiscreteHostParams);

TYPED_TEST(InternalDiscreteHostTest, discrete_host_internal)
{
    using input_type                         = typename TestFixture::input_type;
    static constexpr bool use_discrete_alias = TestFixture::use_discrete_alias;

    if constexpr(use_discrete_alias)
    {
        run_internal_host_test<input_type>(
            [=](const input_type& x, rocrand_discrete_distribution_st& dis)
            { return rocrand_device::detail::discrete_alias(x, dis); });
    }
    else
    {
        run_internal_host_test<input_type>(
            [=](const input_type& x, rocrand_discrete_distribution_st& dis)
            { return rocrand_device::detail::discrete_cdf(x, dis); });
    }
}

template<class RocrandPRNGType>
inline void GetRocrandState(RocrandPRNGType* host_state)
{

    if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol32 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol32>)
    {
        const unsigned int* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors32(&directions, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, host_state);
    }
    // scrambled sobol64 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_scrambled_sobol64>)
    {
        const unsigned long long* directions;
        ROCRAND_CHECK(
            rocrand_get_direction_vectors64(&directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6));
        rocrand_init(directions, 123456, 654321, host_state);
    }
    // lfsr113 case
    else if constexpr(std::is_same_v<RocrandPRNGType, rocrand_state_lfsr113>)
    {
        rocrand_init({0xabcd, 0xdabc, 0xcdab, 0xbcda}, 0, 0, host_state);
    }
    else
    {
        rocrand_init(123456, 654321, 0, host_state);
    }
}

template<typename RocrandPRNGType, class DiscreteFunc>
void run_host_test(const DiscreteFunc& df)
{
    constexpr size_t test_size = 100000;

    std::vector<std::vector<double>> all_distributions = {
        {10, 10, 10, 10},
        {1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1},
        {1234, 1677, 1519, 1032, 561, 254, 98, 33, 10, 2},
        {1, 2, 8, 4, 3, 2, 1}
    };

    RocrandPRNGType generator;
    GetRocrandState(&generator);

    for(std::vector<double> distribution : all_distributions)
    {
        // Getting expected Results
        double              sum = std::accumulate(distribution.begin(), distribution.end(), 0);
        std::vector<double> expected_prob(distribution.size());
        for(size_t i = 0; i < distribution.size(); i++)
            expected_prob[i] = distribution[i] / sum;

        // Creating the discrete distribution
        rocrand_discrete_distribution_st discrete_dis;

        using namespace rocrand_impl::host;

        rocrand_status rocrand_err
            = discrete_distribution_factory<discrete_method::DISCRETE_METHOD_UNIVERSAL,
                                            true>::create(distribution,
                                                          distribution.size(),
                                                          0,
                                                          discrete_dis);

        ROCRAND_CHECK(rocrand_err);
        std::vector<double> histogram(distribution.size());
        for(size_t i = 0; i < test_size; i++)
        {
            histogram[df(&generator, &discrete_dis)]++;
        }

        std::vector<double> actual_prob(distribution.size());
        for(size_t i = 0; i < actual_prob.size(); i++)
            actual_prob[i] = histogram[i] / static_cast<double>(test_size);

        // If the original probability is bigger than 5% then expected should be within 1% difference.
        // Otherwise it should be within 0.01
        for(size_t i = 0; i < expected_prob.size(); i++)
        {
            double eps = expected_prob[i] > 0.05 ? expected_prob[i] * 0.05 : 0.01;
            ASSERT_NEAR(expected_prob[i], actual_prob[i], eps);
        }
    }
}

using DiscreteHostParams = ::testing::Types<rocrand_state_philox4x32_10,
                                            rocrand_state_mrg31k3p,
                                            rocrand_state_mrg32k3a,
                                            rocrand_state_xorwow,
                                            rocrand_state_sobol32,
                                            rocrand_state_scrambled_sobol32,
                                            rocrand_state_sobol64,
                                            rocrand_state_scrambled_sobol64,
                                            rocrand_state_lfsr113,
                                            rocrand_state_threefry2x32_20,
                                            rocrand_state_threefry2x64_20,
                                            rocrand_state_threefry4x32_20,
                                            rocrand_state_threefry4x64_20>;

template<typename T>
class DiscreteHostTest : public ::testing::Test
{
public:
    using rocrand_prng_type = T;
};

TYPED_TEST_SUITE(DiscreteHostTest, DiscreteHostParams);

TYPED_TEST(DiscreteHostTest, discrete_host)
{
    using rocrand_prng_type = typename TestFixture::rocrand_prng_type;

    run_host_test<rocrand_prng_type>(
        [=](rocrand_prng_type* x, rocrand_discrete_distribution_st* dis)
        { return rocrand_discrete(x, dis); });
}
