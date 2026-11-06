#include <gtest/gtest.h>
#include <stdio.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_utils_hipgraphs.hpp"

// The rng type, and whether the generator is created with rocrand_create_generator_host.
using generate_test_params = std::tuple<rocrand_rng_type, bool>;

class rocrand_hipgraph_generate_tests : public ::testing::TestWithParam<generate_test_params>
{};

std::string generate_test_name(const ::testing::TestParamInfo<generate_test_params>& info)
{
    return rocrand_rng_name(std::get<0>(info.param))
           + (std::get<1>(info.param) ? "_Host" : "_Device");
}

void create_generator(rocrand_generator* generator, const generate_test_params& params)
{
    const auto [rng_type, host_generator] = params;
    if(host_generator)
    {
        ROCRAND_CHECK(rocrand_create_generator_host(generator, rng_type));
    }
    else
    {
        ROCRAND_CHECK(rocrand_create_generator(generator, rng_type));
    }
}

// Host generators write the output on the host, so they need memory that the host can access.
template<typename T>
hipError_t allocate_output(T** data, size_t size, const generate_test_params& params)
{
    return std::get<1>(params) ? hipHostMalloc(data, size * sizeof(T))
                               : hipMallocHelper(data, size * sizeof(T));
}

template<typename T>
hipError_t free_output(T* data, const generate_test_params& params)
{
    return std::get<1>(params) ? hipHostFree(data) : hipFree(data);
}

// Launch every graph twice: a graph must stay valid after its first launch.
void create_and_launch_graph(test_utils::GraphHelper& gHelper, hipStream_t& stream)
{
    gHelper.createAndLaunchGraph(stream);
    gHelper.launchGraphHelper(stream, true);
}

void test_float(
    std::function<rocrand_status(rocrand_generator, float*, size_t, float, float)> generate_fn,
    const generate_test_params&                                                    params)
{
    rocrand_generator generator;
    create_generator(&generator, params);

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size   = 12563;
    float        mean   = 5.0f;
    float        stddev = 2.0f;
    float*       data;
    HIP_CHECK(allocate_output(&data, size, params));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    test_utils::GraphHelper gHelper;

    gHelper.startStreamCapture(stream);

    // Any sizes
    ROCRAND_CHECK(generate_fn(generator, data, 1, mean, stddev));
    create_and_launch_graph(gHelper, stream);
    gHelper.resetGraphHelper(stream);

    // Any alignment
    ROCRAND_CHECK(generate_fn(generator, data + 1, 2, mean, stddev));

    create_and_launch_graph(gHelper, stream);
    gHelper.resetGraphHelper(stream);

    ROCRAND_CHECK(generate_fn(generator, data, size, mean, stddev));

    create_and_launch_graph(gHelper, stream);

    HIP_CHECK(free_output(data, params));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_hipgraph_generate_tests, normal_float_test)
{
    auto generator_fcn
        = [](rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
    { return rocrand_generate_normal(generator, output_data, n, mean, stddev); };

    test_float(generator_fcn, GetParam());
}

TEST_P(rocrand_hipgraph_generate_tests, log_normal_float_test)
{
    auto generator_fcn
        = [](rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
    { return rocrand_generate_log_normal(generator, output_data, n, mean, stddev); };

    test_float(generator_fcn, GetParam());
}

TEST_P(rocrand_hipgraph_generate_tests, uniform_float_test)
{
    const generate_test_params params = GetParam();

    rocrand_generator generator;
    create_generator(&generator, params);

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size = 12563;
    float*       data;
    HIP_CHECK(allocate_output(&data, size, params));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(stream);

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, 1));

    create_and_launch_graph(gHelper, stream);
    gHelper.resetGraphHelper(stream);

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data + 1, 2));

    create_and_launch_graph(gHelper, stream);
    gHelper.resetGraphHelper(stream);

    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, size));

    create_and_launch_graph(gHelper, stream);

    HIP_CHECK(free_output(data, params));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_hipgraph_generate_tests, poisson_test)
{
    const generate_test_params params = GetParam();

    rocrand_generator generator;
    create_generator(&generator, params);

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    constexpr size_t size = 12563;
    unsigned int*    data;
    HIP_CHECK(allocate_output(&data, size, params));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(stream);

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_poisson(generator, data, 1, 10.0));

    create_and_launch_graph(gHelper, stream);
    gHelper.resetGraphHelper(stream);

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_poisson(generator, data + 1, 2, 500.0));

    create_and_launch_graph(gHelper, stream);
    gHelper.resetGraphHelper(stream);

    ROCRAND_CHECK(rocrand_generate_poisson(generator, data, size, 5000.0));

    create_and_launch_graph(gHelper, stream);

    HIP_CHECK(free_output(data, params));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST(rocrand_hipgraph_generate_tests, hipgraphs_doc_sample)
{

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"

#include "hipgraphs_doc_sample.hpp"

#pragma GCC diagnostic pop
}

INSTANTIATE_TEST_SUITE_P(rocrand_hipgraph_generate_tests,
                         rocrand_hipgraph_generate_tests,
                         ::testing::Combine(::testing::ValuesIn(rng_types), ::testing::Bool()),
                         generate_test_name);

class rocrand_hipgraph_host_generator_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

// A non-blocking host generator enqueues generation as a host function, so capturing it
// creates a host node. The graph must be launchable more than once, and must keep working
// after the graph it was instantiated from is destroyed.
TEST_P(rocrand_hipgraph_host_generator_tests, relaunch)
{
    const rocrand_rng_type rng_type     = GetParam();
    constexpr size_t       size         = 4096;
    constexpr int          launch_count = 3;

    if(rng_type == ROCRAND_RNG_PSEUDO_MT19937)
    {
        ROCRAND_SKIP_SLOW_TEST_IF_NOT_ENABLED();
    }

    std::vector<float> expected(size);
    {
        rocrand_generator generator;
        ROCRAND_CHECK(rocrand_create_generator_host_blocking(&generator, rng_type));
        ROCRAND_CHECK(rocrand_generate_uniform(generator, expected.data(), size));
        ROCRAND_CHECK(rocrand_destroy_generator(generator));
    }

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator_host(&generator, rng_type));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    ROCRAND_CHECK(rocrand_set_stream(generator, stream));
    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    std::vector<float> output(size);

    hipGraph_t graph;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    ROCRAND_CHECK(rocrand_generate_uniform(generator, output.data(), size));
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    hipGraphExec_t graph_exec;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));

    for(int launch = 0; launch < launch_count; ++launch)
    {
        SCOPED_TRACE(testing::Message() << "with launch = " << launch);

        std::fill(output.begin(), output.end(), -1.0f);
        HIP_CHECK(hipGraphLaunch(graph_exec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        if(launch == 0)
        {
            ASSERT_EQ(output, expected);
        }
        else
        {
            // Stateful engines advance with every launch, so only check that values were written
            ASSERT_TRUE(std::all_of(output.begin(),
                                    output.end(),
                                    [](float value) { return value > 0.0f && value <= 1.0f; }));
        }
    }

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipStreamDestroy(stream));
}

constexpr rocrand_rng_type host_generator_rng_types[] = {ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
                                                         ROCRAND_RNG_PSEUDO_MRG31K3P,
                                                         ROCRAND_RNG_PSEUDO_MRG32K3A,
                                                         ROCRAND_RNG_PSEUDO_XORWOW,
                                                         ROCRAND_RNG_PSEUDO_MTGP32,
                                                         ROCRAND_RNG_PSEUDO_LFSR113,
                                                         ROCRAND_RNG_PSEUDO_MT19937,
                                                         ROCRAND_RNG_PSEUDO_THREEFRY2_32_20,
                                                         ROCRAND_RNG_PSEUDO_THREEFRY2_64_20,
                                                         ROCRAND_RNG_PSEUDO_THREEFRY4_32_20,
                                                         ROCRAND_RNG_PSEUDO_THREEFRY4_64_20,
                                                         ROCRAND_RNG_QUASI_SOBOL32,
                                                         ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32,
                                                         ROCRAND_RNG_QUASI_SOBOL64,
                                                         ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64};

INSTANTIATE_TEST_SUITE_P(rocrand_hipgraph_host_generator_tests,
                         rocrand_hipgraph_host_generator_tests,
                         ::testing::ValuesIn(host_generator_rng_types),
                         rocrand_rng_type_test_name);
