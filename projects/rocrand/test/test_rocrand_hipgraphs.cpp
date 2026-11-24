#include <gtest/gtest.h>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_utils_hipgraphs.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

class rocrand_hipgraph_generate_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

void test_float(
    std::function<rocrand_status(rocrand_generator, float*, size_t, float, float)> generate_fn,
    rocrand_rng_type                                                               rng_type)
{
    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size   = 12563;
    float        mean   = 5.0f;
    float        stddev = 2.0f;
    float*       data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    test_utils::GraphHelper gHelper;

    gHelper.startStreamCapture(stream);

    // Any sizes
    ROCRAND_CHECK(generate_fn(generator, data, 1, mean, stddev));
    gHelper.createAndLaunchGraph(stream);
    gHelper.resetGraphHelper(stream);

    // Any alignment
    ROCRAND_CHECK(generate_fn(generator, data + 1, 2, mean, stddev));

    gHelper.createAndLaunchGraph(stream);
    gHelper.resetGraphHelper(stream);

    ROCRAND_CHECK(generate_fn(generator, data, size, mean, stddev));

    gHelper.createAndLaunchGraph(stream);

    HIP_CHECK(hipFree(data));
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
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size = 12563;
    float*       data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(stream);

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, 1));

    gHelper.createAndLaunchGraph(stream);
    gHelper.resetGraphHelper(stream);

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data + 1, 2));

    gHelper.createAndLaunchGraph(stream);
    gHelper.resetGraphHelper(stream);

    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, size));

    gHelper.createAndLaunchGraph(stream);

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_hipgraph_generate_tests, poisson_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    constexpr size_t size = 12563;
    unsigned int*    data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(*data)));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(stream);

    // Any sizes
    ROCRAND_CHECK(rocrand_generate_poisson(generator, data, 1, 10.0));

    gHelper.createAndLaunchGraph(stream);
    gHelper.resetGraphHelper(stream);

    // Any alignment
    ROCRAND_CHECK(rocrand_generate_poisson(generator, data + 1, 2, 500.0));

    gHelper.createAndLaunchGraph(stream);
    gHelper.resetGraphHelper(stream);

    ROCRAND_CHECK(rocrand_generate_poisson(generator, data, size, 5000.0));

    gHelper.createAndLaunchGraph(stream);

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipStreamDestroy(stream));
}

// Number of positions where the two equally sized buffers differ.
size_t count_mismatches(const std::vector<unsigned int>& a, const std::vector<unsigned int>& b)
{
    size_t mismatches = 0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        mismatches += a[i] != b[i];
    }
    return mismatches;
}

// ASSERT_EQ-based ROCRAND_CHECK cannot be used in a constructor or destructor
// (fatal assertions require a void-returning function), so use the same
// lambda+exit workaround that HIP_CHECK above uses for hipError_t.
#define ROCRAND_CHECK_CTOR(state)                                      \
    {                                                                  \
        const rocrand_status status_ = (state);                        \
        if(status_ != ROCRAND_STATUS_SUCCESS)                          \
        {                                                              \
            [status_]() { FAIL() << "rocRAND error: " << status_; }(); \
            exit(status_);                                             \
        }                                                              \
    }

// Generator, stream and device buffers for the Poisson lambda-change tests.
class poisson_lambda_change_fixture
{
public:
    static constexpr size_t size = 65536;

    poisson_lambda_change_fixture(const rocrand_rng_type rng_type, const size_t num_buffers)
        : buffers(num_buffers)
    {
        ROCRAND_CHECK_CTOR(rocrand_create_generator(&generator, rng_type));
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        ROCRAND_CHECK_CTOR(rocrand_set_stream(generator, stream));
        for(auto& buffer : buffers)
        {
            HIP_CHECK(hipMallocHelper(&buffer, sizeof(*buffer) * size));
        }
    }

    ~poisson_lambda_change_fixture()
    {
        for(auto* buffer : buffers)
        {
            HIP_CHECK(hipFree(buffer));
        }
        ROCRAND_CHECK_CTOR(rocrand_destroy_generator(generator));
        HIP_CHECK(hipStreamDestroy(stream));
    }

    void generate(const size_t buffer, const double lambda)
    {
        ROCRAND_CHECK(rocrand_generate_poisson(generator, buffers[buffer], size, lambda));
    }

    std::vector<unsigned int> to_host(const size_t buffer)
    {
        std::vector<unsigned int> result(size);
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipMemcpy(result.data(),
                            buffers[buffer],
                            sizeof(*buffers[buffer]) * size,
                            hipMemcpyDeviceToHost));
        return result;
    }

    rocrand_generator          generator;
    hipStream_t                stream;
    std::vector<unsigned int*> buffers;
};

// Reference output: the lambdas generated in order, synchronizing after each call.
std::vector<std::vector<unsigned int>> poisson_synced_reference(const rocrand_rng_type     rng_type,
                                                                const std::vector<double>& lambdas)
{
    poisson_lambda_change_fixture          f(rng_type, lambdas.size());
    std::vector<std::vector<unsigned int>> result;
    for(size_t i = 0; i < lambdas.size(); ++i)
    {
        f.generate(i, lambdas[i]);
        result.push_back(f.to_host(i));
    }
    return result;
}

// Eager Poisson(A), Poisson(B); capture Poisson(A), Poisson(B); launch.
// The captured Poisson(B) must not read the table of Poisson(A).
TEST_P(rocrand_hipgraph_generate_tests, poisson_lambda_change_warmup_capture)
{
    constexpr double la       = 1000.0;
    constexpr double lb       = 10.0;
    const auto       expected = poisson_synced_reference(GetParam(), {la, lb, la, lb});

    poisson_lambda_change_fixture f(GetParam(), 4);
    f.generate(0, la);
    f.generate(1, lb);
    HIP_CHECK(hipStreamSynchronize(f.stream));

    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(f.stream);
    f.generate(2, la);
    f.generate(3, lb);
    gHelper.createAndLaunchGraph(f.stream);
    gHelper.cleanupGraphHelper();

    EXPECT_EQ(count_mismatches(f.to_host(2), expected[2]), 0u);
    EXPECT_EQ(count_mismatches(f.to_host(3), expected[3]), 0u);
}

// Eager Poisson(A); capture Poisson(B), Poisson(A); launch.
// The captured Poisson(A) must not read the table of Poisson(B).
TEST_P(rocrand_hipgraph_generate_tests, poisson_lambda_change_capture_hit)
{
    constexpr double la       = 10.0;
    constexpr double lb       = 1000.0;
    const auto       expected = poisson_synced_reference(GetParam(), {la, lb, la});

    poisson_lambda_change_fixture f(GetParam(), 3);
    f.generate(0, la);
    HIP_CHECK(hipStreamSynchronize(f.stream));

    test_utils::GraphHelper gHelper;
    gHelper.startStreamCapture(f.stream);
    f.generate(1, lb);
    f.generate(2, la);
    gHelper.createAndLaunchGraph(f.stream);
    gHelper.cleanupGraphHelper();

    EXPECT_EQ(count_mismatches(f.to_host(1), expected[1]), 0u);
    EXPECT_EQ(count_mismatches(f.to_host(2), expected[2]), 0u);
}

// Eager Poisson(A); capture Poisson(A); eager Poisson(B); launch.
// The graph must carry its own table update instead of relying on the table
// that was current at capture time.
TEST_P(rocrand_hipgraph_generate_tests, poisson_lambda_change_after_capture)
{
    constexpr double la = 10.0;
    constexpr double lb = 1000.0;

    poisson_lambda_change_fixture f(GetParam(), 3);
    f.generate(0, la);
    HIP_CHECK(hipStreamSynchronize(f.stream));

    hipGraph_t graph;
    HIP_CHECK(hipStreamBeginCapture(f.stream, hipStreamCaptureModeGlobal));
    f.generate(1, la);
    HIP_CHECK(hipStreamEndCapture(f.stream, &graph));

    f.generate(2, lb);
    HIP_CHECK(hipStreamSynchronize(f.stream));

    hipGraphExec_t graph_instance;
    HIP_CHECK(hipGraphInstantiate(&graph_instance, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graph_instance, f.stream));
    HIP_CHECK(hipStreamSynchronize(f.stream));
    HIP_CHECK(hipGraphExecDestroy(graph_instance));
    HIP_CHECK(hipGraphDestroy(graph));

    // The generator state differs from a synchronized run, so check the distribution.
    const auto   values   = f.to_host(1);
    const double mean     = get_mean(values);
    const double variance = get_variance(values, mean);
    EXPECT_NEAR(mean, la, std::max(1.0, la * 3e-2));
    EXPECT_NEAR(variance, la, std::max(1.0, la * 2e-2));
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
                         ::testing::ValuesIn(rng_types),
                         rocrand_rng_type_test_name);

class rocrand_hipgraph_reinit_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

// Generators that keep their engine states in an array passed as the first kernel argument
constexpr rocrand_rng_type engine_rng_types[] = {ROCRAND_RNG_PSEUDO_MRG31K3P,
                                                 ROCRAND_RNG_PSEUDO_MRG32K3A,
                                                 ROCRAND_RNG_PSEUDO_XORWOW,
                                                 ROCRAND_RNG_PSEUDO_MTGP32,
                                                 ROCRAND_RNG_PSEUDO_LFSR113,
                                                 ROCRAND_RNG_PSEUDO_MT19937};

struct reinit_case
{
    const char*                                      name;
    rocrand_ordering                                 initial_order;
    std::function<rocrand_status(rocrand_generator)> reconfigure;
};

// Captures a generation into a graph, reconfigures and reinitializes the generator, then
// checks that the engine array used by the graph is still allocated and that launching the
// graph doesn't write into memory allocated after the reinitialization.
void test_reinit_after_capture(const rocrand_rng_type rng_type, const reinit_case& test_case)
{
    SCOPED_TRACE(testing::Message() << "with " << test_case.name);

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));
    if(rocrand_set_ordering(generator, test_case.initial_order) != ROCRAND_STATUS_SUCCESS)
    {
        ROCRAND_CHECK(rocrand_destroy_generator(generator));
        return;
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    ROCRAND_CHECK(rocrand_set_stream(generator, stream));
    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size = 1 << 20;
    float*       data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(*data)));

    hipGraph_t graph;
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    ROCRAND_CHECK(rocrand_generate_uniform(generator, data, size));
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    hipGraphExec_t graph_exec;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // Find the engine arrays used by the captured kernels
    size_t node_count = 0;
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &node_count));
    std::vector<hipGraphNode_t> nodes(node_count);
    HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &node_count));
    std::vector<std::pair<hipDeviceptr_t, size_t>> engine_ranges;
    for(hipGraphNode_t node : nodes)
    {
        hipGraphNodeType type;
        HIP_CHECK(hipGraphNodeGetType(node, &type));
        if(type != hipGraphNodeTypeKernel)
        {
            continue;
        }
        hipKernelNodeParams params{};
        HIP_CHECK(hipGraphKernelNodeGetParams(node, &params));
        ASSERT_NE(params.kernelParams, nullptr);
        const auto engines
            = static_cast<hipDeviceptr_t>(*static_cast<void**>(params.kernelParams[0]));
        hipDeviceptr_t base;
        size_t         range_size;
        HIP_CHECK(hipMemGetAddressRange(&base, &range_size, engines));
        engine_ranges.emplace_back(base, range_size);
    }
    ASSERT_FALSE(engine_ranges.empty());

    const rocrand_status status = test_case.reconfigure(generator);
    if(status == ROCRAND_STATUS_SUCCESS)
    {
        ROCRAND_CHECK(rocrand_initialize_generator(generator));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<unsigned char*> victims;
        for(const auto& [base, range_size] : engine_ranges)
        {
            // The engine array must not have been freed
            hipDeviceptr_t current_base = nullptr;
            size_t         current_size = 0;
            EXPECT_EQ(hipMemGetAddressRange(&current_base, &current_size, base), hipSuccess);
            static_cast<void>(hipGetLastError());
            EXPECT_EQ(current_base, base);
            EXPECT_EQ(current_size, range_size);

            // If it was freed, an allocation of the same size would likely reuse it
            unsigned char* victim;
            HIP_CHECK(hipMalloc(&victim, range_size));
            HIP_CHECK(hipMemset(victim, 0xAB, range_size));
            victims.push_back(victim);
        }
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipGraphLaunch(graph_exec, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        for(size_t i = 0; i < victims.size(); ++i)
        {
            std::vector<unsigned char> h_victim(engine_ranges[i].second);
            HIP_CHECK(
                hipMemcpy(h_victim.data(), victims[i], h_victim.size(), hipMemcpyDeviceToHost));
            const size_t changed = std::count_if(h_victim.begin(),
                                                 h_victim.end(),
                                                 [](const unsigned char v) { return v != 0xAB; });
            EXPECT_EQ(changed, 0);
            HIP_CHECK(hipFree(victims[i]));
        }
    }
    else
    {
        // Setting not supported by this generator
        EXPECT_TRUE(status == ROCRAND_STATUS_TYPE_ERROR || status == ROCRAND_STATUS_OUT_OF_RANGE);
    }

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_hipgraph_reinit_tests, reinit_after_capture)
{
    const reinit_case cases[] = {
        {                 "set_seed",
         ROCRAND_ORDERING_PSEUDO_DEFAULT,               [](rocrand_generator g) { return rocrand_set_seed(g, 12345); }},
        {               "set_offset",
         ROCRAND_ORDERING_PSEUDO_DEFAULT,              [](rocrand_generator g) { return rocrand_set_offset(g, 1000); }},
        {  "set_ordering to dynamic",
         ROCRAND_ORDERING_PSEUDO_DEFAULT, [](rocrand_generator g)
 { return rocrand_set_ordering(g, ROCRAND_ORDERING_PSEUDO_DYNAMIC); }        },
        {"set_ordering from dynamic",
         ROCRAND_ORDERING_PSEUDO_DYNAMIC, [](rocrand_generator g)
 { return rocrand_set_ordering(g, ROCRAND_ORDERING_PSEUDO_DEFAULT); }        },
        {   "set_ordering to legacy",
         ROCRAND_ORDERING_PSEUDO_DEFAULT,  [](rocrand_generator g)
  { return rocrand_set_ordering(g, ROCRAND_ORDERING_PSEUDO_LEGACY); }        },
    };
    for(const reinit_case& test_case : cases)
    {
        test_reinit_after_capture(GetParam(), test_case);
    }
}

INSTANTIATE_TEST_SUITE_P(rocrand_hipgraph_reinit_tests,
                         rocrand_hipgraph_reinit_tests,
                         ::testing::ValuesIn(engine_rng_types),
                         rocrand_rng_type_test_name);
