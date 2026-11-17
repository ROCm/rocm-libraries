#include <gtest/gtest.h>
#include <stdio.h>

#include <vector>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_utils_hipgraphs.hpp"

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

class rocrand_hipgraph_replay_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

namespace
{

constexpr rocrand_rng_type replay_rng_types[] = {ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
                                                 ROCRAND_RNG_PSEUDO_MRG31K3P,
                                                 ROCRAND_RNG_PSEUDO_MRG32K3A,
                                                 ROCRAND_RNG_PSEUDO_XORWOW,
                                                 ROCRAND_RNG_PSEUDO_MTGP32,
                                                 ROCRAND_RNG_PSEUDO_MT19937,
                                                 ROCRAND_RNG_PSEUDO_LFSR113,
                                                 ROCRAND_RNG_PSEUDO_THREEFRY2_32_20,
                                                 ROCRAND_RNG_PSEUDO_THREEFRY2_64_20,
                                                 ROCRAND_RNG_PSEUDO_THREEFRY4_32_20,
                                                 ROCRAND_RNG_PSEUDO_THREEFRY4_64_20,
                                                 ROCRAND_RNG_QUASI_SOBOL32,
                                                 ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32,
                                                 ROCRAND_RNG_QUASI_SOBOL64,
                                                 ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64};

// Generators that continue from where the previous launch stopped, exactly like eager calls.
// The others (XORWOW, MRG, LFSR113, MTGP32) also return new values on every launch, but they
// map engines to outputs from a host-side index fixed at capture time.
bool replay_matches_eager(const rocrand_rng_type rng_type)
{
    switch(rng_type)
    {
        case ROCRAND_RNG_PSEUDO_PHILOX4_32_10:
        case ROCRAND_RNG_PSEUDO_MT19937:
        case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20:
        case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20:
        case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20:
        case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20:
        case ROCRAND_RNG_QUASI_SOBOL32:
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
        case ROCRAND_RNG_QUASI_SOBOL64:
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64: return true;
        default: return false;
    }
}

struct replay_generator
{
    rocrand_generator generator = nullptr;
    hipStream_t       stream    = nullptr;

    explicit replay_generator(const rocrand_rng_type rng_type)
    {
        HIP_CHECK_NON_VOID(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        EXPECT_EQ(rocrand_create_generator(&generator, rng_type), ROCRAND_STATUS_SUCCESS);
        EXPECT_EQ(rocrand_set_stream(generator, stream), ROCRAND_STATUS_SUCCESS);
        EXPECT_EQ(rocrand_initialize_generator(generator), ROCRAND_STATUS_SUCCESS);
    }

    ~replay_generator()
    {
        EXPECT_EQ(rocrand_destroy_generator(generator), ROCRAND_STATUS_SUCCESS);
        HIP_CHECK_NON_VOID(hipStreamDestroy(stream));
    }

    std::vector<unsigned int> read(const unsigned int* data, const size_t size) const
    {
        std::vector<unsigned int> host(size);
        HIP_CHECK_NON_VOID(hipStreamSynchronize(stream));
        HIP_CHECK_NON_VOID(
            hipMemcpy(host.data(), data, size * sizeof(unsigned int), hipMemcpyDeviceToHost));
        return host;
    }
};

size_t count_equal(const std::vector<unsigned int>& a, const std::vector<unsigned int>& b)
{
    size_t count = 0;
    for(size_t i = 0; i < a.size(); ++i)
    {
        count += a[i] == b[i];
    }
    return count;
}

} // namespace

// A generate call captured once and launched several times must continue the sequence on every
// launch instead of repeating the values of the first launch.
TEST_P(rocrand_hipgraph_replay_tests, replay_continues_sequence)
{
    const rocrand_rng_type rng_type = GetParam();
    constexpr int          launches = 3;

    // A size that is not a multiple of any block or engine count, and (for MT19937) one small
    // enough to be served from values already generated by the previous call.
    for(const size_t size : {size_t{100003}, size_t{1000}})
    {
        SCOPED_TRACE(testing::Message() << "with size = " << size);

        unsigned int* data;
        HIP_CHECK(hipMallocHelper(&data, size * sizeof(*data)));

        // Reference: the same calls made eagerly on an identical generator.
        std::vector<std::vector<unsigned int>> expected;
        {
            replay_generator reference(rng_type);
            for(int i = 0; i < launches + 2; ++i)
            {
                ROCRAND_CHECK(rocrand_generate(reference.generator, data, size));
                expected.push_back(reference.read(data, size));
            }
        }

        replay_generator        replay(rng_type);
        test_utils::GraphHelper gHelper;

        // One eager call before the capture.
        ROCRAND_CHECK(rocrand_generate(replay.generator, data, size));
        ASSERT_EQ(replay.read(data, size), expected[0]);

        gHelper.startStreamCapture(replay.stream);
        ROCRAND_CHECK(rocrand_generate(replay.generator, data, size));
        gHelper.createAndLaunchGraph(replay.stream, false, false);

        std::vector<unsigned int> previous = expected[0];
        for(int i = 0; i < launches; ++i)
        {
            SCOPED_TRACE(testing::Message() << "with launch = " << i);
            gHelper.launchGraphHelper(replay.stream, true);
            const std::vector<unsigned int> output = replay.read(data, size);

            ASSERT_LT(count_equal(output, previous), size / 100);
            if(replay_matches_eager(rng_type))
            {
                ASSERT_EQ(output, expected[i + 1]);
            }
            previous = output;
        }

        // An eager call after the launches continues after them.
        ROCRAND_CHECK(rocrand_generate(replay.generator, data, size));
        const std::vector<unsigned int> output = replay.read(data, size);
        ASSERT_LT(count_equal(output, previous), size / 100);
        if(replay_matches_eager(rng_type))
        {
            ASSERT_EQ(output, expected[launches + 1]);
        }

        gHelper.cleanupGraphHelper();
        HIP_CHECK(hipFree(data));
    }
}

// Re-initializing a generator (here after changing the seed or offset) must keep the memory
// that a captured graph uses, so launching the graph afterwards continues from the
// re-initialized state instead of reading freed memory.
TEST_P(rocrand_hipgraph_replay_tests, replay_after_reinitialization)
{
    const rocrand_rng_type rng_type = GetParam();
    const bool             quasi    = rng_type == ROCRAND_RNG_QUASI_SOBOL32
                       || rng_type == ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
                       || rng_type == ROCRAND_RNG_QUASI_SOBOL64
                       || rng_type == ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    constexpr size_t   size = 100003;
    constexpr unsigned seed = 1234;

    auto reinitialize = [&](rocrand_generator generator)
    {
        if(quasi)
        {
            ROCRAND_CHECK(rocrand_set_offset(generator, seed));
        }
        else
        {
            ROCRAND_CHECK(rocrand_set_seed(generator, seed));
        }
        ROCRAND_CHECK(rocrand_initialize_generator(generator));
    };

    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(*data)));

    // Reference: the first call after the same re-initialization.
    std::vector<unsigned int> expected;
    {
        replay_generator reference(rng_type);
        reinitialize(reference.generator);
        ROCRAND_CHECK(rocrand_generate(reference.generator, data, size));
        expected = reference.read(data, size);
    }

    replay_generator        replay(rng_type);
    test_utils::GraphHelper gHelper;

    gHelper.startStreamCapture(replay.stream);
    ROCRAND_CHECK(rocrand_generate(replay.generator, data, size));
    gHelper.createAndLaunchGraph(replay.stream);
    const std::vector<unsigned int> before = replay.read(data, size);

    reinitialize(replay.generator);

    gHelper.launchGraphHelper(replay.stream, true);
    const std::vector<unsigned int> output = replay.read(data, size);
    ASSERT_EQ(output, expected);
    ASSERT_LT(count_equal(output, before), size / 100);

    gHelper.cleanupGraphHelper();
    HIP_CHECK(hipFree(data));
}

INSTANTIATE_TEST_SUITE_P(rocrand_hipgraph_replay_tests,
                         rocrand_hipgraph_replay_tests,
                         ::testing::ValuesIn(replay_rng_types),
                         rocrand_rng_type_test_name);
