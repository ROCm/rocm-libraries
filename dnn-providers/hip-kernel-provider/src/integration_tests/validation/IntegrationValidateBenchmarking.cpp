// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>
#include <hipdnn_test_sdk/utilities/ScopedEnvironmentVariableSetter.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "../kernel_ingestor_engine/ScopedPluginLogCapture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_test_sdk::utilities;
using namespace hip_kernel_provider::kernel_ingestor_engine::integration;

/**
 * @file IntegrationValidateBenchmarking.cpp
 * @brief Output validation of benchmark candidates, end to end on a device, against the
 *        validation_fixture descriptor set: an engine whose catalog holds two kernels that
 *        compute a + b slowly and one kernel that computes a - b and is the fastest.
 *
 * The cases live in their own binary because the descriptor catalog is discovered once
 * per process, and this binary must see only the validation_fixture set.
 */
namespace hip_kernel_provider::validation_tests
{

namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:validation_pointwise";

/// The id of the fixture's wrong kernel, `validation_fixture/pointwise/
/// validation_pointwise_add_wrong.ukd.json`. Plan logs name a kernel by this id.
constexpr const char* WRONG_KERNEL_ID = "a3a1df80-9fbd-442c-af8d-c809d41330bd";

/// Chosen so that a + b and a - b differ, and so that both are exact in float.
constexpr float INPUT_A = 3.0f;
constexpr float INPUT_B = 1.5f;
constexpr float EXPECTED_SUM = INPUT_A + INPUT_B;
constexpr float EXPECTED_DIFFERENCE = INPUT_A - INPUT_B;

/// The harness is parameterized on a case type this binary does not use.
struct NoCase
{
};

std::shared_ptr<TensorAttributes> makeScalarTensor(int64_t uid, const std::string& name)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({1, 1, 1, 1})
        .set_stride({1, 1, 1, 1})
        .set_data_type(DataType::FLOAT);
    return tensor;
}

std::shared_ptr<Graph> buildPointwiseAddGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("validation_pointwise_add")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto a = makeScalarTensor(1, "A");
    auto b = makeScalarTensor(2, "B");

    PointwiseAttributes attrs;
    attrs.set_name("pointwise").set_mode(PointwiseMode::ADD);
    auto c = graph->pointwise(a, b, attrs);
    c->set_uid(3).set_name("C").set_output(true).set_data_type(DataType::FLOAT);

    return graph;
}

/// One float on the device.
class DeviceScalar
{
public:
    explicit DeviceScalar(float value)
    {
        EXPECT_EQ(hipMalloc(&_device, sizeof(float)), hipSuccess);
        EXPECT_EQ(hipMemcpy(_device, &value, sizeof(float), hipMemcpyHostToDevice), hipSuccess);
    }

    ~DeviceScalar()
    {
        static_cast<void>(hipFree(_device));
    }

    DeviceScalar(const DeviceScalar&) = delete;
    DeviceScalar& operator=(const DeviceScalar&) = delete;
    DeviceScalar(DeviceScalar&&) = delete;
    DeviceScalar& operator=(DeviceScalar&&) = delete;

    void* get() const
    {
        return _device;
    }

    float read() const
    {
        float value = 0.0f;
        EXPECT_EQ(hipMemcpy(&value, _device, sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
        return value;
    }

private:
    void* _device = nullptr;
};

} // namespace

class IntegrationValidateBenchmarking
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float, NoCase>
{
protected:
    /// A cache root private to one case. A winner shard recorded by one case would be
    /// served to the next, and the cases differ in whether validation ran.
    ScopedTestCacheDir _cacheDir{"validation-case", ScopedTestCacheDir::Scope::TEST};

    /// Builds the graph with benchmarking on and the fixture engine pinned, then executes it
    /// twice. The first execute resolves the sweep; the second serves the winner. Returns
    /// the value each execute left in C.
    std::vector<float> buildAndExecuteTwice()
    {
        auto graph = buildPointwiseAddGraph();
        graph->set_preferred_engine_id_ext(hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME));

        auto result = graph->build_operation_graph(_handle);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        std::vector<int64_t> rankedEngineIds;
        result = graph->get_ranked_engine_ids(rankedEngineIds);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(rankedEngineIds.empty())
        {
            ADD_FAILURE() << "the fixture engine '" << ENGINE_NAME << "' was not ranked";
            return {};
        }

        const std::vector<KnobSetting> knobSettings{
            KnobSetting(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, int64_t{1})};
        result = graph->create_execution_plan_ext(rankedEngineIds.front(), knobSettings);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph->check_support();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph->build_plans();
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        int64_t workspaceSize = 0;
        result = graph->get_workspace_size(workspaceSize);
        EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        if(workspaceSize < 0)
        {
            ADD_FAILURE() << "negative workspace size " << workspaceSize;
            return {};
        }
        const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        const DeviceScalar a(INPUT_A);
        const DeviceScalar b(INPUT_B);
        const DeviceScalar c(0.0f);

        std::vector<float> outputs;
        for(int execution = 0; execution < 2; ++execution)
        {
            std::unordered_map<int64_t, void*> variantPack{
                {1, a.get()}, {2, b.get()}, {3, c.get()}};
            result = graph->execute(_handle, variantPack, workspace.get());
            EXPECT_EQ(result.code, ErrorCode::OK) << result.err_msg;
            EXPECT_EQ(hipStreamSynchronize(stream()), hipSuccess);
            outputs.push_back(c.read());
        }

        return outputs;
    }
};

/// Mode on: the wrong kernel is the fastest, but its output disagrees with the other two,
/// so it cannot win. The assertions do not depend on timing: a validated resolve serves a
/// passing kernel whether or not any candidate yielded a usable time (HIP event readings
/// are unreliable on Windows, ROCm/rocm-systems#12925), so the winner log line is asserted
/// only where timing is reliable.
TEST_F(IntegrationValidateBenchmarking, TheWrongKernelIsExcludedWhenValidationIsOn)
{
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const ScopedEnvironmentVariableSetter validation(
        hipdnn_plugin_sdk::VALIDATE_BENCHMARKING_ENV_NAME, "1");

    const auto outputs = buildAndExecuteTwice();

    ASSERT_EQ(outputs.size(), 2U) << recorder.getRecordedLogsAsString();
    EXPECT_EQ(outputs[0], EXPECTED_SUM) << "the first execute served a kernel whose output is "
                                           "not a + b. Captured logs:\n"
                                        << recorder.getRecordedLogsAsString();
    EXPECT_EQ(outputs[1], EXPECTED_SUM) << "the second execute served a kernel whose output is "
                                           "not a + b. Captured logs:\n"
                                        << recorder.getRecordedLogsAsString();

    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, WRONG_KERNEL_ID))
        << "no WARN names the wrong kernel '" << WRONG_KERNEL_ID << "'. Captured logs:\n"
        << recorder.getRecordedLogsAsString();

#ifndef _WIN32
    // HIP event timing is reliable here, so a sweep that validated its candidates resolves a
    // winner.
    EXPECT_TRUE(recorder.hasLogContaining("benchmarking selected kernel"))
        << "the sampling sweep did not resolve a winner. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
#endif
}

/// Mode off: nothing compares the candidates' outputs, so the fastest kernel wins, and the
/// fixture's wrong kernel is the fastest. This is the proof that the mode-on case above
/// does not pass vacuously. It depends on timing, which is unreliable on Windows, so the
/// DISABLED_ prefix keeps it out of every CI tier; run it with
/// --gtest_also_run_disabled_tests.
TEST_F(IntegrationValidateBenchmarking, DISABLED_TheWrongKernelWinsWhenValidationIsOff)
{
    const ScopedPluginLogCapture capture(this);
    auto& recorder = capture.recorder();
    const ScopedEnvironmentVariableSetter validation(
        hipdnn_plugin_sdk::VALIDATE_BENCHMARKING_ENV_NAME, "0");

    const auto outputs = buildAndExecuteTwice();

    ASSERT_EQ(outputs.size(), 2U) << recorder.getRecordedLogsAsString();
    EXPECT_EQ(outputs[0], EXPECTED_DIFFERENCE)
        << "the fastest kernel is not the fixture's wrong one, so the mode-on case proves "
           "nothing. Captured logs:\n"
        << recorder.getRecordedLogsAsString();
    EXPECT_EQ(outputs[1], EXPECTED_DIFFERENCE) << "Captured logs:\n"
                                               << recorder.getRecordedLogsAsString();
}

} // namespace hip_kernel_provider::validation_tests
