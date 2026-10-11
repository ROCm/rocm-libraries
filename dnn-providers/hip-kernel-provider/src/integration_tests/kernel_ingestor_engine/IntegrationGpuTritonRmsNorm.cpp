// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/TensorView.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "TestDescriptorRoot.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

/**
 * @file IntegrationGpuTritonRmsNorm.cpp
 * @brief Runs the Triton AOT RMS-norm engine end to end through the public frontend API,
 *        asserts that it served the plan, and compares the output with the CPU reference
 *        executor. Skips when no Triton shard is staged for the device's arch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

namespace
{

/// The engine this suite pins; the descriptors and the native pack restate this name.
constexpr const char* TRITON_RMSNORM_ENGINE = "hipkernel:TritonRmsNorm";

/// True when @p directory holds at least one `.ukd.json` file.
bool holdsKernelDescriptor(const std::filesystem::path& directory)
{
    std::error_code failed;
    for(const auto& entry : std::filesystem::directory_iterator(directory, failed))
    {
        const auto& path = entry.path();
        if(entry.is_regular_file(failed) && path.extension() == ".json"
           && path.stem().extension() == ".ukd")
        {
            return true;
        }
    }
    return false;
}

constexpr int64_t X_UID = 1;
constexpr int64_t SCALE_UID = 2;
constexpr int64_t EPSILON_UID = 3;
constexpr int64_t Y_UID = 4;

struct RmsNormCase
{
    hipdnn_frontend::DataType dataType;
    int64_t rows;
    int64_t columns;
    float epsilon;
};

std::string rmsNormCaseName(const ::testing::TestParamInfo<RmsNormCase>& info)
{
    const auto& c = info.param;
    return std::string(c.dataType == hipdnn_frontend::DataType::BFLOAT16 ? "Bf16" : "F16") + "_N"
           + std::to_string(c.columns) + "_rows" + std::to_string(c.rows)
           + (c.epsilon > 1e-3F ? "_epsLarge" : "_epsSmall");
}

/// A single-node RMS-norm forward graph: x (rows, columns), scale (1, columns), a baked
/// epsilon. The default tensor alignment (16) is what the pack's graph match requires.
std::shared_ptr<Graph> buildRmsNormGraph(const RmsNormCase& c)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("triton_rmsnorm")
        .set_io_data_type(c.dataType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    const std::vector<int64_t> xDims = {c.rows, c.columns};
    const std::vector<int64_t> scaleDims = {1, c.columns};

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(X_UID)
        .set_name("x")
        .set_dim(xDims)
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(xDims))
        .set_data_type(c.dataType);

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(SCALE_UID)
        .set_name("scale")
        .set_dim(scaleDims)
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(scaleDims))
        .set_data_type(c.dataType);

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(EPSILON_UID).set_name("epsilon").set_compile_time_constant(c.epsilon);

    RMSNormAttributes attributes;
    attributes.set_epsilon(epsilon);
    attributes.set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);

    auto [y, invRms] = graph->rmsnorm(x, scale, attributes);
    y->set_uid(Y_UID).set_name("y").set_output(true).set_data_type(c.dataType);
    return graph;
}

/// The worst element of |got - ref| / max(|ref|, 1e-3) must not exceed the tolerance; the
/// floor keeps a near-zero reference from turning rounding into a large ratio.
template <typename T>
class FlooredRelativeErrorValidation : public hipdnn_test_sdk::utilities::IReferenceValidation
{
public:
    explicit FlooredRelativeErrorValidation(double tolerance)
        : _tolerance(tolerance)
    {
    }

    bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                  hipdnn_data_sdk::utilities::ITensor& implementation) const override
    {
        if(reference.dims() != implementation.dims() || reference.dims().size() != 2)
        {
            ADD_FAILURE() << "reference and implementation shapes differ or are not rank 2";
            return false;
        }

        hipdnn_data_sdk::utilities::TensorView<T> refView(reference);
        hipdnn_data_sdk::utilities::TensorView<T> implView(implementation);

        double worst = 0.0;
        std::vector<int64_t> worstIndex{0, 0};
        double worstGot = 0.0;
        double worstRef = 0.0;
        for(int64_t row = 0; row < reference.dims()[0]; ++row)
        {
            for(int64_t column = 0; column < reference.dims()[1]; ++column)
            {
                const std::vector<int64_t> index{row, column};
                const auto ref
                    = static_cast<double>(static_cast<float>(refView.getHostValue(index)));
                const auto got
                    = static_cast<double>(static_cast<float>(implView.getHostValue(index)));
                // NaN compares false, so it must be caught explicitly or it would pass.
                const double error = std::isfinite(got) && std::isfinite(ref)
                                         ? std::abs(got - ref) / std::max(std::abs(ref), 1e-3)
                                         : std::numeric_limits<double>::infinity();
                if(error > worst)
                {
                    worst = error;
                    worstIndex = index;
                    worstGot = got;
                    worstRef = ref;
                }
            }
        }

        if(worst > _tolerance)
        {
            ADD_FAILURE() << "worst floored relative error " << worst << " exceeds " << _tolerance
                          << " at element (" << worstIndex[0] << ", " << worstIndex[1] << "): got "
                          << worstGot << ", reference " << worstRef;
            return false;
        }
        return true;
    }

private:
    double _tolerance;
};

} // namespace

class IntegrationGpuTritonRmsNorm
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                      RmsNormCase>
{
protected:
    void SetUp() override
    {
        IntegrationGraphVerificationHarness<float, RmsNormCase>::SetUp();
        if(IsSkipped() || HasFatalFailure())
        {
            return;
        }

        const auto root = hip_kernel_provider::testing::descriptorSetRoot(
            HIPKERNELPROVIDER_PRODUCT_DESCRIPTOR_RELDIR);
        std::error_code failed;
        if(root.empty() || !std::filesystem::is_directory(root, failed))
        {
            GTEST_SKIP() << "no production descriptor root at '" << root.string() << "'";
        }

        auto arch = hip_kernel_provider_common::getDeviceString(_stream);
        arch = arch.substr(0, arch.find(':'));
        // The packer stages each arch's tree under its own shard, keeping the source path.
        const auto shard = root / arch / "Triton" / "rmsnorm" / arch;
        if(!holdsKernelDescriptor(shard))
        {
            GTEST_SKIP() << "no Triton RMSNorm descriptors are staged for " << arch << " at '"
                         << shard.string() << "'";
        }
    }

    /// Fills x and scale only. The epsilon tensor is a pass-by-value constant the graph
    /// already carries, so randomizing it would overwrite the value under test.
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        for(auto& tensorPair : bundle.tensors)
        {
            if(tensorPair.first == EPSILON_UID)
            {
                continue;
            }
            bundle.randomizeTensor(tensorPair.first, DEFAULT_MIN, DEFAULT_MAX, seed + 1);
        }
    }

    /// Registers the floored-relative-error comparator for every non-virtual output.
    template <typename T>
    void registerOutputValidators(
        hip_kernel_provider::test_utilities::GraphVerificationContext& context,
        Graph& graph,
        double tolerance)
    {
        graph.visit([&](const hipdnn_frontend::graph::INode& node) {
            for(const auto& attr : node.getNodeOutputTensorAttributes())
            {
                if(!attr->get_is_virtual())
                {
                    registerValidator(
                        context,
                        attr,
                        std::make_unique<FlooredRelativeErrorValidation<T>>(tolerance));
                }
            }
        });
    }

    static int64_t tritonEngineId()
    {
        return hipdnn_data_sdk::utilities::engineNameToId(TRITON_RMSNORM_ENGINE);
    }

    /// Pins the Triton engine and builds the plan, asserting every step.
    void buildAndCompile(Graph& graph)
    {
        graph.set_preferred_engine_id_ext(tritonEngineId());

        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // hip_mlops and the CPU path claim RMSNorm too: this checks catalog membership,
        // the serving engine is checked below.
        std::vector<int64_t> rankedEngineIds;
        result = graph.get_ranked_engine_ids(rankedEngineIds);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        ASSERT_NE(std::find(rankedEngineIds.begin(), rankedEngineIds.end(), tritonEngineId()),
                  rankedEngineIds.end())
            << "the Triton RMSNorm engine did not offer itself for this graph";

        result = graph.create_execution_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        // The preferred engine falls back silently, so only the plan's own engine id proves
        // which engine will run.
        int64_t servingEngineId = 0;
        ASSERT_EQ(graph.get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
        ASSERT_EQ(servingEngineId, tritonEngineId())
            << "engine id " << servingEngineId << " served the graph, not the "
            << TRITON_RMSNORM_ENGINE << " engine";
    }
};

TEST_P(IntegrationGpuTritonRmsNorm, ServesTheGraphThroughThePublicApiAndMatchesTheCpuReference)
{
    const auto& c = GetParam();
    auto graph = buildRmsNormGraph(c);
    ASSERT_NO_FATAL_FAILURE(buildAndCompile(*graph));

    hip_kernel_provider::test_utilities::GraphVerificationContext context(*graph);
    if(c.dataType == hipdnn_frontend::DataType::BFLOAT16)
    {
        registerOutputValidators<hipdnn_data_sdk::types::bfloat16>(context, *graph, 2e-2);
    }
    else
    {
        registerOutputValidators<hipdnn_data_sdk::types::half>(context, *graph, 5e-3);
    }
    verifyBuiltGraph(context, /*seed=*/0);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    IntegrationGpuTritonRmsNorm,
    ::testing::ValuesIn([] {
        std::vector<RmsNormCase> cases;
        for(const auto dataType :
            {hipdnn_frontend::DataType::BFLOAT16, hipdnn_frontend::DataType::HALF})
        {
            // Width 1, a small odd width, an odd width inside a block, an exact block
            // boundary, and the two largest blocks (8192 is the widest served).
            for(const int64_t columns : {1, 7, 1001, 1024, 4096, 8192})
            {
                for(const float epsilon : {1e-5F, 1e-2F})
                {
                    cases.push_back({dataType, 7, columns, epsilon});
                }
            }
        }
        // The FlyDSL RMSNorm workload's widths, at its row count.
        for(const auto dataType :
            {hipdnn_frontend::DataType::BFLOAT16, hipdnn_frontend::DataType::HALF})
        {
            for(const int64_t columns : {640, 2048, 3072, 3584, 5120})
            {
                cases.push_back({dataType, 4, columns, 1e-5F});
            }
        }
        return cases;
    }()),
    rmsNormCaseName);

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
