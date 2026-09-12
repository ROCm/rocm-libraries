// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

namespace hip_kernel_provider::test_utilities
{
namespace
{
using hipdnn_frontend::DataType;
using hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::TensorAttributes;
using hipdnn_test_sdk::utilities::GraphTensorBundle;

std::shared_ptr<TensorAttributes> makeInput(int64_t uid, DataType dataType)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1}).set_data_type(dataType);
    return tensor;
}

std::shared_ptr<TensorAttributes> makePointwise(Graph& graph, DataType dataType)
{
    graph.set_io_data_type(dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);
    graph.set_preferred_engine_id_ext(
        hipdnn_data_sdk::utilities::engineNameToId("hipkernel:Pointwise"));
    hipdnn_frontend::graph::PointwiseAttributes attributes;
    attributes.set_mode(hipdnn_frontend::PointwiseMode::ADD);
    auto output = graph.pointwise(makeInput(1, dataType), makeInput(2, dataType), attributes);
    output->set_uid(3).set_output(true).set_data_type(dataType);
    return output;
}

// Capture only the intentionally rejected verification, not setup or its positive control.
template <typename Verify>
void expectVerificationFailure(Verify&& verify)
{
    testing::TestPartResultArray results;
    {
        const testing::ScopedFakeTestPartResultReporter reporter(
            testing::ScopedFakeTestPartResultReporter::INTERCEPT_ONLY_CURRENT_THREAD, &results);
        verify();
    }
    ASSERT_GT(results.size(), 0);
    for(int i = 0; i < results.size(); ++i)
    {
        EXPECT_TRUE(results.GetTestPartResult(i).fatally_failed());
    }
}

class IntegrationGraphVerificationContext : public IntegrationGraphVerificationHarness<float, int>
{
protected:
    void initializeBundle(const Graph& /*graph*/,
                          GraphTensorBundle& bundle,
                          unsigned int /*seed*/) override
    {
        const bool cpu = (_initializations++ % 2) != 0;
        bundle.getTensor(1).fillTensorWithValue(1.0f);
        bundle.getTensor(2).fillTensorWithValue(cpu && _differentInputs ? 2.0f : 1.0f);
    }

    int _initializations = 0;
    bool _differentInputs = false;
};

TEST_F(IntegrationGraphVerificationContext, NewGraphCannotBorrowPreviousRegistration)
{
    Graph previous;
    auto previousOutput = makePointwise(previous, DataType::FLOAT);
    GraphVerificationContext previousContext(previous);
    registerValidator(previousContext, previousOutput, 1.0f);
    ASSERT_NO_FATAL_FAILURE(verifyGraph(previousContext, 0));

    Graph current;
    auto currentOutput = makePointwise(current, DataType::FLOAT);
    GraphVerificationContext currentContext(current);
    expectVerificationFailure([&] { verifyGraph(currentContext, 0); });
    EXPECT_EQ(_initializations, 4);

    registerValidator(currentContext, currentOutput, 0.0f);
    ASSERT_NO_FATAL_FAILURE(verifyBuiltGraph(currentContext, 0));
    ASSERT_NO_FATAL_FAILURE(verifyBuiltGraph(previousContext, 0));
}

TEST_F(IntegrationGraphVerificationContext, FutureGraphCannotReplaceCurrentComparator)
{
    Graph current;
    auto currentOutput = makePointwise(current, DataType::FLOAT);
    GraphVerificationContext currentContext(current);
    registerValidator(currentContext, currentOutput, 0.0f);

    Graph future;
    auto futureOutput = makePointwise(future, DataType::HALF);
    GraphVerificationContext futureContext(future);
    registerValidator(futureContext, futureOutput, 0.0f);
    ASSERT_NO_FATAL_FAILURE(verifyGraph(currentContext, 0));

    _differentInputs = true;
    expectVerificationFailure([&] { verifyBuiltGraph(currentContext, 0); });
    EXPECT_EQ(_initializations, 4);

    // The latest tolerance belongs to this output alone and survives another call.
    registerValidator(currentContext, currentOutput, 1.0f);
    ASSERT_NO_FATAL_FAILURE(verifyBuiltGraph(currentContext, 0));
    registerValidator(currentContext, currentOutput, 0.0f);
    expectVerificationFailure([&] { verifyBuiltGraph(currentContext, 0); });
}

#ifdef HIPDNN_ENABLE_SDPA
class IntegrationGraphVerificationOutputs : public IntegrationGraphVerificationHarness<float, int>
{
protected:
    // This regression exercises real frontend output discovery and registration resolution;
    // it does not execute an SDPA engine.
    void SetUp() override {}
    void TearDown() override {}

    void resolve(Graph& graph, GraphVerificationContext& context)
    {
        GraphTensorBundle cpu;
        GraphTensorBundle gpu;
        std::vector<OutputTensor> outputs;
        generateBundles(graph, cpu, gpu, outputs);
        resolveOutputValidators(context, outputs);
    }

    static auto makeSdpa(Graph& graph)
    {
        graph.set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);
        hipdnn_frontend::graph::SdpaAttributes attributes;
        attributes.set_generate_stats(true);
        auto outputs = graph.sdpa(makeInput(1, DataType::FLOAT),
                                  makeInput(2, DataType::FLOAT),
                                  makeInput(3, DataType::FLOAT),
                                  attributes);
        outputs[0]->set_uid(4).set_name("O").set_output(true).set_data_type(DataType::FLOAT);
        outputs[1]->set_uid(5).set_name("STATS").set_output(true).set_data_type(DataType::FLOAT);
        return outputs;
    }
};

TEST_F(IntegrationGraphVerificationOutputs, CurrentStatsRequiresItsOwnRegistration)
{
    Graph previous;
    auto previousOutputs = makeSdpa(previous);
    ASSERT_TRUE(previous.validate().is_good());
    GraphVerificationContext previousContext(previous);
    registerValidator(previousContext, previousOutputs[0], 0.0f);
    registerValidator(previousContext, previousOutputs[1], 100.0f);
    ASSERT_NO_FATAL_FAILURE(resolve(previous, previousContext));

    Graph current;
    auto currentOutputs = makeSdpa(current);
    ASSERT_TRUE(current.validate().is_good());
    GraphVerificationContext currentContext(current);
    registerValidator(currentContext, currentOutputs[0], 0.0f);
    expectVerificationFailure([&] { resolve(current, currentContext); });

    registerValidator(currentContext, currentOutputs[1], 0.0f);
    ASSERT_NO_FATAL_FAILURE(resolve(current, currentContext));
    ASSERT_NO_FATAL_FAILURE(resolve(previous, previousContext));
}
#endif

} // namespace
} // namespace hip_kernel_provider::test_utilities
