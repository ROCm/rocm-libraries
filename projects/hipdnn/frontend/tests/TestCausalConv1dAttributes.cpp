// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/CausalConv1dAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// --- Test suite: TestCausalConv1dAttributes ---

TEST(TestCausalConv1dAttributes, CreateCausalConv1dAttributes)
{
    CausalConv1dAttributes attrs;

    // Set all tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9600);
    attrs.set_x(xTensor);
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(9601);
    attrs.set_weight(weightTensor);
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(9602);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(9604);
    attrs.set_bias(biasTensor);
    auto initialStateTensor = std::make_shared<TensorAttributes>();
    initialStateTensor->set_uid(9605);
    attrs.set_initial_state(initialStateTensor);
    auto stateIndicesTensor = std::make_shared<TensorAttributes>();
    stateIndicesTensor->set_uid(9606);
    attrs.set_state_indices(stateIndicesTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9603);
    attrs.set_y(yTensor);
    auto finalStateTensor = std::make_shared<TensorAttributes>();
    finalStateTensor->set_uid(9607);
    attrs.set_final_state(finalStateTensor);

    // Set data fields
    attrs.set_activation(CausalConv1dActivation::SILU);

    // Verify tensor getters
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 9600);
    EXPECT_NE(attrs.get_weight(), nullptr);
    EXPECT_EQ(attrs.get_weight()->get_uid(), 9601);
    EXPECT_NE(attrs.get_cu_seqlens(), nullptr);
    EXPECT_EQ(attrs.get_cu_seqlens()->get_uid(), 9602);
    EXPECT_NE(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_bias()->get_uid(), 9604);
    EXPECT_NE(attrs.get_initial_state(), nullptr);
    EXPECT_EQ(attrs.get_initial_state()->get_uid(), 9605);
    EXPECT_NE(attrs.get_state_indices(), nullptr);
    EXPECT_EQ(attrs.get_state_indices()->get_uid(), 9606);
    EXPECT_NE(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_y()->get_uid(), 9603);
    EXPECT_NE(attrs.get_final_state(), nullptr);
    EXPECT_EQ(attrs.get_final_state()->get_uid(), 9607);

    // Verify data field getters
}

TEST(TestCausalConv1dAttributes, DefaultValues)
{
    const CausalConv1dAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_weight(), nullptr);
    EXPECT_EQ(attrs.get_cu_seqlens(), nullptr);
    EXPECT_EQ(attrs.get_bias(), nullptr);
    EXPECT_EQ(attrs.get_initial_state(), nullptr);
    EXPECT_EQ(attrs.get_state_indices(), nullptr);
    EXPECT_EQ(attrs.get_y(), nullptr);
    EXPECT_EQ(attrs.get_final_state(), nullptr);

    EXPECT_EQ(attrs.get_activation(), CausalConv1dActivation::IDENTITY);
    EXPECT_FALSE(attrs.get_output_final_state());
}

TEST(TestCausalConv1dAttributes, SetOutputFinalState)
{
    CausalConv1dAttributes attrs;
    attrs.set_output_final_state(true).set_activation(CausalConv1dActivation::SILU);
    EXPECT_TRUE(attrs.get_output_final_state());
    EXPECT_EQ(attrs.get_activation(), CausalConv1dActivation::SILU);
}

TEST(TestCausalConv1dAttributes, SetXMove)
{
    CausalConv1dAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9600).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetWeightMove)
{
    CausalConv1dAttributes attrs;

    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(9601)
        .set_name("MovedWeightTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = weightTensor.get();

    attrs.set_weight(std::move(weightTensor));

    // After move, original should be nullptr
    EXPECT_EQ(weightTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_weight();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetCuSeqlensMove)
{
    CausalConv1dAttributes attrs;

    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(9602)
        .set_name("MovedCuSeqlensTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = cuSeqlensTensor.get();

    attrs.set_cu_seqlens(std::move(cuSeqlensTensor));

    // After move, original should be nullptr
    EXPECT_EQ(cuSeqlensTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_cu_seqlens();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetBiasMove)
{
    CausalConv1dAttributes attrs;

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(9604)
        .set_name("MovedBiasTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = biasTensor.get();

    attrs.set_bias(std::move(biasTensor));

    // After move, original should be nullptr
    EXPECT_EQ(biasTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_bias();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetInitialStateMove)
{
    CausalConv1dAttributes attrs;

    auto initialStateTensor = std::make_shared<TensorAttributes>();
    initialStateTensor->set_uid(9605)
        .set_name("MovedInitialStateTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = initialStateTensor.get();

    attrs.set_initial_state(std::move(initialStateTensor));

    // After move, original should be nullptr
    EXPECT_EQ(initialStateTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_initial_state();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetStateIndicesMove)
{
    CausalConv1dAttributes attrs;

    auto stateIndicesTensor = std::make_shared<TensorAttributes>();
    stateIndicesTensor->set_uid(9606)
        .set_name("MovedStateIndicesTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = stateIndicesTensor.get();

    attrs.set_state_indices(std::move(stateIndicesTensor));

    // After move, original should be nullptr
    EXPECT_EQ(stateIndicesTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_state_indices();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetYMove)
{
    CausalConv1dAttributes attrs;

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9603).set_name("MovedYTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = yTensor.get();

    attrs.set_y(std::move(yTensor));

    // After move, original should be nullptr
    EXPECT_EQ(yTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_y();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetFinalStateMove)
{
    CausalConv1dAttributes attrs;

    auto finalStateTensor = std::make_shared<TensorAttributes>();
    finalStateTensor->set_uid(9607)
        .set_name("MovedFinalStateTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = finalStateTensor.get();

    attrs.set_final_state(std::move(finalStateTensor));

    // After move, original should be nullptr
    EXPECT_EQ(finalStateTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_final_state();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestCausalConv1dAttributes, SetTensorsConstRef)
{
    CausalConv1dAttributes attrs;

    // Create tensors
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(9600).set_name("XConstRef");
    auto weightTensor = std::make_shared<TensorAttributes>();
    weightTensor->set_uid(9601).set_name("WeightConstRef");
    auto cuSeqlensTensor = std::make_shared<TensorAttributes>();
    cuSeqlensTensor->set_uid(9602).set_name("CuSeqlensConstRef");
    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(9604).set_name("BiasConstRef");
    auto initialStateTensor = std::make_shared<TensorAttributes>();
    initialStateTensor->set_uid(9605).set_name("InitialStateConstRef");
    auto stateIndicesTensor = std::make_shared<TensorAttributes>();
    stateIndicesTensor->set_uid(9606).set_name("StateIndicesConstRef");
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(9603).set_name("YConstRef");
    auto finalStateTensor = std::make_shared<TensorAttributes>();
    finalStateTensor->set_uid(9607).set_name("FinalStateConstRef");

    // Set using const reference (copy)
    attrs.set_x(xTensor);
    attrs.set_weight(weightTensor);
    attrs.set_cu_seqlens(cuSeqlensTensor);
    attrs.set_bias(biasTensor);
    attrs.set_initial_state(initialStateTensor);
    attrs.set_state_indices(stateIndicesTensor);
    attrs.set_y(yTensor);
    attrs.set_final_state(finalStateTensor);

    // Original tensors should still be valid
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(weightTensor, nullptr);
    EXPECT_NE(cuSeqlensTensor, nullptr);
    EXPECT_NE(biasTensor, nullptr);
    EXPECT_NE(initialStateTensor, nullptr);
    EXPECT_NE(stateIndicesTensor, nullptr);
    EXPECT_NE(yTensor, nullptr);
    EXPECT_NE(finalStateTensor, nullptr);
}

TEST(TestCausalConv1dAttributes, LogicalAndStrictEquality)
{
    CausalConv1dAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_activation(CausalConv1dActivation::SILU);
    auto x1 = std::make_shared<TensorAttributes>();
    x1->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);
    auto weight1 = std::make_shared<TensorAttributes>();
    weight1->set_uid(2).set_name("Weight").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_weight(weight1);
    auto cuSeqlens1 = std::make_shared<TensorAttributes>();
    cuSeqlens1->set_uid(3).set_name("CuSeqlens").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_cu_seqlens(cuSeqlens1);
    auto bias1 = std::make_shared<TensorAttributes>();
    bias1->set_uid(4).set_name("Bias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_bias(bias1);
    auto initialState1 = std::make_shared<TensorAttributes>();
    initialState1->set_uid(5)
        .set_name("InitialState")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_initial_state(initialState1);
    auto stateIndices1 = std::make_shared<TensorAttributes>();
    stateIndices1->set_uid(6)
        .set_name("StateIndices")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_state_indices(stateIndices1);
    auto y1 = std::make_shared<TensorAttributes>();
    y1->set_uid(7).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_y(y1);
    auto finalState1 = std::make_shared<TensorAttributes>();
    finalState1->set_uid(8).set_name("FinalState").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_final_state(finalState1);

    CausalConv1dAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_activation(CausalConv1dActivation::SILU);
    auto x2 = std::make_shared<TensorAttributes>();
    x2->set_uid(1).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);
    auto weight2 = std::make_shared<TensorAttributes>();
    weight2->set_uid(2).set_name("Weight").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_weight(weight2);
    auto cuSeqlens2 = std::make_shared<TensorAttributes>();
    cuSeqlens2->set_uid(3).set_name("CuSeqlens").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_cu_seqlens(cuSeqlens2);
    auto bias2 = std::make_shared<TensorAttributes>();
    bias2->set_uid(4).set_name("Bias").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_bias(bias2);
    auto initialState2 = std::make_shared<TensorAttributes>();
    initialState2->set_uid(5)
        .set_name("InitialState")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_initial_state(initialState2);
    auto stateIndices2 = std::make_shared<TensorAttributes>();
    stateIndices2->set_uid(6)
        .set_name("StateIndices")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_state_indices(stateIndices2);
    auto y2 = std::make_shared<TensorAttributes>();
    y2->set_uid(7).set_name("Y").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_y(y2);
    auto finalState2 = std::make_shared<TensorAttributes>();
    finalState2->set_uid(8).set_name("FinalState").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_final_state(finalState2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchX = std::make_shared<TensorAttributes>();
    structuralMismatchX->set_uid(9999).set_name("MismatchedX");
    attr2.set_x(structuralMismatchX);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_x(x2); // Revert

    // activation mismatch: semantic, must fail both checks
    attr2.set_activation(CausalConv1dActivation::IDENTITY);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_activation(CausalConv1dActivation::SILU); // Revert

    // output_final_state mismatch: semantic, must fail both checks
    attr2.set_output_final_state(true);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_output_final_state(false); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchX = std::make_shared<TensorAttributes>();
    logicalMatchX
        ->set_uid(555) // Diverges from attr1's x1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's x1 ("X")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_x(logicalMatchX);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
