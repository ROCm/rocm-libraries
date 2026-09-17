// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <memory>

using namespace hipdnn_frontend::graph;

// --- Test suite: TestLayernormBackwardAttributes ---

TEST(TestLayernormBackwardAttributes, CreateLayernormBackwardAttributes)
{
    LayernormBackwardAttributes attrs;

    // Set all tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(10);
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(11);
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(12);
    attrs.set_scale(scaleTensor);
    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(13);
    attrs.set_mean(meanTensor);
    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(14);
    attrs.set_inv_variance(invVarianceTensor);
    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(15);
    attrs.set_epsilon(epsilonTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(16);
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(17);
    attrs.set_dscale(dscaleTensor);
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(18);
    attrs.set_dbias(dbiasTensor);

    // Set data fields
    attrs.set_normalized_dim_count(2);

    // Verify tensor getters
    EXPECT_NE(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_dy()->get_uid(), 10);
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 11);
    EXPECT_NE(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_scale()->get_uid(), 12);
    EXPECT_NE(attrs.get_mean(), nullptr);
    EXPECT_EQ(attrs.get_mean()->get_uid(), 13);
    EXPECT_NE(attrs.get_inv_variance(), nullptr);
    EXPECT_EQ(attrs.get_inv_variance()->get_uid(), 14);
    EXPECT_NE(attrs.get_epsilon(), nullptr);
    EXPECT_EQ(attrs.get_epsilon()->get_uid(), 15);
    EXPECT_NE(attrs.get_dx(), nullptr);
    EXPECT_EQ(attrs.get_dx()->get_uid(), 16);
    EXPECT_NE(attrs.get_dscale(), nullptr);
    EXPECT_EQ(attrs.get_dscale()->get_uid(), 17);
    EXPECT_NE(attrs.get_dbias(), nullptr);
    EXPECT_EQ(attrs.get_dbias()->get_uid(), 18);

    // Verify data field getters
    EXPECT_EQ(attrs.get_normalized_dim_count(), 2);
}

TEST(TestLayernormBackwardAttributes, DefaultValues)
{
    const LayernormBackwardAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_mean(), nullptr);
    EXPECT_EQ(attrs.get_inv_variance(), nullptr);
    EXPECT_EQ(attrs.get_epsilon(), nullptr);
    EXPECT_EQ(attrs.get_dx(), nullptr);
    EXPECT_EQ(attrs.get_dscale(), nullptr);
    EXPECT_EQ(attrs.get_dbias(), nullptr);
}

TEST(TestLayernormBackwardAttributes, SetDyMove)
{
    LayernormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(10).set_name("MovedDyTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dyTensor.get();

    attrs.set_dy(std::move(dyTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dyTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_dy();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedDyTensor");
}

TEST(TestLayernormBackwardAttributes, SetXMove)
{
    LayernormBackwardAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(11).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedXTensor");
}

TEST(TestLayernormBackwardAttributes, SetScaleMove)
{
    LayernormBackwardAttributes attrs;

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(12)
        .set_name("MovedScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = scaleTensor.get();

    attrs.set_scale(std::move(scaleTensor));

    // After move, original should be nullptr
    EXPECT_EQ(scaleTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_scale();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedScaleTensor");
}

TEST(TestLayernormBackwardAttributes, SetMeanMove)
{
    LayernormBackwardAttributes attrs;

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(13)
        .set_name("MovedMeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = meanTensor.get();

    attrs.set_mean(std::move(meanTensor));

    // After move, original should be nullptr
    EXPECT_EQ(meanTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_mean();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedMeanTensor");
}

TEST(TestLayernormBackwardAttributes, SetInvVarianceMove)
{
    LayernormBackwardAttributes attrs;

    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(14)
        .set_name("MovedInvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = invVarianceTensor.get();

    attrs.set_inv_variance(std::move(invVarianceTensor));

    // After move, original should be nullptr
    EXPECT_EQ(invVarianceTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_inv_variance();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedInvVarianceTensor");
}

TEST(TestLayernormBackwardAttributes, SetMeanAndInvVarianceMove)
{
    LayernormBackwardAttributes attrs;

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(13)
        .set_name("MovedMeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(14)
        .set_name("MovedInvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawMeanPtr = meanTensor.get();
    auto rawInvVariancePtr = invVarianceTensor.get();

    attrs.set_saved_mean_and_inv_variance(std::move(meanTensor), std::move(invVarianceTensor));

    // After move, original should be nullptr
    EXPECT_EQ(meanTensor, nullptr);
    EXPECT_EQ(invVarianceTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedMeanTensor = attrs.get_mean();
    EXPECT_EQ(retrievedMeanTensor.get(), rawMeanPtr);
    EXPECT_EQ(retrievedMeanTensor->get_name(), "MovedMeanTensor");
    auto retrievedInvVarianceTensor = attrs.get_inv_variance();
    EXPECT_EQ(retrievedInvVarianceTensor.get(), rawInvVariancePtr);
    EXPECT_EQ(retrievedInvVarianceTensor->get_name(), "MovedInvVarianceTensor");
}

TEST(TestLayernormBackwardAttributes, SetMeanAndInvVarianceCopy)
{
    LayernormBackwardAttributes attrs;

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(13)
        .set_name("CopiedMeanTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);
    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(14)
        .set_name("CopiedInvVarianceTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawMeanPtr = meanTensor.get();
    auto rawInvVariancePtr = invVarianceTensor.get();

    attrs.set_saved_mean_and_inv_variance(meanTensor, invVarianceTensor);

    // After copy, original should still exist
    EXPECT_NE(meanTensor, nullptr);
    EXPECT_NE(invVarianceTensor, nullptr);

    // The copied tensor should be accessible through the getter and should have the correct name
    auto retrievedMeanTensor = attrs.get_mean();
    EXPECT_EQ(retrievedMeanTensor.get(), rawMeanPtr);
    EXPECT_EQ(retrievedMeanTensor->get_name(), "CopiedMeanTensor");
    auto retrievedInvVarianceTensor = attrs.get_inv_variance();
    EXPECT_EQ(retrievedInvVarianceTensor.get(), rawInvVariancePtr);
    EXPECT_EQ(retrievedInvVarianceTensor->get_name(), "CopiedInvVarianceTensor");
}

TEST(TestLayernormBackwardAttributes, SetEpsilonMove)
{
    LayernormBackwardAttributes attrs;

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(15)
        .set_name("MovedEpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = epsilonTensor.get();

    attrs.set_epsilon(std::move(epsilonTensor));

    // After move, original should be nullptr
    EXPECT_EQ(epsilonTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_epsilon();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedEpsilonTensor");
}

TEST(TestLayernormBackwardAttributes, SetDxMove)
{
    LayernormBackwardAttributes attrs;

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(16).set_name("MovedDxTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dxTensor.get();

    attrs.set_dx(std::move(dxTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dxTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_dx();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedDxTensor");
}

TEST(TestLayernormBackwardAttributes, SetDscaleMove)
{
    LayernormBackwardAttributes attrs;

    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(17)
        .set_name("MovedDscaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dscaleTensor.get();

    attrs.set_dscale(std::move(dscaleTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dscaleTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_dscale();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedDscaleTensor");
}

TEST(TestLayernormBackwardAttributes, SetDbiasMove)
{
    LayernormBackwardAttributes attrs;

    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(18)
        .set_name("MovedDbiasTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dbiasTensor.get();

    attrs.set_dbias(std::move(dbiasTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dbiasTensor, nullptr);

    // The moved tensor should be accessible through the getter and should have the correct name
    auto retrievedTensor = attrs.get_dbias();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
    EXPECT_EQ(retrievedTensor->get_name(), "MovedDbiasTensor");
}

TEST(TestLayernormBackwardAttributes, SetTensorsConstRef)
{
    LayernormBackwardAttributes attrs;

    // Create tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(10).set_name("DyConstRef");
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(11).set_name("XConstRef");
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(12).set_name("ScaleConstRef");
    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(13).set_name("MeanConstRef");
    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(14).set_name("InvVarianceConstRef");
    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(15).set_name("EpsilonConstRef");
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(16).set_name("DxConstRef");
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(17).set_name("DscaleConstRef");
    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_uid(18).set_name("DbiasConstRef");

    // Set using const reference (copy)
    attrs.set_dy(dyTensor);
    attrs.set_x(xTensor);
    attrs.set_scale(scaleTensor);
    attrs.set_mean(meanTensor);
    attrs.set_inv_variance(invVarianceTensor);
    attrs.set_epsilon(epsilonTensor);
    attrs.set_dx(dxTensor);
    attrs.set_dscale(dscaleTensor);
    attrs.set_dbias(dbiasTensor);

    // Original tensors should still be valid
    EXPECT_NE(dyTensor, nullptr);
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(scaleTensor, nullptr);
    EXPECT_NE(meanTensor, nullptr);
    EXPECT_NE(invVarianceTensor, nullptr);
    EXPECT_NE(epsilonTensor, nullptr);
    EXPECT_NE(dxTensor, nullptr);
    EXPECT_NE(dscaleTensor, nullptr);
    EXPECT_NE(dbiasTensor, nullptr);
}

TEST(TestLayernormBackwardAttributes, LogicalAndStrictEquality)
{
    hipdnn_frontend::graph::LayernormBackwardAttributes attr1;
    attr1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_normalized_dim_count(2);

    auto dy1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dy1->set_uid(1).set_name("DY").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dy(dy1);

    auto x1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x1->set_uid(2).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_x(x1);

    auto scale1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scale1->set_uid(3).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_scale(scale1);

    auto mean1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    mean1->set_uid(4).set_name("Mean").set_data_type(hipdnn_frontend::DataType::FLOAT);
    auto invVar1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVar1->set_uid(5).set_name("InvVar").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_saved_mean_and_inv_variance(mean1, invVar1);

    auto epsilon1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon1->set_uid(6).set_name("Epsilon").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_epsilon(epsilon1);

    auto dx1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dx1->set_uid(7).set_name("DX").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dx(dx1);

    auto dscale1 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dscale1->set_uid(8).set_name("DScale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr1.set_dscale(dscale1);

    hipdnn_frontend::graph::LayernormBackwardAttributes attr2;
    attr2.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_normalized_dim_count(2);

    auto dy2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dy2->set_uid(1).set_name("DY").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dy(dy2);

    auto x2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    x2->set_uid(2).set_name("X").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_x(x2);

    auto scale2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    scale2->set_uid(3).set_name("Scale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_scale(scale2);

    auto mean2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    mean2->set_uid(4).set_name("Mean").set_data_type(hipdnn_frontend::DataType::FLOAT);
    auto invVar2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    invVar2->set_uid(5).set_name("InvVar").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_saved_mean_and_inv_variance(mean2, invVar2);

    auto epsilon2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilon2->set_uid(6).set_name("Epsilon").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_epsilon(epsilon2);

    auto dx2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dx2->set_uid(7).set_name("DX").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dx(dx2);

    auto dscale2 = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    dscale2->set_uid(8).set_name("DScale").set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr2.set_dscale(dscale2);

    // Initial check: everything matches exactly
    EXPECT_TRUE(attr1 == attr2);
    EXPECT_FALSE(attr1 != attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));

    // Structural tensor mismatch: different UID/name/type entirely
    auto structuralMismatchDy = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    structuralMismatchDy->set_uid(99).set_name("MismatchedDY");
    attr2.set_dy(structuralMismatchDy);

    EXPECT_TRUE(attr1 != attr2);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2)); // Structural/type gap implies logical inequality
    attr2.set_dy(dy2); // Revert

    // Normalized-dim-count mismatch: semantic, must fail both checks
    attr2.set_normalized_dim_count(1);
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_FALSE(attr1.logicallyEquals(attr2));
    attr2.set_normalized_dim_count(2); // Revert

    // Change metadata (UID/Name) on a tensor while keeping mathematical layout intact
    auto logicalMatchDy = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    logicalMatchDy
        ->set_uid(555) // Diverges from attr1's dy1 (uid: 1)
        .set_name("DIVERGENT_NAME") // Diverges from attr1's dy1 ("DY")
        .set_data_type(hipdnn_frontend::DataType::FLOAT); // Layout matches
    attr2.set_dy(logicalMatchDy);

    // Expecting: strict evaluation fails, but functional logical comparison passes
    EXPECT_FALSE(attr1 == attr2);
    EXPECT_TRUE(attr1.logicallyEquals(attr2));
}
