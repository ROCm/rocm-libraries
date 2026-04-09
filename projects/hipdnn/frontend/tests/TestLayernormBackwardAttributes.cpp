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

    // Vector fields should be empty by default
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dy();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_scale();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_mean();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_inv_variance();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_epsilon();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dx();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dscale();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dbias();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
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
