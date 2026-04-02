// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/data_objects/rmsnorm_backward_attributes_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_frontend::graph;

// --- Test suite: TestRMSNormBackwardAttributes ---

TEST(TestRMSNormBackwardAttributes, CreateRMSNormBackwardAttributes)
{
    RMSNormBackwardAttributes attrs;

    // Set all tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70);
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71);
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72);
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74);
    attrs.set_dx(dxTensor);

    // Set data fields

    // Verify tensor getters
    EXPECT_NE(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_dy()->get_uid(), 70);
    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 71);
    EXPECT_NE(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_scale()->get_uid(), 72);
    EXPECT_NE(attrs.get_dx(), nullptr);
    EXPECT_EQ(attrs.get_dx()->get_uid(), 74);

    // Verify data field getters
}

TEST(TestRMSNormBackwardAttributes, DefaultValues)
{
    const RMSNormBackwardAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_dx(), nullptr);

    // Vector fields should be empty by default
}

TEST(TestRMSNormBackwardAttributes, PackAttributes)
{
    RMSNormBackwardAttributes attrs;

    // Set tensors with UIDs
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70);
    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71);
    attrs.set_x(xTensor);
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72);
    attrs.set_scale(scaleTensor);
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74);
    attrs.set_dx(dxTensor);
    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_uid(75);
    attrs.set_dscale(dscaleTensor);

    // Set data fields

    // Pack attributes
    flatbuffers::FlatBufferBuilder builder;
    auto packedAttributes = attrs.pack_attributes(builder);
    builder.Finish(packedAttributes);

    auto buffer = builder.GetBufferPointer();
    auto attrsFb
        = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::RMSNormBackwardAttributes>(buffer);

    // Verify packed tensor UIDs
    EXPECT_EQ(attrsFb->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(attrsFb->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(attrsFb->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(attrsFb->dx_tensor_uid(), dxTensor->get_uid());
    EXPECT_EQ(attrsFb->dscale_tensor_uid(), dscaleTensor->get_uid());

    // Verify packed data fields
}

TEST(TestRMSNormBackwardAttributes, SetDyMove)
{
    RMSNormBackwardAttributes attrs;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70).set_name("MovedDyTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dyTensor.get();

    attrs.set_dy(std::move(dyTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dyTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dy();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestRMSNormBackwardAttributes, SetXMove)
{
    RMSNormBackwardAttributes attrs;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71).set_name("MovedXTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = xTensor.get();

    attrs.set_x(std::move(xTensor));

    // After move, original should be nullptr
    EXPECT_EQ(xTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_x();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestRMSNormBackwardAttributes, SetScaleMove)
{
    RMSNormBackwardAttributes attrs;

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72)
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

TEST(TestRMSNormBackwardAttributes, SetDxMove)
{
    RMSNormBackwardAttributes attrs;

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74).set_name("MovedDxTensor").set_data_type(hipdnn_frontend::DataType::FLOAT);

    // Store the raw pointer before moving
    auto rawPtr = dxTensor.get();

    attrs.set_dx(std::move(dxTensor));

    // After move, original should be nullptr
    EXPECT_EQ(dxTensor, nullptr);

    // The moved tensor should be accessible through the getter
    auto retrievedTensor = attrs.get_dx();
    EXPECT_EQ(retrievedTensor.get(), rawPtr);
}

TEST(TestRMSNormBackwardAttributes, SetTensorsConstRef)
{
    RMSNormBackwardAttributes attrs;

    // Create tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70).set_name("DyConstRef");
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71).set_name("XConstRef");
    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72).set_name("ScaleConstRef");
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74).set_name("DxConstRef");

    // Set using const reference (copy)
    attrs.set_dy(dyTensor);
    attrs.set_x(xTensor);
    attrs.set_scale(scaleTensor);
    attrs.set_dx(dxTensor);

    // Original tensors should still be valid
    EXPECT_NE(dyTensor, nullptr);
    EXPECT_NE(xTensor, nullptr);
    EXPECT_NE(scaleTensor, nullptr);
    EXPECT_NE(dxTensor, nullptr);
}
