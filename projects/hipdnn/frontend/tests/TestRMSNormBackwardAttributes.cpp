// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_frontend::graph;

// --- Test suite: TestRMSNormBackwardAttributes ---

TEST(TestRMSNormBackwardAttributes, CreateRMSNormBackwardAttributes)
{
    RMSNormBackwardAttributes attrs;

    // Set all tensors
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(70)
        .set_name("dyTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    attrs.set_dy(dyTensor);
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(71)
        .set_name("xTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});
    attrs.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(72)
        .set_name("scaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});
    attrs.set_scale(scaleTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(74)
        .set_name("dxTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});
    attrs.set_dx(dxTensor);

    auto dScaleTensor = std::make_shared<TensorAttributes>();
    dScaleTensor->set_uid(75)
        .set_name("dScaleTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});
    attrs.set_dscale(dScaleTensor);

    auto invRMSTensor = std::make_shared<TensorAttributes>();
    invRMSTensor->set_uid(76)
        .set_name("invRMSTensor")
        .set_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});
    attrs.set_inv_rms(invRMSTensor);

    // Verify tensor getters
    EXPECT_NE(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_dy()->get_uid(), 70);
    EXPECT_EQ(attrs.get_dy()->get_name(), "dyTensor");
    EXPECT_EQ(attrs.get_dy()->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(attrs.get_dy()->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(attrs.get_dy()->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_NE(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_x()->get_uid(), 71);
    EXPECT_EQ(attrs.get_x()->get_name(), "xTensor");
    EXPECT_EQ(attrs.get_x()->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(attrs.get_x()->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(attrs.get_x()->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_NE(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_scale()->get_uid(), 72);
    EXPECT_EQ(attrs.get_scale()->get_name(), "scaleTensor");
    EXPECT_EQ(attrs.get_scale()->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(attrs.get_scale()->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(attrs.get_scale()->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_NE(attrs.get_dx(), nullptr);
    EXPECT_EQ(attrs.get_dx()->get_uid(), 74);
    EXPECT_EQ(attrs.get_dx()->get_name(), "dxTensor");
    EXPECT_EQ(attrs.get_dx()->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(attrs.get_dx()->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(attrs.get_dx()->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_NE(attrs.get_dscale(), nullptr);
    EXPECT_EQ(attrs.get_dscale()->get_uid(), 75);
    EXPECT_EQ(attrs.get_dscale()->get_name(), "dScaleTensor");
    EXPECT_EQ(attrs.get_dscale()->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(attrs.get_dscale()->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(attrs.get_dscale()->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_NE(attrs.get_inv_rms(), nullptr);
    EXPECT_EQ(attrs.get_inv_rms()->get_uid(), 76);
    EXPECT_EQ(attrs.get_inv_rms()->get_name(), "invRMSTensor");
    EXPECT_EQ(attrs.get_inv_rms()->get_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(attrs.get_inv_rms()->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(attrs.get_inv_rms()->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestRMSNormBackwardAttributes, DefaultValues)
{
    const RMSNormBackwardAttributes attrs;

    // Tensors should be null by default
    EXPECT_EQ(attrs.get_dy(), nullptr);
    EXPECT_EQ(attrs.get_x(), nullptr);
    EXPECT_EQ(attrs.get_scale(), nullptr);
    EXPECT_EQ(attrs.get_dx(), nullptr);
    EXPECT_EQ(attrs.get_inv_rms(), nullptr);

    // dbias computation is opt-in
    EXPECT_FALSE(attrs.get_compute_dbias());
}

TEST(TestRMSNormBackwardAttributes, SetComputeDbias)
{
    RMSNormBackwardAttributes attrs;
    EXPECT_FALSE(attrs.get_compute_dbias());

    attrs.set_compute_dbias(true);
    EXPECT_TRUE(attrs.get_compute_dbias());

    attrs.set_compute_dbias(false);
    EXPECT_FALSE(attrs.get_compute_dbias());
}

TEST(TestRMSNormBackwardAttributes, Hasbias)
{
    RMSNormBackwardAttributes attrs;
    EXPECT_FALSE(attrs.get_compute_dbias());

    attrs.has_dbias(true);
    EXPECT_TRUE(attrs.get_compute_dbias());

    attrs.has_dbias(false);
    EXPECT_FALSE(attrs.get_compute_dbias());
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
