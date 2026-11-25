// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

// Runs output inference for x = {1, 3, 8, 8} with the given window, stride and padding;
// yDims receives the y dimensions afterwards.
Error inferYFromX8x8(const std::vector<int64_t>& window,
                     const std::vector<int64_t>& stride,
                     std::vector<int64_t>& yDims,
                     const std::vector<int64_t>& prePadding = {0, 0},
                     const std::vector<int64_t>& postPadding = {0, 0})
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    attr.set_pre_padding(prePadding);
    attr.set_post_padding(postPadding);
    attr.set_window(window);
    attr.set_stride(stride);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
    yDims = yTensor->get_dim();
    return error;
}

// x = {1, 3, 8, 8} max-pooled to y = {1, 3, 4, 4} by a 2x2 window with stride 2 and zero
// padding mode.
ResampleFwdAttributes makeX8x8Attributes()
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 4, 4});
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_window(std::vector<int64_t>{2, 2});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);
    return attr;
}

Error preValidate(ResampleFwdAttributes attr)
{
    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);
    return node.pre_validate_node();
}

// Pre-validates makeX8x8Attributes() with the given resample parameters.
Error preValidateX8x8(const std::vector<int64_t>& window,
                      const std::vector<int64_t>& stride,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding)
{
    auto attr = makeX8x8Attributes();
    attr.set_pre_padding(prePadding);
    attr.set_post_padding(postPadding);
    attr.set_window(window);
    attr.set_stride(stride);
    return preValidate(std::move(attr));
}

// Mode values probed by the lowering drift guards; values past the last enumerator are included
// so that an enumerator added later is covered too.
constexpr int K_MODE_PROBE_END = 16;

} // namespace

TEST(TestResampleFwdNode, ResampleFwdProperties)
{
    ResampleFwdAttributes attr;
    attr.set_x(std::make_shared<TensorAttributes>());
    attr.set_y(std::make_shared<TensorAttributes>());
    attr.set_index(std::make_shared<TensorAttributes>());

    const auto inputTensor = attr.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    const auto outputTensor = attr.get_y();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    const auto indexTensor = attr.get_index();
    indexTensor->set_uid(3).set_name("IndexTensor");

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    const auto error = node.infer_properties_node();

    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));

    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
    EXPECT_EQ(indexTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestResampleFwdNode, PreValidateNode)
{
    ResampleFwdAttributes attr;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_y(yTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_data_type(hipdnn_frontend::DataType::INT32);
    attr.set_index(indexTensor);
    attr.set_generate_index(true);
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::NEG_INF_PAD);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestResampleFwdNode, PreValidateNodeMissingValues)
{
    ResampleFwdAttributes attr;
    attr.set_resample_mode(ResampleMode::AVGPOOL_INCLUDE_PADDING);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);

    const GraphAttributes graphAttributes;
    auto resampleFwdAttributesCopy = attr;
    const ResampleFwdNode node(std::move(resampleFwdAttributesCopy), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    attr.set_x(std::make_shared<TensorAttributes>());
    resampleFwdAttributesCopy = attr;
    const ResampleFwdNode nodeWithX(std::move(resampleFwdAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    attr.set_y(std::make_shared<TensorAttributes>());
    resampleFwdAttributesCopy = attr;
    const ResampleFwdNode nodeWithY(std::move(resampleFwdAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestResampleFwdNode, MissingIndexTensor)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_y(yTensor);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);

    const GraphAttributes graphAttributes;

    // Ok to not have index tensor when generate index set to false
    {
        ResampleFwdAttributes attrCopy = attr;
        attrCopy.set_generate_index(false);
        attrCopy.set_resample_mode(ResampleMode::MAXPOOL);
        const ResampleFwdNode node(std::move(attrCopy), graphAttributes);

        const auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::OK);
    }

    // Ok to not have index tensor when using a resample mode other than MAXPOOL
    {
        ResampleFwdAttributes attrCopy = attr;
        attrCopy.set_resample_mode(ResampleMode::AVGPOOL_EXCLUDE_PADDING);
        const ResampleFwdNode node(std::move(attrCopy), graphAttributes);

        const auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::OK);
    }

    // Not ok to not have index tensor when MAXPOOL index generation is requested
    {
        ResampleFwdAttributes attrCopy = attr;
        attrCopy.set_generate_index(true);
        attrCopy.set_resample_mode(ResampleMode::MAXPOOL);
        const ResampleFwdNode node(std::move(attrCopy), graphAttributes);

        const auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    }
}

TEST(TestResampleFwdNode, InvalidIndexType)
{
    ResampleFwdAttributes attr;
    attr.set_x(std::make_shared<TensorAttributes>());
    attr.set_y(std::make_shared<TensorAttributes>());
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);
    const auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InvalidXDim)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32, 32});
    attr.set_y(yTensor);
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InvalidYDim)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32, 32, 32});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64});
    attr.set_y(yTensor);
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InferYDim3D)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    attr.set_pre_padding(std::vector<int64_t>{1});
    attr.set_post_padding(std::vector<int64_t>{1});
    attr.set_stride(std::vector<int64_t>{2});
    attr.set_window(std::vector<int64_t>{3});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 3, 16}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 3, 16}));
}

TEST(TestResampleFwdNode, InferDim4D)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    attr.set_pre_padding(std::vector<int64_t>{1, 1});
    attr.set_post_padding(std::vector<int64_t>{1, 1});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{3, 3});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 3, 16, 16}));
}

TEST(TestResampleFwdNode, InferDim5D)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 16, 16, 16});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    attr.set_pre_padding(std::vector<int64_t>{2, 2, 2});
    attr.set_post_padding(std::vector<int64_t>{2, 2, 2});
    attr.set_stride(std::vector<int64_t>{3, 3, 3});
    attr.set_window(std::vector<int64_t>{1, 1, 1});

    const GraphAttributes graphAttributes;

    ResampleFwdNode node(std::move(attr), graphAttributes);
    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 3, 7, 7, 7}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 3, 7, 7, 7}));
}

TEST(TestResampleFwdNode, InferRejectsZeroStride)
{
    std::vector<int64_t> yDims;
    const auto error = inferYFromX8x8({2, 2}, {0, 0}, yDims);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferRejectsEmptyWindow)
{
    std::vector<int64_t> yDims;
    const auto error = inferYFromX8x8({}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferRejectsWindowShorterThanSpatialRank)
{
    std::vector<int64_t> yDims;
    const auto error = inferYFromX8x8({2}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferTruncatesOutputExtentAsCudnn)
{
    // (8 - 9) / 2 + 1 = 1: a window wider than the padded input by less than the stride
    std::vector<int64_t> yDims;
    const auto error = inferYFromX8x8({9, 9}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(yDims, (std::vector<int64_t>{1, 3, 1, 1}));
}

TEST(TestResampleFwdNode, InferRejectsOutputExtentBelowOne)
{
    // (8 - 11) / 2 + 1 = 0
    std::vector<int64_t> yDims;
    const auto error = inferYFromX8x8({11, 11}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, PreValidateRejectsWindowShorterThanSpatialRankWithYSet)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 4, 4});
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{2});
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, PreValidateAcceptsWindowWiderThanPaddedInputWithYSet)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 1, 1});
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{9, 9});
    attr.set_resample_mode(ResampleMode::MAXPOOL);
    attr.set_padding_mode(PaddingMode::ZERO_PAD);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
}

TEST(TestResampleFwdNode, PreValidateRejectsZeroWindowValue)
{
    auto error = preValidateX8x8({1, 2}, {2, 2}, {0, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    error = preValidateX8x8({0, 2}, {2, 2}, {0, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, PreValidateRejectsNegativePadding)
{
    auto error = preValidateX8x8({2, 2}, {2, 2}, {0, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;

    error = preValidateX8x8({2, 2}, {2, 2}, {-1, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);

    error = preValidateX8x8({2, 2}, {2, 2}, {0, 0}, {0, -1});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, PreValidateRejectsPaddingCountMismatch)
{
    auto error = preValidateX8x8({2, 2}, {2, 2}, {0, 0, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);

    error = preValidateX8x8({2, 2}, {2, 2}, {0, 0}, {0, 0, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, PreValidateRejectsStrideCountMismatch)
{
    const auto error = preValidateX8x8({2, 2}, {2, 2, 2}, {0, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, PreValidateRejectsUnsetPaddingMode)
{
    auto attr = makeX8x8Attributes();
    attr.set_padding_mode(PaddingMode::NOT_SET);

    const auto error = preValidate(std::move(attr));
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET) << error.err_msg;
}

TEST(TestResampleFwdNode, PreValidateAcceptsExactlyTheResampleModesLoweringMaps)
{
    for(int raw = 0; raw < K_MODE_PROBE_END; ++raw)
    {
        const auto mode = static_cast<ResampleMode>(raw);
        auto attr = makeX8x8Attributes();
        attr.set_resample_mode(mode);

        const auto error = preValidate(std::move(attr));
        const auto expected
            = toBackendResampleMode(mode).has_value() ? ErrorCode::OK : ErrorCode::INVALID_VALUE;
        EXPECT_EQ(error.code, expected) << "resample mode " << raw << ": " << error.err_msg;
    }
}

TEST(TestResampleFwdNode, PreValidateAcceptsExactlyThePaddingModesLoweringMaps)
{
    for(int raw = 0; raw < K_MODE_PROBE_END; ++raw)
    {
        const auto mode = static_cast<PaddingMode>(raw);
        auto attr = makeX8x8Attributes();
        attr.set_padding_mode(mode);

        const auto error = preValidate(std::move(attr));
        // NOT_SET has a backend value but is rejected as missing: padding_mode has no default
        auto expected
            = toBackendPaddingMode(mode).has_value() ? ErrorCode::OK : ErrorCode::INVALID_VALUE;
        if(mode == PaddingMode::NOT_SET)
        {
            expected = ErrorCode::ATTRIBUTE_NOT_SET;
        }
        EXPECT_EQ(error.code, expected) << "padding mode " << raw << ": " << error.err_msg;
    }
}

TEST(TestResampleFwdNode, PreValidateAcceptsGenerateIndexForAvgpool)
{
    for(const auto mode :
        {ResampleMode::AVGPOOL_EXCLUDE_PADDING, ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        // generate_index is ignored for average pooling, so no index tensor is needed
        auto attr = makeX8x8Attributes();
        attr.set_resample_mode(mode).set_generate_index(true);

        const auto error = preValidate(std::move(attr));
        EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    }
}

TEST(TestResampleFwdNode, InferRejectsPaddingOverflow)
{
    // 8 + pre_padding + post_padding exceeds int64; wrapped, it gives a y extent of 5
    constexpr int64_t K_MAX = std::numeric_limits<int64_t>::max();
    std::vector<int64_t> yDims;
    const auto error = inferYFromX8x8({2, 2}, {1, 1}, yDims, {K_MAX, 0}, {K_MAX, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.err_msg.find("overflows int64"), std::string::npos) << error.err_msg;
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferRejectsXWithoutDims)
{
    ResampleFwdAttributes attr;
    attr.set_x(std::make_shared<TensorAttributes>());
    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{2, 2});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(yTensor->get_dim().empty());
}

TEST(TestResampleFwdNode, InferRejectsXRankBelowThree)
{
    // No resample parameters are set: the rank check must reject x before they are read
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32});
    attr.set_x(xTensor);
    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yTensor->get_dim().empty());
}

TEST(TestResampleFwdNode, InferRejectsEmptyPrePadding)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);
    const auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{2, 2});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(yTensor->get_dim().empty());
}

TEST(TestResampleFwdNode, InferStride)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 3, 7, 5});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 3, 3, 2});
    attr.set_y(yTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Stride should default to channel last
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{18, 1, 6, 3}));
    EXPECT_EQ(indexTensor->get_stride(), (std::vector<int64_t>{18, 1, 6, 3}));
}

TEST(TestResampleFwdNode, InferStrideFromX)
{
    ResampleFwdAttributes attr;
    const auto xTensor = std::make_shared<TensorAttributes>();
    // NCHW
    xTensor->set_dim({1, 3, 32, 32}).set_stride({3072, 1024, 32, 1});
    attr.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 16, 16});
    attr.set_y(yTensor);

    const auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{768, 256, 16, 1}));
    EXPECT_EQ(indexTensor->get_stride(), (std::vector<int64_t>{768, 256, 16, 1}));
}

TEST(TestResampleFwdNode, GetNodeTypeReturnsResampleFwd)
{
    const GraphAttributes graphAttrs;
    const ResampleFwdNode node(ResampleFwdAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RESAMPLE_FWD);
}

namespace
{
class TestResampleFwdNodeCreateOperation : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;
    std::array<char, 8> _fakeDescriptors{};
    size_t _nextDescriptor = 0;
    std::vector<bool> _sentGenerateIndex;

    void SetUp() override
    {
        using ::testing::_;
        using ::testing::Return;

        _mockBackend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t,
                                  hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    std::addressof(_fakeDescriptors[_nextDescriptor++ % _fakeDescriptors.size()]));
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend,
                backendSetAttribute(
                    _, HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT, HIPDNN_TYPE_BOOLEAN, 1, _))
            .WillByDefault([this](hipdnnBackendDescriptor_t,
                                  hipdnnBackendAttributeName_t,
                                  hipdnnBackendAttributeType_t,
                                  int64_t,
                                  const void* value) {
                _sentGenerateIndex.push_back(*static_cast<const bool*>(value));
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    // makeX8x8Attributes() with the uids, data types and strides lowering needs.
    static ResampleFwdAttributes makeLowerableX8x8Attributes()
    {
        auto attr = makeX8x8Attributes();
        attr.get_x()
            ->set_uid(1)
            .set_name("X")
            .set_data_type(DataType::FLOAT)
            .set_stride({192, 64, 8, 1});
        attr.get_y()
            ->set_uid(2)
            .set_name("Y")
            .set_data_type(DataType::FLOAT)
            .set_stride({48, 16, 4, 1});
        attr.set_compute_data_type(DataType::FLOAT);
        return attr;
    }

    static Error preValidateAndCreateOperation(ResampleFwdAttributes attr)
    {
        const GraphAttributes graphAttributes;
        const ResampleFwdNode node(std::move(attr), graphAttributes);

        auto error = node.pre_validate_node();
        if(!error.is_good())
        {
            return error;
        }

        std::unordered_map<int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor>
            tensorDescs;
        std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;
        return node.create_operation(tensorDescs, operations);
    }
};
} // namespace

TEST_F(TestResampleFwdNodeCreateOperation, AvgpoolLowersGenerateIndexFalse)
{
    for(const auto mode :
        {ResampleMode::AVGPOOL_EXCLUDE_PADDING, ResampleMode::AVGPOOL_INCLUDE_PADDING})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        _sentGenerateIndex.clear();

        auto attr = makeLowerableX8x8Attributes();
        attr.set_resample_mode(mode).set_generate_index(true);

        const auto error = preValidateAndCreateOperation(std::move(attr));
        ASSERT_TRUE(error.is_good()) << error.get_message();
        EXPECT_EQ(_sentGenerateIndex, std::vector<bool>{false});
    }
}

TEST_F(TestResampleFwdNodeCreateOperation, MaxpoolWithIndexLowersGenerateIndexTrue)
{
    auto attr = makeLowerableX8x8Attributes();
    const auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_uid(3)
        .set_name("Index")
        .set_data_type(DataType::INT32)
        .set_dim({1, 3, 4, 4})
        .set_stride({48, 16, 4, 1});
    attr.set_index(indexTensor);
    attr.set_resample_mode(ResampleMode::MAXPOOL).set_generate_index(true);

    const auto error = preValidateAndCreateOperation(std::move(attr));
    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(_sentGenerateIndex, std::vector<bool>{true});
}
