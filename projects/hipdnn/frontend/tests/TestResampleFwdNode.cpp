// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{

// Runs output inference for x = {1, 3, 8, 8} with zero padding and the given window and
// stride; yDims receives the y dimensions afterwards.
Error inferYFromX8x8(const std::vector<int64_t>& window,
                     const std::vector<int64_t>& stride,
                     std::vector<int64_t>& yDims)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_window(window);
    attr.set_stride(stride);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
    yDims = yTensor->get_dim();
    return error;
}

// Pre-validates x = {1, 3, 8, 8} and y = {1, 3, 4, 4} with the given resample parameters.
Error preValidateX8x8(const std::vector<int64_t>& window,
                      const std::vector<int64_t>& stride,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 4, 4});
    attr.set_y(yTensor);

    attr.set_pre_padding(prePadding);
    attr.set_post_padding(postPadding);
    attr.set_window(window);
    attr.set_stride(stride);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);
    return node.pre_validate_node();
}

} // namespace

TEST(TestResampleFwdNode, ResampleFwdProperties)
{
    ResampleFwdAttributes attr;
    attr.set_x(std::make_shared<TensorAttributes>());
    attr.set_y(std::make_shared<TensorAttributes>());
    attr.set_index(std::make_shared<TensorAttributes>());

    auto inputTensor = attr.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = attr.get_y();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto indexTensor = attr.get_index();
    indexTensor->set_uid(3).set_name("IndexTensor");

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();

    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));

    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
    EXPECT_EQ(indexTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestResampleFwdNode, PreValidateNode)
{
    ResampleFwdAttributes attr;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_y(yTensor);

    auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_data_type(hipdnn_frontend::DataType::INT32);
    attr.set_index(indexTensor);
    attr.set_generate_index(true);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestResampleFwdNode, PreValidateNodeMissingValues)
{
    ResampleFwdAttributes attr;

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
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    attr.set_y(yTensor);

    const GraphAttributes graphAttributes;

    // Ok to not have index tensor when generate index set to false
    {
        ResampleFwdAttributes attrCopy = attr;
        attrCopy.set_generate_index(false);
        attrCopy.set_resample_mode(ResampleMode::MAXPOOL);
        const ResampleFwdNode node(std::move(attrCopy), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::OK);
    }

    // Ok to not have index tensor when using a resample mode other than MAXPOOL
    {
        ResampleFwdAttributes attrCopy = attr;
        attrCopy.set_generate_index(true);
        attrCopy.set_resample_mode(ResampleMode::AVGPOOL_EXCLUDE_PADDING);
        const ResampleFwdNode node(std::move(attrCopy), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::OK);
    }

    // Not ok to not have index tensor when using a resample mode other than MAXPOOL
    {
        ResampleFwdAttributes attrCopy = attr;
        attrCopy.set_generate_index(true);
        attrCopy.set_resample_mode(ResampleMode::MAXPOOL);
        const ResampleFwdNode node(std::move(attrCopy), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    }
}

TEST(TestResampleFwdNode, InvalidIndexType)
{
    ResampleFwdAttributes attr;
    attr.set_x(std::make_shared<TensorAttributes>());
    attr.set_y(std::make_shared<TensorAttributes>());
    auto indexTensor = std::make_shared<TensorAttributes>();
    indexTensor->set_data_type(hipdnn_frontend::DataType::FLOAT);
    attr.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InvalidXDim)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 32, 32});
    attr.set_y(yTensor);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InvalidYDim)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32, 32, 32});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64});
    attr.set_y(yTensor);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InferYDim3D)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    attr.set_pre_padding(std::vector<int64_t>{1});
    attr.set_post_padding(std::vector<int64_t>{1});
    attr.set_stride(std::vector<int64_t>{2});
    attr.set_window(std::vector<int64_t>{3});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 3, 16}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 3, 16}));
}

TEST(TestResampleFwdNode, InferDim4D)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    attr.set_pre_padding(std::vector<int64_t>{1, 1});
    attr.set_post_padding(std::vector<int64_t>{1, 1});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{3, 3});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 3, 16, 16}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 3, 16, 16}));
}

TEST(TestResampleFwdNode, InferDim5D)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 16, 16, 16});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    attr.set_pre_padding(std::vector<int64_t>{2, 2, 2});
    attr.set_post_padding(std::vector<int64_t>{2, 2, 2});
    attr.set_stride(std::vector<int64_t>{3, 3, 3});
    attr.set_window(std::vector<int64_t>{1, 1, 1});

    const GraphAttributes graphAttributes;

    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(yTensor->get_dim(), (std::vector<int64_t>{1, 3, 7, 7, 7}));
    EXPECT_EQ(indexTensor->get_dim(), (std::vector<int64_t>{1, 3, 7, 7, 7}));
}

TEST(TestResampleFwdNode, InferRejectsZeroStride)
{
    std::vector<int64_t> yDims;
    auto error = inferYFromX8x8({2, 2}, {0, 0}, yDims);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferRejectsEmptyWindow)
{
    std::vector<int64_t> yDims;
    auto error = inferYFromX8x8({}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferRejectsWindowShorterThanSpatialRank)
{
    std::vector<int64_t> yDims;
    auto error = inferYFromX8x8({2}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, InferTruncatesOutputExtentAsCudnn)
{
    // (8 - 9) / 2 + 1 = 1: a window wider than the padded input by less than the stride
    std::vector<int64_t> yDims;
    auto error = inferYFromX8x8({9, 9}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    EXPECT_EQ(yDims, (std::vector<int64_t>{1, 3, 1, 1}));
}

TEST(TestResampleFwdNode, InferRejectsOutputExtentBelowOne)
{
    // (8 - 11) / 2 + 1 = 0
    std::vector<int64_t> yDims;
    auto error = inferYFromX8x8({11, 11}, {2, 2}, yDims);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yDims.empty());
}

TEST(TestResampleFwdNode, PreValidateRejectsWindowShorterThanSpatialRankWithYSet)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 4, 4});
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{2});

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, PreValidateAcceptsWindowWiderThanPaddedInputWithYSet)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 1, 1});
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{9, 9});

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
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
    auto error = preValidateX8x8({2, 2}, {2, 2, 2}, {0, 0}, {0, 0});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestResampleFwdNode, InferRejectsXWithoutDims)
{
    ResampleFwdAttributes attr;
    attr.set_x(std::make_shared<TensorAttributes>());
    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    attr.set_pre_padding(std::vector<int64_t>{0, 0});
    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{2, 2});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(yTensor->get_dim().empty());
}

TEST(TestResampleFwdNode, InferRejectsXRankBelowThree)
{
    // No resample parameters are set: the rank check must reject x before they are read
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32});
    attr.set_x(xTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(yTensor->get_dim().empty());
}

TEST(TestResampleFwdNode, InferRejectsEmptyPrePadding)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 8, 8});
    attr.set_x(xTensor);
    auto yTensor = std::make_shared<TensorAttributes>();
    attr.set_y(yTensor);

    attr.set_post_padding(std::vector<int64_t>{0, 0});
    attr.set_stride(std::vector<int64_t>{2, 2});
    attr.set_window(std::vector<int64_t>{2, 2});

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(yTensor->get_dim().empty());
}

TEST(TestResampleFwdNode, InferStride)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 3, 7, 5});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 3, 3, 2});
    attr.set_y(yTensor);

    auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Stride should default to channel last
    EXPECT_EQ(yTensor->get_stride(), (std::vector<int64_t>{18, 1, 6, 3}));
    EXPECT_EQ(indexTensor->get_stride(), (std::vector<int64_t>{18, 1, 6, 3}));
}

TEST(TestResampleFwdNode, InferStrideFromX)
{
    ResampleFwdAttributes attr;
    auto xTensor = std::make_shared<TensorAttributes>();
    // NCHW
    xTensor->set_dim({1, 3, 32, 32}).set_stride({3072, 1024, 32, 1});
    attr.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 3, 16, 16});
    attr.set_y(yTensor);

    auto indexTensor = std::make_shared<TensorAttributes>();
    attr.set_index(indexTensor);

    const GraphAttributes graphAttributes;
    ResampleFwdNode node(std::move(attr), graphAttributes);
    auto error = node.infer_properties_node();
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
