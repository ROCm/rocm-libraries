// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/node/ResampleFwdNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

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
    attr.set_x(std::make_shared<TensorAttributes>());
    attr.set_y(std::make_shared<TensorAttributes>());
    attr.set_generate_index(true);

    const GraphAttributes graphAttributes;
    const ResampleFwdNode node(std::move(attr), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
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

TEST(TestResampleFwdNode, GetNodeTypeReturnsResampleFwd)
{
    const GraphAttributes graphAttrs;
    const ResampleFwdNode node(ResampleFwdAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RESAMPLE_FWD);
}
