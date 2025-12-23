// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/Attributes.hpp>
#include <hipdnn_frontend/node/Node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace ::testing;

namespace
{

struct FakeAttributes : public Attributes<FakeAttributes>
{
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>> outputs;
};

class FakeNode : public NodeCRTP<FakeNode>
{
public:
    FakeNode(FakeAttributes&& fakeAttrs, GraphAttributes const& graphAttrs)
        : NodeCRTP<FakeNode>(graphAttrs)
        , attributes(std::move(fakeAttrs))
    {
    }
    FakeAttributes attributes;
};

TEST(TestNode, PostValidateNodeComputeDataType)
{
    GraphAttributes graphAttributes;
    FakeNode node(FakeAttributes{}, graphAttributes);

    std::vector<std::pair<DataType, ErrorCode>> expectedResults
        = {{DataType::NOT_SET, ErrorCode::ATTRIBUTE_NOT_SET},
           {DataType::FLOAT, ErrorCode::OK},
           {DataType::HALF, ErrorCode::OK},
           {DataType::BFLOAT16, ErrorCode::OK},
           {DataType::DOUBLE, ErrorCode::OK},
           {DataType::UINT8, ErrorCode::OK},
           {DataType::INT32, ErrorCode::OK},
           {DataType::INT8, ErrorCode::OK},
           {DataType::FP8_E4M3, ErrorCode::OK}};

    for(auto [dataType, errorCode] : expectedResults)
    {
        node.attributes.set_compute_data_type(dataType);
        auto result = node.post_validate_node();
        EXPECT_EQ(result.code, errorCode) << "For " + std::string(to_string(dataType));
    }
}

enum class TensorAttributeValidity
{
    VALID,
    INVALID
};

struct FakeNodeParams
{
    std::vector<TensorAttributeValidity> inputs;
    std::vector<TensorAttributeValidity> outputs;
    ErrorCode expectedResult;
};

class TestNodePostValidateNodeTensors : public ::testing::TestWithParam<FakeNodeParams>
{
protected:
    void SetUp() override
    {
        auto param = GetParam();

        auto validTensorAttribute = std::make_shared<TensorAttributes>();
        validTensorAttribute->set_data_type(DataType::FLOAT)
            .set_dim({1, 2, 3, 4})
            .set_stride({1, 2, 3, 4});

        auto invalidTensorAttribute = std::make_shared<TensorAttributes>();
        invalidTensorAttribute->set_dim({1, 2, 3, 4}).set_stride({1, 2, 3});

        _attributes.set_compute_data_type(DataType::FLOAT);
        int64_t id = 0;
        for(const auto& input : param.inputs)
        {
            _attributes.inputs[id++] = (input == TensorAttributeValidity::VALID)
                                           ? validTensorAttribute
                                           : invalidTensorAttribute;
        }
        for(const auto& output : param.outputs)
        {
            _attributes.outputs[id++] = (output == TensorAttributeValidity::VALID)
                                            ? validTensorAttribute
                                            : invalidTensorAttribute;
        }
        _expectedResult = param.expectedResult;
    }

    GraphAttributes _graphAttributes;
    FakeAttributes _attributes;
    ErrorCode _expectedResult;
};

TEST_P(TestNodePostValidateNodeTensors, Correctness)
{
    auto tensorsToString
        = [&](const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensors) {
              std::string ret = "[";

              for(const auto& [id, tensor] : tensors)
              {
                  std::string isValid = (tensor->validate().is_good()) ? "VALID" : "INVALID";
                  ret += isValid + ", ";
              }

              if(!tensors.empty())
              {
                  ret.resize(ret.size() - 2);
              }

              ret.push_back(']');
              return ret;
          };

    std::string caseString = "Inputs: " + tensorsToString(_attributes.inputs)
                             + " Outputs: " + tensorsToString(_attributes.outputs);
    FakeNode node(std::move(_attributes), _graphAttributes);

    auto nodes = node.getNodeOutputTensorAttributes();

    EXPECT_EQ(node.post_validate_node().code, _expectedResult) << caseString;
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestNodePostValidateNodeTensors,
    ::testing::Values(
        FakeNodeParams{{}, {TensorAttributeValidity::VALID}, ErrorCode::OK},
        FakeNodeParams{
            {TensorAttributeValidity::INVALID}, {TensorAttributeValidity::VALID}, ErrorCode::OK},
        FakeNodeParams{{TensorAttributeValidity::INVALID},
                       {TensorAttributeValidity::INVALID},
                       ErrorCode::ATTRIBUTE_NOT_SET},
        FakeNodeParams{{},
                       {TensorAttributeValidity::VALID, TensorAttributeValidity::INVALID},
                       ErrorCode::ATTRIBUTE_NOT_SET}));

}
