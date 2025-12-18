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
}

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
           {DataType::INT32, ErrorCode::OK}};

    for(auto [dataType, errorCode] : expectedResults)
    {
        node.attributes.set_compute_data_type(dataType);
        auto result = node.post_validate_node();
        EXPECT_EQ(result.code, errorCode) << "For " + std::string(to_string(dataType));
    }
}

TEST(TestNode, PostValidateNodeTensors)
{
    GraphAttributes graphAttributes;
    FakeAttributes nodeAttributes;

    auto validTensorAttribute = std::make_shared<TensorAttributes>();
    validTensorAttribute->set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({1, 2, 3, 4});

    auto invalidTensorAttribute = std::make_shared<TensorAttributes>();
    invalidTensorAttribute->set_dim({1, 2, 3, 4}).set_stride({1, 2, 3});

    auto tensorsToString
        = [&](const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensors) {
              std::string ret = "[";

              for(const auto& [id, tensor] : tensors)
              {
                  std::string isValid = (tensor == validTensorAttribute) ? "VALID" : "INVALID";
                  ret += isValid + ", ";
              }

              if(!tensors.empty())
              {
                  ret.resize(ret.size() - 2);
              }

              ret.push_back(']');
              return ret;
          };

    auto toAttributes = [&](const std::vector<std::shared_ptr<TensorAttributes>>& inputs,
                            const std::vector<std::shared_ptr<TensorAttributes>>& outputs) {
        FakeAttributes attributes;
        attributes.set_compute_data_type(DataType::FLOAT);
        int64_t id = 0;
        for(const auto& input : inputs)
        {
            attributes.inputs[id++] = input;
        }
        for(const auto& output : outputs)
        {
            attributes.outputs[id++] = output;
        }

        return attributes;
    };

    std::vector<std::pair<FakeAttributes, ErrorCode>> expectedResults
        = {{toAttributes({}, {validTensorAttribute}), ErrorCode::OK},
           {toAttributes({invalidTensorAttribute}, {validTensorAttribute}), ErrorCode::OK},
           {toAttributes({invalidTensorAttribute}, {invalidTensorAttribute}),
            ErrorCode::ATTRIBUTE_NOT_SET},
           {toAttributes({}, {validTensorAttribute, invalidTensorAttribute}),
            ErrorCode::ATTRIBUTE_NOT_SET}};

    for(auto [attributes, errorCode] : expectedResults)
    {
        std::string caseString = "Inputs: " + tensorsToString(attributes.inputs)
                                 + " Outputs: " + tensorsToString(attributes.outputs);
        FakeNode node(std::move(attributes), graphAttributes);

        auto nodes = node.getNodeOutputTensorAttributes();

        EXPECT_EQ(node.post_validate_node().code, errorCode) << caseString;
    }
}
