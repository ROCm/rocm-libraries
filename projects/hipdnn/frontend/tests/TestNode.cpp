// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/Attributes.hpp>
#include <hipdnn_frontend/node/Node.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace ::testing;

namespace hipdnn_frontend
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

namespace graph::test
{
class INode
{
public:
    GraphAttributes graph_attributes; // NOLINT(readability-identifier-naming)
    INode(GraphAttributes attributes)
        : graph_attributes(std::move(attributes))
    {
    }
    virtual ~INode() = default;

    virtual Error pre_validate_node() const // NOLINT(readability-identifier-naming)
    {
        return {};
    }
    virtual Error infer_properties_node() // NOLINT(readability-identifier-naming)
    {
        return {};
    }
    virtual Error post_validate_node() const // NOLINT(readability-identifier-naming)
    {
        return {};
    }
    virtual void
        // NOLINTNEXTLINE(readability-identifier-naming)
        gather_hipdnn_tensors(
            [[maybe_unused]] std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors)
            const
    {
    }

    virtual flatbuffers::Offset<hipdnn_sdk::data_objects::Node>
        pack_node([[maybe_unused]] flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return {};
    }

    virtual std::vector<std::shared_ptr<TensorAttributes>> getNodeInputTensorAttributes() const
    {
        return {};
    }

    virtual std::vector<std::shared_ptr<TensorAttributes>> getNodeOutputTensorAttributes() const
    {
        std::cout << "INode::getNodeOutputTensorAttributes()\n";
        return {};
    }

    void visit(const std::function<void(INode&)>& visitor)
    {
        // Visit current node first (pre-order traversal)
        visitor(*this);

        // Then visit all children
        for(const auto& child : _sub_nodes)
        {
            if(child)
            {
                child->visit(visitor);
            }
        }
    }

    void visit(const std::function<void(const INode&)>& visitor) const
    {
        // Visit current node first (pre-order traversal)
        visitor(*this);

        // Then visit all children
        for(const auto& child : _sub_nodes)
        {
            if(child)
            {
                // Explicitly call const version by getting const reference
                const INode& constChild = *child;
                constChild.visit(visitor);
            }
        }
    }

protected:
    std::vector<std::shared_ptr<INode>> _sub_nodes;

    Error validateSubtree()
    {
        HIPDNN_CHECK_ERROR(pre_validate_node());
        HIPDNN_CHECK_ERROR(infer_properties_node());
        for(const auto& node : _sub_nodes)
        {
            HIPDNN_CHECK_ERROR(node->validateSubtree());
        }
        HIPDNN_CHECK_ERROR(post_validate_node());
        return {};
    }

    void gatherHipdnnTensorsSubtree(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const
    {
        gather_hipdnn_tensors(allTensors);

        for(const auto& node : _sub_nodes)
        {
            node->gatherHipdnnTensorsSubtree(allTensors);
        }
    }

    void gatherOutputTensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& outputTensors) const
    {
        auto outputs = getNodeOutputTensorAttributes();
        outputTensors.insert(outputs.begin(), outputs.end());

        for(const auto& node : _sub_nodes)
        {
            node->gatherOutputTensors(outputTensors);
        }
    }

    void gatherInputTensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& inputTensors,
        const std::unordered_set<std::shared_ptr<TensorAttributes>>& outputTensors) const
    {
        auto inputs = getNodeInputTensorAttributes();
        for(const auto& input : inputs)
        {
            if(outputTensors.count(input) == 0)
            {
                inputTensors.insert(input);
            }
        }

        for(const auto& node : _sub_nodes)
        {
            node->gatherInputTensors(inputTensors, outputTensors);
        }
    }
};

// Any class extending BaseNode must have an attributes member with an inputs & outputs map.
// The map needs to have TensorAttributes as the value.
// BaseNode uses this to gather tensor uids, and populate unset ones.
template <typename DerivedT>
class BaseNode : public INode
{
private:
    DerivedT& self()
    {
        return static_cast<DerivedT&>(*this);
    }
    const DerivedT& self() const
    {
        return static_cast<const DerivedT&>(*this);
    }

public:
    // NOLINTNEXTLINE(readability-identifier-naming)
    void gather_hipdnn_tensors(
        std::unordered_set<std::shared_ptr<TensorAttributes>>& allTensors) const override
    {
        for(auto& [_, tensor] : self().attributes.inputs)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }

        for(auto& [_, tensor] : self().attributes.outputs)
        {
            if(tensor)
            {
                allTensors.insert(tensor);
            }
        }
    }

    Error post_validate_node() const override // NOLINT(readability-identifier-naming)
    {
        if(self().attributes.compute_data_type == DataType::NOT_SET)
        {
            return {ErrorCode::ATTRIBUTE_NOT_SET,
                    "Node " + self().attributes.name + " does not have a compute_data_type set"};
        }

        std::cout << "outputs.size() = " << this->getNodeOutputTensorAttributes().size() << "\n";

        for(const auto& tensorAttr : this->getNodeOutputTensorAttributes())
        {
            HIPDNN_CHECK_ERROR(tensorAttr->validate());
        }

        return {ErrorCode::OK, ""};
    }

    std::vector<std::shared_ptr<TensorAttributes>> getNodeInputTensorAttributes() const override
    {
        std::vector<std::shared_ptr<TensorAttributes>> inputAttributes;
        for(auto& tensorAttrPair : self().attributes.inputs)
        {
            if(tensorAttrPair.second)
            {
                inputAttributes.push_back(tensorAttrPair.second);
            }
        }

        return inputAttributes;
    }

    std::vector<std::shared_ptr<TensorAttributes>> getNodeOutputTensorAttributes() const override
    {
        std::cout << "BaseNode::getNodeOutputTensorAttributes()\n";
        std::vector<std::shared_ptr<TensorAttributes>> outputAttributes;
        for(auto& tensorAttrPair : self().attributes.outputs)
        {
            if(tensorAttrPair.second)
            {
                outputAttributes.push_back(tensorAttrPair.second);
            }
        }

        return outputAttributes;
    }

protected:
    using INode::INode;
};

template <typename DerivedT>
using NodeCRTP = BaseNode<DerivedT>; // NOLINT

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

TEST(TestNode, Scratch)
{
    GraphAttributes graphAttributes;
    graph::test::FakeAttributes nodeAttributes;
    nodeAttributes.set_compute_data_type(DataType::FLOAT);

    graph::test::FakeNode node(std::move(nodeAttributes), graphAttributes);

    auto nodes = node.getNodeOutputTensorAttributes();

    EXPECT_EQ(node.post_validate_node(), ErrorCode::OK);
}

TEST(TestNode, Scratch2)
{
    GraphAttributes graphAttributes;
    FakeAttributes nodeAttributes;
    nodeAttributes.set_compute_data_type(DataType::FLOAT);

    FakeNode node(std::move(nodeAttributes), graphAttributes);

    auto nodes = node.getNodeOutputTensorAttributes();

    EXPECT_EQ(node.post_validate_node(), ErrorCode::OK);
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

TEST(TestNode, PostValidateTensors)
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

} // namespace hipdnn_frontend
