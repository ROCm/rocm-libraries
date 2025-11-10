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

} // namespace hipdnn_frontend
