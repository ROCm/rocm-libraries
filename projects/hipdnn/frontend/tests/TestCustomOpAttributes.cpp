// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
std::shared_ptr<TensorAttributes>
    makeTensor(const std::string& name, int64_t uid, const std::vector<int64_t>& dims)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_name(name).set_uid(uid).set_dim(dims).set_data_type(DataType::FLOAT);
    return tensor;
}

CustomOpAttributes makeEqualityAttributes()
{
    CustomOpAttributes attr;
    attr.set_name("op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT)
        .set_inputs({makeTensor("a", 1, {2, 3}), makeTensor("b", 2, {4, 5})})
        .set_outputs({makeTensor("out", 3, {2, 3})})
        .set_data({0x01, 0x02});
    return attr;
}
} // namespace

TEST(TestCustomOpAttributes, DefaultValues)
{
    const CustomOpAttributes attr;

    EXPECT_TRUE(attr.get_name().empty());
    EXPECT_TRUE(attr.get_custom_op_id().empty());
    EXPECT_EQ(attr.get_compute_data_type(), DataType::NOT_SET);
    EXPECT_TRUE(attr.get_inputs().empty());
    EXPECT_TRUE(attr.get_outputs().empty());
    EXPECT_TRUE(attr.get_data().empty());
}

TEST(TestCustomOpAttributes, SettersAndGetters)
{
    CustomOpAttributes attr;
    attr.set_name("test_custom_op")
        .set_custom_op_id("example.my_add")
        .set_compute_data_type(DataType::FLOAT);

    EXPECT_EQ(attr.get_name(), "test_custom_op");
    EXPECT_EQ(attr.get_custom_op_id(), "example.my_add");
    EXPECT_EQ(attr.get_compute_data_type(), DataType::FLOAT);
}

TEST(TestCustomOpAttributes, InputsAndOutputs)
{
    const auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_name("input_a").set_uid(1);

    const auto inputB = std::make_shared<TensorAttributes>();
    inputB->set_name("input_b").set_uid(2);

    const auto output = std::make_shared<TensorAttributes>();
    output->set_name("output").set_uid(3);

    CustomOpAttributes attr;
    attr.set_inputs({inputA, inputB}).set_outputs({output});

    EXPECT_EQ(attr.get_inputs().size(), 2u);
    EXPECT_EQ(attr.get_outputs().size(), 1u);
    EXPECT_EQ(attr.get_inputs()[0]->get_uid(), 1);
    EXPECT_EQ(attr.get_inputs()[1]->get_uid(), 2);
    EXPECT_EQ(attr.get_outputs()[0]->get_uid(), 3);
}

TEST(TestCustomOpAttributes, Data)
{
    const std::vector<uint8_t> payload = {0x01, 0x02, 0x03, 0xFF};

    CustomOpAttributes attr;
    attr.set_data(payload);

    EXPECT_EQ(attr.get_data(), payload);
}

TEST(TestCustomOpAttributes, FillFromContext)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);
    graphAttrs.set_io_data_type(DataType::HALF);

    const auto inputA = std::make_shared<TensorAttributes>();
    inputA->set_uid(1).set_dim({2, 3}).set_stride({3, 1});

    const auto output = std::make_shared<TensorAttributes>();
    output->set_uid(2).set_dim({2, 3}).set_stride({3, 1});

    CustomOpAttributes attr;
    attr.set_inputs({inputA}).set_outputs({output});

    const auto err = attr.fill_from_context(graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(attr.get_compute_data_type(), DataType::FLOAT);

    // Tensors should have their data_type filled from graph context
    EXPECT_EQ(inputA->get_data_type(), DataType::HALF);
    EXPECT_EQ(output->get_data_type(), DataType::HALF);
}

TEST(TestCustomOpAttributes, FillFromContextPreservesExplicitComputeType)
{
    GraphAttributes graphAttrs;
    graphAttrs.set_compute_data_type(DataType::FLOAT);

    CustomOpAttributes attr;
    attr.set_compute_data_type(DataType::HALF);

    const auto err = attr.fill_from_context(graphAttrs);
    EXPECT_EQ(err.code, ErrorCode::OK);
    EXPECT_EQ(attr.get_compute_data_type(), DataType::HALF);
}

TEST(TestCustomOpAttributes, EqualAttributesCompareEqual)
{
    const auto lhs = makeEqualityAttributes();
    const auto rhs = makeEqualityAttributes();

    EXPECT_TRUE(lhs.logicallyEquals(rhs));
    EXPECT_TRUE(lhs == rhs);
    EXPECT_FALSE(lhs != rhs);
}

TEST(TestCustomOpAttributes, LogicalEqualityIgnoresNamesAndUids)
{
    const auto lhs = makeEqualityAttributes();

    auto rhs = makeEqualityAttributes();
    rhs.set_name("other_op")
        .set_inputs({makeTensor("x", 11, {2, 3}), makeTensor("y", 12, {4, 5})})
        .set_outputs({makeTensor("z", 13, {2, 3})});

    EXPECT_TRUE(lhs.logicallyEquals(rhs));
    EXPECT_FALSE(lhs == rhs);
}

TEST(TestCustomOpAttributes, EqualityComparesCustomOpIdAndData)
{
    const auto lhs = makeEqualityAttributes();

    auto otherId = makeEqualityAttributes();
    otherId.set_custom_op_id("example.my_mul");
    EXPECT_FALSE(lhs.logicallyEquals(otherId));
    EXPECT_FALSE(lhs == otherId);

    auto otherData = makeEqualityAttributes();
    otherData.set_data({0x01, 0x03});
    EXPECT_FALSE(lhs.logicallyEquals(otherData));
    EXPECT_FALSE(lhs == otherData);

    auto otherComputeType = makeEqualityAttributes();
    otherComputeType.set_compute_data_type(DataType::HALF);
    EXPECT_FALSE(lhs.logicallyEquals(otherComputeType));
    EXPECT_FALSE(lhs == otherComputeType);
}

TEST(TestCustomOpAttributes, EqualityRespectsTensorOrder)
{
    const auto lhs = makeEqualityAttributes();

    auto swapped = makeEqualityAttributes();
    swapped.set_inputs({makeTensor("b", 2, {4, 5}), makeTensor("a", 1, {2, 3})});

    EXPECT_FALSE(lhs.logicallyEquals(swapped));
    EXPECT_FALSE(lhs == swapped);

    auto extraOutput = makeEqualityAttributes();
    extraOutput.set_outputs({makeTensor("out", 3, {2, 3}), makeTensor("out2", 4, {2, 3})});
    EXPECT_FALSE(lhs.logicallyEquals(extraOutput));
    EXPECT_FALSE(lhs == extraOutput);
}

TEST(TestCustomOpAttributes, EqualityTreatsNullEntriesPositionally)
{
    auto lhs = makeEqualityAttributes();
    lhs.set_inputs({nullptr, makeTensor("b", 2, {4, 5})});

    auto rhs = makeEqualityAttributes();
    rhs.set_inputs({nullptr, makeTensor("b", 2, {4, 5})});
    EXPECT_TRUE(lhs.logicallyEquals(rhs));
    EXPECT_TRUE(lhs == rhs);

    auto nonNull = makeEqualityAttributes();
    EXPECT_FALSE(lhs.logicallyEquals(nonNull));
    EXPECT_FALSE(nonNull.logicallyEquals(lhs));
    EXPECT_FALSE(lhs == nonNull);
}
