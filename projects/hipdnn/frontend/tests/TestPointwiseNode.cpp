// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
namespace
{
// Generic helper function to generate vectors of pointwise modes based on a checker function
template <typename CheckerFunc>
std::vector<PointwiseMode> getPointwiseModesByChecker(CheckerFunc checker)
{
    std::vector<PointwiseMode> modes;
    // Iterate through all possible PointwiseMode values and check if they match the criteria
    for(int i = static_cast<int>(hipdnn_sdk::data_objects::PointwiseMode::MIN);
        i <= static_cast<int>(hipdnn_sdk::data_objects::PointwiseMode::MAX);
        ++i)
    {
        auto mode = static_cast<PointwiseMode>(i);
        if(checker(mode))
        {
            modes.push_back(mode);
        }
    }
    return modes;
}

// Helper functions that use the generic function with specific checkers
std::vector<PointwiseMode> getUnaryPointwiseModes()
{
    return getPointwiseModesByChecker(isUnaryPointwiseMode);
}

std::vector<PointwiseMode> getBinaryPointwiseModes()
{
    return getPointwiseModesByChecker(isBinaryPointwiseMode);
}

std::vector<PointwiseMode> getTernaryPointwiseModes()
{
    return getPointwiseModesByChecker(isTernaryPointwiseMode);
}
}
}

// Parameterized test for unary operations
class TestPointwiseNodeUnaryOps : public ::testing::TestWithParam<PointwiseMode>
{
protected:
    void SetUp() override
    {
        _mode = GetParam();
    }

    PointwiseMode _mode;
};

TEST_P(TestPointwiseNodeUnaryOps, ValidWithOneInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2).set_name("OutputTensor").set_dim({2, 3, 4});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Unary operation " << static_cast<int>(_mode) << " should be valid with one input";
}

TEST_P(TestPointwiseNodeUnaryOps, InvalidWithTwoInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE)
        << "Unary operation " << static_cast<int>(_mode) << " should be invalid with two inputs";
}

TEST_P(TestPointwiseNodeUnaryOps, InvalidWithThreeInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE)
        << "Unary operation " << static_cast<int>(_mode) << " should be invalid with three inputs";
}

// Instantiate test suite with all unary operations from Types.hpp
INSTANTIATE_TEST_SUITE_P(,
                         TestPointwiseNodeUnaryOps,
                         ::testing::ValuesIn(getUnaryPointwiseModes()));

// Parameterized test for binary operations
class TestPointwiseNodeBinaryOps : public ::testing::TestWithParam<PointwiseMode>
{
protected:
    void SetUp() override
    {
        _mode = GetParam();
    }

    PointwiseMode _mode;
};

TEST_P(TestPointwiseNodeBinaryOps, InvalidWithOneInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE)
        << "Binary operation " << static_cast<int>(_mode) << " should be invalid with one input";
}

TEST_P(TestPointwiseNodeBinaryOps, ValidWithTwoInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_dim({2, 3, 4});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Binary operation " << static_cast<int>(_mode) << " should be valid with two inputs";
}

TEST_P(TestPointwiseNodeBinaryOps, InvalidWithThreeInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE)
        << "Binary operation " << static_cast<int>(_mode) << " should be invalid with three inputs";
}

// Instantiate test suite with all binary operations from Types.hpp
INSTANTIATE_TEST_SUITE_P(,
                         TestPointwiseNodeBinaryOps,
                         ::testing::ValuesIn(getBinaryPointwiseModes()));

// Parameterized test for ternary operations
class TestPointwiseNodeTernaryOps : public ::testing::TestWithParam<PointwiseMode>
{
protected:
    void SetUp() override
    {
        _mode = GetParam();
    }

    PointwiseMode _mode;
};

TEST_P(TestPointwiseNodeTernaryOps, InvalidWithOneInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE)
        << "Ternary operation " << static_cast<int>(_mode) << " should be invalid with one input";
}

TEST_P(TestPointwiseNodeTernaryOps, InvalidWithTwoInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE)
        << "Ternary operation " << static_cast<int>(_mode) << " should be invalid with two inputs";
}

TEST_P(TestPointwiseNodeTernaryOps, ValidWithThreeInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto inputTensor2 = attributes.get_input_2();
    inputTensor2->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(4).set_name("OutputTensor").set_dim({2, 3, 4});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Ternary operation " << static_cast<int>(_mode) << " should be valid with three inputs";
}

// Instantiate test suite with all ternary operations from Types.hpp
INSTANTIATE_TEST_SUITE_P(,
                         TestPointwiseNodeTernaryOps,
                         ::testing::ValuesIn(getTernaryPointwiseModes()));

TEST(TestPointwiseNode, ZeroInputsError)
{
    PointwiseAttributes attributes;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, MoreThanThreeInputsError)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());

    attributes.inputs[static_cast<PointwiseAttributes::input_names>(3)]
        = std::make_shared<TensorAttributes>();

    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::BINARY_SELECT);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, MissingOutputError)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, BroadcastableDimsValid)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Input0: [1, 3, 1, 4] - broadcastable to output
    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 3, 1, 4})
        .set_stride({12, 4, 4, 1});

    // Input1: [2, 1, 5, 1] - broadcastable to output
    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 1, 5, 1})
        .set_stride({5, 5, 1, 1});

    // Output: [2, 3, 5, 4] - common broadcast shape
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_dim({2, 3, 5, 4})
        .set_stride({60, 20, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestPointwiseNode, StrideInferenceFailsWithMismatchedDims)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    // Input: [2, 3, 4] with strides
    auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Output: [5, 6, 7] - completely different dims, no stride set
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({5, 6, 7});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, InferDimsAndStridesFromSingleInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    // Input with dims and strides
    auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Output with no dims or strides - should be inferred
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Check that dims and strides were inferred
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{12, 4, 1}));
}

TEST(TestPointwiseNode, BroadcastTwoInputsWithMismatchedDims)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Input0: [1, 3, 1] - smaller shape
    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 3, 1})
        .set_stride({3, 1, 1});

    // Input1: [2, 1, 4] - different shape
    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 1, 4})
        .set_stride({4, 4, 1});

    // Output: [2, 3, 4] - matches the broadcast result
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestPointwiseNode, InferCommonShapeFromMultipleInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Input0: [2, 3, 4]
    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({3, 1, 1});

    // Input1: [2, 1, 4]
    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 1, 4})
        .set_stride({4, 4, 1});

    // Output with no dims - should be inferred as [2, 3, 4]
    // strides not set, should be inferred as [3, 1, 1]
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Check that common shape was inferred correctly
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{3, 1, 1}));
}

TEST(TestPointwiseNode, ScalarBroadcast)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Input0: scalar [1]
    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("ScalarInput")
        .set_data_type(DataType::FLOAT)
        .set_dim({1})
        .set_stride({1});

    // Input1: [2, 3, 4]
    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("TensorInput")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Output: [2, 3, 4]
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestPointwiseNode, BroadcastWithFewerDimensions)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Input0: [3, 4] - can broadcast to [2, 3, 4]
    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({3, 4})
        .set_stride({4, 1});

    // Input1: [4] - can broadcast to [2, 3, 4]
    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({4})
        .set_stride({1});

    // Output: [2, 3, 4]
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestPointwiseNode, NonBroadcastableDimensionsError)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Input0: [2, 3, 4] - NOT broadcastable to output [2, 5, 4]
    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Input1: [2, 5, 4] - matches output
    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 5, 4})
        .set_stride({20, 4, 1});

    // Output: [2, 5, 4]
    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 5, 4})
        .set_stride({20, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, MultipleOutputsError)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());

    // Manually add a second output
    attributes.outputs[static_cast<PointwiseAttributes::OutputNames>(2)]
        = std::make_shared<TensorAttributes>();

    attributes.set_mode(PointwiseMode::RELU_FWD);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, NullInputTensorPreValidation)
{
    PointwiseAttributes attributes;
    // Set input_0 to nullptr explicitly
    attributes.inputs[PointwiseAttributes::InputNames::IN_0] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(1)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, NullInputAmongMultipleInputsPreValidation)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    // Set input_1 to nullptr
    attributes.inputs[PointwiseAttributes::InputNames::IN_1] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, NullInputTensorInferProperties)
{
    PointwiseAttributes attributes;
    // Set input_0 to nullptr
    attributes.inputs[PointwiseAttributes::InputNames::IN_0] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(1).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, NullInputAmongMultipleInferProperties)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    // Set input_2 to nullptr
    attributes.inputs[PointwiseAttributes::InputNames::IN_2] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::BINARY_SELECT);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(4).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, NullSecondInputInferProperties)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    // Manually add nullptr as second input
    attributes.inputs[PointwiseAttributes::InputNames::IN_1] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}
