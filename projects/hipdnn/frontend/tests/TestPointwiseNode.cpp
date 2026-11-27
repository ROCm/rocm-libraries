// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

namespace
{
namespace
{
// Sentinel: PointwiseMode::COUNT is always one past the last valid mode.
// New modes inserted before COUNT automatically update the iteration range.
constexpr auto K_POINTWISE_MODE_COUNT = PointwiseMode::COUNT;
static_assert(static_cast<int>(K_POINTWISE_MODE_COUNT) == 51,
              "PointwiseMode enum changed — update this assertion");

// Generic helper function to generate vectors of pointwise modes based on a checker function.
// Only modes the lowering can express (toBackendPointwiseMode) are returned; modes without a
// backend mapping are rejected by pre-validation and covered by the agreement test below.
template <typename CheckerFunc>
std::vector<PointwiseMode> getPointwiseModesByChecker(CheckerFunc checker)
{
    std::vector<PointwiseMode> modes;
    // Iterate through all possible PointwiseMode values and check if they match the criteria
    for(int i = static_cast<int>(PointwiseMode::NOT_SET);
        i < static_cast<int>(K_POINTWISE_MODE_COUNT);
        ++i)
    {
        const auto mode = static_cast<PointwiseMode>(i);
        if(checker(mode) && toBackendPointwiseMode(mode).has_value())
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

    const auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2).set_name("OutputTensor").set_dim({2, 3, 4});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Unary operation " << static_cast<int>(_mode) << " should be valid with one input";
}

TEST_P(TestPointwiseNodeUnaryOps, ValidWithSurplusSecondInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Unary operation " << static_cast<int>(_mode) << " should accept a surplus second input";
}

TEST_P(TestPointwiseNodeUnaryOps, ValidWithSurplusSecondAndThirdInputs)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << "Unary operation " << static_cast<int>(_mode)
                                         << " should accept surplus second and third inputs";
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

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_dim({2, 3, 4});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Binary operation " << static_cast<int>(_mode) << " should be valid with two inputs";
}

TEST_P(TestPointwiseNodeBinaryOps, ValidWithSurplusThirdInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(_mode);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Binary operation " << static_cast<int>(_mode) << " should accept a surplus third input";
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

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto inputTensor2 = attributes.get_input_2();
    inputTensor2->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(4).set_name("OutputTensor").set_dim({2, 3, 4});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Ternary operation " << static_cast<int>(_mode) << " should be valid with three inputs";
}

// Instantiate test suite with all ternary operations from Types.hpp
INSTANTIATE_TEST_SUITE_P(,
                         TestPointwiseNodeTernaryOps,
                         ::testing::ValuesIn(getTernaryPointwiseModes()));

namespace
{
// A pointwise node for `mode` whose inputs match the mode's operand count, with valid shapes.
PointwiseNode makeNodeWithModeOperands(PointwiseMode mode)
{
    PointwiseAttributes attributes;
    attributes.set_mode(mode);
    const size_t inputCount = std::max<size_t>(PointwiseNode::modeInputCount(mode), 1);
    for(size_t slot = 0; slot < inputCount; ++slot)
    {
        const auto input = std::make_shared<TensorAttributes>();
        input->set_uid(static_cast<int64_t>(slot) + 1)
            .set_name("InputTensor" + std::to_string(slot))
            .set_data_type(DataType::FLOAT)
            .set_dim({2, 3, 4})
            .set_stride({12, 4, 1});
        attributes.inputs[static_cast<PointwiseAttributes::InputNames>(slot)] = input;
    }
    const auto output = std::make_shared<TensorAttributes>();
    output->set_uid(10).set_name("OutputTensor").set_dim({2, 3, 4});
    attributes.set_output_0(output);

    const GraphAttributes graphAttributes;
    return {std::move(attributes), graphAttributes};
}
} // namespace

// pre_validate_node and the lowering share one table (toBackendPointwiseMode): every mode is
// either accepted by both or rejected by both.
TEST(TestPointwiseNode, PreValidateAgreesWithLoweringForEveryMode)
{
    for(int i = static_cast<int>(PointwiseMode::NOT_SET) + 1;
        i < static_cast<int>(K_POINTWISE_MODE_COUNT);
        ++i)
    {
        const auto mode = static_cast<PointwiseMode>(i);
        const auto node = makeNodeWithModeOperands(mode);
        const auto error = node.pre_validate_node();
        EXPECT_EQ(error.is_good(), toBackendPointwiseMode(mode).has_value())
            << "Mode " << to_string(mode) << ": " << error.get_message();
    }
}

TEST(TestPointwiseNode, PreValidateRejectsModesWithoutBackendMapping)
{
    for(const auto mode : {PointwiseMode::COS, PointwiseMode::MOD, PointwiseMode::POW})
    {
        const auto node = makeNodeWithModeOperands(mode);
        const auto error = node.pre_validate_node();
        EXPECT_EQ(error,
                  Error(ErrorCode::INVALID_VALUE,
                        std::string("PointwiseNode mode ") + to_string(mode)
                            + " is not supported by hipDNN lowering"));
    }
}

TEST(TestPointwiseNode, ZeroInputsError)
{
    PointwiseAttributes attributes;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, MissingOutputError)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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
    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 3, 1, 4})
        .set_stride({12, 4, 4, 1});

    // Input1: [2, 1, 5, 1] - broadcastable to output
    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 1, 5, 1})
        .set_stride({5, 5, 1, 1});

    // Output: [2, 3, 5, 4] - common broadcast shape
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_dim({2, 3, 5, 4})
        .set_stride({60, 20, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestPointwiseNode, StrideInferenceFailsWithMismatchedDims)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    // Input: [2, 3, 4] with strides
    const auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Output: [5, 6, 7] - completely different dims, no stride set
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({5, 6, 7});

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, InferDimsAndStridesFromSingleInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    // Input with dims and strides
    const auto inputTensor = attributes.get_input_0();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Output with no dims or strides - should be inferred
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(2).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
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
    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 3, 1})
        .set_stride({3, 1, 1});

    // Input1: [2, 1, 4] - different shape
    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 1, 4})
        .set_stride({4, 4, 1});

    // Output: [2, 3, 4] - matches the broadcast result
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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
    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({3, 1, 1});

    // Input1: [2, 1, 4]
    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 1, 4})
        .set_stride({4, 4, 1});

    // Output with no dims - should be inferred as [2, 3, 4]
    // strides not set, should be inferred as [3, 1, 1]
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
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
    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("ScalarInput")
        .set_data_type(DataType::FLOAT)
        .set_dim({1})
        .set_stride({1});

    // Input1: [2, 3, 4]
    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("TensorInput")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Output: [2, 3, 4]
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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
    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({3, 4})
        .set_stride({4, 1});

    // Input1: [4] - can broadcast to [2, 3, 4]
    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({4})
        .set_stride({1});

    // Output: [2, 3, 4]
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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
    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    // Input1: [2, 5, 4] - matches output
    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 5, 4})
        .set_stride({20, 4, 1});

    // Output: [2, 5, 4]
    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 5, 4})
        .set_stride({20, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestPointwiseNode, NullInputTensorPreValidation)
{
    PointwiseAttributes attributes;
    // Set input_0 to nullptr explicitly
    attributes.inputs[PointwiseAttributes::InputNames::IN_0] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(1)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, NullInputAmongMultipleInputsPreValidation)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    // Set input_1 to nullptr
    attributes.inputs[PointwiseAttributes::InputNames::IN_1] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, NullInputTensorInferProperties)
{
    PointwiseAttributes attributes;
    // Set input_0 to nullptr
    attributes.inputs[PointwiseAttributes::InputNames::IN_0] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::RELU_FWD);

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(1).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
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

    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto inputTensor1 = attributes.get_input_1();
    inputTensor1->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(4).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, NullSecondInputInferProperties)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    // Manually add nullptr as second input
    attributes.inputs[PointwiseAttributes::InputNames::IN_1] = nullptr;
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    const auto inputTensor0 = attributes.get_input_0();
    inputTensor0->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

TEST(TestPointwiseNode, GetNodeTypeReturnsPointwise)
{
    const GraphAttributes graphAttrs;
    const PointwiseNode node(PointwiseAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::POINTWISE);
}

TEST(TestPointwiseNode, InferBroadcastAlignsLowerRankInputOnTrailingDims)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    attributes.get_input_0()
        ->set_uid(1)
        .set_name("X")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("Bias")
        .set_data_type(DataType::FLOAT)
        .set_dim({4})
        .set_stride({1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3).set_name("OutputTensor").set_data_type(DataType::FLOAT);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{12, 4, 1}));
}

TEST(TestPointwiseNode, InferBroadcastExpandsUnitDimForVirtualOutput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    attributes.get_input_0()
        ->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({4, 1})
        .set_stride({1, 1});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({4})
        .set_stride({1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_is_virtual(true);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{4, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{4, 1}));

    // Graph lowering validates again, now with the inferred output dims set.
    error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestPointwiseNode, InferVirtualOutputStridePutsExtraLeadingAxisOutermost)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    // Column-major IN_0 of lower rank than the output; no input matches the output shape
    attributes.get_input_0()
        ->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3})
        .set_stride({1, 2});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({4, 1, 1})
        .set_stride({1, 1, 1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_is_virtual(true);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{4, 2, 3}));
    // The trailing axes keep IN_0's column-major order; the leading axis is outermost
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{6, 1, 2}));
}

TEST(TestPointwiseNode, InferBroadcastStrideFailsForNonVirtualOutputMatchingNoInput)
{
    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);

    attributes.get_input_0()
        ->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({4, 1})
        .set_stride({1, 1});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({4})
        .set_stride({1});

    const auto outputTensor = attributes.get_output_0();
    outputTensor->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_is_virtual(false);

    const GraphAttributes graphAttributes;
    PointwiseNode node(std::move(attributes), graphAttributes);

    const auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
}

namespace
{
class TestPointwiseNodeCreateOperation : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;
    std::array<char, 8> _fakeDescriptors{};
    size_t _nextDescriptor = 0;

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
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }
};
} // namespace

TEST_F(TestPointwiseNodeCreateOperation, IdentityWithTwoInputsLeavesSurplusInputOut)
{
    using ::testing::_;
    using ::testing::AnyNumber;

    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::IDENTITY);
    attributes.set_compute_data_type(DataType::FLOAT);

    attributes.get_input_0()
        ->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});
    attributes.get_output_0()
        ->set_uid(3)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, _, _, _))
        .Times(0);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    std::unordered_map<int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;
    error = node.create_operation(tensorDescs, operations);

    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(tensorDescs.size(), 2u);
    EXPECT_EQ(operations.size(), 1u);
    EXPECT_NE(node.attributes.get_input_1(), nullptr);
}

TEST_F(TestPointwiseNodeCreateOperation, AddWithThreeInputsLeavesSurplusInputOut)
{
    using ::testing::_;
    using ::testing::AnyNumber;

    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::ADD);
    attributes.set_compute_data_type(DataType::FLOAT);

    attributes.get_input_0()
        ->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});
    attributes.get_input_2()
        ->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});
    attributes.get_output_0()
        ->set_uid(4)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3, 4})
        .set_stride({12, 4, 1});

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, _, _, _))
        .Times(1);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, _, _, _))
        .Times(0);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    std::unordered_map<int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;
    error = node.create_operation(tensorDescs, operations);

    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(tensorDescs.size(), 3u);
    EXPECT_EQ(operations.size(), 1u);
    EXPECT_NE(node.attributes.get_input_2(), nullptr);
}

TEST_F(TestPointwiseNodeCreateOperation, IdentityWithThreeInputsLeavesBothSurplusInputsOut)
{
    using ::testing::_;
    using ::testing::AnyNumber;

    PointwiseAttributes attributes;
    attributes.set_input_0(std::make_shared<TensorAttributes>());
    attributes.set_input_1(std::make_shared<TensorAttributes>());
    attributes.set_input_2(std::make_shared<TensorAttributes>());
    attributes.set_output_0(std::make_shared<TensorAttributes>());
    attributes.set_mode(PointwiseMode::IDENTITY);
    attributes.set_compute_data_type(DataType::FLOAT);

    attributes.get_input_0()
        ->set_uid(1)
        .set_name("InputTensor0")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3})
        .set_stride({3, 1});
    attributes.get_input_1()
        ->set_uid(2)
        .set_name("InputTensor1")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3})
        .set_stride({3, 1});
    attributes.get_input_2()
        ->set_uid(3)
        .set_name("InputTensor2")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3})
        .set_stride({3, 1});
    attributes.get_output_0()
        ->set_uid(4)
        .set_name("OutputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 3})
        .set_stride({3, 1});

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, _, _, _))
        .Times(0);
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT, _, _, _))
        .Times(0);

    const GraphAttributes graphAttributes;
    const PointwiseNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    std::unordered_map<int64_t, hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor> operations;
    error = node.create_operation(tensorDescs, operations);

    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(tensorDescs.size(), 2u);
    EXPECT_EQ(operations.size(), 1u);
    EXPECT_NE(node.attributes.get_input_1(), nullptr);
    EXPECT_NE(node.attributes.get_input_2(), nullptr);
}
