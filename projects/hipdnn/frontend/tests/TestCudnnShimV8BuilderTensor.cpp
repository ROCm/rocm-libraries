// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// TensorBuilder rules, read through Tensor getters and the hipDNN tensors a graph build lowers to.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderFixture.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace
{

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::buildFailure;
using hipdnn_shim_test::Dims;
using hipdnn_shim_test::TensorSpec;
namespace v8fe = hipdnn_shim_test::v8fe;

using v8fe::DataType_t;
using v8fe::DescriptorType_t;
using v8fe::PointwiseMode_t;
using v8fe::TensorReordering_t;

const Dims X_DIMS = {2, 3, 4, 5};
const Dims X_STRIDES = {60, 20, 5, 1};

Dims dimsOf(const v8fe::Tensor& tensor)
{
    return {tensor.getDim(), tensor.getDim() + tensor.getDimCount()};
}

Dims stridesOf(const v8fe::Tensor& tensor)
{
    return {tensor.getStride(), tensor.getStride() + tensor.getDimCount()};
}

// An enumerator value the enum does not declare.
template <typename Enum>
Enum undeclared(int value)
{
    return static_cast<Enum>(value);
}

struct StatusCase
{
    const char* label;
    std::function<BuildFailure()> build;
    cudnnStatus_t expected;
};

class TestCudnnShimV8BuilderTensor : public hipdnn_shim_test::ShimV8BuilderFixture
{
protected:
    // Every field build() needs, on X_DIMS with packed strides.
    static v8fe::TensorBuilder& complete(v8fe::TensorBuilder& builder, std::int64_t uid = 1)
    {
        return describe(builder, TensorSpec{uid, X_DIMS});
    }

    // Lowers y = x + b in a one-op graph.
    void lowerAdd(const v8fe::Tensor& b)
    {
        const v8fe::Tensor x = tensor(1, X_DIMS);
        const v8fe::Tensor y = tensor(3, X_DIMS);
        const v8fe::Operation add = pointwise(pointwiseDesc(PointwiseMode_t::ADD), x, y, &b);
        [[maybe_unused]] const v8fe::OperationGraph built = graph({&add});
    }

    template <typename T>
    static v8fe::Tensor constant(std::int64_t uid, DataType_t dataType, const T& value)
    {
        v8fe::TensorBuilder builder;
        describe(builder, TensorSpec{uid, {1, 1, 1, 1}});
        return builder.setDataType(dataType).setConstValue(value).build();
    }

    std::optional<hipdnn_shim_test::CapturedSetAttribute>
        loweredField(std::int64_t uid, hipdnnBackendAttributeName_t name) const
    {
        return callOn(hipdnnTensor(uid), name);
    }
};

TEST_F(TestCudnnShimV8BuilderTensor, GettersReportBuilderFields)
{
    const v8fe::Tensor fromC = v8fe::TensorBuilder()
                                   .setDataType(CUDNN_DATA_HALF)
                                   .setDim(4, X_DIMS.data())
                                   .setStride(4, X_STRIDES.data())
                                   .setId(7)
                                   .setAlignment(16)
                                   .setVirtual()
                                   .build();
    const v8fe::Tensor fromFe = v8fe::TensorBuilder()
                                    .setDataType(DataType_t::HALF)
                                    .setDim(4, X_DIMS.data())
                                    .setStrides(4, X_STRIDES.data())
                                    .setId(8)
                                    .setAlignment(32)
                                    .setByValue()
                                    .build();

    EXPECT_EQ(fromC.getDataType(), static_cast<std::int64_t>(DataType_t::HALF));
    EXPECT_EQ(fromFe.getDataType(), fromC.getDataType());
    EXPECT_EQ(fromC.getDimCount(), 4);
    EXPECT_EQ(dimsOf(fromC), X_DIMS);
    EXPECT_EQ(stridesOf(fromC), X_STRIDES);
    EXPECT_EQ(stridesOf(fromFe), X_STRIDES);
    EXPECT_EQ(fromC.getId(), 7);
    EXPECT_EQ(fromFe.getId(), 8);
    EXPECT_EQ(fromC.getAlignment(), 16);
    EXPECT_EQ(fromFe.getAlignment(), 32);
    EXPECT_TRUE(fromC.isVirtualTensor());
    EXPECT_FALSE(fromFe.isVirtualTensor());
    EXPECT_EQ(fromC.getPackedElementCount(), 2 * 3 * 4 * 5);
    EXPECT_EQ(fromC.get_status(), CUDNN_STATUS_SUCCESS);
}

TEST_F(TestCudnnShimV8BuilderTensor, LoweringCarriesTensorFields)
{
    TensorSpec inputSpec{1, X_DIMS};
    inputSpec.dataType = DataType_t::HALF;
    TensorSpec virtualSpec{2, X_DIMS, true};
    virtualSpec.dataType = DataType_t::HALF;
    TensorSpec outputSpec{3, X_DIMS};
    outputSpec.dataType = DataType_t::HALF;
    const v8fe::Tensor x = tensor(inputSpec);
    const v8fe::Tensor v = tensor(virtualSpec);
    const v8fe::Tensor y = tensor(outputSpec);
    const v8fe::Operation first = pointwise(pointwiseDesc(PointwiseMode_t::RELU_FWD), x, v);
    const v8fe::Operation second = pointwise(pointwiseDesc(PointwiseMode_t::RELU_FWD), v, y);
    [[maybe_unused]] const v8fe::OperationGraph built = graph({&first, &second});

    for(const std::int64_t uid : {1, 2, 3})
    {
        SCOPED_TRACE(uid);
        ASSERT_TRUE(loweredField(uid, HIPDNN_ATTR_TENSOR_DATA_TYPE));
        EXPECT_EQ(loweredField(uid, HIPDNN_ATTR_TENSOR_DATA_TYPE)->value<hipdnnDataType_t>(),
                  HIPDNN_DATA_HALF);
        ASSERT_TRUE(loweredField(uid, HIPDNN_ATTR_TENSOR_DIMENSIONS));
        EXPECT_EQ(loweredField(uid, HIPDNN_ATTR_TENSOR_DIMENSIONS)->values<std::int64_t>(), X_DIMS);
        ASSERT_TRUE(loweredField(uid, HIPDNN_ATTR_TENSOR_STRIDES));
        EXPECT_EQ(loweredField(uid, HIPDNN_ATTR_TENSOR_STRIDES)->values<std::int64_t>(), X_STRIDES);
        ASSERT_TRUE(loweredField(uid, HIPDNN_ATTR_TENSOR_IS_VIRTUAL));
        EXPECT_EQ(loweredField(uid, HIPDNN_ATTR_TENSOR_IS_VIRTUAL)->value<bool>(), uid == 2);
        EXPECT_FALSE(loweredField(uid, HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT));
        EXPECT_FALSE(loweredField(uid, HIPDNN_ATTR_TENSOR_VALUE_EXT));
        EXPECT_FALSE(loweredField(uid, HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC));
        EXPECT_FALSE(loweredField(uid, HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT));
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, AlignmentForwardedOnlyBelowSixteen)
{
    TensorSpec wideSpec{1, X_DIMS};
    wideSpec.alignment = 32;
    TensorSpec narrowSpec{2, X_DIMS};
    narrowSpec.alignment = 4;
    const v8fe::Tensor wide = tensor(wideSpec);
    const v8fe::Tensor narrow = tensor(narrowSpec);
    const v8fe::Tensor exact = tensor(3, X_DIMS);
    const v8fe::Operation add
        = pointwise(pointwiseDesc(PointwiseMode_t::ADD), wide, exact, &narrow);
    [[maybe_unused]] const v8fe::OperationGraph built = graph({&add});

    const auto alignments = callsNamed(HIPDNN_ATTR_TENSOR_BYTE_ALIGNMENT);
    ASSERT_EQ(alignments.size(), 1U);
    EXPECT_EQ(alignments.front().value<std::int64_t>(), 4);
    EXPECT_EQ(alignments.front().descriptor, hipdnnTensor(2));
    // The builder still reports the cuDNN value.
    EXPECT_EQ(wide.getAlignment(), 32);
}

TEST_F(TestCudnnShimV8BuilderTensor, IdAndAlignmentBoundsEnforced)
{
    const auto withIdAndAlignment = [](std::int64_t uid, std::int64_t alignment) {
        return [uid, alignment] {
            v8fe::TensorBuilder builder;
            complete(builder, uid).setAlignment(alignment);
            return buildFailure(builder);
        };
    };
    const std::vector<StatusCase> cases = {
        {"negative UID", withIdAndAlignment(-1, 16), CUDNN_STATUS_BAD_PARAM},
        {"zero alignment", withIdAndAlignment(1, 0), CUDNN_STATUS_BAD_PARAM},
        {"negative alignment", withIdAndAlignment(1, -8), CUDNN_STATUS_BAD_PARAM},
        {"zero UID", withIdAndAlignment(0, 16), CUDNN_STATUS_SUCCESS},
        {"alignment one", withIdAndAlignment(1, 1), CUDNN_STATUS_SUCCESS},
    };
    for(const StatusCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        const BuildFailure failure = item.build();
        EXPECT_EQ(failure.thrown, item.expected != CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(failure.status, item.expected);
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, MissingRequiredFieldRejected)
{
    const auto missing = [](const std::function<void(v8fe::TensorBuilder&)>& fill) {
        return [fill] {
            v8fe::TensorBuilder builder;
            fill(builder);
            return buildFailure(builder);
        };
    };
    const std::vector<StatusCase> cases = {
        {"data type",
         missing([](v8fe::TensorBuilder& b) {
             b.setDim(4, X_DIMS.data()).setStride(4, X_STRIDES.data()).setId(1).setAlignment(16);
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"dimensions",
         missing([](v8fe::TensorBuilder& b) {
             b.setDataType(DataType_t::FLOAT)
                 .setStride(4, X_STRIDES.data())
                 .setId(1)
                 .setAlignment(16);
         }),
         CUDNN_STATUS_BAD_PARAM},
        {"strides",
         missing([](v8fe::TensorBuilder& b) {
             b.setDataType(DataType_t::FLOAT).setDim(4, X_DIMS.data()).setId(1).setAlignment(16);
         }),
         CUDNN_STATUS_BAD_PARAM},
        {"UID",
         missing([](v8fe::TensorBuilder& b) {
             b.setDataType(DataType_t::FLOAT)
                 .setDim(4, X_DIMS.data())
                 .setStride(4, X_STRIDES.data())
                 .setAlignment(16);
         }),
         CUDNN_STATUS_BAD_PARAM},
        {"alignment",
         missing([](v8fe::TensorBuilder& b) {
             b.setDataType(DataType_t::FLOAT)
                 .setDim(4, X_DIMS.data())
                 .setStride(4, X_STRIDES.data())
                 .setId(1);
         }),
         CUDNN_STATUS_BAD_PARAM},
    };
    for(const StatusCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        const BuildFailure failure = item.build();
        EXPECT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, item.expected);
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, ShapeRulesEnforced)
{
    const auto shaped = [](const Dims& dims, const Dims& strides) {
        return [dims, strides] {
            v8fe::TensorBuilder builder;
            complete(builder)
                .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
                .setStride(static_cast<std::int64_t>(strides.size()), strides.data());
            return buildFailure(builder);
        };
    };
    const std::vector<StatusCase> cases = {
        {"zero inner dimension", shaped({2, 0, 4, 5}, X_STRIDES), CUDNN_STATUS_BAD_PARAM},
        {"zero outer dimension", shaped({0, 3, 4, 5}, X_STRIDES), CUDNN_STATUS_BAD_PARAM},
        {"negative inner dimension", shaped({2, 3, -4, 5}, X_STRIDES), CUDNN_STATUS_BAD_PARAM},
        {"negative inner stride", shaped(X_DIMS, {60, 20, -5, 1}), CUDNN_STATUS_BAD_PARAM},
        {"negative outer stride", shaped(X_DIMS, {-60, 20, 5, 1}), CUDNN_STATUS_BAD_PARAM},
        {"zero stride", shaped(X_DIMS, {60, 20, 0, 1}), CUDNN_STATUS_SUCCESS},
    };
    for(const StatusCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        const BuildFailure failure = item.build();
        EXPECT_EQ(failure.thrown, item.expected != CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(failure.status, item.expected);
    }

    // An empty shape never reaches hipDNN. The builder already holds X_DIMS / X_STRIDES, so the
    // shim's count-0 rule, not upstream's first-element precheck, is what rejects it.
    v8fe::TensorBuilder empty;
    complete(empty).setDim(0, X_DIMS.data()).setStride(0, X_STRIDES.data());
    const BuildFailure failure = buildFailure(empty);
    EXPECT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCudnnShimV8BuilderTensor, VectorCountOtherThanOneNotSupported)
{
    for(const std::int64_t count : {2, 4, 32})
    {
        SCOPED_TRACE(count);
        v8fe::TensorBuilder builder;
        complete(builder).setVectorCountAndDimension(count, 1);
        const BuildFailure failure = buildFailure(builder);
        EXPECT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, ReorderingOtherThanNoneNotSupported)
{
    const std::vector<StatusCase> cases = {
        {"INT8x32",
         [] {
             v8fe::TensorBuilder builder;
             complete(builder).setReorderType(TensorReordering_t::INT8x32);
             return buildFailure(builder);
         },
         CUDNN_STATUS_NOT_SUPPORTED},
        {"F16x16",
         [] {
             v8fe::TensorBuilder builder;
             complete(builder).setReorderType(TensorReordering_t::F16x16);
             return buildFailure(builder);
         },
         CUDNN_STATUS_NOT_SUPPORTED},
        {"C INT8x32",
         [] {
             v8fe::TensorBuilder builder;
             complete(builder).setReorderType(CUDNN_TENSOR_REORDERING_INT8x32);
             return buildFailure(builder);
         },
         CUDNN_STATUS_NOT_SUPPORTED},
    };
    for(const StatusCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        const BuildFailure failure = item.build();
        EXPECT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, item.expected);
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, ExplicitDefaultVectorCountAndReorderingAccepted)
{
    v8fe::TensorBuilder feBuilder;
    complete(feBuilder, 1)
        .setVectorCountAndDimension(1, 1)
        .setReorderType(TensorReordering_t::NONE);
    v8fe::TensorBuilder cBuilder;
    complete(cBuilder, 2)
        .setVectorCountAndDimension(1, 0)
        .setReorderType(CUDNN_TENSOR_REORDERING_NONE);

    const BuildFailure feFailure = buildFailure(feBuilder);
    EXPECT_FALSE(feFailure.thrown) << feFailure.what;
    const BuildFailure cFailure = buildFailure(cBuilder);
    EXPECT_FALSE(cFailure.thrown) << cFailure.what;
}

TEST_F(TestCudnnShimV8BuilderTensor, UnlowerableDataTypeNotSupported)
{
    v8fe::TensorBuilder feBuilder;
    complete(feBuilder).setDataType(DataType_t::INT8x4);
    v8fe::TensorBuilder cBuilder;
    complete(cBuilder).setDataType(CUDNN_DATA_INT8x4);

    for(v8fe::TensorBuilder* builder : {&feBuilder, &cBuilder})
    {
        const BuildFailure failure = buildFailure(*builder);
        EXPECT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, ByValueLowersToRuntimePassByValue)
{
    lowerAdd(scalar(2));

    const auto byValue = callsNamed(HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT);
    ASSERT_EQ(byValue.size(), 1U);
    EXPECT_TRUE(byValue.front().value<bool>());
    EXPECT_EQ(byValue.front().descriptor, hipdnnTensor(2));
    EXPECT_FALSE(loweredField(2, HIPDNN_ATTR_TENSOR_VALUE_EXT));
}

TEST_F(TestCudnnShimV8BuilderTensor, ConstantValueLowersAsValueNotRuntimeByValue)
{
    // setConstValue implies setByValue; an explicit setByValue changes nothing.
    for(const bool explicitByValue : {false, true})
    {
        SCOPED_TRACE(explicitByValue);
        v8fe::TensorBuilder builder;
        describe(builder, TensorSpec{2, {1, 1, 1, 1}});
        builder.setByValue(explicitByValue).setConstValue(2.5F);
        lowerAdd(builder.build());

        const auto value = loweredField(2, HIPDNN_ATTR_TENSOR_VALUE_EXT);
        ASSERT_TRUE(value);
        EXPECT_EQ(value->values<float>(), (std::vector<float>{2.5F}));
        EXPECT_EQ(loweredField(2, HIPDNN_ATTR_TENSOR_DATA_TYPE)->value<hipdnnDataType_t>(),
                  HIPDNN_DATA_FLOAT);
        EXPECT_FALSE(loweredField(2, HIPDNN_ATTR_TENSOR_IS_RUNTIME_PASS_BY_VALUE_EXT));
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, ConstantValueCopiedBeforeSourceIsFreed)
{
    std::optional<v8fe::Operation> add;
    {
        auto source = std::make_unique<double>(-3.25);
        const std::array<std::int64_t, 1> one = {1};
        v8fe::TensorBuilder builder;
        builder.setDataType(DataType_t::DOUBLE)
            .setDim(1, one.data())
            .setStride(1, one.data())
            .setId(2)
            .setAlignment(8)
            .setConstValue(*source);
        source.reset();
        const v8fe::Tensor constantTensor = builder.build();
        add.emplace(pointwise(pointwiseDesc(PointwiseMode_t::ADD),
                              tensor(1, X_DIMS),
                              tensor(3, X_DIMS),
                              &constantTensor));
    }
    // The builder, the Tensor and the caller's value are all gone before lowering.
    [[maybe_unused]] const v8fe::OperationGraph built = graph({&*add});

    const auto value = loweredField(2, HIPDNN_ATTR_TENSOR_VALUE_EXT);
    ASSERT_TRUE(value);
    EXPECT_EQ(value->values<double>(), (std::vector<double>{-3.25}));
    EXPECT_EQ(loweredField(2, HIPDNN_ATTR_TENSOR_DATA_TYPE)->value<hipdnnDataType_t>(),
              HIPDNN_DATA_DOUBLE);
}

TEST_F(TestCudnnShimV8BuilderTensor, BooleanConstantLowersAsOneByte)
{
    lowerAdd(constant(2, DataType_t::BOOLEAN, false));
    const auto falseValue = loweredField(2, HIPDNN_ATTR_TENSOR_VALUE_EXT);
    ASSERT_TRUE(falseValue);
    EXPECT_EQ(falseValue->values<std::uint8_t>(), (std::vector<std::uint8_t>{0}));

    lowerAdd(constant(4, DataType_t::BOOLEAN, true));
    const auto trueValue = loweredField(4, HIPDNN_ATTR_TENSOR_VALUE_EXT);
    ASSERT_TRUE(trueValue);
    EXPECT_EQ(trueValue->values<std::uint8_t>(), (std::vector<std::uint8_t>{1}));
    EXPECT_EQ(loweredField(4, HIPDNN_ATTR_TENSOR_DATA_TYPE)->value<hipdnnDataType_t>(),
              HIPDNN_DATA_BOOLEAN);
}

TEST_F(TestCudnnShimV8BuilderTensor, WiderIntegerConstantLowersNarrowed)
{
    lowerAdd(constant(2, DataType_t::UINT8, 3));
    const auto byteValue = loweredField(2, HIPDNN_ATTR_TENSOR_VALUE_EXT);
    ASSERT_TRUE(byteValue);
    EXPECT_EQ(byteValue->values<std::uint8_t>(), (std::vector<std::uint8_t>{3}));

    lowerAdd(constant(4, DataType_t::INT32, std::int64_t{-5}));
    const auto intValue = loweredField(4, HIPDNN_ATTR_TENSOR_VALUE_EXT);
    ASSERT_TRUE(intValue);
    EXPECT_EQ(intValue->value<std::int32_t>(), -5);
}

TEST_F(TestCudnnShimV8BuilderTensor, ConstantSizeMismatchRejected)
{
    // A narrower constant would be read past its end; a wider one would be read as its leading
    // bytes, which are its value only for an integer that an integer data type holds.
    const auto build = [](DataType_t dataType, const auto& value) {
        v8fe::TensorBuilder builder;
        describe(builder, TensorSpec{2, {1, 1, 1, 1}});
        builder.setDataType(dataType).setConstValue(value);
        return buildFailure(builder);
    };
    struct Case
    {
        const char* label;
        BuildFailure failure;
        cudnnStatus_t expected;
    };
    const std::vector<Case> cases = {
        {"INT64 from int32", build(DataType_t::INT64, std::int32_t{5}), CUDNN_STATUS_BAD_PARAM},
        {"DOUBLE from float", build(DataType_t::DOUBLE, 1.0F), CUDNN_STATUS_BAD_PARAM},
        {"FLOAT from double", build(DataType_t::FLOAT, 1.0), CUDNN_STATUS_BAD_PARAM},
        {"BOOLEAN from 4 bytes",
         build(DataType_t::BOOLEAN, std::array<std::uint8_t, 4>{0, 0x7F, 0x7F, 0x7F}),
         CUDNN_STATUS_BAD_PARAM},
        {"INT64 from int64", build(DataType_t::INT64, std::int64_t{5}), CUDNN_STATUS_SUCCESS},
        {"UINT8 from int", build(DataType_t::UINT8, 3), CUDNN_STATUS_SUCCESS},
        {"BOOLEAN from int", build(DataType_t::BOOLEAN, 1), CUDNN_STATUS_SUCCESS},
        {"INT32 from int64", build(DataType_t::INT32, std::int64_t{5}), CUDNN_STATUS_SUCCESS},
        {"UINT8 from out-of-range int", build(DataType_t::UINT8, 256), CUDNN_STATUS_BAD_PARAM},
        {"UINT8 from negative int", build(DataType_t::UINT8, -1), CUDNN_STATUS_BAD_PARAM},
        {"BOOLEAN from int 2", build(DataType_t::BOOLEAN, 2), CUDNN_STATUS_BAD_PARAM},
        {"FLOAT from int64", build(DataType_t::FLOAT, std::int64_t{5}), CUDNN_STATUS_BAD_PARAM},
    };
    for(const Case& item : cases)
    {
        SCOPED_TRACE(item.label);
        EXPECT_EQ(item.failure.thrown, item.expected != CUDNN_STATUS_SUCCESS) << item.failure.what;
        EXPECT_EQ(item.failure.status, item.expected) << item.failure.what;
        if(item.expected != CUDNN_STATUS_SUCCESS)
        {
            EXPECT_NE(item.failure.what.find("TensorBuilder: constant value"), std::string::npos)
                << item.failure.what;
        }
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, VirtualTensorWithValueRejected)
{
    v8fe::TensorBuilder byValue;
    complete(byValue).setVirtual().setByValue();
    v8fe::TensorBuilder constantValue;
    complete(constantValue).setVirtual().setConstValue(1.0F);
    for(v8fe::TensorBuilder* builder : {&byValue, &constantValue})
    {
        const BuildFailure failure = buildFailure(*builder);
        EXPECT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM) << failure.what;
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, RaggedOffsetLowersAsLinkedTensor)
{
    TensorSpec offsetSpec{9, {3, 1, 1, 1}};
    offsetSpec.dataType = DataType_t::INT32;
    auto offsets = std::make_shared<v8fe::Tensor>(tensor(offsetSpec));
    v8fe::TensorBuilder builder;
    complete(builder, 1).setRaggedOffset(offsets);
    const v8fe::Tensor x = builder.build();
    const v8fe::Tensor y = tensor(2, X_DIMS);
    const v8fe::Operation abs = pointwise(pointwiseDesc(PointwiseMode_t::ABS), x, y);
    [[maybe_unused]] const v8fe::OperationGraph built = graph({&abs});

    const auto ragged = loweredField(1, HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC);
    ASSERT_TRUE(ragged);
    EXPECT_EQ(ragged->value<hipdnnBackendDescriptor_t>(), hipdnnTensor(9));
    EXPECT_FALSE(loweredField(2, HIPDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC));
}

TEST_F(TestCudnnShimV8BuilderTensor, DescribeMatchesUpstream)
{
    const std::array<std::int64_t, 2> dims = {1, 2};
    const std::array<std::int64_t, 2> strides = {2, 1};
    auto offsets = std::make_shared<v8fe::Tensor>(tensor(9, {1, 1}, false, DataType_t::INT32));
    const v8fe::Tensor built = v8fe::TensorBuilder()
                                   .setDataType(DataType_t::FLOAT)
                                   .setDim(2, dims.data())
                                   .setStride(2, strides.data())
                                   .setId(4)
                                   .setAlignment(16)
                                   .setRaggedOffset(offsets)
                                   .build();

    const std::string text = built.describe();
    EXPECT_NE(text.find("CUDNN_BACKEND_TENSOR_DESCRIPTOR :"), std::string::npos) << text;
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    EXPECT_NE(text.find("Datatype: [\"FLOAT\"]"), std::string::npos) << text;
#else
    EXPECT_NE(text.find("Datatype: " + std::to_string(static_cast<int>(DataType_t::FLOAT))),
              std::string::npos)
        << text;
#endif
    EXPECT_NE(text.find("Id: 4"), std::string::npos) << text;
    EXPECT_NE(text.find("Dim [ 1,2 ]"), std::string::npos) << text;
    EXPECT_NE(text.find("Str [ 2,1 ]"), std::string::npos) << text;
    EXPECT_NE(text.find("Alignment: 16"), std::string::npos) << text;
    EXPECT_NE(text.find("raggedOffset: Enabled UID: 9"), std::string::npos) << text;
}

TEST_F(TestCudnnShimV8BuilderTensor, UndeclaredEnumValuesRejected)
{
    const auto tensorWith = [](const std::function<void(v8fe::TensorBuilder&)>& set) {
        return [set] {
            v8fe::TensorBuilder builder;
            complete(builder);
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto pointwiseWith = [](const std::function<void(v8fe::PointWiseDescBuilder&)>& set) {
        return [set] {
            v8fe::PointWiseDescBuilder builder;
            builder.setMode(PointwiseMode_t::RELU_FWD).setComputeType(DataType_t::FLOAT);
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto convWith = [](const std::function<void(v8fe::ConvDescBuilder&)>& set) {
        return [set] {
            const std::array<std::int64_t, 2> ones = {1, 1};
            v8fe::ConvDescBuilder builder;
            builder.setComputeType(DataType_t::FLOAT)
                .setMathMode(CUDNN_CROSS_CORRELATION)
                .setSpatialDimCount(2)
                .setPrePadding(2, ones.data())
                .setPostPadding(2, ones.data())
                .setDilation(2, ones.data())
                .setSpatialStride(2, ones.data());
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto resampleWith = [](const std::function<void(v8fe::ResampleDescBuilder_v8&)>& set) {
        return [set] {
            const std::array<std::int64_t, 2> window = {2, 2};
            const std::array<std::int64_t, 2> zeros = {0, 0};
            v8fe::ResampleDescBuilder_v8 builder;
            builder.setResampleMode(v8fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
                .setComputeType(DataType_t::FLOAT)
                .setPaddingMode(v8fe::PaddingMode_t::ZERO_PAD)
                .setSpatialDim(2, window.data())
                .setPrePadding(2, zeros.data())
                .setPostPadding(2, zeros.data())
                .setSpatialStride(2, window.data());
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto reductionWith = [](const std::function<void(v8fe::ReductionDescBuilder&)>& set) {
        return [set] {
            v8fe::ReductionDescBuilder builder;
            builder.setComputeType(DataType_t::FLOAT).setReductionOp(CUDNN_REDUCE_TENSOR_ADD);
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto normWith = [](const std::function<void(v8fe::OperationBuilder&)>& set) {
        return [set] {
            v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
            builder.setxDesc(tensor(1, X_DIMS))
                .setNormalizationMode(v8fe::NormMode_t::LAYER_NORM)
                .setNormFwdPhase(v8fe::NormFwdPhase_t::INFERENCE);
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto bnFinalizeWith = [](const std::function<void(v8fe::OperationBuilder&)>& set) {
        return [set] {
            const Dims channel = {1, 3, 1, 1};
            v8fe::OperationBuilder builder(
                DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
            builder.setSumDesc(tensor(1, channel))
                .setSqSumDesc(tensor(2, channel))
                .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
                .setMathPrecision(CUDNN_DATA_FLOAT);
            set(builder);
            return buildFailure(builder);
        };
    };
    const auto heuristicsWith = [this](cudnnBackendHeurMode_t mode) {
        return [this, mode] {
            const v8fe::Tensor x = tensor(1, X_DIMS);
            const v8fe::Tensor y = tensor(2, X_DIMS);
            const v8fe::Operation abs = pointwise(pointwiseDesc(PointwiseMode_t::ABS), x, y);
            v8fe::OperationGraph opGraph = graph({&abs});
            v8fe::EngineHeuristicsBuilder builder;
            builder.setOperationGraph(opGraph).setHeurMode(mode);
            return buildFailure(builder);
        };
    };

    // Converting setters map an undeclared C value to NOT_SET, which build() rejects.
    const std::vector<StatusCase> cases = {
        {"tensor C data type",
         tensorWith(
             [](v8fe::TensorBuilder& b) { b.setDataType(undeclared<cudnnDataType_t>(999)); }),
         CUDNN_STATUS_INVALID_VALUE},
        {"tensor data type",
         tensorWith([](v8fe::TensorBuilder& b) { b.setDataType(undeclared<DataType_t>(999)); }),
         CUDNN_STATUS_INVALID_VALUE},
        {"tensor reordering",
         tensorWith(
             [](v8fe::TensorBuilder& b) { b.setReorderType(undeclared<TensorReordering_t>(999)); }),
         CUDNN_STATUS_INVALID_VALUE},
        // Upstream's converter falls back to NONE for an unknown C reordering mode.
        {"tensor C reordering",
         tensorWith([](v8fe::TensorBuilder& b) {
             b.setReorderType(
                 undeclared<cudnnBackendTensorReordering_t>(CUDNN_TENSOR_REORDERING_F8_128x4 + 1));
         }),
         CUDNN_STATUS_SUCCESS},
        {"pointwise C mode",
         pointwiseWith([](v8fe::PointWiseDescBuilder& b) {
             b.setMode(undeclared<cudnnPointwiseMode_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"pointwise mode",
         pointwiseWith(
             [](v8fe::PointWiseDescBuilder& b) { b.setMode(undeclared<PointwiseMode_t>(999)); }),
         CUDNN_STATUS_INVALID_VALUE},
        {"pointwise C compute type",
         pointwiseWith([](v8fe::PointWiseDescBuilder& b) {
             b.setComputeType(undeclared<cudnnDataType_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"convolution C compute type",
         convWith(
             [](v8fe::ConvDescBuilder& b) { b.setComputeType(undeclared<cudnnDataType_t>(999)); }),
         CUDNN_STATUS_INVALID_VALUE},
        // Upstream's converter falls back to CROSS_CORRELATION for an unknown mode.
        {"convolution C mode",
         convWith([](v8fe::ConvDescBuilder& b) {
             b.setMathMode(undeclared<cudnnConvolutionMode_t>(CUDNN_CROSS_CORRELATION + 1));
         }),
         CUDNN_STATUS_SUCCESS},
        {"matmul C compute type",
         [] {
             v8fe::MatMulDescBuilder builder;
             builder.setComputeType(undeclared<cudnnDataType_t>(999));
             return buildFailure(builder);
         },
         CUDNN_STATUS_INVALID_VALUE},
        {"reduction C operator",
         reductionWith([](v8fe::ReductionDescBuilder& b) {
             b.setReductionOp(undeclared<cudnnReduceTensorOp_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"reduction C compute type",
         reductionWith([](v8fe::ReductionDescBuilder& b) {
             b.setComputeType(undeclared<cudnnDataType_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"resample C mode",
         resampleWith([](v8fe::ResampleDescBuilder_v8& b) {
             b.setResampleMode(undeclared<cudnnResampleMode_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"resample C padding mode",
         resampleWith([](v8fe::ResampleDescBuilder_v8& b) {
             b.setPaddingMode(undeclared<cudnnPaddingMode_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"resample C compute type",
         resampleWith([](v8fe::ResampleDescBuilder_v8& b) {
             b.setComputeType(undeclared<cudnnDataType_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"RNG C distribution",
         [] {
             v8fe::RngDescBuilder builder;
             builder.setRngDistribution(undeclared<cudnnRngDistribution_t>(999));
             return buildFailure(builder);
         },
         CUDNN_STATUS_INVALID_VALUE},
        {"norm C mode",
         normWith([](v8fe::OperationBuilder& b) {
             b.setNormalizationMode(undeclared<cudnnBackendNormMode_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
        {"norm C forward phase",
         normWith([](v8fe::OperationBuilder& b) {
             b.setNormFwdPhase(undeclared<cudnnBackendNormFwdPhase_t>(999));
         }),
         CUDNN_STATUS_INVALID_VALUE},
    };
    for(const StatusCase& item : cases)
    {
        SCOPED_TRACE(item.label);
        const BuildFailure failure = item.build();
        EXPECT_EQ(failure.thrown, item.expected != CUDNN_STATUS_SUCCESS) << failure.what;
        EXPECT_EQ(failure.status, item.expected) << failure.what;
    }

    const auto genStatsWith = [](cudnnGenStatsMode_t mode, cudnnDataType_t precision) {
        const Dims channel = {1, 3, 1, 1};
        v8fe::OperationBuilder builder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);
        builder.setxDesc(tensor(1, X_DIMS))
            .setSumDesc(tensor(2, channel))
            .setSqSumDesc(tensor(3, channel))
            .setGenStatsMode(mode)
            .setMathPrecision(precision);
        return buildFailure(builder);
    };
    // An engine that reports TILE_SIZE, chosen, plus an appended knob of `type`, so the
    // appended choice is sent.
    const auto knobTypeWith = [this](cudnnBackendKnobType_t type) {
        constexpr std::int64_t ENGINE_ID = 7;
        installHeuristicResults({ENGINE_ID});
        installEngineKnobs(ENGINE_ID, {{"prov.tile_size", 8, 0, 64, 8, {}}});
        const v8fe::Tensor x = tensor(1, X_DIMS);
        const v8fe::Tensor y = tensor(2, X_DIMS);
        const v8fe::Operation abs = pointwise(pointwiseDesc(PointwiseMode_t::ABS), x, y);
        const v8fe::OperationGraph opGraph = graph({&abs});
        v8fe::EngineBuilder engineBuilder;
        engineBuilder.setOperationGraph(opGraph).setGlobalEngineIdx(0);
        v8fe::Engine engine = engineBuilder.build();
        auto& knobs = engine.getSupportedKnobs();
        EXPECT_EQ(knobs.size(), 1U);
        for(auto& knob : knobs)
        {
            knob.setChoice(8);
        }
        using Knob = std::remove_reference_t<decltype(knobs.front())>;
        Knob appended(type, 64, 0, 1);
        appended.setChoice(8);
        knobs.push_back(appended);
        return hipdnn_shim_test::failureOf([&engine] {
            [[maybe_unused]] auto config = v8fe::EngineConfigBuilder().setEngine(engine).build();
        });
    };

    // Setters that pass a raw C enum through: the range check rejects every value outside
    // [0, last] with BAD_PARAM, and `last` goes on to the next rule (`lastStatus`).
    struct RawCase
    {
        const char* label;
        std::function<BuildFailure(int)> build;
        int last;
        cudnnStatus_t lastStatus;
    };
    const std::vector<RawCase> rawCases = {
        {"pointwise NaN propagation",
         [&pointwiseWith](int value) {
             return pointwiseWith([value](v8fe::PointWiseDescBuilder& b) {
                 b.setMode(undeclared<cudnnNanPropagation_t>(value));
             })();
         },
         CUDNN_PROPAGATE_NAN,
         CUDNN_STATUS_SUCCESS},
        {"resample NaN propagation",
         [&resampleWith](int value) {
             return resampleWith([value](v8fe::ResampleDescBuilder_v8& b) {
                 b.setNanPropagation(undeclared<cudnnNanPropagation_t>(value));
             })();
         },
         CUDNN_PROPAGATE_NAN,
         CUDNN_STATUS_SUCCESS},
        // BN finalize and gen stats have no lowering, so a value in range ends NOT_SUPPORTED.
        {"BN finalize stats mode",
         [&bnFinalizeWith](int value) {
             return bnFinalizeWith([value](v8fe::OperationBuilder& b) {
                 b.setBNFinalizeMode(undeclared<cudnnBnFinalizeStatsMode_t>(value));
             })();
         },
         CUDNN_BN_FINALIZE_STATISTICS_TRAINING,
         CUDNN_STATUS_NOT_SUPPORTED},
        {"BN finalize C math precision",
         [&bnFinalizeWith](int value) {
             return bnFinalizeWith([value](v8fe::OperationBuilder& b) {
                 b.setMathPrecision(undeclared<cudnnDataType_t>(value));
             })();
         },
         CUDNN_DATA_UINT8x4,
         CUDNN_STATUS_NOT_SUPPORTED},
        {"gen stats mode",
         [&genStatsWith](int value) {
             return genStatsWith(undeclared<cudnnGenStatsMode_t>(value), CUDNN_DATA_FLOAT);
         },
         CUDNN_GENSTATS_SUM_SQSUM,
         CUDNN_STATUS_NOT_SUPPORTED},
        {"gen stats C math precision",
         [&genStatsWith](int value) {
             return genStatsWith(CUDNN_GENSTATS_SUM_SQSUM, undeclared<cudnnDataType_t>(value));
         },
         CUDNN_DATA_UINT8x4,
         CUDNN_STATUS_NOT_SUPPORTED},
        {"heuristics mode",
         [&heuristicsWith](int value) {
             return heuristicsWith(undeclared<cudnnBackendHeurMode_t>(value))();
         },
         CUDNN_HEUR_MODES_COUNT - 1,
         CUDNN_STATUS_SUCCESS},
        // A declared but unreported knob type is hidden from this engine: NOT_SUPPORTED.
        {"knob type",
         [&knobTypeWith](int value) {
             return knobTypeWith(undeclared<cudnnBackendKnobType_t>(value));
         },
         CUDNN_KNOB_TYPE_COUNTS - 1,
         CUDNN_STATUS_NOT_SUPPORTED},
    };
    for(const RawCase& item : rawCases)
    {
        for(const int value :
            {-1, item.last + 1, std::numeric_limits<std::int32_t>::max(), item.last})
        {
            SCOPED_TRACE(std::string(item.label) + " = " + std::to_string(value));
            const cudnnStatus_t expected
                = value == item.last ? item.lastStatus : CUDNN_STATUS_BAD_PARAM;
            const BuildFailure failure = item.build(value);
            EXPECT_EQ(failure.thrown, expected != CUDNN_STATUS_SUCCESS) << failure.what;
            EXPECT_EQ(failure.status, expected) << failure.what;
        }
    }
}

TEST_F(TestCudnnShimV8BuilderTensor, RetryAfterFailedPrecheckSucceeds)
{
    // As upstream: a precheck runs before the descriptor exists, so fixing the builder and calling
    // build() again works.
    v8fe::TensorBuilder builder;
    builder.setDataType(DataType_t::FLOAT)
        .setDim(4, X_DIMS.data())
        .setStride(4, X_STRIDES.data())
        .setId(1);
    const BuildFailure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);

    const BuildFailure retry = hipdnn_shim_test::failureOf([&builder] {
        const v8fe::Tensor built = builder.setAlignment(16).build();
        EXPECT_EQ(built.get_status(), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(built.getAlignment(), 16);
        EXPECT_NE(v8fe::detail::v8::recordOf<v8fe::detail::v8::TensorRecord>(built.get_desc()),
                  nullptr);
    });
    EXPECT_FALSE(retry.thrown) << retry.what;
}

TEST_F(TestCudnnShimV8BuilderTensor, RetryAfterFailedCoreCheckSucceeds)
{
    // A failed core check leaves only a placeholder pointer, which does not count as built.
    v8fe::TensorBuilder builder;
    describe(builder, TensorSpec{2, {1, 1, 1, 1}});
    builder.setDataType(DataType_t::INT64).setConstValue(std::int32_t{5});
    const BuildFailure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);

    builder.setConstValue(std::int64_t{5});
    const BuildFailure retry = buildFailure(builder);
    EXPECT_FALSE(retry.thrown) << retry.what;
}

TEST_F(TestCudnnShimV8BuilderTensor, SecondBuildOfAKeptTensorRejected)
{
    v8fe::TensorBuilder builder;
    complete(builder);
    // Binding the result by reference leaves the built tensor in the builder.
    [[maybe_unused]] const v8fe::Tensor& first = builder.build();
    const BuildFailure second = buildFailure(builder);
    EXPECT_TRUE(second.thrown);
    EXPECT_EQ(second.status, CUDNN_STATUS_BAD_PARAM) << second.what;
}

} // namespace
