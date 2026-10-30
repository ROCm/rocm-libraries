// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// OperationBuilder validation per operation kind, driven only through the v0.x builders.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8BuilderFixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace
{

namespace v8fe = hipdnn_shim_test::v8fe;

using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::buildFailure;
using hipdnn_shim_test::Dims;
using hipdnn_shim_test::failureOf;
using hipdnn_shim_test::packedStrides;
using v8fe::DataType_t;
using v8fe::DescriptorType_t;
using v8fe::NormFwdPhase_t;
using v8fe::NormMode_t;
using v8fe::PointwiseMode_t;

using Builder = v8fe::OperationBuilder;
using Setter = std::function<void(Builder&)>;
using TensorSetter = Builder& (Builder::*)(const v8fe::Tensor&);
using TensorPairSetter = Builder& (Builder::*)(const v8fe::Tensor&, const v8fe::Tensor&);

const Dims X_DIMS = {2, 4, 8, 8};
const Dims W_DIMS = {8, 4, 3, 3};
const Dims Y_DIMS = {2, 8, 8, 8};
const Dims BIAS_DIMS = {1, 4, 1, 1};
const Dims POOLED_DIMS = {2, 4, 4, 4};
const Dims NORM_DIMS = {2, 3, 4, 4};
const Dims CHANNEL_DIMS = {1, 3, 1, 1};
const Dims FEATURE_DIMS = {1, 3, 4, 4};
const Dims ROW_DIMS = {2, 1, 1, 1};

// A UID no case uses, for tensors the builder must not forward.
constexpr std::int64_t STRAY_UID = 99;

struct Slot
{
    const char* name;
    Setter set;
};

// One operation kind (and mode class or norm mode/phase): what its builder
// requires, what it optionally accepts, its tag, and what it lowers to.
struct OpCase
{
    const char* label;
    DescriptorType_t kind;
    Setter setup;
    std::vector<Slot> required;
    std::vector<Slot> optional;
    // Null for pointwise, whose tag depends on the JSON build option.
    const char* tag;
    hipdnnBackendDescriptorType_t lowered;
};

// NOLINTNEXTLINE(readability-identifier-naming): gtest finds the printer by this name.
void PrintTo(const OpCase& opCase, std::ostream* out)
{
    *out << opCase.label;
}

void expectStatus(const BuildFailure& failure, cudnnStatus_t expected)
{
    EXPECT_EQ(failure.thrown, expected != CUDNN_STATUS_SUCCESS) << failure.what;
    EXPECT_EQ(failure.status, expected) << failure.what;
}

// The tag upstream's build_pointwise_op gives: the serialized mode name, or the
// mode's integer when the JSON library is compiled out.
std::string pointwiseTag([[maybe_unused]] PointwiseMode_t mode, [[maybe_unused]] const char* name)
{
#if !defined(CUDNN_FRONTEND_SKIP_JSON_LIB) && !defined(HIPDNN_FRONTEND_SKIP_JSON_LIB)
    return name;
#else
    return std::to_string(static_cast<int>(mode));
#endif
}

std::int64_t bytesAsInt64(float low, float high)
{
    const std::array<float, 2> pair = {low, high};
    std::int64_t value = 0;
    static_assert(sizeof(pair) == sizeof(value));
    std::memcpy(&value, pair.data(), sizeof(value));
    return value;
}

struct ConvSpec
{
    Dims x;
    Dims w;
    Dims y;
    Dims padding;
    Dims dilation;
    Dims stride;
    DataType_t xType = DataType_t::FLOAT;
    DataType_t wType = DataType_t::FLOAT;
    DataType_t yType = DataType_t::FLOAT;
    DataType_t computeType = DataType_t::FLOAT;
};

// Upstream's extract_feature_vector, restated: op type, X/W/Y dims, the first
// three pads/dilations/strides (the third zeroed for 2-D), X/W/Y/compute types,
// math mode, X/W/Y strides, then the eight bytes at alpha_s and at beta_s
// (alpha_s|beta_s and beta_s|alpha2_s).
v8fe::feature_vector_t upstreamConvFeatureVector(DescriptorType_t opType, const ConvSpec& spec)
{
    v8fe::feature_vector_t expected;
    expected.push_back(static_cast<int>(opType));
    for(const Dims* dims : {&spec.x, &spec.w, &spec.y})
    {
        expected.insert(expected.end(), dims->begin(), dims->end());
    }
    const bool is2D = spec.padding.size() == 2;
    for(const Dims* values : {&spec.padding, &spec.dilation, &spec.stride})
    {
        for(std::size_t i = 0; i < 3; ++i)
        {
            expected.push_back(i == 2 && is2D ? 0 : (*values)[i]);
        }
    }
    expected.push_back(static_cast<std::int64_t>(spec.xType));
    expected.push_back(static_cast<std::int64_t>(spec.wType));
    expected.push_back(static_cast<std::int64_t>(spec.yType));
    expected.push_back(static_cast<int>(spec.computeType));
    expected.push_back(static_cast<std::int64_t>(CUDNN_CROSS_CORRELATION));
    for(const Dims* dims : {&spec.x, &spec.w, &spec.y})
    {
        const Dims strides = packedStrides(*dims);
        expected.insert(expected.end(), strides.begin(), strides.end());
    }
    expected.push_back(bytesAsInt64(1.0F, 0.0F));
    expected.push_back(bytesAsInt64(0.0F, 1.0F));
    return expected;
}

class TestCudnnShimV8BuilderOperations : public hipdnn_shim_test::ShimV8BuilderFixture
{
public:
    // Public for the free scale-case helpers below.
    using ShimV8BuilderFixture::convDesc;

    static const std::vector<OpCase>& cases();

    static const OpCase& find(std::string_view label)
    {
        const std::vector<OpCase>& all = cases();
        for(const OpCase& opCase : all)
        {
            if(label == opCase.label)
            {
                return opCase;
            }
        }
        ADD_FAILURE() << "no case " << label;
        return all.front();
    }

protected:
    static v8fe::ConvDesc convolution(const Dims& padding,
                                      const Dims& dilation,
                                      const Dims& stride,
                                      DataType_t computeType = DataType_t::FLOAT)
    {
        const auto count = static_cast<std::int64_t>(padding.size());
        return v8fe::ConvDescBuilder()
            .setComputeType(computeType)
            .setMathMode(CUDNN_CROSS_CORRELATION)
            .setSpatialDimCount(count)
            .setPrePadding(count, padding.data())
            .setPostPadding(count, padding.data())
            .setDilation(count, dilation.data())
            .setSpatialStride(count, stride.data())
            .build();
    }

    static v8fe::MatMulDesc matmul()
    {
        return v8fe::MatMulDescBuilder().setComputeType(DataType_t::FLOAT).build();
    }

    static v8fe::ReductionDesc reduction()
    {
        return v8fe::ReductionDescBuilder()
            .setComputeType(DataType_t::FLOAT)
            .setReductionOp(v8fe::ReductionMode_t::ADD)
            .build();
    }

    static v8fe::ResampleDesc_v8 pool(v8fe::ResampleMode_t mode)
    {
        const std::array<std::int64_t, 2> window = {2, 2};
        const std::array<std::int64_t, 2> zeros = {0, 0};
        return v8fe::ResampleDescBuilder_v8()
            .setResampleMode(mode)
            .setComputeType(DataType_t::FLOAT)
            .setPaddingMode(mode == v8fe::ResampleMode_t::MAXPOOL ? v8fe::PaddingMode_t::NEG_INF_PAD
                                                                  : v8fe::PaddingMode_t::ZERO_PAD)
            .setSpatialDim(2, window.data())
            .setPrePadding(2, zeros.data())
            .setPostPadding(2, zeros.data())
            .setSpatialStride(2, window.data())
            .build();
    }

    static Slot tensorSlot(const char* name,
                           TensorSetter setter,
                           std::int64_t uid,
                           const Dims& dims,
                           DataType_t dataType = DataType_t::FLOAT)
    {
        return {name, [setter, uid, dims, dataType](Builder& builder) {
                    (builder.*setter)(tensor(uid, dims, false, dataType));
                }};
    }

    static Slot scalarSlot(const char* name, TensorSetter setter, std::int64_t uid)
    {
        return {name, [setter, uid](Builder& builder) { (builder.*setter)(scalar(uid)); }};
    }

    static Slot pairSlot(const char* name,
                         TensorPairSetter setter,
                         std::int64_t firstUid,
                         std::int64_t secondUid,
                         const Dims& dims)
    {
        return {name, [setter, firstUid, secondUid, dims](Builder& builder) {
                    (builder.*setter)(tensor(firstUid, dims), tensor(secondUid, dims));
                }};
    }

    static Slot pointwiseSlot(PointwiseMode_t mode)
    {
        return {"Pointwise", [mode](Builder& builder) { builder.setpwDesc(pointwiseDesc(mode)); }};
    }

    // The case's setup and every required slot except `skip`.
    static void fill(Builder& builder, const OpCase& opCase, const char* skip = nullptr)
    {
        if(opCase.setup)
        {
            opCase.setup(builder);
        }
        for(const Slot& slot : opCase.required)
        {
            if(skip == nullptr || std::string_view(slot.name) != skip)
            {
                slot.set(builder);
            }
        }
    }

    static const Slot& optionalSlot(const OpCase& opCase, std::string_view name)
    {
        for(const Slot& slot : opCase.optional)
        {
            if(name == slot.name)
            {
                return slot;
            }
        }
        ADD_FAILURE() << opCase.label << " has no optional slot " << name;
        return opCase.optional.front();
    }

    static BuildFailure failureOfCase(const OpCase& opCase, const char* skip, const Setter& extra)
    {
        Builder builder(opCase.kind);
        fill(builder, opCase, skip);
        if(extra)
        {
            extra(builder);
        }
        return buildFailure(builder);
    }

    static BuildFailure failureWith(std::string_view label, const Setter& extra)
    {
        return failureOfCase(find(label), nullptr, extra);
    }

    // Builds the case's operation (plus `extra`), then a one-op graph from it,
    // which must lower to one more hipDNN operation of the case's type.
    void expectBuildsAndLowers(const OpCase& opCase, const Setter& extra)
    {
        Builder builder(opCase.kind);
        fill(builder, opCase);
        if(extra)
        {
            extra(builder);
        }
        const std::size_t loweredBefore = created(opCase.lowered).size();
        std::optional<v8fe::Operation> op;
        const BuildFailure failure = failureOf([&] { op.emplace(builder.build()); });
        ASSERT_FALSE(failure.thrown) << failure.what;
        ASSERT_EQ(op->get_status(), CUDNN_STATUS_SUCCESS) << op->get_error();
        if(opCase.tag != nullptr)
        {
            EXPECT_EQ(op->getTag(), opCase.tag);
        }
        std::optional<v8fe::OperationGraph> opGraph;
        const BuildFailure lowering = failureOf([&] { opGraph.emplace(graph({&*op})); });
        ASSERT_FALSE(lowering.thrown) << lowering.what;
        ASSERT_EQ(opGraph->get_status(), CUDNN_STATUS_SUCCESS) << opGraph->get_error();
        if(opCase.tag != nullptr)
        {
            EXPECT_EQ(opGraph->getTag(), std::string(opCase.tag) + "_");
        }
        EXPECT_EQ(created(opCase.lowered).size(), loweredBefore + 1);
    }

    // The UID each named slot of the latest hipDNN operation of `type` refers to.
    void expectSlotUids(
        hipdnnBackendDescriptorType_t type,
        const std::vector<std::pair<hipdnnBackendAttributeName_t, std::int64_t>>& expected) const
    {
        const std::vector<hipdnnBackendDescriptor_t> ops = created(type);
        ASSERT_FALSE(ops.empty());
        hipdnnBackendDescriptor_t op = ops.back();
        for(const auto& [name, uid] : expected)
        {
            EXPECT_EQ(slotUid(op, name), uid) << "attribute " << static_cast<int>(name);
        }
    }

    bool reachedHipdnn(std::int64_t uid) const
    {
        for(const hipdnn_shim_test::CapturedSetAttribute& call :
            callsNamed(HIPDNN_ATTR_TENSOR_UNIQUE_ID))
        {
            if(call.value<std::int64_t>() == uid)
            {
                return true;
            }
        }
        return false;
    }

    static v8fe::Operation relu(std::int64_t xUid, std::int64_t yUid)
    {
        return pointwise(
            pointwiseDesc(PointwiseMode_t::RELU_FWD), tensor(xUid, X_DIMS), tensor(yUid, X_DIMS));
    }
};

// One case per operation kind, as builder setters.
const std::vector<OpCase>& TestCudnnShimV8BuilderOperations::cases()
{
    const auto normForward = [](NormMode_t mode, NormFwdPhase_t phase) -> Setter {
        return [mode, phase](Builder& builder) {
            builder.setNormalizationMode(mode).setNormFwdPhase(phase);
        };
    };
    const auto normBackward = [](NormMode_t mode) -> Setter {
        return [mode](Builder& builder) { builder.setNormalizationMode(mode); };
    };
    const Slot conv = {"Convolution", [](Builder& builder) { builder.setcDesc(convDesc()); }};
    const Slot epsilon = scalarSlot("Epsilon", &Builder::setEpsilonTensor, 4);
    const Slot normX = tensorSlot("X", &Builder::setxDesc, 1, NORM_DIMS);
    const Slot normY = tensorSlot("Y", &Builder::setyDesc, 5, NORM_DIMS);
    const Slot bwdX = tensorSlot("X", &Builder::setxDesc, 1, NORM_DIMS);
    const Slot bwdDy = tensorSlot("DY", &Builder::setdyDesc, 2, NORM_DIMS);
    const Slot bwdDx = tensorSlot("DX", &Builder::setdxDesc, 6, NORM_DIMS);
    const Slot bwdEpsilon = scalarSlot("Epsilon", &Builder::setEpsilonTensor, 9);

    static const std::vector<OpCase> s_cases = {
        {"ConvolutionForward",
         DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
         nullptr,
         {conv,
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("W", &Builder::setwDesc, 2, W_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 3, Y_DIMS)},
         {},
         "ConvFwd",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR},
        {"ConvolutionBackwardData",
         DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
         nullptr,
         {conv,
          tensorSlot("DX", &Builder::setdxDesc, 1, X_DIMS),
          tensorSlot("W", &Builder::setwDesc, 2, W_DIMS),
          tensorSlot("DY", &Builder::setdyDesc, 3, Y_DIMS)},
         {},
         "ConvBwdData",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR},
        {"ConvolutionBackwardFilter",
         DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
         nullptr,
         {conv,
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("DW", &Builder::setdwDesc, 2, W_DIMS),
          tensorSlot("DY", &Builder::setdyDesc, 3, Y_DIMS)},
         {},
         "ConvBwdFilter",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR},
        {"PointwiseMathUnary",
         DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR,
         nullptr,
         {pointwiseSlot(PointwiseMode_t::SQRT),
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 2, X_DIMS)},
         {},
         nullptr,
         HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR},
        {"PointwiseMathBinary",
         DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR,
         nullptr,
         {pointwiseSlot(PointwiseMode_t::ADD),
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("B", &Builder::setbDesc, 2, BIAS_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 3, X_DIMS)},
         {},
         nullptr,
         HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR},
        {"PointwiseBinarySelect",
         DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR,
         nullptr,
         {pointwiseSlot(PointwiseMode_t::BINARY_SELECT),
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("B", &Builder::setbDesc, 2, X_DIMS),
          tensorSlot("T", &Builder::settDesc, 3, X_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 4, X_DIMS)},
         {},
         nullptr,
         HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR},
        {"PointwiseIdentity",
         DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR,
         nullptr,
         {pointwiseSlot(PointwiseMode_t::IDENTITY),
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 2, X_DIMS)},
         {},
         nullptr,
         HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR},
        {"PointwiseActivationForward",
         DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR,
         nullptr,
         {pointwiseSlot(PointwiseMode_t::RELU_FWD),
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 2, X_DIMS)},
         {},
         nullptr,
         HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR},
        {"PointwiseActivationBackward",
         DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR,
         nullptr,
         {pointwiseSlot(PointwiseMode_t::RELU_BWD),
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("DY", &Builder::setdyDesc, 2, X_DIMS),
          tensorSlot("DX", &Builder::setdxDesc, 3, X_DIMS)},
         {},
         nullptr,
         HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR},
        {"Matmul",
         DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR,
         nullptr,
         {{"Matmul", [](Builder& builder) { builder.setmatmulDesc(matmul()); }},
          tensorSlot("A", &Builder::setaMatDesc, 1, {2, 4, 8}),
          tensorSlot("B", &Builder::setbMatDesc, 2, {2, 8, 16}),
          tensorSlot("C", &Builder::setcMatDesc, 3, {2, 4, 16})},
         {},
         "Matmul",
         HIPDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR},
        {"Reduction",
         DescriptorType_t::OPERATION_REDUCTION_DESCRIPTOR,
         nullptr,
         {{"Reduction", [](Builder& builder) { builder.setreductionDesc(reduction()); }},
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 2, {2, 4, 1, 1})},
         {},
         "Reduction",
         HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR},
        {"ResampleForward",
         DescriptorType_t::OPERATION_RESAMPLE_FWD_DESCRIPTOR,
         nullptr,
         {{"Resample",
           [](Builder& builder) { builder.setResampleDesc(pool(v8fe::ResampleMode_t::MAXPOOL)); }},
          tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
          tensorSlot("Y", &Builder::setyDesc, 2, POOLED_DIMS)},
         {tensorSlot("Index", &Builder::setidxDesc, 3, POOLED_DIMS, DataType_t::INT32)},
         "Resample_fwd",
         HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR},
        // A max-pool backward needs the forward pass's indices.
        {"ResampleBackward",
         DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR,
         nullptr,
         {{"Resample",
           [](Builder& builder) { builder.setResampleDesc(pool(v8fe::ResampleMode_t::MAXPOOL)); }},
          tensorSlot("DX", &Builder::setdxDesc, 1, X_DIMS),
          tensorSlot("DY", &Builder::setdyDesc, 2, POOLED_DIMS),
          tensorSlot("Index", &Builder::setidxDesc, 3, POOLED_DIMS, DataType_t::INT32)},
         {},
         "Resample_bwd",
         HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR},
        {"ResampleBackwardAverage",
         DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR,
         nullptr,
         {{"Resample",
           [](Builder& builder) {
               builder.setResampleDesc(pool(v8fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING));
           }},
          tensorSlot("DX", &Builder::setdxDesc, 1, X_DIMS),
          tensorSlot("DY", &Builder::setdyDesc, 2, POOLED_DIMS)},
         {},
         "Resample_bwd",
         HIPDNN_BACKEND_OPERATION_RESAMPLE_BWD_DESCRIPTOR},
        {"NormForwardBatchTraining",
         DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR,
         normForward(NormMode_t::BATCH_NORM, NormFwdPhase_t::TRAINING),
         {normX,
          tensorSlot("Scale", &Builder::setScale, 2, CHANNEL_DIMS),
          tensorSlot("Bias", &Builder::setBias, 3, CHANNEL_DIMS),
          epsilon,
          normY},
         {pairSlot("SavedStats", &Builder::setSavedMeanAndInvVar, 6, 7, CHANNEL_DIMS),
          pairSlot("InputRunningStats", &Builder::setPrevRunningMeanAndVar, 8, 9, CHANNEL_DIMS),
          scalarSlot("ExpAvgFactor", &Builder::setExpDecayFactorTensor, 10),
          pairSlot("OutputRunningStats", &Builder::setNextRunningMeanAndVar, 11, 12, CHANNEL_DIMS)},
         "Norm_Fwd",
         HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT},
        {"NormForwardBatchInference",
         DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR,
         normForward(NormMode_t::BATCH_NORM, NormFwdPhase_t::INFERENCE),
         {normX,
          pairSlot("SavedStats", &Builder::setSavedMeanAndInvVar, 6, 7, CHANNEL_DIMS),
          tensorSlot("Scale", &Builder::setScale, 2, CHANNEL_DIMS),
          tensorSlot("Bias", &Builder::setBias, 3, CHANNEL_DIMS),
          normY},
         {epsilon},
         "Norm_Fwd",
         HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT},
        {"NormForwardLayerTraining",
         DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR,
         normForward(NormMode_t::LAYER_NORM, NormFwdPhase_t::TRAINING),
         {normX,
          tensorSlot("Scale", &Builder::setScale, 2, FEATURE_DIMS),
          tensorSlot("Bias", &Builder::setBias, 3, FEATURE_DIMS),
          epsilon,
          normY},
         {pairSlot("SavedStats", &Builder::setSavedMeanAndInvVar, 6, 7, ROW_DIMS)},
         "Norm_Fwd",
         HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT},
        {"NormForwardLayerInference",
         DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR,
         normForward(NormMode_t::LAYER_NORM, NormFwdPhase_t::INFERENCE),
         {normX,
          tensorSlot("Scale", &Builder::setScale, 2, FEATURE_DIMS),
          tensorSlot("Bias", &Builder::setBias, 3, FEATURE_DIMS),
          epsilon,
          normY},
         {pairSlot("SavedStats", &Builder::setSavedMeanAndInvVar, 6, 7, ROW_DIMS)},
         "Norm_Fwd",
         HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT},
        {"NormForwardRmsTraining",
         DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR,
         normForward(NormMode_t::RMS_NORM, NormFwdPhase_t::TRAINING),
         {normX, tensorSlot("Scale", &Builder::setScale, 2, FEATURE_DIMS), epsilon, normY},
         {tensorSlot("Bias", &Builder::setBias, 3, FEATURE_DIMS),
          tensorSlot("InvVariance", &Builder::setSavedInvVar, 7, ROW_DIMS)},
         "Norm_Fwd",
         HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT},
        {"NormForwardRmsInference",
         DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR,
         normForward(NormMode_t::RMS_NORM, NormFwdPhase_t::INFERENCE),
         {normX, tensorSlot("Scale", &Builder::setScale, 2, FEATURE_DIMS), epsilon, normY},
         {tensorSlot("Bias", &Builder::setBias, 3, FEATURE_DIMS),
          tensorSlot("InvVariance", &Builder::setSavedInvVar, 7, ROW_DIMS)},
         "Norm_Fwd",
         HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT},
        {"NormBackwardBatch",
         DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR,
         normBackward(NormMode_t::BATCH_NORM),
         {bwdX,
          bwdDy,
          tensorSlot("Scale", &Builder::setScale, 3, CHANNEL_DIMS),
          pairSlot("SavedStats", &Builder::setSavedMeanAndInvVar, 4, 5, CHANNEL_DIMS),
          bwdDx,
          tensorSlot("DScale", &Builder::setDScale, 7, CHANNEL_DIMS),
          tensorSlot("DBias", &Builder::setDBias, 8, CHANNEL_DIMS)},
         {bwdEpsilon},
         "Norm_Bwd",
         HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT},
        {"NormBackwardLayer",
         DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR,
         normBackward(NormMode_t::LAYER_NORM),
         {bwdX,
          bwdDy,
          tensorSlot("Scale", &Builder::setScale, 3, FEATURE_DIMS),
          bwdDx,
          tensorSlot("DScale", &Builder::setDScale, 7, FEATURE_DIMS),
          tensorSlot("DBias", &Builder::setDBias, 8, FEATURE_DIMS)},
         {pairSlot("SavedStats", &Builder::setSavedMeanAndInvVar, 4, 5, ROW_DIMS), bwdEpsilon},
         "Norm_Bwd",
         HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR_EXT},
        {"NormBackwardRms",
         DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR,
         normBackward(NormMode_t::RMS_NORM),
         {bwdX,
          bwdDy,
          tensorSlot("Scale", &Builder::setScale, 3, FEATURE_DIMS),
          tensorSlot("InvVariance", &Builder::setSavedInvVar, 5, ROW_DIMS),
          bwdDx,
          tensorSlot("DScale", &Builder::setDScale, 7, FEATURE_DIMS)},
         {tensorSlot("DBias", &Builder::setDBias, 8, FEATURE_DIMS)},
         "Norm_Bwd",
         HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT},
    };
    return s_cases;
}

class TestCudnnShimV8BuilderOperationKinds : public TestCudnnShimV8BuilderOperations,
                                             public ::testing::WithParamInterface<OpCase>
{
};

TEST_P(TestCudnnShimV8BuilderOperationKinds, RequiredSlotsSuffice)
{
    expectBuildsAndLowers(GetParam(), nullptr);
}

TEST_P(TestCudnnShimV8BuilderOperationKinds, OptionalSlotsAccepted)
{
    const OpCase& opCase = GetParam();
    expectBuildsAndLowers(opCase, [&opCase](Builder& builder) {
        for(const Slot& slot : opCase.optional)
        {
            slot.set(builder);
        }
    });
}

TEST_P(TestCudnnShimV8BuilderOperationKinds, MissingRequiredSlotIsBadParam)
{
    const OpCase& opCase = GetParam();
    for(const Slot& slot : opCase.required)
    {
        SCOPED_TRACE(slot.name);
        expectStatus(failureOfCase(opCase, slot.name, nullptr), CUDNN_STATUS_BAD_PARAM);
    }
}

INSTANTIATE_TEST_SUITE_P(Table,
                         TestCudnnShimV8BuilderOperationKinds,
                         ::testing::ValuesIn(TestCudnnShimV8BuilderOperations::cases()),
                         [](const ::testing::TestParamInfo<OpCase>& info) {
                             return std::string(info.param.label);
                         });

TEST_F(TestCudnnShimV8BuilderOperations, UnloweredKindsNotSupported)
{
    const Dims channel = {1, 4, 1, 1};

    Builder genStats(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);
    genStats.setxDesc(tensor(1, X_DIMS))
        .setSumDesc(tensor(2, channel))
        .setSqSumDesc(tensor(3, channel))
        .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
        .setMathPrecision(CUDNN_DATA_FLOAT);

    Builder bnFinalize(DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
    bnFinalize.setSumDesc(tensor(1, channel))
        .setSqSumDesc(tensor(2, channel))
        .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
        .setMathPrecision(CUDNN_DATA_FLOAT);

    Builder bnBackwardWeights(DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR);
    bnBackwardWeights.setxDesc(tensor(1, X_DIMS))
        .setdyDesc(tensor(2, X_DIMS))
        .setSavedMeanAndInvVar(tensor(3, channel), tensor(4, channel))
        .setMathPrecision(CUDNN_DATA_FLOAT);

    const v8fe::RngDesc rngDesc = v8fe::RngDescBuilder()
                                      .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                      .setBernoulliDistProbability(0.5)
                                      .build();
    ASSERT_EQ(rngDesc.get_status(), CUDNN_STATUS_SUCCESS) << rngDesc.get_error();
    Builder rng(DescriptorType_t::OPERATION_RNG_DESCRIPTOR);
    rng.setyDesc(tensor(1, X_DIMS)).setRngDesc(rngDesc).setSeed(7);

    Builder reshape(DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR);
    reshape.setxDesc(tensor(1, X_DIMS)).setyDesc(tensor(2, {2, 256}));

    Builder pagedCacheLoad(DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR);
    pagedCacheLoad.setcontainerDesc(tensor(1, {8, 2, 16, 64}))
        .setpageTableDesc(tensor(2, {2, 1, 4, 1}, false, DataType_t::INT32))
        .setsequenceDesc(tensor(3, {2, 1, 1, 1}, false, DataType_t::INT32))
        .setyDesc(tensor(4, {2, 2, 64, 64}));

    const std::array<std::pair<Builder*, const char*>, 6> builders = {{
        {&genStats, "GenStats"},
        {&bnFinalize, "BnFinalizeStatistics"},
        {&bnBackwardWeights, "BnBackwardWeights"},
        {&rng, "Rng"},
        {&reshape, "Reshape"},
        {&pagedCacheLoad, "PagedCacheLoad"},
    }};
    for(const auto& [builder, kind] : builders)
    {
        SCOPED_TRACE(kind);
        expectStatus(buildFailure(*builder), CUDNN_STATUS_NOT_SUPPORTED);
    }
}

// Upstream checks and sends an unlowerable kind's slots before the kind itself is
// rejected, so a missing or non-tensor slot is still BAD_PARAM.
TEST_F(TestCudnnShimV8BuilderOperations, UnloweredKindsCheckSlotsFirst)
{
    const Dims channel = {1, 4, 1, 1};
    const v8fe::RngDesc rngDesc = v8fe::RngDescBuilder()
                                      .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                      .setBernoulliDistProbability(0.5)
                                      .build();
    const v8fe::ManagedOpaqueDescriptor notTensor = convDesc().get_desc();
    struct MissingSlot
    {
        const char* label;
        DescriptorType_t kind;
        Setter set;
    };
    const std::vector<MissingSlot> cases = {
        {"GenStats without sum",
         DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR,
         [&channel](Builder& b) {
             b.setxDesc(tensor(1, X_DIMS)).setSqSumDesc(tensor(3, channel));
         }},
        {"GenStats with a non-tensor X",
         DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR,
         [&channel, &notTensor](Builder& b) {
             b.setxDesc(notTensor).setSumDesc(tensor(2, channel)).setSqSumDesc(tensor(3, channel));
         }},
        {"BnBackwardWeights without DY",
         DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
         [&channel](Builder& b) {
             b.setxDesc(tensor(1, X_DIMS))
                 .setSavedMeanAndInvVar(tensor(3, channel), tensor(4, channel));
         }},
        {"Rng without Y",
         DescriptorType_t::OPERATION_RNG_DESCRIPTOR,
         [&rngDesc](Builder& b) { b.setRngDesc(rngDesc).setSeed(7); }},
        {"Rng without descriptor",
         DescriptorType_t::OPERATION_RNG_DESCRIPTOR,
         [](Builder& b) { b.setyDesc(tensor(1, X_DIMS)).setSeed(7); }},
        {"Reshape without Y",
         DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR,
         [](Builder& b) { b.setxDesc(tensor(1, X_DIMS)); }},
        {"PagedCacheLoad without sequence",
         DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR,
         [](Builder& b) {
             b.setcontainerDesc(tensor(1, {8, 2, 16, 64}))
                 .setpageTableDesc(tensor(2, {2, 1, 4, 1}, false, DataType_t::INT32))
                 .setyDesc(tensor(4, {2, 2, 64, 64}));
         }},
    };
    for(const MissingSlot& missing : cases)
    {
        SCOPED_TRACE(missing.label);
        Builder builder(missing.kind);
        missing.set(builder);
        expectStatus(buildFailure(builder), CUDNN_STATUS_BAD_PARAM);
    }
}

// A descriptor without a tensor record in a slot the builder sends is a caller
// error, even in a slot where a built tensor is NOT_SUPPORTED (resample backward X).
TEST_F(TestCudnnShimV8BuilderOperations, NonTensorInSentSlotIsBadParam)
{
    const v8fe::ManagedOpaqueDescriptor notTensor = convDesc().get_desc();
    const v8fe::ManagedOpaqueDescriptor unbuilt
        = v8fe::make_shared_backend_pointer(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    for(const v8fe::ManagedOpaqueDescriptor& desc : {notTensor, unbuilt})
    {
        SCOPED_TRACE(desc == notTensor ? "convolution descriptor" : "unbuilt tensor");
        expectStatus(failureWith("ResampleBackward", [&desc](Builder& b) { b.setxDesc(desc); }),
                     CUDNN_STATUS_BAD_PARAM);
        expectStatus(
            failureWith("PointwiseActivationBackward", [&desc](Builder& b) { b.setdyDesc(desc); }),
            CUDNN_STATUS_BAD_PARAM);
        expectStatus(failureWith("Matmul", [&desc](Builder& b) { b.setaMatDesc(desc); }),
                     CUDNN_STATUS_BAD_PARAM);
    }
}

// Upstream's OperationBuilder has no build path for these kinds.
TEST_F(TestCudnnShimV8BuilderOperations, KindsWithoutBuilderPathBadParam)
{
    for(const DescriptorType_t kind :
        {DescriptorType_t::OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR,
         DescriptorType_t::OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR,
         DescriptorType_t::OPERATION_CONCAT_DESCRIPTOR,
         DescriptorType_t::OPERATION_CONCATENATE_DESCRIPTOR,
         DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR,
         DescriptorType_t::OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR,
         DescriptorType_t::OPERATION_SIGNAL_DESCRIPTOR,
         DescriptorType_t::OPERATION_SLICE_DESCRIPTOR,
         DescriptorType_t::OPERATION_TRANSPOSE_DESCRIPTOR})
    {
        SCOPED_TRACE(static_cast<int>(kind));
        Builder builder(kind);
        builder.setxDesc(tensor(1, X_DIMS)).setyDesc(tensor(2, X_DIMS));
        expectStatus(buildFailure(builder), CUDNN_STATUS_BAD_PARAM);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, WrongKindSetterRejected)
{
    const v8fe::RngDesc rngDesc = v8fe::RngDescBuilder()
                                      .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                      .setBernoulliDistProbability(0.5)
                                      .build();
    struct WrongKind
    {
        const char* label;
        DescriptorType_t kind;
        Setter set;
    };
    const DescriptorType_t pointwiseKind = DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR;
    const DescriptorType_t convKind = DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
    const std::vector<WrongKind> cases = {
        {"W on pointwise", pointwiseKind, [](Builder& b) { b.setwDesc(tensor(1, W_DIMS)); }},
        {"Convolution on pointwise", pointwiseKind, [](Builder& b) { b.setcDesc(convDesc()); }},
        {"A on pointwise", pointwiseKind, [](Builder& b) { b.setaMatDesc(tensor(1, {2, 4, 8})); }},
        {"B matrix on pointwise",
         pointwiseKind,
         [](Builder& b) { b.setbMatDesc(tensor(1, {2, 8, 16})); }},
        {"C on pointwise", pointwiseKind, [](Builder& b) { b.setcMatDesc(tensor(1, {2, 4, 16})); }},
        {"M override on pointwise",
         pointwiseKind,
         [](Builder& b) { b.setmOverrideDesc(tensor(1, {2, 1, 1}, false, DataType_t::INT64)); }},
        {"N override on pointwise",
         pointwiseKind,
         [](Builder& b) { b.setnOverrideDesc(tensor(1, {2, 1, 1}, false, DataType_t::INT64)); }},
        {"K override on pointwise",
         pointwiseKind,
         [](Builder& b) { b.setkOverrideDesc(tensor(1, {2, 1, 1}, false, DataType_t::INT64)); }},
        {"Matmul on pointwise", pointwiseKind, [](Builder& b) { b.setmatmulDesc(matmul()); }},
        {"Reduction on pointwise",
         pointwiseKind,
         [](Builder& b) { b.setreductionDesc(reduction()); }},
        {"Resample on pointwise",
         pointwiseKind,
         [](Builder& b) { b.setResampleDesc(pool(v8fe::ResampleMode_t::MAXPOOL)); }},
        {"Rng on pointwise", pointwiseKind, [&rngDesc](Builder& b) { b.setRngDesc(rngDesc); }},
        {"B on convolution", convKind, [](Builder& b) { b.setbDesc(tensor(1, {1, 8, 1, 1})); }},
        {"T on convolution", convKind, [](Builder& b) { b.settDesc(tensor(1, Y_DIMS)); }},
        {"Pointwise on convolution",
         convKind,
         [](Builder& b) { b.setpwDesc(pointwiseDesc(PointwiseMode_t::ADD)); }},
    };
    for(const WrongKind& wrongKind : cases)
    {
        SCOPED_TRACE(wrongKind.label);
        Builder builder(wrongKind.kind);
        const BuildFailure failure = failureOf([&] { wrongKind.set(builder); });
        expectStatus(failure, CUDNN_STATUS_BAD_PARAM);
    }
}

// Port count alone does not decide the mode class.
TEST_F(TestCudnnShimV8BuilderOperations, PointwiseModeClassesBuild)
{
    struct ModeClass
    {
        std::vector<PointwiseMode_t> modes;
        Setter slots;
    };
    const auto x = [](Builder& b) { b.setxDesc(tensor(1, X_DIMS)); };
    // COS, MOD and POW have no hipDNN mapping, so their PointWiseDesc already
    // fails to build.
    const std::vector<ModeClass> classes = {
        {{PointwiseMode_t::ADD,
          PointwiseMode_t::MUL,
          PointwiseMode_t::DIV,
          PointwiseMode_t::SUB,
          PointwiseMode_t::ADD_SQUARE,
          PointwiseMode_t::MIN,
          PointwiseMode_t::MAX,
          PointwiseMode_t::CMP_EQ,
          PointwiseMode_t::CMP_NEQ,
          PointwiseMode_t::CMP_GT,
          PointwiseMode_t::CMP_GE,
          PointwiseMode_t::CMP_LT,
          PointwiseMode_t::CMP_LE,
          PointwiseMode_t::LOGICAL_AND,
          PointwiseMode_t::LOGICAL_OR},
         [x](Builder& b) {
             x(b);
             b.setbDesc(tensor(2, X_DIMS)).setyDesc(tensor(3, X_DIMS));
         }},
        {{PointwiseMode_t::SQRT,         PointwiseMode_t::RSQRT,
          PointwiseMode_t::SIN,          PointwiseMode_t::TAN,
          PointwiseMode_t::LOG,          PointwiseMode_t::NEG,
          PointwiseMode_t::ABS,          PointwiseMode_t::CEIL,
          PointwiseMode_t::FLOOR,        PointwiseMode_t::LOGICAL_NOT,
          PointwiseMode_t::GEN_INDEX,    PointwiseMode_t::ERF,
          PointwiseMode_t::RECIPROCAL,   PointwiseMode_t::IDENTITY,
          PointwiseMode_t::RELU_FWD,     PointwiseMode_t::TANH_FWD,
          PointwiseMode_t::SIGMOID_FWD,  PointwiseMode_t::ELU_FWD,
          PointwiseMode_t::GELU_FWD,     PointwiseMode_t::GELU_APPROX_TANH_FWD,
          PointwiseMode_t::SOFTPLUS_FWD, PointwiseMode_t::SWISH_FWD,
          PointwiseMode_t::EXP},
         [x](Builder& b) {
             x(b);
             b.setyDesc(tensor(2, X_DIMS));
         }},
        {{PointwiseMode_t::BINARY_SELECT},
         [x](Builder& b) {
             x(b);
             b.setbDesc(tensor(2, X_DIMS)).settDesc(tensor(3, X_DIMS)).setyDesc(tensor(4, X_DIMS));
         }},
        {{PointwiseMode_t::RELU_BWD,
          PointwiseMode_t::TANH_BWD,
          PointwiseMode_t::SIGMOID_BWD,
          PointwiseMode_t::ELU_BWD,
          PointwiseMode_t::GELU_BWD,
          PointwiseMode_t::GELU_APPROX_TANH_BWD,
          PointwiseMode_t::SOFTPLUS_BWD,
          PointwiseMode_t::SWISH_BWD},
         [x](Builder& b) {
             x(b);
             b.setdyDesc(tensor(2, X_DIMS)).setdxDesc(tensor(3, X_DIMS));
         }},
    };
    for(const ModeClass& modeClass : classes)
    {
        for(const PointwiseMode_t mode : modeClass.modes)
        {
            SCOPED_TRACE(static_cast<int>(mode));
            Builder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
            builder.setpwDesc(pointwiseDesc(mode));
            modeClass.slots(builder);
            expectStatus(buildFailure(builder), CUDNN_STATUS_SUCCESS);
        }
    }
}

// The builder forwards only the ports a pointwise mode class uses, so a tensor
// set in any other slot is dropped rather than rejected.
TEST_F(TestCudnnShimV8BuilderOperations, PointwiseSlotsOutsideModeClassIgnored)
{
    struct Extra
    {
        const char* row;
        const char* slot;
        TensorSetter set;
    };
    const std::vector<Extra> extras = {
        {"PointwiseActivationBackward", "Y", &Builder::setyDesc},
        {"PointwiseActivationBackward", "B", &Builder::setbDesc},
        {"PointwiseActivationBackward", "T", &Builder::settDesc},
        {"PointwiseMathUnary", "DY", &Builder::setdyDesc},
        {"PointwiseMathUnary", "DX", &Builder::setdxDesc},
        {"PointwiseMathBinary", "DY", &Builder::setdyDesc},
        {"PointwiseMathBinary", "DX", &Builder::setdxDesc},
        {"PointwiseBinarySelect", "DY", &Builder::setdyDesc},
        {"PointwiseBinarySelect", "DX", &Builder::setdxDesc},
        {"PointwiseIdentity", "DY", &Builder::setdyDesc},
        {"PointwiseIdentity", "DX", &Builder::setdxDesc},
        {"PointwiseActivationForward", "DY", &Builder::setdyDesc},
        {"PointwiseActivationForward", "DX", &Builder::setdxDesc},
        {"PointwiseMathUnary", "T", &Builder::settDesc},
        {"PointwiseMathBinary", "T", &Builder::settDesc},
        {"PointwiseIdentity", "T", &Builder::settDesc},
        {"PointwiseActivationForward", "T", &Builder::settDesc},
        {"PointwiseMathUnary", "B", &Builder::setbDesc},
        {"PointwiseIdentity", "B", &Builder::setbDesc},
        {"PointwiseActivationForward", "B", &Builder::setbDesc},
    };
    for(const Extra& extra : extras)
    {
        SCOPED_TRACE(std::string(extra.row) + " " + extra.slot);
        expectBuildsAndLowers(find(extra.row), [&extra](Builder& builder) {
            (builder.*extra.set)(tensor(STRAY_UID, X_DIMS));
        });
        EXPECT_TRUE(reachedHipdnn(1));
        EXPECT_FALSE(reachedHipdnn(STRAY_UID));
    }

    for(const PointwiseMode_t mode : {PointwiseMode_t::GEN_INDEX, PointwiseMode_t::EXP})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        Builder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        builder.setpwDesc(pointwiseDesc(mode))
            .setxDesc(tensor(1, X_DIMS))
            .setbDesc(tensor(STRAY_UID, X_DIMS))
            .setyDesc(tensor(2, X_DIMS));
        const v8fe::Operation op = builder.build();
        [[maybe_unused]] const v8fe::OperationGraph lowered = graph({&op});
        EXPECT_TRUE(reachedHipdnn(1));
        EXPECT_FALSE(reachedHipdnn(STRAY_UID));
    }
}

enum class Scale
{
    ALPHA,
    BETA,
    ALPHA2
};

struct ScaleCase
{
    const char* row;
    Scale scale;
    double identity;
    bool doubleCompute;
};

const std::vector<ScaleCase>& scaleCases()
{
    static const std::vector<ScaleCase> s_cases = {
        {"ConvolutionForward", Scale::ALPHA, 1.0, false},
        {"ConvolutionForward", Scale::BETA, 0.0, false},
        {"ConvolutionForward", Scale::ALPHA, 1.0, true},
        {"ConvolutionForward", Scale::BETA, 0.0, true},
        {"ConvolutionBackwardData", Scale::ALPHA, 1.0, false},
        {"ConvolutionBackwardData", Scale::BETA, 0.0, true},
        {"ConvolutionBackwardFilter", Scale::ALPHA, 1.0, true},
        {"ConvolutionBackwardFilter", Scale::BETA, 0.0, false},
        {"PointwiseMathBinary", Scale::ALPHA, 1.0, false},
        {"PointwiseMathBinary", Scale::ALPHA2, 1.0, false},
        {"ResampleForward", Scale::ALPHA, 1.0, false},
        {"ResampleForward", Scale::BETA, 0.0, false},
        {"ResampleBackward", Scale::ALPHA, 1.0, false},
        {"ResampleBackward", Scale::BETA, 0.0, false},
    };
    return s_cases;
}

// Sets `scale` through the float or the double overload; a double-compute
// convolution makes the builder store alpha/beta as double.
Setter scaleSetter(const ScaleCase& scaleCase, double value, bool asFloat)
{
    return [scaleCase, value, asFloat](Builder& builder) {
        if(scaleCase.doubleCompute)
        {
            builder.setcDesc(TestCudnnShimV8BuilderOperations::convDesc(DataType_t::DOUBLE));
        }
        const auto narrow = static_cast<float>(value);
        switch(scaleCase.scale)
        {
        case Scale::ALPHA:
            if(asFloat)
            {
                builder.setAlpha(narrow);
            }
            else
            {
                builder.setAlpha(value);
            }
            break;
        case Scale::BETA:
            if(asFloat)
            {
                builder.setBeta(narrow);
            }
            else
            {
                builder.setBeta(value);
            }
            break;
        case Scale::ALPHA2:
            if(asFloat)
            {
                builder.setAlpha2(narrow);
            }
            else
            {
                builder.setAlpha2(value);
            }
            break;
        default:
            break;
        }
    };
}

std::string scaleLabel(const ScaleCase& scaleCase, bool asFloat)
{
    const std::array<const char*, 3> names = {"alpha", "beta", "alpha2"};
    return std::string(scaleCase.row) + " " + names.at(static_cast<std::size_t>(scaleCase.scale))
           + (asFloat ? " float" : " double") + (scaleCase.doubleCompute ? " double compute" : "");
}

TEST_F(TestCudnnShimV8BuilderOperations, ScaleOtherThanIdentityNotSupported)
{
    for(const ScaleCase& scaleCase : scaleCases())
    {
        for(const bool asFloat : {true, false})
        {
            SCOPED_TRACE(scaleLabel(scaleCase, asFloat));
            for(const double value :
                {scaleCase.identity + 0.5, std::numeric_limits<double>::quiet_NaN()})
            {
                expectStatus(failureWith(scaleCase.row, scaleSetter(scaleCase, value, asFloat)),
                             CUDNN_STATUS_NOT_SUPPORTED);
            }
        }
    }
}

// A value float rounds to the identity is kept only by the double overload on a double-compute
// convolution, the one case where upstream sends the double.
TEST_F(TestCudnnShimV8BuilderOperations, ScaleBelowFloatResolutionFollowsComputeType)
{
    for(const ScaleCase& scaleCase : scaleCases())
    {
        if(std::string_view(scaleCase.row).rfind("Convolution", 0) != 0)
        {
            continue;
        }
        const double nearIdentity = scaleCase.identity == 0.0 ? 1e-50 : scaleCase.identity + 1e-12;
        ASSERT_EQ(static_cast<float>(nearIdentity), static_cast<float>(scaleCase.identity));
        for(const bool asFloat : {true, false})
        {
            SCOPED_TRACE(scaleLabel(scaleCase, asFloat));
            const bool sentAsDouble = scaleCase.doubleCompute && !asFloat;
            expectStatus(failureWith(scaleCase.row, scaleSetter(scaleCase, nearIdentity, asFloat)),
                         sentAsDouble ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_SUCCESS);
        }
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, IdentityScalesAccepted)
{
    for(const ScaleCase& scaleCase : scaleCases())
    {
        for(const bool asFloat : {true, false})
        {
            SCOPED_TRACE(scaleLabel(scaleCase, asFloat));
            expectStatus(
                failureWith(scaleCase.row, scaleSetter(scaleCase, scaleCase.identity, asFloat)),
                CUDNN_STATUS_SUCCESS);
        }
    }
    expectStatus(failureWith("PointwiseMathBinary",
                             [](Builder& builder) { builder.setAlpha(1.0F).setAlpha2(1.0); }),
                 CUDNN_STATUS_SUCCESS);
}

// Only X is set: the mode is rejected before any required tensor is checked.
TEST_F(TestCudnnShimV8BuilderOperations, NormModesWithoutLoweringNotSupported)
{
    for(const NormMode_t mode :
        {NormMode_t::INSTANCE_NORM, NormMode_t::GROUP_NORM, NormMode_t::ADA_LAYER_NORM})
    {
        SCOPED_TRACE(static_cast<int>(mode));
        for(const NormFwdPhase_t phase : {NormFwdPhase_t::TRAINING, NormFwdPhase_t::INFERENCE})
        {
            Builder forward(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR);
            forward.setNormalizationMode(mode).setNormFwdPhase(phase).setxDesc(
                tensor(1, NORM_DIMS));
            expectStatus(buildFailure(forward), CUDNN_STATUS_NOT_SUPPORTED);
        }
        Builder backward(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR);
        backward.setNormalizationMode(mode).setxDesc(tensor(1, NORM_DIMS));
        expectStatus(buildFailure(backward), CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, NormPeerStatsNotSupported)
{
    for(const OpCase& opCase : cases())
    {
        if(opCase.kind != DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR
           && opCase.kind != DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
        {
            continue;
        }
        SCOPED_TRACE(opCase.label);
        expectStatus(failureOfCase(opCase,
                                   nullptr,
                                   [](Builder& builder) {
                                       builder.addPeerStatTensor(tensor(20, CHANNEL_DIMS));
                                   }),
                     CUDNN_STATUS_NOT_SUPPORTED);
        expectStatus(failureOfCase(opCase,
                                   nullptr,
                                   [](Builder& builder) {
                                       std::vector<v8fe::Tensor> peers;
                                       peers.push_back(tensor(20, CHANNEL_DIMS));
                                       peers.push_back(tensor(21, CHANNEL_DIMS));
                                       builder.setPeerStatTensor(peers);
                                   }),
                     CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, EmptyPeerStatsAccepted)
{
    expectBuildsAndLowers(find("NormForwardBatchTraining"), [](Builder& builder) {
        builder.setPeerStatTensor(std::vector<v8fe::Tensor>{});
    });
}

// Only the inverse variance can be set alone through the builder.
TEST_F(TestCudnnShimV8BuilderOperations, NormSavedStatsBothOrNone)
{
    struct Pair
    {
        const char* row;
        const char* skip;
        const Dims* dims;
    };
    for(const Pair& pair : {Pair{"NormForwardBatchTraining", nullptr, &CHANNEL_DIMS},
                            Pair{"NormForwardBatchInference", "SavedStats", &CHANNEL_DIMS},
                            Pair{"NormForwardLayerTraining", nullptr, &ROW_DIMS},
                            Pair{"NormForwardLayerInference", nullptr, &ROW_DIMS},
                            Pair{"NormBackwardBatch", "SavedStats", &CHANNEL_DIMS},
                            Pair{"NormBackwardLayer", nullptr, &ROW_DIMS}})
    {
        SCOPED_TRACE(pair.row);
        const Dims& dims = *pair.dims;
        expectStatus(
            failureOfCase(find(pair.row),
                          pair.skip,
                          [&dims](Builder& builder) { builder.setSavedInvVar(tensor(7, dims)); }),
            CUDNN_STATUS_BAD_PARAM);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, RunningStatsAllOrNone)
{
    const OpCase& opCase = find("NormForwardBatchTraining");
    const std::array<const Slot*, 3> groups = {&optionalSlot(opCase, "InputRunningStats"),
                                               &optionalSlot(opCase, "ExpAvgFactor"),
                                               &optionalSlot(opCase, "OutputRunningStats")};
    // Every non-empty proper subset of the three setter groups.
    for(unsigned mask = 1; mask < 7; ++mask)
    {
        SCOPED_TRACE(mask);
        expectStatus(failureOfCase(opCase,
                                   nullptr,
                                   [&groups, mask](Builder& builder) {
                                       for(std::size_t i = 0; i < groups.size(); ++i)
                                       {
                                           if((mask & (1U << i)) != 0)
                                           {
                                               groups.at(i)->set(builder);
                                           }
                                       }
                                   }),
                     CUDNN_STATUS_BAD_PARAM);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, RunningStatsOutsideBatchTrainingNotSupported)
{
    const OpCase& batchTraining = find("NormForwardBatchTraining");
    for(const char* row : {"NormForwardBatchInference",
                           "NormForwardLayerTraining",
                           "NormForwardLayerInference",
                           "NormForwardRmsTraining",
                           "NormForwardRmsInference"})
    {
        for(const char* group : {"InputRunningStats", "ExpAvgFactor", "OutputRunningStats"})
        {
            SCOPED_TRACE(std::string(row) + " " + group);
            expectStatus(failureWith(row, optionalSlot(batchTraining, group).set),
                         CUDNN_STATUS_NOT_SUPPORTED);
        }
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, RmsMeanNotSupported)
{
    for(const char* row : {"NormForwardRmsTraining", "NormForwardRmsInference", "NormBackwardRms"})
    {
        SCOPED_TRACE(row);
        expectStatus(failureWith(row,
                                 [](Builder& builder) {
                                     builder.setSavedMeanAndInvVar(tensor(6, ROW_DIMS),
                                                                   tensor(7, ROW_DIMS));
                                 }),
                     CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, MatmulOverridesNotSupported)
{
    for(const auto& [setter, name] :
        {std::pair<TensorSetter, const char*>{&Builder::setmOverrideDesc, "M"},
         std::pair<TensorSetter, const char*>{&Builder::setnOverrideDesc, "N"},
         std::pair<TensorSetter, const char*>{&Builder::setkOverrideDesc, "K"}})
    {
        SCOPED_TRACE(name);
        const TensorSetter set = setter;
        expectStatus(failureWith("Matmul",
                                 [set](Builder& builder) {
                                     (builder.*set)(tensor(4, {2, 1, 1}, false, DataType_t::INT64));
                                 }),
                     CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, ResampleBackwardForwardTensorsNotSupported)
{
    for(const char* row : {"ResampleBackward", "ResampleBackwardAverage"})
    {
        SCOPED_TRACE(row);
        expectStatus(
            failureWith(row, [](Builder& builder) { builder.setxDesc(tensor(4, X_DIMS)); }),
            CUDNN_STATUS_NOT_SUPPORTED);
        expectStatus(
            failureWith(row, [](Builder& builder) { builder.setyDesc(tensor(4, POOLED_DIMS)); }),
            CUDNN_STATUS_NOT_SUPPORTED);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, ConvolutionTensorRankMismatchRejected)
{
    const std::vector<std::pair<const char*, std::vector<TensorSetter>>> rows = {
        {"ConvolutionForward", {&Builder::setxDesc, &Builder::setwDesc, &Builder::setyDesc}},
        {"ConvolutionBackwardData", {&Builder::setdxDesc, &Builder::setwDesc, &Builder::setdyDesc}},
        {"ConvolutionBackwardFilter",
         {&Builder::setxDesc, &Builder::setdwDesc, &Builder::setdyDesc}},
    };
    for(const auto& [row, setters] : rows)
    {
        for(std::size_t i = 0; i < setters.size(); ++i)
        {
            SCOPED_TRACE(std::string(row) + " slot " + std::to_string(i));
            const TensorSetter set = setters[i];
            expectStatus(
                failureWith(row, [set](Builder& builder) { (builder.*set)(tensor(9, {2, 4, 8})); }),
                CUDNN_STATUS_BAD_PARAM);
        }
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, ConvolutionSpatialDimsMismatchRejected)
{
    for(const char* row :
        {"ConvolutionForward", "ConvolutionBackwardData", "ConvolutionBackwardFilter"})
    {
        SCOPED_TRACE(row);
        expectStatus(failureWith(row,
                                 [](Builder& builder) {
                                     builder.setcDesc(convolution({1, 1, 1}, {1, 1, 1}, {1, 1, 1}));
                                 }),
                     CUDNN_STATUS_BAD_PARAM);
    }
}

// Required tensors are checked before alpha/beta, and alpha/beta before
// tensors that cannot be lowered.
TEST_F(TestCudnnShimV8BuilderOperations, FirstFailingRuleDecidesStatus)
{
    expectStatus(failureOfCase(find("ConvolutionForward"),
                               "Y",
                               [](Builder& builder) { builder.setAlpha(2.0); }),
                 CUDNN_STATUS_BAD_PARAM);
    expectStatus(failureOfCase(find("ResampleBackward"),
                               "Index",
                               [](Builder& builder) { builder.setBeta(0.5); }),
                 CUDNN_STATUS_BAD_PARAM);
    expectStatus(failureOfCase(find("NormForwardRmsTraining"),
                               "Epsilon",
                               [](Builder& builder) {
                                   builder.setSavedMeanAndInvVar(tensor(6, ROW_DIMS),
                                                                 tensor(7, ROW_DIMS));
                               }),
                 CUDNN_STATUS_BAD_PARAM);
    expectStatus(
        failureWith("ResampleBackward",
                    [](Builder& builder) { builder.setxDesc(tensor(4, X_DIMS)).setBeta(0.5); }),
        CUDNN_STATUS_NOT_SUPPORTED);
}

// The builder keeps what it was given alive, so the caller's objects may go
// first.
TEST_F(TestCudnnShimV8BuilderOperations, ReferentsMayBeReleasedBeforeBuild)
{
    const Dims xDims = {2, 4, 9, 7};
    Builder builder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    {
        const v8fe::Tensor x = tensor(1, xDims);
        const v8fe::Tensor w = tensor(2, W_DIMS);
        const v8fe::Tensor y = tensor(3, {2, 8, 9, 7});
        const v8fe::ConvDesc conv = convDesc();
        builder.setxDesc(x).setwDesc(w).setyDesc(y).setcDesc(conv);
    }
    std::optional<v8fe::Operation> op;
    const BuildFailure failure = failureOf([&] { op.emplace(builder.build()); });
    ASSERT_FALSE(failure.thrown) << failure.what;
    const BuildFailure lowering
        = failureOf([&] { [[maybe_unused]] const auto built = graph({&*op}); });
    ASSERT_FALSE(lowering.thrown) << lowering.what;

    const auto dims = callOn(hipdnnTensor(1), HIPDNN_ATTR_TENSOR_DIMENSIONS);
    ASSERT_TRUE(dims.has_value());
    EXPECT_EQ(dims->values<std::int64_t>(), xDims);
}

TEST_F(TestCudnnShimV8BuilderOperations, ConvolutionSlotsReachHipdnn)
{
    struct ConvSlots
    {
        const char* row;
        hipdnnBackendDescriptorType_t type;
        std::vector<std::pair<hipdnnBackendAttributeName_t, std::int64_t>> slots;
    };
    const std::vector<ConvSlots> rows = {
        {"ConvolutionForward",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, 1},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, 2},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, 3}}},
        {"ConvolutionBackwardData",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, 1},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, 2},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, 3}}},
        {"ConvolutionBackwardFilter",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, 1},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, 2},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, 3}}},
    };
    for(const ConvSlots& row : rows)
    {
        SCOPED_TRACE(row.row);
        expectBuildsAndLowers(find(row.row), nullptr);
        expectSlotUids(row.type, row.slots);
    }
}

// setxDesc, setwDesc and setyDesc stand in for the unset gradient slots of a backward convolution,
// the way PyTorch builds them.
TEST_F(TestCudnnShimV8BuilderOperations, BackwardConvolutionForwardSettersStandIn)
{
    struct AliasRow
    {
        const char* row;
        hipdnnBackendDescriptorType_t type;
        std::vector<std::pair<hipdnnBackendAttributeName_t, std::int64_t>> slots;
    };
    const std::vector<AliasRow> rows = {
        {"ConvolutionBackwardData",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX, 1},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W, 2},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY, 3}}},
        {"ConvolutionBackwardFilter",
         HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
         {{HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, 1},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, 2},
          {HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, 3}}},
    };
    for(const AliasRow& row : rows)
    {
        SCOPED_TRACE(row.row);
        OpCase aliased = find(row.row);
        aliased.required = {{"Convolution", [](Builder& builder) { builder.setcDesc(convDesc()); }},
                            tensorSlot("X", &Builder::setxDesc, 1, X_DIMS),
                            tensorSlot("W", &Builder::setwDesc, 2, W_DIMS),
                            tensorSlot("Y", &Builder::setyDesc, 3, Y_DIMS)};
        expectBuildsAndLowers(aliased, nullptr);
        expectSlotUids(row.type, row.slots);
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, BackwardConvolutionBothSettersAmbiguous)
{
    const std::vector<std::pair<const char*, Slot>> rows = {
        {"ConvolutionBackwardData", tensorSlot("Y", &Builder::setyDesc, STRAY_UID, Y_DIMS)},
        {"ConvolutionBackwardData", tensorSlot("X", &Builder::setxDesc, STRAY_UID, X_DIMS)},
        {"ConvolutionBackwardFilter", tensorSlot("Y", &Builder::setyDesc, STRAY_UID, Y_DIMS)},
        {"ConvolutionBackwardFilter", tensorSlot("W", &Builder::setwDesc, STRAY_UID, W_DIMS)},
    };
    for(const auto& [row, extra] : rows)
    {
        SCOPED_TRACE(std::string(row) + " with set" + extra.name + "Desc");
        const BuildFailure failure = failureWith(row, extra.set);
        expectStatus(failure, CUDNN_STATUS_BAD_PARAM);
        EXPECT_NE(failure.what.find("Ambiguous specification"), std::string::npos) << failure.what;
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, PointwiseTagIsUpstreamName)
{
    const v8fe::Tensor bias = tensor(2, BIAS_DIMS);
    const v8fe::Operation add = pointwise(
        pointwiseDesc(PointwiseMode_t::ADD), tensor(1, X_DIMS), tensor(3, X_DIMS), &bias);
    const v8fe::Operation activation = relu(4, 5);
    const v8fe::Operation gelu = pointwise(
        pointwiseDesc(PointwiseMode_t::GELU_APPROX_TANH_FWD), tensor(6, X_DIMS), tensor(7, X_DIMS));

    EXPECT_EQ(add.getTag(), pointwiseTag(PointwiseMode_t::ADD, "ADD"));
    EXPECT_EQ(activation.getTag(), pointwiseTag(PointwiseMode_t::RELU_FWD, "RELU_FWD"));
    EXPECT_EQ(gelu.getTag(),
              pointwiseTag(PointwiseMode_t::GELU_APPROX_TANH_FWD, "GELU_APPROX_TANH_FWD"));

    [[maybe_unused]] const auto built = graph({&add});
    expectSlotUids(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                   {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 1},
                    {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, 2},
                    {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 3}});
}

TEST_F(TestCudnnShimV8BuilderOperations, PointwiseBackwardUsesGradientSlots)
{
    expectBuildsAndLowers(find("PointwiseActivationBackward"), nullptr);
    expectSlotUids(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
                   {{HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT, 2},
                    {HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT, 1},
                    {HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT, 3}});
}

TEST_F(TestCudnnShimV8BuilderOperations, ConvFeatureVectorMatchesUpstream)
{
    ConvSpec twoD;
    twoD.x = {2, 4, 9, 7};
    twoD.w = {8, 4, 3, 5};
    twoD.y = {2, 8, 5, 6};
    twoD.padding = {1, 2};
    twoD.dilation = {2, 1};
    twoD.stride = {1, 3};
    twoD.xType = DataType_t::HALF;
    twoD.wType = DataType_t::BFLOAT16;

    ConvSpec threeD;
    threeD.x = {2, 4, 6, 9, 7};
    threeD.w = {8, 4, 3, 3, 5};
    threeD.y = {2, 8, 4, 5, 6};
    threeD.padding = {1, 2, 3};
    threeD.dilation = {2, 1, 3};
    threeD.stride = {1, 3, 2};
    threeD.yType = DataType_t::HALF;

    // Tensors of rank CUDNN_DIM_MAX: setcDesc copies CUDNN_DIM_MAX + 1
    // convolution entries into the builder.
    ConvSpec maxRank;
    maxRank.x = {2, 4, 3, 4, 5, 6, 7, 8};
    maxRank.w = {8, 4, 1, 1, 1, 1, 1, 1};
    maxRank.y = {2, 8, 3, 4, 5, 6, 7, 8};
    maxRank.padding = {1, 2, 3, 4, 5, 6};
    maxRank.dilation = {1, 2, 1, 2, 1, 2};
    maxRank.stride = {3, 1, 2, 1, 2, 1};
    ASSERT_EQ(maxRank.x.size(), static_cast<std::size_t>(CUDNN_DIM_MAX));

    for(const ConvSpec* spec : {&twoD, &threeD, &maxRank})
    {
        SCOPED_TRACE(spec->x.size());
        const v8fe::Operation op = convolutionForward(
            tensor(1, spec->x, false, spec->xType),
            tensor(2, spec->w, false, spec->wType),
            tensor(3, spec->y, false, spec->yType),
            convolution(spec->padding, spec->dilation, spec->stride, spec->computeType));

        ASSERT_EQ(op.get_status(), CUDNN_STATUS_SUCCESS) << op.get_error();
        EXPECT_EQ(op.getFeatureVector(),
                  upstreamConvFeatureVector(
                      DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, *spec));
    }
}

TEST_F(TestCudnnShimV8BuilderOperations, OutputTensorFollowsOpKind)
{
    const v8fe::Tensor convY = tensor(3, Y_DIMS);
    v8fe::Operation conv
        = convolutionForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convY, convDesc());

    const v8fe::Tensor reluY = tensor(5, X_DIMS);
    v8fe::Operation activation
        = pointwise(pointwiseDesc(PointwiseMode_t::RELU_FWD), tensor(4, X_DIMS), reluY);

    const v8fe::Tensor matmulC = tensor(8, {2, 4, 16});
    v8fe::Operation product = Builder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                  .setaMatDesc(tensor(6, {2, 4, 8}))
                                  .setbMatDesc(tensor(7, {2, 8, 16}))
                                  .setcMatDesc(matmulC)
                                  .setmatmulDesc(matmul())
                                  .build();

    EXPECT_EQ(conv.getOutputTensor(), convY.get_desc());
    EXPECT_EQ(activation.getOutputTensor(), reluY.get_desc());
    EXPECT_EQ(product.getOutputTensor(), matmulC.get_desc());
}

TEST_F(TestCudnnShimV8BuilderOperations, GraphTagConcatenatesOps)
{
    const v8fe::Tensor convOut = tensor(3, Y_DIMS, true);
    const v8fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const v8fe::Tensor biasOut = tensor(5, Y_DIMS, true);
    std::vector<v8fe::Operation> ops;
    ops.emplace_back(convolutionForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut, convDesc()));
    ops.emplace_back(pointwise(pointwiseDesc(PointwiseMode_t::ADD), convOut, biasOut, &bias));
    ops.emplace_back(
        pointwise(pointwiseDesc(PointwiseMode_t::RELU_FWD), biasOut, tensor(6, Y_DIMS)));

    v8fe::OperationGraphBuilder builder;
    builder.setHandle(_handle).setOperationGraph(ops);
    const v8fe::OperationGraph opGraph = builder.build();

    ASSERT_EQ(opGraph.get_status(), CUDNN_STATUS_SUCCESS) << opGraph.get_error();
    EXPECT_EQ(opGraph.getOpCount(), 3U);
    EXPECT_EQ(opGraph.getTag(),
              "ConvFwd_" + pointwiseTag(PointwiseMode_t::ADD, "ADD") + "_"
                  + pointwiseTag(PointwiseMode_t::RELU_FWD, "RELU_FWD") + "_");
    EXPECT_FALSE(ops.front().getFeatureVector().empty());
    EXPECT_EQ(opGraph.getFeatureVector(), ops.front().getFeatureVector());
    EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8BuilderOperations, GraphWithoutHandleRejected)
{
    const v8fe::Operation op = relu(1, 2);
    const v8fe::Operation* ops = &op;
    v8fe::OperationGraphBuilder builder;
    builder.setOperationGraph(1, &ops);

    expectStatus(buildFailure(builder), CUDNN_STATUS_BAD_PARAM);
    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());
}

// Over-limit counts are rejected before any operation is copied. A removed check makes the huge
// count throw std::length_error from the feature-vector resize rather than read past the array; a
// check that is off by one or moved after the copy dereferences the null entries and aborts.
TEST_F(TestCudnnShimV8BuilderOperations, GraphOpLimitEnforced)
{
    static_assert(MAX_OPGRAPH_OPS == 250);
    const v8fe::Operation op = relu(1, 2);
    std::vector<const v8fe::Operation*> ops(MAX_OPGRAPH_OPS, &op);
    std::vector<const v8fe::Operation*> tooMany(MAX_OPGRAPH_OPS + 1, nullptr);

    v8fe::OperationGraphBuilder huge;
    expectStatus(failureOf([&] {
                     huge.setOperationGraph(std::numeric_limits<std::int64_t>::max(),
                                            tooMany.data());
                 }),
                 CUDNN_STATUS_BAD_PARAM);

    v8fe::OperationGraphBuilder atLimit;
    EXPECT_FALSE(failureOf([&] { atLimit.setOperationGraph(MAX_OPGRAPH_OPS, ops.data()); }).thrown);

    v8fe::OperationGraphBuilder overLimit;
    expectStatus(
        failureOf([&] { overLimit.setOperationGraph(MAX_OPGRAPH_OPS + 1, tooMany.data()); }),
        CUDNN_STATUS_BAD_PARAM);

    std::vector<v8fe::Operation> vectorOps;
    vectorOps.reserve(static_cast<std::size_t>(MAX_OPGRAPH_OPS) + 1);
    for(std::int64_t i = 0; i < MAX_OPGRAPH_OPS; ++i)
    {
        vectorOps.emplace_back(relu(1, 2));
    }
    v8fe::OperationGraphBuilder vectorAtLimit;
    EXPECT_FALSE(failureOf([&] { vectorAtLimit.setOperationGraph(vectorOps); }).thrown);
    vectorOps.emplace_back(relu(1, 2));
    v8fe::OperationGraphBuilder vectorOverLimit;
    expectStatus(failureOf([&] { vectorOverLimit.setOperationGraph(vectorOps); }),
                 CUDNN_STATUS_BAD_PARAM);

    v8fe::OperationGraphBuilder appended;
    for(std::int64_t i = 0; i < MAX_OPGRAPH_OPS; ++i)
    {
        appended.addOperation(op.get_desc());
    }
    expectStatus(failureOf([&] { appended.addOperation(op.get_desc()); }), CUDNN_STATUS_BAD_PARAM);
}

// Upstream indexes ops[numOps] with numOps still -1 on a fresh builder; the
// shim starts counting at zero.
TEST_F(TestCudnnShimV8BuilderOperations, GraphAddOperationOnFreshBuilder)
{
    const v8fe::Tensor convOut = tensor(3, Y_DIMS, true);
    const v8fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const v8fe::Tensor biasOut = tensor(5, Y_DIMS, true);
    const v8fe::Operation conv
        = convolutionForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut, convDesc());
    const v8fe::Operation add
        = pointwise(pointwiseDesc(PointwiseMode_t::ADD), convOut, biasOut, &bias);
    const v8fe::Operation activation
        = pointwise(pointwiseDesc(PointwiseMode_t::RELU_FWD), biasOut, tensor(6, Y_DIMS));

    v8fe::OperationGraphBuilder builder;
    builder.setHandle(_handle);
    for(const v8fe::Operation* op : {&conv, &add, &activation})
    {
        builder.addOperation(op->get_desc());
    }
    const v8fe::OperationGraph opGraph = builder.build();

    ASSERT_EQ(opGraph.get_status(), CUDNN_STATUS_SUCCESS) << opGraph.get_error();
    EXPECT_EQ(opGraph.getOpCount(), 3U);
    EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8BuilderOperations, GraphEngineCountFromHeuristics)
{
    installHeuristicResults({5, -2, 7});
    const v8fe::Operation op = relu(1, 2);
    const v8fe::OperationGraph opGraph = graph({&op});

    ASSERT_EQ(opGraph.get_status(), CUDNN_STATUS_SUCCESS) << opGraph.get_error();
    EXPECT_EQ(opGraph.getEngineCount(), 3);
}

} // namespace
