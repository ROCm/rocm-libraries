// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported v0.x Operation and OperationGraph builders against the shim's
// backend C API and the mock hipDNN backend.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_OperationGraph.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8MockBackendFixture.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::DataType_t;
using fe::DescriptorType_t;
using fe::PointwiseMode_t;

using Dims = std::vector<std::int64_t>;

const Dims X_DIMS = {2, 4, 8, 8};
const Dims W_DIMS = {8, 4, 3, 3};
const Dims Y_DIMS = {2, 8, 8, 8};

std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

struct Failure
{
    bool thrown = false;
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    std::string what;
};

template <typename Action>
Failure failureOf(Action&& action)
{
    Failure failure;
    try
    {
        std::forward<Action>(action)();
    }
    catch(fe::cudnnException& e)
    {
        failure.thrown = true;
        failure.status = e.getCudnnStatus();
        failure.what = e.what();
    }
    return failure;
}

template <typename Builder>
Failure buildFailure(Builder& builder)
{
    return failureOf([&builder] { [[maybe_unused]] auto built = builder.build(); });
}

// The tag upstream's build_pointwise_op gives: the serialized mode name, or the
// mode's integer when the JSON library is compiled out.
std::string pointwiseTag([[maybe_unused]] PointwiseMode_t mode, [[maybe_unused]] const char* name)
{
#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
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
fe::feature_vector_t upstreamConvFeatureVector(DescriptorType_t opType, const ConvSpec& spec)
{
    fe::feature_vector_t expected;
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

class TestCudnnShimV8OperationBuilders : public hipdnn_shim_test::ShimV8MockBackendFixture
{
protected:
    void SetUp() override
    {
        ShimV8MockBackendFixture::SetUp();
        lastError();
    }

    static fe::Tensor tensor(std::int64_t uid,
                             const Dims& dims,
                             bool isVirtual = false,
                             DataType_t dataType = DataType_t::FLOAT)
    {
        const Dims strides = packedStrides(dims);
        return fe::TensorBuilder()
            .setDataType(dataType)
            .setDim(static_cast<std::int64_t>(dims.size()), dims.data())
            .setStride(static_cast<std::int64_t>(strides.size()), strides.data())
            .setId(uid)
            .setAlignment(16)
            .setVirtual(isVirtual)
            .build();
    }

    static fe::ConvDesc convolution(const Dims& padding,
                                    const Dims& dilation,
                                    const Dims& stride,
                                    DataType_t computeType = DataType_t::FLOAT)
    {
        const auto count = static_cast<std::int64_t>(padding.size());
        return fe::ConvDescBuilder()
            .setComputeType(computeType)
            .setMathMode(CUDNN_CROSS_CORRELATION)
            .setSpatialDimCount(count)
            .setPrePadding(count, padding.data())
            .setPostPadding(count, padding.data())
            .setDilation(count, dilation.data())
            .setSpatialStride(count, stride.data())
            .build();
    }

    static fe::ConvDesc convolution2d(DataType_t computeType = DataType_t::FLOAT)
    {
        return convolution({1, 1}, {1, 1}, {1, 1}, computeType);
    }

    static fe::PointWiseDesc pointwise(PointwiseMode_t mode)
    {
        return fe::PointWiseDescBuilder().setMode(mode).setComputeType(DataType_t::FLOAT).build();
    }

    static fe::MatMulDesc matmul()
    {
        return fe::MatMulDescBuilder().setComputeType(DataType_t::FLOAT).build();
    }

    static fe::ResampleDesc_v8 averagePool()
    {
        const std::array<std::int64_t, 2> window = {2, 2};
        const std::array<std::int64_t, 2> zeros = {0, 0};
        return fe::ResampleDescBuilder_v8()
            .setResampleMode(fe::ResampleMode_t::AVGPOOL_INCLUDE_PADDING)
            .setComputeType(DataType_t::FLOAT)
            .setPaddingMode(fe::PaddingMode_t::ZERO_PAD)
            .setSpatialDim(2, window.data())
            .setPrePadding(2, zeros.data())
            .setPostPadding(2, zeros.data())
            .setSpatialStride(2, window.data())
            .build();
    }

    static fe::Operation convForward(const fe::Tensor& x,
                                     const fe::Tensor& w,
                                     const fe::Tensor& y,
                                     const fe::ConvDesc& conv)
    {
        return fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
            .setxDesc(x)
            .setwDesc(w)
            .setyDesc(y)
            .setcDesc(conv)
            .build();
    }

    static fe::Operation pointwiseOp(PointwiseMode_t mode,
                                     const fe::Tensor& x,
                                     const fe::Tensor* b,
                                     const fe::Tensor& y)
    {
        fe::OperationBuilder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
        builder.setpwDesc(pointwise(mode)).setxDesc(x).setyDesc(y);
        if(b != nullptr)
        {
            builder.setbDesc(*b);
        }
        return builder.build();
    }

    static fe::Operation relu(std::int64_t xUid, std::int64_t yUid)
    {
        return pointwiseOp(
            PointwiseMode_t::RELU_FWD, tensor(xUid, X_DIMS), nullptr, tensor(yUid, X_DIMS));
    }

    // UID of the tensor an operation holds in a descriptor-valued slot, read
    // through an in-place GET into a caller-created tensor descriptor.
    static std::int64_t slotUid(cudnnBackendDescriptor_t op, cudnnBackendAttributeName_t name)
    {
        cudnnBackendDescriptor_t target = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &target),
                  CUDNN_STATUS_SUCCESS);
        std::int64_t count = 0;
        EXPECT_EQ(
            cudnnBackendGetAttribute(
                op, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &count, static_cast<void*>(&target)),
            CUDNN_STATUS_SUCCESS)
            << lastError();
        EXPECT_EQ(count, 1);
        std::int64_t uid = -1;
        EXPECT_EQ(cudnnBackendGetAttribute(target,
                                           CUDNN_ATTR_TENSOR_UNIQUE_ID,
                                           CUDNN_TYPE_INT64,
                                           1,
                                           &count,
                                           static_cast<void*>(&uid)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        EXPECT_EQ(cudnnBackendDestroyDescriptor(target), CUDNN_STATUS_SUCCESS);
        return uid;
    }

    static bool slotSet(cudnnBackendDescriptor_t op, cudnnBackendAttributeName_t name)
    {
        std::int64_t count = 0;
        const cudnnStatus_t status
            = cudnnBackendGetAttribute(op, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);
        lastError();
        return status == CUDNN_STATUS_SUCCESS && count > 0;
    }

    // nullopt when the attribute is absent or stored with another type.
    template <typename T>
    static std::optional<T> scalarAs(cudnnBackendDescriptor_t descriptor,
                                     cudnnBackendAttributeName_t name,
                                     cudnnBackendAttributeType_t type)
    {
        T value{};
        std::int64_t count = 0;
        const cudnnStatus_t status = cudnnBackendGetAttribute(
            descriptor, name, type, 1, &count, static_cast<void*>(&value));
        lastError();
        if(status != CUDNN_STATUS_SUCCESS || count != 1)
        {
            return std::nullopt;
        }
        return value;
    }
};

TEST_F(TestCudnnShimV8OperationBuilders, ConvForwardSetsShimAttributes)
{
    const fe::Operation op
        = convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), tensor(3, Y_DIMS), convolution2d());

    ASSERT_EQ(op.get_status(), CUDNN_STATUS_SUCCESS) << op.get_error();
    cudnnBackendDescriptor_t raw = op.get_raw_desc();
    EXPECT_EQ(slotUid(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X), 1);
    EXPECT_EQ(slotUid(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W), 2);
    EXPECT_EQ(slotUid(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y), 3);
    EXPECT_TRUE(slotSet(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC));
    EXPECT_EQ(
        scalarAs<float>(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT),
        1.0F);
    EXPECT_EQ(scalarAs<float>(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT),
              0.0F);
    EXPECT_FALSE(
        scalarAs<double>(raw, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE));
    EXPECT_EQ(op.getTag(), "ConvFwd");
    // Building an operation stays inside the shim.
    EXPECT_TRUE(setAttributes().empty());

    const fe::Operation asDouble = convForward(
        tensor(1, X_DIMS), tensor(2, W_DIMS), tensor(3, Y_DIMS), convolution2d(DataType_t::DOUBLE));

    ASSERT_EQ(asDouble.get_status(), CUDNN_STATUS_SUCCESS) << asDouble.get_error();
    cudnnBackendDescriptor_t rawDouble = asDouble.get_raw_desc();
    EXPECT_EQ(scalarAs<double>(
                  rawDouble, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_DOUBLE),
              1.0);
    EXPECT_EQ(scalarAs<double>(
                  rawDouble, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_DOUBLE),
              0.0);
    EXPECT_FALSE(scalarAs<float>(
        rawDouble, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT));
}

TEST_F(TestCudnnShimV8OperationBuilders, ConvBackwardDataAndFilterBuild)
{
    const fe::Operation dgrad
        = fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR)
              .setdxDesc(tensor(1, X_DIMS))
              .setwDesc(tensor(2, W_DIMS))
              .setdyDesc(tensor(3, Y_DIMS))
              .setcDesc(convolution2d())
              .build();

    ASSERT_EQ(dgrad.get_status(), CUDNN_STATUS_SUCCESS) << dgrad.get_error();
    EXPECT_EQ(dgrad.getTag(), "ConvBwdData");
    EXPECT_EQ(slotUid(dgrad.get_raw_desc(), CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX), 1);
    EXPECT_EQ(slotUid(dgrad.get_raw_desc(), CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W), 2);
    EXPECT_EQ(slotUid(dgrad.get_raw_desc(), CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY), 3);
    EXPECT_EQ(scalarAs<float>(dgrad.get_raw_desc(),
                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA,
                              CUDNN_TYPE_FLOAT),
              1.0F);

    const fe::Operation wgrad
        = fe::OperationBuilder(DescriptorType_t::OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR)
              .setxDesc(tensor(4, X_DIMS))
              .setdwDesc(tensor(5, W_DIMS))
              .setdyDesc(tensor(6, Y_DIMS))
              .setcDesc(convolution2d())
              .build();

    ASSERT_EQ(wgrad.get_status(), CUDNN_STATUS_SUCCESS) << wgrad.get_error();
    EXPECT_EQ(wgrad.getTag(), "ConvBwdFilter");
    EXPECT_EQ(slotUid(wgrad.get_raw_desc(), CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X), 4);
    EXPECT_EQ(slotUid(wgrad.get_raw_desc(), CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW), 5);
    EXPECT_EQ(slotUid(wgrad.get_raw_desc(), CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY), 6);
    EXPECT_EQ(scalarAs<float>(wgrad.get_raw_desc(),
                              CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA,
                              CUDNN_TYPE_FLOAT),
              0.0F);
}

TEST_F(TestCudnnShimV8OperationBuilders, PointwiseTagIsUpstreamName)
{
    const fe::Tensor bias = tensor(2, {1, 4, 1, 1});
    const fe::Operation add
        = pointwiseOp(PointwiseMode_t::ADD, tensor(1, X_DIMS), &bias, tensor(3, X_DIMS));
    const fe::Operation activation = relu(4, 5);
    const fe::Operation gelu = pointwiseOp(
        PointwiseMode_t::GELU_APPROX_TANH_FWD, tensor(6, X_DIMS), nullptr, tensor(7, X_DIMS));

    ASSERT_EQ(add.get_status(), CUDNN_STATUS_SUCCESS) << add.get_error();
    ASSERT_EQ(activation.get_status(), CUDNN_STATUS_SUCCESS) << activation.get_error();
    ASSERT_EQ(gelu.get_status(), CUDNN_STATUS_SUCCESS) << gelu.get_error();
    EXPECT_EQ(add.getTag(), pointwiseTag(PointwiseMode_t::ADD, "ADD"));
    EXPECT_EQ(activation.getTag(), pointwiseTag(PointwiseMode_t::RELU_FWD, "RELU_FWD"));
    EXPECT_EQ(gelu.getTag(),
              pointwiseTag(PointwiseMode_t::GELU_APPROX_TANH_FWD, "GELU_APPROX_TANH_FWD"));
    EXPECT_EQ(slotUid(add.get_raw_desc(), CUDNN_ATTR_OPERATION_POINTWISE_BDESC), 2);
}

TEST_F(TestCudnnShimV8OperationBuilders, PointwiseBackwardUsesGradientSlots)
{
    const fe::Operation op = fe::OperationBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR)
                                 .setpwDesc(pointwise(PointwiseMode_t::RELU_BWD))
                                 .setxDesc(tensor(1, X_DIMS))
                                 .setdyDesc(tensor(2, X_DIMS))
                                 .setdxDesc(tensor(3, X_DIMS))
                                 .build();

    ASSERT_EQ(op.get_status(), CUDNN_STATUS_SUCCESS) << op.get_error();
    cudnnBackendDescriptor_t raw = op.get_raw_desc();
    EXPECT_EQ(slotUid(raw, CUDNN_ATTR_OPERATION_POINTWISE_XDESC), 1);
    EXPECT_EQ(slotUid(raw, CUDNN_ATTR_OPERATION_POINTWISE_DYDESC), 2);
    EXPECT_EQ(slotUid(raw, CUDNN_ATTR_OPERATION_POINTWISE_DXDESC), 3);
    EXPECT_FALSE(slotSet(raw, CUDNN_ATTR_OPERATION_POINTWISE_YDESC));
}

TEST_F(TestCudnnShimV8OperationBuilders, ScaledAddNotSupported)
{
    const fe::Tensor bias = tensor(2, {1, 4, 1, 1});
    fe::OperationBuilder builder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
    builder.setpwDesc(pointwise(PointwiseMode_t::ADD))
        .setxDesc(tensor(1, X_DIMS))
        .setbDesc(bias)
        .setyDesc(tensor(3, X_DIMS))
        .setAlpha2(0.5);

    const Failure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("cudnnFinalize Failed"), std::string::npos) << failure.what;
    EXPECT_NE(failure.what.find("CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2"), std::string::npos)
        << failure.what;
}

TEST_F(TestCudnnShimV8OperationBuilders, MatmulOverrideNotSupported)
{
    const Dims aDims = {2, 4, 8};
    const Dims bDims = {2, 8, 16};
    const Dims cDims = {2, 4, 16};
    const Dims overrideDims = {2, 1, 1};

    const fe::Operation plain = fe::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                    .setaMatDesc(tensor(1, aDims))
                                    .setbMatDesc(tensor(2, bDims))
                                    .setcMatDesc(tensor(3, cDims))
                                    .setmatmulDesc(matmul())
                                    .build();
    ASSERT_EQ(plain.get_status(), CUDNN_STATUS_SUCCESS) << plain.get_error();
    EXPECT_EQ(plain.getTag(), "Matmul");

    using Setter = fe::OperationBuilder& (fe::OperationBuilder::*)(const fe::Tensor&);
    const std::array<std::pair<Setter, const char*>, 3> overrides = {{
        {&fe::OperationBuilder::setmOverrideDesc,
         "CUDNN_ATTR_OPERATION_MATMUL_GEMM_M_OVERRIDE_DESC"},
        {&fe::OperationBuilder::setnOverrideDesc,
         "CUDNN_ATTR_OPERATION_MATMUL_GEMM_N_OVERRIDE_DESC"},
        {&fe::OperationBuilder::setkOverrideDesc,
         "CUDNN_ATTR_OPERATION_MATMUL_GEMM_K_OVERRIDE_DESC"},
    }};
    for(const auto& [setter, attribute] : overrides)
    {
        SCOPED_TRACE(attribute);
        fe::OperationBuilder builder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR);
        builder.setaMatDesc(tensor(1, aDims))
            .setbMatDesc(tensor(2, bDims))
            .setcMatDesc(tensor(3, cDims))
            .setmatmulDesc(matmul());
        (builder.*setter)(tensor(4, overrideDims, false, DataType_t::INT64));

        const Failure failure = buildFailure(builder);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
        EXPECT_NE(failure.what.find(attribute), std::string::npos) << failure.what;
    }
}

TEST_F(TestCudnnShimV8OperationBuilders, ResampleBackwardXyNotSupported)
{
    const Dims pooled = {2, 4, 4, 4};

    const fe::Operation plain
        = fe::OperationBuilder(DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR)
              .setResampleDesc(averagePool())
              .setdyDesc(tensor(1, pooled))
              .setdxDesc(tensor(2, X_DIMS))
              .build();
    ASSERT_EQ(plain.get_status(), CUDNN_STATUS_SUCCESS) << plain.get_error();
    EXPECT_EQ(plain.getTag(), "Resample_bwd");
    EXPECT_EQ(scalarAs<double>(
                  plain.get_raw_desc(), CUDNN_ATTR_OPERATION_RESAMPLE_BWD_ALPHA, CUDNN_TYPE_DOUBLE),
              1.0);

    for(const bool setX : {true, false})
    {
        const char* attribute = setX ? "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_XDESC"
                                     : "CUDNN_ATTR_OPERATION_RESAMPLE_BWD_YDESC";
        SCOPED_TRACE(attribute);
        fe::OperationBuilder builder(DescriptorType_t::OPERATION_RESAMPLE_BWD_DESCRIPTOR);
        builder.setResampleDesc(averagePool())
            .setdyDesc(tensor(1, pooled))
            .setdxDesc(tensor(2, X_DIMS));
        if(setX)
        {
            builder.setxDesc(tensor(3, X_DIMS));
        }
        else
        {
            builder.setyDesc(tensor(3, pooled));
        }

        const Failure failure = buildFailure(builder);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
        EXPECT_NE(failure.what.find(attribute), std::string::npos) << failure.what;
    }
}

TEST_F(TestCudnnShimV8OperationBuilders, UnloweredOpsNotSupported)
{
    const Dims channel = {1, 4, 1, 1};

    fe::OperationBuilder genStats(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR);
    genStats.setxDesc(tensor(1, X_DIMS))
        .setSumDesc(tensor(2, channel))
        .setSqSumDesc(tensor(3, channel))
        .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
        .setMathPrecision(CUDNN_DATA_FLOAT);

    fe::OperationBuilder bnFinalize(DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR);
    bnFinalize.setSumDesc(tensor(1, channel))
        .setSqSumDesc(tensor(2, channel))
        .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
        .setMathPrecision(CUDNN_DATA_FLOAT);

    fe::OperationBuilder bnBackwardWeights(DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR);
    bnBackwardWeights.setxDesc(tensor(1, X_DIMS))
        .setdyDesc(tensor(2, X_DIMS))
        .setSavedMeanAndInvVar(tensor(3, channel), tensor(4, channel))
        .setMathPrecision(CUDNN_DATA_FLOAT);

    const fe::RngDesc rngDesc = fe::RngDescBuilder()
                                    .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                    .setBernoulliDistProbability(0.5)
                                    .build();
    ASSERT_EQ(rngDesc.get_status(), CUDNN_STATUS_SUCCESS) << rngDesc.get_error();
    fe::OperationBuilder rng(DescriptorType_t::OPERATION_RNG_DESCRIPTOR);
    rng.setyDesc(tensor(1, X_DIMS)).setRngDesc(rngDesc).setSeed(7);

    fe::OperationBuilder reshape(DescriptorType_t::OPERATION_RESHAPE_DESCRIPTOR);
    reshape.setxDesc(tensor(1, X_DIMS)).setyDesc(tensor(2, {2, 256}));

    fe::OperationBuilder pagedCacheLoad(DescriptorType_t::OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR);
    pagedCacheLoad.setcontainerDesc(tensor(1, {8, 2, 16, 64}))
        .setpageTableDesc(tensor(2, {2, 1, 4, 1}, false, DataType_t::INT32))
        .setsequenceDesc(tensor(3, {2, 1, 1, 1}, false, DataType_t::INT32))
        .setyDesc(tensor(4, {2, 2, 64, 64}));

    const std::array<std::pair<fe::OperationBuilder*, const char*>, 6> builders = {{
        {&genStats, "CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR"},
        {&bnFinalize, "CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR"},
        {&bnBackwardWeights, "CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR"},
        {&rng, "CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR"},
        {&reshape, "CUDNN_BACKEND_OPERATION_RESHAPE_DESCRIPTOR"},
        {&pagedCacheLoad, "CUDNN_BACKEND_OPERATION_PAGED_CACHE_LOAD_DESCRIPTOR"},
    }};
    for(const auto& [builder, type] : builders)
    {
        SCOPED_TRACE(type);
        const Failure failure = buildFailure(*builder);
        ASSERT_TRUE(failure.thrown);
        EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
        EXPECT_NE(failure.what.find("cudnnFinalize Failed"), std::string::npos) << failure.what;
        EXPECT_NE(failure.what.find(std::string(type) + " is not supported"), std::string::npos)
            << failure.what;
    }
}

TEST_F(TestCudnnShimV8OperationBuilders, WrongKindSetterRejected)
{
    fe::OperationBuilder pointwiseBuilder(DescriptorType_t::OPERATION_POINTWISE_DESCRIPTOR);
    const fe::Tensor w = tensor(1, W_DIMS);
    const Failure onPointwise = failureOf([&] { pointwiseBuilder.setwDesc(w); });
    ASSERT_TRUE(onPointwise.thrown);
    EXPECT_EQ(onPointwise.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(onPointwise.what.find("Non Convolution operation does not need wTensor"),
              std::string::npos)
        << onPointwise.what;

    fe::OperationBuilder convBuilder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    const fe::Tensor b = tensor(2, {1, 8, 1, 1});
    const Failure onConv = failureOf([&] { convBuilder.setbDesc(b); });
    ASSERT_TRUE(onConv.thrown);
    EXPECT_EQ(onConv.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(onConv.what.find("Non Pointwise operation does not need bTensor"), std::string::npos)
        << onConv.what;
}

TEST_F(TestCudnnShimV8OperationBuilders, ConvFeatureVectorMatchesUpstream)
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
        const fe::Operation op = convForward(
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

TEST_F(TestCudnnShimV8OperationBuilders, OutputTensorFollowsOpKind)
{
    const fe::Tensor convY = tensor(3, Y_DIMS);
    fe::Operation conv = convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convY, convolution2d());

    const fe::Tensor reluY = tensor(5, X_DIMS);
    fe::Operation activation
        = pointwiseOp(PointwiseMode_t::RELU_FWD, tensor(4, X_DIMS), nullptr, reluY);

    const fe::Tensor matmulC = tensor(8, {2, 4, 16});
    fe::Operation product = fe::OperationBuilder(DescriptorType_t::OPERATION_MATMUL_DESCRIPTOR)
                                .setaMatDesc(tensor(6, {2, 4, 8}))
                                .setbMatDesc(tensor(7, {2, 8, 16}))
                                .setcMatDesc(matmulC)
                                .setmatmulDesc(matmul())
                                .build();

    ASSERT_EQ(conv.get_status(), CUDNN_STATUS_SUCCESS) << conv.get_error();
    ASSERT_EQ(activation.get_status(), CUDNN_STATUS_SUCCESS) << activation.get_error();
    ASSERT_EQ(product.get_status(), CUDNN_STATUS_SUCCESS) << product.get_error();
    EXPECT_EQ(conv.getOutputTensor(), convY.get_desc());
    EXPECT_EQ(activation.getOutputTensor(), reluY.get_desc());
    EXPECT_EQ(product.getOutputTensor(), matmulC.get_desc());
}

TEST_F(TestCudnnShimV8OperationBuilders, OptionalOperationEmplaces)
{
    fe::OperationBuilder builder(DescriptorType_t::OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    builder.setxDesc(tensor(1, X_DIMS))
        .setwDesc(tensor(2, W_DIMS))
        .setyDesc(tensor(3, Y_DIMS))
        .setcDesc(convolution2d());

    std::optional<fe::Operation> op;
    op.emplace(builder.build());

    ASSERT_TRUE(op.has_value());
    ASSERT_EQ(op->get_status(), CUDNN_STATUS_SUCCESS) << op->get_error();
    EXPECT_EQ(op->getTag(), "ConvFwd");

    const fe::Operation* ops = &*op;
    fe::OperationGraphBuilder graphBuilder;
    graphBuilder.setHandle(_handle).setOperationGraph(1, &ops);
    const fe::OperationGraph graph = graphBuilder.build();
    EXPECT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
    EXPECT_EQ(graph.getTag(), "ConvFwd_");
    EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8OperationBuilders, GraphTagConcatenatesOps)
{
    const fe::Tensor convOut = tensor(3, Y_DIMS, true);
    const fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const fe::Tensor biasOut = tensor(5, Y_DIMS, true);
    std::vector<fe::Operation> ops;
    ops.emplace_back(convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut, convolution2d()));
    ops.emplace_back(pointwiseOp(PointwiseMode_t::ADD, convOut, &bias, biasOut));
    ops.emplace_back(pointwiseOp(PointwiseMode_t::RELU_FWD, biasOut, nullptr, tensor(6, Y_DIMS)));

    fe::OperationGraphBuilder builder;
    builder.setHandle(_handle).setOperationGraph(ops);
    const fe::OperationGraph graph = builder.build();

    ASSERT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
    EXPECT_EQ(graph.getOpCount(), 3U);
    EXPECT_EQ(graph.getTag(),
              "ConvFwd_" + pointwiseTag(PointwiseMode_t::ADD, "ADD") + "_"
                  + pointwiseTag(PointwiseMode_t::RELU_FWD, "RELU_FWD") + "_");
    EXPECT_FALSE(ops.front().getFeatureVector().empty());
    EXPECT_EQ(graph.getFeatureVector(), ops.front().getFeatureVector());
    EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8OperationBuilders, GraphWithoutHandleRejected)
{
    const fe::Operation op = relu(1, 2);
    const fe::Operation* ops = &op;
    fe::OperationGraphBuilder builder;
    builder.setOperationGraph(1, &ops);

    const Failure failure = buildFailure(builder);
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("CUDNN_ATTR_OPERATIONGRAPH_HANDLE"), std::string::npos)
        << failure.what;
    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());
}

TEST_F(TestCudnnShimV8OperationBuilders, GraphOpLimitEnforced)
{
    static_assert(MAX_OPGRAPH_OPS == 250);
    const fe::Operation op = relu(1, 2);
    std::vector<const fe::Operation*> ops(MAX_OPGRAPH_OPS, &op);
    // Null entries crash the case if the limit check ever moves after the copy.
    std::vector<const fe::Operation*> tooMany(MAX_OPGRAPH_OPS + 1, nullptr);

    fe::OperationGraphBuilder atLimit;
    EXPECT_FALSE(failureOf([&] { atLimit.setOperationGraph(MAX_OPGRAPH_OPS, ops.data()); }).thrown);

    fe::OperationGraphBuilder overLimit;
    const Failure failure
        = failureOf([&] { overLimit.setOperationGraph(MAX_OPGRAPH_OPS + 1, tooMany.data()); });
    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_BAD_PARAM);
    EXPECT_NE(failure.what.find("numOps exceeds MAX_OPGRAPH_OPS"), std::string::npos)
        << failure.what;

    fe::OperationGraphBuilder appended;
    for(std::int64_t i = 0; i < MAX_OPGRAPH_OPS; ++i)
    {
        appended.addOperation(op.get_desc());
    }
    const Failure appendFailure = failureOf([&] { appended.addOperation(op.get_desc()); });
    ASSERT_TRUE(appendFailure.thrown);
    EXPECT_EQ(appendFailure.status, CUDNN_STATUS_BAD_PARAM);

    EXPECT_TRUE(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).empty());
}

// Upstream indexes ops[numOps] with numOps still -1 on a fresh builder; the
// shim starts counting at zero.
TEST_F(TestCudnnShimV8OperationBuilders, GraphAddOperationOnFreshBuilder)
{
    const fe::Tensor convOut = tensor(3, Y_DIMS, true);
    const fe::Tensor bias = tensor(4, {1, 8, 1, 1});
    const fe::Tensor biasOut = tensor(5, Y_DIMS, true);
    const fe::Operation conv
        = convForward(tensor(1, X_DIMS), tensor(2, W_DIMS), convOut, convolution2d());
    const fe::Operation add = pointwiseOp(PointwiseMode_t::ADD, convOut, &bias, biasOut);
    const fe::Operation activation
        = pointwiseOp(PointwiseMode_t::RELU_FWD, biasOut, nullptr, tensor(6, Y_DIMS));

    fe::OperationGraphBuilder builder;
    builder.setHandle(_handle);
    for(const fe::Operation* op : {&conv, &add, &activation})
    {
        builder.addOperation(op->get_desc());
    }
    const fe::OperationGraph graph = builder.build();

    ASSERT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
    EXPECT_EQ(graph.getOpCount(), 3U);
    EXPECT_EQ(created(HIPDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR).size(), 1U);
}

TEST_F(TestCudnnShimV8OperationBuilders, GraphEngineCountFromHeuristics)
{
    installHeuristicResults({5, -2, 7});
    const fe::Operation op = relu(1, 2);
    const fe::Operation* ops = &op;
    fe::OperationGraphBuilder builder;
    builder.setHandle(_handle).setOperationGraph(1, &ops);
    const fe::OperationGraph graph = builder.build();

    ASSERT_EQ(graph.get_status(), CUDNN_STATUS_SUCCESS) << graph.get_error();
    EXPECT_EQ(graph.getEngineCount(), 3);
}

} // namespace
