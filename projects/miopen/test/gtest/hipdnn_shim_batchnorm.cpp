// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Batch-norm inference through miopenBatchNormalizationForwardInference, checked against a CPU
// reference in both forwarding modes, and the problems the forwarded path declines.

#include "hipdnn_shim_helpers.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

using namespace hipdnn_shim_test;

namespace {

constexpr double kEpsilon = 1e-5;

// The parameter tensor as PyTorch describes it: lengths 1 x C x 1 ... with packed strides,
// whatever x's layout, or 1 x C x spatial... for per-activation mode.
tensor<float> MakeParams(const std::vector<std::size_t>& x_lengths, bool per_activation)
{
    std::vector<std::size_t> lengths(x_lengths.size(), 1);
    lengths[1] = x_lengths[1];
    if(per_activation)
        std::copy(x_lengths.begin() + 2, x_lengths.end(), lengths.begin() + 2);
    return tensor<float>{lengths};
}

// Whole numbers, so the expected result depends only on the square root. Variances are
// perfect squares near 1 to 16 and scales differ per element, so a kernel that mixes up the
// parameters or channels cannot match.
struct Params
{
    tensor<float> scale;
    tensor<float> bias;
    tensor<float> mean;
    tensor<float> variance;

    explicit Params(const tensor<float>& like)
        : scale(like.desc), bias(like.desc), mean(like.desc), variance(like.desc)
    {
        for(std::size_t i = 0; i < scale.data.size(); ++i)
        {
            scale.data[i]    = static_cast<float>(1 + i % 3);
            bias.data[i]     = static_cast<float>(static_cast<int>(i % 7) - 3);
            mean.data[i]     = static_cast<float>(i % 5);
            variance.data[i] = static_cast<float>((1 + i % 4) * (1 + i % 4));
        }
    }
};

// The element's position in the tensor's buffer, from its strides; GetIndex takes no vector.
std::size_t Offset(const miopen::TensorDescriptor& desc, const std::vector<std::size_t>& index)
{
    std::size_t offset = 0;
    for(std::size_t d = 0; d < index.size(); ++d)
        offset += index[d] * desc.GetStrides()[d];
    return offset;
}

// Index of x's element (n, c, rest...) in a parameter tensor with the same rank: channel only
// for spatial, channel and spatial position for per-activation.
std::size_t ParamIndex(const tensor<float>& params, std::vector<std::size_t> index)
{
    index[0] = 0;
    for(std::size_t d = 2; d < index.size(); ++d)
    {
        if(params.desc.GetLengths()[d] == 1)
            index[d] = 0;
    }
    return Offset(params.desc, index);
}

template <class T>
tensor<T> Reference(const tensor<T>& x,
                    const Params& p,
                    const tensor<float>& mean,
                    const tensor<float>& variance,
                    double epsilon)
{
    tensor<T> ref{x.desc};
    const auto& lengths = x.desc.GetLengths();
    std::vector<std::size_t> index(lengths.size(), 0);
    const std::size_t total = x.desc.GetElementSize();
    for(std::size_t flat = 0; flat < total; ++flat)
    {
        std::size_t rest = flat;
        for(std::size_t d = lengths.size(); d-- > 0;)
        {
            index[d] = rest % lengths[d];
            rest /= lengths[d];
        }
        const std::size_t pi = ParamIndex(p.scale, index);
        const double v       = static_cast<double>(x.data[Offset(x.desc, index)]);
        const double normalized =
            (v - mean.data[pi]) / std::sqrt(static_cast<double>(variance.data[pi]) + epsilon);
        ref.data[Offset(ref.desc, index)] =
            static_cast<T>(static_cast<float>(p.scale.data[pi] * normalized + p.bias.data[pi]));
    }
    return ref;
}

// The mean and biased variance of each channel over every other axis, which is what MIOpen
// normalizes by when it is given no running statistics.
template <class T>
void BatchStatistics(const tensor<T>& x, tensor<float>& mean, tensor<float>& variance)
{
    const auto& lengths      = x.desc.GetLengths();
    const std::size_t c_size = lengths[1];
    std::vector<double> sum(c_size, 0.0);
    std::vector<double> sum_sq(c_size, 0.0);
    std::size_t per_channel = 0;
    std::vector<std::size_t> index(lengths.size(), 0);
    const std::size_t total = x.desc.GetElementSize();
    for(std::size_t flat = 0; flat < total; ++flat)
    {
        std::size_t rest = flat;
        for(std::size_t d = lengths.size(); d-- > 0;)
        {
            index[d] = rest % lengths[d];
            rest /= lengths[d];
        }
        const double v = static_cast<double>(x.data[Offset(x.desc, index)]);
        sum[index[1]] += v;
        sum_sq[index[1]] += v * v;
    }
    per_channel = total / c_size;
    for(std::size_t c = 0; c < c_size; ++c)
    {
        const double m   = sum[c] / static_cast<double>(per_channel);
        mean.data[c]     = static_cast<float>(m);
        variance.data[c] = static_cast<float>(sum_sq[c] / static_cast<double>(per_channel) - m * m);
    }
}

struct Call
{
    miopenBatchNormMode_t mode = miopenBNSpatial;
    const float* alpha         = &kOne;
    const float* beta          = &kZero;
    bool with_mean             = true;
    bool with_variance         = true;
    double epsilon             = kEpsilon;
};

// Runs the public entry point. y holds the result when the call succeeds. The tensors are
// non-const because the C API takes `const miopenTensorDescriptor_t`, a const pointer to a
// non-const descriptor.
template <class T, class TY = T>
CallResult RunInference(const Call& call, tensor<T>& x, Params& p, tensor<TY>& y)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    auto x_dev     = handle_deref.Write(x.data);
    auto y_dev     = handle_deref.Write(y.data);
    auto scale_dev = handle_deref.Write(p.scale.data);
    auto bias_dev  = handle_deref.Write(p.bias.data);
    auto mean_dev  = handle_deref.Write(p.mean.data);
    auto var_dev   = handle_deref.Write(p.variance.data);

    auto result = CaptureCall([&] {
        return miopenBatchNormalizationForwardInference(handle,
                                                        call.mode,
                                                        const_cast<float*>(call.alpha),
                                                        const_cast<float*>(call.beta),
                                                        &x.desc,
                                                        x_dev.get(),
                                                        &y.desc,
                                                        y_dev.get(),
                                                        &p.scale.desc,
                                                        scale_dev.get(),
                                                        bias_dev.get(),
                                                        call.with_mean ? mean_dev.get() : nullptr,
                                                        call.with_variance ? var_dev.get()
                                                                           : nullptr,
                                                        call.epsilon);
    });
    // A call that was served by mistake still has to finish before its buffers are freed.
    handle_deref.Finish();
    if(result.status == miopenStatusSuccess)
        y.data = handle_deref.Read<TY>(y_dev, y.data.size());
    return result;
}

template <class T>
void CheckServed(miopenTensorLayout_t layout, const std::vector<std::size_t>& lengths)
{
    tensor<T> x = MakeFilled<T>(layout, lengths, {}, 17);
    Params p{MakeParams(lengths, false)};
    tensor<T> y{layout, lengths};

    const auto result = RunInference(Call{}, x, p, y);
    ASSERT_EQ(result.status, miopenStatusSuccess) << result.message << '\n' << result.logged;
    CheckWithinTolerance(Reference(x, p, p.mean, p.variance, kEpsilon), y, "batch-norm result");
}

struct ServedCase
{
    const char* name;
    miopenTensorLayout_t layout;
    std::vector<std::size_t> lengths;
};

void PrintTo(const ServedCase& c, std::ostream* os) { *os << c.name; }

std::string CaseName(const ::testing::TestParamInfo<ServedCase>& info) { return info.param.name; }

// NCHW and NHWC, as the strides say, and the 5-D layouts, which MIOpen serves by folding depth
// into height.
std::vector<ServedCase> ServedCases()
{
    return {
        {"Nchw", miopenTensorNCHW, {2, 6, 5, 7}},
        {"Nhwc", miopenTensorNHWC, {2, 6, 5, 7}},
        {"Ncdhw", miopenTensorNCDHW, {2, 4, 3, 5, 6}},
        {"Ndhwc", miopenTensorNDHWC, {2, 4, 3, 5, 6}},
    };
}

} // namespace

class GPU_HipdnnShimBatchNormInfer_FP32 : public ::testing::TestWithParam<ServedCase>
{
};

TEST_P(GPU_HipdnnShimBatchNormInfer_FP32, MatchesCpuReference)
{
    CheckServed<float>(GetParam().layout, GetParam().lengths);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_HipdnnShimBatchNormInfer_FP32,
                         ::testing::ValuesIn(ServedCases()),
                         CaseName);

// fp16 and bf16 x and y with fp32 parameters, as PyTorch passes them under autocast or .half().
class GPU_HipdnnShimBatchNormInfer_FP16 : public ::testing::TestWithParam<ServedCase>
{
};

TEST_P(GPU_HipdnnShimBatchNormInfer_FP16, MatchesCpuReference)
{
    CheckServed<half_float::half>(GetParam().layout, GetParam().lengths);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_HipdnnShimBatchNormInfer_FP16,
                         ::testing::ValuesIn(ServedCases()),
                         CaseName);

class GPU_HipdnnShimBatchNormInfer_BFP16 : public ::testing::TestWithParam<ServedCase>
{
};

TEST_P(GPU_HipdnnShimBatchNormInfer_BFP16, MatchesCpuReference)
{
    CheckServed<bfloat16>(GetParam().layout, GetParam().lengths);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_HipdnnShimBatchNormInfer_BFP16,
                         ::testing::ValuesIn(ServedCases()),
                         CaseName);

// Same shapes, different epsilon. Epsilon is baked into the forwarded graph, so a plan key
// without it would reuse the first plan for the second call. A large epsilon makes the two
// results far apart.
TEST(GPU_HipdnnShimBatchNormPlanKey_FP32, SameShapeDifferentEpsilon)
{
    const std::vector<std::size_t> lengths{2, 6, 5, 7};
    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, lengths, {}, 17);
    Params p{MakeParams(lengths, false)};
    for(const double epsilon : {kEpsilon, 3.0})
    {
        SCOPED_TRACE(epsilon);
        Call call;
        call.epsilon = epsilon;
        tensor<float> y{lengths};
        const auto result = RunInference(call, x, p, y);
        ASSERT_EQ(result.status, miopenStatusSuccess) << result.message << '\n' << result.logged;
        CheckWithinTolerance(Reference(x, p, p.mean, p.variance, epsilon), y, "batch-norm result");
    }
}

// MIOpen serves these; the forwarded path declines them. Each records whether it was served,
// and checks the result when it was.
TEST(GPU_HipdnnShimBatchNormGaps_FP32, MatchesCpuReferenceWhenServed)
{
    const std::vector<std::size_t> lengths{2, 4, 3, 5};
    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, lengths, {}, 17);

    {
        SCOPED_TRACE("per-activation");
        Params p{MakeParams(lengths, true)};
        Call call;
        call.mode = miopenBNPerActivation;
        tensor<float> y{lengths};
        const auto result = RunInference(call, x, p, y);
        RecordServed("per-activation", result.status == miopenStatusSuccess);
        if(result.status == miopenStatusSuccess)
            CheckWithinTolerance(
                Reference(x, p, p.mean, p.variance, kEpsilon), y, "per-activation result");
        else
            ExpectForwardedDecline(result, "per-activation batch norm");
    }

    {
        SCOPED_TRACE("batch-statistics");
        Params p{MakeParams(lengths, false)};
        Call call;
        call.with_mean     = false;
        call.with_variance = false;
        tensor<float> y{lengths};
        const auto result = RunInference(call, x, p, y);
        RecordServed("batch-statistics", result.status == miopenStatusSuccess);
        if(result.status == miopenStatusSuccess)
        {
            tensor<float> mean{p.mean.desc};
            tensor<float> variance{p.variance.desc};
            BatchStatistics(x, mean, variance);
            CheckWithinTolerance(
                Reference(x, p, mean, variance, kEpsilon), y, "batch-statistics result");
        }
        else
        {
            ExpectForwardedDecline(result, "without a running mean and variance");
        }
    }
}

// Both modes return BadParm for these, as MIOpen checks them before choosing a kernel. In the
// forwarded mode the reason shows which check fired.
TEST(GPU_HipdnnShimBatchNormInvalid_FP32, StatusMatchesMiopen)
{
    const std::vector<std::size_t> lengths{2, 4, 3, 5};
    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, lengths, {}, 17);
    Params p{MakeParams(lengths, false)};

    const float two = 2.0f;
    struct ScaleCase
    {
        const char* name;
        const float* alpha;
        const float* beta;
    };
    for(const ScaleCase& scale :
        {ScaleCase{"alpha", &two, &kZero}, ScaleCase{"beta", &kOne, &kOne}})
    {
        SCOPED_TRACE(scale.name);
        Call call;
        call.alpha = scale.alpha;
        call.beta  = scale.beta;
        tensor<float> y{lengths};
        const auto result = RunInference(call, x, p, y);
        RecordStatus(scale.name, result.status);
        EXPECT_EQ(result.status, miopenStatusBadParm);
        ExpectForwardedDecline(result, "supports only alpha=1, beta=0");
    }

    {
        SCOPED_TRACE("rank");
        // A 3-D parameter tensor for 4-D x.
        Params p3{tensor<float>{std::vector<std::size_t>{1, 4, 1}}};
        tensor<float> y{lengths};
        const auto result = RunInference(Call{}, x, p3, y);
        RecordStatus("rank", result.status);
        EXPECT_EQ(result.status, miopenStatusBadParm);
        ExpectForwardedDecline(result, "must have the same rank");
    }

    {
        SCOPED_TRACE("y-type");
        tensor<half_float::half> y{lengths};
        const auto result = RunInference(Call{}, x, p, y);
        RecordStatus("y-type", result.status);
        EXPECT_EQ(result.status, miopenStatusBadParm);
        ExpectForwardedDecline(result, "x and y must have the same data type");
    }
}

// One call per check the forwarded path makes before building a graph, forwarding only:
// natively a null alpha crashes and a vectorized layout is not a batch-norm problem. The reason
// is checked, not just the status, because a hipDNN failure for lack of an engine would hide a
// missing check. The gaps and invalid-parameter tests above run the rest in both modes.
TEST(GPU_HipdnnShimBatchNormDeclined_FP32, UnsupportedProblemsAreDeclinedWithReason)
{
    if(!ForwardingEnabled())
        return;

    const std::vector<std::size_t> lengths{2, 4, 3, 5};
    tensor<float> x = MakeFilled<float>(miopenTensorNCHW, lengths, {}, 17);
    Params p{MakeParams(lengths, false)};

    const auto expect_declined_because =
        [](const CallResult& result, const char* reason, miopenStatus_t expected_status) {
            EXPECT_EQ(result.status, expected_status) << reason;
            ExpectForwardedDecline(result, reason);
        };

    {
        Call call;
        call.alpha = nullptr;
        tensor<float> y{lengths};
        expect_declined_because(
            RunInference(call, x, p, y), "supports only alpha=1, beta=0", miopenStatusBadParm);
    }
    {
        Call call;
        call.beta = nullptr;
        tensor<float> y{lengths};
        expect_declined_because(
            RunInference(call, x, p, y), "supports only alpha=1, beta=0", miopenStatusBadParm);
    }
    {
        Call call;
        call.with_variance = false;
        tensor<float> y{lengths};
        expect_declined_because(RunInference(call, x, p, y),
                                "without a running mean and variance",
                                miopenStatusUnsupportedOp);
    }
    {
        tensor<float> vectorized_x{miopenTensorNCHWc4, lengths};
        tensor<float> vectorized_y{miopenTensorNCHWc4, lengths};
        expect_declined_because(RunInference(Call{}, vectorized_x, p, vectorized_y),
                                "vectorized tensor layouts",
                                miopenStatusUnsupportedOp);
    }
}
