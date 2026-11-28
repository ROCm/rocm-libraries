// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <half/half.hpp>
#include <miopen/config.h>
#include <miopen/miopen.h>
#include <miopen/bfloat16.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

#define RNN_OK(call) ASSERT_EQ(miopenStatusSuccess, (call)) << #call
#define HIP_OK(call) ASSERT_EQ(hipSuccess, (call)) << #call

template <class T, auto Destroy>
struct Resource
{
    T value = nullptr;
    ~Resource()
    {
        if(value != nullptr)
            (void)Destroy(value);
    }
    operator T() const { return value; }
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
};

using Handle = Resource<miopenHandle_t, miopenDestroy>;
using Rnn = Resource<miopenRNNDescriptor_t, miopenDestroyRNNDescriptor>;
using Dropout = Resource<miopenDropoutDescriptor_t, miopenDestroyDropoutDescriptor>;
using Tensor = Resource<miopenTensorDescriptor_t, miopenDestroyTensorDescriptor>;
using Sequence = Resource<miopenSeqTensorDescriptor_t, miopenDestroySeqTensorDescriptor>;
using Buffer = Resource<void*, hipFree>;

constexpr int Steps = 3;
constexpr int Variables = Steps + 2;
using Inputs = std::vector<double>;
struct Result
{
    std::array<double, Steps> y;
    double h;
    double c;
};

// Logical order: x, hx, cx, input weights, recurrent weights, input biases,
// recurrent biases. Each group uses the public gate order, not a kernel layout.
Result Reference(miopenRNNMode_t mode, int gates, const Inputs& p)
{
    Result out{{}, p[Steps], p[Steps + 1]};
    const auto sigmoid = [](double x) { return 1 / (1 + std::exp(-x)); };
    for(int t = 0; t < Steps; ++t)
    {
        const auto input = [&](int g) { return p[t] * p[Variables + g] + p[Variables + 2 * gates + g]; };
        const auto recurrent = [&](int g) {
            return out.h * p[Variables + gates + g] + p[Variables + 3 * gates + g];
        };
        const auto gate = [&](int g) { return input(g) + recurrent(g); };
        if(mode == miopenLSTM)
        {
            out.c = sigmoid(gate(1)) * out.c + sigmoid(gate(0)) * std::tanh(gate(3));
            out.h = sigmoid(gate(2)) * std::tanh(out.c);
        }
        else if(mode == miopenGRU)
        {
            const auto z = sigmoid(gate(0));
            const auto n = std::tanh(input(2) + sigmoid(gate(1)) * recurrent(2));
            out.h = z * out.h + (1 - z) * n;
        }
        else if(mode == miopenRNNRELU)
            out.h = std::max(0.0, gate(0));
        else
            out.h = std::tanh(gate(0));
        out.y[t] = out.h;
    }
    return out;
}

constexpr std::array<float, Steps> Dy{0.3125f, -0.1875f, 0.4375f};
constexpr float Dhy = -0.125f;
constexpr float Dcy = 0.25f;

Inputs ReferenceGradients(miopenRNNMode_t mode, int gates, Inputs p, double scale)
{
    const auto loss = [&] {
        const auto out = Reference(mode, gates, p);
        double value = out.h * Dhy + (mode == miopenLSTM ? out.c * Dcy : 0);
        for(int t = 0; t < Steps; ++t)
            value += out.y[t] * Dy[t];
        return value;
    };
    // Differentiate the independent smooth reference at the quantized inputs,
    // not the discontinuous storage-rounding steps in the implementation.
    Inputs grad(p.size());
    for(std::size_t i = 0; i < p.size(); ++i)
    {
        // Scale the perturbation to each variable's range so it survives
        // double precision for x without saturating gates for input weights.
        const double eps = 1e-5 * (i < Steps ? scale : 1) /
                           (i >= Variables && i < static_cast<std::size_t>(Variables + gates)
                                ? scale
                                : 1);
        const auto original = p[i];
        p[i] = original + eps;
        const auto upper = loss();
        p[i] = original - eps;
        grad[i] = (upper - loss()) / (2 * eps);
        p[i] = original;
    }
    return grad;
}

template <class T>
void Upload(Buffer& buffer, const T* data, std::size_t count)
{
    HIP_OK(hipMalloc(&buffer.value, count * sizeof(T)));
    HIP_OK(hipMemcpy(buffer, data, count * sizeof(T), hipMemcpyHostToDevice));
}

void Scratch(Buffer& buffer, std::size_t bytes)
{
    if(bytes != 0)
        HIP_OK(hipMalloc(&buffer.value, bytes));
}

template <class T>
void Check(Buffer& buffer, const Inputs& expected, double tolerance, const char* name)
{
    SCOPED_TRACE(name);
    std::vector<T> output(expected.size());
    HIP_OK(hipMemcpy(output.data(), buffer, output.size() * sizeof(T), hipMemcpyDeviceToHost));
    for(std::size_t i = 0; i < output.size(); ++i)
        EXPECT_NEAR(static_cast<float>(output[i]), expected[i], tolerance * (0.25 + std::abs(expected[i]))) << "element " << i;
}

template <class T>
void Run(miopenHandle_t handle, miopenRNNMode_t mode, bool sequence, miopenDataType_t type)
{
    SCOPED_TRACE(::testing::Message() << "mode=" << mode << " sequence=" << sequence << " type=" << type);
    const bool lstm = mode == miopenLSTM;
    const int gates = lstm ? 4 : mode == miopenGRU ? 3 : 1;
    const double tolerance = type == miopenBFloat16 ? 0.035 : type == miopenHalf ? 0.006 : 0.0001;
    Dropout dropout;
    Rnn rnn;
    Tensor xy, state, weights, gateDesc;
    Sequence seq;
    RNN_OK(miopenCreateRNNDescriptor(&rnn.value));
    RNN_OK(miopenCreateTensorDescriptor(&xy.value));
    RNN_OK(miopenCreateTensorDescriptor(&state.value));
    RNN_OK(miopenCreateTensorDescriptor(&weights.value));
    RNN_OK(miopenCreateTensorDescriptor(&gateDesc.value));
    const int ones[] = {1, 1, 1};
    RNN_OK(miopenSetTensorDescriptor(xy, type, 2, ones, ones));
    RNN_OK(miopenSetTensorDescriptor(state, type, 3, ones, ones));
    const std::array<miopenTensorDescriptor_t, Steps> desc{xy, xy, xy};
    if(sequence)
    {
        RNN_OK(miopenCreateDropoutDescriptor(&dropout.value));
        RNN_OK(miopenSetRNNDescriptor_V2(rnn, 1, 1, dropout, miopenRNNlinear,
                                        miopenRNNunidirection, mode, miopenRNNwithBias,
                                        miopenRNNdefault, type));
        RNN_OK(miopenCreateSeqTensorDescriptor(&seq.value));
        const int lengths[] = {Steps};
        RNN_OK(miopenSetRNNDataSeqTensorDescriptor(seq, type, miopenRNNDataSeqMajorNotPadded,
                                                 Steps, 1, 1, lengths, nullptr));
    }
    else
        RNN_OK(miopenSetRNNDescriptor(rnn, 1, 1, miopenRNNlinear, miopenRNNunidirection,
                                     mode, miopenRNNwithBias, miopenRNNdefault, type));

    std::size_t weightBytes = 0;
    RNN_OK(miopenGetRNNParamsSize(handle, rnn, xy, &weightBytes, type));
    ASSERT_EQ(weightBytes, 4 * gates * sizeof(T));
    RNN_OK(miopenGetRNNParamsDescriptor(handle, rnn, xy, weights, type));
    std::vector<std::size_t> offsets(4 * gates);
    for(int id = 0; id < 2 * gates; ++id)
    {
        std::size_t paramBytes = 0, biasBytes = 0;
        RNN_OK(miopenGetRNNLayerParamSize(handle, rnn, 0, xy, id, &paramBytes));
        RNN_OK(miopenGetRNNLayerBiasSize(handle, rnn, 0, id, &biasBytes));
        ASSERT_EQ(paramBytes, sizeof(T));
        ASSERT_EQ(biasBytes, sizeof(T));
        RNN_OK(miopenGetRNNLayerParamOffset(rnn, 0, xy, id, gateDesc, &offsets[id]));
        RNN_OK(miopenGetRNNLayerBiasOffset(rnn, 0, xy, id, gateDesc, &offsets[2 * gates + id]));
    }
    auto sorted = offsets;
    std::sort(sorted.begin(), sorted.end());
    for(std::size_t i = 0; i < sorted.size(); ++i)
        ASSERT_EQ(sorted[i], i);

    // The first product is ordinary magnitude in BF16 but its x operand would
    // overflow in FP16. Its other operand would underflow in FP16.
    const float scale = type == miopenHalf ? 1.0f : 16777216.0f;
    Inputs p{0.3125f * scale, -0.375f * scale, 0.625f * scale, 0.1875f, -0.3125f};
    if(mode == miopenRNNRELU)
        p[1] = -4.0f * scale; // Exercise both sides of the ReLU derivative.
    for(int i = 0; i < 4 * gates; ++i)
    {
        const float v = 0.09375f + 0.02734375f * (i % 7);
        p.push_back(i < gates ? v / scale : v);
    }
    for(auto& value : p)
        value = static_cast<float>(T(static_cast<float>(value)));
    const auto ref = Reference(mode, gates, p);
    std::array<T, Steps> xHost{}, dyHost{};
    for(int i = 0; i < Steps; ++i)
    {
        xHost[i] = T(static_cast<float>(p[i]));
        dyHost[i] = T(Dy[i]);
    }
    const T hxHost(static_cast<float>(p[Steps])), cxHost(static_cast<float>(p[Steps + 1]));
    const T dhyHost(Dhy), dcyHost(Dcy);
    const T poison(std::numeric_limits<float>::quiet_NaN());
    std::array<T, Steps> poisonSeq{poison, poison, poison};
    std::vector<T> packed(offsets.size()), poisonWeights(offsets.size(), poison);
    for(std::size_t i = 0; i < offsets.size(); ++i)
        packed[offsets[i]] = T(static_cast<float>(p[Variables + i]));
    Buffer x, hx, cx, w, y, hy, cy, dx, dhx, dcx, dw, dy, dhy, dcy;
    ASSERT_NO_FATAL_FAILURE(Upload(x, xHost.data(), Steps));
    ASSERT_NO_FATAL_FAILURE(Upload(hx, &hxHost, 1));
    ASSERT_NO_FATAL_FAILURE(Upload(w, packed.data(), packed.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(y, poisonSeq.data(), Steps));
    ASSERT_NO_FATAL_FAILURE(Upload(hy, &poison, 1));
    ASSERT_NO_FATAL_FAILURE(Upload(dx, poisonSeq.data(), Steps));
    ASSERT_NO_FATAL_FAILURE(Upload(dhx, &poison, 1));
    ASSERT_NO_FATAL_FAILURE(Upload(dw, poisonWeights.data(), poisonWeights.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(dy, dyHost.data(), Steps));
    ASSERT_NO_FATAL_FAILURE(Upload(dhy, &dhyHost, 1));
    if(lstm)
    {
        ASSERT_NO_FATAL_FAILURE(Upload(cx, &cxHost, 1));
        ASSERT_NO_FATAL_FAILURE(Upload(cy, &poison, 1));
        ASSERT_NO_FATAL_FAILURE(Upload(dcx, &poison, 1));
        ASSERT_NO_FATAL_FAILURE(Upload(dcy, &dcyHost, 1));
    }
    miopenAcceleratorQueue_t stream = nullptr;
    RNN_OK(miopenGetStream(handle, &stream));
    const auto checkForward = [&] {
        HIP_OK(hipStreamSynchronize(stream));
        Check<T>(y, Inputs(ref.y.begin(), ref.y.end()), tolerance, "y");
        Check<T>(hy, {ref.h}, tolerance, "hy");
        if(lstm)
            Check<T>(cy, {ref.c}, tolerance, "cy");
    };
    std::size_t trainBytes = 0, inferBytes = 0, reserveBytes = 0;
    if(sequence)
    {
        RNN_OK(miopenGetRNNTempSpaceSizes(
            handle, rnn, seq, miopenRNNInference, &inferBytes, nullptr));
        RNN_OK(miopenGetRNNTempSpaceSizes(handle, rnn, seq, miopenRNNTraining, &trainBytes, &reserveBytes));
    }
    else
    {
        RNN_OK(miopenGetRNNWorkspaceSize(handle, rnn, Steps, desc.data(), &trainBytes));
        RNN_OK(miopenGetRNNTrainingReserveSize(handle, rnn, Steps, desc.data(), &reserveBytes));
        inferBytes = trainBytes;
    }
    EXPECT_EQ(trainBytes % sizeof(T), 0);
    EXPECT_EQ(inferBytes % sizeof(T), 0);
    Buffer infer, workspace, reserve;
    ASSERT_NO_FATAL_FAILURE(Scratch(infer, inferBytes));
    ASSERT_NO_FATAL_FAILURE(Scratch(workspace, trainBytes));
    ASSERT_NO_FATAL_FAILURE(Scratch(reserve, reserveBytes));
    if(type == miopenBFloat16)
    {
        Tensor wrongState;
        RNN_OK(miopenCreateTensorDescriptor(&wrongState.value));
        RNN_OK(miopenSetTensorDescriptor(wrongState, miopenHalf, 3, ones, ones));
        if(sequence)
        {
            EXPECT_EQ(miopenStatusBadParm,
                      miopenRNNForward(handle, rnn, miopenRNNInference, seq, x, wrongState, hx,
                                       hy, state, cx, cy, seq, y, w, weightBytes, infer,
                                       inferBytes, nullptr, 0));
            EXPECT_NE(miopenStatusSuccess,
                      miopenRNNForward(handle, rnn, miopenRNNInference, seq, x, state, hx,
                                       hy, state, cx, cy, seq, y, w, weightBytes - 1, infer,
                                       inferBytes, nullptr, 0));
        }
        else
            EXPECT_EQ(miopenStatusBadParm,
                      miopenRNNForwardInference(handle, rnn, Steps, desc.data(), x, wrongState,
                                                hx, state, cx, weights, w, desc.data(), y, state,
                                                hy, state, cy, infer, inferBytes));
    }
    if(sequence)
        RNN_OK(miopenRNNForward(handle, rnn, miopenRNNInference, seq, x, state, hx, hy,
                               state, cx, cy, seq, y, w, weightBytes, infer, inferBytes, nullptr, 0));
    else
        RNN_OK(miopenRNNForwardInference(handle, rnn, Steps, desc.data(), x, state, hx,
                                        state, cx, weights, w, desc.data(), y, state, hy,
                                        state, cy, infer, inferBytes));
    ASSERT_NO_FATAL_FAILURE(checkForward());
    HIP_OK(hipMemcpy(y, poisonSeq.data(), sizeof(poisonSeq), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(hy, &poison, sizeof(T), hipMemcpyHostToDevice));
    if(lstm)
        HIP_OK(hipMemcpy(cy, &poison, sizeof(T), hipMemcpyHostToDevice));
    const auto forwardTraining = [&](std::size_t workSize, std::size_t reserveSize) {
        if(sequence)
            return miopenRNNForward(handle, rnn, miopenRNNTraining, seq, x, state, hx, hy,
                                   state, cx, cy, seq, y, w, weightBytes, workspace, workSize,
                                   reserve, reserveSize);
        return miopenRNNForwardTraining(handle, rnn, Steps, desc.data(), x, state, hx, state,
                                        cx, weights, w, desc.data(), y, state, hy, state, cy,
                                        workspace, workSize, reserve, reserveSize);
    };
    ASSERT_GT(trainBytes, 0);
    ASSERT_GT(reserveBytes, 0);
    if(type == miopenBFloat16)
    {
        EXPECT_NE(miopenStatusSuccess, forwardTraining(trainBytes - 1, reserveBytes));
        EXPECT_NE(miopenStatusSuccess, forwardTraining(trainBytes, reserveBytes - 1));
    }
    RNN_OK(forwardTraining(trainBytes, reserveBytes));
    ASSERT_NO_FATAL_FAILURE(checkForward());
    if(sequence)
    {
        RNN_OK(miopenRNNBackwardSeqData(handle, rnn, seq, y, dy, state, hx, dhy, dhx,
                                       state, cx, dcy, dcx, seq, dx, w, weightBytes, workspace,
                                       trainBytes, reserve, reserveBytes));
        RNN_OK(miopenRNNBackwardWeightsSeqTensor(handle, rnn, seq, x, state, hx, seq, y,
                                                dw, weightBytes, workspace, trainBytes, reserve,
                                                reserveBytes));
    }
    else
    {
        RNN_OK(miopenRNNBackwardData(handle, rnn, Steps, desc.data(), y, desc.data(), dy,
                                   state, dhy, state, dcy, weights, w, state, hx, state, cx,
                                   desc.data(), dx, state, dhx, state, dcx, workspace,
                                   trainBytes, reserve, reserveBytes));
        RNN_OK(miopenRNNBackwardWeights(handle, rnn, Steps, desc.data(), x, state, hx,
                                      desc.data(), y, weights, dw, workspace, trainBytes,
                                      reserve, reserveBytes));
    }
    HIP_OK(hipStreamSynchronize(stream));
    const auto grad = ReferenceGradients(mode, gates, p, scale);
    Check<T>(dx, Inputs(grad.begin(), grad.begin() + Steps), tolerance / scale, "dx");
    Check<T>(dhx, {grad[Steps]}, tolerance, "dhx");
    if(lstm)
        Check<T>(dcx, {grad[Steps + 1]}, tolerance, "dcx");
    std::vector<T> dwHost(offsets.size());
    HIP_OK(hipMemcpy(dwHost.data(), dw, weightBytes, hipMemcpyDeviceToHost));
    for(std::size_t i = 0; i < offsets.size(); ++i)
    {
        const double factor = i < static_cast<std::size_t>(gates) ? scale : 1;
        const auto expected = grad[Variables + i] / factor;
        EXPECT_NEAR(static_cast<float>(dwHost[offsets[i]]) / factor, expected,
                    tolerance * (0.25 + std::abs(expected))) << "weight/bias ID " << i;
    }
}

using GPU_RNNPublicApi_BFP16 = ::testing::TestWithParam<std::tuple<miopenRNNMode_t, bool>>;

TEST_P(GPU_RNNPublicApi_BFP16, AllPasses)
{
#if !MIOPEN_USE_ROCBLAS
    GTEST_SKIP() << "BF16 RNN requires rocBLAS";
#endif
    const auto [mode, sequence] = GetParam();
    Handle handle;
    RNN_OK(miopenCreate(&handle.value));
    ASSERT_NO_FATAL_FAILURE(Run<bfloat16>(handle, mode, sequence, miopenBFloat16));
    // Interleave dtypes on the same handle with the same shapes and pass order.
    // LSTM uses dedicated typed gate kernels in addition to the common operations.
    if(mode == miopenLSTM)
    {
        ASSERT_NO_FATAL_FAILURE(Run<half_float::half>(handle, mode, sequence, miopenHalf));
        ASSERT_NO_FATAL_FAILURE(Run<float>(handle, mode, sequence, miopenFloat));
        ASSERT_NO_FATAL_FAILURE(Run<bfloat16>(handle, mode, sequence, miopenBFloat16));
    }
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         GPU_RNNPublicApi_BFP16,
                         ::testing::Combine(::testing::Values(miopenRNNRELU, miopenRNNTANH, miopenLSTM, miopenGRU),
                                            ::testing::Bool()));

using GPU_RNNLegacyNullHx_BFP16 = ::testing::TestWithParam<std::vector<int>>;

TEST_P(GPU_RNNLegacyNullHx_BFP16, BidirectionalRaggedBiasGradients)
{
#if !MIOPEN_USE_ROCBLAS
    GTEST_SKIP() << "BF16 RNN requires rocBLAS";
#endif
    constexpr auto type = miopenBFloat16;
    constexpr int directions = 2;
    const auto& batches = GetParam();
    const int steps     = static_cast<int>(batches.size());
    const int rows      = std::accumulate(batches.begin(), batches.end(), 0);
    Handle handle;
    Rnn rnn;
    Tensor state, weights, bias;
    std::vector<Tensor> xTensors(steps), yTensors(steps);
    std::vector<miopenTensorDescriptor_t> xDesc(steps), yDesc(steps);
    RNN_OK(miopenCreate(&handle.value));
    RNN_OK(miopenCreateRNNDescriptor(&rnn.value));
    RNN_OK(miopenSetRNNDescriptor(rnn, 1, 1, miopenRNNlinear, miopenRNNbidirection,
                                 miopenRNNRELU, miopenRNNwithBias, miopenRNNdefault, type));
    for(int t = 0; t < steps; ++t)
    {
        RNN_OK(miopenCreateTensorDescriptor(&xTensors[t].value));
        RNN_OK(miopenCreateTensorDescriptor(&yTensors[t].value));
        xDesc[t] = xTensors[t];
        yDesc[t] = yTensors[t];
        const int xLengths[] = {batches[t], 1}, xStrides[] = {1, 1};
        const int yLengths[] = {batches[t], directions}, yStrides[] = {directions, 1};
        RNN_OK(miopenSetTensorDescriptor(xDesc[t], type, 2, xLengths, xStrides));
        RNN_OK(miopenSetTensorDescriptor(yDesc[t], type, 2, yLengths, yStrides));
    }
    RNN_OK(miopenCreateTensorDescriptor(&state.value));
    RNN_OK(miopenCreateTensorDescriptor(&weights.value));
    RNN_OK(miopenCreateTensorDescriptor(&bias.value));
    const int stateLengths[] = {directions, batches[0], 1};
    const int stateStrides[] = {batches[0], 1, 1};
    RNN_OK(miopenSetTensorDescriptor(state, type, 3, stateLengths, stateStrides));
    RNN_OK(miopenGetRNNParamsDescriptor(handle, rnn, xDesc[0], weights, type));
    std::size_t weightBytes = 0, workBytes = 0, reserveBytes = 0;
    RNN_OK(miopenGetRNNParamsSize(handle, rnn, xDesc[0], &weightBytes, type));
    RNN_OK(miopenGetRNNWorkspaceSize(handle, rnn, steps, xDesc.data(), &workBytes));
    RNN_OK(miopenGetRNNTrainingReserveSize(handle, rnn, steps, xDesc.data(), &reserveBytes));
    ASSERT_EQ(weightBytes, 8 * sizeof(bfloat16));
    std::vector<bfloat16> packed(weightBytes / sizeof(bfloat16), bfloat16(0.0f));
    constexpr float inputBias[] = {0.25f, 0.5f}, recurrentBias[] = {0.5f, 0.75f};
    std::size_t offsets[directions][2]{};
    for(int d = 0; d < directions; ++d)
    {
        for(int id = 0; id < 2; ++id)
        {
            RNN_OK(miopenGetRNNLayerBiasOffset(rnn, d, xDesc[0], id, bias, &offsets[d][id]));
            ASSERT_LT(offsets[d][id], packed.size());
            packed[offsets[d][id]] = bfloat16(id == 0 ? inputBias[d] : recurrentBias[d]);
        }
    }

    const bfloat16 poison(std::numeric_limits<float>::quiet_NaN());
    const std::vector<bfloat16> xHost(rows, bfloat16(0.0f)), yPoison(rows * directions, poison);
    const std::vector<bfloat16> dwPoison(packed.size(), poison);
    std::vector<bfloat16> dyHost(rows * directions);
    Inputs expectedY(rows * directions);
    double expectedBias[directions][2]{};
    int row = 0;
    for(int t = 0; t < steps; ++t)
    {
        for(int b = 0; b < batches[t]; ++b, ++row)
        {
            // Reverse recurrence starts at each sample's own last row, not
            // necessarily the last timestep of the packed batch.
            const bool recurrent[] = {t != 0, t + 1 < steps && b < batches[t + 1]};
            const float grad[] = {1.0f, t == 0 ? 2.0f : 1.0f};
            for(int d = 0; d < directions; ++d)
            {
                const int i = row * directions + d;
                expectedY[i] = inputBias[d] + (recurrent[d] ? recurrentBias[d] : 0);
                dyHost[i] = bfloat16(grad[d]);
                expectedBias[d][0] += grad[d];
                if(recurrent[d])
                    expectedBias[d][1] += grad[d];
            }
        }
    }
    Buffer x, y, w, dy, dx, dw, workspace, reserve;
    ASSERT_NO_FATAL_FAILURE(Upload(x, xHost.data(), xHost.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(y, yPoison.data(), yPoison.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(w, packed.data(), packed.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(dy, dyHost.data(), dyHost.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(dw, dwPoison.data(), dwPoison.size()));
    ASSERT_NO_FATAL_FAILURE(Scratch(dx, xHost.size() * sizeof(bfloat16)));
    ASSERT_NO_FATAL_FAILURE(Scratch(workspace, workBytes));
    ASSERT_NO_FATAL_FAILURE(Scratch(reserve, reserveBytes));
    RNN_OK(miopenRNNForwardTraining(handle, rnn, steps, xDesc.data(), x, state, nullptr,
                                    state, nullptr, weights, w, yDesc.data(), y, state, nullptr,
                                    state, nullptr, workspace, workBytes, reserve, reserveBytes));
    RNN_OK(miopenRNNBackwardData(handle, rnn, steps, yDesc.data(), y, yDesc.data(), dy,
                                state, nullptr, state, nullptr, weights, w, state, nullptr,
                                state, nullptr, xDesc.data(), dx, state, nullptr, state, nullptr,
                                workspace, workBytes, reserve, reserveBytes));
    RNN_OK(miopenRNNBackwardWeights(handle, rnn, steps, xDesc.data(), x, state, nullptr,
                                   yDesc.data(), y, weights, dw, workspace, workBytes, reserve,
                                   reserveBytes));
    miopenAcceleratorQueue_t stream = nullptr;
    RNN_OK(miopenGetStream(handle, &stream));
    HIP_OK(hipStreamSynchronize(stream));
    ASSERT_NO_FATAL_FAILURE(Check<bfloat16>(y, expectedY, 0, "y"));
    std::vector<bfloat16> dwHost(packed.size());
    HIP_OK(hipMemcpy(dwHost.data(), dw, weightBytes, hipMemcpyDeviceToHost));
    // In particular, 300 forward recurrent contributions of +1 exceed
    // BF16's consecutive integer range. Rounding after each row loses terms.
    for(int d = 0; d < directions; ++d)
        for(int id = 0; id < 2; ++id)
            EXPECT_EQ(static_cast<float>(bfloat16(static_cast<float>(expectedBias[d][id]))),
                      static_cast<float>(dwHost[offsets[d][id]]))
                << "direction " << d << ", bias ID " << id;
}

INSTANTIATE_TEST_SUITE_P(
    Quick,
    GPU_RNNLegacyNullHx_BFP16,
    ::testing::Values(std::vector<int>{300, 200, 100},
                      std::vector<int>{19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                                       9, 8, 7, 6, 5, 4, 3, 2, 1}));

using GPU_RNNDropout_BFP16 = ::testing::TestWithParam<int>;

TEST_P(GPU_RNNDropout_BFP16, MultilayerTraining)
{
#if !MIOPEN_USE_ROCBLAS
    GTEST_SKIP() << "BF16 RNN requires rocBLAS";
#endif
    constexpr auto type = miopenBFloat16;
    constexpr int batch = 32, layers = 2, rows = batch * Steps;
    Handle handle;
    Buffer rng;
    Dropout dropout;
    Rnn rnn;
    Tensor xy, state, weights, param;
    RNN_OK(miopenCreate(&handle.value));
    RNN_OK(miopenCreateDropoutDescriptor(&dropout.value));
    size_t stateBytes = 0;
    RNN_OK(miopenDropoutGetStatesSize(handle, &stateBytes));
    ASSERT_NO_FATAL_FAILURE(Scratch(rng, stateBytes));
    RNN_OK(miopenSetDropoutDescriptor(dropout, handle, 0.5f, rng, stateBytes, 42, false, false,
                                     MIOPEN_RNG_PSEUDO_XORWOW));
    RNN_OK(miopenCreateRNNDescriptor(&rnn.value));
    RNN_OK(miopenSetRNNDescriptor_V2(rnn, 1, layers, dropout, miopenRNNlinear,
                                    miopenRNNunidirection, miopenRNNRELU, miopenRNNNoBias,
                                    miopenRNNdefault, type));
    for(auto* desc : {&xy, &state, &weights, &param})
        RNN_OK(miopenCreateTensorDescriptor(&desc->value));
    const int xyLengths[] = {batch, 1}, xyStrides[] = {1, 1};
    const int stateLengths[] = {layers, batch, 1}, stateStrides[] = {batch, 1, 1};
    RNN_OK(miopenSetTensorDescriptor(xy, type, 2, xyLengths, xyStrides));
    RNN_OK(miopenSetTensorDescriptor(state, type, 3, stateLengths, stateStrides));
    const std::array<miopenTensorDescriptor_t, Steps> descs{xy, xy, xy};
    size_t weightBytes = 0, workBytes = 0, reserveBytes = 0;
    RNN_OK(miopenGetRNNParamsSize(handle, rnn, xy, &weightBytes, type));
    RNN_OK(miopenGetRNNParamsDescriptor(handle, rnn, xy, weights, type));
    RNN_OK(miopenGetRNNWorkspaceSize(handle, rnn, Steps, descs.data(), &workBytes));
    RNN_OK(miopenGetRNNTrainingReserveSize(handle, rnn, Steps, descs.data(), &reserveBytes));
    std::vector<bfloat16> packed(weightBytes / sizeof(bfloat16), bfloat16(0));
    std::array<size_t, layers> inputWeights{};
    for(int layer = 0; layer < layers; ++layer)
    {
        RNN_OK(miopenGetRNNLayerParamOffset(rnn, layer, xy, 0, param, &inputWeights[layer]));
        ASSERT_LT(inputWeights[layer], packed.size());
        packed[inputWeights[layer]] = bfloat16(1);
    }

    // With unit input weights and zero recurrent weights, y is exactly 0 or 2*x.
    // Backward data must replay that same mask, not sample again.
    std::vector<bfloat16> xHost(rows), dyHost(rows, bfloat16(1));
    for(int i = 0; i < rows; ++i)
        xHost[i] = bfloat16(1.0f + 0.25f * (i % 3));
    Buffer x, y, w, dy, dx, dw, work, reserve;
    ASSERT_NO_FATAL_FAILURE(Upload(x, xHost.data(), xHost.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(dy, dyHost.data(), dyHost.size()));
    ASSERT_NO_FATAL_FAILURE(Upload(w, packed.data(), packed.size()));
    ASSERT_NO_FATAL_FAILURE(Scratch(y, rows * sizeof(bfloat16)));
    ASSERT_NO_FATAL_FAILURE(Scratch(dx, rows * sizeof(bfloat16)));
    ASSERT_NO_FATAL_FAILURE(Scratch(dw, weightBytes));
    ASSERT_NO_FATAL_FAILURE(Scratch(work, workBytes));
    ASSERT_NO_FATAL_FAILURE(Scratch(reserve, reserveBytes));
    RNN_OK(miopenRNNForwardTraining(handle, rnn, Steps, descs.data(), x, state, nullptr, state,
                                    nullptr, weights, w, descs.data(), y, state, nullptr, state,
                                    nullptr, work, workBytes, reserve, reserveBytes));
    RNN_OK(miopenRNNBackwardData(handle, rnn, Steps, descs.data(), y, descs.data(), dy, state,
                                nullptr, state, nullptr, weights, w, state, nullptr, state,
                                nullptr, descs.data(), dx, state, nullptr, state, nullptr, work,
                                workBytes, reserve, reserveBytes));
    RNN_OK(miopenRNNBackwardWeights(handle, rnn, Steps, descs.data(), x, state, nullptr,
                                   descs.data(), y, weights, dw, work, workBytes, reserve,
                                   reserveBytes));
    std::vector<bfloat16> yHost(rows), dwHost(packed.size());
    HIP_OK(hipMemcpy(yHost.data(), y, rows * sizeof(bfloat16), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(dwHost.data(), dw, weightBytes, hipMemcpyDeviceToHost));
    Inputs expectedDx(rows);
    int kept = 0;
    float expectedDw = 0;
    for(int i = 0; i < rows; ++i)
    {
        const float output = static_cast<float>(yHost[i]);
        const float scaled = 2 * static_cast<float>(xHost[i]);
        ASSERT_TRUE(output == 0 || output == scaled) << "row " << i;
        if(output != 0)
        {
            ++kept;
            expectedDx[i] = 2;
            expectedDw += scaled;
        }
    }
    EXPECT_GT(kept, 0);
    EXPECT_LT(kept, rows);
    Check<bfloat16>(dx, expectedDx, 0, "dropout dx");
    for(auto offset : inputWeights)
        EXPECT_EQ(static_cast<float>(dwHost[offset]), static_cast<float>(bfloat16(expectedDw)));
}

INSTANTIATE_TEST_SUITE_P(Quick, GPU_RNNDropout_BFP16, ::testing::Values(0));

using CPU_RNNPublicErrors_NONE = ::testing::TestWithParam<bool>;

TEST_P(CPU_RNNPublicErrors_NONE, RejectsUnsupportedAlgorithm)
{
    Rnn rnn;
    RNN_OK(miopenCreateRNNDescriptor(&rnn.value));
    for(const auto algorithm : {miopenRNNfundamental, miopenRNNroundedDynamic})
    {
        if(GetParam())
            EXPECT_EQ(miopenStatusNotImplemented,
                      miopenSetRNNDescriptor_V2(rnn, 1, 1, nullptr, miopenRNNlinear,
                                               miopenRNNunidirection, miopenLSTM, miopenRNNNoBias,
                                               algorithm, miopenBFloat16));
        else
            EXPECT_EQ(miopenStatusNotImplemented,
                      miopenSetRNNDescriptor(rnn, 1, 1, miopenRNNlinear, miopenRNNunidirection,
                                            miopenLSTM, miopenRNNNoBias, algorithm, miopenBFloat16));
    }
}

INSTANTIATE_TEST_SUITE_P(Quick, CPU_RNNPublicErrors_NONE, ::testing::Bool());

#undef HIP_OK
#undef RNN_OK

} // namespace
