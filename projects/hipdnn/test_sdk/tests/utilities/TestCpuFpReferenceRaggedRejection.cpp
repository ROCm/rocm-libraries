// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The convolution and normalization references address memory through hoisted base
// pointers and strides, which describes the dense layout only: RaggedTensorBase
// overrides getIndexImpl to rebase each batch at ragged_offset[b], and flat stride
// arithmetic bypasses that override entirely.
//
// The CPU graph executor already rejects ragged graphs one level up (see
// cpu_graph_executor/TestCpuReferenceRaggedRejection.cpp). These tests cover the direct
// callers the executor does not sit in front of - the samples and the gpu-ref fixtures -
// which would otherwise get silently wrong numbers rather than an error.

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRMSNorm.hpp>

#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;

namespace
{

// Canonical BSHD-packed ragged geometry, mirroring the data SDK's ragged tests:
// dims [B, S_max, H, D] with batch 0 holding 2 sequence rows and batch 1 holding 3.
// Read as NCHW this is a legal convolution/normalization input, so the references get
// past their shape checks and reach the ragged guard under test.
const std::vector<int64_t> RAGGED_DIMS = {2, 3, 2, 2};
const std::vector<int64_t> RAGGED_STRIDES = {12, 4, 2, 1};
const std::vector<int64_t> RAGGED_OFFSETS = {0, 8, 20};
constexpr int RAGGED_SEQ_AXIS = 1;

std::shared_ptr<ITensor> makeOffsetAux()
{
    auto aux = std::make_shared<Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(RAGGED_OFFSETS.size()), 1, 1, 1});
    for(size_t i = 0; i < RAGGED_OFFSETS.size(); ++i)
    {
        aux->setHostValue(
            static_cast<int32_t>(RAGGED_OFFSETS[i]), static_cast<int64_t>(i), 0, 0, 0);
    }
    return aux;
}

RaggedTensor<float> makeRaggedTensor(const std::shared_ptr<ITensor>& aux)
{
    return RaggedTensor<float>(RAGGED_DIMS, RAGGED_STRIDES, RAGGED_SEQ_AXIS, aux);
}

// Asserts the call was rejected *for raggedness* rather than by some unrelated shape
// check that happens to throw from the same function.
template <typename Callable>
void expectRaggedRejection(Callable&& call, const std::string& tensorName)
{
    try
    {
        call();
        ADD_FAILURE() << "expected the ragged " << tensorName << " tensor to be rejected";
    }
    catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("ragged " + tensorName + " tensor"), std::string::npos)
            << "rejected, but not as a ragged " << tensorName << ": " << message;
    }
}

} // namespace

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionFpropRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> w({1, 3, 1, 1});
    Tensor<float> y({2, 1, 2, 2});

    expectRaggedRejection(
        [&] {
            CpuFpReferenceConvolution::fprop<float, float, float, float>(
                x, w, y, {1, 1}, {1, 1}, {0, 0});
        },
        "x");
}

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionDgradRejectsRaggedGradient)
{
    auto aux = makeOffsetAux();
    auto gradX = makeRaggedTensor(aux);
    Tensor<float> w({1, 3, 1, 1});
    Tensor<float> gradY({2, 1, 2, 2});

    expectRaggedRejection(
        [&] {
            CpuFpReferenceConvolution::dgrad<float, float, float, float>(
                gradX, w, gradY, {1, 1}, {1, 1}, {0, 0});
        },
        "x");
}

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionWgradRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> gradW({1, 3, 1, 1});
    Tensor<float> gradY({2, 1, 2, 2});

    expectRaggedRejection(
        [&] {
            CpuFpReferenceConvolution::wgrad<float, float, float, float>(
                x, gradW, gradY, {1, 1}, {1, 1}, {0, 0});
        },
        "x");
}

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionFpropRejectsRaggedOutput)
{
    // The output is written through the same hoisted pointer, so it needs the guard too.
    auto aux = makeOffsetAux();
    Tensor<float> x({2, 3, 2, 2});
    Tensor<float> w({3, 3, 1, 1});
    auto y = makeRaggedTensor(aux);

    expectRaggedRejection(
        [&] {
            CpuFpReferenceConvolution::fprop<float, float, float, float>(
                x, w, y, {1, 1}, {1, 1}, {0, 0});
        },
        "y");
}

TEST(TestCpuFpReferenceRaggedRejection, LayernormFpropRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> y(RAGGED_DIMS);

    expectRaggedRejection(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, nullptr, nullptr, y, 1e-5, 1);
        },
        "x");
}

TEST(TestCpuFpReferenceRaggedRejection, LayernormBpropRejectsRaggedGradient)
{
    auto aux = makeOffsetAux();
    auto dx = makeRaggedTensor(aux);
    Tensor<float> dy(RAGGED_DIMS);
    Tensor<float> x(RAGGED_DIMS);
    Tensor<float> scale({1, 1, 1, 2});
    Tensor<float> dscale({1, 1, 1, 2});
    Tensor<float> dbias({1, 1, 1, 2});

    expectRaggedRejection(
        [&] {
            CpuFpReferenceLayernorm::bprop<float, float, float, float, float>(
                dy, x, scale, dx, dscale, dbias, 1e-5, nullptr, nullptr, 1);
        },
        "dx");
}

TEST(TestCpuFpReferenceRaggedRejection, RmsNormForwardRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> scale({1, 1, 2, 2});
    Tensor<float> y(RAGGED_DIMS);

    expectRaggedRejection(
        [&] { CpuFpReferenceRMSNorm::forward<float, float, float, float>(x, scale, y, 1e-5); },
        "x");
}

TEST(TestCpuFpReferenceRaggedRejection, RmsNormBackwardRejectsRaggedGradient)
{
    auto aux = makeOffsetAux();
    auto dx = makeRaggedTensor(aux);
    Tensor<float> dy(RAGGED_DIMS);
    Tensor<float> x(RAGGED_DIMS);
    Tensor<float> scale({1, 1, 2, 2});
    Tensor<float> invRms({2, 3, 1, 1});
    Tensor<float> dscale({1, 1, 2, 2});

    expectRaggedRejection(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale);
        },
        "dx");
}

// ============================================================================
// Rank validation
// ============================================================================
//
// Hoisting raw base pointers past TensorBase::getIndex also skipped its argument-count
// guard, which used to reject rank-mismatched inputs on every access. These cover the
// reachable instances: without the guards the references read or write out of bounds,
// and in the RMSNorm backward case return success with plausible but wrong numbers.

namespace
{

template <typename Callable>
void expectRejectedWith(Callable&& call, const std::string& needle)
{
    try
    {
        call();
        ADD_FAILURE() << "expected a rejection mentioning \"" << needle << "\"";
    }
    catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find(needle), std::string::npos)
            << "rejected, but not for the expected reason: " << message;
    }
}

} // namespace

TEST(TestCpuFpReferenceRankValidation, LayernormFpropRejectsScaleRankBelowNormalizedDimCount)
{
    // normSuffixStart = scale.rank() - normalizedDimCount underflows size_t, and the walk
    // strides are then written far out of bounds.
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({4});
    Tensor<float> bias({4});
    Tensor<float> y({2, 3, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, &scale, &bias, y, 1e-5, 2);
        },
        "at least normalizedDimCount");
}

TEST(TestCpuFpReferenceRankValidation, LayernormBpropRejectsScaleRankBelowNormalizedDimCount)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> scale({4});
    Tensor<float> dscale({4});
    Tensor<float> dbias({4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::bprop<float, float, float, float, float>(
                dy, x, scale, dx, dscale, dbias, 1e-5, nullptr, nullptr, 2);
        },
        "at least normalizedDimCount");
}

TEST(TestCpuFpReferenceRankValidation, LayernormFpropRejectsRankZeroStats)
{
    // Every dimension is normalized, so the scalar-batch padding kicks in; a rank-0 mean
    // is then walked with one index against an empty stride vector.
    Tensor<float> x({4});
    Tensor<float> scale({4});
    Tensor<float> bias({4});
    Tensor<float> y({4});
    Tensor<float> mean({});
    Tensor<float> rstd({});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, &scale, &bias, y, 1e-5, 1, &mean, &rstd);
        },
        "at least one dimension");
}

TEST(TestCpuFpReferenceRankValidation, LayernormFpropRejectsMismatchedOutputRank)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({4});
    Tensor<float> bias({4});
    Tensor<float> y({2, 3, 4, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, &scale, &bias, y, 1e-5, 1);
        },
        "y rank to equal input rank");
}

TEST(TestCpuFpReferenceRankValidation, RmsNormBackwardRejectsShortWeightGradientRank)
{
    // The silent one: pre-guard this returned exit 0 with plausible dscale/dbias values,
    // having read past the end of their stride vectors.
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> invRms({2, 1, 1});
    Tensor<float> dscale({3, 4});
    Tensor<float> dbias({3, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale, &dbias);
        },
        "dx and dscale");
}

TEST(TestCpuFpReferenceRankValidation, RmsNormBackwardRejectsShortBiasGradientRank)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> invRms({2, 1, 1});
    Tensor<float> dscale({1, 3, 4});
    Tensor<float> dbias({3, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale, &dbias);
        },
        "dbias rank");
}

TEST(TestCpuFpReferenceRankValidation, RmsNormForwardRejectsShortOptionalRanks)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> y({2, 3, 4});
    Tensor<float> invRms({2, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::forward<float, float, float, float>(x, scale, y, 1e-5, &invRms);
        },
        "invRms rank");
}

TEST(TestCpuFpReferenceRankValidation, RmsNormForwardRejectsMismatchedOutputRank)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> y({2, 3, 4, 1});

    expectRejectedWith(
        [&] { CpuFpReferenceRMSNorm::forward<float, float, float, float>(x, scale, y, 1e-5); },
        "y rank to equal input rank");
}
