// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The convolution and normalization references address memory through hoisted base
// pointers and strides, so every tensor they touch must be dense and must have the shape
// its offsets are computed against. A ragged tensor rebases each batch at
// ragged_offset[b] inside getIndexImpl, which flat stride arithmetic never calls; a
// mis-shaped tensor is read or written out of bounds. Both are rejected up front.
//
// The CPU graph executor already validates graphs one level up (see
// cpu_graph_executor/TestCpuReferenceRaggedRejection.cpp). These tests cover the direct
// callers the executor does not sit in front of - the samples and the gpu-ref fixtures.

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
    return {RAGGED_DIMS, RAGGED_STRIDES, RAGGED_SEQ_AXIS, aux};
}

// Asserts the call was rejected for the reason under test rather than by some unrelated
// check that happens to throw from the same function.
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

// ============================================================================
// Ragged rejection
// ============================================================================

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionFpropRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> w({1, 3, 1, 1});
    Tensor<float> y({2, 1, 2, 2});

    expectRejectedWith(
        [&] {
            CpuFpReferenceConvolution::fprop<float, float, float, float>(
                x, w, y, {1, 1}, {1, 1}, {0, 0});
        },
        "ragged x tensor");
}

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionDgradRejectsRaggedGradient)
{
    auto aux = makeOffsetAux();
    auto gradX = makeRaggedTensor(aux);
    Tensor<float> w({1, 3, 1, 1});
    Tensor<float> gradY({2, 1, 2, 2});

    expectRejectedWith(
        [&] {
            CpuFpReferenceConvolution::dgrad<float, float, float, float>(
                gradX, w, gradY, {1, 1}, {1, 1}, {0, 0});
        },
        "ragged dx tensor");
}

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionWgradRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> gradW({1, 3, 1, 1});
    Tensor<float> gradY({2, 1, 2, 2});

    expectRejectedWith(
        [&] {
            CpuFpReferenceConvolution::wgrad<float, float, float, float>(
                x, gradW, gradY, {1, 1}, {1, 1}, {0, 0});
        },
        "ragged x tensor");
}

TEST(TestCpuFpReferenceRaggedRejection, ConvolutionFpropRejectsRaggedOutput)
{
    // The output is written through the same hoisted pointer, so it needs the guard too.
    auto aux = makeOffsetAux();
    Tensor<float> x({2, 3, 2, 2});
    Tensor<float> w({3, 3, 1, 1});
    auto y = makeRaggedTensor(aux);

    expectRejectedWith(
        [&] {
            CpuFpReferenceConvolution::fprop<float, float, float, float>(
                x, w, y, {1, 1}, {1, 1}, {0, 0});
        },
        "ragged y tensor");
}

TEST(TestCpuFpReferenceRaggedRejection, LayernormFpropRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> y(RAGGED_DIMS);

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, nullptr, nullptr, y, 1e-5, 1);
        },
        "ragged x tensor");
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

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::bprop<float, float, float, float, float>(
                dy, x, scale, dx, dscale, dbias, 1e-5, nullptr, nullptr, 1);
        },
        "ragged dx tensor");
}

TEST(TestCpuFpReferenceRaggedRejection, RmsNormForwardRejectsRaggedInput)
{
    auto aux = makeOffsetAux();
    auto x = makeRaggedTensor(aux);
    Tensor<float> scale({1, 1, 2, 2});
    Tensor<float> y(RAGGED_DIMS);

    expectRejectedWith(
        [&] { CpuFpReferenceRMSNorm::forward<float, float, float, float>(x, scale, y, 1e-5); },
        "ragged x tensor");
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

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale);
        },
        "ragged dx tensor");
}

// ============================================================================
// Shape validation: layernorm
// ============================================================================

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsOutputNotShapedLikeInput)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> y({2, 3, 5});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, nullptr, nullptr, y, 1e-5, 1);
        },
        "y to have the same shape as x");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsBiasNotShapedLikeScale)
{
    // bias is walked with scale's offsets, so a shorter bias is read past its end.
    Tensor<float> x({2, 4});
    Tensor<float> scale({4});
    Tensor<float> bias({3});
    Tensor<float> y({2, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, &scale, &bias, y, 1e-5, 1);
        },
        "scale and bias to have the same shape");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsRstdNotShapedLikeMean)
{
    // The batch walk is taken from mean, so a lower-rank rstd is written through strides
    // it does not have.
    Tensor<float> x({2, 3, 4});
    Tensor<float> y({2, 3, 4});
    Tensor<float> mean({2, 3});
    Tensor<float> rstd({2});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, nullptr, nullptr, y, 1e-5, 1, &mean, &rstd);
        },
        "mean and rstd to have the same shape");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsScaleRankBelowNormalizedDimCount)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({4});
    Tensor<float> bias({4});
    Tensor<float> y({2, 3, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, &scale, &bias, y, 1e-5, 2);
        },
        "trailing normalizedDimCount dims");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsScaleNotMatchingNormalizedDims)
{
    Tensor<float> x({2, 4});
    Tensor<float> scale({3});
    Tensor<float> y({2, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, &scale, nullptr, y, 1e-5, 1);
        },
        "trailing normalizedDimCount dims");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsStatsNotMatchingBatchDims)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> y({2, 3, 4});
    Tensor<float> mean({2, 4});
    Tensor<float> rstd({2, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::fprop<float, float, float, float, float>(
                x, nullptr, nullptr, y, 1e-5, 1, &mean, &rstd);
        },
        "leading batch dims");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormFpropRejectsRankZeroStats)
{
    // A rank-0 tensor holds no element, so a whole-tensor normalization still needs a
    // one-element mean/rstd to write its single batch position to.
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

TEST(TestCpuFpReferenceShapeValidation, LayernormBpropRejectsInputNotShapedLikeGradient)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 4, 4});
    Tensor<float> scale({4});
    Tensor<float> dscale({4});
    Tensor<float> dbias({4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::bprop<float, float, float, float, float>(
                dy, x, scale, dx, dscale, dbias, 1e-5, nullptr, nullptr, 1);
        },
        "x and dx to have the same shape as dy");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormBpropRejectsScaleRankBelowNormalizedDimCount)
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
        "trailing normalizedDimCount dims");
}

TEST(TestCpuFpReferenceShapeValidation, LayernormBpropRejectsRankZeroStats)
{
    Tensor<float> dy({4});
    Tensor<float> x({4});
    Tensor<float> dx({4});
    Tensor<float> scale({4});
    Tensor<float> dscale({4});
    Tensor<float> dbias({4});
    Tensor<float> mean({});
    Tensor<float> rstd({});

    expectRejectedWith(
        [&] {
            CpuFpReferenceLayernorm::bprop<float, float, float, float, float>(
                dy, x, scale, dx, dscale, dbias, 1e-5, &mean, &rstd, 1);
        },
        "at least one dimension");
}

// ============================================================================
// Shape validation: RMSNorm
// ============================================================================

TEST(TestCpuFpReferenceShapeValidation, RmsNormForwardRejectsOutputNotShapedLikeInput)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> y({2, 3, 5});

    expectRejectedWith(
        [&] { CpuFpReferenceRMSNorm::forward<float, float, float, float>(x, scale, y, 1e-5); },
        "y to have the same shape as x");
}

TEST(TestCpuFpReferenceShapeValidation, RmsNormForwardRejectsBiasNotShapedLikeScale)
{
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> y({2, 3, 4});
    Tensor<float> bias({1, 1, 4});
    Tensor<float>* noInvRms = nullptr;

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::forward<float, float, float, float>(
                x, scale, y, 1e-5, noInvRms, &bias);
        },
        "bias to have the same shape as scale");
}

TEST(TestCpuFpReferenceShapeValidation, RmsNormForwardRejectsInvRmsNotCollapsingNormalizedDims)
{
    // scale [1, 1, 4] normalizes over the last dim only, so invRms is [2, 3, 1]; the
    // [2, 1, 1] of a wider normalization is written past its end.
    Tensor<float> x({2, 3, 4});
    Tensor<float> scale({1, 1, 4});
    Tensor<float> y({2, 3, 4});
    Tensor<float> invRms({2, 1, 1});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::forward<float, float, float, float>(x, scale, y, 1e-5, &invRms);
        },
        "invRms to have the input shape");
}

TEST(TestCpuFpReferenceShapeValidation, RmsNormBackwardRejectsGradientNotShapedLikeInput)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 5});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> invRms({2, 1, 1});
    Tensor<float> dscale({1, 3, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale);
        },
        "dy and dx to have the same shape as x");
}

TEST(TestCpuFpReferenceShapeValidation, RmsNormBackwardRejectsWeightGradientNotShapedLikeScale)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> invRms({2, 1, 1});
    Tensor<float> dscale({3, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale);
        },
        "dscale and dbias to have the same shape as scale");
}

TEST(TestCpuFpReferenceShapeValidation, RmsNormBackwardRejectsBiasGradientNotShapedLikeScale)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> scale({1, 3, 4});
    Tensor<float> invRms({2, 1, 1});
    Tensor<float> dscale({1, 3, 4});
    Tensor<float> dbias({1, 1, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale, &dbias);
        },
        "dscale and dbias to have the same shape as scale");
}

TEST(TestCpuFpReferenceShapeValidation, RmsNormBackwardRejectsInvRmsNotCollapsingNormalizedDims)
{
    Tensor<float> dy({2, 3, 4});
    Tensor<float> x({2, 3, 4});
    Tensor<float> dx({2, 3, 4});
    Tensor<float> scale({1, 1, 4});
    Tensor<float> invRms({2, 1, 1});
    Tensor<float> dscale({1, 1, 4});

    expectRejectedWith(
        [&] {
            CpuFpReferenceRMSNorm::backward<float, float, float, float, float>(
                dy, x, scale, invRms, dx, dscale);
        },
        "invRms to have the input shape");
}
