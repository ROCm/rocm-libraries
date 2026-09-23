// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_blocked_backend_test_support.hpp"

namespace blocked_backend_test {
void testBlockScaleAppliedAfterCompleteScaleSegment() {
    using namespace roc::host_numerics;

    constexpr size_t reductionElements = 16;
    const std::vector<float> a(reductionElements, 1.0f);
    std::vector<float> b(reductionElements, 1.0e37f);
    std::fill(b.begin() + reductionElements / 2, b.end(), -1.0e37f);
    const std::array<float, 1> scaleA{8.0f};
    Tensor blockedOutput = makeOutput(1, 1, untouchedValue);

    GemmTestCase blockedProblem = makeProblem(a, b, blockedOutput, 1, reductionElements, 1);
    const Tensor blockScaleA = Tensor::copyNativeValues<float>(Shape{1, 1}, scaleA);
    blockedProblem.blockScaleA = blockScaleA;
    blockedProblem.blockSizeA = reductionElements;

    runAndCheck(blockedProblem, blockedOutput,
                "Blocked GEMM applied a one-sided block scale before its complete reduction "
                "segment.");
    require(std::isfinite(blockedOutput.loadAs<float>({0, 0})),
            "Blocked GEMM overflowed scale-segment partial sums.");
}

void testOneSidedBlockScaling() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 2;
    constexpr size_t reductionElements = 16;
    constexpr size_t columns = 2;
    const std::vector<float> a(rows * reductionElements, 1.0f);
    const std::vector<float> b(reductionElements * columns, 1.0f);
    const std::array<float, 4> scales{2.0f, 3.0f, 4.0f, 5.0f};
    const Tensor blockScale = Tensor::copyNativeValues<float>(Shape{2, 2}, scales);

    const auto checkOneSide = [&](bool scaleOperandA, const std::array<float, 4>& expected) {
        Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
        GemmTestCase blockedProblem =
            makeProblem(a, b, blockedOutput, rows, reductionElements, columns);
        if (scaleOperandA) {
            blockedProblem.blockScaleA = blockScale;
            blockedProblem.blockSizeA = 8;
        } else {
            blockedProblem.blockScaleB = blockScale;
            blockedProblem.blockSizeB = 8;
        }

        runAndCheck(blockedProblem, blockedOutput, "One-sided block scaling result mismatch.");
        require(
            compare(blockedOutput, Tensor::copyNativeValues<float>(Shape{rows, columns}, expected))
                .passed(),
            "One-sided block scaling produced an incorrect result.");
    };

    checkOneSide(true, {40.0f, 40.0f, 72.0f, 72.0f});
    checkOneSide(false, {40.0f, 72.0f, 40.0f, 72.0f});
}

void testOneSidedBlockScalingWithZeroReductionExtent() {
    using namespace roc::host_numerics;

    constexpr size_t rows = 2;
    constexpr size_t columns = 2;
    const std::vector<float> empty;
    Tensor blockedOutput = makeOutput(rows, columns, untouchedValue);
    GemmTestCase blockedProblem = makeProblem(empty, empty, blockedOutput, rows, 0, columns);
    const Tensor emptyScale(ScalarType::Float32, Shape{rows, 0});
    blockedProblem.blockScaleA = emptyScale;
    blockedProblem.blockSizeA = 8;
    runAndCheck(blockedProblem, blockedOutput,
                "One-sided block scaling read scales for an empty reduction dimension.");
    require(compare(blockedOutput,
                    Tensor::copyNativeValues<float>(Shape{rows, columns},
                                                    std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}))
                .passed(),
            "One-sided zero-K block scaling did not produce a null product.");
}

}  // namespace blocked_backend_test
