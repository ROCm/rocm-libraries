// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <roc/host_numerics/validation.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gemm_test_adapter.hpp"

#ifdef HOST_NUMERICS_TEST_OPENMP
#include <omp.h>
#endif

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void requireInvalidArgument(Operation&& operation, const char* message) {
    bool invalidArgumentThrown = false;
    try {
        std::forward<Operation>(operation)();
    } catch (const std::invalid_argument&) {
        invalidArgumentThrown = true;
    }
    require(invalidArgumentThrown, message);
}

std::vector<std::byte> copyRawEncodedBackingStorage(const roc::host_numerics::Tensor& tensor) {
    const std::span<const std::byte> storage = tensor.rawEncodedBackingStorage();
    return {storage.begin(), storage.end()};
}

void requireRawEncodedBackingStorageEquals(const roc::host_numerics::Tensor& tensor,
                                           std::span<const std::byte> expected,
                                           const char* message) {
    const std::span<const std::byte> actual = tensor.rawEncodedBackingStorage();
    require(actual.size() == expected.size() &&
                std::equal(actual.begin(), actual.end(), expected.begin()),
            message);
}

void testRuntimeReferenceGemm() {
    using namespace roc::host_numerics;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    Tensor d(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}));
    const std::array<float, 2> bias{1, -10000};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    GemmTestCase problem(
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b)),
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)), d,
        ScalarType::Float32);
    problem.beta = 1.0;
    problem.scaleC = 2.0;
    problem.bias =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2}),
                                         std::span<const float>(bias))
            .expandDims(1);
    problem.scaleA =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2}),
                                         std::span<const float>(scaleA))
            .expandDims(1);
    problem.scaleB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2}), std::span<const float>(scaleB));
    problem.activation = Activation::Relu;

    const GemmSupportInfo mixedSupport = queryGemmSupport(problem, GemmBackend::Mixed);
    require(!mixedSupport && mixedSupport.reason == "Mixed is a reporting-only GEMM backend value.",
            "GEMM accepted Mixed as an execution request.");
    bool mixedExecutionRejected = false;
    try {
        (void)referenceGemm(problem, GemmBackend::Mixed);
    } catch (const std::invalid_argument& error) {
        mixedExecutionRejected = error.what() == mixedSupport.reason;
    }
    require(mixedExecutionRejected, "GEMM executed the reporting-only Mixed backend value.");

    GemmBackend backend = GemmBackend::Automatic;
    require(static_cast<bool>(queryGemmSupport(problem, backend)),
            "Runtime reference GEMM request support mismatch.");
    const GemmTestRunInfo runInfo = referenceGemm(problem, backend);
    require(runInfo.backendUsed == GemmBackend::Pointwise && runInfo.outputElementsWritten == 4 &&
                runInfo.outputElementsCovered == 4,
            "Runtime reference GEMM run information mismatch.");

    const std::array<float, 4> expected{
        58 * 2 * 5 + 2 + 1,
        0,
        64 * 2 * 7 + 2 + 1,
        0,
    };
    require(compare(d, Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 2}),
                                                        std::span<const float>(expected)))
                .passed(),
            "Runtime reference GEMM result mismatch.");

    backend = GemmBackend::Blocked;
    require(static_cast<bool>(queryGemmSupport(problem, backend)),
            "Built-in Blocked reference GEMM unexpectedly rejected the request.");
    const GemmTestRunInfo blocked = referenceGemm(problem, backend);
    require(blocked.backendUsed == GemmBackend::Blocked && !blocked.fallbackReason.has_value(),
            "Built-in Blocked reference GEMM dispatch mismatch.");

    const GemmTestSpecification owningProblem = problem;
    const Layout owningLayout(Shape{2, 2}, {7, 2}, 1);
    const GemmTestOutputOptions owningOutput{
        .layout = owningLayout,
        .selection = OutputSelection::explicitIndices({0, 1}),
    };
    const GemmTestResult owned = referenceGemm(owningProblem, owningOutput, GemmBackend::Pointwise);
    require(owned.output.layout() == owningLayout && owned.runInfo.outputElementsWritten == 2 &&
                owned.output.loadAs<float>({0, 0}) == expected[0] &&
                owned.output.loadAs<float>({0, 1}) == expected[2] &&
                owned.output.loadAs<float>({1, 0}) == 0 && owned.output.loadAs<float>({1, 1}) == 0,
            "Owning reference GEMM result contract mismatch.");
    std::array<float, 11> ownedStorage;
    std::memcpy(ownedStorage.data(), owned.output.rawEncodedBackingStorage().data(),
                sizeof(ownedStorage));
    std::array<float, 11> expectedOwnedStorage{};
    expectedOwnedStorage[1] = expected[0];
    expectedOwnedStorage[3] = expected[2];
    require(ownedStorage == expectedOwnedStorage,
            "Owning reference GEMM did not zero unselected storage.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceGemm(owningProblem,
                            {.layout = Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                             .selection = OutputSelection::all()});
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference GEMM accepted an invalid output layout.");
}

void testZeroGemmScalarsSuppressNonFiniteOperands() {
    using namespace roc::host_numerics;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<float, 4> nonFiniteA{nan, nan, nan, nan};
    const std::array<float, 4> nonFiniteB{infinity, infinity, infinity, infinity};
    const std::array<float, 4> finiteC{1, 2, 3, 4};
    Tensor output(ScalarType::Float32, Shape{2, 2});

    GemmTestCase alphaZero(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(nonFiniteA)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(nonFiniteB)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(finiteC)),
        output, ScalarType::Float32);
    alphaZero.alpha = 0.0;
    alphaZero.beta = 2.0;
    referenceGemm(alphaZero);
    require(compare(output, Tensor::copyNativeStorage<float>(
                                Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                std::span<const float>(std::array<float, 4>{2, 4, 6, 8})))
                .passed(),
            "Zero alpha propagated a non-finite GEMM operand.");

    alphaZero.scaleA = Tensor::copyNativeValues<float>(Shape{1}, std::array<float, 1>{nan});
    alphaZero.scaleB = Tensor::copyNativeValues<float>(Shape{1}, std::array<float, 1>{3.0f});
    referenceGemm(alphaZero);
    require(compare(output, Tensor::copyNativeStorage<float>(
                                Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                std::span<const float>(std::array<float, 4>{2, 4, 6, 8})))
                .passed(),
            "Zero alpha evaluated a broadcast scale or non-finite GEMM operand.");
    alphaZero.scaleA =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{nan, nan}).expandDims(1);
    referenceGemm(alphaZero);
    require(compare(output, Tensor::copyNativeStorage<float>(
                                Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                std::span<const float>(std::array<float, 4>{2, 4, 6, 8})))
                .passed(),
            "Zero alpha treated broadcast and expanded non-finite scales differently.");

    alphaZero.alpha = 1.0;
    alphaZero.scaleA = Tensor::copyNativeValues<float>(Shape{1}, std::array<float, 1>{0.0f});
    referenceGemm(alphaZero);
    require(std::isnan(output.loadAs<float>({0, 0})),
            "A zero broadcast scale did not preserve IEEE multiplication of a non-finite dot "
            "product.");
    alphaZero.scaleA =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{0.0f, 0.0f}).expandDims(1);
    referenceGemm(alphaZero);
    require(std::isnan(output.loadAs<float>({0, 0})),
            "Broadcast and expanded zero scales produced different non-finite behavior.");

    const std::array<float, 4> finiteA{1, 2, 3, 4};
    const std::array<float, 4> finiteB{5, 6, 7, 8};
    const std::array<float, 4> nonFiniteC{infinity, infinity, infinity, infinity};
    GemmTestCase betaZero(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(finiteA)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(finiteB)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(nonFiniteC)),
        output, ScalarType::Float32);
    betaZero.beta = 0.0;
    referenceGemm(betaZero);
    require(compare(output, Tensor::copyNativeStorage<float>(
                                Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                std::span<const float>(std::array<float, 4>{19, 22, 43, 50})))
                .passed(),
            "Zero beta propagated a non-finite C operand.");
}

void testGemmScaleCValidationMatchesExecution() {
    using namespace roc::host_numerics;

    const std::array<float, 1> realValues{1.0f};
    Tensor realOutput = Tensor::copyNativeValues<float>(Shape{1, 1}, std::array<float, 1>{-7.0f});
    GemmTestCase realProblem(Tensor::copyNativeStorage<float>(std::span<const float>(realValues)),
                             Tensor::copyNativeStorage<float>(std::span<const float>(realValues)),
                             Tensor::copyNativeStorage<float>(std::span<const float>(realValues)),
                             realOutput, ScalarType::Float32);
    realProblem.scaleC = std::complex<double>(1.0, 1.0);

    require(!queryGemmSupport(realProblem, GemmBackend::Pointwise),
            "GEMM support query accepted a complex C scale for a real accumulator.");
    bool rejectedComplexScaleC = false;
    try {
        referenceGemm(realProblem, GemmBackend::Pointwise);
    } catch (const std::invalid_argument&) {
        rejectedComplexScaleC = true;
    }
    require(rejectedComplexScaleC && realOutput.loadAs<float>({0, 0}) == -7.0f,
            "GEMM execution did not reject an incompatible C scale before writing output.");

    const std::array<int32_t, 1> integerValues{1};
    Tensor integerOutput =
        Tensor::copyNativeValues<int32_t>(Shape{1, 1}, std::array<int32_t, 1>{-7});
    GemmTestCase integerProblem(
        Tensor::copyNativeStorage<int32_t>(std::span<const int32_t>(integerValues)),
        Tensor::copyNativeStorage<int32_t>(std::span<const int32_t>(integerValues)),
        Tensor::copyNativeStorage<int32_t>(std::span<const int32_t>(integerValues)), integerOutput,
        ScalarType::Int32);
    integerProblem.scaleC = 0.5;

    require(!queryGemmSupport(integerProblem, GemmBackend::Pointwise),
            "GEMM support query accepted a fractional C scale for an integer accumulator.");
    bool rejectedFractionalScaleC = false;
    try {
        referenceGemm(integerProblem, GemmBackend::Pointwise);
    } catch (const std::invalid_argument&) {
        rejectedFractionalScaleC = true;
    }
    require(rejectedFractionalScaleC && integerOutput.loadAs<int32_t>({0, 0}) == -7,
            "GEMM execution did not reject a fractional C scale before writing output.");
}

void testRuntimeMixedAndBlockScaledGemm() {
    using namespace roc::host_numerics;

    const std::array<float, 2> aValues{1.25f, 2.5f};
    const std::array<float, 2> bValues{2.0f, 3.0f};
    const std::array<float, 1> cValues{1.0f};
    Tensor a = Tensor::copyValuesWithConversion(ScalarType::Float8E4M3, Shape{1, 2},
                                                std::span<const float>(aValues));
    Tensor b = Tensor::copyValuesWithConversion(ScalarType::Float8E5M2, Shape{2, 1},
                                                std::span<const float>(bValues));
    Tensor c = Tensor::copyValuesWithConversion(ScalarType::BFloat16, Shape{1, 1},
                                                std::span<const float>(cValues));
    Tensor d(ScalarType::Float16, Shape{1, 1});

    GemmTestCase mixed(a, b, c, d, ScalarType::Float32);
    mixed.computeTypeA = ScalarType::Float4E2M1;
    mixed.beta = 1.0;
    referenceGemm(mixed);
    require(d.loadAs<float>({0, 0}) == 9.0f, "Runtime mixed-type GEMM result mismatch.");

    const std::array<float, 4> ones{1, 1, 1, 1};
    const std::array<float, 1> zero{0};
    const std::array<float, 2> scaleAValues{2, 4};
    const std::array<float, 2> scaleBValues{8, 16};
    Tensor blockA = Tensor::copyValuesWithConversion(ScalarType::Float32, Shape{1, 4},
                                                     std::span<const float>(ones));
    Tensor blockB = Tensor::copyValuesWithConversion(ScalarType::Float32, Shape{4, 1},
                                                     std::span<const float>(ones));
    Tensor blockC = Tensor::copyValuesWithConversion(ScalarType::Float32, Shape{1, 1},
                                                     std::span<const float>(zero));
    Tensor blockD(ScalarType::Float32, Shape{1, 1});
    Tensor scalesA = Tensor::copyValuesWithConversion(ScalarType::E8M0, Shape{1, 2},
                                                      std::span<const float>(scaleAValues));
    Tensor scalesB = Tensor::copyValuesWithConversion(ScalarType::E8M0, Shape{1, 2},
                                                      std::span<const float>(scaleBValues));

    GemmTestCase blockScaled(blockA, blockB, blockC, blockD, ScalarType::Float32);
    blockScaled.blockScaleA = scalesA;
    blockScaled.blockSizeA = 2;
    blockScaled.blockScaleB = scalesB;
    blockScaled.blockSizeB = 2;
    referenceGemm(blockScaled);
    require(blockD.loadAs<float>({0, 0}) == 2 * 2 * 8 + 2 * 4 * 16,
            "Runtime block-scaled GEMM result mismatch.");
}

void testPointwiseRoutes() {
    using namespace roc::host_numerics;

    const std::array<float, 7> a{1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 14> b{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 2> c{};
    const std::array<float, 3> scaleA{2, 3, 5};
    const std::array<float, 4> scaleB{1, 1, 7, 11};

    auto makeProblem = [&](Tensor output) {
        Tensor operandA = Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{1, 7}), std::span<const float>(a));
        Tensor operandB = Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{7, 2}), std::span<const float>(b));
        GemmTestCase problem(
            std::move(operandA), std::move(operandB),
            Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 2}),
                                             std::span<const float>(c)),
            std::move(output), ScalarType::Float32);
        problem.blockScaleA = Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{1, 3}), std::span<const float>(scaleA));
        problem.blockSizeA = 3;
        problem.blockScaleB = Tensor::copyNativeStorage<float>(
            Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(scaleB));
        problem.blockSizeB = 4;
        problem.outputSelection = OutputSelection::explicitIndices({1});
        return problem;
    };

    Tensor automaticOutput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, -99});
    GemmTestCase automaticProblem = makeProblem(automaticOutput);
    const GemmTestRunInfo automatic = referenceGemm(automaticProblem);

    Tensor pointwiseOutput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, -99});
    GemmTestCase pointwiseProblem = makeProblem(pointwiseOutput);
    const GemmTestRunInfo pointwise = referenceGemm(pointwiseProblem, GemmBackend::Pointwise);

    const Tensor expected =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, 184});
    require(
        compare(automaticOutput, expected).passed() && compare(pointwiseOutput, expected).passed(),
        "Automatic and explicit Pointwise routes diverged.");
    require(automatic.backendUsed == GemmBackend::Pointwise &&
                pointwise.backendUsed == GemmBackend::Pointwise &&
                automatic.outputElementsWritten == 1 && pointwise.outputElementsWritten == 1 &&
                automatic.outputElementsCovered == 1 && pointwise.outputElementsCovered == 1 &&
                automatic.fallbackReason && !pointwise.fallbackReason,
            "Pointwise route information changed.");
}

void testExactIntegerGemm() {
    using namespace roc::host_numerics;

    const std::array<int32_t, 1> a{std::numeric_limits<int32_t>::max()};
    const std::array<int32_t, 1> b{2};
    const std::array<int32_t, 1> c{std::numeric_limits<int32_t>::max()};
    Tensor d(ScalarType::Int32, Shape{1, 1});
    GemmTestCase problem(
        Tensor::copyNativeStorage(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                  std::span<const int32_t>(a)),
        Tensor::copyNativeStorage(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                  std::span<const int32_t>(b)),
        Tensor::copyNativeStorage(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                  std::span<const int32_t>(c)),
        d, ScalarType::Int32);
    problem.beta = 2.0;
    referenceGemm(problem);

    const auto wrapMultiply = [](int32_t left, int32_t right) {
        return std::bit_cast<int32_t>(static_cast<uint32_t>(left) * static_cast<uint32_t>(right));
    };
    const auto wrapAdd = [](int32_t left, int32_t right) {
        return std::bit_cast<int32_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
    };
    const int32_t expected = wrapAdd(wrapMultiply(a[0], b[0]), wrapMultiply(int32_t{2}, c[0]));
    require(d.loadAs<int32_t>({0, 0}) == expected,
            "Int32 GEMM did not use defined wrapping arithmetic.");

    problem.alpha = 0.5;
    bool rejectedFractionalIntegerScalar = false;
    try {
        referenceGemm(problem);
    } catch (const std::invalid_argument&) {
        rejectedFractionalIntegerScalar = true;
    }
    require(rejectedFractionalIntegerScalar,
            "Int32 GEMM accepted a fractional floating-point scalar proxy.");
}

void testRuntimeComplexAndExplicitAxisGemm() {
    using namespace roc::host_numerics;

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1.0f, 2.0f)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3.0f, 4.0f)};
    const std::array<std::complex<float>, 1> complexC{};
    Tensor complexD(ScalarType::ComplexFloat32, Shape{1, 1});

    Tensor complexOperandA = Tensor::copyNativeStorage<std::complex<float>>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}),
        std::span<const std::complex<float>>(complexA));
    GemmTestCase complexProblem(std::move(complexOperandA),
                                Tensor::copyNativeStorage<std::complex<float>>(
                                    Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                    std::span<const std::complex<float>>(complexB)),
                                Tensor::copyNativeStorage<std::complex<float>>(
                                    Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                    std::span<const std::complex<float>>(complexC)),
                                complexD, ScalarType::ComplexFloat32);
    complexProblem.conjugateA = true;
    referenceGemm(complexProblem);
    require(complexD.loadAs<std::complex<float>>({0, 0}) == std::complex<float>(11.0f, -2.0f),
            "Runtime complex GEMM result mismatch.");

    const std::array<float, 1> realA{1};
    const std::array<float, 2> realB{0, 0};
    const std::array<float, 2> realC{0, 0};
    const std::array<float, 2> columnBias{2, 3};
    Tensor realD(ScalarType::Float32, Shape{1, 2});
    GemmTestCase axisProblem(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(realA)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 2}),
                                         std::span<const float>(realB)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 2}),
                                         std::span<const float>(realC)),
        realD, ScalarType::Float32);
    axisProblem.bias = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2}), std::span<const float>(columnBias));
    referenceGemm(axisProblem);
    require(compare(realD, Tensor::copyNativeStorage<float>(
                               Layout::contiguousLastDimensionFastest(Shape{1, 2}),
                               std::span<const float>(columnBias)))
                .passed(),
            "Runtime GEMM explicit column-axis bias mismatch.");
}

void testOutputSelection() {
    using namespace roc::host_numerics;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{};
    Tensor d =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});

    GemmTestCase problem(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(b)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(c)),
        d, ScalarType::Float32);
    problem.outputSelection = OutputSelection::explicitIndices({0, 3});
    const GemmTestRunInfo runInfo = referenceGemm(problem);
    require(runInfo.outputElementsWritten == 2 && runInfo.outputElementsCovered == 2,
            "Selected-output GEMM reported the wrong element count.");
    require(d.loadAs<float>({0, 0}) == 19 && d.loadAs<float>({0, 1}) == -99 &&
                d.loadAs<float>({1, 0}) == -99 && d.loadAs<float>({1, 1}) == 50,
            "Selected-output GEMM modified the wrong elements.");

    Tensor firstDimensionFastestOutput =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
    GemmTestCase firstDimensionFastestProblem(problem, firstDimensionFastestOutput);
    firstDimensionFastestProblem.outputSelection =
        OutputSelection::explicitIndices({1}, IndexOrder::FirstDimensionFastest);
    referenceGemm(firstDimensionFastestProblem);
    require(firstDimensionFastestOutput.loadAs<float>({0, 0}) == -99 &&
                firstDimensionFastestOutput.loadAs<float>({0, 1}) == -99 &&
                firstDimensionFastestOutput.loadAs<float>({1, 0}) == 43 &&
                firstDimensionFastestOutput.loadAs<float>({1, 1}) == -99,
            "Selected-output GEMM ignored the selection index order.");

    const auto prime = OutputSelection::primeStride(10, 10, 3).indices(10);
    require(prime == std::vector<size_t>({0, 3, 6, 9}), "Prime-stride output selection mismatch.");
    require(OutputSelection::primeStride(20, 9, 1).indices(20) == std::vector<size_t>({0, 11}),
            "Prime-stride output selection failed after a multiple of three.");
    require(OutputSelection::primeStride(60, 49, 1).indices(60) == std::vector<size_t>({0, 53}),
            "Prime-stride output selection failed after a squared factor.");
    require(OutputSelection::primeStride(128, 121, 1).indices(128) == std::vector<size_t>({0, 127}),
            "Prime-stride output selection failed after a larger squared factor.");
    require(OutputSelection::strided(2, 3).selectedCount(10) == 3,
            "Strided output selection reported the wrong count.");
    require(OutputSelection::strided(1, 2, 1).indices(10) == std::vector<size_t>({1}),
            "Bounded strided output selection ignored its maximum count.");
    require(OutputSelection::strided(10, 3).selectedCount(10) == 0,
            "Out-of-range strided output selection reported a nonzero count.");
    require(OutputSelection::primeStride(10, 10, 0).selectsAll(),
            "Zero requested elements did not preserve all-output behavior.");
    const OutputSelection explicitSelection = OutputSelection::explicitIndices({3, 0, 3});
    require(explicitSelection.indices(4) == std::vector<size_t>({0, 3}),
            "Explicit output selection did not normalize to a unique ordered set.");
    require(explicitSelection.selectedCount(4) == 2,
            "Explicit output selection reported the wrong count.");
    bool rejectedOutOfRange = false;
    try {
        (void)explicitSelection.selectedCount(3);
    } catch (const std::out_of_range&) {
        rejectedOutOfRange = true;
    }
    require(rejectedOutOfRange, "Out-of-range explicit selection count did not fail.");
    rejectedOutOfRange = false;
    try {
        (void)explicitSelection.indices(3);
    } catch (const std::out_of_range&) {
        rejectedOutOfRange = true;
    }
    require(rejectedOutOfRange, "Out-of-range explicit selection materialization did not fail.");
}

void testStreamingGemmValidation() {
    using namespace roc::host_numerics;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{};
    const Tensor tensorA = Tensor::copyNativeValues<float>(Shape{2, 2}, a);
    const Tensor tensorB = Tensor::copyNativeValues<float>(Shape{2, 2}, b);
    const Tensor tensorC = Tensor::copyNativeValues<float>(Shape{2, 2}, c);
    const GemmOptions gemmOptions(ScalarType::Float32);

    Tensor observed =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{19, 999, 43, 50});
    GemmValidationOptions options;
    options.comparison.computeFrobenius = false;
    options.comparison.selection =
        OutputSelection::explicitIndices({1}, IndexOrder::FirstDimensionFastest);

    ComparisonReport pointwise =
        validateGemm(tensorA, tensorB, tensorC, observed, gemmOptions, options);
    require(pointwise.passed() && pointwise.compared == 1,
            "Streaming GEMM validation did not isolate the selected output.");

    options.backend = GemmBackend::Blocked;
    ComparisonReport blocked =
        validateGemm(tensorA, tensorB, tensorC, observed, gemmOptions, options);
    require(blocked.passed(), "Streaming blocked GEMM validation reported the wrong work.");

    observed.storeFrom({1, 0}, 44.0f);
    ComparisonReport mismatch =
        validateGemm(tensorA, tensorB, tensorC, observed, gemmOptions, options);
    require(!mismatch.passed() && mismatch.mismatches == 1 &&
                mismatch.reportedMismatches[0].index == 1 &&
                mismatch.reportedMismatches[0].coordinates == std::vector<size_t>({1, 0}) &&
                mismatch.reportedMismatches[0].observedOffset == 2,
            "Streaming GEMM validation did not preserve the original logical location.");
}

void testReferenceEpilogue() {
    using namespace roc::host_numerics;

    const std::array<float, 4> input{-2, 1, 3, -4};
    const std::array<float, 2> bias{1, 2};
    Tensor output(ScalarType::Float16, Shape{2, 2});
    Tensor rawOutput(ScalarType::Float32, Shape{2, 2});
    Tensor auxiliary(ScalarType::BFloat16, Shape{2, 2});
    Tensor amax(ScalarType::Float32, Shape{1});

    const Tensor inputTensor = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(input));
    EpilogueOptions options(ScalarType::Float32);
    require(options.outputScale.type() == ScalarType::Float32 &&
                options.auxiliaryScale.type() == ScalarType::Float32 &&
                options.activationParameter0.type() == ScalarType::Float32 &&
                options.activationParameter1.type() == ScalarType::Float32,
            "Reference epilogue defaults do not use the requested compute type.");
    options.bias = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 1}), std::span<const float>(bias));
    options.outputScale = 2.0;
    options.auxiliaryScale = 3.0;
    options.activation = Activation::Relu;
    referenceEpilogueInto(
        inputTensor,
        {.output = output, .rawOutput = rawOutput, .auxiliaryOutput = auxiliary, .amax = amax},
        options);

    require(output.loadAs<float>({0, 0}) == 0 && output.loadAs<float>({0, 1}) == 4 &&
                output.loadAs<float>({1, 0}) == 10 && output.loadAs<float>({1, 1}) == 0,
            "Reference epilogue output mismatch.");
    require(compare(rawOutput,
                    Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{0, 4, 10, 0}))
                .passed(),
            "Reference epilogue raw output mismatch.");
    require(auxiliary.loadAs<float>({0, 0}) == -3 && auxiliary.loadAs<float>({0, 1}) == 6 &&
                auxiliary.loadAs<float>({1, 0}) == 15 && auxiliary.loadAs<float>({1, 1}) == -6,
            "Reference epilogue auxiliary output mismatch.");
    require(amax.loadAs<float>({0}) == 5, "Reference epilogue AMax mismatch.");

    const std::array<float, 4> gradientInput{10, 20, 30, 40};
    const std::array<float, 4> activationInput{-1, 1, 2, -2};
    Tensor gradientOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueOptions gradientOptions(ScalarType::Float32);
    gradientOptions.auxiliaryInput =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(activationInput));
    gradientOptions.activation = Activation::Relu;
    gradientOptions.activationApplication = ActivationApplication::Gradient;
    referenceEpilogueInto(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(gradientInput)),
        {.output = gradientOutput}, gradientOptions);
    require(compare(gradientOutput, Tensor::copyNativeValues<float>(
                                        Shape{2, 2}, std::array<float, 4>{0, 20, 30, 0}))
                .passed(),
            "Reference gradient epilogue mismatch.");

    const std::array<float, 4> gate{0.5f, 2.0f, -1.0f, 0.25f};
    Tensor gatedOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueOptions gatedOptions(ScalarType::Float32);
    gatedOptions.gateResidual = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(gate));
    gatedOptions.outputScale = 2.0;
    referenceEpilogueInto(inputTensor, {.output = gatedOutput}, gatedOptions);
    require(compare(gatedOutput, Tensor::copyNativeValues<float>(
                                     Shape{2, 2}, std::array<float, 4>{-1.5f, 6.0f, -7.0f, -1.75f}))
                .passed(),
            "Reference gate-residual epilogue mismatch.");

    const std::array<float, 4> int8Input{-200.0f, -128.5f, 126.5f, 300.0f};
    Tensor int8Output(ScalarType::Int8, Shape{2, 2});
    EpilogueOptions int8Options(ScalarType::Float32);
    int8Options.outputConversion = OutputConversion::SaturatingInt8;
    referenceEpilogueInto(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2, 2}),
                                         std::span<const float>(int8Input)),
        {.output = int8Output}, int8Options);
    require(compare(int8Output, Tensor::copyNativeValues<int8_t>(
                                    Shape{2, 2}, std::array<int8_t, 4>{-128, -128, 126, 127}))
                .passed(),
            "Reference epilogue Int8 saturation mismatch.");

    constexpr double highPrecisionInput = 1.0000000001;
    const std::array<double, 1> highPrecisionValues{highPrecisionInput};
    Tensor highPrecisionOutput(ScalarType::Float64, Shape{1, 1});
    EpilogueOptions highPrecisionOptions(ScalarType::Float64);
    highPrecisionOptions.activation = Activation::Sigmoid;
    referenceEpilogueInto(
        Tensor::copyNativeStorage<double>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                          highPrecisionValues),
        {.output = highPrecisionOutput}, highPrecisionOptions);
    const double expectedHighPrecision = 1.0 / (1.0 + std::exp(-highPrecisionInput));
    require(std::abs(highPrecisionOutput.loadAs<double>({0, 0}) - expectedHighPrecision) < 1e-15,
            "Float64 reference activation used reduced-precision intermediates.");

    for (const Activation activation : {Activation::Tanh, Activation::Swish}) {
        Tensor gradientResult(ScalarType::Float64, Shape{1, 1});
        EpilogueOptions highPrecisionGradient(ScalarType::Float64);
        highPrecisionGradient.auxiliaryInput = Tensor::copyNativeValues<double>(
            Shape{1, 1}, std::array<double, 1>{highPrecisionInput});
        highPrecisionGradient.activation = activation;
        highPrecisionGradient.activationApplication = ActivationApplication::Gradient;
        highPrecisionGradient.activationParameter0 = 1.0;
        highPrecisionGradient.activationParameter1 = 1.0;
        referenceEpilogueInto(
            Tensor::copyNativeValues<double>(Shape{1, 1}, std::array<double, 1>{1.0}),
            {.output = gradientResult}, highPrecisionGradient);

        const double sigmoid = 1.0 / (1.0 + std::exp(-highPrecisionInput));
        const double hyperbolicTangent = std::tanh(highPrecisionInput);
        const double expectedGradient =
            activation == Activation::Tanh
                ? 1.0 - hyperbolicTangent * hyperbolicTangent
                : sigmoid + highPrecisionInput * sigmoid * (1.0 - sigmoid);
        require(std::abs(gradientResult.loadAs<double>({0, 0}) - expectedGradient) < 1e-15,
                "Float64 reference activation gradient used reduced-precision intermediates.");
    }

    const Tensor ownedInput = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(input));
    EpilogueOptions ownedOptions = options;
    ownedOptions.outputSelection = OutputSelection::explicitIndices({1, 2});
    const EpilogueOutputs owned = referenceEpilogue(ownedInput,
                                                    {.output = ScalarType::Float16,
                                                     .rawOutput = ScalarType::Float32,
                                                     .auxiliaryOutput = ScalarType::BFloat16,
                                                     .amax = ScalarType::Float32},
                                                    ownedOptions);
    require(owned.output.layout() == Layout::contiguousLastDimensionFastest(Shape{2, 2}) &&
                owned.rawOutput && owned.auxiliaryOutput && owned.amax,
            "Owning reference epilogue result contract mismatch.");
    require(owned.output.loadAs<float>({0, 0}) == 0 && owned.output.loadAs<float>({0, 1}) == 4 &&
                owned.output.loadAs<float>({1, 0}) == 10 &&
                owned.output.loadAs<float>({1, 1}) == 0 &&
                owned.rawOutput->loadAs<float>({0, 0}) == 0 &&
                owned.rawOutput->loadAs<float>({1, 1}) == 0 &&
                owned.auxiliaryOutput->loadAs<float>({0, 0}) == 0 &&
                owned.auxiliaryOutput->loadAs<float>({1, 1}) == 0 &&
                owned.amax->loadAs<float>({0}) == 5,
            "Owning reference epilogue did not zero unselected values.");

    EpilogueOptions emptyOptions = ownedOptions;
    emptyOptions.outputSelection = OutputSelection::strided(4, 1);
    const EpilogueOutputs empty = referenceEpilogue(ownedInput,
                                                    {.output = ScalarType::Float16,
                                                     .rawOutput = ScalarType::Float32,
                                                     .auxiliaryOutput = ScalarType::BFloat16,
                                                     .amax = ScalarType::Float32},
                                                    emptyOptions);
    require(empty.output.loadAs<float>({0, 0}) == 0 && empty.rawOutput &&
                empty.rawOutput->loadAs<float>({0, 0}) == 0 && empty.auxiliaryOutput &&
                empty.auxiliaryOutput->loadAs<float>({0, 0}) == 0 && empty.amax &&
                empty.amax->loadAs<float>({0}) == 0,
            "Owning reference epilogue empty selection was not zero initialized.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceEpilogue(ownedInput, {.output = ScalarType::E8M0});
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference epilogue accepted an invalid output type.");

    Tensor preservedOutput =
        Tensor::copyNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
    Tensor preservedRaw = preservedOutput.deepCopy();
    Tensor preservedAuxiliary = preservedOutput.deepCopy();
    Tensor accumulatedAmax = Tensor::copyNativeValues<float>(Shape{1}, std::array<float, 1>{100});
    EpilogueOptions preservedOptions;
    preservedOptions.outputSelection = OutputSelection::explicitIndices({0, 3});
    preservedOptions.accumulateAmax = true;
    referenceEpilogueInto(ownedInput,
                          {.output = preservedOutput,
                           .rawOutput = preservedRaw,
                           .auxiliaryOutput = preservedAuxiliary,
                           .amax = accumulatedAmax},
                          preservedOptions);
    require(preservedOutput.loadAs<float>({0, 1}) == -99 &&
                preservedOutput.loadAs<float>({1, 0}) == -99 &&
                preservedRaw.loadAs<float>({0, 1}) == -99 &&
                preservedAuxiliary.loadAs<float>({1, 0}) == -99 &&
                accumulatedAmax.loadAs<float>({0}) == 100,
            "Explicit reference epilogue did not preserve unselected or accumulated state.");
}

void testReferenceReduction() {
    using namespace roc::host_numerics;

    std::array<float, 30> storage;
    storage.fill(-1);
    Tensor input = Tensor::copyNativeStorage<float>(Layout(Shape{2, 3, 4}, {15, 5, 1}),
                                                    std::span<float>(storage));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 4; ++column)
                input.storeFrom({batch, row, column}, 100 * batch + 10 * row + column);
        }
    }

    Tensor output(ScalarType::Float32, Layout(Shape{3}, {2}));
    referenceSumInto(input, output, {0, 2}, ScalarType::Float32);
    require(compare(output,
                    Tensor::copyNativeValues<float>(Shape{3}, std::array<float, 3>{412, 492, 572}))
                .passed(),
            "Reference reduction result mismatch.");

    const std::array<int32_t, 2> wrappingInputValues{std::numeric_limits<int32_t>::max(), 1};
    const Tensor wrappingSum =
        referenceSum(Tensor::copyNativeStorage<int32_t>(
                         Layout::contiguousLastDimensionFastest(Shape{2}), wrappingInputValues),
                     {0}, ScalarType::Int32, ScalarType::Int32);
    require(wrappingSum.loadAs<int32_t>({}) == std::numeric_limits<int32_t>::min(),
            "Int32 reference reduction did not use defined wrapping arithmetic.");

    Tensor maximumAbsolute(ScalarType::Float32, Shape{});
    referenceMaximumAbsoluteInto(input, maximumAbsolute, ScalarType::Float32);
    require(maximumAbsolute.loadAs<float>({}) == 123.0f,
            "Reference maximum-absolute result mismatch.");

    const Tensor owned = referenceSum(input, {0, 2}, ScalarType::Float32, ScalarType::Float32);
    require(owned.layout() == Layout::contiguousLastDimensionFastest(Shape{3}) &&
                owned.type() == ScalarType::Float32 &&
                compare(owned, Tensor::copyNativeValues<float>(Shape{3},
                                                               std::array<float, 3>{412, 492, 572}))
                    .passed(),
            "Owning reference sum result contract mismatch.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceSum(input, {0, 0}, ScalarType::Float32, ScalarType::Float32);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference sum accepted invalid axes.");

    const Tensor ownedMaximum =
        referenceMaximumAbsolute(input, ScalarType::Float32, ScalarType::Float32);
    require(ownedMaximum.shape() == Shape{} &&
                ownedMaximum.layout() == Layout::contiguousLastDimensionFastest(Shape{}) &&
                ownedMaximum.loadAs<float>({}) == 123.0f,
            "Owning reference maximum-absolute result contract mismatch.");
}

void testStructuredSparsity() {
    using namespace roc::host_numerics;

    std::array<float, 20> inputStorage;
    inputStorage.fill(-99);
    Tensor input =
        Tensor::copyEncodedBackingStorage(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                                          std::as_writable_bytes(std::span<float>(inputStorage)));
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 8; ++column)
            input.storeFrom({row, column}, static_cast<float>(1 + row * 8 + column));

    std::array<float, 20> prunedStorage;
    prunedStorage.fill(-7);
    Tensor pruned =
        Tensor::copyEncodedBackingStorage(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                                          std::as_writable_bytes(std::span<float>(prunedStorage)));
    std::array<float, 8> compressedStorage{};
    Tensor compressed = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 4}), std::span<float>(compressedStorage));
    std::array<uint8_t, 8> indexStorage{};
    Tensor retainedIndices = Tensor::copyNativeStorage<uint8_t>(
        Layout::contiguousLastDimensionFastest(Shape{2, 4}), std::span<uint8_t>(indexStorage));

    StructuredSparsityPattern pattern;
    pattern.axis = 1;
    pattern.fixedPositions = {1, 3};
    applyStructuredSparsityInto(
        input, {.pruned = pruned, .compressed = compressed, .retainedIndices = retainedIndices},
        pattern);

    for (size_t row = 0; row < 2; ++row) {
        for (size_t group = 0; group < 2; ++group) {
            const size_t inputBase = group * 4;
            const size_t compressedBase = group * 2;
            require(pruned.loadAs<float>({row, inputBase}) == 0 &&
                        pruned.loadAs<float>({row, inputBase + 1}) ==
                            input.loadAs<float>({row, inputBase + 1}) &&
                        pruned.loadAs<float>({row, inputBase + 2}) == 0 &&
                        pruned.loadAs<float>({row, inputBase + 3}) ==
                            input.loadAs<float>({row, inputBase + 3}),
                    "Structured sparsity pruned output mismatch.");
            require(compressed.loadAs<float>({row, compressedBase}) ==
                            input.loadAs<float>({row, inputBase + 1}) &&
                        compressed.loadAs<float>({row, compressedBase + 1}) ==
                            input.loadAs<float>({row, inputBase + 3}),
                    "Structured sparsity compressed output mismatch.");
            require(retainedIndices.loadAs<uint8_t>({row, compressedBase}) == 1 &&
                        retainedIndices.loadAs<uint8_t>({row, compressedBase + 1}) == 3,
                    "Structured sparsity retained-index output mismatch.");
        }
    }

    Tensor metadata(ScalarType::UInt8, Shape{2, 1});
    encodeTwoOfFourMetadataInto(retainedIndices, metadata, 1);
    require(metadata.loadAs<uint8_t>({0, 0}) == 0xdd && metadata.loadAs<uint8_t>({1, 0}) == 0xdd,
            "Two-of-four metadata encoding mismatch.");

    Tensor fusedPruned(ScalarType::Float32, Shape{2, 8});
    Tensor fusedCompressed(ScalarType::Float32, Shape{2, 4});
    Tensor fusedMetadata(ScalarType::UInt8, Shape{2, 1});
    const StructuredSparseTensor fusedOutputs{
        .pruned = fusedPruned,
        .compressed = fusedCompressed,
        .twoOfFourMetadata = fusedMetadata,
    };
    applyStructuredSparsityInto(input, fusedOutputs, pattern, {.firstSlice = 0, .sliceCount = 1});
    applyStructuredSparsityInto(input, fusedOutputs, pattern, {.firstSlice = 1, .sliceCount = 1});
    require(compare(fusedMetadata, metadata).passed(),
            "Fused structured sparsity metadata mismatch.");

    Tensor inPlace =
        Tensor::copyNativeValues<float>(Shape{8}, std::array<float, 8>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor inPlaceCompressed(ScalarType::Float32, Shape{4});
    Tensor inPlaceIndices(ScalarType::UInt8, Shape{4});
    pattern.axis = 0;
    pattern.fixedPositions = {0, 2};
    applyStructuredSparsityInto(
        inPlace,
        {.pruned = inPlace, .compressed = inPlaceCompressed, .retainedIndices = inPlaceIndices},
        pattern);
    require(inPlace.loadAs<float>({0}) == 1 && inPlace.loadAs<float>({1}) == 0 &&
                inPlace.loadAs<float>({2}) == 3 && inPlace.loadAs<float>({3}) == 0,
            "In-place structured sparsity mismatch.");

    const std::array<float, 12> ownedValues{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    StructuredSparsityPattern ownedPattern;
    ownedPattern.axis = 0;
    ownedPattern.fixedPositions = {0, 2};
    const Tensor ownedInput =
        Tensor::copyNativeValues<float>(Shape{12}, std::span<const float>(ownedValues));
    const StructuredSparseTensor owned = applyStructuredSparsity(
        ownedInput, ownedPattern, {.retainedIndices = true, .twoOfFourMetadata = true});
    require(
        owned.pruned.layout() == Layout::contiguousLastDimensionFastest(Shape{12}) &&
            owned.compressed.layout() == Layout::contiguousLastDimensionFastest(Shape{6}) &&
            owned.retainedIndices &&
            owned.retainedIndices->layout() == Layout::contiguousLastDimensionFastest(Shape{6}) &&
            owned.twoOfFourMetadata &&
            owned.twoOfFourMetadata->layout() == Layout::contiguousLastDimensionFastest(Shape{2}),
        "Owning structured-sparsity result contract mismatch.");
    require(owned.twoOfFourMetadata->loadAs<uint8_t>({0}) == 0x88 &&
                owned.twoOfFourMetadata->loadAs<uint8_t>({1}) == 0x08,
            "Owning structured-sparsity metadata retained unexpected bits.");

    StructuredSparsityPattern invalidMetadataPattern = ownedPattern;
    invalidMetadataPattern.retainedElements = 1;
    invalidMetadataPattern.fixedPositions = {0};
    bool rejectedBeforeAllocation = false;
    try {
        (void)applyStructuredSparsity(ownedInput, invalidMetadataPattern,
                                      {.retainedIndices = false, .twoOfFourMetadata = true});
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation,
            "Owning structured sparsity accepted an invalid metadata policy.");

    const Tensor ownedMetadata = encodeTwoOfFourMetadata(*owned.retainedIndices, 0);
    require(ownedMetadata.shape() == Shape{2} && ownedMetadata.loadAs<uint8_t>({0}) == 0x88 &&
                ownedMetadata.loadAs<uint8_t>({1}) == 0x08,
            "Owning two-of-four metadata result contract mismatch.");

    const std::array<uint8_t, 2> invalidRetainedValues{2, 1};
    const Tensor invalidRetained = Tensor::copyNativeValues<uint8_t>(
        Shape{2}, std::span<const uint8_t>(invalidRetainedValues));
    rejectedBeforeAllocation = false;
    try {
        (void)encodeTwoOfFourMetadata(invalidRetained, 0);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation,
            "Owning two-of-four metadata accepted invalid retained positions.");

    StructuredSparsityPattern widePattern;
    widePattern.axis = 0;
    widePattern.groupSize = 257;
    widePattern.retainedElements = 1;
    widePattern.fixedPositions = {0};
    const StructuredSparseTensor wide =
        applyStructuredSparsity(Tensor(ScalarType::Float32, Shape{257}), widePattern);
    require(wide.compressed.shape() == Shape{1} && !wide.retainedIndices,
            "Structured sparsity imposed the UInt8 index limit without an index output.");

    Tensor overlappingPrunedA(ScalarType::Float32, Layout(Shape{2, 8}, {1, 1}));
    Tensor overlappingPrunedB(ScalarType::Float32, Layout(Shape{2, 8}, {1, 1}));
    Tensor independentCompressedA(ScalarType::Float32, Shape{2, 4});
    Tensor independentCompressedB(ScalarType::Float32, Shape{2, 4});
    StructuredSparsityPattern overlappingPattern;
    overlappingPattern.axis = 1;
    overlappingPattern.fixedPositions = {1, 3};
    applyStructuredSparsityInto(
        input, {.pruned = overlappingPrunedA, .compressed = independentCompressedA},
        overlappingPattern);
    applyStructuredSparsityInto(
        input, {.pruned = overlappingPrunedB, .compressed = independentCompressedB},
        overlappingPattern);
    require(std::ranges::equal(overlappingPrunedA.rawEncodedBackingStorage(),
                               overlappingPrunedB.rawEncodedBackingStorage()) &&
                compare(independentCompressedA, independentCompressedB).passed(),
            "Structured sparsity was nondeterministic for overlapping nonzero strides.");
}

void testIndexedGeneration() {
    using namespace roc::host_numerics;

    Tensor serial(ScalarType::Float32, Shape{2, 3});
    generate(serial, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(serial.loadAs<float>({0, 0}) == 0 && serial.loadAs<float>({1, 0}) == 1 &&
                serial.loadAs<float>({0, 1}) == 2 && serial.loadAs<float>({1, 2}) == 5,
            "First-dimension-fast serial generation mismatch.");

    Tensor complex(ScalarType::ComplexFloat32, Shape{2, 2});
    generate(complex,
             GenerationRecipe::cartesian(GenerationRecipe::sine(), GenerationRecipe::cosine()));
    const std::complex<float> value = complex.loadAs<std::complex<float>>({1, 0});
    require(std::abs(value.real() - std::sin(1.0f)) < 1e-6f &&
                std::abs(value.imag() - std::cos(1.0f)) < 1e-6f,
            "Complex trigonometric generation mismatch.");

    Tensor candidates(ScalarType::Float32, Shape{8});
    const std::vector<double> candidateValues{-6.0, -1.5, 0.0, 4.0};
    const std::array<double, 8> expectedCandidateValues{0.0, -1.5, 0.0, -6.0, 0.0, -6.0, -1.5, 0.0};
    constexpr uint64_t candidateSeed = 37;
    generate(candidates,
             GenerationRecipe::realOnly(GenerationRecipe::choice({.values = candidateValues}),
                                        {.seed = candidateSeed}));
    for (size_t index = 0; index < candidates.elementCount(); ++index) {
        require(candidates.loadAs<float>({index}) == expectedCandidateValues[index],
                "Choice generation mismatch.");
    }

    Tensor point(ScalarType::Float32, Shape{2, 3, 2});
    generateAt(point, 3, GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 9.0})));
    require(point.loadAs<float>({1, 1, 0}) == 9.0f && point.loadAs<float>({0, 1, 1}) == 0.0f,
            "First-dimension-fast point generation mismatch.");

    generateAt(point, 3,
               GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 7.0}),
                                          {.indexOrder = IndexOrder::LastDimensionFastest}));
    require(point.loadAs<float>({0, 1, 1}) == 7.0f,
            "Last-dimension-fast point generation mismatch.");

    for (const IndexOrder order :
         {IndexOrder::FirstDimensionFastest, IndexOrder::LastDimensionFastest}) {
        Tensor whole(ScalarType::Float4E2M1, Shape{2, 3, 2});
        Tensor elementwise(ScalarType::Float4E2M1, Shape{2, 3, 2});
        const GenerationRecipe exactRecipe =
            GenerationRecipe::realOnly(GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}),
                                       {.seed = 0x12345678, .indexOrder = order});
        generate(whole, exactRecipe);
        for (size_t index = 0; index < whole.elementCount(); ++index)
            generateAt(elementwise, index, exactRecipe);
        require(std::equal(whole.rawEncodedBackingStorage().begin(),
                           whole.rawEncodedBackingStorage().end(),
                           elementwise.rawEncodedBackingStorage().begin(),
                           elementwise.rawEncodedBackingStorage().end()),
                "Whole-tensor and elementwise generation encodings differ.");
    }

    Tensor affine(ScalarType::Float32, Shape{2, 3, 2});
    generate(affine,
             GenerationRecipe::realOnly(
                 GenerationRecipe::affineIndexRemainder(
                     {.dimensionCoefficients = {1, -1, 2}, .offset = -2, .positiveDivisor = 5})
                     .withAffineValueMapping({.offset = 1.0})));
    require(affine.loadAs<float>({0, 0, 0}) == -1.0f && affine.loadAs<float>({1, 2, 1}) == 0.0f,
            "Affine-index remainder generation mismatch.");

#ifdef HOST_NUMERICS_TEST_OPENMP
    const int originalDynamic = omp_get_dynamic();
    const int originalThreadCount = omp_get_max_threads();
    omp_set_dynamic(0);

    const Layout paddedLayout(Shape{128, 96}, {1, 137});
    Tensor oneThread(ScalarType::Float32, paddedLayout);
    Tensor fourThreads(ScalarType::Float32, paddedLayout);
    const GenerationRecipe parallelRecipe =
        GenerationRecipe::realOnly(GenerationRecipe::uniformReal({.lower = -3.0, .upper = 7.0}),
                                   {.seed = 0x1020304050607080ULL});
    omp_set_num_threads(1);
    generate(oneThread, parallelRecipe);
    omp_set_num_threads(4);
    generate(fourThreads, parallelRecipe);
    require(std::equal(oneThread.rawEncodedBackingStorage().begin(),
                       oneThread.rawEncodedBackingStorage().end(),
                       fourThreads.rawEncodedBackingStorage().begin(),
                       fourThreads.rawEncodedBackingStorage().end()),
            "Ordinary generation changed with OpenMP thread count.");

    Tensor aliased(ScalarType::Float32, Layout(Shape{8192}, {0}));
    generate(aliased, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(aliased.loadAs<float>({0}) == 8191.0f,
            "Aliased generation did not preserve deterministic traversal order.");

    omp_set_num_threads(originalThreadCount);
    omp_set_dynamic(originalDynamic);
#endif
}

void testLinearCombination() {
    using namespace roc::host_numerics;

    const Shape shape{2, 2, 2};
    Tensor x(ScalarType::Float16, Layout(shape, {1, 3, 6}));
    Tensor y(ScalarType::BFloat16, Layout(shape, {3, 1, 6}));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 8}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                x.storeFrom(indices, static_cast<float>(1 + row + 2 * column + batch));
                y.storeFrom(indices, static_cast<float>(2 - 2 * row + column + batch));
            }
        }
    }

    LinearCombinationOptions options(ScalarType::Float32);
    require(
        options.alpha.type() == ScalarType::Float32 && options.beta.type() == ScalarType::Float32,
        "Linear-combination defaults do not use the requested accumulator type.");
    options.alpha = 2.0;
    options.beta = -0.5;
    linearCombinationInto(x, y, output, options);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                const float expected =
                    2.0f * x.loadAs<float>(indices) - 0.5f * y.loadAs<float>(indices);
                require(output.loadAs<float>(indices) == expected,
                        "Linear-combination value mismatch.");
            }
        }
    }

    Tensor yOnlyOutput(ScalarType::Float32, shape);
    LinearCombinationOptions yOnlyOptions(ScalarType::Float32);
    yOnlyOptions.beta = 3.0;
    linearCombinationInto(std::nullopt, y, yOnlyOutput, yOnlyOptions);
    require(yOnlyOutput.loadAs<float>({1, 0, 1}) == 3.0f * y.loadAs<float>({1, 0, 1}),
            "Linear-combination optional-input mismatch.");

    Tensor coefficientTensor(ScalarType::Float32, Shape{});
    coefficientTensor.storeFrom({}, 3.0f);
    LinearCombinationOptions tensorCoefficientOptions(ScalarType::Float32);
    tensorCoefficientOptions.alpha = coefficientTensor;
    coefficientTensor.storeFrom({}, 7.0f);
    const Tensor tensorCoefficientOutput =
        linearCombination(x, std::nullopt, ScalarType::Float32, tensorCoefficientOptions);
    require(tensorCoefficientOutput.loadAs<float>({1, 0, 1}) == 3.0f * x.loadAs<float>({1, 0, 1}),
            "Rank-zero Tensor coefficient did not retain snapshot semantics.");

    const std::array<float, 2> columnValues{1.0f, 2.0f};
    const std::array<float, 3> rowValues{10.0f, 20.0f, 30.0f};
    const Tensor column = Tensor::copyNativeValues<float>(Shape{2, 1}, columnValues);
    const Tensor row = Tensor::copyNativeValues<float>(Shape{1, 3}, rowValues);
    LinearCombinationOptions broadcastOptions(ScalarType::Float32);
    broadcastOptions.alpha = 2.0f;
    broadcastOptions.beta = -1.0f;
    const Tensor broadcastOutput =
        linearCombination(column, row, ScalarType::Float32, broadcastOptions);
    require(broadcastOutput.shape() == Shape{2, 3} &&
                broadcastOutput.loadAs<float>({0, 0}) == -8.0f &&
                broadcastOutput.loadAs<float>({0, 2}) == -28.0f &&
                broadcastOutput.loadAs<float>({1, 0}) == -6.0f &&
                broadcastOutput.loadAs<float>({1, 2}) == -26.0f,
            "Linear combination did not apply NumPy-style broadcasting.");

    const Tensor emptyBroadcast = linearCombination(
        Tensor(ScalarType::Float32, Shape{0, 3}),
        Tensor::copyNativeValues<float>(Shape{1, 3}, rowValues), ScalarType::Float32);
    require(emptyBroadcast.shape() == Shape{0, 3} && emptyBroadcast.elementCount() == 0,
            "Linear combination did not preserve a broadcast zero extent.");

    bool rejectedIncompatibleBroadcast = false;
    try {
        (void)linearCombination(Tensor(ScalarType::Float32, Shape{2}),
                                Tensor(ScalarType::Float32, Shape{3}), ScalarType::Float32);
    } catch (const std::invalid_argument&) {
        rejectedIncompatibleBroadcast = true;
    }
    require(rejectedIncompatibleBroadcast,
            "Linear combination accepted incompatible broadcast shapes.");

    const std::array<std::complex<float>, 1> complexXValues{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexYValues{std::complex<float>(3, -1)};
    Tensor complexX = Tensor::copyNativeValues<std::complex<float>>(Shape{1}, complexXValues);
    Tensor complexY = Tensor::copyNativeValues<std::complex<float>>(Shape{1}, complexYValues);
    Tensor complexOutput(ScalarType::ComplexFloat32, Shape{1});
    LinearCombinationOptions complexOptions(ScalarType::ComplexFloat32);
    complexOptions.alpha = std::complex<double>(0.5, 1.0);
    complexOptions.beta = -2.0;
    linearCombinationInto(complexX, complexY, complexOutput, complexOptions);
    require(complexOutput.loadAs<std::complex<float>>({0}) == std::complex<float>(-7.5f, 4.0f),
            "Complex linear combination mismatch.");

    const Tensor owned = linearCombination(x, y, ScalarType::Float32, options);
    require(owned.layout() == Layout::contiguousLastDimensionFastest(shape) &&
                owned.type() == ScalarType::Float32 &&
                owned.loadAs<float>({1, 0, 1}) ==
                    2.0f * x.loadAs<float>({1, 0, 1}) - 0.5f * y.loadAs<float>({1, 0, 1}),
            "Owning linear combination result contract mismatch.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)linearCombination(std::nullopt, std::nullopt, ScalarType::Float32);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning linear combination accepted an invalid problem.");

    LinearCombinationOptions invalidCoefficient(ScalarType::Float32);
    invalidCoefficient.alpha = std::complex<double>(1.0, 1.0);
    rejectedBeforeAllocation = false;
    try {
        (void)linearCombination(x, std::nullopt, ScalarType::Float32, invalidCoefficient);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning linear combination accepted invalid coefficients.");
}

void testReferenceSoftmax() {
    using namespace roc::host_numerics;

    const Shape shape{2, 3, 2};
    Tensor input(ScalarType::Float16, Layout(shape, {1, 3, 10}));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                input.storeFrom({row, column, batch}, static_cast<float>(static_cast<int>(column) +
                                                                         2 * static_cast<int>(row) -
                                                                         static_cast<int>(batch)));
            }
        }
    }

    referenceSoftmaxInto(input, output, 1, ScalarType::Float32);

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            float sum = 0;
            for (size_t column = 0; column < 3; ++column)
                sum += output.loadAs<float>({row, column, batch});
            require(std::abs(sum - 1.0f) < 1e-6f, "Reference softmax slice does not sum to one.");
        }
    }
    require(output.loadAs<float>({0, 0, 0}) < output.loadAs<float>({0, 1, 0}) &&
                output.loadAs<float>({0, 1, 0}) < output.loadAs<float>({0, 2, 0}),
            "Reference softmax ordering mismatch.");

    const Tensor owned = referenceSoftmax(input, 1, ScalarType::Float32, ScalarType::Float32);
    require(owned.layout() == Layout::contiguousLastDimensionFastest(shape) &&
                owned.type() == ScalarType::Float32,
            "Owning reference softmax result contract mismatch.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceSoftmax(input, shape.rank(), ScalarType::Float32, ScalarType::Float32);
    } catch (const std::out_of_range&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference softmax accepted an invalid problem.");
}

void testReferenceLayerNorm() {
    using namespace roc::host_numerics;

    const Shape shape{2, 3, 2};
    Tensor input(ScalarType::Float32, Layout(shape, {1, 3, 10}));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 3; ++column)
                input.storeFrom({row, column, batch},
                                static_cast<float>(1 + column + 3 * row + 6 * batch));
        }
    }
    const std::array<float, 3> gammaValues{1.0f, 2.0f, 0.5f};
    const std::array<float, 3> betaValues{0.25f, -0.5f, 1.0f};
    const Tensor gamma = Tensor::copyValuesWithConversion(ScalarType::Float16, Shape{3},
                                                          std::span<const float>(gammaValues));
    const Tensor beta = Tensor::copyValuesWithConversion(ScalarType::BFloat16, Shape{3},
                                                         std::span<const float>(betaValues));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    Tensor mean(ScalarType::Float32, Layout(Shape{2, 2}, {3, 1}));
    Tensor inverseVariance(ScalarType::Float32, Layout(Shape{2, 2}, {1, 3}));

    LayerNormOptions options;
    options.axis = 1;
    options.gamma = gamma;
    options.beta = beta;
    referenceLayerNormInto(
        input, {.output = output, .mean = mean, .inverseVariance = inverseVariance}, options);
    require(mean.loadAs<float>({0, 0}) == 2.0f, "Reference LayerNorm mean mismatch.");
    require(std::abs(inverseVariance.loadAs<float>({0, 0}) -
                     1.0f / std::sqrt(2.0f / 3.0f + 1e-5f)) < 1e-6f,
            "Reference LayerNorm inverse variance mismatch.");
    require(output.loadAs<float>({0, 1, 0}) == -0.5f,
            "Reference LayerNorm affine output mismatch.");

    const LayerNormOutputs owned = referenceLayerNorm(input,
                                                      {.output = ScalarType::Float32,
                                                       .mean = ScalarType::Float32,
                                                       .inverseVariance = ScalarType::Float32},
                                                      options);
    require(owned.output.layout() == Layout::contiguousLastDimensionFastest(shape) && owned.mean &&
                owned.mean->layout() == Layout::contiguousLastDimensionFastest(Shape{2, 2}) &&
                owned.inverseVariance &&
                owned.inverseVariance->layout() ==
                    Layout::contiguousLastDimensionFastest(Shape{2, 2}) &&
                owned.output.loadAs<float>({0, 1, 0}) == -0.5f,
            "Owning reference LayerNorm result contract mismatch.");

    LayerNormOptions invalidOptions = options;
    invalidOptions.epsilon = std::numeric_limits<double>::quiet_NaN();
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceLayerNorm(input, {}, invalidOptions);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference LayerNorm accepted invalid epsilon.");

    const LayerNormOutputs outputOnly = referenceLayerNorm(input, {}, options);
    require(!outputOnly.mean && !outputOnly.inverseVariance,
            "Owning reference LayerNorm created unrequested statistics.");

    const std::array<float, 3> rankOneValues{1.0f, 2.0f, 3.0f};
    const LayerNormOutputs rankOneResult = referenceLayerNorm(
        Tensor::copyNativeValues<float>(Shape{3}, std::span<const float>(rankOneValues)),
        {.output = ScalarType::Float32, .mean = ScalarType::Float32});
    require(rankOneResult.mean && rankOneResult.mean->shape() == Shape{} &&
                !rankOneResult.inverseVariance,
            "Owning reference LayerNorm did not preserve a requested rank-zero statistic.");
}

void testReferenceOperationAliasing() {
    using namespace roc::host_numerics;

    Tensor linearCombinationInPlace =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    const Tensor linearCombinationY =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{3.0f, 4.0f});
    LinearCombinationOptions linearCombinationOptions(ScalarType::Float32);
    linearCombinationOptions.alpha = 2.0;
    linearCombinationOptions.beta = 1.0;
    linearCombinationInto(linearCombinationInPlace, linearCombinationY, linearCombinationInPlace,
                          linearCombinationOptions);
    require(linearCombinationInPlace.loadAs<float>({0}) == 5.0f &&
                linearCombinationInPlace.loadAs<float>({1}) == 8.0f,
            "Linear-combination rejected or corrupted an exact in-place mapping.");

    Tensor linearCombinationOverlap =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{11.0f, 22.0f});
    Tensor linearCombinationReversedOutput =
        linearCombinationOverlap.shareStorageWithLayout(Layout(Shape{2}, {-1}, 1));
    const std::vector<std::byte> linearCombinationOverlapBefore =
        copyRawEncodedBackingStorage(linearCombinationOverlap);
    requireInvalidArgument(
        [&] {
            linearCombinationInto(linearCombinationOverlap, std::nullopt,
                                  linearCombinationReversedOutput);
        },
        "Linear-combination accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        linearCombinationOverlap, linearCombinationOverlapBefore,
        "Linear-combination modified destination storage before rejecting overlap.");

    const Tensor distinctLinearCombinationInput =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    Tensor selfCollidingLinearCombinationOutput(ScalarType::Float32, Layout(Shape{2}, {0}));
    selfCollidingLinearCombinationOutput.storeFrom({0}, 17.0f);
    const std::vector<std::byte> selfCollidingLinearCombinationOutputBefore =
        copyRawEncodedBackingStorage(selfCollidingLinearCombinationOutput);
    requireInvalidArgument(
        [&] {
            linearCombinationInto(distinctLinearCombinationInput, std::nullopt,
                                  selfCollidingLinearCombinationOutput);
        },
        "Linear-combination accepted a self-colliding destination.");
    requireRawEncodedBackingStorageEquals(
        selfCollidingLinearCombinationOutput, selfCollidingLinearCombinationOutputBefore,
        "Linear-combination modified a self-colliding destination before rejecting it.");

    Tensor softmaxInPlace =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    referenceSoftmaxInto(softmaxInPlace, softmaxInPlace, 0);
    require(std::abs(softmaxInPlace.loadAs<float>({0}) + softmaxInPlace.loadAs<float>({1}) - 1.0f) <
                    1e-6f &&
                softmaxInPlace.loadAs<float>({0}) < softmaxInPlace.loadAs<float>({1}),
            "Reference softmax rejected or corrupted an exact in-place mapping.");

    Tensor softmaxOverlap =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{1.0f, 2.0f});
    Tensor softmaxReversedOutput = softmaxOverlap.shareStorageWithLayout(Layout(Shape{2}, {-1}, 1));
    const std::vector<std::byte> softmaxOverlapBefore =
        copyRawEncodedBackingStorage(softmaxOverlap);
    requireInvalidArgument([&] { referenceSoftmaxInto(softmaxOverlap, softmaxReversedOutput, 0); },
                           "Reference softmax accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        softmaxOverlap, softmaxOverlapBefore,
        "Reference softmax modified destination storage before rejecting overlap.");

    Tensor layerNormInPlace =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, 3.0f});
    LayerNormOptions layerNormOptions;
    layerNormOptions.axis = 1;
    referenceLayerNormInto(layerNormInPlace, {.output = layerNormInPlace}, layerNormOptions);
    require(layerNormInPlace.loadAs<float>({0, 0}) < 0.0f &&
                layerNormInPlace.loadAs<float>({0, 1}) > 0.0f &&
                std::abs(layerNormInPlace.loadAs<float>({0, 0}) +
                         layerNormInPlace.loadAs<float>({0, 1})) < 1e-6f,
            "Reference LayerNorm rejected or corrupted an exact in-place mapping.");

    Tensor layerNormOverlap =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, 3.0f});
    Tensor layerNormReversedOutput =
        layerNormOverlap.shareStorageWithLayout(Layout(Shape{1, 2}, {2, -1}, 1));
    const std::vector<std::byte> layerNormOverlapBefore =
        copyRawEncodedBackingStorage(layerNormOverlap);
    requireInvalidArgument(
        [&] {
            referenceLayerNormInto(layerNormOverlap, {.output = layerNormReversedOutput},
                                   layerNormOptions);
        },
        "Reference LayerNorm accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        layerNormOverlap, layerNormOverlapBefore,
        "Reference LayerNorm modified destination storage before rejecting overlap.");

    Tensor reductionInPlace =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{5.0f, 7.0f});
    referenceSumInto(reductionInPlace, reductionInPlace, {}, ScalarType::Float32);
    require(
        reductionInPlace.loadAs<float>({0}) == 5.0f && reductionInPlace.loadAs<float>({1}) == 7.0f,
        "Reference reduction rejected or corrupted an exact pointwise mapping.");

    Tensor reductionOverlap =
        Tensor::copyNativeValues<float>(Shape{2}, std::array<float, 2>{5.0f, 7.0f});
    Tensor reductionReversedOutput =
        reductionOverlap.shareStorageWithLayout(Layout(Shape{2}, {-1}, 1));
    const std::vector<std::byte> reductionOverlapBefore =
        copyRawEncodedBackingStorage(reductionOverlap);
    requireInvalidArgument(
        [&] {
            referenceSumInto(reductionOverlap, reductionReversedOutput, {}, ScalarType::Float32);
        },
        "Reference reduction accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        reductionOverlap, reductionOverlapBefore,
        "Reference reduction modified destination storage before rejecting overlap.");

    Tensor epilogueInPlace =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, -2.0f});
    EpilogueOptions epilogueInPlaceOptions;
    epilogueInPlaceOptions.outputScale = 2.0;
    referenceEpilogueInto(epilogueInPlace, {.output = epilogueInPlace}, epilogueInPlaceOptions);
    require(epilogueInPlace.loadAs<float>({0, 0}) == 2.0f &&
                epilogueInPlace.loadAs<float>({0, 1}) == -4.0f,
            "Reference epilogue rejected or corrupted an exact in-place mapping.");

    Tensor epilogueOverlap =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{1.0f, -2.0f});
    Tensor epilogueReversedOutput =
        epilogueOverlap.shareStorageWithLayout(Layout(Shape{1, 2}, {2, -1}, 1));
    const std::vector<std::byte> epilogueOverlapBefore =
        copyRawEncodedBackingStorage(epilogueOverlap);
    requireInvalidArgument(
        [&] { referenceEpilogueInto(epilogueOverlap, {.output = epilogueReversedOutput}); },
        "Reference epilogue accepted differently mapped overlapping storage.");
    requireRawEncodedBackingStorageEquals(
        epilogueOverlap, epilogueOverlapBefore,
        "Reference epilogue modified destination storage before rejecting overlap.");

    const Tensor epilogueInput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{3.0f, 4.0f});
    Tensor overlappingEpilogueOutputs =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{19.0f, 23.0f});
    const std::vector<std::byte> overlappingEpilogueOutputsBefore =
        copyRawEncodedBackingStorage(overlappingEpilogueOutputs);
    requireInvalidArgument(
        [&] {
            referenceEpilogueInto(epilogueInput, {.output = overlappingEpilogueOutputs,
                                                  .rawOutput = overlappingEpilogueOutputs});
        },
        "Reference epilogue accepted overlapping result tensors.");
    requireRawEncodedBackingStorageEquals(
        overlappingEpilogueOutputs, overlappingEpilogueOutputsBefore,
        "Reference epilogue modified result storage before rejecting overlapping outputs.");
}

void testActivations() {
    using namespace roc::host_numerics;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{1};
    const std::array<float, 1> c{0};
    Tensor d(ScalarType::Float32, Shape{1, 1});

    GemmTestCase problem(
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(b)),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(c)),
        d, ScalarType::Float32);

    problem.activation = Activation::Gelu;
    referenceGemm(problem);
    require(std::abs(d.loadAs<float>({0, 0}) - 1.9545977f) < 1e-6f, "GELU result mismatch.");

    problem.activation = Activation::Silu;
    problem.activationParameter0 = 1;
    referenceGemm(problem);
    require(std::abs(d.loadAs<float>({0, 0}) - 1.7615942f) < 1e-6f, "SiLU result mismatch.");

    problem.activation = Activation::Clamp;
    problem.activationParameter0 = -1;
    problem.activationParameter1 = 1;
    referenceGemm(problem);
    require(d.loadAs<float>({0, 0}) == 1, "Clamp result mismatch.");
}

void testStridedAndOffsetViews() {
    using namespace roc::host_numerics;

    // Logical A and B are the same matrices as testReferenceGemm, but both
    // are stored transposed with padded leading dimensions. C and D use
    // different padding, and D begins at an adjusted base pointer.
    const std::array<float, 8> a{1, 2, 3, -1, 4, 5, 6, -1};
    const std::array<float, 9> b{7, 8, -1, 9, 10, -1, 11, 12, -1};
    const std::array<float, 8> c{1, 1, -1, -1, 1, 1, -1, -1};
    std::array<float, 12> initialStorage;
    initialStorage.fill(-99);
    std::vector<std::byte> dStorage(sizeof(initialStorage));
    std::memcpy(dStorage.data(), initialStorage.data(), dStorage.size());

    const Layout outputLayout(Shape{2, 2}, {1, 5}, 1);
    GemmTestCase problem(
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {4, 1}), std::span<const float>(a)),
        Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {3, 1}), std::span<const float>(b)),
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 4}), std::span<const float>(c)),
        Tensor::takeOwnershipOfEncodedBackingStorage(ScalarType::Float32, outputLayout,
                                                     std::move(dStorage)),
        ScalarType::Float32);
    Tensor d = problem.d;
    problem.alpha = 2.0;
    problem.beta = 3.0;

    referenceGemm(problem);

    std::array<float, 12> expected;
    expected.fill(-99);
    expected[1] = 2 * 58 + 3;
    expected[2] = 2 * 139 + 3;
    expected[6] = 2 * 64 + 3;
    expected[7] = 2 * 154 + 3;
    const auto comparison = compare(
        d, Tensor::copyNativeStorage<float>(outputLayout, std::span<const float>(expected)));
    require(comparison.passed(), "Strided GEMM matrix comparison failed.");
    const auto storageValue = [&d](size_t index) {
        float value;
        std::memcpy(&value, d.rawEncodedBackingStorage().data() + index * sizeof(float),
                    sizeof(value));
        return value;
    };
    require(storageValue(0) == -99 && storageValue(3) == -99 && storageValue(11) == -99,
            "Strided GEMM modified padding.");
}

void testGenerationAndComparison() {
    using namespace roc::host_numerics;

    const GenerationRecipe binaryGeneration =
        GenerationRecipe::realOnly(GenerationRecipe::choice({.values = {-1.0, 1.0}}), {.seed = 42});
    Tensor a(ScalarType::Float32, Shape{32});
    Tensor b(ScalarType::Float32, Shape{32});
    generate(a, binaryGeneration);
    generate(b, binaryGeneration);
    require(compare(b, a).passed(), "Random generation is not repeatable for equal seeds.");

    b.storeFrom({7}, b.loadAs<float>({7}) + 1.0f);
    ComparisonOptions mismatchOptions;
    mismatchOptions.absoluteTolerance = 0.0;
    mismatchOptions.relativeTolerance = 0.0;
    mismatchOptions.maxReportedMismatches = 4;
    const auto result = compare(b, a, mismatchOptions);
    require(result.mismatches == 1, "Comparison did not count one mismatch.");
    require(result.reportedMismatches.size() == 1, "Comparison did not report one mismatch.");
    require(result.reportedMismatches[0].index == 7,
            "Comparison reported the wrong mismatch index.");

    const std::array<double, 2> nonFiniteA{
        std::numeric_limits<double>::infinity(),
        1.0,
    };
    const std::array<double, 2> nonFiniteB{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    ComparisonOptions nonFiniteOptions;
    nonFiniteOptions.relativeTolerance = 1.0;
    const auto nonFiniteResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(nonFiniteA)),
                Tensor::copyNativeStorage(std::span<const double>(nonFiniteB)), nonFiniteOptions);
    require(nonFiniteResult.mismatches == 1,
            "Comparison did not distinguish finite and infinite values.");

    std::array<int, 8> generatedStorage;
    generatedStorage.fill(-1);
    std::vector<std::byte> generatedBytes(sizeof(generatedStorage));
    std::memcpy(generatedBytes.data(), generatedStorage.data(), generatedBytes.size());
    Tensor generated = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Int32, Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1),
        std::move(generatedBytes));
    generate(generated,
             [](std::span<const size_t> indices) { return 10 * indices[1] + indices[0]; });
    require(generated.loadAs<int>({0, 0}) == 0 && generated.loadAs<int>({1, 0}) == 1 &&
                generated.loadAs<int>({0, 1}) == 10 && generated.loadAs<int>({1, 1}) == 11,
            "Matrix generation produced incorrect logical values.");
    const auto generatedStorageValue = [&generated](size_t index) {
        int value;
        std::memcpy(&value, generated.rawEncodedBackingStorage().data() + index * sizeof(int),
                    sizeof(value));
        return value;
    };
    require(generatedStorageValue(0) == -1 && generatedStorageValue(3) == -1 &&
                generatedStorageValue(7) == -1,
            "Matrix generation modified padding.");

    Tensor runtimeExpected(ScalarType::Float32, Shape{2, 3});
    const GenerationRecipe runtimeGeneration = GenerationRecipe::realOnly(
        GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}), {.seed = 7});
    generate(runtimeExpected, runtimeGeneration);
    Tensor runtimeObserved = runtimeExpected.deepCopy();
    runtimeObserved.storeFrom({1, 2}, runtimeExpected.loadAs<float>({1, 2}) + 1.0f);
    ComparisonOptions runtimeComparisonOptions;
    runtimeComparisonOptions.absoluteTolerance = 0.0;
    runtimeComparisonOptions.maxReportedMismatches = 2;
    const auto runtimeComparison =
        compare(runtimeObserved, runtimeExpected, runtimeComparisonOptions);
    require(runtimeComparison.compared == 6 && runtimeComparison.mismatches == 1 &&
                runtimeComparison.reportedMismatches[0].index == 5,
            "Runtime tensor generation/comparison mismatch.");
}

void testComparisonProgram() {
    using namespace roc::host_numerics;

    const std::array<float, 0> emptyStorage{};
    const Layout emptyLayout(Shape{0, 3}, {1, 1});
    ComparisonOptions emptyOptions;
    emptyOptions.computePointwiseStatistics = false;
    emptyOptions.computeFrobenius = false;
    emptyOptions.maxReportedMismatches = 0;
    emptyOptions.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    const auto compareEmpty = [&](const ComparisonOptions& options) {
        return compare(Tensor::copyNativeStorage(emptyLayout, std::span<const float>(emptyStorage)),
                       Tensor::copyNativeStorage(emptyLayout, std::span<const float>(emptyStorage)),
                       options);
    };
    const ComparisonReport emptyResult = compareEmpty(emptyOptions);
    require(emptyResult.compared == 0 && emptyResult.pointwiseEvaluated &&
                emptyResult.pointwisePassed && !emptyResult.frobeniusEvaluated &&
                !emptyResult.ulpEvaluated,
            "Runtime fast comparison rejected an empty tensor.");

    const auto requireInvalidEmptyOptions = [&](const ComparisonOptions& options,
                                                const char* message) {
        bool rejected = false;
        try {
            (void)compareEmpty(options);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, message);
    };
    ComparisonOptions missingUlpType;
    missingUlpType.computeUlp = true;
    requireInvalidEmptyOptions(missingUlpType,
                               "Empty comparison accepted ULP evidence without a scalar type.");
    ComparisonOptions invalidUlpType = missingUlpType;
    invalidUlpType.ulpType = ScalarType::Count;
    requireInvalidEmptyOptions(invalidUlpType,
                               "Empty comparison accepted ScalarType::Count as a ULP type.");
    ComparisonOptions missingUlpEvidence;
    missingUlpEvidence.maximumUlpTolerance = 0.0;
    requireInvalidEmptyOptions(missingUlpEvidence,
                               "Empty comparison accepted a ULP criterion without ULP evidence.");
    ComparisonOptions missingFrobeniusEvidence;
    missingFrobeniusEvidence.computeFrobenius = false;
    missingFrobeniusEvidence.relativeFrobeniusTolerance = 0.0;
    requireInvalidEmptyOptions(
        missingFrobeniusEvidence,
        "Empty comparison accepted a Frobenius criterion without Frobenius evidence.");
    bool rejectedZeroSelectionStride = false;
    try {
        (void)OutputSelection::strided(0, 0);
    } catch (const std::invalid_argument&) {
        rejectedZeroSelectionStride = true;
    }
    require(rejectedZeroSelectionStride, "Output selection accepted a zero stride.");

    const std::array<double, 1> evidenceObserved{2.0};
    const std::array<double, 1> evidenceExpected{1.0};
    ComparisonOptions evidenceOnly;
    evidenceOnly.pointwise = false;
    evidenceOnly.computeUlp = true;
    evidenceOnly.ulpType = ScalarType::Float64;
    const ComparisonReport evidenceResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(evidenceObserved)),
                Tensor::copyNativeStorage(std::span<const double>(evidenceExpected)), evidenceOnly);
    require(!evidenceResult.pointwiseEvaluated && !evidenceResult.frobeniusEvaluated &&
                !evidenceResult.ulpEvaluated && evidenceResult.passed() &&
                evidenceResult.frobeniusDifference == 1.0 && evidenceResult.ulpCompared == 1,
            "Evidence-only comparison reported an evaluated criterion.");

    const std::array<double, 3> reversedStorage{1.0, 2.0, 3.0};
    const Layout reversedLayout(Shape{3}, {-1}, 2);
    ComparisonOptions reversedMetrics;
    reversedMetrics.pointwise = false;
    reversedMetrics.computePointwiseStatistics = false;
    reversedMetrics.computeFrobenius = true;
    reversedMetrics.maxReportedMismatches = 0;
    const auto reversedRuntimeResult =
        compare(Tensor::copyNativeStorage(reversedLayout, std::span<const double>(reversedStorage)),
                Tensor::copyNativeStorage(reversedLayout, std::span<const double>(reversedStorage)),
                reversedMetrics);
    require(reversedRuntimeResult.compared == 3 && reversedRuntimeResult.frobeniusDifference == 0.0,
            "Negative-stride comparison produced incorrect evidence.");

    const std::array<float, 8> expectedStorage{1.0f, 2.0f, -99.0f, 3.0f, 4.0f, -99.0f, 5.0f, 6.0f};
    auto observedStorage = expectedStorage;
    observedStorage[4] += 0.5f;
    observedStorage[6] += 1.0f;
    const Layout layout(Shape{2, 3}, {1, 3});

    ComparisonOptions selected;
    selected.selection = OutputSelection::strided(0, 2, std::numeric_limits<size_t>::max(),
                                                  IndexOrder::FirstDimensionFastest);
    selected.computeFrobenius = false;
    const auto selectedResult = compare(
        Tensor::copyNativeStorage(layout, std::span<const float>(observedStorage)),
        Tensor::copyNativeStorage(layout, std::span<const float>(expectedStorage)), selected);
    require(selectedResult.compared == 3 && selectedResult.mismatches == 1,
            "Selected comparison visited the wrong logical elements.");
    require(selectedResult.reportedMismatches[0].index == 4 &&
                selectedResult.reportedMismatches[0].coordinates == std::vector<size_t>({0, 2}) &&
                selectedResult.reportedMismatches[0].observedOffset == 6,
            "Selected comparison reported the wrong logical location.");
    selected.selection = OutputSelection::explicitIndices({4}, IndexOrder::FirstDimensionFastest);
    const auto explicitSelectedResult = compare(
        Tensor::copyNativeStorage(layout, std::span<const float>(observedStorage)),
        Tensor::copyNativeStorage(layout, std::span<const float>(expectedStorage)), selected);
    require(
        explicitSelectedResult.compared == 1 && explicitSelectedResult.mismatches == 1 &&
            explicitSelectedResult.reportedMismatches[0].coordinates == std::vector<size_t>({0, 2}),
        "Explicit comparison selection visited the wrong logical element.");
    ComparisonOptions paddedMetrics;
    paddedMetrics.pointwise = false;
    paddedMetrics.computePointwiseStatistics = false;
    paddedMetrics.computeFrobenius = true;
    paddedMetrics.maxReportedMismatches = 0;
    paddedMetrics.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    const auto paddedMetricResult = compare(
        Tensor::copyNativeStorage(layout, std::span<const float>(observedStorage)),
        Tensor::copyNativeStorage(layout, std::span<const float>(expectedStorage)), paddedMetrics);
    require(paddedMetricResult.compared == 6 && paddedMetricResult.frobeniusDifference > 0.0,
            "Regular strided comparison produced incorrect evidence.");

    const std::array<uint64_t, 1> wideIntegerObserved{uint64_t{1} << 53};
    const std::array<uint64_t, 1> wideIntegerExpected{(uint64_t{1} << 53) + 1};
    ComparisonOptions fastIntegerOptions;
    fastIntegerOptions.computePointwiseStatistics = false;
    fastIntegerOptions.computeFrobenius = false;
    fastIntegerOptions.maxReportedMismatches = 0;
    const auto fastIntegerComparison =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)),
                fastIntegerOptions);
    require(!fastIntegerComparison.passed() && fastIntegerComparison.mismatches == 1,
            "Fast comparison rounded distinct wide integers together.");

    fastIntegerOptions.maxReportedMismatches = 1;
    const auto reportedIntegerComparison =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)),
                fastIntegerOptions);
    require(!reportedIntegerComparison.passed() && reportedIntegerComparison.mismatches == 1 &&
                reportedIntegerComparison.reportedMismatches.size() == 1 &&
                !reportedIntegerComparison.reportedMismatches[0].matched,
            "Detailed comparison changed the wide-integer pointwise decision.");

    const auto runtimeIntegerComparison =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)));
    require(!runtimeIntegerComparison.passed() && runtimeIntegerComparison.mismatches == 1,
            "Runtime detailed comparison rounded distinct wide integers together.");

    ComparisonOptions subUnitIntegerTolerance;
    subUnitIntegerTolerance.absoluteTolerance = 0.5;
    subUnitIntegerTolerance.computePointwiseStatistics = false;
    subUnitIntegerTolerance.computeFrobenius = false;
    subUnitIntegerTolerance.maxReportedMismatches = 0;
    require(!compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                     Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)),
                     subUnitIntegerTolerance)
                 .passed(),
            "Runtime comparison lost a sub-unit tolerance at the uint64 precision boundary.");

    const std::array<uint64_t, 2> exactUnsignedObserved{
        uint64_t{1} << 53,
        std::numeric_limits<uint64_t>::max() - 1,
    };
    const std::array<uint64_t, 2> exactUnsignedExpected{
        (uint64_t{1} << 53) + 1,
        std::numeric_limits<uint64_t>::max(),
    };
    ComparisonOptions exactUnsignedOptions;
    exactUnsignedOptions.computeUlp = true;
    exactUnsignedOptions.ulpType = ScalarType::UInt64;
    exactUnsignedOptions.maximumUlpTolerance = 0.0;
    exactUnsignedOptions.maxReportedMismatches = 1;
    const ComparisonReport exactUnsignedResult =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(exactUnsignedObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(exactUnsignedExpected)),
                exactUnsignedOptions);
    require(exactUnsignedResult.mismatches == 2 &&
                exactUnsignedResult.maxAbsoluteDifference == 1.0 &&
                exactUnsignedResult.frobeniusDifference == std::sqrt(2.0) &&
                exactUnsignedResult.maximumUlp == 1.0 && exactUnsignedResult.sumUlp == 2.0 &&
                exactUnsignedResult.averageUlp == 1.0 && !exactUnsignedResult.ulpPassed &&
                exactUnsignedResult.reportedMismatches[0].absoluteDifference == 1.0,
            "UInt64 comparison evidence lost adjacent differences above 2^53.");

    const std::array<int64_t, 3> exactSignedObserved{
        int64_t{1} << 53,
        std::numeric_limits<int64_t>::max() - 1,
        std::numeric_limits<int64_t>::lowest(),
    };
    const std::array<int64_t, 3> exactSignedExpected{
        (int64_t{1} << 53) + 1,
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::lowest() + 1,
    };
    ComparisonOptions exactSignedOptions = exactUnsignedOptions;
    exactSignedOptions.ulpType = ScalarType::Int64;
    const ComparisonReport exactSignedResult =
        compare(Tensor::copyNativeStorage(std::span<const int64_t>(exactSignedObserved)),
                Tensor::copyNativeStorage(std::span<const int64_t>(exactSignedExpected)),
                exactSignedOptions);
    require(exactSignedResult.mismatches == 3 && exactSignedResult.maxAbsoluteDifference == 1.0 &&
                exactSignedResult.frobeniusDifference == std::sqrt(3.0) &&
                exactSignedResult.maximumUlp == 1.0 && exactSignedResult.sumUlp == 3.0 &&
                exactSignedResult.averageUlp == 1.0 && !exactSignedResult.ulpPassed,
            "Int64 comparison evidence lost adjacent differences at its precision boundaries.");

    const std::array<uint64_t, 1> mixedUnsignedObserved{uint64_t{1} << 53};
    const std::array<int64_t, 1> mixedSignedExpected{(int64_t{1} << 53) + 1};
    const ComparisonReport mixedIntegerResult =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(mixedUnsignedObserved)),
                Tensor::copyNativeStorage(std::span<const int64_t>(mixedSignedExpected)),
                exactUnsignedOptions);
    require(mixedIntegerResult.mismatches == 1 && mixedIntegerResult.maxAbsoluteDifference == 1.0 &&
                mixedIntegerResult.maximumUlp == 1.0 && !mixedIntegerResult.ulpPassed,
            "Mixed UInt64/Int64 comparison lost an adjacent difference above 2^53.");

    const std::array<uint64_t, 1> unsignedMaximum{std::numeric_limits<uint64_t>::max()};
    const std::array<uint64_t, 1> unsignedZero{};
    const ComparisonReport unsignedExtremeResult = compare(
        Tensor::copyNativeStorage(std::span<const uint64_t>(unsignedMaximum)),
        Tensor::copyNativeStorage(std::span<const uint64_t>(unsignedZero)), exactUnsignedOptions);
    require(unsignedExtremeResult.maximumUlp ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()) &&
                unsignedExtremeResult.maxAbsoluteDifference ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()),
            "UInt64 comparison evidence overflowed at the full-width difference.");

    const std::array<int64_t, 1> negativeOne{-1};
    const ComparisonReport mixedSignExtremeResult = compare(
        Tensor::copyNativeStorage(std::span<const uint64_t>(unsignedMaximum)),
        Tensor::copyNativeStorage(std::span<const int64_t>(negativeOne)), exactUnsignedOptions);
    require(mixedSignExtremeResult.mismatches == 1 &&
                mixedSignExtremeResult.maximumUlp == std::ldexp(1.0, 64) &&
                mixedSignExtremeResult.maxAbsoluteDifference == std::ldexp(1.0, 64),
            "Mixed signed/unsigned comparison confused -1 with UInt64 maximum.");

    const std::array<int64_t, 1> signedIntegerObserved{std::numeric_limits<int64_t>::lowest()};
    const std::array<int64_t, 1> signedIntegerExpected{std::numeric_limits<int64_t>::max()};
    const ComparisonReport signedExtremeResult =
        compare(Tensor::copyNativeStorage(std::span<const int64_t>(signedIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const int64_t>(signedIntegerExpected)),
                exactSignedOptions);
    require(!signedExtremeResult.passed() &&
                signedExtremeResult.maximumUlp ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()) &&
                signedExtremeResult.maxAbsoluteDifference ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()),
            "Runtime comparison overflowed the signed-integer decision.");

    const std::array<double, 3> expected{3.0, 4.0, 0.0};
    const std::array<double, 3> observed{0.0, 4.0, 3.0};
    ComparisonOptions metrics;
    metrics.pointwise = false;
    metrics.relativeFrobeniusTolerance = 0.9;
    metrics.computeUlp = true;
    metrics.ulpType = ScalarType::Float64;
    const auto metricResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(observed)),
                Tensor::copyNativeStorage(std::span<const double>(expected)), metrics);
    require(std::abs(metricResult.frobeniusExpected - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusObserved - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusDifference - std::sqrt(18.0)) < 1e-12 &&
                std::abs(metricResult.relativeFrobeniusError - std::sqrt(18.0) / 5.0) < 1e-12 &&
                std::abs(metricResult.relativeMaximumError - 0.75) < 1e-12 &&
                !metricResult.pointwiseEvaluated && metricResult.frobeniusEvaluated &&
                !metricResult.ulpEvaluated && metricResult.frobeniusPassed,
            "Comparison Frobenius evidence is incorrect.");

    const std::array<double, 1> unitExpected{1.0};
    const std::array<double, 1> unitDifference{2.0};
    ComparisonOptions strictNorm;
    strictNorm.pointwise = false;
    strictNorm.computePointwiseStatistics = true;
    strictNorm.computeFrobenius = true;
    strictNorm.relativeFrobeniusTolerance = 1.0;
    strictNorm.strictTolerance = true;
    require(!compare(Tensor::copyNativeStorage(std::span<const double>(unitDifference)),
                     Tensor::copyNativeStorage(std::span<const double>(unitExpected)), strictNorm)
                 .passed(),
            "Strict relative Frobenius comparison accepted its tolerance boundary.");

    const std::array<double, 1> zeroNorm{};
    strictNorm.zeroExpectedNormIsNaN = true;
    const ComparisonReport zeroNormResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(zeroNorm)),
                Tensor::copyNativeStorage(std::span<const double>(zeroNorm)), strictNorm);
    require(std::isnan(zeroNormResult.relativeFrobeniusError) &&
                std::isnan(zeroNormResult.relativeMaximumError) && !zeroNormResult.passed(),
            "IEEE zero-norm comparison policy did not preserve 0/0 as NaN.");

    const std::array<double, 1> infiniteNormValue{std::numeric_limits<double>::infinity()};
    strictNorm.nonFiniteValuesInvalidateRelativeNorms = true;
    const ComparisonReport nonFiniteNormResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(infiniteNormValue)),
                Tensor::copyNativeStorage(std::span<const double>(infiniteNormValue)), strictNorm);
    require(std::isnan(nonFiniteNormResult.relativeFrobeniusError) &&
                std::isnan(nonFiniteNormResult.relativeMaximumError) &&
                !nonFiniteNormResult.passed(),
            "Non-finite norm invalidation did not produce a failing NaN ratio.");
    ComparisonOptions sampledMetrics = metrics;
    sampledMetrics.selection = OutputSelection::strided(1, 2);
    const auto sampledMetricResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(observed)),
                Tensor::copyNativeStorage(std::span<const double>(expected)), sampledMetrics);
    require(sampledMetricResult.compared == 1,
            "Irregular comparison selection visited the wrong element count.");

    const double oneUlp = std::ldexp(1.0, -52);
    const std::array<double, 1> ulpObserved{1.0 + oneUlp};
    const std::array<double, 1> ulpExpected{1.0};
    ComparisonOptions ulp;
    ulp.computeUlp = true;
    ulp.ulpType = ScalarType::Float64;
    ulp.maximumUlpTolerance = 1.0;
    const auto ulpResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(ulpObserved)),
                Tensor::copyNativeStorage(std::span<const double>(ulpExpected)), ulp);
    require(ulpResult.maximumUlp == 1.0 && ulpResult.averageUlp == 1.0 &&
                ulpResult.pointwiseEvaluated && !ulpResult.frobeniusEvaluated &&
                ulpResult.ulpEvaluated && ulpResult.ulpPassed,
            "Comparison ULP evidence is incorrect.");
    require(encodedUlpDistance(0.0, static_cast<double>(std::numeric_limits<float>::denorm_min()),
                               ScalarType::Float32) == 1.0,
            "Encoded ULP distance mishandled the F32 zero/subnormal boundary.");
    require(encodedUlpDistance(1.0, 1.5, ScalarType::Float4E2M1) == 1.0,
            "Encoded ULP distance mishandled a packed scalar sign width.");
    require(encodedUlpDistance(1.0, 2.0, ScalarType::E5M3) == 8.0,
            "Encoded ULP distance treated unsigned E5M3 as a signed encoding.");

    const std::array<std::complex<double>, 2> complexExpected{
        std::complex<double>(std::numeric_limits<double>::infinity(), 2.0),
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 4.0)};
    const auto complexObserved = complexExpected;
    ComparisonOptions nonFinite;
    nonFinite.equalNaNs = true;
    const auto nonFiniteResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexObserved)),
                Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexExpected)),
                nonFinite);
    require(nonFiniteResult.passed() && nonFiniteResult.matchedInfinities == 1 &&
                nonFiniteResult.matchedNaNs == 1,
            "Complex non-finite comparison policy is incorrect.");
    const std::array<double, 1> doubleInfinity{std::numeric_limits<double>::infinity()};
    const std::array<float, 1> floatInfinity{std::numeric_limits<float>::infinity()};
    ComparisonOptions noPointwiseStatistics;
    noPointwiseStatistics.computePointwiseStatistics = false;
    noPointwiseStatistics.computeFrobenius = false;
    const ComparisonReport noPointwiseStatisticsResult = compare(
        Tensor::copyNativeStorage(std::span<const double>(doubleInfinity)),
        Tensor::copyNativeStorage(std::span<const float>(floatInfinity)), noPointwiseStatistics);
    require(
        noPointwiseStatisticsResult.passed() && noPointwiseStatisticsResult.matchedInfinities == 0,
        "Disabled pointwise statistics collected matched infinities.");

    const ComparisonOptions numpyDefaults = allCloseComparisonOptions();
    require(numpyDefaults.absoluteTolerance == 1e-8 && numpyDefaults.relativeTolerance == 1e-5 &&
                numpyDefaults.symmetricRelativeTolerance == 0.0 && !numpyDefaults.strictTolerance &&
                !numpyDefaults.equalNaNs &&
                numpyDefaults.complexPointwiseMode == ComplexPointwiseMode::Magnitude,
            "Allclose options do not expose NumPy's finite-value defaults.");
    const std::array<double, 1> lowerValue{8.0};
    const std::array<double, 1> referenceValue{10.0};
    ComparisonOptions numpyBoundary = allCloseComparisonOptions(0.0, 0.2);
    numpyBoundary.computeFrobenius = false;
    require(
        compare(Tensor::copyNativeStorage(std::span<const double>(lowerValue)),
                Tensor::copyNativeStorage(std::span<const double>(referenceValue)), numpyBoundary)
            .passed(),
        "Allclose rejected the inclusive NumPy boundary.");
    require(!compare(Tensor::copyNativeStorage(std::span<const double>(referenceValue)),
                     Tensor::copyNativeStorage(std::span<const double>(lowerValue)), numpyBoundary)
                 .passed(),
            "Allclose lost NumPy's expected-reference asymmetry.");
    numpyBoundary.strictTolerance = true;
    require(
        !compare(Tensor::copyNativeStorage(std::span<const double>(lowerValue)),
                 Tensor::copyNativeStorage(std::span<const double>(referenceValue)), numpyBoundary)
             .passed(),
        "Strict tolerance did not preserve the legacy exclusive boundary.");

    const std::array<std::complex<double>, 1> componentwiseObserved{std::complex<double>(1.0, 1.0)};
    const std::array<std::complex<double>, 1> componentwiseExpected{std::complex<double>(0.0, 0.0)};
    ComparisonOptions componentwiseComplex = allCloseComparisonOptions(1.0, 0.0);
    componentwiseComplex.computeFrobenius = false;
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseObserved)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseExpected)),
                     componentwiseComplex)
                 .passed(),
            "Complex allclose did not apply magnitude tolerance.");
    componentwiseComplex.complexPointwiseMode = ComplexPointwiseMode::Componentwise;
    require(
        compare(
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(componentwiseObserved)),
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(componentwiseExpected)),
            componentwiseComplex)
            .passed(),
        "Explicit componentwise complex comparison rejected passing components.");

    ComparisonOptions magnitudePointwiseOnly = allCloseComparisonOptions(1.0, 0.0);
    magnitudePointwiseOnly.computePointwiseStatistics = false;
    magnitudePointwiseOnly.computeFrobenius = false;
    magnitudePointwiseOnly.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseObserved)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseExpected)),
                     magnitudePointwiseOnly)
                 .passed(),
            "The optimized pointwise-only path ignored complex magnitude mode.");

    const std::array<std::complex<double>, 1> magnitudeBoundaryObserved{
        std::complex<double>(0.0, 0.0)};
    const std::array<std::complex<double>, 1> magnitudeBoundaryExpected{
        std::complex<double>(3.0, 4.0)};
    ComparisonOptions magnitudeBoundary = allCloseComparisonOptions(0.0, 1.0);
    magnitudeBoundary.computeFrobenius = false;
    require(compare(Tensor::copyNativeStorage(
                        std::span<const std::complex<double>>(magnitudeBoundaryObserved)),
                    Tensor::copyNativeStorage(
                        std::span<const std::complex<double>>(magnitudeBoundaryExpected)),
                    magnitudeBoundary)
                .passed(),
            "Complex allclose rejected its inclusive magnitude boundary.");
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(magnitudeBoundaryExpected)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(magnitudeBoundaryObserved)),
                     magnitudeBoundary)
                 .passed(),
            "Complex allclose lost expected-magnitude asymmetry.");

    const double quietNaN = std::numeric_limits<double>::quiet_NaN();
    const std::array<std::complex<double>, 1> crossComponentNaNObserved{
        std::complex<double>(quietNaN, 1.0)};
    const std::array<std::complex<double>, 1> crossComponentNaNExpected{
        std::complex<double>(1.0, quietNaN)};
    ComparisonOptions magnitudeEqualNaNs = allCloseComparisonOptions(0.0, 0.0, true);
    magnitudeEqualNaNs.computeFrobenius = false;
    const ComparisonReport crossComponentNaNResult = compare(
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(crossComponentNaNObserved)),
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(crossComponentNaNExpected)),
        magnitudeEqualNaNs);
    require(crossComponentNaNResult.passed() && crossComponentNaNResult.matchedNaNs == 1,
            "Complex magnitude comparison did not match logical NaN values.");

    const double infinity = std::numeric_limits<double>::infinity();
    const std::array<std::complex<double>, 1> complexInfinity{
        std::complex<double>(infinity, infinity)};
    const ComparisonReport complexInfinityResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                magnitudeEqualNaNs);
    require(complexInfinityResult.passed() && complexInfinityResult.matchedInfinities == 1,
            "Complex magnitude comparison did not count a matched infinity as one logical value.");
    ComparisonOptions magnitudeInfinityWithFrobenius = allCloseComparisonOptions(0.0, 0.0, true);
    const ComparisonReport complexInfinityWithFrobeniusResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                magnitudeInfinityWithFrobenius);
    require(complexInfinityWithFrobeniusResult.passed() &&
                complexInfinityWithFrobeniusResult.matchedInfinities == 1,
            "Complex magnitude infinity statistics depended on Frobenius evidence.");

    const std::array<std::complex<double>, 1> mismatchedComplexInfinityObserved{
        std::complex<double>(infinity, 1.0)};
    const std::array<std::complex<double>, 1> mismatchedComplexInfinityExpected{
        std::complex<double>(infinity, 2.0)};
    ComparisonOptions permissiveComplexInfinity = allCloseComparisonOptions(infinity, infinity);
    permissiveComplexInfinity.computeFrobenius = false;
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(mismatchedComplexInfinityObserved)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(mismatchedComplexInfinityExpected)),
                     permissiveComplexInfinity)
                 .passed(),
            "Complex magnitude comparison applied finite tolerances to unequal infinities.");

    const std::array<double, 1> mixedReal{3.0};
    const std::array<std::complex<double>, 1> mixedComplex{std::complex<double>(3.0, 4.0)};
    ComparisonOptions mixedMagnitude = allCloseComparisonOptions(0.0, 1.0);
    mixedMagnitude.computeFrobenius = false;
    require(compare(Tensor::copyNativeStorage(std::span<const double>(mixedReal)),
                    Tensor::copyNativeStorage(std::span<const std::complex<double>>(mixedComplex)),
                    mixedMagnitude)
                .passed(),
            "Mixed real/complex comparison did not scale tolerance by the complex reference.");
    require(!compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(mixedComplex)),
                     Tensor::copyNativeStorage(std::span<const double>(mixedReal)), mixedMagnitude)
                 .passed(),
            "Mixed real/complex comparison lost expected-magnitude asymmetry.");

    const std::array<std::complex<double>, 1> signedZeroNaNObserved{
        std::complex<double>(quietNaN, 0.0)};
    const std::array<std::complex<double>, 1> signedZeroNaNExpected{
        std::complex<double>(quietNaN, -0.0)};
    ComparisonOptions signedZeroNaN = allCloseComparisonOptions(0.0, 0.0, true);
    signedZeroNaN.equalSignedZero = false;
    signedZeroNaN.computeFrobenius = false;
    const ComparisonReport signedZeroNaNResult = compare(
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(signedZeroNaNObserved)),
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(signedZeroNaNExpected)),
        signedZeroNaN);
    require(signedZeroNaNResult.passed() && signedZeroNaNResult.matchedNaNs == 1 &&
                signedZeroNaNResult.signedZeroMismatches == 0,
            "Logical complex NaN matching did not precede signed-zero classification.");

    const std::array<double, 4> absoluteCandidates{1e-6, 1e-5, 1e-4, 1e-3};
    const std::array<double, 1> relativeCandidates{0.0};
    const std::array<double, 1> closeObserved{1.00009};
    const std::array<double, 1> closeExpected{1.0};
    const auto tolerance = findAllCloseTolerance(
        Tensor::copyNativeStorage(std::span<const double>(closeObserved)),
        Tensor::copyNativeStorage(std::span<const double>(closeExpected)),
        std::span<const double>(absoluteCandidates), std::span<const double>(relativeCandidates));
    require(tolerance && tolerance->absolute == 1e-4 && tolerance->relative == 0.0,
            "Allclose tolerance search selected the wrong candidate.");

    const std::array<std::complex<double>, 1> complexSearchObserved{
        std::complex<double>(0.09, 0.09)};
    const std::array<std::complex<double>, 1> complexSearchExpected{std::complex<double>(0.0, 0.0)};
    const std::array<double, 1> complexSearchAbsoluteCandidates{0.1};
    const std::array<double, 1> complexSearchRelativeCandidates{0.0};
    require(
        !findAllCloseTolerance(
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchObserved)),
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchExpected)),
            std::span<const double>(complexSearchAbsoluteCandidates),
            std::span<const double>(complexSearchRelativeCandidates)),
        "Allclose tolerance search defaulted complex values to componentwise comparison.");
    ComparisonOptions componentwiseSearch = allCloseComparisonOptions();
    componentwiseSearch.complexPointwiseMode = ComplexPointwiseMode::Componentwise;
    require(
        findAllCloseTolerance(
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchObserved)),
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchExpected)),
            std::span<const double>(complexSearchAbsoluteCandidates),
            std::span<const double>(complexSearchRelativeCandidates), componentwiseSearch)
            .has_value(),
        "Allclose tolerance search did not preserve explicit componentwise comparison.");

    std::array<float, 5> rangedSentinel{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 0.0f, std::numeric_limits<float>::infinity()};
    const SentinelReport rangedSentinelResult = checkUnwrittenSentinel(
        ScalarType::Float32, std::as_bytes(std::span<const float>(rangedSentinel)), 2, 2,
        SentinelRegion::Before);
    require(rangedSentinelResult.checked == 2 && rangedSentinelResult.mismatches == 1 &&
                rangedSentinelResult.reportedMismatches[0].region == SentinelRegion::Before &&
                rangedSentinelResult.reportedMismatches[0].index == 3,
            "Sentinel range did not report an absolute storage element index.");

    std::array<float, 5> guarded{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    Tensor guardedView = Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 3}),
                                                          std::span<const float>(guarded));
    require(checkUnusedTensorStorage(guardedView, guarded.size()).passed(),
            "Unwritten tensor padding sentinel was rejected.");
    const float writtenPadding = 0.0f;
    std::memcpy(guardedView.rawEncodedBackingStorage().data() + 2 * sizeof(float), &writtenPadding,
                sizeof(writtenPadding));
    const auto sentinel = checkUnusedTensorStorage(guardedView, guarded.size());
    require(
        !sentinel.passed() && sentinel.mismatches == 1 && sentinel.reportedMismatches[0].index == 2,
        "Written tensor padding was not detected.");
    bool rejectedOversizedSentinelAllocation = false;
    try {
        (void)checkUnusedTensorStorage(guardedView, guarded.size() + 1);
    } catch (const std::invalid_argument&) {
        rejectedOversizedSentinelAllocation = true;
    }
    require(rejectedOversizedSentinelAllocation,
            "Sentinel check accepted an allocation larger than its storage.");
}
}  // namespace

int main() {
    testRuntimeReferenceGemm();
    testZeroGemmScalarsSuppressNonFiniteOperands();
    testGemmScaleCValidationMatchesExecution();
    testRuntimeMixedAndBlockScaledGemm();
    testPointwiseRoutes();
    testExactIntegerGemm();
    testRuntimeComplexAndExplicitAxisGemm();
    testOutputSelection();
    testStreamingGemmValidation();
    testReferenceEpilogue();
    testReferenceReduction();
    testStructuredSparsity();
    testIndexedGeneration();
    testLinearCombination();
    testReferenceSoftmax();
    testReferenceLayerNorm();
    testReferenceOperationAliasing();
    testActivations();
    testStridedAndOffsetViews();
    testGenerationAndComparison();
    testComparisonProgram();
    return 0;
}
