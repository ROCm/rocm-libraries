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
#include <roc/host_validation/validation.hpp>
#include <stdexcept>

#ifdef HOST_VALIDATION_TEST_OPENMP
#include <omp.h>
#endif

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void testRuntimeReferenceGemm() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    Tensor d(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}));
    const std::array<float, 2> bias{1, -10000};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a))),
        GemmOperand(
            Tensor::fromNative<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b))),
        Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)), d,
        ScalarType::Float32);
    problem.epilogue.beta = 1.0;
    problem.epilogue.bias = VectorBinding{
        Tensor::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.epilogue.scaleA =
        Tensor::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(scaleA));
    problem.epilogue.scaleB =
        Tensor::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(scaleB));
    problem.epilogue.activation = Activation::Relu;

    GemmExecution execution;
    require(static_cast<bool>(queryGemmSupport(problem, execution)),
            "Runtime reference GEMM request support mismatch.");
    const GemmResult result = referenceGemm(problem, execution);
    require(result.runInfo.backendUsed == GemmBackend::Pointwise &&
                result.runInfo.outputElementsWritten == 4 &&
                result.runInfo.outputElementsCovered == 4,
            "Runtime reference GEMM run information mismatch.");
    require(result.output.shape() == Shape{2, 2} &&
                result.output.storage().data() == d.storage().data(),
            "Runtime reference GEMM result did not retain shared output storage.");

    const std::array<float, 4> expected{
        58 * 2 * 5 + 1 + 1,
        0,
        64 * 2 * 7 + 1 + 1,
        0,
    };
    require(compare(d, Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 2}),
                                                 std::span<const float>(expected)))
                .passed(),
            "Runtime reference GEMM result mismatch.");

    execution.backend = GemmBackend::Blocked;
    require(!queryGemmSupport(problem, execution),
            "Runtime reference GEMM request unexpectedly supports a missing backend.");
    const GemmResult fallback = referenceGemm(problem, execution);
    require(fallback.runInfo.backendUsed == GemmBackend::Pointwise &&
                fallback.runInfo.fallbackReason.has_value(),
            "Runtime reference GEMM backend fallback mismatch.");
}

void testZeroGemmScalarsSuppressNonFiniteOperands() {
    using namespace roc::host_validation;

    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<float, 4> nonFiniteA{nan, nan, nan, nan};
    const std::array<float, 4> nonFiniteB{infinity, infinity, infinity, infinity};
    const std::array<float, 4> finiteC{1, 2, 3, 4};
    Tensor output(ScalarType::Float32, Shape{2, 2});

    GemmRequest alphaZero(
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                              std::span<const float>(nonFiniteA))),
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                              std::span<const float>(nonFiniteB))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(finiteC)),
        output, ScalarType::Float32);
    alphaZero.epilogue.alpha = 0.0;
    alphaZero.epilogue.beta = 2.0;
    referenceGemm(alphaZero);
    require(compare(output, Tensor::fromNative<float>(
                                Layout::contiguous(Shape{2, 2}),
                                std::span<const float>(std::array<float, 4>{2, 4, 6, 8})))
                .passed(),
            "Zero alpha propagated a non-finite GEMM operand.");

    const std::array<float, 4> finiteA{1, 2, 3, 4};
    const std::array<float, 4> finiteB{5, 6, 7, 8};
    const std::array<float, 4> nonFiniteC{infinity, infinity, infinity, infinity};
    GemmRequest betaZero(GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                               std::span<const float>(finiteA))),
                         GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                               std::span<const float>(finiteB))),
                         Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                   std::span<const float>(nonFiniteC)),
                         output, ScalarType::Float32);
    betaZero.epilogue.beta = 0.0;
    referenceGemm(betaZero);
    require(compare(output, Tensor::fromNative<float>(
                                Layout::contiguous(Shape{2, 2}),
                                std::span<const float>(std::array<float, 4>{19, 22, 43, 50})))
                .passed(),
            "Zero beta propagated a non-finite C operand.");
}

void testRuntimeMixedAndBlockScaledGemm() {
    using namespace roc::host_validation;

    const std::array<float, 2> aValues{1.25f, 2.5f};
    const std::array<float, 2> bValues{2.0f, 3.0f};
    const std::array<float, 1> cValues{1.0f};
    Tensor a =
        Tensor::fromValues(ScalarType::Float8E4M3, Shape{1, 2}, std::span<const float>(aValues));
    Tensor b =
        Tensor::fromValues(ScalarType::Float8E5M2, Shape{2, 1}, std::span<const float>(bValues));
    Tensor c =
        Tensor::fromValues(ScalarType::BFloat16, Shape{1, 1}, std::span<const float>(cValues));
    Tensor d(ScalarType::Float16, Shape{1, 1});

    GemmOperand operandA(a);
    operandA.computeType = ScalarType::Float4E2M1;
    GemmRequest mixed(std::move(operandA), GemmOperand(b), c, d, ScalarType::Float32);
    mixed.epilogue.beta = 1.0;
    referenceGemm(mixed);
    require(d.loadAs<float>({0, 0}) == 9.0f, "Runtime mixed-type GEMM result mismatch.");

    const std::array<float, 4> ones{1, 1, 1, 1};
    const std::array<float, 1> zero{0};
    const std::array<float, 2> scaleAValues{2, 4};
    const std::array<float, 2> scaleBValues{8, 16};
    Tensor blockA =
        Tensor::fromValues(ScalarType::Float32, Shape{1, 4}, std::span<const float>(ones));
    Tensor blockB =
        Tensor::fromValues(ScalarType::Float32, Shape{4, 1}, std::span<const float>(ones));
    Tensor blockC =
        Tensor::fromValues(ScalarType::Float32, Shape{1, 1}, std::span<const float>(zero));
    Tensor blockD(ScalarType::Float32, Shape{1, 1});
    Tensor scalesA =
        Tensor::fromValues(ScalarType::E8M0, Shape{1, 2}, std::span<const float>(scaleAValues));
    Tensor scalesB =
        Tensor::fromValues(ScalarType::E8M0, Shape{1, 2}, std::span<const float>(scaleBValues));

    GemmOperand blockOperandA(blockA);
    GemmOperand blockOperandB(blockB);
    blockOperandA.blockScale = BlockScaleBinding{scalesA, 2};
    blockOperandB.blockScale = BlockScaleBinding{scalesB, 2};
    GemmRequest blockScaled(std::move(blockOperandA), std::move(blockOperandB), blockC, blockD,
                            ScalarType::Float32);
    referenceGemm(blockScaled);
    require(blockD.loadAs<float>({0, 0}) == 2 * 2 * 8 + 2 * 4 * 16,
            "Runtime block-scaled GEMM result mismatch.");
}

void testPointwiseRoutes() {
    using namespace roc::host_validation;

    const std::array<float, 7> a{1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 14> b{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 2> c{};
    const std::array<float, 3> scaleA{2, 3, 5};
    const std::array<float, 4> scaleB{1, 1, 7, 11};

    auto makeProblem = [&](Tensor output) {
        GemmOperand operandA(
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 7}), std::span<const float>(a)));
        GemmOperand operandB(
            Tensor::fromNative<float>(Layout::contiguous(Shape{7, 2}), std::span<const float>(b)));
        operandA.blockScale = BlockScaleBinding{
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 3}),
                                      std::span<const float>(scaleA)),
            3,
        };
        operandB.blockScale = BlockScaleBinding{
            Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                      std::span<const float>(scaleB)),
            4,
        };
        GemmRequest problem(
            std::move(operandA), std::move(operandB),
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 2}), std::span<const float>(c)),
            std::move(output), ScalarType::Float32);
        problem.outputSelection = OutputSelection::explicitIndices({1});
        return problem;
    };

    Tensor automaticOutput =
        Tensor::fromNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, -99});
    GemmRequest automaticProblem = makeProblem(automaticOutput);
    const GemmResult automatic = referenceGemm(automaticProblem);

    Tensor pointwiseOutput =
        Tensor::fromNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, -99});
    GemmRequest pointwiseProblem = makeProblem(pointwiseOutput);
    const GemmResult pointwise =
        referenceGemm(pointwiseProblem, {
                                            .backend = GemmBackend::Pointwise,
                                            .requireRequestedBackend = true,
                                        });

    const Tensor expected =
        Tensor::fromNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, 184});
    require(
        compare(automaticOutput, expected).passed() && compare(pointwiseOutput, expected).passed(),
        "Automatic and explicit Pointwise routes diverged.");
    require(automatic.runInfo.backendUsed == GemmBackend::Pointwise &&
                pointwise.runInfo.backendUsed == GemmBackend::Pointwise &&
                automatic.runInfo.outputElementsWritten == 1 &&
                pointwise.runInfo.outputElementsWritten == 1 &&
                automatic.runInfo.outputElementsCovered == 1 &&
                pointwise.runInfo.outputElementsCovered == 1 && !automatic.runInfo.fallbackReason &&
                !pointwise.runInfo.fallbackReason,
            "Pointwise route information changed.");
}

void testExactIntegerGemm() {
    using namespace roc::host_validation;

    const std::array<int32_t, 1> a{std::numeric_limits<int32_t>::max()};
    const std::array<int32_t, 1> b{2};
    const std::array<int32_t, 1> c{std::numeric_limits<int32_t>::max()};
    Tensor d(ScalarType::Int32, Shape{1, 1});
    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative(Layout::contiguous(Shape{1, 1}), std::span<const int32_t>(a))),
        GemmOperand(
            Tensor::fromNative(Layout::contiguous(Shape{1, 1}), std::span<const int32_t>(b))),
        Tensor::fromNative(Layout::contiguous(Shape{1, 1}), std::span<const int32_t>(c)), d,
        ScalarType::Int32);
    problem.epilogue.beta = 2.0;
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

    problem.epilogue.alpha = 0.5;
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
    using namespace roc::host_validation;

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1.0f, 2.0f)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3.0f, 4.0f)};
    const std::array<std::complex<float>, 1> complexC{};
    Tensor complexD(ScalarType::ComplexFloat32, Shape{1, 1});

    GemmOperand complexOperandA(Tensor::fromNative<std::complex<float>>(
        Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexA)));
    complexOperandA.conjugate = true;
    GemmRequest complexProblem(
        std::move(complexOperandA),
        GemmOperand(Tensor::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexB))),
        Tensor::fromNative<std::complex<float>>(Layout::contiguous(Shape{1, 1}),
                                                std::span<const std::complex<float>>(complexC)),
        complexD, ScalarType::ComplexFloat32);
    referenceGemm(complexProblem);
    require(complexD.loadAs<std::complex<float>>({0, 0}) == std::complex<float>(11.0f, -2.0f),
            "Runtime complex GEMM result mismatch.");

    const std::array<float, 1> realA{1};
    const std::array<float, 2> realB{0, 0};
    const std::array<float, 2> realC{0, 0};
    const std::array<float, 2> columnBias{2, 3};
    Tensor realD(ScalarType::Float32, Shape{1, 2});
    GemmRequest axisProblem(
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                              std::span<const float>(realA))),
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                              std::span<const float>(realB))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{1, 2}), std::span<const float>(realC)),
        realD, ScalarType::Float32);
    axisProblem.epilogue.bias = VectorBinding{
        Tensor::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(columnBias)),
        MatrixAxis::Column,
    };
    referenceGemm(axisProblem);
    require(compare(realD, Tensor::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                                     std::span<const float>(columnBias)))
                .passed(),
            "Runtime GEMM explicit column-axis bias mismatch.");
}

void testOutputSelection() {
    using namespace roc::host_validation;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{};
    Tensor d =
        Tensor::fromNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});

    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(a))),
        GemmOperand(
            Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(b))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(c)), d,
        ScalarType::Float32);
    problem.outputSelection = OutputSelection::explicitIndices({0, 3});
    const GemmResult result = referenceGemm(problem);
    require(result.runInfo.outputElementsWritten == 2 && result.runInfo.outputElementsCovered == 2,
            "Selected-output GEMM reported the wrong element count.");
    require(d.loadAs<float>({0, 0}) == 19 && d.loadAs<float>({0, 1}) == -99 &&
                d.loadAs<float>({1, 0}) == -99 && d.loadAs<float>({1, 1}) == 50,
            "Selected-output GEMM modified the wrong elements.");

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

void testReferenceEpilogue() {
    using namespace roc::host_validation;

    const std::array<float, 4> input{-2, 1, 3, -4};
    const std::array<float, 2> bias{1, 2};
    Tensor output(ScalarType::Float16, Shape{2, 2});
    Tensor rawOutput(ScalarType::Float32, Shape{2, 2});
    Tensor auxiliary(ScalarType::BFloat16, Shape{2, 2});
    Tensor amax(ScalarType::Float32, Shape{1});

    EpilogueRequest problem(
        Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(input)),
        output, rawOutput, auxiliary, amax, ScalarType::Float32);
    problem.bias = VectorBinding{
        Tensor::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.outputScale = 2.0;
    problem.auxiliaryScale = 3.0;
    problem.activation = Activation::Relu;
    const EpilogueRunInfo run = referenceEpilogue(problem);
    require(run.outputElementsWritten == 4 && run.rawOutputElementsWritten == 4 &&
                run.auxiliaryOutputElementsWritten == 4 && run.amaxElementsWritten == 1,
            "Reference epilogue run information mismatch.");

    require(output.loadAs<float>({0, 0}) == 0 && output.loadAs<float>({0, 1}) == 4 &&
                output.loadAs<float>({1, 0}) == 10 && output.loadAs<float>({1, 1}) == 0,
            "Reference epilogue output mismatch.");
    require(compare(rawOutput,
                    Tensor::fromNativeValues<float>(Shape{2, 2}, std::array<float, 4>{0, 4, 10, 0}))
                .passed(),
            "Reference epilogue raw output mismatch.");
    require(auxiliary.loadAs<float>({0, 0}) == -3 && auxiliary.loadAs<float>({0, 1}) == 6 &&
                auxiliary.loadAs<float>({1, 0}) == 15 && auxiliary.loadAs<float>({1, 1}) == -6,
            "Reference epilogue auxiliary output mismatch.");
    require(amax.loadAs<float>({0}) == 5, "Reference epilogue AMax mismatch.");

    const std::array<float, 4> gradientInput{10, 20, 30, 40};
    const std::array<float, 4> activationInput{-1, 1, 2, -2};
    Tensor gradientOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueRequest gradient(Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                       std::span<const float>(gradientInput)),
                             gradientOutput, ScalarType::Float32);
    gradient.auxiliaryInput = Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                        std::span<const float>(activationInput));
    gradient.activation = Activation::Relu;
    gradient.activationApplication = ActivationApplication::Gradient;
    referenceEpilogue(gradient);
    require(compare(gradientOutput, Tensor::fromNativeValues<float>(
                                        Shape{2, 2}, std::array<float, 4>{0, 20, 30, 0}))
                .passed(),
            "Reference gradient epilogue mismatch.");

    const std::array<float, 4> gate{0.5f, 2.0f, -1.0f, 0.25f};
    Tensor gatedOutput(ScalarType::Float32, Shape{2, 2});
    EpilogueRequest gated(
        Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(input)),
        gatedOutput, ScalarType::Float32);
    gated.gateResidual =
        Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(gate));
    gated.outputScale = 2.0;
    referenceEpilogue(gated);
    require(compare(gatedOutput, Tensor::fromNativeValues<float>(
                                     Shape{2, 2}, std::array<float, 4>{-1.5f, 6.0f, -7.0f, -1.75f}))
                .passed(),
            "Reference gate-residual epilogue mismatch.");

    const std::array<float, 4> int8Input{-200.0f, -128.5f, 126.5f, 300.0f};
    Tensor int8Output(ScalarType::Int8, Shape{2, 2});
    EpilogueRequest saturatingInt8(Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                             std::span<const float>(int8Input)),
                                   int8Output, ScalarType::Float32);
    saturatingInt8.outputConversion = OutputConversion::SaturatingInt8;
    referenceEpilogue(saturatingInt8);
    require(compare(int8Output, Tensor::fromNativeValues<int8_t>(
                                    Shape{2, 2}, std::array<int8_t, 4>{-128, -128, 126, 127}))
                .passed(),
            "Reference epilogue Int8 saturation mismatch.");

    const Tensor ownedInput =
        Tensor::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(input));
    EpilogueProblem ownedProblem(ownedInput, ScalarType::Float16, ScalarType::Float32);
    ownedProblem.rawOutputType = ScalarType::Float32;
    ownedProblem.auxiliaryOutputType = ScalarType::BFloat16;
    ownedProblem.amaxType = ScalarType::Float32;
    ownedProblem.bias = problem.bias;
    ownedProblem.outputScale = 2.0;
    ownedProblem.auxiliaryScale = 3.0;
    ownedProblem.activation = Activation::Relu;
    ownedProblem.outputSelection = OutputSelection::explicitIndices({1, 2});

    size_t allocatorCalls = 0;
    const TensorStorageAllocator nonzeroAllocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        TensorStorage storage = TensorStorage::allocate(bytes);
        std::fill(storage.mutableBytes().begin(), storage.mutableBytes().end(), std::byte{0xa5});
        return storage;
    };
    const EpilogueResult owned = referenceEpilogue(ownedProblem, nonzeroAllocator);
    require(allocatorCalls == 4 && owned.output.layout() == Layout::contiguous(Shape{2, 2}) &&
                owned.rawOutput && owned.auxiliaryOutput && owned.amax &&
                owned.runInfo.outputElementsWritten == 2 &&
                owned.runInfo.rawOutputElementsWritten == 2 &&
                owned.runInfo.auxiliaryOutputElementsWritten == 2 &&
                owned.runInfo.amaxElementsWritten == 1,
            "Owning reference epilogue result contract mismatch.");
    require(owned.output.loadAs<float>({0, 0}) == 0 && owned.output.loadAs<float>({0, 1}) == 4 &&
                owned.output.loadAs<float>({1, 0}) == 10 &&
                owned.output.loadAs<float>({1, 1}) == 0 &&
                owned.rawOutput->loadAs<float>({0, 0}) == 0 &&
                owned.rawOutput->loadAs<float>({1, 1}) == 0 &&
                owned.auxiliaryOutput->loadAs<float>({0, 0}) == 0 &&
                owned.auxiliaryOutput->loadAs<float>({1, 1}) == 0 &&
                owned.amax->loadAs<float>({0}) == 5,
            "Owning reference epilogue retained allocator-dependent unselected values.");

    EpilogueProblem emptySelection = ownedProblem;
    emptySelection.outputSelection = OutputSelection::strided(4, 1);
    const EpilogueResult empty = referenceEpilogue(emptySelection);
    require(empty.runInfo.outputElementsWritten == 0 &&
                empty.runInfo.rawOutputElementsWritten == 0 &&
                empty.runInfo.auxiliaryOutputElementsWritten == 0 &&
                empty.runInfo.amaxElementsWritten == 1 && empty.output.loadAs<float>({0, 0}) == 0 &&
                empty.rawOutput && empty.rawOutput->loadAs<float>({0, 0}) == 0 &&
                empty.auxiliaryOutput && empty.auxiliaryOutput->loadAs<float>({0, 0}) == 0 &&
                empty.amax && empty.amax->loadAs<float>({0}) == 0,
            "Owning reference epilogue empty selection was not zero initialized.");

    EpilogueProblem invalidProblem(ownedInput, ScalarType::E8M0, ScalarType::Float32);
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceEpilogue(invalidProblem, nonzeroAllocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 4,
            "Owning reference epilogue allocated before validating output type.");

    TensorStorage overlappingStorage = TensorStorage::allocate(ownedInput.storage().size());
    size_t overlappingAllocatorCalls = 0;
    const TensorStorageAllocator overlappingAllocator = [&overlappingStorage,
                                                         &overlappingAllocatorCalls](size_t) {
        ++overlappingAllocatorCalls;
        return overlappingStorage;
    };
    bool rejectedOverlap = false;
    try {
        (void)referenceEpilogue(ownedProblem, overlappingAllocator);
    } catch (const std::invalid_argument&) {
        rejectedOverlap = true;
    }
    require(rejectedOverlap && overlappingAllocatorCalls == 4,
            "Owning reference epilogue accepted overlapping allocator results.");

    Tensor preservedOutput =
        Tensor::fromNativeValues<float>(Shape{2, 2}, std::array<float, 4>{-99, -99, -99, -99});
    Tensor preservedRaw = preservedOutput.clone();
    Tensor preservedAuxiliary = preservedOutput.clone();
    Tensor accumulatedAmax = Tensor::fromNativeValues<float>(Shape{1}, std::array<float, 1>{100});
    EpilogueRequest preserved(ownedInput, preservedOutput, preservedRaw, preservedAuxiliary,
                              accumulatedAmax, ScalarType::Float32);
    preserved.outputSelection = OutputSelection::explicitIndices({0, 3});
    preserved.accumulateAmax = true;
    const EpilogueRunInfo preservedRun = referenceEpilogue(preserved);
    require(preservedRun.outputElementsWritten == 2 &&
                preservedOutput.loadAs<float>({0, 1}) == -99 &&
                preservedOutput.loadAs<float>({1, 0}) == -99 &&
                preservedRaw.loadAs<float>({0, 1}) == -99 &&
                preservedAuxiliary.loadAs<float>({1, 0}) == -99 &&
                accumulatedAmax.loadAs<float>({0}) == 100,
            "Explicit reference epilogue did not preserve unselected or accumulated state.");
}

void testReferenceReduction() {
    using namespace roc::host_validation;

    std::array<float, 30> storage;
    storage.fill(-1);
    Tensor input =
        Tensor::fromNative<float>(Layout(Shape{2, 3, 4}, {15, 5, 1}), std::span<float>(storage));
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 3; ++row) {
            for (size_t column = 0; column < 4; ++column)
                input.storeFrom({batch, row, column}, 100 * batch + 10 * row + column);
        }
    }

    Tensor output(ScalarType::Float32, Layout(Shape{3}, {2}));
    ReductionRequest request(input, output, ScalarType::Float32, {0, 2});
    const ReductionRunInfo run = referenceSum(request);
    require(run.outputElementsWritten == 3 && run.inputElementsRead == 24,
            "Reference reduction run information mismatch.");
    require(compare(output,
                    Tensor::fromNativeValues<float>(Shape{3}, std::array<float, 3>{412, 492, 572}))
                .passed(),
            "Reference reduction result mismatch.");

    Tensor maximumAbsolute(ScalarType::Float32, Shape{});
    const ReductionRunInfo maximumRun =
        referenceMaximumAbsolute(input, maximumAbsolute, ScalarType::Float32);
    require(maximumRun.outputElementsWritten == 1 && maximumRun.inputElementsRead == 24,
            "Reference maximum-absolute run information mismatch.");
    require(maximumAbsolute.loadAs<float>({}) == 123.0f,
            "Reference maximum-absolute result mismatch.");

    const ReductionProblem ownedProblem(input, ScalarType::Float32, ScalarType::Float32, {0, 2});
    const ReductionResult owned = referenceSum(ownedProblem);
    require(owned.output.layout() == Layout::contiguous(Shape{3}) &&
                owned.output.type() == ScalarType::Float32 &&
                owned.runInfo.outputElementsWritten == 3 && owned.runInfo.inputElementsRead == 24 &&
                compare(owned.output, Tensor::fromNativeValues<float>(
                                          Shape{3}, std::array<float, 3>{412, 492, 572}))
                    .passed(),
            "Owning reference sum result contract mismatch.");

    size_t allocatorCalls = 0;
    const TensorStorageAllocator allocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        return TensorStorage::allocate(bytes);
    };
    const ReductionResult allocated = referenceSum(ownedProblem, allocator);
    require(allocatorCalls == 1 && allocated.output.layout() == Layout::contiguous(Shape{3}),
            "Owning reference sum did not use the supplied allocator exactly once.");

    const ReductionProblem invalidProblem(input, ScalarType::Float32, ScalarType::Float32, {0, 0});
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceSum(invalidProblem, allocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 1,
            "Owning reference sum allocated before validating its axes.");

    const ReductionResult ownedMaximum =
        referenceMaximumAbsolute(input, ScalarType::Float32, ScalarType::Float32);
    require(ownedMaximum.output.shape() == Shape{} &&
                ownedMaximum.output.layout() == Layout::contiguous(Shape{}) &&
                ownedMaximum.runInfo.outputElementsWritten == 1 &&
                ownedMaximum.runInfo.inputElementsRead == 24 &&
                ownedMaximum.output.loadAs<float>({}) == 123.0f,
            "Owning reference maximum-absolute result contract mismatch.");
}

void testStructuredSparsity() {
    using namespace roc::host_validation;

    std::array<float, 20> inputStorage;
    inputStorage.fill(-99);
    Tensor input(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                 std::as_writable_bytes(std::span<float>(inputStorage)));
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 8; ++column)
            input.storeFrom({row, column}, static_cast<float>(1 + row * 8 + column));

    std::array<float, 20> prunedStorage;
    prunedStorage.fill(-7);
    Tensor pruned(ScalarType::Float32, Layout(Shape{2, 8}, {10, 1}),
                  std::as_writable_bytes(std::span<float>(prunedStorage)));
    std::array<float, 8> compressedStorage{};
    Tensor compressed = Tensor::fromNative<float>(Layout::contiguous(Shape{2, 4}),
                                                  std::span<float>(compressedStorage));
    std::array<uint8_t, 8> indexStorage{};
    Tensor retainedIndices = Tensor::fromNative<uint8_t>(Layout::contiguous(Shape{2, 4}),
                                                         std::span<uint8_t>(indexStorage));

    StructuredSparsityPattern pattern;
    pattern.axis = 1;
    pattern.fixedPositions = {1, 3};
    const StructuredSparsityRunInfo run = applyStructuredSparsity(
        StructuredSparsityRequest(input, pruned, compressed, retainedIndices, pattern));
    require(run.groupsProcessed == 4 && run.prunedElementsWritten == 16 &&
                run.compressedElementsWritten == 8,
            "Structured sparsity run information mismatch.");

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
    const TwoOfFourMetadataRunInfo metadataRun =
        encodeTwoOfFourMetadata(TwoOfFourMetadataRequest(retainedIndices, metadata, 1));
    require(metadataRun.sparsityGroupsEncoded == 4 && metadataRun.metadataBytesWritten == 2,
            "Two-of-four metadata run information mismatch.");
    require(metadata.loadAs<uint8_t>({0, 0}) == 0xdd && metadata.loadAs<uint8_t>({1, 0}) == 0xdd,
            "Two-of-four metadata encoding mismatch.");

    Tensor fusedPruned(ScalarType::Float32, Shape{2, 8});
    Tensor fusedCompressed(ScalarType::Float32, Shape{2, 4});
    Tensor fusedMetadata(ScalarType::UInt8, Shape{2, 1});
    StructuredSparsityRequest fusedRequest(input, fusedPruned, fusedCompressed, std::nullopt,
                                           fusedMetadata, pattern);
    const StructuredSparsityRunInfo firstFusedRun =
        applyStructuredSparsity(fusedRequest, {.firstSlice = 0, .sliceCount = 1});
    const StructuredSparsityRunInfo secondFusedRun =
        applyStructuredSparsity(fusedRequest, {.firstSlice = 1, .sliceCount = 1});
    require(
        firstFusedRun.groupsProcessed == 2 && secondFusedRun.groupsProcessed == 2 &&
            firstFusedRun.retainedIndicesWritten == 0 &&
            secondFusedRun.retainedIndicesWritten == 0 && firstFusedRun.metadataBytesWritten == 1 &&
            secondFusedRun.metadataBytesWritten == 1 && compare(fusedMetadata, metadata).passed(),
        "Fused structured sparsity metadata mismatch.");

    Tensor inPlace =
        Tensor::fromNativeValues<float>(Shape{8}, std::array<float, 8>{1, 2, 3, 4, 5, 6, 7, 8});
    Tensor inPlaceCompressed(ScalarType::Float32, Shape{4});
    Tensor inPlaceIndices(ScalarType::UInt8, Shape{4});
    pattern.axis = 0;
    pattern.fixedPositions = {0, 2};
    applyStructuredSparsity(
        StructuredSparsityRequest(inPlace, inPlace, inPlaceCompressed, inPlaceIndices, pattern));
    require(inPlace.loadAs<float>({0}) == 1 && inPlace.loadAs<float>({1}) == 0 &&
                inPlace.loadAs<float>({2}) == 3 && inPlace.loadAs<float>({3}) == 0,
            "In-place structured sparsity mismatch.");

    const std::array<float, 12> ownedValues{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    StructuredSparsityPattern ownedPattern;
    ownedPattern.axis = 0;
    ownedPattern.fixedPositions = {0, 2};
    StructuredSparsityProblem ownedProblem(
        Tensor::fromNativeValues<float>(Shape{12}, std::span<const float>(ownedValues)),
        ownedPattern, {.retainedIndices = true, .twoOfFourMetadata = true});
    size_t allocatorCalls = 0;
    const TensorStorageAllocator nonzeroAllocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        TensorStorage storage = TensorStorage::allocate(bytes);
        std::fill(storage.mutableBytes().begin(), storage.mutableBytes().end(), std::byte{0xa5});
        return storage;
    };
    const StructuredSparsityResult owned = applyStructuredSparsity(ownedProblem, nonzeroAllocator);
    require(allocatorCalls == 4 && owned.pruned.layout() == Layout::contiguous(Shape{12}) &&
                owned.compressed.layout() == Layout::contiguous(Shape{6}) &&
                owned.retainedIndices &&
                owned.retainedIndices->layout() == Layout::contiguous(Shape{6}) &&
                owned.twoOfFourMetadata &&
                owned.twoOfFourMetadata->layout() == Layout::contiguous(Shape{2}) &&
                owned.runInfo.groupsProcessed == 3,
            "Owning structured-sparsity result contract mismatch.");
    require(owned.twoOfFourMetadata->loadAs<uint8_t>({0}) == 0x88 &&
                owned.twoOfFourMetadata->loadAs<uint8_t>({1}) == 0x08,
            "Owning structured-sparsity metadata retained allocator-dependent bits.");

    StructuredSparsityPattern invalidMetadataPattern = ownedPattern;
    invalidMetadataPattern.retainedElements = 1;
    invalidMetadataPattern.fixedPositions = {0};
    StructuredSparsityProblem invalidMetadataProblem(
        ownedProblem.input, invalidMetadataPattern,
        {.retainedIndices = false, .twoOfFourMetadata = true});
    bool rejectedBeforeAllocation = false;
    try {
        (void)applyStructuredSparsity(invalidMetadataProblem, nonzeroAllocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 4,
            "Owning structured sparsity allocated before validating metadata policy.");

    TensorStorage overlappingStorage = TensorStorage::allocate(ownedProblem.input.storage().size());
    size_t overlappingAllocatorCalls = 0;
    const TensorStorageAllocator overlappingAllocator = [&overlappingStorage,
                                                         &overlappingAllocatorCalls](size_t) {
        ++overlappingAllocatorCalls;
        return overlappingStorage;
    };
    bool rejectedOverlap = false;
    try {
        (void)applyStructuredSparsity(ownedProblem, overlappingAllocator);
    } catch (const std::invalid_argument&) {
        rejectedOverlap = true;
    }
    require(rejectedOverlap && overlappingAllocatorCalls == 4,
            "Owning structured sparsity accepted overlapping allocator results.");

    const TwoOfFourMetadataResult ownedMetadata = encodeTwoOfFourMetadata(
        TwoOfFourMetadataProblem(*owned.retainedIndices, 0), nonzeroAllocator);
    require(allocatorCalls == 5 && ownedMetadata.metadata.shape() == Shape{2} &&
                ownedMetadata.metadata.loadAs<uint8_t>({0}) == 0x88 &&
                ownedMetadata.metadata.loadAs<uint8_t>({1}) == 0x08,
            "Owning two-of-four metadata result contract mismatch.");

    const std::array<uint8_t, 2> invalidRetainedValues{2, 1};
    const TwoOfFourMetadataProblem invalidRetained(
        Tensor::fromNativeValues<uint8_t>(Shape{2},
                                          std::span<const uint8_t>(invalidRetainedValues)),
        0);
    rejectedBeforeAllocation = false;
    try {
        (void)encodeTwoOfFourMetadata(invalidRetained, nonzeroAllocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 5,
            "Owning two-of-four metadata allocated before validating retained positions.");

    StructuredSparsityPattern widePattern;
    widePattern.axis = 0;
    widePattern.groupSize = 257;
    widePattern.retainedElements = 1;
    widePattern.fixedPositions = {0};
    const StructuredSparsityResult wide = applyStructuredSparsity(
        StructuredSparsityProblem(Tensor(ScalarType::Float32, Shape{257}), widePattern));
    require(wide.compressed.shape() == Shape{1} && !wide.retainedIndices,
            "Structured sparsity imposed the UInt8 index limit without an index output.");
}

void testIndexedGeneration() {
    using namespace roc::host_validation;

    Tensor serial(ScalarType::Float32, Shape{2, 3});
    const GenerationRunInfo serialRun =
        generate(serial, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(serialRun.elementsGenerated == 6, "Indexed generation count mismatch.");
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
    constexpr uint64_t candidateSeed = 37;
    generate(candidates,
             GenerationRecipe::realOnly(GenerationRecipe::candidateSet({.values = candidateValues}),
                                        {.seed = candidateSeed}));
    for (size_t index = 0; index < candidates.size(); ++index) {
        const double expected =
            candidateValues[counterRandom(candidateSeed,
                                          generation_random_domain_version_1::realComponent,
                                          index) %
                            candidateValues.size()];
        require(candidates.loadAs<float>({index}) == expected,
                "Candidate-set generation mismatch.");
    }

    Tensor point(ScalarType::Float32, Shape{2, 3, 2});
    const GenerationRunInfo pointRun = generateAt(
        point, 3, GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 9.0})));
    require(pointRun.elementsGenerated == 1 && point.loadAs<float>({1, 1, 0}) == 9.0f &&
                point.loadAs<float>({0, 1, 1}) == 0.0f,
            "First-dimension-fast point generation mismatch.");

    generateAt(point, 3,
               GenerationRecipe::realOnly(GenerationRecipe::constant({.value = 7.0}),
                                          {.indexOrder = LogicalIndexOrder::LastDimensionFastest}));
    require(point.loadAs<float>({0, 1, 1}) == 7.0f,
            "Last-dimension-fast point generation mismatch.");

    for (const LogicalIndexOrder order :
         {LogicalIndexOrder::FirstDimensionFastest, LogicalIndexOrder::LastDimensionFastest}) {
        Tensor whole(ScalarType::Float4E2M1, Shape{2, 3, 2});
        Tensor elementwise(ScalarType::Float4E2M1, Shape{2, 3, 2});
        const GenerationRecipe exactRecipe =
            GenerationRecipe::realOnly(GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}),
                                       {.seed = 0x12345678, .indexOrder = order});
        generate(whole, exactRecipe);
        for (size_t index = 0; index < whole.size(); ++index)
            generateAt(elementwise, index, exactRecipe);
        require(std::equal(whole.storage().begin(), whole.storage().end(),
                           elementwise.storage().begin(), elementwise.storage().end()),
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

#ifdef HOST_VALIDATION_TEST_OPENMP
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
    require(std::equal(oneThread.storage().begin(), oneThread.storage().end(),
                       fourThreads.storage().begin(), fourThreads.storage().end()),
            "Ordinary generation changed with OpenMP thread count.");

    Tensor aliased(ScalarType::Float32, Layout(Shape{8192}, {0}));
    generate(aliased, GenerationRecipe::realOnly(GenerationRecipe::serialIndex()));
    require(aliased.loadAs<float>({0}) == 8191.0f,
            "Aliased generation did not preserve deterministic traversal order.");

    omp_set_num_threads(originalThreadCount);
    omp_set_dynamic(originalDynamic);
#endif
}

void testReferenceAxpby() {
    using namespace roc::host_validation;

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

    AxpbyRequest request(x, y, output, ScalarType::Float32);
    request.alpha = 2.0;
    request.beta = -0.5;
    const AxpbyRunInfo run = referenceAxpby(request);
    require(run.outputElementsWritten == shape.elementCount(),
            "Reference AXPBY element count mismatch.");

    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t column = 0; column < 2; ++column) {
                const std::array<size_t, 3> indices{row, column, batch};
                const float expected =
                    2.0f * x.loadAs<float>(indices) - 0.5f * y.loadAs<float>(indices);
                require(output.loadAs<float>(indices) == expected,
                        "Reference AXPBY value mismatch.");
            }
        }
    }

    Tensor yOnlyOutput(ScalarType::Float32, shape);
    AxpbyRequest yOnly(std::nullopt, y, yOnlyOutput, ScalarType::Float32);
    yOnly.beta = 3.0;
    referenceAxpby(yOnly);
    require(yOnlyOutput.loadAs<float>({1, 0, 1}) == 3.0f * y.loadAs<float>({1, 0, 1}),
            "Reference AXPBY optional-input mismatch.");

    const std::array<std::complex<float>, 1> complexXValues{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexYValues{std::complex<float>(3, -1)};
    Tensor complexX = Tensor::fromNativeValues<std::complex<float>>(Shape{1}, complexXValues);
    Tensor complexY = Tensor::fromNativeValues<std::complex<float>>(Shape{1}, complexYValues);
    Tensor complexOutput(ScalarType::ComplexFloat32, Shape{1});
    AxpbyRequest complexRequest(complexX, complexY, complexOutput, ScalarType::ComplexFloat32);
    complexRequest.alpha = std::complex<double>(0.5, 1.0);
    complexRequest.beta = -2.0;
    referenceAxpby(complexRequest);
    require(complexOutput.loadAs<std::complex<float>>({0}) == std::complex<float>(-7.5f, 4.0f),
            "Complex reference AXPBY mismatch.");

    AxpbyProblem ownedProblem(x, y, ScalarType::Float32, ScalarType::Float32);
    ownedProblem.alpha = 2.0;
    ownedProblem.beta = -0.5;
    const AxpbyResult owned = referenceAxpby(ownedProblem);
    require(owned.output.layout() == Layout::contiguous(shape) &&
                owned.output.type() == ScalarType::Float32 &&
                owned.runInfo.outputElementsWritten == shape.elementCount() &&
                owned.output.loadAs<float>({1, 0, 1}) ==
                    2.0f * x.loadAs<float>({1, 0, 1}) - 0.5f * y.loadAs<float>({1, 0, 1}),
            "Owning reference AXPBY result contract mismatch.");

    size_t allocatorCalls = 0;
    const TensorStorageAllocator allocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        return TensorStorage::allocate(bytes);
    };
    const AxpbyResult allocated = referenceAxpby(ownedProblem, allocator);
    require(allocatorCalls == 1 && allocated.output.layout() == Layout::contiguous(shape),
            "Owning reference AXPBY did not use the supplied allocator exactly once.");

    AxpbyProblem invalidProblem(std::nullopt, std::nullopt, ScalarType::Float32,
                                ScalarType::Float32);
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceAxpby(invalidProblem, allocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 1,
            "Owning reference AXPBY allocated before validating its problem.");

    AxpbyProblem invalidCoefficient(x, std::nullopt, ScalarType::Float32, ScalarType::Float32);
    invalidCoefficient.alpha = std::complex<double>(1.0, 1.0);
    rejectedBeforeAllocation = false;
    try {
        (void)referenceAxpby(invalidCoefficient, allocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 1,
            "Owning reference AXPBY allocated before validating its coefficients.");
}

void testReferenceSoftmax() {
    using namespace roc::host_validation;

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

    const SoftmaxRunInfo run =
        referenceSoftmax(SoftmaxRequest(input, output, 1, ScalarType::Float32));
    require(run.slicesProcessed == 4 && run.outputElementsWritten == shape.elementCount(),
            "Reference softmax run information mismatch.");

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

    const SoftmaxProblem ownedProblem(input, ScalarType::Float32, 1, ScalarType::Float32);
    const SoftmaxResult owned = referenceSoftmax(ownedProblem);
    require(owned.output.layout() == Layout::contiguous(shape) &&
                owned.output.type() == ScalarType::Float32 && owned.runInfo.slicesProcessed == 4 &&
                owned.runInfo.outputElementsWritten == shape.elementCount(),
            "Owning reference softmax result contract mismatch.");

    size_t allocatorCalls = 0;
    const TensorStorageAllocator allocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        return TensorStorage::allocate(bytes);
    };
    const SoftmaxResult allocated = referenceSoftmax(ownedProblem, allocator);
    require(allocatorCalls == 1 && allocated.output.layout() == Layout::contiguous(shape),
            "Owning reference softmax did not use the supplied allocator exactly once.");

    const SoftmaxProblem invalidProblem(input, ScalarType::Float32, shape.rank(),
                                        ScalarType::Float32);
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceSoftmax(invalidProblem, allocator);
    } catch (const std::out_of_range&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 1,
            "Owning reference softmax allocated before validating its problem.");
}

void testReferenceLayerNorm() {
    using namespace roc::host_validation;

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
    const Tensor gamma =
        Tensor::fromValues(ScalarType::Float16, Shape{3}, std::span<const float>(gammaValues));
    const Tensor beta =
        Tensor::fromValues(ScalarType::BFloat16, Shape{3}, std::span<const float>(betaValues));
    Tensor output(ScalarType::Float32, Layout(shape, {1, 4, 12}));
    Tensor mean(ScalarType::Float32, Layout(Shape{2, 2}, {3, 1}));
    Tensor inverseVariance(ScalarType::Float32, Layout(Shape{2, 2}, {1, 3}));

    LayerNormRequest request(input, output, mean, inverseVariance, 1, ScalarType::Float32);
    request.gamma = gamma;
    request.beta = beta;
    const LayerNormRunInfo run = referenceLayerNorm(request);
    require(run.slicesProcessed == 4 && run.outputElementsWritten == shape.elementCount() &&
                run.meanElementsWritten == 4 && run.inverseVarianceElementsWritten == 4,
            "Reference LayerNorm run information mismatch.");
    require(mean.loadAs<float>({0, 0}) == 2.0f, "Reference LayerNorm mean mismatch.");
    require(std::abs(inverseVariance.loadAs<float>({0, 0}) -
                     1.0f / std::sqrt(2.0f / 3.0f + 1e-5f)) < 1e-6f,
            "Reference LayerNorm inverse variance mismatch.");
    require(output.loadAs<float>({0, 1, 0}) == -0.5f,
            "Reference LayerNorm affine output mismatch.");

    LayerNormProblem ownedProblem(input, ScalarType::Float32, 1, ScalarType::Float32);
    ownedProblem.meanType = ScalarType::Float32;
    ownedProblem.inverseVarianceType = ScalarType::Float32;
    ownedProblem.gamma = gamma;
    ownedProblem.beta = beta;
    const LayerNormResult owned = referenceLayerNorm(ownedProblem);
    require(owned.output.layout() == Layout::contiguous(shape) && owned.mean &&
                owned.mean->layout() == Layout::contiguous(Shape{2, 2}) && owned.inverseVariance &&
                owned.inverseVariance->layout() == Layout::contiguous(Shape{2, 2}) &&
                owned.runInfo.slicesProcessed == 4 &&
                owned.runInfo.outputElementsWritten == shape.elementCount() &&
                owned.runInfo.meanElementsWritten == 4 &&
                owned.runInfo.inverseVarianceElementsWritten == 4 &&
                owned.output.loadAs<float>({0, 1, 0}) == -0.5f,
            "Owning reference LayerNorm result contract mismatch.");

    size_t allocatorCalls = 0;
    const TensorStorageAllocator allocator = [&allocatorCalls](size_t bytes) {
        ++allocatorCalls;
        return TensorStorage::allocate(bytes);
    };
    const LayerNormResult allocated = referenceLayerNorm(ownedProblem, allocator);
    require(allocatorCalls == 3 && allocated.mean && allocated.inverseVariance,
            "Owning reference LayerNorm did not allocate each requested result exactly once.");

    LayerNormProblem invalidProblem = ownedProblem;
    invalidProblem.epsilon = std::numeric_limits<double>::quiet_NaN();
    bool rejectedBeforeAllocation = false;
    try {
        (void)referenceLayerNorm(invalidProblem, allocator);
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation && allocatorCalls == 3,
            "Owning reference LayerNorm allocated before validating epsilon.");

    LayerNormProblem outputOnly(input, ScalarType::Float32, 1, ScalarType::Float32);
    const LayerNormResult outputOnlyResult = referenceLayerNorm(outputOnly);
    require(!outputOnlyResult.mean && !outputOnlyResult.inverseVariance &&
                outputOnlyResult.runInfo.meanElementsWritten == 0 &&
                outputOnlyResult.runInfo.inverseVarianceElementsWritten == 0,
            "Owning reference LayerNorm created unrequested statistics.");

    const std::array<float, 3> rankOneValues{1.0f, 2.0f, 3.0f};
    LayerNormProblem rankOne(
        Tensor::fromNativeValues<float>(Shape{3}, std::span<const float>(rankOneValues)),
        ScalarType::Float32, 0, ScalarType::Float32);
    rankOne.meanType = ScalarType::Float32;
    const LayerNormResult rankOneResult = referenceLayerNorm(rankOne);
    require(rankOneResult.mean && rankOneResult.mean->shape() == Shape{} &&
                !rankOneResult.inverseVariance && rankOneResult.runInfo.meanElementsWritten == 1,
            "Owning reference LayerNorm did not preserve a requested rank-zero statistic.");
}

void testActivations() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{1};
    const std::array<float, 1> c{0};
    Tensor d(ScalarType::Float32, Shape{1, 1});

    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(a))),
        GemmOperand(
            Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(b))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(c)), d,
        ScalarType::Float32);

    problem.epilogue.activation = Activation::Gelu;
    referenceGemm(problem);
    require(std::abs(d.loadAs<float>({0, 0}) - 1.9545977f) < 1e-6f, "GELU result mismatch.");

    problem.epilogue.activation = Activation::Silu;
    problem.epilogue.activationParameter0 = 1;
    referenceGemm(problem);
    require(std::abs(d.loadAs<float>({0, 0}) - 1.7615942f) < 1e-6f, "SiLU result mismatch.");

    problem.epilogue.activation = Activation::Clamp;
    problem.epilogue.activationParameter0 = -1;
    problem.epilogue.activationParameter1 = 1;
    referenceGemm(problem);
    require(d.loadAs<float>({0, 0}) == 1, "Clamp result mismatch.");
}

void testStridedAndOffsetViews() {
    using namespace roc::host_validation;

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
    GemmRequest problem(
        GemmOperand(
            Tensor::fromNative<float>(Layout(Shape{2, 3}, {4, 1}), std::span<const float>(a))),
        GemmOperand(
            Tensor::fromNative<float>(Layout(Shape{3, 2}, {3, 1}), std::span<const float>(b))),
        Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 4}), std::span<const float>(c)),
        Tensor::fromStorage(ScalarType::Float32, outputLayout, std::move(dStorage)),
        ScalarType::Float32);
    Tensor d = problem.d;
    problem.epilogue.alpha = 2.0;
    problem.epilogue.beta = 3.0;

    referenceGemm(problem);

    std::array<float, 12> expected;
    expected.fill(-99);
    expected[1] = 2 * 58 + 3;
    expected[2] = 2 * 139 + 3;
    expected[6] = 2 * 64 + 3;
    expected[7] = 2 * 154 + 3;
    const auto comparison =
        compare(d, Tensor::fromNative<float>(outputLayout, std::span<const float>(expected)));
    require(comparison.passed(), "Strided GEMM matrix comparison failed.");
    const auto storageValue = [&d](size_t index) {
        float value;
        std::memcpy(&value, d.storage().data() + index * sizeof(float), sizeof(value));
        return value;
    };
    require(storageValue(0) == -99 && storageValue(3) == -99 && storageValue(11) == -99,
            "Strided GEMM modified padding.");
}

void testGenerationAndComparison() {
    using namespace roc::host_validation;

    require(counterRandom(7, 3, 11) == counterRandom(7, 3, 11),
            "Counter-based generation is not deterministic.");
    require(counterRandom(7, 3, 11) != counterRandom(7, 3, 12),
            "Counter-based generation does not vary by logical index.");
    require(counterRandom(0, 0, 0) == 0x6e789e6aa1b965f4ULL &&
                counterRandom(7, 3, 11) == 0xf6dd3a1482c56d3fULL &&
                counterRandom(42, 9, 123456789) == 0x91a0834ef3c62df8ULL,
            "Counter-based generation sequence changed.");
    const int indexedValue = indexedUniformInteger(7, 3, 11, -4, 5);
    require(indexedValue == 5 && indexedUniformInteger(42, 9, 123456789, -100, 100) == -31,
            "Counter-based integer generation sequence changed.");

    const GenerationRecipe binaryGeneration = GenerationRecipe::realOnly(
        GenerationRecipe::candidateSet({.values = {-1.0, 1.0}}), {.seed = 42});
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
        compare(Tensor::fromNative(std::span<const double>(nonFiniteA)),
                Tensor::fromNative(std::span<const double>(nonFiniteB)), nonFiniteOptions);
    require(nonFiniteResult.mismatches == 1,
            "Comparison did not distinguish finite and infinite values.");

    std::array<int, 8> generatedStorage;
    generatedStorage.fill(-1);
    std::vector<std::byte> generatedBytes(sizeof(generatedStorage));
    std::memcpy(generatedBytes.data(), generatedStorage.data(), generatedBytes.size());
    Tensor generated =
        Tensor::fromStorage(ScalarType::Int32, Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1),
                            std::move(generatedBytes));
    generate(generated,
             [](std::span<const size_t> indices) { return 10 * indices[1] + indices[0]; });
    require(generated.loadAs<int>({0, 0}) == 0 && generated.loadAs<int>({1, 0}) == 1 &&
                generated.loadAs<int>({0, 1}) == 10 && generated.loadAs<int>({1, 1}) == 11,
            "Matrix generation produced incorrect logical values.");
    const auto generatedStorageValue = [&generated](size_t index) {
        int value;
        std::memcpy(&value, generated.storage().data() + index * sizeof(int), sizeof(value));
        return value;
    };
    require(generatedStorageValue(0) == -1 && generatedStorageValue(3) == -1 &&
                generatedStorageValue(7) == -1,
            "Matrix generation modified padding.");

    Tensor runtimeExpected(ScalarType::Float32, Shape{2, 3});
    const GenerationRecipe runtimeGeneration = GenerationRecipe::realOnly(
        GenerationRecipe::uniformInteger({.lower = -2, .upper = 2}), {.seed = 7});
    generate(runtimeExpected, runtimeGeneration);
    Tensor runtimeObserved = runtimeExpected.clone();
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
    using namespace roc::host_validation;

    const std::array<float, 0> emptyStorage{};
    const Layout emptyLayout(Shape{0, 3}, {1, 1});
    ComparisonOptions emptyOptions;
    emptyOptions.computePointwiseStatistics = false;
    emptyOptions.computeFrobenius = false;
    emptyOptions.maxReportedMismatches = 0;
    emptyOptions.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    const auto compareEmpty = [&](const ComparisonOptions& options) {
        return compare(Tensor::fromNative(emptyLayout, std::span<const float>(emptyStorage)),
                       Tensor::fromNative(emptyLayout, std::span<const float>(emptyStorage)),
                       options);
    };
    const ComparisonResult emptyResult = compareEmpty(emptyOptions);
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
    ComparisonOptions zeroSelectionStride;
    zeroSelectionStride.selection.stride = 0;
    requireInvalidEmptyOptions(zeroSelectionStride,
                               "Empty comparison accepted a zero selection stride.");

    const std::array<double, 1> evidenceObserved{2.0};
    const std::array<double, 1> evidenceExpected{1.0};
    ComparisonOptions evidenceOnly;
    evidenceOnly.pointwise = false;
    evidenceOnly.computeUlp = true;
    evidenceOnly.ulpType = ScalarType::Float64;
    const ComparisonResult evidenceResult =
        compare(Tensor::fromNative(std::span<const double>(evidenceObserved)),
                Tensor::fromNative(std::span<const double>(evidenceExpected)), evidenceOnly);
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
        compare(Tensor::fromNative(reversedLayout, std::span<const double>(reversedStorage)),
                Tensor::fromNative(reversedLayout, std::span<const double>(reversedStorage)),
                reversedMetrics);
    require(reversedRuntimeResult.compared == 3 && reversedRuntimeResult.frobeniusDifference == 0.0,
            "Negative-stride comparison produced incorrect evidence.");

    const std::array<float, 8> expectedStorage{1.0f, 2.0f, -99.0f, 3.0f, 4.0f, -99.0f, 5.0f, 6.0f};
    auto observedStorage = expectedStorage;
    observedStorage[4] += 0.5f;
    observedStorage[6] += 1.0f;
    const Layout layout(Shape{2, 3}, {1, 3});

    ComparisonOptions selected;
    selected.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    selected.selection.stride = 2;
    selected.computeFrobenius = false;
    const auto selectedResult =
        compare(Tensor::fromNative(layout, std::span<const float>(observedStorage)),
                Tensor::fromNative(layout, std::span<const float>(expectedStorage)), selected);
    require(selectedResult.compared == 3 && selectedResult.mismatches == 1,
            "Selected comparison visited the wrong logical elements.");
    require(selectedResult.reportedMismatches[0].index == 4 &&
                selectedResult.reportedMismatches[0].coordinates == std::vector<size_t>({0, 2}) &&
                selectedResult.reportedMismatches[0].observedOffset == 6,
            "Selected comparison reported the wrong logical location.");
    ComparisonOptions paddedMetrics;
    paddedMetrics.pointwise = false;
    paddedMetrics.computePointwiseStatistics = false;
    paddedMetrics.computeFrobenius = true;
    paddedMetrics.maxReportedMismatches = 0;
    paddedMetrics.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    const auto paddedMetricResult =
        compare(Tensor::fromNative(layout, std::span<const float>(observedStorage)),
                Tensor::fromNative(layout, std::span<const float>(expectedStorage)), paddedMetrics);
    require(paddedMetricResult.compared == 6 && paddedMetricResult.frobeniusDifference > 0.0,
            "Regular strided comparison produced incorrect evidence.");

    const std::array<uint64_t, 1> wideIntegerObserved{uint64_t{1} << 53};
    const std::array<uint64_t, 1> wideIntegerExpected{(uint64_t{1} << 53) + 1};
    ComparisonOptions fastIntegerOptions;
    fastIntegerOptions.computePointwiseStatistics = false;
    fastIntegerOptions.computeFrobenius = false;
    fastIntegerOptions.maxReportedMismatches = 0;
    const auto fastIntegerComparison = compare(
        Tensor::fromNative(std::span<const uint64_t>(wideIntegerObserved)),
        Tensor::fromNative(std::span<const uint64_t>(wideIntegerExpected)), fastIntegerOptions);
    require(!fastIntegerComparison.passed() && fastIntegerComparison.mismatches == 1,
            "Fast comparison rounded distinct wide integers together.");

    fastIntegerOptions.maxReportedMismatches = 1;
    const auto reportedIntegerComparison = compare(
        Tensor::fromNative(std::span<const uint64_t>(wideIntegerObserved)),
        Tensor::fromNative(std::span<const uint64_t>(wideIntegerExpected)), fastIntegerOptions);
    require(!reportedIntegerComparison.passed() && reportedIntegerComparison.mismatches == 1 &&
                reportedIntegerComparison.reportedMismatches.size() == 1 &&
                !reportedIntegerComparison.reportedMismatches[0].matched,
            "Detailed comparison changed the wide-integer pointwise decision.");

    const auto runtimeIntegerComparison =
        compare(Tensor::fromNative(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::fromNative(std::span<const uint64_t>(wideIntegerExpected)));
    require(!runtimeIntegerComparison.passed() && runtimeIntegerComparison.mismatches == 1,
            "Runtime detailed comparison rounded distinct wide integers together.");

    ComparisonOptions subUnitIntegerTolerance;
    subUnitIntegerTolerance.absoluteTolerance = 0.5;
    subUnitIntegerTolerance.computePointwiseStatistics = false;
    subUnitIntegerTolerance.computeFrobenius = false;
    subUnitIntegerTolerance.maxReportedMismatches = 0;
    require(!compare(Tensor::fromNative(std::span<const uint64_t>(wideIntegerObserved)),
                     Tensor::fromNative(std::span<const uint64_t>(wideIntegerExpected)),
                     subUnitIntegerTolerance)
                 .passed(),
            "Runtime comparison lost a sub-unit tolerance at the uint64 precision boundary.");

    const std::array<int64_t, 1> signedIntegerObserved{std::numeric_limits<int64_t>::lowest()};
    const std::array<int64_t, 1> signedIntegerExpected{std::numeric_limits<int64_t>::max()};
    require(!compare(Tensor::fromNative(std::span<const int64_t>(signedIntegerObserved)),
                     Tensor::fromNative(std::span<const int64_t>(signedIntegerExpected)))
                 .passed(),
            "Runtime comparison overflowed the signed-integer decision.");

    const std::array<double, 3> expected{3.0, 4.0, 0.0};
    const std::array<double, 3> observed{0.0, 4.0, 3.0};
    ComparisonOptions metrics;
    metrics.pointwise = false;
    metrics.relativeFrobeniusTolerance = 0.9;
    metrics.computeUlp = true;
    metrics.ulpType = ScalarType::Float64;
    const auto metricResult =
        compare(Tensor::fromNative(std::span<const double>(observed)),
                Tensor::fromNative(std::span<const double>(expected)), metrics);
    require(std::abs(metricResult.frobeniusExpected - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusObserved - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusDifference - std::sqrt(18.0)) < 1e-12 &&
                std::abs(metricResult.relativeFrobeniusError - std::sqrt(18.0) / 5.0) < 1e-12 &&
                !metricResult.pointwiseEvaluated && metricResult.frobeniusEvaluated &&
                !metricResult.ulpEvaluated && metricResult.frobeniusPassed,
            "Comparison Frobenius evidence is incorrect.");
    ComparisonOptions sampledMetrics = metrics;
    sampledMetrics.selection.first = 1;
    sampledMetrics.selection.stride = 2;
    const auto sampledMetricResult =
        compare(Tensor::fromNative(std::span<const double>(observed)),
                Tensor::fromNative(std::span<const double>(expected)), sampledMetrics);
    require(sampledMetricResult.compared == 1,
            "Irregular comparison selection visited the wrong element count.");

    const double oneUlp = std::ldexp(1.0, -52);
    const std::array<double, 1> ulpObserved{1.0 + oneUlp};
    const std::array<double, 1> ulpExpected{1.0};
    ComparisonOptions ulp;
    ulp.computeUlp = true;
    ulp.ulpType = ScalarType::Float64;
    ulp.maximumUlpTolerance = 1.0;
    const auto ulpResult = compare(Tensor::fromNative(std::span<const double>(ulpObserved)),
                                   Tensor::fromNative(std::span<const double>(ulpExpected)), ulp);
    require(ulpResult.maximumUlp == 1.0 && ulpResult.averageUlp == 1.0 &&
                ulpResult.pointwiseEvaluated && !ulpResult.frobeniusEvaluated &&
                ulpResult.ulpEvaluated && ulpResult.ulpPassed,
            "Comparison ULP evidence is incorrect.");
    require(encodedUlpDistance(0.0, static_cast<double>(std::numeric_limits<float>::denorm_min()),
                               ScalarType::Float32) == 1.0,
            "Encoded ULP distance mishandled the F32 zero/subnormal boundary.");
    require(encodedUlpDistance(1.0, 1.5, ScalarType::Float4E2M1) == 1.0,
            "Encoded ULP distance mishandled a packed scalar sign width.");

    const std::array<std::complex<double>, 2> complexExpected{
        std::complex<double>(std::numeric_limits<double>::infinity(), 2.0),
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 4.0)};
    const auto complexObserved = complexExpected;
    ComparisonOptions nonFinite;
    nonFinite.equalNaNs = true;
    const auto nonFiniteResult = compare(
        Tensor::fromNative(std::span<const std::complex<double>>(complexObserved)),
        Tensor::fromNative(std::span<const std::complex<double>>(complexExpected)), nonFinite);
    require(nonFiniteResult.passed() && nonFiniteResult.matchedInfinities == 1 &&
                nonFiniteResult.matchedNaNs == 1,
            "Complex non-finite comparison policy is incorrect.");
    const std::array<double, 1> doubleInfinity{std::numeric_limits<double>::infinity()};
    const std::array<float, 1> floatInfinity{std::numeric_limits<float>::infinity()};
    ComparisonOptions noPointwiseStatistics;
    noPointwiseStatistics.computePointwiseStatistics = false;
    noPointwiseStatistics.computeFrobenius = false;
    const ComparisonResult noPointwiseStatisticsResult =
        compare(Tensor::fromNative(std::span<const double>(doubleInfinity)),
                Tensor::fromNative(std::span<const float>(floatInfinity)), noPointwiseStatistics);
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
    require(compare(Tensor::fromNative(std::span<const double>(lowerValue)),
                    Tensor::fromNative(std::span<const double>(referenceValue)), numpyBoundary)
                .passed(),
            "Allclose rejected the inclusive NumPy boundary.");
    require(!compare(Tensor::fromNative(std::span<const double>(referenceValue)),
                     Tensor::fromNative(std::span<const double>(lowerValue)), numpyBoundary)
                 .passed(),
            "Allclose lost NumPy's expected-reference asymmetry.");
    numpyBoundary.strictTolerance = true;
    require(!compare(Tensor::fromNative(std::span<const double>(lowerValue)),
                     Tensor::fromNative(std::span<const double>(referenceValue)), numpyBoundary)
                 .passed(),
            "Strict tolerance did not preserve the legacy exclusive boundary.");

    const std::array<std::complex<double>, 1> componentwiseObserved{std::complex<double>(1.0, 1.0)};
    const std::array<std::complex<double>, 1> componentwiseExpected{std::complex<double>(0.0, 0.0)};
    ComparisonOptions componentwiseComplex = allCloseComparisonOptions(1.0, 0.0);
    componentwiseComplex.computeFrobenius = false;
    require(
        !compare(Tensor::fromNative(std::span<const std::complex<double>>(componentwiseObserved)),
                 Tensor::fromNative(std::span<const std::complex<double>>(componentwiseExpected)),
                 componentwiseComplex)
             .passed(),
        "Complex allclose did not apply magnitude tolerance.");
    componentwiseComplex.complexPointwiseMode = ComplexPointwiseMode::Componentwise;
    require(
        compare(Tensor::fromNative(std::span<const std::complex<double>>(componentwiseObserved)),
                Tensor::fromNative(std::span<const std::complex<double>>(componentwiseExpected)),
                componentwiseComplex)
            .passed(),
        "Explicit componentwise complex comparison rejected passing components.");

    ComparisonOptions magnitudePointwiseOnly = allCloseComparisonOptions(1.0, 0.0);
    magnitudePointwiseOnly.computePointwiseStatistics = false;
    magnitudePointwiseOnly.computeFrobenius = false;
    magnitudePointwiseOnly.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    require(
        !compare(Tensor::fromNative(std::span<const std::complex<double>>(componentwiseObserved)),
                 Tensor::fromNative(std::span<const std::complex<double>>(componentwiseExpected)),
                 magnitudePointwiseOnly)
             .passed(),
        "The optimized pointwise-only path ignored complex magnitude mode.");

    const std::array<std::complex<double>, 1> magnitudeBoundaryObserved{
        std::complex<double>(0.0, 0.0)};
    const std::array<std::complex<double>, 1> magnitudeBoundaryExpected{
        std::complex<double>(3.0, 4.0)};
    ComparisonOptions magnitudeBoundary = allCloseComparisonOptions(0.0, 1.0);
    magnitudeBoundary.computeFrobenius = false;
    require(
        compare(
            Tensor::fromNative(std::span<const std::complex<double>>(magnitudeBoundaryObserved)),
            Tensor::fromNative(std::span<const std::complex<double>>(magnitudeBoundaryExpected)),
            magnitudeBoundary)
            .passed(),
        "Complex allclose rejected its inclusive magnitude boundary.");
    require(
        !compare(
             Tensor::fromNative(std::span<const std::complex<double>>(magnitudeBoundaryExpected)),
             Tensor::fromNative(std::span<const std::complex<double>>(magnitudeBoundaryObserved)),
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
    const ComparisonResult crossComponentNaNResult = compare(
        Tensor::fromNative(std::span<const std::complex<double>>(crossComponentNaNObserved)),
        Tensor::fromNative(std::span<const std::complex<double>>(crossComponentNaNExpected)),
        magnitudeEqualNaNs);
    require(crossComponentNaNResult.passed() && crossComponentNaNResult.matchedNaNs == 1,
            "Complex magnitude comparison did not match logical NaN values.");

    const double infinity = std::numeric_limits<double>::infinity();
    const std::array<std::complex<double>, 1> complexInfinity{
        std::complex<double>(infinity, infinity)};
    const ComparisonResult complexInfinityResult =
        compare(Tensor::fromNative(std::span<const std::complex<double>>(complexInfinity)),
                Tensor::fromNative(std::span<const std::complex<double>>(complexInfinity)),
                magnitudeEqualNaNs);
    require(complexInfinityResult.passed() && complexInfinityResult.matchedInfinities == 1,
            "Complex magnitude comparison did not count a matched infinity as one logical value.");
    ComparisonOptions magnitudeInfinityWithFrobenius = allCloseComparisonOptions(0.0, 0.0, true);
    const ComparisonResult complexInfinityWithFrobeniusResult =
        compare(Tensor::fromNative(std::span<const std::complex<double>>(complexInfinity)),
                Tensor::fromNative(std::span<const std::complex<double>>(complexInfinity)),
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
    require(!compare(Tensor::fromNative(
                         std::span<const std::complex<double>>(mismatchedComplexInfinityObserved)),
                     Tensor::fromNative(
                         std::span<const std::complex<double>>(mismatchedComplexInfinityExpected)),
                     permissiveComplexInfinity)
                 .passed(),
            "Complex magnitude comparison applied finite tolerances to unequal infinities.");

    const std::array<double, 1> mixedReal{3.0};
    const std::array<std::complex<double>, 1> mixedComplex{std::complex<double>(3.0, 4.0)};
    ComparisonOptions mixedMagnitude = allCloseComparisonOptions(0.0, 1.0);
    mixedMagnitude.computeFrobenius = false;
    require(compare(Tensor::fromNative(std::span<const double>(mixedReal)),
                    Tensor::fromNative(std::span<const std::complex<double>>(mixedComplex)),
                    mixedMagnitude)
                .passed(),
            "Mixed real/complex comparison did not scale tolerance by the complex reference.");
    require(!compare(Tensor::fromNative(std::span<const std::complex<double>>(mixedComplex)),
                     Tensor::fromNative(std::span<const double>(mixedReal)), mixedMagnitude)
                 .passed(),
            "Mixed real/complex comparison lost expected-magnitude asymmetry.");

    const std::array<std::complex<double>, 1> signedZeroNaNObserved{
        std::complex<double>(quietNaN, 0.0)};
    const std::array<std::complex<double>, 1> signedZeroNaNExpected{
        std::complex<double>(quietNaN, -0.0)};
    ComparisonOptions signedZeroNaN = allCloseComparisonOptions(0.0, 0.0, true);
    signedZeroNaN.equalSignedZero = false;
    signedZeroNaN.computeFrobenius = false;
    const ComparisonResult signedZeroNaNResult =
        compare(Tensor::fromNative(std::span<const std::complex<double>>(signedZeroNaNObserved)),
                Tensor::fromNative(std::span<const std::complex<double>>(signedZeroNaNExpected)),
                signedZeroNaN);
    require(signedZeroNaNResult.passed() && signedZeroNaNResult.matchedNaNs == 1 &&
                signedZeroNaNResult.signedZeroMismatches == 0,
            "Logical complex NaN matching did not precede signed-zero classification.");

    const std::array<double, 4> absoluteCandidates{1e-6, 1e-5, 1e-4, 1e-3};
    const std::array<double, 1> relativeCandidates{0.0};
    const std::array<double, 1> closeObserved{1.00009};
    const std::array<double, 1> closeExpected{1.0};
    const auto tolerance = findAllCloseTolerance(
        Tensor::fromNative(std::span<const double>(closeObserved)),
        Tensor::fromNative(std::span<const double>(closeExpected)),
        std::span<const double>(absoluteCandidates), std::span<const double>(relativeCandidates));
    require(tolerance && tolerance->absolute == 1e-4 && tolerance->relative == 0.0,
            "Allclose tolerance search selected the wrong candidate.");

    const std::array<std::complex<double>, 1> complexSearchObserved{
        std::complex<double>(0.09, 0.09)};
    const std::array<std::complex<double>, 1> complexSearchExpected{std::complex<double>(0.0, 0.0)};
    const std::array<double, 1> complexSearchAbsoluteCandidates{0.1};
    const std::array<double, 1> complexSearchRelativeCandidates{0.0};
    require(!findAllCloseTolerance(
                Tensor::fromNative(std::span<const std::complex<double>>(complexSearchObserved)),
                Tensor::fromNative(std::span<const std::complex<double>>(complexSearchExpected)),
                std::span<const double>(complexSearchAbsoluteCandidates),
                std::span<const double>(complexSearchRelativeCandidates)),
            "Allclose tolerance search defaulted complex values to componentwise comparison.");
    ComparisonOptions componentwiseSearch = allCloseComparisonOptions();
    componentwiseSearch.complexPointwiseMode = ComplexPointwiseMode::Componentwise;
    require(findAllCloseTolerance(
                Tensor::fromNative(std::span<const std::complex<double>>(complexSearchObserved)),
                Tensor::fromNative(std::span<const std::complex<double>>(complexSearchExpected)),
                std::span<const double>(complexSearchAbsoluteCandidates),
                std::span<const double>(complexSearchRelativeCandidates), componentwiseSearch)
                .has_value(),
            "Allclose tolerance search did not preserve explicit componentwise comparison.");

    std::array<float, 5> rangedSentinel{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 0.0f, std::numeric_limits<float>::infinity()};
    const SentinelResult rangedSentinelResult = checkUnwrittenSentinel(
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
    Tensor guardedView =
        Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 3}), std::span<const float>(guarded));
    require(checkUnusedTensorStorage(guardedView, guarded.size()).passed(),
            "Unwritten tensor padding sentinel was rejected.");
    const float writtenPadding = 0.0f;
    std::memcpy(guardedView.storage().data() + 2 * sizeof(float), &writtenPadding,
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
    testRuntimeMixedAndBlockScaledGemm();
    testPointwiseRoutes();
    testExactIntegerGemm();
    testRuntimeComplexAndExplicitAxisGemm();
    testOutputSelection();
    testReferenceEpilogue();
    testReferenceReduction();
    testStructuredSparsity();
    testIndexedGeneration();
    testReferenceAxpby();
    testReferenceSoftmax();
    testReferenceLayerNorm();
    testActivations();
    testStridedAndOffsetViews();
    testGenerationAndComparison();
    testComparisonProgram();
    return 0;
}
