// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <roc/host_validation/validation.hpp>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void testRuntimeReferenceGemm() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    std::array<float, 4> d{};
    const std::array<float, 2> bias{1, -10000};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    GemmProblem problem(
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a))),
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b))),
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<float>(d)),
        ScalarType::Float32);
    problem.epilogue.beta = 1.0;
    problem.epilogue.bias = VectorBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.epilogue.scaleA =
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(scaleA));
    problem.epilogue.scaleB =
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(scaleB));
    problem.epilogue.activation = Activation::Relu;

    const GemmRunInfo run = referenceGemm(problem);
    require(run.backendUsed == GemmBackend::Canonical && run.outputElementsComputed == 4,
            "Runtime reference GEMM run information mismatch.");

    const std::array<float, 4> expected{
        58 * 2 * 5 + 1 + 1,
        0,
        64 * 2 * 7 + 1 + 1,
        0,
    };
    require(compare(std::span<const float>(d), std::span<const float>(expected)).passed(),
            "Runtime reference GEMM result mismatch.");

    const GemmRunInfo fallback = referenceGemm(problem, {.backend = GemmBackend::Tiled});
    require(fallback.backendUsed == GemmBackend::Canonical && fallback.fallbackReason.has_value(),
            "Runtime reference GEMM backend fallback mismatch.");
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

    GemmOperand operandA(a.view());
    operandA.computeType = ScalarType::Float4E2M1;
    GemmProblem mixed(std::move(operandA), GemmOperand(b.view()), c.view(), d.mutableView(),
                      ScalarType::Float32);
    mixed.epilogue.beta = 1.0;
    referenceGemm(mixed);
    require(d.view().loadAs<float>({0, 0}) == 9.0f, "Runtime mixed-type GEMM result mismatch.");

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

    GemmOperand blockOperandA(blockA.view());
    GemmOperand blockOperandB(blockB.view());
    blockOperandA.blockScale = BlockScaleBinding{scalesA.view(), 2};
    blockOperandB.blockScale = BlockScaleBinding{scalesB.view(), 2};
    GemmProblem blockScaled(std::move(blockOperandA), std::move(blockOperandB), blockC.view(),
                            blockD.mutableView(), ScalarType::Float32);
    referenceGemm(blockScaled);
    require(blockD.view().loadAs<float>({0, 0}) == 2 * 2 * 8 + 2 * 4 * 16,
            "Runtime block-scaled GEMM result mismatch.");
}

void testRuntimeComplexAndExplicitAxisGemm() {
    using namespace roc::host_validation;

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1.0f, 2.0f)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3.0f, 4.0f)};
    const std::array<std::complex<float>, 1> complexC{};
    std::array<std::complex<float>, 1> complexD{};

    GemmOperand complexOperandA(TensorView::fromNative<std::complex<float>>(
        Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexA)));
    complexOperandA.conjugate = true;
    GemmProblem complexProblem(
        std::move(complexOperandA),
        GemmOperand(TensorView::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexB))),
        TensorView::fromNative<std::complex<float>>(Layout::contiguous(Shape{1, 1}),
                                                    std::span<const std::complex<float>>(complexC)),
        MutableTensorView::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<std::complex<float>>(complexD)),
        ScalarType::ComplexFloat32);
    referenceGemm(complexProblem);
    require(complexD[0] == std::complex<float>(11.0f, -2.0f),
            "Runtime complex GEMM result mismatch.");

    const std::array<float, 1> realA{1};
    const std::array<float, 2> realB{0, 0};
    const std::array<float, 2> realC{0, 0};
    const std::array<float, 2> columnBias{2, 3};
    std::array<float, 2> realD{};
    GemmProblem axisProblem(GemmOperand(TensorView::fromNative<float>(
                                Layout::contiguous(Shape{1, 1}), std::span<const float>(realA))),
                            GemmOperand(TensorView::fromNative<float>(
                                Layout::contiguous(Shape{1, 2}), std::span<const float>(realB))),
                            TensorView::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                                          std::span<const float>(realC)),
                            MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                                                 std::span<float>(realD)),
                            ScalarType::Float32);
    axisProblem.epilogue.bias = VectorBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}),
                                      std::span<const float>(columnBias)),
        MatrixAxis::Column,
    };
    referenceGemm(axisProblem);
    require(realD == columnBias, "Runtime GEMM explicit column-axis bias mismatch.");
}

void testOutputSelection() {
    using namespace roc::host_validation;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{};
    std::array<float, 4> d{-99, -99, -99, -99};

    GemmProblem problem(
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                  std::span<const float>(a))),
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}),
                                                  std::span<const float>(b))),
        TensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{2, 2}), std::span<float>(d)),
        ScalarType::Float32);
    problem.outputSelection = OutputSelection::explicitIndices({0, 3});
    const GemmRunInfo run = referenceGemm(problem);
    require(run.outputElementsComputed == 2,
            "Selected-output GEMM reported the wrong element count.");
    require(d[0] == 19 && d[1] == -99 && d[2] == -99 && d[3] == 50,
            "Selected-output GEMM modified the wrong elements.");

    const auto prime = OutputSelection::primeStride(10, 10, 3).indices(10);
    require(prime == std::vector<size_t>({0, 3, 6, 9}), "Prime-stride output selection mismatch.");
    require(OutputSelection::primeStride(10, 10, 0).selectsAll(),
            "Zero requested elements did not preserve all-output behavior.");
}

void testActivations() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{1};
    const std::array<float, 1> c{0};
    std::array<float, 1> d{};

    GemmProblem problem(
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                  std::span<const float>(a))),
        GemmOperand(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                  std::span<const float>(b))),
        TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}), std::span<float>(d)),
        ScalarType::Float32);

    problem.epilogue.activation = Activation::Gelu;
    referenceGemm(problem);
    require(std::abs(d[0] - 1.9545977f) < 1e-6f, "GELU result mismatch.");

    problem.epilogue.activation = Activation::Silu;
    problem.epilogue.activationParameter0 = 1;
    referenceGemm(problem);
    require(std::abs(d[0] - 1.7615942f) < 1e-6f, "SiLU result mismatch.");

    problem.epilogue.activation = Activation::Clamp;
    problem.epilogue.activationParameter0 = -1;
    problem.epilogue.activationParameter1 = 1;
    referenceGemm(problem);
    require(d[0] == 1, "Clamp result mismatch.");
}

void testStridedAndOffsetViews() {
    using namespace roc::host_validation;

    // Logical A and B are the same matrices as testReferenceGemm, but both
    // are stored transposed with padded leading dimensions. C and D use
    // different padding, and D begins at an adjusted base pointer.
    const std::array<float, 8> a{1, 2, 3, -1, 4, 5, 6, -1};
    const std::array<float, 9> b{7, 8, -1, 9, 10, -1, 11, 12, -1};
    const std::array<float, 8> c{1, 1, -1, -1, 1, 1, -1, -1};
    std::array<float, 12> d;
    d.fill(-99);

    const Layout outputLayout(Shape{2, 2}, {1, 5}, 1);
    GemmProblem problem(
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{2, 3}, {4, 1}), std::span<const float>(a))),
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{3, 2}, {3, 1}), std::span<const float>(b))),
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 4}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(outputLayout, std::span<float>(d)),
        ScalarType::Float32);
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
        compare(TensorView::fromNative<float>(outputLayout, std::span<const float>(d)),
                TensorView::fromNative<float>(outputLayout, std::span<const float>(expected)));
    require(comparison.passed(), "Strided GEMM matrix comparison failed.");
    require(d[0] == -99 && d[3] == -99 && d[11] == -99, "Strided GEMM modified padding.");
}

void testGenerationAndComparison() {
    using namespace roc::host_validation;

    require(counterRandom(7, 3, 11) == counterRandom(7, 3, 11),
            "Counter-based generation is not deterministic.");
    require(counterRandom(7, 3, 11) != counterRandom(7, 3, 12),
            "Counter-based generation does not vary by logical index.");
    const int indexedValue = indexedUniformInteger(7, 3, 11, -4, 5);
    require(indexedValue >= -4 && indexedValue <= 5,
            "Counter-based integer generation exceeded its bounds.");

    RandomGenerator generatorA(42);
    RandomGenerator generatorB(42);
    std::array<float, 32> a{};
    std::array<float, 32> b{};
    generatorA.fillBinary<float>(a);
    generatorB.fillBinary<float>(b);
    require(a == b, "Random generation is not repeatable for equal seeds.");

    b[7] += 1;
    const auto result =
        compare(std::span<const float>(b), std::span<const float>(a),
                {.absoluteTolerance = 0.0, .relativeTolerance = 0.0, .maxReportedMismatches = 4});
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
    const auto nonFiniteResult =
        compare(std::span<const double>(nonFiniteA), std::span<const double>(nonFiniteB),
                {.relativeTolerance = 1.0});
    require(nonFiniteResult.mismatches == 1,
            "Comparison did not distinguish finite and infinite values.");

    std::array<int, 8> generated;
    generated.fill(-1);
    generate(MatrixView<int>(generated.data() + 1, 2, 2, 1, 3),
             [](size_t row, size_t column) { return 10 * column + row; });
    require(generated[1] == 0 && generated[2] == 1 && generated[4] == 10 && generated[5] == 11,
            "Matrix generation produced incorrect logical values.");
    require(generated[0] == -1 && generated[3] == -1 && generated[7] == -1,
            "Matrix generation modified padding.");

    Tensor runtimeExpected(ScalarType::Float32, Shape{2, 3});
    RandomGenerator runtimeGenerator(7);
    fill(runtimeExpected.mutableView(), DataPattern::UniformInteger, runtimeGenerator, -2, 2);
    Tensor runtimeObserved = runtimeExpected;
    runtimeObserved.mutableView().storeFrom({1, 2},
                                            runtimeExpected.view().loadAs<float>({1, 2}) + 1.0f);
    const auto runtimeComparison =
        compare(runtimeObserved.view(), runtimeExpected.view(),
                {.absoluteTolerance = 0.0, .maxReportedMismatches = 2});
    require(runtimeComparison.compared == 6 && runtimeComparison.mismatches == 1 &&
                runtimeComparison.reportedMismatches[0].index == 5,
            "Runtime tensor generation/comparison mismatch.");
}
}  // namespace

int main() {
    testRuntimeReferenceGemm();
    testRuntimeMixedAndBlockScaledGemm();
    testRuntimeComplexAndExplicitAxisGemm();
    testOutputSelection();
    testActivations();
    testStridedAndOffsetViews();
    testGenerationAndComparison();
    return 0;
}
