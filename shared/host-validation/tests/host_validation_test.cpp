// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <cmath>
#include <limits>
#include <roc/host_validation/validation.hpp>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void testReferenceGemm() {
    using namespace roc::host_validation;

    // Column-major A(2x3), B(3x2), C/D(2x2).
    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    std::array<float, 4> d{};
    const std::array<float, 2> bias{1, -10000};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    GemmInvocation<float, float, float, float, float> invocation{
        ConstMatrixView<float>(a.data(), 2, 3, 1, 2), ConstMatrixView<float>(b.data(), 3, 2, 1, 3),
        ConstMatrixView<float>(c.data(), 2, 2, 1, 2), MatrixView<float>(d.data(), 2, 2, 1, 2)};
    invocation.alpha = 1;
    invocation.beta = 1;
    invocation.bias = ConstVectorView<float>(bias.data(), bias.size());
    invocation.scaleA = ConstVectorView<float>(scaleA.data(), scaleA.size());
    invocation.scaleB = ConstVectorView<float>(scaleB.data(), scaleB.size());
    invocation.activation = Activation::Relu;

    referenceGemm(invocation);

    // Unscaled AB is [[58,64],[139,154]].
    const std::array<float, 4> expected{58 * 2 * 5 + 1 + 1, 0, 64 * 2 * 7 + 1 + 1, 0};
    const auto comparison = compare(std::span<const float>(d), std::span<const float>(expected));
    require(comparison.passed(), "Reference GEMM result mismatch.");
}

void testBlockScale() {
    using namespace roc::host_validation;

    const std::array<float, 4> a{1, 1, 1, 1};
    const std::array<float, 4> b{1, 1, 1, 1};
    const std::array<float, 1> c{0};
    std::array<float, 1> d{};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    GemmInvocation<float, float, float, float, float> invocation{
        ConstMatrixView<float>(a.data(), 1, 4, 1, 1), ConstMatrixView<float>(b.data(), 4, 1, 1, 1),
        ConstMatrixView<float>(c.data(), 1, 1, 1, 1), MatrixView<float>(d.data(), 1, 1, 1, 1)};
    invocation.blockScaleA = makeBlockScaleView<float>(scaleA.data(), 2, 2, 1);
    invocation.blockScaleB = makeBlockScaleView<float>(scaleB.data(), 2, 2, 1);

    referenceGemm(invocation);
    require(d[0] == 2 * 2 * 5 + 2 * 3 * 7, "Block-scaled GEMM result mismatch.");
}

void testActivations() {
    using namespace roc::host_validation;

    const std::array<float, 1> a{2};
    const std::array<float, 1> b{1};
    const std::array<float, 1> c{0};
    std::array<float, 1> d{};

    GemmInvocation<float, float, float, float, float> invocation{
        ConstMatrixView<float>(a.data(), 1, 1, 1, 1), ConstMatrixView<float>(b.data(), 1, 1, 1, 1),
        ConstMatrixView<float>(c.data(), 1, 1, 1, 1), MatrixView<float>(d.data(), 1, 1, 1, 1)};

    invocation.activation = Activation::Gelu;
    referenceGemm(invocation);
    require(std::abs(d[0] - 1.9545977f) < 1e-6f, "GELU result mismatch.");

    invocation.activation = Activation::Silu;
    invocation.activationParameter0 = 1;
    referenceGemm(invocation);
    require(std::abs(d[0] - 1.7615942f) < 1e-6f, "SiLU result mismatch.");

    invocation.activation = Activation::Clamp;
    invocation.activationParameter0 = -1;
    invocation.activationParameter1 = 1;
    referenceGemm(invocation);
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

    GemmInvocation<float, float, float, float, float> invocation{
        ConstMatrixView<float>(a.data(), 2, 3, 4, 1), ConstMatrixView<float>(b.data(), 3, 2, 3, 1),
        ConstMatrixView<float>(c.data(), 2, 2, 1, 4), MatrixView<float>(d.data() + 1, 2, 2, 1, 5)};
    invocation.alpha = 2;
    invocation.beta = 3;

    referenceGemm(invocation);

    std::array<float, 12> expected;
    expected.fill(-99);
    expected[1] = 2 * 58 + 3;
    expected[2] = 2 * 139 + 3;
    expected[6] = 2 * 64 + 3;
    expected[7] = 2 * 154 + 3;
    const auto comparison = compare(MatrixView<float>(d.data() + 1, 2, 2, 1, 5).asConst(),
                                    ConstMatrixView<float>(expected.data() + 1, 2, 2, 1, 5));
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
    testReferenceGemm();
    testBlockScale();
    testActivations();
    testStridedAndOffsetViews();
    testGenerationAndComparison();
    return 0;
}
