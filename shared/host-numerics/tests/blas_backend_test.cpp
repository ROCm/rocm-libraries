// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>
#include <cstdint>
#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/comparison.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void testTransformingBlockScaleFallsBack(roc::host_numerics::ScalarType accumulatorType) {
    using namespace roc::host_numerics;

    const std::array<T, 8> a{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<T, 8> b{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<T, 4> c{};
    const std::array<uint8_t, 4> scaleA{128, 129, 130, 131};
    const std::array<uint8_t, 4> scaleB{127, 128, 129, 130};
    const Layout layoutA(Shape{2, 4}, {1, 2});
    const Layout layoutB(Shape{4, 2}, {1, 4});
    const Layout layoutD(Shape{2, 2}, {1, 2});
    const Layout scaleLayout = Layout::contiguousLastDimensionFastest(Shape{2, 2});
    Tensor pointwiseD(nativeScalarType<T>, layoutD);
    Tensor transformingD(nativeScalarType<T>, layoutD);

    auto makeOperandA = [&]() {
        GemmOperand operand(Tensor::copyNativeStorage<T>(layoutA, std::span<const T>(a)));
        operand.blockScale = BlockScaleBinding{
            Tensor::copyEncodedBackingStorage(ScalarType::E8M0, scaleLayout,
                                              std::as_bytes(std::span(scaleA))),
            2,
        };
        return operand;
    };
    auto makeOperandB = [&]() {
        GemmOperand operand(Tensor::copyNativeStorage<T>(layoutB, std::span<const T>(b)));
        operand.blockScale = BlockScaleBinding{
            Tensor::copyEncodedBackingStorage(ScalarType::E8M0, scaleLayout,
                                              std::as_bytes(std::span(scaleB))),
            2,
        };
        return operand;
    };

    GemmRequest pointwiseProblem(makeOperandA(), makeOperandB(),
                                 Tensor::copyNativeStorage<T>(layoutD, std::span<const T>(c)),
                                 pointwiseD, accumulatorType);
    referenceGemm(pointwiseProblem);

    GemmRequest transformingProblem(makeOperandA(), makeOperandB(),
                                    Tensor::copyNativeStorage<T>(layoutD, std::span<const T>(c)),
                                    transformingD, accumulatorType);
    const GemmSupportInfo support =
        queryGemmSupportWithBlasBackend(transformingProblem, GemmBackend::Blas);
    require(!support.supported &&
                support.reason ==
                    "Transforming BLAS backend cannot preserve block-scale reduction boundaries.",
            "Transforming BLAS did not reject block scaling with a precise reason.");
    const GemmRunInfo runInfo = referenceGemmWithBlasBackend(transformingProblem);
    require(runInfo.backendUsed == GemmBackend::Pointwise && runInfo.fallbackReason.has_value(),
            "Transforming BLAS block scaling did not fall back to Pointwise.");

    const std::array<T, 4> expected{20, 80, 80, 320};
    const Tensor expectedTensor =
        Tensor::copyNativeStorage<T>(layoutD, std::span<const T>(expected));
    require(compare(pointwiseD, expectedTensor).passed(),
            "Pointwise block-scale reference mismatch.");
    require(compare(transformingD, pointwiseD).passed(),
            "Transforming BLAS block-scale fallback differs from pointwise reference.");
}

void testPartialOutputSelection() {
    using namespace roc::host_numerics;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    constexpr std::array<float, 4> untouched{-99, -99, -99, -99};
    Tensor d = Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 2}),
                                                std::span<const float>(untouched));
    GemmRequest problem(GemmOperand(Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}),
                                                                     std::span<const float>(a))),
                        GemmOperand(Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}),
                                                                     std::span<const float>(b))),
                        d, d, ScalarType::Float32);
    problem.outputSelection = OutputSelection::explicitIndices({0, 3});

    const GemmSupportInfo support = queryGemmSupportWithBlasBackend(problem, GemmBackend::Blas);
    require(!support.supported &&
                support.reason == "Transforming BLAS backend requires complete output selection.",
            "BLAS backend support query accepted partial output selection.");

    bool rejectedRequiredBackend = false;
    try {
        referenceGemmWithBlasBackend(problem, GemmBackend::Blas);
    } catch (const std::invalid_argument& error) {
        rejectedRequiredBackend = support.reason == error.what();
    }
    require(rejectedRequiredBackend, "Required BLAS execution accepted partial output selection.");
    require(
        compare(d, Tensor::copyNativeStorage<float>(d.layout(), std::span<const float>(untouched)))
            .passed(),
        "Rejected BLAS execution modified output.");

    const GemmRunInfo fallback = referenceGemmWithBlasBackend(problem);
    require(fallback.backendUsed == GemmBackend::Pointwise &&
                fallback.fallbackReason == support.reason && fallback.outputElementsWritten == 2 &&
                fallback.outputElementsCovered == 2,
            "Partial-output BLAS request did not report pointwise fallback.");
    require(d.loadAs<float>({0, 0}) == 58 && d.loadAs<float>({1, 0}) == -99 &&
                d.loadAs<float>({0, 1}) == -99 && d.loadAs<float>({1, 1}) == 154,
            "Pointwise BLAS fallback did not preserve unselected outputs.");
}

void testTransformingAutomaticCostPolicy() {
    using namespace roc::host_numerics;

    auto makeProblem = [](ScalarType inputType, size_t rows, size_t columns, size_t reductions) {
        return GemmRequest(GemmOperand(Tensor(inputType, Shape{rows, reductions})),
                           GemmOperand(Tensor(inputType, Shape{reductions, columns})),
                           Tensor(ScalarType::Float32, Shape{rows, columns}),
                           Tensor(ScalarType::Float32, Shape{rows, columns}), ScalarType::Float32);
    };

    const GemmSupportInfo dense = queryGemmSupportWithBlasBackend(
        makeProblem(ScalarType::Float16, 128, 128, 128), GemmBackend::Blas);
    require(dense.supported && dense.preferredForAutomaticExecution,
            "Automatic GEMM did not prefer BLAS for a compute-dense transformed request.");

    const GemmSupportInfo skinny = queryGemmSupportWithBlasBackend(
        makeProblem(ScalarType::Float16, 1, 128, 8192), GemmBackend::Blas);
    require(skinny.supported && !skinny.preferredForAutomaticExecution,
            "Automatic GEMM preferred transformed BLAS for a staging-dominated skinny request.");

    const GemmSupportInfo reusableSkinny = queryGemmSupportWithBlasBackend(
        makeProblem(ScalarType::Float32, 1, 128, 8192), GemmBackend::Blas);
    require(reusableSkinny.supported && reusableSkinny.preferredForAutomaticExecution,
            "Automatic GEMM ignored reusable BLAS inputs for a skinny request.");
}

void testModeratelyLargeExactGemm() {
    using namespace roc::host_numerics;

    constexpr size_t dimension = 256;
    const Layout layout(Shape{dimension, dimension}, {1, static_cast<ptrdiff_t>(dimension)});
    const std::vector<float> ones(dimension * dimension, 1.0f);
    Tensor output(ScalarType::Float32, layout);

    GemmRequest problem(
        GemmOperand(Tensor::copyNativeStorage<float>(layout, std::span<const float>(ones))),
        GemmOperand(Tensor::copyNativeStorage<float>(layout, std::span<const float>(ones))), output,
        output, ScalarType::Float32);
    referenceGemmWithBlasBackend(problem, GemmBackend::Blas);

    for (size_t row = 0; row < dimension; ++row)
        for (size_t column = 0; column < dimension; ++column)
            require(output.loadAs<float>({row, column}) == static_cast<float>(dimension),
                    "Moderately large BLAS GEMM result mismatch.");
}
}  // namespace

int main() {
    using namespace roc::host_numerics;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    Tensor d = Tensor::copyNativeStorage<float>(
        Layout(Shape{2, 2}, {1, 2}), std::span<const float>(std::array<float, 4>{1, 1, 1, 1}));
    GemmRequest problem(GemmOperand(Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}),
                                                                     std::span<const float>(a))),
                        GemmOperand(Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}),
                                                                     std::span<const float>(b))),
                        d, d, ScalarType::Float32);
    problem.epilogue.alpha = 2.0;
    problem.epilogue.beta = 3.0;

    require(queryGemmSupportWithBlasBackend(problem, GemmBackend::Blas).supported,
            "BLAS backend unexpectedly rejected F32 GEMM.");
    const GemmRunInfo runInfo = referenceGemmWithBlasBackend(problem, GemmBackend::Blas);
    require(runInfo.backendUsed == GemmBackend::Blas, "BLAS backend run information mismatch.");
    require(d.loadAs<float>({0, 0}) == 119 && d.loadAs<float>({1, 0}) == 281 &&
                d.loadAs<float>({0, 1}) == 131 && d.loadAs<float>({1, 1}) == 311,
            "BLAS backend F32 result mismatch.");

    const Tensor ones = Tensor::copyNativeStorage<float>(
        d.layout(), std::span<const float>(std::array<float, 4>{1, 1, 1, 1}));
    d.copyLogicalElementsFrom(ones);
    const GemmRunInfo automatic = referenceGemmWithBlasBackend(problem);
    require(automatic.backendUsed == GemmBackend::Blas && d.loadAs<float>({0, 0}) == 119 &&
                d.loadAs<float>({1, 0}) == 281 && d.loadAs<float>({0, 1}) == 131 &&
                d.loadAs<float>({1, 1}) == 311,
            "Automatic runtime backend selection mismatch.");

    d.copyLogicalElementsFrom(ones);
    problem.epilogue.activation = Activation::Relu;
    const GemmRunInfo fallback = referenceGemmWithBlasBackend(problem);
    require(fallback.backendUsed == GemmBackend::Pointwise && fallback.fallbackReason.has_value() &&
                d.loadAs<float>({0, 0}) == 119 && d.loadAs<float>({1, 0}) == 281 &&
                d.loadAs<float>({0, 1}) == 131 && d.loadAs<float>({1, 1}) == 311,
            "Automatic runtime backend fallback mismatch.");
    problem.epilogue.activation = Activation::None;

    testPartialOutputSelection();
    testTransformingAutomaticCostPolicy();
    testModeratelyLargeExactGemm();

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3, 4)};
    Tensor complexD(ScalarType::ComplexFloat32, Shape{1, 1});
    GemmOperand operandA(Tensor::copyNativeStorage<std::complex<float>>(
        Layout(Shape{1, 1}, {2, 1}), std::span<const std::complex<float>>(complexA)));
    operandA.conjugate = true;
    GemmRequest complexProblem(std::move(operandA),
                               GemmOperand(Tensor::copyNativeStorage<std::complex<float>>(
                                   Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                   std::span<const std::complex<float>>(complexB))),
                               complexD, complexD, ScalarType::ComplexFloat32);
    referenceGemmWithBlasBackend(complexProblem, GemmBackend::Blas);
    require(complexD.loadAs<std::complex<float>>({0, 0}) == std::complex<float>(11, -2),
            "BLAS backend complex result mismatch.");

    const std::array<float, 1> transformedA{0.3f};
    const std::array<float, 1> transformedB{1.0f};
    const std::array<float, 1> transformedC{1.0f};
    const std::array<float, 1> transformedScaleA{0.7f};
    const std::array<float, 1> transformedAlphaVector{0.6f};
    Tensor transformedD(ScalarType::Float32, Shape{1, 1});
    GemmOperand transformedOperandA(Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const float>(transformedA)));
    transformedOperandA.computeType = ScalarType::Float8E4M3;
    transformedOperandA.preQuantizationScales.push_back(VectorBinding{
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1}),
                                         std::span<const float>(transformedScaleA)),
        MatrixAxis::Row});
    transformedOperandA.preQuantizationScales.push_back(VectorBinding{
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1}),
                                         std::span<const float>(transformedAlphaVector)),
        MatrixAxis::Row});
    GemmRequest transformedProblem(
        std::move(transformedOperandA),
        GemmOperand(
            Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                             std::span<const float>(transformedB))),
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                         std::span<const float>(transformedC)),
        transformedD, ScalarType::Float32);
    transformedProblem.epilogue.alpha = 2.0;
    transformedProblem.epilogue.beta = 3.0;
    transformedProblem.epilogue.scaleC = 2.0;
    transformedProblem.epilogue.outputScale = 4.0;
    referenceGemmWithBlasBackend(transformedProblem, GemmBackend::Blas);
    require(transformedD.loadAs<float>({0, 0}) == 25.0f,
            "Transforming BLAS pre-quantization/finalization result mismatch.");

    const std::array<float, 1> scalarScaleA{2.0f};
    const std::array<float, 1> scalarScaleB{3.0f};
    transformedProblem.epilogue.scaleA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1}), std::span<const float>(scalarScaleA));
    transformedProblem.epilogue.scaleB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1}), std::span<const float>(scalarScaleB));
    referenceGemmWithBlasBackend(transformedProblem, GemmBackend::Blas);
    require(transformedD.loadAs<float>({0, 0}) == 30.0f,
            "Transforming BLAS scalar A/B scale result mismatch.");
    const float transformedBlasResult = transformedD.loadAs<float>({0, 0});
    transformedD.storeFrom({0, 0}, 0.0f);
    referenceGemm(transformedProblem, GemmBackend::Pointwise);
    require(transformedD.loadAs<float>({0, 0}) == transformedBlasResult,
            "Transforming BLAS scalar A/B scales differ from Pointwise.");
    transformedProblem.epilogue.scaleA.reset();
    transformedProblem.epilogue.scaleB.reset();

    transformedD.storeFrom({0, 0}, 0.0f);
    const GemmRunInfo smallAutomatic = referenceGemmWithBlasBackend(transformedProblem);
    require(smallAutomatic.backendUsed == GemmBackend::Pointwise &&
                transformedD.loadAs<float>({0, 0}) == 25.0f,
            "Automatic GEMM did not avoid staging a tiny transformed request.");

    transformedProblem.a.values.storeFrom({0, 0}, std::numeric_limits<float>::quiet_NaN());
    transformedProblem.epilogue.alpha = 0.0;
    transformedProblem.epilogue.beta = 1.0;
    transformedProblem.epilogue.scaleC = 3.0;
    transformedProblem.epilogue.outputScale = 1.0;
    referenceGemmWithBlasBackend(transformedProblem, GemmBackend::Blas);
    require(transformedD.loadAs<float>({0, 0}) == 3.0f,
            "Transforming BLAS did not suppress an unused non-finite product.");

    const std::array<float, 1> saturatingA{63.75f};
    const std::array<float, 1> saturatingB{2.0f};
    const std::array<int8_t, 1> saturatingC{};
    Tensor saturatingD(ScalarType::Int8, Shape{1, 1});
    GemmRequest saturatingProblem(
        GemmOperand(
            Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                             std::span<const float>(saturatingA))),
        GemmOperand(
            Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                             std::span<const float>(saturatingB))),
        Tensor::copyNativeStorage<int8_t>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                          std::span<const int8_t>(saturatingC)),
        saturatingD, ScalarType::Float32);
    saturatingProblem.epilogue.outputConversion = OutputConversion::SaturatingInt8;
    referenceGemmWithBlasBackend(saturatingProblem, GemmBackend::Blas);
    require(saturatingD.loadAs<int8_t>({0, 0}) == 127,
            "Transforming BLAS saturating output mismatch.");

    testTransformingBlockScaleFallsBack<float>(ScalarType::Float32);
    testTransformingBlockScaleFallsBack<double>(ScalarType::Float64);

    return 0;
}
