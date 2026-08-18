// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>
#include <cstdint>
#include <roc/host_validation/backends/blas.hpp>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename T>
void testTransformingBlockScale(roc::host_validation::ScalarType accumulatorType) {
    using namespace roc::host_validation;

    const std::array<T, 8> a{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<T, 8> b{1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<T, 4> c{};
    const std::array<uint8_t, 4> scaleA{128, 129, 130, 131};
    const std::array<uint8_t, 4> scaleB{127, 128, 129, 130};
    const Layout layoutA(Shape{2, 4}, {1, 2});
    const Layout layoutB(Shape{4, 2}, {1, 4});
    const Layout layoutD(Shape{2, 2}, {1, 2});
    const Layout scaleLayout = Layout::contiguous(Shape{2, 2});
    Tensor canonicalD(nativeScalarType<T>, layoutD);
    Tensor transformingD(nativeScalarType<T>, layoutD);

    auto makeOperandA = [&]() {
        GemmOperand operand(Tensor::fromNative<T>(layoutA, std::span<const T>(a)));
        operand.blockScale = BlockScaleBinding{
            Tensor(ScalarType::E8M0, scaleLayout, std::as_bytes(std::span(scaleA))),
            2,
        };
        return operand;
    };
    auto makeOperandB = [&]() {
        GemmOperand operand(Tensor::fromNative<T>(layoutB, std::span<const T>(b)));
        operand.blockScale = BlockScaleBinding{
            Tensor(ScalarType::E8M0, scaleLayout, std::as_bytes(std::span(scaleB))),
            2,
        };
        return operand;
    };

    GemmRequest canonicalProblem(makeOperandA(), makeOperandB(),
                                 Tensor::fromNative<T>(layoutD, std::span<const T>(c)), canonicalD,
                                 accumulatorType);
    referenceGemm(canonicalProblem);

    GemmRequest transformingProblem(makeOperandA(), makeOperandB(),
                                    Tensor::fromNative<T>(layoutD, std::span<const T>(c)),
                                    transformingD, accumulatorType);
    TransformingBlasGemmBackend backend;
    require(queryGemmSupport(transformingProblem,
                             {
                                 .backend = GemmBackend::Blas,
                                 .requireRequestedBackend = true,
                             },
                             &backend)
                .supported,
            "Transforming BLAS unexpectedly rejected block scaling.");
    referenceGemm(transformingProblem,
                  {
                      .backend = GemmBackend::Blas,
                      .requireRequestedBackend = true,
                  },
                  &backend);

    const std::array<T, 4> expected{20, 80, 80, 320};
    const Tensor expectedTensor = Tensor::fromNative<T>(layoutD, std::span<const T>(expected));
    require(compare(canonicalD, expectedTensor).passed(),
            "Canonical block-scale reference mismatch.");
    require(compare(transformingD, canonicalD).passed(),
            "Transforming BLAS block-scale result differs from canonical reference.");
}

void testPartialOutputSelection() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    constexpr std::array<float, 4> untouched{-99, -99, -99, -99};
    Tensor d =
        Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(untouched));
    BlasGemmBackend backend;

    GemmRequest problem(GemmOperand(Tensor::fromNative<float>(Layout(Shape{2, 3}, {1, 2}),
                                                              std::span<const float>(a))),
                        GemmOperand(Tensor::fromNative<float>(Layout(Shape{3, 2}, {1, 3}),
                                                              std::span<const float>(b))),
                        d, d, ScalarType::Float32);
    problem.outputSelection = OutputSelection::explicitIndices({0, 3});

    const GemmSupportInfo support = queryGemmSupport(problem,
                                                     {
                                                         .backend = GemmBackend::Blas,
                                                         .requireRequestedBackend = true,
                                                     },
                                                     &backend);
    require(
        !support.supported && support.reason == "BLAS backend requires complete output selection.",
        "BLAS backend support query accepted partial output selection.");

    bool rejectedRequiredBackend = false;
    try {
        referenceGemm(problem,
                      {
                          .backend = GemmBackend::Blas,
                          .requireRequestedBackend = true,
                      },
                      &backend);
    } catch (const std::invalid_argument& error) {
        rejectedRequiredBackend = support.reason == error.what();
    }
    require(rejectedRequiredBackend, "Required BLAS execution accepted partial output selection.");
    require(compare(d, Tensor::fromNative<float>(d.layout(), std::span<const float>(untouched)))
                .passed(),
            "Rejected BLAS execution modified output.");

    const GemmResult fallback = referenceGemm(problem,
                                              {
                                                  .backend = GemmBackend::Blas,
                                                  .requireRequestedBackend = false,
                                              },
                                              &backend);
    require(fallback.runInfo.backendUsed == GemmBackend::Canonical &&
                fallback.runInfo.fallbackReason == support.reason &&
                fallback.runInfo.outputElementsComputed == 2,
            "Partial-output BLAS request did not report canonical fallback.");
    require(d.loadAs<float>({0, 0}) == 58 && d.loadAs<float>({1, 0}) == -99 &&
                d.loadAs<float>({0, 1}) == -99 && d.loadAs<float>({1, 1}) == 154,
            "Canonical BLAS fallback did not preserve unselected outputs.");
}

void testModeratelyLargeExactGemm() {
    using namespace roc::host_validation;

    constexpr size_t dimension = 256;
    const Layout layout(Shape{dimension, dimension},
                        {1, static_cast<ptrdiff_t>(dimension)});
    const std::vector<float> ones(dimension * dimension, 1.0f);
    Tensor output(ScalarType::Float32, layout);

    GemmRequest problem(
        GemmOperand(Tensor::fromNative<float>(layout, std::span<const float>(ones))),
        GemmOperand(Tensor::fromNative<float>(layout, std::span<const float>(ones))),
        output, output, ScalarType::Float32);
    BlasGemmBackend backend;
    referenceGemm(problem,
                  {
                      .backend = GemmBackend::Blas,
                      .requireRequestedBackend = true,
                  },
                  &backend);

    for (size_t row = 0; row < dimension; ++row)
        for (size_t column = 0; column < dimension; ++column)
            require(output.loadAs<float>({row, column}) == static_cast<float>(dimension),
                    "Moderately large BLAS GEMM result mismatch.");
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    Tensor d = Tensor::fromNative<float>(Layout(Shape{2, 2}, {1, 2}),
                                         std::span<const float>(std::array<float, 4>{1, 1, 1, 1}));
    BlasGemmBackend backend;

    GemmRequest problem(GemmOperand(Tensor::fromNative<float>(Layout(Shape{2, 3}, {1, 2}),
                                                              std::span<const float>(a))),
                        GemmOperand(Tensor::fromNative<float>(Layout(Shape{3, 2}, {1, 3}),
                                                              std::span<const float>(b))),
                        d, d, ScalarType::Float32);
    problem.epilogue.alpha = 2.0;
    problem.epilogue.beta = 3.0;

    require(queryGemmSupport(problem,
                             {
                                 .backend = GemmBackend::Blas,
                                 .requireRequestedBackend = true,
                             },
                             &backend)
                .supported,
            "BLAS backend unexpectedly rejected F32 GEMM.");
    const GemmResult result = referenceGemm(problem,
                                            {
                                                .backend = GemmBackend::Blas,
                                                .requireRequestedBackend = true,
                                            },
                                            &backend);
    require(result.runInfo.backendUsed == GemmBackend::Blas,
            "BLAS backend run information mismatch.");
    require(d.loadAs<float>({0, 0}) == 119 && d.loadAs<float>({1, 0}) == 281 &&
                d.loadAs<float>({0, 1}) == 131 && d.loadAs<float>({1, 1}) == 311,
            "BLAS backend F32 result mismatch.");

    const Tensor ones = Tensor::fromNative<float>(
        d.layout(), std::span<const float>(std::array<float, 4>{1, 1, 1, 1}));
    d.copyFrom(ones);
    const GemmRunInfo automatic = referenceGemm(problem, {}, &backend).runInfo;
    require(automatic.backendUsed == GemmBackend::Blas && d.loadAs<float>({0, 0}) == 119 &&
                d.loadAs<float>({1, 0}) == 281 && d.loadAs<float>({0, 1}) == 131 &&
                d.loadAs<float>({1, 1}) == 311,
            "Automatic runtime backend selection mismatch.");

    d.copyFrom(ones);
    problem.epilogue.activation = Activation::Relu;
    const GemmRunInfo fallback = referenceGemm(problem, {}, &backend).runInfo;
    require(fallback.backendUsed == GemmBackend::Canonical && fallback.fallbackReason.has_value() &&
                d.loadAs<float>({0, 0}) == 119 && d.loadAs<float>({1, 0}) == 281 &&
                d.loadAs<float>({0, 1}) == 131 && d.loadAs<float>({1, 1}) == 311,
            "Automatic runtime backend fallback mismatch.");
    problem.epilogue.activation = Activation::None;

    testPartialOutputSelection();
    testModeratelyLargeExactGemm();

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3, 4)};
    Tensor complexD(ScalarType::ComplexFloat32, Shape{1, 1});
    GemmOperand operandA(Tensor::fromNative<std::complex<float>>(
        Layout(Shape{1, 1}, {2, 1}), std::span<const std::complex<float>>(complexA)));
    operandA.conjugate = true;
    GemmRequest complexProblem(
        std::move(operandA),
        GemmOperand(Tensor::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexB))),
        complexD, complexD, ScalarType::ComplexFloat32);
    referenceGemm(complexProblem,
                  {
                      .backend = GemmBackend::Blas,
                      .requireRequestedBackend = true,
                  },
                  &backend);
    require(complexD.loadAs<std::complex<float>>({0, 0}) == std::complex<float>(11, -2),
            "BLAS backend complex result mismatch.");

    const std::array<float, 1> transformedA{0.3f};
    const std::array<float, 1> transformedB{1.0f};
    const std::array<float, 1> transformedC{1.0f};
    const std::array<float, 1> transformedScaleA{0.7f};
    const std::array<float, 1> transformedAlphaVector{0.6f};
    Tensor transformedD(ScalarType::Float32, Shape{1, 1});
    GemmOperand transformedOperandA(Tensor::fromNative<float>(
        Layout::contiguous(Shape{1, 1}), std::span<const float>(transformedA)));
    transformedOperandA.computeType = ScalarType::Float8E4M3;
    transformedOperandA.preQuantizationScales.push_back(
        VectorBinding{Tensor::fromNative<float>(Layout::contiguous(Shape{1}),
                                                std::span<const float>(transformedScaleA)),
                      MatrixAxis::Row});
    transformedOperandA.preQuantizationScales.push_back(
        VectorBinding{Tensor::fromNative<float>(Layout::contiguous(Shape{1}),
                                                std::span<const float>(transformedAlphaVector)),
                      MatrixAxis::Row});
    GemmRequest transformedProblem(
        std::move(transformedOperandA),
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                              std::span<const float>(transformedB))),
        Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                  std::span<const float>(transformedC)),
        transformedD, ScalarType::Float32);
    transformedProblem.epilogue.alpha = 2.0;
    transformedProblem.epilogue.beta = 3.0;
    transformedProblem.epilogue.outputScale = 4.0;
    TransformingBlasGemmBackend transformingBackend;
    referenceGemm(transformedProblem,
                  {
                      .backend = GemmBackend::Blas,
                      .requireRequestedBackend = true,
                  },
                  &transformingBackend);
    require(transformedD.loadAs<float>({0, 0}) == 13.0f,
            "Transforming BLAS pre-quantization/finalization result mismatch.");

    const std::array<float, 1> saturatingA{63.75f};
    const std::array<float, 1> saturatingB{2.0f};
    const std::array<int8_t, 1> saturatingC{};
    Tensor saturatingD(ScalarType::Int8, Shape{1, 1});
    GemmRequest saturatingProblem(
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                              std::span<const float>(saturatingA))),
        GemmOperand(Tensor::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                              std::span<const float>(saturatingB))),
        Tensor::fromNative<int8_t>(Layout::contiguous(Shape{1, 1}),
                                   std::span<const int8_t>(saturatingC)),
        saturatingD, ScalarType::Float32);
    saturatingProblem.epilogue.outputConversion = OutputConversion::SaturatingInt8;
    referenceGemm(saturatingProblem,
                  {
                      .backend = GemmBackend::Blas,
                      .requireRequestedBackend = true,
                  },
                  &transformingBackend);
    require(saturatingD.loadAs<int8_t>({0, 0}) == 127,
            "Transforming BLAS saturating output mismatch.");

    testTransformingBlockScale<float>(ScalarType::Float32);
    testTransformingBlockScale<double>(ScalarType::Float64);

    return 0;
}
