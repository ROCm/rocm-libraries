// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>
#include <roc/host_validation/backends/blas.hpp>
#include <span>
#include <stdexcept>
#include <utility>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    std::array<float, 4> d{1, 1, 1, 1};
    BlasGemmBackend backend;

    GemmProblem problem(
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a))),
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b))),
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(d)),
        MutableTensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<float>(d)),
        ScalarType::Float32);
    problem.epilogue.alpha = 2.0;
    problem.epilogue.beta = 3.0;

    require(queryGemmSupport(problem, GemmBackend::Blas, &backend).supported,
            "BLAS backend unexpectedly rejected F32 GEMM.");
    const GemmRunInfo run = referenceGemm(problem, {
                                                       .backend = GemmBackend::Blas,
                                                       .requireRequestedBackend = true,
                                                       .backendImplementation = &backend,
                                                   });
    require(run.backendUsed == GemmBackend::Blas, "BLAS backend run information mismatch.");
    require(d == std::array<float, 4>{119, 281, 131, 311}, "BLAS backend F32 result mismatch.");

    d.fill(1);
    const GemmRunInfo automatic = referenceGemm(problem, {.backendImplementation = &backend});
    require(
        automatic.backendUsed == GemmBackend::Blas && d == std::array<float, 4>{119, 281, 131, 311},
        "Automatic runtime backend selection mismatch.");

    d.fill(1);
    problem.epilogue.activation = Activation::Relu;
    const GemmRunInfo fallback = referenceGemm(problem, {.backendImplementation = &backend});
    require(fallback.backendUsed == GemmBackend::Canonical && fallback.fallbackReason.has_value() &&
                d == std::array<float, 4>{119, 281, 131, 311},
            "Automatic runtime backend fallback mismatch.");
    problem.epilogue.activation = Activation::None;

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1, 2)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3, 4)};
    std::array<std::complex<float>, 1> complexD{};
    GemmOperand operandA(TensorView::fromNative<std::complex<float>>(
        Layout(Shape{1, 1}, {2, 1}), std::span<const std::complex<float>>(complexA)));
    operandA.conjugate = true;
    GemmProblem complexProblem(
        std::move(operandA),
        GemmOperand(TensorView::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<const std::complex<float>>(complexB))),
        TensorView::fromNative<std::complex<float>>(Layout::contiguous(Shape{1, 1}),
                                                    std::span<const std::complex<float>>(complexD)),
        MutableTensorView::fromNative<std::complex<float>>(
            Layout::contiguous(Shape{1, 1}), std::span<std::complex<float>>(complexD)),
        ScalarType::ComplexFloat32);
    referenceGemm(complexProblem, {
                                      .backend = GemmBackend::Blas,
                                      .requireRequestedBackend = true,
                                      .backendImplementation = &backend,
                                  });
    require(complexD[0] == std::complex<float>(11, -2), "BLAS backend complex result mismatch.");

    return 0;
}
