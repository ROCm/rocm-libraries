// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_validation/backends/tiled.hpp>
#include <span>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    const std::array<float, 2> bias{1, -1000};
    std::array<float, 4> d{};
    TiledGemmBackend backend;

    GemmProblem problem(
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a))),
        GemmOperand(
            TensorView::fromNative<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b))),
        TensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c)),
        MutableTensorView::fromNative<float>(Layout(Shape{2, 2}, {1, 2}), std::span<float>(d)),
        ScalarType::Float32);
    problem.epilogue.alpha = 2;
    problem.epilogue.beta = 3;
    problem.epilogue.bias = VectorBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{2}), std::span<const float>(bias)),
        MatrixAxis::Row,
    };
    problem.epilogue.activation = Activation::Relu;

    require(queryGemmSupport(problem, GemmBackend::Tiled, &backend).supported,
            "Tiled backend unexpectedly rejected the test GEMM.");
    const GemmRunInfo run = referenceGemm(problem, {
                                                       .backend = GemmBackend::Tiled,
                                                       .requireRequestedBackend = true,
                                                       .backendImplementation = &backend,
                                                   });
    require(run.backendUsed == GemmBackend::Tiled, "Tiled backend run information mismatch.");
    require(d == std::array<float, 4>{120, 0, 132, 0}, "Tiled backend result mismatch.");

    problem.outputSelection = OutputSelection::explicitIndices({0});
    require(!queryGemmSupport(problem, GemmBackend::Tiled, &backend).supported,
            "Tiled backend accepted partial output selection.");
    return 0;
}
