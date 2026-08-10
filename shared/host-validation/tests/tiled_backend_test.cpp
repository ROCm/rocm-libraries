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

    GemmRequest problem(
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

    require(queryGemmSupport(problem,
                             {
                                 .backend = GemmBackend::Tiled,
                                 .requireRequestedBackend = true,
                             },
                             &backend)
                .supported,
            "Tiled backend unexpectedly rejected the test GEMM.");
    const GemmResult result = referenceGemm(problem,
                                            {
                                                .backend = GemmBackend::Tiled,
                                                .requireRequestedBackend = true,
                                            },
                                            &backend);
    require(result.runInfo.backendUsed == GemmBackend::Tiled,
            "Tiled backend run information mismatch.");
    require(d == std::array<float, 4>{120, 0, 132, 0}, "Tiled backend result mismatch.");

    d.fill(-99);
    problem.outputSelection = OutputSelection::explicitIndices({0});
    referenceGemm(problem,
                  {
                      .backend = GemmBackend::Tiled,
                      .requireRequestedBackend = true,
                  },
                  &backend);
    require(d == std::array<float, 4>{120, -99, -99, -99},
            "Tiled backend partial output selection mismatch.");

    const std::array<float, 16> ones{
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    };
    const std::array<float, 1> zero{0};
    const std::array<float, 2> blockScaleA{2, 4};
    const std::array<float, 2> blockScaleB{8, 16};
    std::array<float, 1> blockOutput{};
    GemmOperand blockA(TensorView::fromNative<float>(Layout::contiguous(Shape{1, 16}),
                                                     std::span<const float>(ones)));
    GemmOperand blockB(TensorView::fromNative<float>(Layout::contiguous(Shape{16, 1}),
                                                     std::span<const float>(ones)));
    blockA.blockScale = BlockScaleBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                      std::span<const float>(blockScaleA)),
        8,
    };
    blockB.blockScale = BlockScaleBinding{
        TensorView::fromNative<float>(Layout::contiguous(Shape{1, 2}),
                                      std::span<const float>(blockScaleB)),
        8,
    };
    GemmRequest blockProblem(std::move(blockA), std::move(blockB),
                             TensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                           std::span<const float>(zero)),
                             MutableTensorView::fromNative<float>(Layout::contiguous(Shape{1, 1}),
                                                                  std::span<float>(blockOutput)),
                             ScalarType::Float32);
    referenceGemm(blockProblem,
                  {
                      .backend = GemmBackend::Tiled,
                      .requireRequestedBackend = true,
                  },
                  &backend);
    require(blockOutput[0] == 640, "Tiled backend block scaling mismatch.");
    return 0;
}
