// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testRuntimeReferenceGemm() {
    using namespace roc::host_numerics;

    const std::array<float, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6> b{7, 9, 11, 8, 10, 12};
    const std::array<float, 4> c{1, 1, 1, 1};
    Tensor d(ScalarType::Float32, Layout(Shape{2, 2}, {1, 2}));
    const std::array<float, 2> bias{1, -10000};
    const std::array<float, 2> scaleA{2, 3};
    const std::array<float, 2> scaleB{5, 7};

    const Tensor operandA =
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {1, 2}), std::span<const float>(a));
    const Tensor operandB =
        Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {1, 3}), std::span<const float>(b));
    const Tensor inputC =
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 2}), std::span<const float>(c));
    const Tensor biasTensor =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2}),
                                         std::span<const float>(bias))
            .expandDims(1);
    const Tensor scaleATensor =
        Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{2}),
                                         std::span<const float>(scaleA))
            .expandDims(1);
    const Tensor scaleBTensor = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2}), std::span<const float>(scaleB));
    Tensor product = matmul(operandA, operandB, ScalarType::Float32);
    product = product * scaleATensor * scaleBTensor;
    const Tensor combined = product + inputC * 2.0f;
    EpilogueOptions epilogue(ScalarType::Float32);
    epilogue.bias = biasTensor;
    epilogue.activation = ReluActivation{};
    referenceEpilogueInto(combined, {.output = d}, epilogue);

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

    const Layout owningLayout(Shape{2, 2}, {7, 2}, 1);
    Tensor owned(ScalarType::Float32, owningLayout);
    matmulInto(operandA, operandB, owned, MatmulOptions(ScalarType::Float32),
               OutputSelection::explicitIndices({0, 1}), GemmBackend::Blocked);
    require(owned.layout() == owningLayout && owned.loadAs<float>({0, 0}) == 58.0f &&
                owned.loadAs<float>({0, 1}) == 64.0f && owned.loadAs<float>({1, 0}) == 0 &&
                owned.loadAs<float>({1, 1}) == 0,
            "Caller-layout matmul result mismatch.");
    std::array<float, 11> ownedStorage;
    std::memcpy(ownedStorage.data(), owned.rawEncodedBackingStorage().data(), sizeof(ownedStorage));
    std::array<float, 11> expectedOwnedStorage{};
    expectedOwnedStorage[1] = 58.0f;
    expectedOwnedStorage[3] = 64.0f;
    require(ownedStorage == expectedOwnedStorage,
            "Caller-layout matmul did not preserve unselected storage.");

    bool rejectedBeforeAllocation = false;
    try {
        (void)matmul(operandA, operandB, ScalarType::Float32, MatmulOptions(ScalarType::Float32),
                     Layout::contiguousLastDimensionFastest(Shape{1, 1}));
    } catch (const std::invalid_argument&) {
        rejectedBeforeAllocation = true;
    }
    require(rejectedBeforeAllocation, "Owning reference GEMM accepted an invalid output layout.");

    const Tensor zeroReduction =
        matmul(Tensor(ScalarType::Float32, Shape{2, 0}), Tensor(ScalarType::Float32, Shape{0, 3}),
               ScalarType::Float32);
    require(zeroReduction.shape() == Shape({2, 3}) &&
                compare(zeroReduction, Tensor(ScalarType::Float32, Shape{2, 3})).passed(),
            "Owning reference GEMM did not initialize an empty reduction to zero.");
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
    const Tensor c = Tensor::copyValuesWithConversion(ScalarType::BFloat16, Shape{1, 1},
                                                      std::span<const float>(cValues));
    MatmulOptions mixedOptions(ScalarType::Float32);
    mixedOptions.computeTypeA = ScalarType::Float4E2M1;
    const Tensor product = matmul(a, b, ScalarType::Float32, mixedOptions);
    const Tensor mixed = add(product, c, ScalarType::Float16, ScalarType::Float32);
    require(mixed.loadAs<float>({0, 0}) == 9.0f, "Runtime mixed-type GEMM result mismatch.");

    const std::array<float, 4> ones{1, 1, 1, 1};
    const std::array<float, 2> scaleAValues{2, 4};
    const std::array<float, 2> scaleBValues{8, 16};
    Tensor blockA = Tensor::copyValuesWithConversion(ScalarType::Float32, Shape{1, 4},
                                                     std::span<const float>(ones));
    Tensor blockB = Tensor::copyValuesWithConversion(ScalarType::Float32, Shape{4, 1},
                                                     std::span<const float>(ones));
    Tensor scalesA = Tensor::copyValuesWithConversion(ScalarType::E8M0, Shape{1, 2},
                                                      std::span<const float>(scaleAValues));
    Tensor scalesB = Tensor::copyValuesWithConversion(ScalarType::E8M0, Shape{1, 2},
                                                      std::span<const float>(scaleBValues));

    MatmulOptions blockOptions(ScalarType::Float32);
    blockOptions.blockScaleA = scalesA;
    blockOptions.blockSizeA = 2;
    blockOptions.blockScaleB = scalesB;
    blockOptions.blockSizeB = 2;
    const Tensor blockResult = matmul(blockA, blockB, ScalarType::Float32, blockOptions,
                                      std::nullopt, GemmBackend::Blocked);
    require(blockResult.loadAs<float>({0, 0}) == 2 * 2 * 8 + 2 * 4 * 16,
            "Runtime block-scaled GEMM result mismatch.");
}

void testBlockedUnalignedScaleSegments() {
    using namespace roc::host_numerics;

    const std::array<float, 7> a{1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 14> b{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    const std::array<float, 3> scaleA{2, 3, 5};
    const std::array<float, 4> scaleB{1, 1, 7, 11};

    const Tensor operandA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 7}), std::span<const float>(a));
    const Tensor operandB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{7, 2}), std::span<const float>(b));
    MatmulOptions options(ScalarType::Float32);
    options.blockScaleA = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{1, 3}), std::span<const float>(scaleA));
    options.blockSizeA = 3;
    options.blockScaleB = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2, 2}), std::span<const float>(scaleB));
    options.blockSizeB = 4;

    Tensor automaticOutput =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, -99});
    matmulInto(operandA, operandB, automaticOutput, options, OutputSelection::explicitIndices({1}));

    const Tensor expected =
        Tensor::copyNativeValues<float>(Shape{1, 2}, std::array<float, 2>{-99, 184});
    require(compare(automaticOutput, expected).passed(),
            "Blocked GEMM with unequal scale segments produced the wrong output.");
}

void testExactIntegerGemm() {
    using namespace roc::host_numerics;

    const std::array<int32_t, 1> a{std::numeric_limits<int32_t>::max()};
    const std::array<int32_t, 1> b{2};
    const std::array<int32_t, 1> c{std::numeric_limits<int32_t>::max()};
    const Tensor operandA = Tensor::copyNativeStorage(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const int32_t>(a));
    const Tensor operandB = Tensor::copyNativeStorage(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const int32_t>(b));
    const Tensor inputC = Tensor::copyNativeStorage(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}), std::span<const int32_t>(c));
    const Tensor product =
        matmul(operandA, operandB, ScalarType::Int32, MatmulOptions(ScalarType::Int32),
               std::nullopt, GemmBackend::Blocked);
    const Tensor d =
        add(product, multiply(inputC, int32_t{2}, ScalarType::Int32, ScalarType::Int32),
            ScalarType::Int32, ScalarType::Int32);

    const auto wrapMultiply = [](int32_t left, int32_t right) {
        return std::bit_cast<int32_t>(static_cast<uint32_t>(left) * static_cast<uint32_t>(right));
    };
    const auto wrapAdd = [](int32_t left, int32_t right) {
        return std::bit_cast<int32_t>(static_cast<uint32_t>(left) + static_cast<uint32_t>(right));
    };
    const int32_t expected = wrapAdd(wrapMultiply(a[0], b[0]), wrapMultiply(int32_t{2}, c[0]));
    require(d.loadAs<int32_t>({0, 0}) == expected,
            "Int32 GEMM did not use defined wrapping arithmetic.");
}

void testRuntimeComplexAndExplicitAxisGemm() {
    using namespace roc::host_numerics;

    const std::array<std::complex<float>, 1> complexA{std::complex<float>(1.0f, 2.0f)};
    const std::array<std::complex<float>, 1> complexB{std::complex<float>(3.0f, 4.0f)};
    const Tensor complexOperandA = Tensor::copyNativeStorage<std::complex<float>>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}),
        std::span<const std::complex<float>>(complexA));
    const Tensor complexOperandB = Tensor::copyNativeStorage<std::complex<float>>(
        Layout::contiguousLastDimensionFastest(Shape{1, 1}),
        std::span<const std::complex<float>>(complexB));
    MatmulOptions complexOptions(ScalarType::ComplexFloat32);
    complexOptions.conjugateA = true;
    const Tensor complexResult =
        matmul(complexOperandA, complexOperandB, ScalarType::ComplexFloat32, complexOptions,
               std::nullopt, GemmBackend::Blocked);
    require(complexResult.loadAs<std::complex<float>>({0, 0}) == std::complex<float>(11.0f, -2.0f),
            "Runtime complex GEMM result mismatch.");

    const std::array<float, 1> realA{1};
    const std::array<float, 2> realB{0, 0};
    const std::array<float, 2> columnBias{2, 3};
    const Tensor realProduct =
        matmul(Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 1}),
                                                std::span<const float>(realA)),
               Tensor::copyNativeStorage<float>(Layout::contiguousLastDimensionFastest(Shape{1, 2}),
                                                std::span<const float>(realB)),
               ScalarType::Float32);
    EpilogueOptions axisOptions(ScalarType::Float32);
    axisOptions.bias = Tensor::copyNativeStorage<float>(
        Layout::contiguousLastDimensionFastest(Shape{2}), std::span<const float>(columnBias));
    const Tensor axisResult = referenceEpilogue(realProduct, {}, axisOptions).output;
    require(compare(axisResult, Tensor::copyNativeStorage<float>(
                                    Layout::contiguousLastDimensionFastest(Shape{1, 2}),
                                    std::span<const float>(columnBias)))
                .passed(),
            "Runtime GEMM explicit column-axis bias mismatch.");
}

}  // namespace host_numerics_test
