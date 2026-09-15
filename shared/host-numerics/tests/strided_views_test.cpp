// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
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
    const Tensor operandA =
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 3}, {4, 1}), std::span<const float>(a));
    const Tensor operandB =
        Tensor::copyNativeStorage<float>(Layout(Shape{3, 2}, {3, 1}), std::span<const float>(b));
    const Tensor inputC =
        Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 4}), std::span<const float>(c));
    Tensor d = Tensor::takeOwnershipOfEncodedBackingStorage(ScalarType::Float32, outputLayout,
                                                            std::move(dStorage));
    const Tensor product = matmul(operandA, operandB, ScalarType::Float32,
                                  MatmulOptions(ScalarType::Float32), outputLayout);
    const Tensor scaledProduct =
        multiply(product, 2.0f, ScalarType::Float32, ScalarType::Float32, outputLayout);
    const Tensor scaledC =
        multiply(inputC, 3.0f, ScalarType::Float32, ScalarType::Float32, outputLayout);
    addInto(scaledProduct, scaledC, d, ScalarType::Float32);

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

}  // namespace host_numerics_test
