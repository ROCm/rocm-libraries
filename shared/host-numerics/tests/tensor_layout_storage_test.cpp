// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testTensorLayoutsAndBounds() {
    using namespace roc::host_numerics;
    Tensor paddedTensor(ScalarType::Int32, Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1));
    paddedTensor.storeFrom({0, 0}, 4);
    paddedTensor.storeFrom({1, 1}, 9);
    require(paddedTensor.loadAs<int32_t>({0, 0}) == 4 && paddedTensor.loadAs<int32_t>({1, 1}) == 9,
            "Strided tensor layout mismatch.");
    requireInvalidArgument([&] { (void)paddedTensor.reshapeSharingStorage(Shape{4}); },
                           "Tensor reshape accepted a noncontiguous layout.");

    Tensor paddedAlias = paddedTensor;
    paddedAlias.storeFrom({0, 1}, 12);
    require(paddedTensor.loadAs<int32_t>({0, 1}) == 12, "Tensor aliases did not share storage.");
    const Tensor constPaddedAlias = paddedTensor;
    constPaddedAlias.storeFrom({1, 0}, 15);
    require(paddedTensor.loadAs<int32_t>({1, 0}) == 15,
            "Const tensor handle did not retain shallow mutability.");

    const std::array<int32_t, 3> reversedStorage{1, 2, 3};
    const Tensor reversed = Tensor::copyEncodedBackingStorage(
        ScalarType::Int32, Layout(Shape{3}, std::vector<ptrdiff_t>{-1}, 2),
        std::as_bytes(std::span<const int32_t>(reversedStorage)));
    require(reversed.loadAs<int32_t>({0}) == 3 && reversed.loadAs<int32_t>({2}) == 1,
            "Negative-stride tensor layout mismatch.");
    const Tensor reversedFloat = reversed.copyConvertedTo(ScalarType::Float32);
    require(reversedFloat.layout() == reversed.layout() &&
                reversedFloat.loadAs<float>({0}) == 3.0f &&
                reversedFloat.loadAs<float>({2}) == 1.0f,
            "Tensor conversion did not preserve the logical layout.");
}
}  // namespace tensor_core_test
