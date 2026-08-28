// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_numerics/tensor.hpp>
#include <span>

int main() {
    using namespace roc::host_numerics;
    Tensor tensor(ScalarType::Float32, Shape{2, 3});
    tensor.storeFrom({1, 2}, 4.0f);
    const Tensor converted = tensor.copyConvertedTo(ScalarType::Float16);
    const std::array<float, 3> native{1.0f, 2.0f, 3.0f};
    const Tensor runtimeTensor = Tensor::copyNativeStorage(std::span<const float>(native));
    const Tensor reversed = runtimeTensor.shareStorageWithLayout(Layout(Shape{3}, {-1}, 2));
    return tensor.elementCount() == 6 && tensor.loadAs<float>({1, 2}) == 4.0f &&
                   converted.type() == ScalarType::Float16 &&
                   converted.loadAs<float>({1, 2}) == 4.0f && reversed.loadAs<float>({0}) == 3.0f &&
                   reversed.loadAs<float>({2}) == 1.0f && runtimeTensor.shape() == Shape{3}
               ? 0
               : 1;
}
