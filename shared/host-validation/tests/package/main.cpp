// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_validation/tensor.hpp>
#include <span>

int main() {
    using namespace roc::host_validation;
    Tensor tensor(ScalarType::Float32, Shape{2, 3});
    tensor.mutableView().storeFrom({1, 2}, 4.0f);
    const Tensor converted = tensor.to(ScalarType::Float16);
    const std::array<float, 3> native{1.0f, 2.0f, 3.0f};
    const TypedTensorView<float> typedView(Layout(Shape{3}, {-1}, 2),
                                           std::span<const float>(native));
    const TensorView runtimeView = TensorView::fromNative(std::span<const float>(native));
    return tensor.size() == 6 && tensor.view().loadAs<float>({1, 2}) == 4.0f &&
                   converted.type() == ScalarType::Float16 &&
                   converted.view().loadAs<float>({1, 2}) == 4.0f && typedView.at({0}) == 3.0f &&
                   typedView.at({2}) == 1.0f && runtimeView.shape() == Shape{3}
               ? 0
               : 1;
}
