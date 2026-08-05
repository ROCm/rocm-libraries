// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_validation/tensor.hpp>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const Shape shape{2, 3};
    require(shape.rank() == 2, "Shape rank mismatch.");
    require(shape.elementCount() == 6, "Shape element count mismatch.");

    Tensor<float> tensor(shape, 0.0f);
    tensor.mutableView().at({1, 2}) = 7.0f;
    require(tensor.view().at({1, 2}) == 7.0f, "Owning tensor view mismatch.");
    require(tensor.layout().strides()[0] == 3 && tensor.layout().strides()[1] == 1,
            "Contiguous tensor strides mismatch.");

    std::array<int, 8> padded{};
    MutableTensorView<int> paddedView(padded.data(),
                                      Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1));
    paddedView.at({0, 0}) = 4;
    paddedView.at({1, 1}) = 9;
    require(padded[1] == 4 && padded[5] == 9, "Strided tensor layout mismatch.");

    const std::array<int, 3> reversedStorage{1, 2, 3};
    const TensorView<int> reversed(reversedStorage.data(),
                                   Layout(Shape{3}, std::vector<ptrdiff_t>{-1}, 2));
    require(reversed.at({0}) == 3 && reversed.at({2}) == 1,
            "Negative-stride tensor layout mismatch.");

    return 0;
}
