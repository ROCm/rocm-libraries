// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/tensor.hpp>

int main() {
    using namespace roc::host_validation;
    Tensor tensor(ScalarType::Float32, Shape{2, 3});
    tensor.mutableView().storeFrom({1, 2}, 4.0f);
    return tensor.size() == 6 && tensor.view().loadAs<float>({1, 2}) == 4.0f ? 0 : 1;
}
