// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/tensor.hpp>

int main() {
    roc::host_validation::Tensor<float> tensor({2, 3}, 1.0f);
    return tensor.size() == 6 ? 0 : 1;
}
