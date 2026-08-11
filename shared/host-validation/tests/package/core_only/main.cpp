// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/tensor.hpp>

int main() {
    roc::host_validation::Tensor tensor(roc::host_validation::ScalarType::Float32,
                                        roc::host_validation::Shape{2, 3});
    return tensor.shape().elementCount() == 6 ? 0 : 1;
}
