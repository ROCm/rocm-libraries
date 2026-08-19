// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>
#include <roc/host_validation/scalar.hpp>
#include <roc/host_validation/tensor.hpp>
#include <vector>

int main() {
    const roc::host_validation::Scalar scalar =
        roc::host_validation::Scalar::from(std::complex<float>{1.5f, -2.0f});
    roc::host_validation::Tensor tensor(roc::host_validation::ScalarType::Float32,
                                        roc::host_validation::Shape{2, 3});
    roc::host_validation::Tensor mxTensor(roc::host_validation::ScalarType::Float4E2M1,
                                          roc::host_validation::Shape{8});
    const roc::host_validation::Shape shape{2, 3};
    const std::array<size_t, 2> coordinates{1, 2};
    return scalar.type() == roc::host_validation::ScalarType::ComplexFloat32 &&
                   scalar.as<std::complex<float>>() == std::complex<float>{1.5f, -2.0f} &&
                   tensor.shape().elementCount() == 6 &&
                   shape.linearIndex(coordinates,
                                     roc::host_validation::IndexOrder::LastDimensionFastest) == 5 &&
                   shape.coordinates(5, roc::host_validation::IndexOrder::LastDimensionFastest) ==
                       std::vector<size_t>({1, 2}) &&
                   mxTensor.type() == roc::host_validation::ScalarType::Float4E2M1 &&
                   mxTensor.shape().elementCount() == 8
               ? 0
               : 1;
}
