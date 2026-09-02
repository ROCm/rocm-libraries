// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>
#include <roc/host_numerics/scalar.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <vector>

int main() {
    const roc::host_numerics::Scalar scalar{std::complex<float>{1.5f, -2.0f}};
    roc::host_numerics::Tensor tensor(roc::host_numerics::ScalarType::Float32,
                                      roc::host_numerics::Shape{2, 3});
    roc::host_numerics::Tensor mxTensor(roc::host_numerics::ScalarType::Float4E2M1,
                                        roc::host_numerics::Shape{8});
    const roc::host_numerics::Tensor reshaped =
        tensor.reshapeSharingStorage(roc::host_numerics::Shape{3, 2});
    const roc::host_numerics::Tensor padded =
        tensor.copyWithZeroPadding(roc::host_numerics::Shape{3, 4});
    const std::array<size_t, 2> permutation{1, 0};
    const roc::host_numerics::Tensor permuted = tensor.copyWithPermutedDimensions(permutation);
    roc::host_numerics::Tensor scalarTensor(roc::host_numerics::ScalarType::Float32,
                                            roc::host_numerics::Shape{});
    scalarTensor.storeFrom({}, 2.0f);
    const roc::host_numerics::Tensor broadcast =
        scalarTensor.broadcastTo(roc::host_numerics::Shape{2, 3});
    const roc::host_numerics::Shape shape{2, 3};
    const std::array<size_t, 2> coordinates{1, 2};
    return scalar.type() == roc::host_numerics::ScalarType::ComplexFloat32 &&
                   scalar.as<std::complex<float>>() == std::complex<float>{1.5f, -2.0f} &&
                   tensor.shape().elementCount() == 6 &&
                   reshaped.shape() == roc::host_numerics::Shape{3, 2} &&
                   padded.shape() == roc::host_numerics::Shape{3, 4} &&
                   permuted.shape() == roc::host_numerics::Shape{3, 2} &&
                   broadcast.shape() == roc::host_numerics::Shape{2, 3} &&
                   broadcast.layout().stride(0) == 0 && broadcast.layout().stride(1) == 0 &&
                   broadcast.loadAs<float>({1, 2}) == 2.0f &&
                   roc::host_numerics::broadcastShapes(roc::host_numerics::Shape{2, 1},
                                                       roc::host_numerics::Shape{1, 3}) ==
                       roc::host_numerics::Shape{2, 3} &&
                   shape.linearIndex(coordinates,
                                     roc::host_numerics::IndexOrder::LastDimensionFastest) == 5 &&
                   shape.coordinates(5, roc::host_numerics::IndexOrder::LastDimensionFastest) ==
                       std::vector<size_t>({1, 2}) &&
                   mxTensor.type() == roc::host_numerics::ScalarType::Float4E2M1 &&
                   mxTensor.shape().elementCount() == 8
               ? 0
               : 1;
}
