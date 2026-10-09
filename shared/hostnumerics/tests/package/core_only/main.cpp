// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>
#include <roc/hostnumerics/scalar.hpp>
#include <roc/hostnumerics/tensor.hpp>
#include <vector>

int main() {
    const roc::hostnumerics::Tensor scalar{std::complex<float>{1.5f, -2.0f}};
    roc::hostnumerics::Tensor tensor(roc::hostnumerics::ScalarType::Float32,
                                     roc::hostnumerics::Shape{2, 3});
    roc::hostnumerics::Tensor mxTensor(roc::hostnumerics::ScalarType::Float4E2M1,
                                       roc::hostnumerics::Shape{8});
    const roc::hostnumerics::Tensor reshaped =
        tensor.reshapeSharingStorage(roc::hostnumerics::Shape{3, 2});
    const roc::hostnumerics::Tensor padded =
        tensor.copyWithZeroPadding(roc::hostnumerics::Shape{3, 4});
    const std::array<size_t, 2> permutation{1, 0};
    const roc::hostnumerics::Tensor permuted = tensor.copyWithPermutedDimensions(permutation);
    roc::hostnumerics::Tensor scalarTensor(roc::hostnumerics::ScalarType::Float32,
                                           roc::hostnumerics::Shape{});
    scalarTensor.storeFrom({}, 2.0f);
    const roc::hostnumerics::Tensor broadcast =
        scalarTensor.broadcastTo(roc::hostnumerics::Shape{2, 3});
    const roc::hostnumerics::Shape shape{2, 3};
    const std::array<size_t, 2> coordinates{1, 2};
    return scalar.type() == roc::hostnumerics::ScalarType::ComplexFloat32 &&
                   scalar.item<std::complex<float>>() == std::complex<float>{1.5f, -2.0f} &&
                   tensor.shape().elementCount() == 6 &&
                   reshaped.shape() == roc::hostnumerics::Shape{3, 2} &&
                   padded.shape() == roc::hostnumerics::Shape{3, 4} &&
                   permuted.shape() == roc::hostnumerics::Shape{3, 2} &&
                   broadcast.shape() == roc::hostnumerics::Shape{2, 3} &&
                   broadcast.layout().stride(0) == 0 && broadcast.layout().stride(1) == 0 &&
                   broadcast.loadAs<float>({1, 2}) == 2.0f &&
                   roc::hostnumerics::broadcastShapes(roc::hostnumerics::Shape{2, 1},
                                                      roc::hostnumerics::Shape{1, 3}) ==
                       roc::hostnumerics::Shape{2, 3} &&
                   shape.linearIndex(coordinates,
                                     roc::hostnumerics::IndexOrder::LastDimensionFastest) == 5 &&
                   shape.coordinates(5, roc::hostnumerics::IndexOrder::LastDimensionFastest) ==
                       std::vector<size_t>({1, 2}) &&
                   mxTensor.type() == roc::hostnumerics::ScalarType::Float4E2M1 &&
                   mxTensor.shape().elementCount() == 8
               ? 0
               : 1;
}
