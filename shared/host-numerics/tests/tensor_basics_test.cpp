// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testTensorBasicsAndBroadcasting() {
    using namespace roc::host_numerics;
    const Shape shape{2, 3};
    static_assert(std::is_same_v<decltype(std::declval<const Tensor&>().rawEncodedBackingStorage()),
                                 std::span<std::byte>>);

    Tensor tensor(ScalarType::Float32, shape);
    tensor.storeFrom({1, 2}, 7.0f);
    require(tensor.loadAs<float>({1, 2}) == 7.0f, "Owning tensor view mismatch.");
    require(tensor.layout().strides()[0] == 3 && tensor.layout().strides()[1] == 1,
            "Contiguous tensor strides mismatch.");
    require(tensor.rawEncodedBackingStorage().size() == 6 * sizeof(float),
            "Float32 storage size mismatch.");

    const std::array<float, 3> nativeValues{2.0f, 4.0f, 6.0f};
    const Tensor nativeTensor =
        Tensor::copyNativeValues<float>(Shape{3}, std::span<const float>(nativeValues));
    require(nativeTensor.type() == ScalarType::Float32 && nativeTensor.loadAs<float>({2}) == 6.0f,
            "Native tensor factory mismatch.");

    Tensor scalarTensor(ScalarType::Float32, Layout(Shape{}, {}, 2));
    scalarTensor.storeFrom({}, -3.5f);
    const float scalarSnapshot = scalarTensor.item<float>();
    scalarTensor.storeFrom({}, 9.0f);
    require(scalarSnapshot == -3.5f,
            "Tensor item did not preserve independent scalar value semantics.");
    require(scalarTensor.item<float>() == 9.0f,
            "Typed Tensor item did not return a native scalar value.");

    Tensor packedScalarTensor(ScalarType::Float6E3M2, Layout(Shape{}, {}, 1));
    packedScalarTensor.storeFrom({}, 1.5f);
    require(packedScalarTensor.item<float>() == 1.5f,
            "Packed rank-zero Tensor did not return its value.");
    requireInvalidArgument([&] { (void)nativeTensor.item<float>(); },
                           "Tensor item accepted a non-scalar shape.");

    require(broadcastShapes(Shape{2, 1, 3}, Shape{1, 4, 1}) == Shape{2, 4, 3},
            "Tensor broadcast shape mismatch.");
    require(broadcastShapes(Shape{0, 3}, Shape{1, 3}) == Shape{0, 3},
            "Tensor broadcasting mishandled a zero extent.");
    requireInvalidArgument([&] { (void)broadcastShapes(Shape{2}, Shape{3}); },
                           "Tensor broadcasting accepted incompatible shapes.");

    const std::array<float, 2> columnValues{2.0f, 5.0f};
    const Tensor column = Tensor::copyNativeValues<float>(Shape{2, 1}, columnValues);
    const Tensor broadcastColumn = column.broadcastTo(Shape{2, 3});
    require(broadcastColumn.shape() == Shape{2, 3} && broadcastColumn.layout().strides()[0] == 1 &&
                broadcastColumn.layout().strides()[1] == 0 &&
                broadcastColumn.loadAs<float>({0, 2}) == 2.0f &&
                broadcastColumn.loadAs<float>({1, 2}) == 5.0f,
            "Tensor broadcast view mismatch.");
    requireInvalidArgument([&] { (void)column.broadcastTo(Shape{3, 2}); },
                           "Tensor broadcast accepted an incompatible destination shape.");
    const Tensor expandedColumn =
        Tensor::copyNativeValues<float>(Shape{2}, columnValues).expandDims(1);
    require(expandedColumn.shape() == Shape{2, 1} && expandedColumn.layout().stride(0) == 1 &&
                expandedColumn.layout().stride(1) == 0 &&
                expandedColumn.loadAs<float>({1, 0}) == 5.0f,
            "Tensor singleton-dimension expansion mismatch.");
    requireOutOfRange([&] { (void)column.expandDims(3); },
                      "Tensor expansion accepted an out-of-range axis.");

    const Tensor broadcastScalar = scalarTensor.broadcastTo(Shape{2, 3});
    require(broadcastScalar.layout().strides()[0] == 0 &&
                broadcastScalar.layout().strides()[1] == 0 &&
                broadcastScalar.loadAs<float>({1, 2}) == 9.0f,
            "Rank-zero Tensor did not broadcast as a scalar.");

    Tensor uninitialized = Tensor::allocateUninitialized(ScalarType::Float32, Shape{2});
    uninitialized.storeFrom({0}, 3.0f);
    uninitialized.storeFrom({1}, 7.0f);
    require(uninitialized.loadAs<float>({0}) == 3.0f && uninitialized.loadAs<float>({1}) == 7.0f,
            "Uninitialized Tensor storage did not retain written values.");
    requireInvalidArgument([&] { (void)Tensor::allocateUninitialized(ScalarType::Int4, Shape{2}); },
                           "Uninitialized Tensor allocation accepted a packed scalar type.");
}
}  // namespace tensor_core_test
