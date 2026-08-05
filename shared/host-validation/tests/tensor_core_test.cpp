// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/tensor.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void requireNear(float observed, float expected, float tolerance, const char* message) {
    if (std::abs(observed - expected) > tolerance) throw std::runtime_error(message);
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const Shape shape{2, 3};
    require(shape.rank() == 2, "Shape rank mismatch.");
    require(shape.elementCount() == 6, "Shape element count mismatch.");
    require(visitScalarType(ScalarType::Float32, []<typename Tag>() {
                return Tag::type == ScalarType::Float32 &&
                       std::is_same_v<typename Tag::Storage, float>;
            }),
            "Runtime scalar tag dispatch mismatch.");
    require(visitScalarType(ScalarType::Int4, []<typename Tag>() {
                return Tag::type == ScalarType::Int4 &&
                       std::is_void_v<typename Tag::Storage>;
            }),
            "Packed scalar tag dispatch mismatch.");

    Tensor tensor(ScalarType::Float32, shape);
    tensor.mutableView().storeFrom({1, 2}, 7.0f);
    require(tensor.view().loadAs<float>({1, 2}) == 7.0f, "Owning tensor view mismatch.");
    require(tensor.layout().strides()[0] == 3 && tensor.layout().strides()[1] == 1,
            "Contiguous tensor strides mismatch.");
    require(tensor.storage().size() == 6 * sizeof(float), "Float32 storage size mismatch.");

    const std::array<float, 3> nativeValues{2.0f, 4.0f, 6.0f};
    const Tensor nativeTensor =
        Tensor::fromNativeValues<float>(Shape{3}, std::span<const float>(nativeValues));
    require(nativeTensor.type() == ScalarType::Float32 &&
                nativeTensor.view().loadAs<float>({2}) == 6.0f,
            "Native tensor factory mismatch.");

    Tensor copied = tensor;
    copied.mutableView().storeFrom({1, 2}, 11.0f);
    require(tensor.view().loadAs<float>({1, 2}) == 7.0f,
            "Copying an owning tensor did not deep-copy storage.");

    std::array<int32_t, 8> padded{};
    MutableTensorView paddedView(
        ScalarType::Int32,
        Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1),
        std::as_writable_bytes(std::span<int32_t>(padded)));
    paddedView.storeFrom({0, 0}, 4);
    paddedView.storeFrom({1, 1}, 9);
    require(padded[1] == 4 && padded[5] == 9, "Strided tensor layout mismatch.");

    auto nativePaddedView = MutableTensorView::fromNative<int32_t>(
        Layout(Shape{2, 2}, std::vector<ptrdiff_t>{1, 3}, 1),
        std::span<int32_t>(padded));
    nativePaddedView.storeFrom({0, 1}, 12);
    require(padded[4] == 12, "Native mutable tensor view factory mismatch.");

    const std::array<int32_t, 3> reversedStorage{1, 2, 3};
    const TensorView reversed(
        ScalarType::Int32,
        Layout(Shape{3}, std::vector<ptrdiff_t>{-1}, 2),
        std::as_bytes(std::span<const int32_t>(reversedStorage)));
    require(reversed.loadAs<int32_t>({0}) == 3 && reversed.loadAs<int32_t>({2}) == 1,
            "Negative-stride tensor layout mismatch.");

    Tensor int4(ScalarType::Int4, Shape{5});
    auto int4View = int4.mutableView();
    int4View.storeFrom({0}, -9);
    int4View.storeFrom({1}, -3);
    int4View.storeFrom({2}, 0);
    int4View.storeFrom({3}, 7);
    int4View.storeFrom({4}, 9);
    require(int4.storage().size() == 3, "Int4 packed storage size mismatch.");
    require(int4.view().loadAs<int32_t>({0}) == -8 &&
                int4.view().loadAs<int32_t>({1}) == -3 &&
                int4.view().loadAs<int32_t>({3}) == 7 &&
                int4.view().loadAs<int32_t>({4}) == 7,
            "Int4 packed codec mismatch.");

    Tensor int12(ScalarType::Int12, Shape{2});
    int12.mutableView().storeFrom({0}, -2048);
    int12.mutableView().storeFrom({1}, 2047);
    require(int12.storage().size() == 3, "Int12 packed storage size mismatch.");
    require(int12.view().loadAs<int32_t>({0}) == -2048 &&
                int12.view().loadAs<int32_t>({1}) == 2047,
            "Int12 cross-byte codec mismatch.");

    Tensor fp4(ScalarType::Float4E2M1, Shape{4});
    fp4.mutableView().storeFrom({0}, -6.0f);
    fp4.mutableView().storeFrom({1}, -0.5f);
    fp4.mutableView().storeFrom({2}, 1.5f);
    fp4.mutableView().storeFrom({3}, 6.0f);
    requireNear(fp4.view().loadAs<float>({0}), -6.0f, 0.0f, "FP4 minimum mismatch.");
    requireNear(fp4.view().loadAs<float>({1}), -0.5f, 0.0f, "FP4 subnormal mismatch.");
    requireNear(fp4.view().loadAs<float>({2}), 1.5f, 0.0f, "FP4 normal mismatch.");
    requireNear(fp4.view().loadAs<float>({3}), 6.0f, 0.0f, "FP4 maximum mismatch.");

    Tensor fp6(ScalarType::Float6E2M3, Shape{4});
    fp6.mutableView().storeFrom({0}, -7.5f);
    fp6.mutableView().storeFrom({1}, -0.125f);
    fp6.mutableView().storeFrom({2}, 0.875f);
    fp6.mutableView().storeFrom({3}, 7.5f);
    requireNear(fp6.view().loadAs<float>({0}), -7.5f, 0.0f, "FP6 minimum mismatch.");
    requireNear(fp6.view().loadAs<float>({3}), 7.5f, 0.0f, "FP6 maximum mismatch.");

    Tensor float16(ScalarType::Float16, Shape{2});
    float16.mutableView().storeFrom({0}, 1.5f);
    float16.mutableView().storeFrom({1}, -0.25f);
    requireNear(float16.view().loadAs<float>({0}), 1.5f, 0.0f, "Float16 codec mismatch.");
    requireNear(float16.view().loadAs<float>({1}), -0.25f, 0.0f, "Float16 codec mismatch.");

    Tensor bfloat16(ScalarType::BFloat16, Shape{1});
    bfloat16.mutableView().storeFrom({0}, 1.25f);
    requireNear(
        bfloat16.view().loadAs<float>({0}), 1.25f, 0.01f, "BFloat16 codec mismatch.");

    Tensor complex(ScalarType::ComplexFloat32, Shape{1});
    complex.mutableView().storeFrom({0}, std::complex<float>(2.0f, -3.0f));
    require(complex.view().loadAs<std::complex<float>>({0}) ==
                std::complex<float>(2.0f, -3.0f),
            "Complex codec mismatch.");

    bool unsupportedThrew = false;
    try {
        Tensor unsupported(ScalarType::Float8E4M3, Shape{1});
        unsupported.mutableView().storeFrom({0}, 1.0f);
    } catch (const std::invalid_argument&) {
        unsupportedThrew = true;
    }
    require(unsupportedThrew, "Unsupported codec did not fail explicitly.");

    return 0;
}
