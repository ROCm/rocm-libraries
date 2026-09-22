// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_tensor_core_test_support.hpp"

namespace tensor_core_test {
void testTensorConversions() {
    using namespace roc::host_numerics;
    const std::array<float, 2> conversionSource{1.1f, -2.25f};
    const Tensor convertedBFloat16 = Tensor::copyNativeValues<float>(Shape{2}, conversionSource)
                                         .copyConvertedTo(ScalarType::BFloat16);
    const Tensor convertedBack = convertedBFloat16.copyConvertedTo(ScalarType::Float32);
    requireNear(convertedBack.loadAs<float>({0}), 1.1015625f, 0.0f,
                "Tensor conversion did not apply BFloat16 rounding.");
    requireNear(convertedBack.loadAs<float>({1}), -2.25f, 0.0f,
                "Tensor conversion changed an exactly representable value.");

    const std::array<float, 6> matrixConversionSource{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const Tensor matrixSource =
        Tensor::copyNativeValues<float>(Shape{2, 3}, matrixConversionSource);
    const Layout columnMajorLayout(Shape{2, 3}, {1, 2});
    const Tensor columnMajorFloat16 =
        matrixSource.copyConvertedTo(ScalarType::Float16, columnMajorLayout);
    require(columnMajorFloat16.layout() == columnMajorLayout,
            "Tensor conversion did not preserve the requested destination layout.");
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            requireNear(columnMajorFloat16.loadAs<float>({row, column}),
                        matrixSource.loadAs<float>({row, column}), 0.0f,
                        "Tensor conversion changed a logical value in a new layout.");
    const Tensor roundTripMatrix = columnMajorFloat16.copyConvertedTo(
        ScalarType::Float32, Layout::contiguousLastDimensionFastest(matrixSource.shape()));
    for (size_t row = 0; row < 2; ++row)
        for (size_t column = 0; column < 3; ++column)
            requireNear(roundTripMatrix.loadAs<float>({row, column}),
                        matrixSource.loadAs<float>({row, column}), 0.0f,
                        "Tensor conversion changed a value in the reverse relayout direction.");
    requireInvalidArgument(
        [&] {
            (void)matrixSource.copyConvertedTo(ScalarType::Float16, Layout(Shape{2, 3}, {0, 1}));
        },
        "Tensor conversion accepted overlapping destination elements.");

    Tensor rawBFloat16(ScalarType::BFloat16, Shape{2, 2});
    const std::array<std::byte, 8> sourceEncodings{
        std::byte{0xa1}, std::byte{0x7f}, std::byte{0xa2}, std::byte{0xff},
        std::byte{0x00}, std::byte{0x80}, std::byte{0x80}, std::byte{0x3f},
    };
    std::copy(sourceEncodings.begin(), sourceEncodings.end(),
              rawBFloat16.rawEncodedBackingStorage().begin());
    const Tensor relaidBFloat16 =
        rawBFloat16.copyConvertedTo(ScalarType::BFloat16, Layout(Shape{2, 2}, {1, 2}));
    const std::array<std::byte, 8> expectedRelaidEncodings{
        std::byte{0xa1}, std::byte{0x7f}, std::byte{0x00}, std::byte{0x80},
        std::byte{0xa2}, std::byte{0xff}, std::byte{0x80}, std::byte{0x3f},
    };
    require(std::equal(expectedRelaidEncodings.begin(), expectedRelaidEncodings.end(),
                       relaidBFloat16.rawEncodedBackingStorage().begin()),
            "Same-type Tensor relayout did not preserve exact logical encodings.");

    const Shape stridedShape{2, 3, 4};
    Tensor stridedSource(ScalarType::BFloat16, Layout(stridedShape, {20, -5, 1}, 10));
    for (size_t first = 0; first < stridedShape[0]; ++first)
        for (size_t second = 0; second < stridedShape[1]; ++second)
            for (size_t third = 0; third < stridedShape[2]; ++third)
                stridedSource.storeFrom({first, second, third},
                                        static_cast<float>(first * 100 + second * 10 + third));
    const Tensor stridedConverted =
        stridedSource.copyConvertedTo(ScalarType::Float32, Layout(stridedShape, {-15, 5, 1}, 15));
    for (size_t first = 0; first < stridedShape[0]; ++first)
        for (size_t second = 0; second < stridedShape[1]; ++second)
            for (size_t third = 0; third < stridedShape[2]; ++third)
                requireNear(stridedConverted.loadAs<float>({first, second, third}),
                            static_cast<float>(first * 100 + second * 10 + third), 0.0f,
                            "Tensor conversion changed an arbitrary strided-layout value.");

    const Shape bulkShape{513, 513};
    std::vector<float> bulkValues(bulkShape.elementCount());
    for (size_t index = 0; index < bulkValues.size(); ++index)
        bulkValues[index] = static_cast<float>(static_cast<int>(index % 17) - 8);
    const Tensor bulkFloat16 = Tensor::copyValuesWithConversion(ScalarType::Float16, bulkShape,
                                                                std::span<const float>(bulkValues));
    const Tensor bulkConverted = bulkFloat16.copyConvertedTo(
        ScalarType::Float32, Layout::contiguousFirstDimensionFastest(bulkShape));
    for (size_t row : {size_t{0}, size_t{256}, size_t{512}})
        for (size_t column : {size_t{0}, size_t{257}, size_t{512}})
            requireNear(bulkConverted.loadAs<float>({row, column}),
                        bulkFloat16.loadAs<float>({row, column}), 0.0f,
                        "Bulk tiled Tensor conversion changed a logical value.");

    const Shape emptyShape{2, 0};
    const Tensor emptyConverted =
        Tensor(ScalarType::Float16, emptyShape)
            .copyConvertedTo(ScalarType::Float32,
                             Layout::contiguousFirstDimensionFastest(emptyShape));
    require(emptyConverted.shape() == emptyShape && emptyConverted.elementCount() == 0 &&
                emptyConverted.rawEncodedBackingStorage().empty(),
            "Tensor conversion did not preserve an empty shape as a no-op.");

    Tensor int4(ScalarType::Int4, Shape{5});
    auto int4View = int4;
    int4View.storeFrom({0}, -9);
    int4View.storeFrom({1}, -3);
    int4View.storeFrom({2}, 0);
    int4View.storeFrom({3}, 7);
    int4View.storeFrom({4}, 9);
    require(int4.rawEncodedBackingStorage().size() == 3, "Int4 packed storage size mismatch.");
    require(int4.loadAs<int32_t>({0}) == -8 && int4.loadAs<int32_t>({1}) == -3 &&
                int4.loadAs<int32_t>({3}) == 7 && int4.loadAs<int32_t>({4}) == 7,
            "Int4 packed codec mismatch.");

    // The second and fourth Float6 elements cross byte boundaries.
    Tensor float6(ScalarType::Float6E3M2, Shape{4});
    float6.storeFrom({0}, -6.0f);
    float6.storeFrom({1}, -1.0f);
    float6.storeFrom({2}, 1.0f);
    float6.storeFrom({3}, 6.0f);
    require(float6.rawEncodedBackingStorage().size() == 3, "Float6 packed storage size mismatch.");
    require(float6.loadAs<float>({0}) == -6.0f && float6.loadAs<float>({1}) == -1.0f &&
                float6.loadAs<float>({2}) == 1.0f && float6.loadAs<float>({3}) == 6.0f,
            "Float6 cross-byte codec mismatch.");

    Tensor fp4(ScalarType::Float4E2M1, Shape{4});
    fp4.storeFrom({0}, -6.0f);
    fp4.storeFrom({1}, -0.5f);
    fp4.storeFrom({2}, 1.5f);
    fp4.storeFrom({3}, 6.0f);
    const Tensor fp4Copy = fp4.copyConvertedTo(ScalarType::Float4E2M1);
    require(fp4Copy.rawEncodedBackingStorage().size() == fp4.rawEncodedBackingStorage().size() &&
                std::equal(fp4Copy.rawEncodedBackingStorage().begin(),
                           fp4Copy.rawEncodedBackingStorage().end(),
                           fp4.rawEncodedBackingStorage().begin()),
            "Same-type tensor conversion changed raw storage.");
    const Tensor fp4Float = fp4.copyConvertedTo(ScalarType::Float32);
    requireNear(fp4Float.loadAs<float>({0}), -6.0f, 0.0f,
                "Packed tensor conversion minimum mismatch.");
    requireNear(fp4Float.loadAs<float>({2}), 1.5f, 0.0f,
                "Packed tensor conversion normal mismatch.");
    requireNear(fp4.loadAs<float>({0}), -6.0f, 0.0f, "FP4 minimum mismatch.");
    requireNear(fp4.loadAs<float>({1}), -0.5f, 0.0f, "FP4 subnormal mismatch.");
    requireNear(fp4.loadAs<float>({2}), 1.5f, 0.0f, "FP4 normal mismatch.");
    requireNear(fp4.loadAs<float>({3}), 6.0f, 0.0f, "FP4 maximum mismatch.");

    Tensor fp6(ScalarType::Float6E2M3, Shape{4});
    fp6.storeFrom({0}, -7.5f);
    fp6.storeFrom({1}, -0.125f);
    fp6.storeFrom({2}, 0.875f);
    fp6.storeFrom({3}, 7.5f);
    requireNear(fp6.loadAs<float>({0}), -7.5f, 0.0f, "FP6 minimum mismatch.");
    requireNear(fp6.loadAs<float>({3}), 7.5f, 0.0f, "FP6 maximum mismatch.");

    Tensor float16(ScalarType::Float16, Shape{2});
    float16.storeFrom({0}, 1.5f);
    float16.storeFrom({1}, -0.25f);
    requireNear(float16.loadAs<float>({0}), 1.5f, 0.0f, "Float16 codec mismatch.");
    requireNear(float16.loadAs<float>({1}), -0.25f, 0.0f, "Float16 codec mismatch.");

    Tensor bfloat16(ScalarType::BFloat16, Shape{1});
    bfloat16.storeFrom({0}, 1.25f);
    requireNear(bfloat16.loadAs<float>({0}), 1.25f, 0.01f, "BFloat16 codec mismatch.");

    Tensor complex(ScalarType::ComplexFloat32, Shape{1});
    complex.storeFrom({0}, std::complex<float>(2.0f, -3.0f));
    require(complex.loadAs<std::complex<float>>({0}) == std::complex<float>(2.0f, -3.0f),
            "Complex codec mismatch.");

    Tensor float8(ScalarType::Float8E4M3, Shape{1});
    float8.storeFrom({0}, 1.25f);
    requireNear(float8.loadAs<float>({0}), 1.25f, 0.0f, "Float8 codec mismatch.");
}
}  // namespace tensor_core_test
