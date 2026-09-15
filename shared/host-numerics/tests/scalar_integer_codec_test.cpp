// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
void testIntegerCodecPolicies() {
    const ScalarConversionOptions rejectNearestEven{
        IntegerRounding::NearestEven,
        IntegerOverflow::Reject,
    };
    const ScalarConversionOptions saturateNearestEven{
        IntegerRounding::NearestEven,
        IntegerOverflow::Saturate,
    };
    const ScalarConversionOptions wrapNearestEven{
        IntegerRounding::NearestEven,
        IntegerOverflow::ModuloWrap,
    };

    const std::array<double, 4> sourceValues{127.5, 128.5, -128.5, -129.5};
    const Tensor source =
        Tensor::copyNativeValues<double>(Shape{sourceValues.size()}, sourceValues);
    const Tensor saturated = source.copyConvertedTo(ScalarType::Int8, saturateNearestEven);
    const Tensor wrapped = source.copyConvertedTo(ScalarType::Int8, wrapNearestEven);
    const std::array<int8_t, 4> saturatedExpected{127, 127, -128, -128};
    const std::array<int8_t, 4> wrappedExpected{-128, -128, -128, 126};
    for (size_t index = 0; index < sourceValues.size(); ++index) {
        require(saturated.loadAs<int8_t>({index}) == saturatedExpected[index],
                "Tensor saturating conversion mismatch.");
        require(wrapped.loadAs<int8_t>({index}) == wrappedExpected[index],
                "Tensor modulo-wrap conversion mismatch.");
    }
    requireThrows<std::overflow_error>(
        [&] { (void)source.copyConvertedTo(ScalarType::Int8, rejectNearestEven); },
        "Tensor rejecting conversion accepted a rounded overflow.");
    require(source.loadAs<int8_t>({0}, saturateNearestEven) == 127 &&
                source.loadAs<int8_t>({0}, wrapNearestEven) == -128,
            "Tensor load conversion options were not applied.");

    Tensor nativeInteger(ScalarType::Int8, Shape{1});
    nativeInteger.storeFrom({0}, 128, saturateNearestEven);
    require(nativeInteger.loadAs<int8_t>({0}) == 127, "Native integer store did not saturate.");
    nativeInteger.storeFrom({0}, 128, wrapNearestEven);
    require(nativeInteger.loadAs<int8_t>({0}) == -128, "Native integer store did not modulo-wrap.");
    requireThrows<std::overflow_error>(
        [&] { nativeInteger.storeFrom({0}, 128, rejectNearestEven); },
        "Native integer store did not reject overflow.");

    const std::array<int16_t, 1> overflowingValue{128};
    const Tensor saturatedFactory = Tensor::copyValuesWithConversion<int16_t>(
        ScalarType::Int8, Shape{1}, overflowingValue, saturateNearestEven);
    require(saturatedFactory.loadAs<int8_t>({0}) == 127,
            "Tensor value factory did not apply conversion options.");

    Tensor packedInteger(ScalarType::Int4, Shape{4});
    packedInteger.storeFrom({0}, 9);
    packedInteger.storeFrom({1}, -9, saturateNearestEven);
    packedInteger.storeFrom({2}, 8, wrapNearestEven);
    packedInteger.storeFrom({3}, -9, wrapNearestEven);
    require(packedInteger.loadAs<int32_t>({0}) == 7 && packedInteger.loadAs<int32_t>({1}) == -8 &&
                packedInteger.loadAs<int32_t>({2}) == -8 && packedInteger.loadAs<int32_t>({3}) == 7,
            "Packed integer policy conversion mismatch.");
    requireThrows<std::overflow_error>([&] { packedInteger.storeFrom({0}, 8, rejectNearestEven); },
                                       "Packed integer store did not reject overflow.");

    const Tensor zeroImaginary(std::complex<double>{3.5, 0.0});
    require(zeroImaginary.item<int32_t>(rejectNearestEven) == 4,
            "Rank-zero tensor zero-imaginary conversion mismatch.");
    const Tensor nonzeroImaginary(std::complex<double>{3.5, 1.0});
    requireThrows<std::domain_error>(
        [&] { (void)nonzeroImaginary.item<double>(rejectNearestEven); },
        "Rank-zero tensor conversion discarded a nonzero imaginary component.");

    Tensor realTensor(ScalarType::Float32, Shape{1});
    realTensor.storeFrom({0}, std::complex<double>{1.25, 0.0}, rejectNearestEven);
    require(realTensor.loadAs<float>({0}) == 1.25f, "Zero-imaginary complex store mismatch.");
    requireThrows<std::domain_error>(
        [&] { realTensor.storeFrom({0}, std::complex<double>{1.25, 0.5}, rejectNearestEven); },
        "Real tensor store discarded a nonzero imaginary component.");

    Tensor floatingCodec(ScalarType::Float8E4M3, Shape{1});
    floatingCodec.storeFrom({0}, 1.0625f, wrapNearestEven);
    require(tensorRaw(floatingCodec) == 0x38,
            "Integer conversion options changed floating codec tie behavior.");
}
}  // namespace scalar_codec_test
