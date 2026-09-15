// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
void testIntegerConversionPrimitives() {
    const ScalarConversionOptions defaults;
    require(defaults.integerRounding == IntegerRounding::TowardZero &&
                defaults.integerOverflow == IntegerOverflow::Reject &&
                defaults.bfloat16Rounding == BFloat16Rounding::NearestEven,
            "Scalar conversion option defaults changed.");

    constexpr float bfloat16RoundingBoundary = 1.005859375f;
    Tensor nearestEven(ScalarType::BFloat16, Shape{1});
    nearestEven.storeFrom({0}, bfloat16RoundingBoundary, defaults);
    require(tensorRaw(nearestEven) == 0x3f81,
            "Default BFloat16 conversion did not round to nearest even.");

    ScalarConversionOptions truncateBFloat16;
    truncateBFloat16.bfloat16Rounding = BFloat16Rounding::Truncate;
    Tensor truncated(ScalarType::BFloat16, Shape{1});
    truncated.storeFrom({0}, bfloat16RoundingBoundary, truncateBFloat16);
    require(tensorRaw(truncated) == 0x3f80,
            "Explicit BFloat16 truncation did not discard the low bits.");

    const ScalarConversionOptions rejectTowardZero{
        IntegerRounding::TowardZero,
        IntegerOverflow::Reject,
    };
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

    require(convertScalar<int32_t>(3.75, rejectTowardZero) == 3 &&
                convertScalar<int32_t>(-3.75, rejectTowardZero) == -3,
            "Toward-zero integer rounding mismatch.");
    require(convertScalar<int8_t>(128.0) == -128,
            "No-options scalar conversion did not use modulo overflow semantics.");

    requireThrows<std::overflow_error>(
        [&] { (void)Tensor::scalar(ScalarType::Int4, 9, rejectTowardZero); },
        "Explicit-policy rank-zero construction accepted integer overflow.");
    const Tensor saturatedInt4 = Tensor::scalar(ScalarType::Int4, 9, saturateNearestEven);
    const Tensor wrappedInt4 = Tensor::scalar(ScalarType::Int4, 9, wrapNearestEven);
    require(saturatedInt4.item<int>() == 7 && wrappedInt4.item<int>() == -7,
            "Explicit-policy rank-zero integer construction mismatch.");
    const Tensor explicitlyTruncatedBFloat16 =
        Tensor::scalar(ScalarType::BFloat16, bfloat16RoundingBoundary, truncateBFloat16);
    require(tensorRaw(explicitlyTruncatedBFloat16) == 0x3f80,
            "Explicit-policy rank-zero BFloat16 construction did not truncate.");

    struct TieCase {
        double value;
        int32_t expected;
    };
    const std::array<TieCase, 6> ties{{
        {0.5, 0},
        {1.5, 2},
        {2.5, 2},
        {-0.5, 0},
        {-1.5, -2},
        {-2.5, -2},
    }};
    for (const auto& tie : ties)
        require(convertScalar<int32_t>(tie.value, rejectNearestEven) == tie.expected,
                "Nearest-even tie rounding mismatch.");

    require(convertScalar<int8_t>(int16_t{-128}, rejectTowardZero) == -128 &&
                convertScalar<int8_t>(int16_t{127}, rejectTowardZero) == 127,
            "Signed integer boundary conversion mismatch.");
    requireThrows<std::overflow_error>(
        [&] { (void)convertScalar<int8_t>(int16_t{-129}, rejectTowardZero); },
        "Signed integer conversion accepted one below its minimum.");
    requireThrows<std::overflow_error>(
        [&] { (void)convertScalar<int8_t>(int16_t{128}, rejectTowardZero); },
        "Signed integer conversion accepted one above its maximum.");

    require(convertScalar<uint8_t>(uint16_t{0}, rejectTowardZero) == 0 &&
                convertScalar<uint8_t>(uint16_t{255}, rejectTowardZero) == 255,
            "Unsigned integer boundary conversion mismatch.");
    requireThrows<std::overflow_error>(
        [&] { (void)convertScalar<uint8_t>(int16_t{-1}, rejectTowardZero); },
        "Unsigned integer conversion accepted a negative value.");
    requireThrows<std::overflow_error>(
        [&] { (void)convertScalar<uint8_t>(uint16_t{256}, rejectTowardZero); },
        "Unsigned integer conversion accepted one above its maximum.");

    require(convertScalar<int64_t>(std::numeric_limits<int64_t>::min(), rejectTowardZero) ==
                    std::numeric_limits<int64_t>::min() &&
                convertScalar<int64_t>(std::numeric_limits<int64_t>::max(), rejectTowardZero) ==
                    std::numeric_limits<int64_t>::max() &&
                convertScalar<uint64_t>(std::numeric_limits<uint64_t>::max(), rejectTowardZero) ==
                    std::numeric_limits<uint64_t>::max(),
            "Wide integer boundary conversion mismatch.");
    requireThrows<std::overflow_error>(
        [&] {
            (void)convertScalar<int64_t>(std::numeric_limits<uint64_t>::max(), rejectTowardZero);
        },
        "Signed conversion accepted UInt64 maximum.");

    const double twoTo63 = std::ldexp(1.0, 63);
    const double twoTo64 = std::ldexp(1.0, 64);
    const double belowTwoTo64 = std::nextafter(twoTo64, 0.0);
    require(
        convertScalar<int64_t>(-twoTo63, rejectTowardZero) == std::numeric_limits<int64_t>::min(),
        "Float-to-Int64 minimum conversion mismatch.");
    requireThrows<std::overflow_error>(
        [&] { (void)convertScalar<int64_t>(twoTo63, rejectTowardZero); },
        "Float-to-Int64 conversion accepted its exclusive upper bound.");
    requireThrows<std::overflow_error>(
        [&] {
            (void)convertScalar<int64_t>(
                std::nextafter(-twoTo63, -std::numeric_limits<double>::infinity()),
                rejectTowardZero);
        },
        "Float-to-Int64 conversion accepted one representable value below its minimum.");
    require(convertScalar<uint64_t>(belowTwoTo64, rejectTowardZero) ==
                std::numeric_limits<uint64_t>::max() - uint64_t{2047},
            "Float-to-UInt64 maximum representable value mismatch.");
    requireThrows<std::overflow_error>(
        [&] { (void)convertScalar<uint64_t>(twoTo64, rejectTowardZero); },
        "Float-to-UInt64 conversion accepted its exclusive upper bound.");

    require(convertScalar<int8_t>(int16_t{-129}, saturateNearestEven) == -128 &&
                convertScalar<int8_t>(int16_t{128}, saturateNearestEven) == 127 &&
                convertScalar<uint8_t>(int16_t{-1}, saturateNearestEven) == 0 &&
                convertScalar<uint8_t>(uint16_t{256}, saturateNearestEven) == 255,
            "Saturating integer conversion mismatch.");
    require(convertScalar<int8_t>(std::numeric_limits<double>::infinity(), saturateNearestEven) ==
                    127 &&
                convertScalar<int8_t>(-std::numeric_limits<double>::infinity(),
                                      saturateNearestEven) == -128 &&
                convertScalar<uint8_t>(-std::numeric_limits<double>::infinity(),
                                       saturateNearestEven) == 0,
            "Saturating infinity conversion mismatch.");

    require(convertScalar<int8_t>(int16_t{128}, wrapNearestEven) == -128 &&
                convertScalar<int8_t>(int16_t{129}, wrapNearestEven) == -127 &&
                convertScalar<int8_t>(int16_t{-129}, wrapNearestEven) == 127 &&
                convertScalar<uint8_t>(int16_t{-1}, wrapNearestEven) == 255 &&
                convertScalar<uint8_t>(uint16_t{256}, wrapNearestEven) == 0 &&
                convertScalar<uint8_t>(258.6, wrapNearestEven) == 3,
            "Modulo-wrap integer conversion mismatch.");
    require(convertScalar<int64_t>(std::numeric_limits<uint64_t>::max(), wrapNearestEven) == -1 &&
                convertScalar<uint64_t>(std::numeric_limits<int64_t>::min(), wrapNearestEven) ==
                    (uint64_t{1} << 63) &&
                convertScalar<uint64_t>(-1.0, wrapNearestEven) ==
                    std::numeric_limits<uint64_t>::max() &&
                convertScalar<int64_t>(twoTo63, wrapNearestEven) ==
                    std::numeric_limits<int64_t>::min() &&
                convertScalar<uint64_t>(
                    std::nextafter(twoTo64, std::numeric_limits<double>::infinity()),
                    wrapNearestEven) == 4096,
            "Wide modulo-wrap conversion mismatch.");

    for (const IntegerOverflow overflow :
         {IntegerOverflow::Reject, IntegerOverflow::Saturate, IntegerOverflow::ModuloWrap}) {
        const ScalarConversionOptions options{IntegerRounding::NearestEven, overflow};
        requireThrows<std::domain_error>(
            [&] {
                (void)convertScalar<int32_t>(std::numeric_limits<double>::quiet_NaN(), options);
            },
            "Integer conversion accepted NaN.");
    }
    requireThrows<std::overflow_error>(
        [&] {
            (void)convertScalar<int32_t>(std::numeric_limits<double>::infinity(),
                                         rejectNearestEven);
        },
        "Rejecting integer conversion accepted infinity.");
    requireThrows<std::overflow_error>(
        [&] {
            (void)convertScalar<int32_t>(std::numeric_limits<double>::infinity(), wrapNearestEven);
        },
        "Modulo-wrap integer conversion accepted infinity.");

    require(convertScalar<int32_t>(std::complex<double>{3.5, 0.0}, rejectNearestEven) == 4 &&
                convertScalar<double>(std::complex<double>{-2.25, -0.0}, rejectTowardZero) == -2.25,
            "Zero-imaginary complex-to-real conversion mismatch.");
    requireThrows<std::domain_error>(
        [&] { (void)convertScalar<int32_t>(std::complex<double>{3.5, 1.0}, rejectNearestEven); },
        "Integer conversion discarded a nonzero imaginary component.");
    requireThrows<std::domain_error>(
        [&] { (void)convertScalar<double>(std::complex<double>{3.5, 1.0}, rejectTowardZero); },
        "Floating conversion discarded a nonzero imaginary component.");
}

}  // namespace scalar_codec_test
