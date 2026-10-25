// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
template <typename Source>
concept ConvertsWithoutPolicies = requires(Source source) { convertScalar<int32_t>(source); };

void testIntegerConversionPrimitives() {
    static_assert(!std::is_default_constructible_v<ScalarConversionOptions>);
    static_assert(!std::is_constructible_v<ScalarConversionOptions, IntegerRounding>);
    static_assert(!ConvertsWithoutPolicies<double>);
    const ScalarConversionOptions nearestBFloat16{
        IntegerRounding::TowardZero, IntegerOverflow::Reject, BFloat16Rounding::NearestEven};

    constexpr float bfloat16RoundingBoundary = 1.005859375f;
    Tensor nearestEven(ScalarType::BFloat16, Shape{1});
    nearestEven.storeFrom({0}, bfloat16RoundingBoundary, nearestBFloat16);
    require(tensorRaw(nearestEven) == 0x3f81,
            "Explicit BFloat16 conversion did not round to nearest even.");

    const ScalarConversionOptions truncateBFloat16{
        IntegerRounding::TowardZero, IntegerOverflow::Reject, BFloat16Rounding::Truncate};
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
    require(convertScalar<int8_t>(
                128.0, {IntegerRounding::TowardZero, IntegerOverflow::ModuloWrap}) == -128,
            "Explicit modulo overflow conversion mismatch.");

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

    require(convertScalar<uint8_t>(uint16_t{0}, rejectTowardZero) == 0 &&
                convertScalar<uint8_t>(uint16_t{255}, rejectTowardZero) == 255,
            "Unsigned integer boundary conversion mismatch.");

    require(convertScalar<int64_t>(std::numeric_limits<int64_t>::min(), rejectTowardZero) ==
                    std::numeric_limits<int64_t>::min() &&
                convertScalar<int64_t>(std::numeric_limits<int64_t>::max(), rejectTowardZero) ==
                    std::numeric_limits<int64_t>::max() &&
                convertScalar<uint64_t>(std::numeric_limits<uint64_t>::max(), rejectTowardZero) ==
                    std::numeric_limits<uint64_t>::max(),
            "Wide integer boundary conversion mismatch.");

    const double twoTo63 = std::ldexp(1.0, 63);
    const double twoTo64 = std::ldexp(1.0, 64);
    const double belowTwoTo64 = std::nextafter(twoTo64, 0.0);
    require(
        convertScalar<int64_t>(-twoTo63, rejectTowardZero) == std::numeric_limits<int64_t>::min(),
        "Float-to-Int64 minimum conversion mismatch.");
    require(convertScalar<uint64_t>(belowTwoTo64, rejectTowardZero) ==
                std::numeric_limits<uint64_t>::max() - uint64_t{2047},
            "Float-to-UInt64 maximum representable value mismatch.");

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

    require(convertScalar<int32_t>(std::complex<double>{3.5, 0.0}, rejectNearestEven) == 4 &&
                convertScalar<double>(std::complex<double>{-2.25, -0.0}, rejectTowardZero) == -2.25,
            "Zero-imaginary complex-to-real conversion mismatch.");
}

}  // namespace scalar_codec_test
