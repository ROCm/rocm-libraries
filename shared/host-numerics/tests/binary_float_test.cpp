// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
void testExhaustiveBinaryFormat(ScalarType type) {
    const uint32_t count = 1U << scalarTypeInfo(type).storageBits;
    for (uint32_t raw = 0; raw < count; ++raw) {
        const Tensor tensor = tensorFromRaw(type, raw);
        const float observed = tensor.loadAs<float>({0});
        const float expected = expectedBinaryDecode(type, raw);
        if (std::isnan(expected)) {
            require(std::isnan(observed), "Binary format NaN decode mismatch.");
        } else if (std::isinf(expected)) {
            require(observed == expected, "Binary format infinity decode mismatch.");
        } else {
            require(observed == expected, "Binary format finite decode mismatch.");
            const uint32_t canonicalRaw = type == ScalarType::E4M3 ? raw & 0x7fU : raw;
            require(encodeRaw(type, observed) == canonicalRaw,
                    "Binary format finite round-trip mismatch.");
        }
    }
}

void testEveryBinaryFormatRoundingBoundary(ScalarType type, uint32_t maximumPositiveRaw,
                                           uint32_t signMask, bool hasSignedZero = true) {
    for (uint32_t upperRaw = 1; upperRaw <= maximumPositiveRaw; ++upperRaw) {
        const uint32_t lowerRaw = upperRaw - 1U;
        const float lower = expectedBinaryDecode(type, lowerRaw);
        const float upper = expectedBinaryDecode(type, upperRaw);
        const float midpoint =
            static_cast<float>((static_cast<double>(lower) + static_cast<double>(upper)) * 0.5);
        const uint32_t midpointRaw = (lowerRaw & 1U) == 0 ? lowerRaw : upperRaw;

        require(encodeRaw(type, std::nextafter(midpoint, lower)) == lowerRaw,
                "Binary format value below a rounding boundary encoded incorrectly.");
        require(encodeRaw(type, midpoint) == midpointRaw,
                "Binary format midpoint did not round to even.");
        require(encodeRaw(type, std::nextafter(midpoint, upper)) == upperRaw,
                "Binary format value above a rounding boundary encoded incorrectly.");

        if (signMask != 0) {
            const uint32_t negativeLowerRaw =
                lowerRaw == 0 && !hasSignedZero ? 0 : signMask | lowerRaw;
            require(encodeRaw(type, -std::nextafter(midpoint, lower)) == negativeLowerRaw,
                    "Negative binary format value below a rounding boundary encoded "
                    "incorrectly.");
            const uint32_t negativeMidpointRaw =
                midpointRaw == 0 && !hasSignedZero ? 0 : signMask | midpointRaw;
            require(encodeRaw(type, -midpoint) == negativeMidpointRaw,
                    "Negative binary format midpoint did not round to even.");
            require(encodeRaw(type, -std::nextafter(midpoint, upper)) == (signMask | upperRaw),
                    "Negative binary format value above a rounding boundary encoded "
                    "incorrectly.");
        }
    }
}

void testScalarEncodings() {
    using namespace roc::host_numerics;
    const int64_t exactInteger = 9'007'199'254'740'993;
    const Tensor integerScalar(exactInteger);
    require(
        integerScalar.type() == ScalarType::Int64 && integerScalar.item<int64_t>() == exactInteger,
        "Rank-zero tensor did not preserve an integer above 2^53.");

    const Tensor overflowingScalar(128.0);
    require(overflowingScalar.item<int8_t>() == -128,
            "No-options Tensor item conversion did not use modulo overflow semantics.");
    requireThrows<std::overflow_error>(
        [&] { (void)overflowingScalar.item<int8_t>(ScalarConversionOptions{}); },
        "Option-bearing Tensor item conversion did not reject integer overflow.");

    const std::complex<float> complexValue{1.25f, -2.5f};
    const Tensor complexScalar(complexValue);
    require(complexScalar.type() == ScalarType::ComplexFloat32 &&
                complexScalar.item<std::complex<float>>() == complexValue,
            "Rank-zero tensor did not preserve a complex native value.");

    // Float6 retains the cross-byte packing coverage while a rank-zero tensor
    // preserves all caller-owned storage bits.
    const std::array<std::byte, 1> float6Storage{std::byte{0xcc}};
    const Tensor float6Scalar = Tensor::copyEncodedBackingStorage(
        ScalarType::Float6E3M2, Layout::contiguousLastDimensionFastest(Shape{}), float6Storage);
    require(float6Scalar.item<float>() == 1.0f,
            "Rank-zero tensor did not decode a packed Float6 value.");
    require(std::to_integer<uint8_t>(float6Scalar.rawEncodedBackingStorage()[0]) == 0xcc,
            "Rank-zero tensor did not preserve packed padding bits.");
    require(Tensor::scalar(ScalarType::Float6E2M3, 0).item<float>() == 0.0f &&
                Tensor::scalar(ScalarType::Float6E2M3, 1).item<float>() == 1.0f,
            "Rank-zero tensor zero/one construction failed for a packed type.");
    require(scalarElementGroupSize(ScalarType::Float32) == 1 &&
                scalarElementGroupSize(ScalarType::Int4) == 2 &&
                scalarElementGroupSize(ScalarType::Float6E2M3) == 4,
            "Scalar element group sizes do not match byte-addressable groups.");

    bool invalidScalarStorageRejected = false;
    try {
        (void)Tensor::copyEncodedBackingStorage(
            ScalarType::Float6E3M2, Layout::contiguousLastDimensionFastest(Shape{}), {});
    } catch (const std::invalid_argument&) {
        invalidScalarStorageRejected = true;
    }
    require(invalidScalarStorageRejected, "Rank-zero tensor accepted incorrectly sized storage.");

    const std::array<float, 16> fp4Expected{
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    std::vector<std::byte> fp4Raw(8);
    for (uint8_t index = 0; index < 16; index += 2)
        fp4Raw[index / 2] = static_cast<std::byte>(index | ((index + 1) << 4));
    const Tensor fp4Decoded = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Float4E2M1, Layout::contiguousLastDimensionFastest(Shape{16}),
        std::move(fp4Raw));
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        require(fp4Decoded.loadAs<float>({index}) == fp4Expected[index],
                "FP4 exhaustive decode mismatch.");

    Tensor fp4Encoded(ScalarType::Float4E2M1, Shape{16});
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        fp4Encoded.storeFrom({index}, fp4Expected[index]);
    for (uint8_t index = 0; index < 16; ++index) {
        const uint8_t byte =
            std::to_integer<uint8_t>(fp4Encoded.rawEncodedBackingStorage()[index / 2]);
        const uint8_t raw = (index & 1) ? byte >> 4 : byte & 0xf;
        require(raw == index, "FP4 exhaustive encode mismatch.");
    }

    std::vector<std::byte> int4Raw(8);
    for (uint8_t index = 0; index < 16; index += 2)
        int4Raw[index / 2] = static_cast<std::byte>(index | ((index + 1) << 4));
    const Tensor int4 = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::Int4, Layout::contiguousLastDimensionFastest(Shape{16}), std::move(int4Raw));
    for (uint8_t index = 0; index < 16; ++index) {
        const int32_t expected = index < 8 ? index : static_cast<int32_t>(index) - 16;
        require(int4.loadAs<int32_t>({index}) == expected, "Int4 exhaustive decode mismatch.");
    }

    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        std::vector<std::byte> storage{
            static_cast<std::byte>(raw & 0xff),
            static_cast<std::byte>(raw >> 8),
        };
        const Tensor value = Tensor::takeOwnershipOfEncodedBackingStorage(
            ScalarType::Float16, Layout::contiguousLastDimensionFastest(Shape{1}),
            std::move(storage));
        const float decoded = value.loadAs<float>({0});
        Tensor roundTrip(ScalarType::Float16, Shape{1});
        roundTrip.storeFrom({0}, decoded);
        const uint16_t encoded = bytesToUint16(roundTrip.rawEncodedBackingStorage());
        if (std::isnan(decoded)) {
            require((encoded & 0x7c00U) == 0x7c00U && (encoded & 0x03ffU) != 0,
                    "Float16 NaN did not remain NaN.");
        } else {
            require(encoded == raw, "Float16 exhaustive round-trip mismatch.");
        }
    }

    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        const Tensor value = tensorFromRaw(ScalarType::BFloat16, raw);
        const float decoded = value.loadAs<float>({0});
        Tensor roundTrip(ScalarType::BFloat16, Shape{1});
        roundTrip.storeFrom({0}, decoded);
        const uint16_t encoded = bytesToUint16(roundTrip.rawEncodedBackingStorage());
        if (std::isnan(decoded)) {
            require((encoded & 0x7f80U) == 0x7f80U && (encoded & 0x007fU) != 0,
                    "BFloat16 NaN did not remain NaN.");
        } else {
            require(encoded == raw, "BFloat16 exhaustive round-trip mismatch.");
        }
    }

    testExhaustiveBinaryFormat(ScalarType::Float4E2M1);
    testExhaustiveBinaryFormat(ScalarType::Float6E2M3);
    testExhaustiveBinaryFormat(ScalarType::Float6E3M2);
    testExhaustiveBinaryFormat(ScalarType::Float8E4M3);
    testExhaustiveBinaryFormat(ScalarType::Float8E5M2);
    testExhaustiveBinaryFormat(ScalarType::Float8E4M3Fnuz);
    testExhaustiveBinaryFormat(ScalarType::Float8E5M2Fnuz);
    testExhaustiveBinaryFormat(ScalarType::E5M3);
    testExhaustiveBinaryFormat(ScalarType::E4M3);

    testEveryBinaryFormatRoundingBoundary(ScalarType::Float4E2M1, 0x07, 0x08);
    testEveryBinaryFormatRoundingBoundary(ScalarType::Float6E2M3, 0x1f, 0x20);
    testEveryBinaryFormatRoundingBoundary(ScalarType::Float6E3M2, 0x1f, 0x20);
    testEveryBinaryFormatRoundingBoundary(ScalarType::Float8E4M3, 0x7e, 0x80);
    testEveryBinaryFormatRoundingBoundary(ScalarType::Float8E5M2, 0x7b, 0x80);
    testEveryBinaryFormatRoundingBoundary(ScalarType::Float8E4M3Fnuz, 0x7f, 0x80, false);
    testEveryBinaryFormatRoundingBoundary(ScalarType::Float8E5M2Fnuz, 0x7f, 0x80, false);
    testEveryBinaryFormatRoundingBoundary(ScalarType::E5M3, 0xfe, 0);
    testEveryBinaryFormatRoundingBoundary(ScalarType::E4M3, 0x7e, 0);

    require(encodeRaw(ScalarType::Float8E4M3, 1.0625f) == 0x38,
            "FP8 E4M3 lower-even midpoint rounding mismatch.");
    require(encodeRaw(ScalarType::Float8E4M3, 1.1875f) == 0x3a,
            "FP8 E4M3 upper-even midpoint rounding mismatch.");
    require(encodeRaw(ScalarType::Float8E4M3, 1000.0f) == 0x7e &&
                encodeRaw(ScalarType::Float8E4M3, -1000.0f) == 0xfe,
            "FP8 E4M3 saturation mismatch.");
    require(encodeRaw(ScalarType::Float8E5M2, std::numeric_limits<float>::infinity()) == 0x7c,
            "FP8 E5M2 infinity encoding mismatch.");
    require(encodeRaw(ScalarType::Float8E5M2, 1.0e10f) == 0x7b,
            "FP8 E5M2 finite saturation mismatch.");
    require(encodeRaw(ScalarType::Float8E4M3Fnuz, std::numeric_limits<float>::infinity()) == 0x7f,
            "FP8 FNUZ saturation mismatch.");
    require(encodeRaw(ScalarType::Float8E4M3, -std::ldexp(1.0f, -20)) == 0x80 &&
                encodeRaw(ScalarType::Float8E4M3Fnuz, -std::ldexp(1.0f, -20)) == 0x00,
            "FP8 underflow zero-sign mismatch.");
    require(encodeRaw(ScalarType::Float8E4M3, std::bit_cast<float>(uint32_t{0xffc00000})) == 0xff,
            "FP8 OCP NaN sign mismatch.");
    require(
        encodeRaw(ScalarType::Float4E2M1, std::numeric_limits<float>::quiet_NaN()) == 0x07 &&
            encodeRaw(ScalarType::Float4E2M1, std::bit_cast<float>(uint32_t{0xffc00000})) == 0x0f,
        "Finite-only minifloat NaN saturation mismatch.");
    require(encodeRaw(ScalarType::E5M3, 1.0f) == 0x78 && encodeRaw(ScalarType::E5M3, 2.0f) == 0x80,
            "E5M3 scale encoding mismatch.");
    require(encodeRaw(ScalarType::E4M3, 1.0f) == 0x38 && encodeRaw(ScalarType::E4M3, 2.0f) == 0x40,
            "E4M3 scale encoding mismatch.");
    require(encodeRaw(ScalarType::E5M3, -0.0f) == 0x00 &&
                encodeRaw(ScalarType::E4M3, -0.0f) == 0x00 &&
                encodeRaw(ScalarType::E8M0, -0.0f) == 0x00,
            "Unsigned scale negative-zero encoding mismatch.");

    for (const ScalarType scaleType : {ScalarType::E5M3, ScalarType::E4M3}) {
        bool negativeScaleThrew = false;
        try {
            (void)encodeRaw(scaleType, -1.0f);
        } catch (const std::domain_error&) {
            negativeScaleThrew = true;
        }
        require(negativeScaleThrew, "Negative scale did not fail.");
    }

    std::vector<std::byte> e8Raw{std::byte{0},   std::byte{1},   std::byte{127},
                                 std::byte{128}, std::byte{254}, std::byte{255}};
    const Tensor e8 = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::E8M0, Layout::contiguousLastDimensionFastest(Shape{6}), std::move(e8Raw));
    require(e8.loadAs<float>({0}) == std::ldexp(1.0f, -127), "E8M0 minimum mismatch.");
    require(e8.loadAs<float>({1}) == std::ldexp(1.0f, -126), "E8M0 exponent mismatch.");
    require(e8.loadAs<float>({2}) == 1.0f, "E8M0 unity mismatch.");
    require(e8.loadAs<float>({3}) == 2.0f, "E8M0 exponent mismatch.");
    require(e8.loadAs<float>({4}) == std::ldexp(1.0f, 127), "E8M0 maximum mismatch.");
    require(std::isnan(e8.loadAs<float>({5})), "E8M0 NaN mismatch.");
    for (uint32_t raw = 0; raw < 0xffU; ++raw) {
        const Tensor value = tensorFromRaw(ScalarType::E8M0, raw);
        require(encodeRaw(ScalarType::E8M0, value.loadAs<float>({0})) == raw,
                "E8M0 finite round-trip mismatch.");
    }
    for (uint32_t lowerRaw = 0; lowerRaw < 0xfeU; ++lowerRaw) {
        const float lower = std::ldexp(1.0f, static_cast<int>(lowerRaw) - 127);
        const float upper = std::ldexp(1.0f, static_cast<int>(lowerRaw) - 126);
        const float midpoint =
            static_cast<float>((static_cast<double>(lower) + static_cast<double>(upper)) * 0.5);
        require(encodeRaw(ScalarType::E8M0, std::nextafter(midpoint, lower)) == lowerRaw,
                "E8M0 value below a rounding boundary encoded incorrectly.");
        require(encodeRaw(ScalarType::E8M0, midpoint) ==
                    ((lowerRaw & 1U) == 0 ? lowerRaw : lowerRaw + 1U),
                "E8M0 midpoint did not round to an even code.");
        require(encodeRaw(ScalarType::E8M0, std::nextafter(midpoint, upper)) == lowerRaw + 1U,
                "E8M0 value above a rounding boundary encoded incorrectly.");
    }
    require(encodeRaw(ScalarType::E8M0, 0.0f) == 0, "E8M0 zero saturation mismatch.");

    std::vector<std::byte> e8ZeroRaw{std::byte{0}, std::byte{1}, std::byte{127}, std::byte{255}};
    const Tensor e8Zero = Tensor::takeOwnershipOfEncodedBackingStorage(
        ScalarType::E8M0Zero, Layout::contiguousLastDimensionFastest(Shape{4}),
        std::move(e8ZeroRaw));
    require(e8Zero.loadAs<float>({0}) == 0.0f, "E8M0Zero zero mismatch.");
    require(e8Zero.loadAs<float>({1}) == std::ldexp(1.0f, -126),
            "E8M0Zero minimum positive mismatch.");
    require(e8Zero.loadAs<float>({2}) == 1.0f, "E8M0Zero unity mismatch.");
    require(std::isnan(e8Zero.loadAs<float>({3})), "E8M0Zero NaN mismatch.");
    for (uint32_t raw = 1; raw < 0xffU; ++raw) {
        const Tensor value = tensorFromRaw(ScalarType::E8M0Zero, raw);
        require(encodeRaw(ScalarType::E8M0Zero, value.loadAs<float>({0})) == raw,
                "E8M0Zero finite round-trip mismatch.");
    }
    require(encodeRaw(ScalarType::E8M0Zero, 0.0f) == 0, "E8M0Zero zero encoding mismatch.");
    require(encodeRaw(ScalarType::E8M0Zero, 1.75f) == 127 &&
                encodeRaw(ScalarType::E8M0Zero, -2.0f) == 128 &&
                encodeRaw(ScalarType::E8M0Zero, std::numeric_limits<float>::infinity()) == 255,
            "E8M0Zero exponent-field encoding mismatch.");
}
}  // namespace scalar_codec_test
