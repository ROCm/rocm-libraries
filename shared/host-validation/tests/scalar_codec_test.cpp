// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using roc::host_validation::Layout;
using roc::host_validation::ScalarType;
using roc::host_validation::scalarTypeInfo;
using roc::host_validation::Shape;
using roc::host_validation::Tensor;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

uint16_t bytesToUint16(std::span<const std::byte> bytes) {
    return static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[0])) |
           static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[1])) << 8;
}

uint32_t rawMask(ScalarType type) {
    const uint16_t bits = scalarTypeInfo(type).storageBits;
    return bits == 32 ? std::numeric_limits<uint32_t>::max() : (1U << bits) - 1U;
}

Tensor tensorFromRaw(ScalarType type, uint32_t raw) {
    const size_t bytes = (scalarTypeInfo(type).storageBits + 7) / 8;
    std::vector<std::byte> storage(bytes);
    for (size_t index = 0; index < bytes; ++index)
        storage[index] = static_cast<std::byte>((raw >> (index * 8)) & 0xffU);
    return Tensor::fromStorage(type, Layout::contiguous(Shape{1}), std::move(storage));
}

uint32_t tensorRaw(const Tensor& tensor) {
    uint32_t raw = 0;
    for (size_t index = 0; index < tensor.storage().size(); ++index)
        raw |= static_cast<uint32_t>(std::to_integer<uint8_t>(tensor.storage()[index]))
               << (index * 8);
    return raw & rawMask(tensor.type());
}

uint32_t encodeRaw(ScalarType type, float value) {
    Tensor tensor(type, Shape{1});
    tensor.mutableView().storeFrom({0}, value);
    return tensorRaw(tensor);
}

struct ExpectedBinaryFormat {
    uint8_t exponentBits;
    uint8_t mantissaBits;
    int exponentBias;
    uint8_t totalBits;
    bool hasSign;
};

ExpectedBinaryFormat expectedFormat(ScalarType type) {
    switch (type) {
        case ScalarType::Float4E2M1:
            return {2, 1, 1, 4, true};
        case ScalarType::Float6E2M3:
            return {2, 3, 1, 6, true};
        case ScalarType::Float6E3M2:
            return {3, 2, 3, 6, true};
        case ScalarType::Float8E4M3:
            return {4, 3, 7, 8, true};
        case ScalarType::Float8E5M2:
            return {5, 2, 15, 8, true};
        case ScalarType::Float8E4M3Fnuz:
            return {4, 3, 8, 8, true};
        case ScalarType::Float8E5M2Fnuz:
            return {5, 2, 16, 8, true};
        case ScalarType::E5M3:
            return {5, 3, 15, 8, false};
        default:
            throw std::invalid_argument("No expected binary format.");
    }
}

bool expectedNaN(ScalarType type, uint32_t raw) {
    switch (type) {
        case ScalarType::Float8E4M3:
            return (raw & 0x7fU) == 0x7fU;
        case ScalarType::Float8E5M2:
            return (raw & 0x7fU) > 0x7cU;
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
            return raw == 0x80U;
        case ScalarType::E5M3:
        case ScalarType::E8M0:
            return raw == 0xffU;
        default:
            return false;
    }
}

bool expectedInfinity(ScalarType type, uint32_t raw) {
    return type == ScalarType::Float8E5M2 && (raw & 0x7fU) == 0x7cU;
}

float expectedBinaryDecode(ScalarType type, uint32_t raw) {
    if (expectedNaN(type, raw)) return std::numeric_limits<float>::quiet_NaN();

    const auto format = expectedFormat(type);
    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const bool negative = format.hasSign && (raw & signMask) != 0;
    if (expectedInfinity(type, raw))
        return negative ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();

    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw;
    const uint32_t exponentMask = (1U << format.exponentBits) - 1U;
    const uint32_t mantissaMask = (1U << format.mantissaBits) - 1U;
    const uint32_t exponent = (magnitude >> format.mantissaBits) & exponentMask;
    const uint32_t mantissa = magnitude & mantissaMask;
    const float fraction =
        static_cast<float>(mantissa) / static_cast<float>(1U << format.mantissaBits);
    const float positive =
        exponent == 0
            ? std::ldexp(fraction, 1 - format.exponentBias)
            : std::ldexp(1.0f + fraction, static_cast<int>(exponent) - format.exponentBias);
    return negative ? -positive : positive;
}

void testExhaustiveBinaryFormat(ScalarType type) {
    const uint32_t count = 1U << scalarTypeInfo(type).storageBits;
    for (uint32_t raw = 0; raw < count; ++raw) {
        const Tensor tensor = tensorFromRaw(type, raw);
        const float observed = tensor.view().loadAs<float>({0});
        const float expected = expectedBinaryDecode(type, raw);
        if (std::isnan(expected)) {
            require(std::isnan(observed), "Binary format NaN decode mismatch.");
        } else if (std::isinf(expected)) {
            require(observed == expected, "Binary format infinity decode mismatch.");
        } else {
            require(observed == expected, "Binary format finite decode mismatch.");
            require(encodeRaw(type, observed) == raw, "Binary format finite round-trip mismatch.");
        }
    }
}
}  // namespace

int main() {
    using namespace roc::host_validation;

    const std::array<float, 16> fp4Expected{
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    std::vector<std::byte> fp4Raw(8);
    for (uint8_t index = 0; index < 16; index += 2)
        fp4Raw[index / 2] = static_cast<std::byte>(index | ((index + 1) << 4));
    const Tensor fp4Decoded = Tensor::fromStorage(ScalarType::Float4E2M1,
                                                  Layout::contiguous(Shape{16}), std::move(fp4Raw));
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        require(fp4Decoded.view().loadAs<float>({index}) == fp4Expected[index],
                "FP4 exhaustive decode mismatch.");

    Tensor fp4Encoded(ScalarType::Float4E2M1, Shape{16});
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        fp4Encoded.mutableView().storeFrom({index}, fp4Expected[index]);
    for (uint8_t index = 0; index < 16; ++index) {
        const uint8_t byte = std::to_integer<uint8_t>(fp4Encoded.storage()[index / 2]);
        const uint8_t raw = (index & 1) ? byte >> 4 : byte & 0xf;
        require(raw == index, "FP4 exhaustive encode mismatch.");
    }

    std::vector<std::byte> int4Raw(8);
    for (uint8_t index = 0; index < 16; index += 2)
        int4Raw[index / 2] = static_cast<std::byte>(index | ((index + 1) << 4));
    const Tensor int4 =
        Tensor::fromStorage(ScalarType::Int4, Layout::contiguous(Shape{16}), std::move(int4Raw));
    for (uint8_t index = 0; index < 16; ++index) {
        const int32_t expected = index < 8 ? index : static_cast<int32_t>(index) - 16;
        require(int4.view().loadAs<int32_t>({index}) == expected,
                "Int4 exhaustive decode mismatch.");
    }

    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        std::vector<std::byte> storage{
            static_cast<std::byte>(raw & 0xff),
            static_cast<std::byte>(raw >> 8),
        };
        const Tensor value = Tensor::fromStorage(ScalarType::Float16, Layout::contiguous(Shape{1}),
                                                 std::move(storage));
        const float decoded = value.view().loadAs<float>({0});
        Tensor roundTrip(ScalarType::Float16, Shape{1});
        roundTrip.mutableView().storeFrom({0}, decoded);
        const uint16_t encoded = bytesToUint16(roundTrip.storage());
        if (std::isnan(decoded)) {
            require((encoded & 0x7c00U) == 0x7c00U && (encoded & 0x03ffU) != 0,
                    "Float16 NaN did not remain NaN.");
        } else {
            require(encoded == raw, "Float16 exhaustive round-trip mismatch.");
        }
    }

    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        const Tensor value = tensorFromRaw(ScalarType::BFloat16, raw);
        const float decoded = value.view().loadAs<float>({0});
        Tensor roundTrip(ScalarType::BFloat16, Shape{1});
        roundTrip.mutableView().storeFrom({0}, decoded);
        const uint16_t encoded = bytesToUint16(roundTrip.storage());
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
    require(
        encodeRaw(ScalarType::E5M3, -0.0f) == 0x00 && encodeRaw(ScalarType::E8M0, -0.0f) == 0x00,
        "Unsigned scale negative-zero encoding mismatch.");

    bool negativeScaleThrew = false;
    try {
        (void)encodeRaw(ScalarType::E5M3, -1.0f);
    } catch (const std::domain_error&) {
        negativeScaleThrew = true;
    }
    require(negativeScaleThrew, "Negative E5M3 scale did not fail.");

    std::vector<std::byte> e8Raw{std::byte{0},   std::byte{1},   std::byte{127},
                                 std::byte{128}, std::byte{254}, std::byte{255}};
    const Tensor e8 =
        Tensor::fromStorage(ScalarType::E8M0, Layout::contiguous(Shape{6}), std::move(e8Raw));
    require(e8.view().loadAs<float>({0}) == std::ldexp(1.0f, -127), "E8M0 minimum mismatch.");
    require(e8.view().loadAs<float>({1}) == std::ldexp(1.0f, -126), "E8M0 exponent mismatch.");
    require(e8.view().loadAs<float>({2}) == 1.0f, "E8M0 unity mismatch.");
    require(e8.view().loadAs<float>({3}) == 2.0f, "E8M0 exponent mismatch.");
    require(e8.view().loadAs<float>({4}) == std::ldexp(1.0f, 127), "E8M0 maximum mismatch.");
    require(std::isnan(e8.view().loadAs<float>({5})), "E8M0 NaN mismatch.");
    for (uint32_t raw = 0; raw < 0xffU; ++raw) {
        const Tensor value = tensorFromRaw(ScalarType::E8M0, raw);
        require(encodeRaw(ScalarType::E8M0, value.view().loadAs<float>({0})) == raw,
                "E8M0 finite round-trip mismatch.");
    }
    require(encodeRaw(ScalarType::E8M0, 0.0f) == 0, "E8M0 zero saturation mismatch.");

    return 0;
}
