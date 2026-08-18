// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using roc::host_validation::convertScalar;
using roc::host_validation::IntegerOverflow;
using roc::host_validation::IntegerRounding;
using roc::host_validation::Layout;
using roc::host_validation::Scalar;
using roc::host_validation::ScalarCategory;
using roc::host_validation::ScalarConversionOptions;
using roc::host_validation::ScalarType;
using roc::host_validation::scalarTypeInfo;
using roc::host_validation::Shape;
using roc::host_validation::Tensor;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const char* message) {
    bool threw = false;
    try {
        std::forward<Function>(function)();
    } catch (const Exception&) {
        threw = true;
    }
    require(threw, message);
}

class RoundingModeRestore {
   public:
    RoundingModeRestore() : m_original(std::fegetround()) {}

    ~RoundingModeRestore() {
        if (m_original != -1) std::fesetround(m_original);
    }

   private:
    int m_original;
};

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
    tensor.storeFrom({0}, value);
    return tensorRaw(tensor);
}

struct ExpectedBinaryFormat {
    uint8_t exponentBits;
    uint8_t mantissaBits;
    int exponentBias;
    uint8_t totalBits;
    bool hasSign;
};

// Keep the test oracle independent of the production tag traits so adding a format requires an
// explicit expectation rather than copying the implementation's metadata path.
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
        case ScalarType::E4M3:
            return {4, 3, 7, 7, false};
        default:
            throw std::invalid_argument("No expected binary format.");
    }
}

// This is likewise an independent encoding oracle; replacing it with visitScalarType would make
// exhaustive tests agree with the implementation by construction.
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
        case ScalarType::E4M3:
            return (raw & 0x7fU) == 0x7fU;
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

    const uint32_t payloadMask = (1U << format.totalBits) - 1U;
    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw & payloadMask;
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

void testScalarTypeInfoContract() {
    const auto& boolean = scalarTypeInfo(ScalarType::Boolean);
    require(boolean.name == "bool" && boolean.category == ScalarCategory::Boolean &&
                boolean.storageBits == 8 && boolean.exponentBits == 0 &&
                boolean.mantissaBits == 0 && boolean.exponentBias == 0 && !boolean.supportsNaN &&
                !boolean.supportsInfinity,
            "Boolean scalar metadata contract mismatch.");

    const auto& complex = scalarTypeInfo(ScalarType::ComplexFloat32);
    require(complex.name == "c64" && complex.category == ScalarCategory::Complex &&
                complex.storageBits == 64 && complex.exponentBits == 8 &&
                complex.mantissaBits == 23 && complex.exponentBias == 127 && complex.supportsNaN &&
                complex.supportsInfinity,
            "Complex scalar metadata contract mismatch.");

    const auto& finiteFloat = scalarTypeInfo(ScalarType::Float4E2M1);
    require(finiteFloat.category == ScalarCategory::FloatingPoint && finiteFloat.storageBits == 4 &&
                finiteFloat.exponentBits == 2 && finiteFloat.mantissaBits == 1 &&
                finiteFloat.exponentBias == 1 && !finiteFloat.supportsNaN &&
                !finiteFloat.supportsInfinity,
            "Finite minifloat metadata contract mismatch.");

    // Int12 is intentionally retained as design-generality coverage for cross-byte packing.
    const auto& int12 = scalarTypeInfo(ScalarType::Int12);
    require(int12.name == "i12" && int12.category == ScalarCategory::SignedInteger &&
                int12.storageBits == 12 && int12.isPacked(),
            "Int12 generality metadata contract mismatch.");

    requireThrows<std::invalid_argument>(
        [] { (void)Scalar::zero(ScalarType::Count); },
        "Runtime scalar construction accepted the Count sentinel.");
}

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

void testIntegerConversionPrimitives() {
    const ScalarConversionOptions defaults;
    require(defaults.integerRounding == IntegerRounding::TowardZero &&
                defaults.integerOverflow == IntegerOverflow::Reject,
            "Scalar conversion option defaults changed.");

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

void testNearestEvenIgnoresHostRoundingMode() {
    const RoundingModeRestore restore;
    const ScalarConversionOptions nearestEvenReject{
        IntegerRounding::NearestEven,
        IntegerOverflow::Reject,
    };
    const ScalarConversionOptions nearestEvenSaturate{
        IntegerRounding::NearestEven,
        IntegerOverflow::Saturate,
    };
    const ScalarConversionOptions nearestEvenWrap{
        IntegerRounding::NearestEven,
        IntegerOverflow::ModuloWrap,
    };

    struct Case {
        double value;
        int32_t expected;
    };
    const std::array<Case, 8> cases{{
        {0.5, 0},
        {1.5, 2},
        {2.5, 2},
        {3.5, 4},
        {-0.5, 0},
        {-1.5, -2},
        {-2.5, -2},
        {-3.5, -4},
    }};
    for (const int mode : {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO}) {
        require(std::fesetround(mode) == 0, "Host did not accept a standard rounding mode.");
        for (const auto& test : cases) {
            volatile double value = test.value;
            require(convertScalar<int32_t>(value, nearestEvenReject) == test.expected,
                    "Nearest-even conversion depended on the host rounding mode.");
        }

        volatile double positiveBoundaryTie = 127.5;
        volatile double negativeBoundaryTie = -128.5;
        require(convertScalar<int8_t>(positiveBoundaryTie, nearestEvenSaturate) == 127 &&
                    convertScalar<int8_t>(positiveBoundaryTie, nearestEvenWrap) == -128 &&
                    convertScalar<int8_t>(negativeBoundaryTie, nearestEvenReject) == -128,
                "Boundary tie conversion depended on the host rounding mode.");
    }
}

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
        Tensor::fromNativeValues<double>(Shape{sourceValues.size()}, sourceValues);
    const Tensor saturated = source.to(ScalarType::Int8, saturateNearestEven);
    const Tensor wrapped = source.to(ScalarType::Int8, wrapNearestEven);
    const std::array<int8_t, 4> saturatedExpected{127, 127, -128, -128};
    const std::array<int8_t, 4> wrappedExpected{-128, -128, -128, 126};
    for (size_t index = 0; index < sourceValues.size(); ++index) {
        require(saturated.loadAs<int8_t>({index}) == saturatedExpected[index],
                "Tensor saturating conversion mismatch.");
        require(wrapped.loadAs<int8_t>({index}) == wrappedExpected[index],
                "Tensor modulo-wrap conversion mismatch.");
    }
    requireThrows<std::overflow_error>(
        [&] { (void)source.to(ScalarType::Int8, rejectNearestEven); },
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
    const Tensor saturatedFactory = Tensor::fromValues<int16_t>(
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

    const Scalar zeroImaginary = Scalar::from(std::complex<double>{3.5, 0.0});
    require(zeroImaginary.as<int32_t>(rejectNearestEven) == 4,
            "Runtime scalar zero-imaginary conversion mismatch.");
    const Scalar nonzeroImaginary = Scalar::from(std::complex<double>{3.5, 1.0});
    requireThrows<std::domain_error>(
        [&] { (void)nonzeroImaginary.as<double>(rejectNearestEven); },
        "Runtime scalar conversion discarded a nonzero imaginary component.");

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
}  // namespace

int main() {
    using namespace roc::host_validation;

    testScalarTypeInfoContract();
    testIntegerConversionPrimitives();
    testNearestEvenIgnoresHostRoundingMode();
    testIntegerCodecPolicies();

    const int64_t exactInteger = 9'007'199'254'740'993;
    const Scalar integerScalar = Scalar::from(exactInteger);
    require(
        integerScalar.type() == ScalarType::Int64 && integerScalar.as<int64_t>() == exactInteger,
        "Runtime scalar did not preserve an integer above 2^53.");

    const std::complex<float> complexValue{1.25f, -2.5f};
    const Scalar complexScalar = Scalar::from(complexValue);
    require(complexScalar.type() == ScalarType::ComplexFloat32 &&
                complexScalar.as<std::complex<float>>() == complexValue,
            "Runtime scalar did not preserve a complex native value.");

    // Int12 is deliberate generality coverage: its second element would straddle byte boundaries
    // in a tensor, while this scalar case also verifies unused high-bit canonicalization.
    const std::array<std::byte, 2> int12Storage{std::byte{0x2e}, std::byte{0xfb}};
    const Scalar int12Scalar = Scalar::fromStorage(ScalarType::Int12, int12Storage);
    require(int12Scalar.as<int32_t>() == -1234,
            "Runtime scalar did not decode a packed Int12 value.");
    require(std::to_integer<uint8_t>(int12Scalar.storage()[1]) == 0x0b,
            "Runtime scalar did not clear packed padding bits.");
    require(Scalar::zero(ScalarType::Float6E2M3).as<float>() == 0.0f &&
                Scalar::one(ScalarType::Float6E2M3).as<float>() == 1.0f,
            "Runtime scalar zero/one construction failed for a packed type.");
    require(scalarElementGroupSize(ScalarType::Float32) == 1 &&
                scalarElementGroupSize(ScalarType::Int4) == 2 &&
                scalarElementGroupSize(ScalarType::Float6E2M3) == 4 &&
                scalarElementGroupSize(ScalarType::Int12) == 2,
            "Scalar element group sizes do not match byte-addressable groups.");

    bool invalidScalarStorageRejected = false;
    try {
        (void)Scalar::fromStorage(ScalarType::Int12,
                                  std::span<const std::byte>(int12Storage).first(1));
    } catch (const std::invalid_argument&) {
        invalidScalarStorageRejected = true;
    }
    require(invalidScalarStorageRejected, "Runtime scalar accepted incorrectly sized storage.");

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
        require(fp4Decoded.loadAs<float>({index}) == fp4Expected[index],
                "FP4 exhaustive decode mismatch.");

    Tensor fp4Encoded(ScalarType::Float4E2M1, Shape{16});
    for (size_t index = 0; index < fp4Expected.size(); ++index)
        fp4Encoded.storeFrom({index}, fp4Expected[index]);
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
        require(int4.loadAs<int32_t>({index}) == expected, "Int4 exhaustive decode mismatch.");
    }

    for (uint32_t raw = 0; raw <= 0xffff; ++raw) {
        std::vector<std::byte> storage{
            static_cast<std::byte>(raw & 0xff),
            static_cast<std::byte>(raw >> 8),
        };
        const Tensor value = Tensor::fromStorage(ScalarType::Float16, Layout::contiguous(Shape{1}),
                                                 std::move(storage));
        const float decoded = value.loadAs<float>({0});
        Tensor roundTrip(ScalarType::Float16, Shape{1});
        roundTrip.storeFrom({0}, decoded);
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
        const float decoded = value.loadAs<float>({0});
        Tensor roundTrip(ScalarType::BFloat16, Shape{1});
        roundTrip.storeFrom({0}, decoded);
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
    testExhaustiveBinaryFormat(ScalarType::E4M3);

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
    const Tensor e8 =
        Tensor::fromStorage(ScalarType::E8M0, Layout::contiguous(Shape{6}), std::move(e8Raw));
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
    require(encodeRaw(ScalarType::E8M0, 0.0f) == 0, "E8M0 zero saturation mismatch.");

    return 0;
}
