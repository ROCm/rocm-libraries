// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <utility>

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
void testScalarTypeInfoContract() {
    static_assert(nativeScalarType<bool> == ScalarType::Boolean);
    static_assert(nativeScalarType<uint8_t> == ScalarType::UInt8);
    static_assert(nativeScalarType<int8_t> == ScalarType::Int8);
    static_assert(nativeScalarType<uint16_t> == ScalarType::UInt16);
    static_assert(nativeScalarType<int16_t> == ScalarType::Int16);
    static_assert(nativeScalarType<uint32_t> == ScalarType::UInt32);
    static_assert(nativeScalarType<int32_t> == ScalarType::Int32);
    static_assert(nativeScalarType<uint64_t> == ScalarType::UInt64);
    static_assert(nativeScalarType<int64_t> == ScalarType::Int64);
    static_assert(nativeScalarType<float> == ScalarType::Float32);
    static_assert(nativeScalarType<double> == ScalarType::Float64);
    static_assert(nativeScalarType<std::complex<float>> == ScalarType::ComplexFloat32);
    static_assert(nativeScalarType<std::complex<double>> == ScalarType::ComplexFloat64);
    static_assert(nativeScalarType<const float&> == ScalarType::Float32);

    // Check each identifier independently of its position in the metadata table.
    constexpr std::pair<ScalarType, std::string_view> expectedNames[] = {
        {ScalarType::Boolean, "bool"},
        {ScalarType::Int4, "i4"},
        {ScalarType::Int8, "i8"},
        {ScalarType::Int16, "i16"},
        {ScalarType::Int32, "i32"},
        {ScalarType::Int64, "i64"},
        {ScalarType::UInt8, "u8"},
        {ScalarType::UInt16, "u16"},
        {ScalarType::UInt32, "u32"},
        {ScalarType::UInt64, "u64"},
        {ScalarType::Float4E2M1, "f4e2m1"},
        {ScalarType::Float6E2M3, "f6e2m3"},
        {ScalarType::Float6E3M2, "f6e3m2"},
        {ScalarType::Float8E4M3, "f8e4m3"},
        {ScalarType::Float8E5M2, "f8e5m2"},
        {ScalarType::Float8E4M3Fnuz, "f8e4m3fnuz"},
        {ScalarType::Float8E5M2Fnuz, "f8e5m2fnuz"},
        {ScalarType::E4M3, "e4m3"},
        {ScalarType::E5M3, "e5m3"},
        {ScalarType::E8M0, "e8m0"},
        {ScalarType::E8M0Zero, "e8m0_zero"},
        {ScalarType::Float16, "f16"},
        {ScalarType::BFloat16, "bf16"},
        {ScalarType::Float32, "f32"},
        {ScalarType::Float64, "f64"},
        {ScalarType::ComplexFloat32, "c64"},
        {ScalarType::ComplexFloat64, "c128"},
    };
    static_assert(std::size(expectedNames) == scalarTypeCount);
    for (const auto& [type, name] : expectedNames) {
        require(scalarTypeName(type) == name, "Scalar identifier has the wrong metadata row.");
        require(scalarTypeInfo(type).category < ScalarCategory::Count,
                "Scalar metadata has an invalid category.");
    }

    for (ScalarType type :
         {ScalarType::E4M3, ScalarType::E5M3, ScalarType::E8M0, ScalarType::E8M0Zero}) {
        const auto& info = scalarTypeInfo(type);
        require(info.category == ScalarCategory::FloatingPoint && !info.isSigned,
                "Unsigned floating-point metadata mismatch.");
    }
    require(scalarTypeInfo(ScalarType::Float32).isSigned &&
                scalarTypeInfo(ScalarType::ComplexFloat32).isSigned &&
                scalarTypeInfo(ScalarType::Int4).isSigned &&
                !scalarTypeInfo(ScalarType::UInt8).isSigned &&
                !scalarTypeInfo(ScalarType::Boolean).isSigned,
            "Scalar signedness metadata mismatch.");

    require(scalarTypeCount == scalarTypeInfos.size(),
            "Scalar type count and metadata table size differ.");
    for (size_t index = 0; index < scalarTypeCount; ++index) {
        const ScalarType type = static_cast<ScalarType>(index);
        require(isConcreteScalarType(type), "Concrete scalar type was classified as a sentinel.");
        require(visitScalarType(type, [type]<typename Tag>() { return Tag::type == type; }),
                "Scalar type visitor and dense enum ordering differ.");
    }
    for (ScalarType invalid : {ScalarType::Count, static_cast<ScalarType>(0xffff)}) {
        require(!isConcreteScalarType(invalid), "Invalid scalar type classified as concrete.");
        requireThrows<std::invalid_argument>(
            [invalid] { (void)visitScalarType(invalid, []<typename>() { return true; }); },
            "Scalar visitor accepted an invalid identifier.");
    }

    require(visitScalarType(ScalarType::Float32,
                            []<typename Tag>() {
                                return Tag::type == ScalarType::Float32 &&
                                       std::is_same_v<typename Tag::Storage, float>;
                            }),
            "Runtime scalar tag dispatch mismatch.");
    require(visitScalarType(ScalarType::Int4,
                            []<typename Tag>() {
                                return Tag::type == ScalarType::Int4 &&
                                       std::is_void_v<typename Tag::Storage>;
                            }),
            "Packed scalar tag dispatch mismatch.");

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

    // Six-bit elements exercise non-power-of-two storage that crosses byte boundaries.
    const auto& float6 = scalarTypeInfo(ScalarType::Float6E3M2);
    require(float6.name == "f6e3m2" && float6.category == ScalarCategory::FloatingPoint &&
                float6.storageBits == 6 && float6.isPacked(),
            "Float6 cross-byte metadata contract mismatch.");

    for (ScalarType invalid : {ScalarType::Count, static_cast<ScalarType>(0xffff)}) {
        requireThrows<std::invalid_argument>(
            [invalid] { (void)Tensor::scalar(invalid, 0); },
            "Rank-zero tensor construction accepted an invalid scalar type.");
        requireThrows<std::invalid_argument>(
            [invalid] { (void)Tensor(invalid, Shape{0}); },
            "Empty tensor construction accepted an invalid scalar type.");
        requireThrows<std::invalid_argument>(
            [invalid] { (void)Tensor::allocateUninitialized(invalid, Shape{0}); },
            "Uninitialized tensor allocation accepted an invalid scalar type.");
    }
}

}  // namespace scalar_codec_test
