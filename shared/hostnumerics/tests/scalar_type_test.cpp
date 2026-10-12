// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <initializer_list>
#include <roc/hostnumerics/scalar_type.hpp>
#include <stdexcept>
#include <utility>

int main() {
    using namespace roc::hostnumerics;
    const auto require = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };

    static_assert(scalarTypeInfo(ScalarType::Float32).exponentBits == 8);

    // IEEE binary16 has a five-bit exponent, ten fraction bits, and bias 15.
    const auto& half = scalarTypeInfo(ScalarType::Float16);
    require(half.name == "f16" && half.category == ScalarCategory::FloatingPoint &&
                half.storageBits == 16 && half.exponentBits == 5 && half.mantissaBits == 10 &&
                half.exponentBias == 15 && half.supportsNaN && half.supportsInfinity &&
                !half.isPacked(),
            "Incorrect binary16 metadata.");
    const auto& complex = scalarTypeInfo(ScalarType::ComplexFloat32);
    require(complex.category == ScalarCategory::Complex && complex.storageBits == 64 &&
                complex.exponentBits == 8 && complex.mantissaBits == 23,
            "Complex metadata must count both components' storage.");
    const auto& packed = scalarTypeInfo(ScalarType::Float6E3M2);
    require(packed.storageBits == 6 && packed.isPacked() && !packed.supportsNaN &&
                !packed.supportsInfinity,
            "Incorrect finite six-bit encoding metadata.");
    require(scalarTypeInfo(ScalarType::Int4).category == ScalarCategory::SignedInteger &&
                scalarTypeInfo(ScalarType::E8M0).category == ScalarCategory::FloatingPoint,
            "Incorrect integer or floating-point category.");

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

    for (ScalarType invalid : {ScalarType::Count, static_cast<ScalarType>(0xffff)}) {
        require(!isConcreteScalarType(invalid), "Invalid scalar type classified as concrete.");
    }
}
