// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    require(scalarTypeCount == scalarTypeInfos.size(),
            "Scalar type count and metadata table size differ.");
    for (size_t index = 0; index < scalarTypeCount; ++index) {
        const ScalarType type = static_cast<ScalarType>(index);
        require(isConcreteScalarType(type), "Concrete scalar type was classified as a sentinel.");
        require(visitScalarType(type, [type]<typename Tag>() { return Tag::type == type; }),
                "Scalar type visitor and dense enum ordering differ.");
    }
    require(!isConcreteScalarType(ScalarType::Count),
            "ScalarType::Count was classified as a concrete scalar type.");
    requireThrows<std::invalid_argument>([] { (void)scalarTypeInfo(ScalarType::Count); },
                                         "Scalar metadata accepted the Count sentinel.");
    requireThrows<std::invalid_argument>(
        [] { (void)visitScalarType(ScalarType::Count, []<typename>() { return true; }); },
        "Scalar visitor accepted the Count sentinel.");

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

    requireThrows<std::invalid_argument>(
        [] { (void)Tensor::scalar(ScalarType::Count, 0); },
        "Rank-zero tensor construction accepted the Count sentinel.");
}

}  // namespace scalar_codec_test
