// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <string>
#include <utility>

#include "_scalar_codec_test_support.hpp"

namespace scalar_codec_test {
void testScalarTypeInfoContract() {
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
        requireThrows<std::runtime_error>(
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
        requireThrows<std::runtime_error>(
            [invalid] { (void)Tensor::scalar(invalid, 0); },
            "Rank-zero tensor construction accepted an invalid scalar type.");
        requireThrows<std::runtime_error>(
            [invalid] { (void)Tensor(invalid, Shape{0}); },
            "Empty tensor construction accepted an invalid scalar type.");
        requireThrows<std::runtime_error>(
            [invalid] { (void)Tensor::allocateUninitialized(invalid, Shape{0}); },
            "Uninitialized tensor allocation accepted an invalid scalar type.");
    }
}

}  // namespace scalar_codec_test
