/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "DataTypes.hpp"

namespace rocRoller
{
    template <typename T>
    std::string friendlyTypeName()
    {
        if constexpr(CHasTypeInfo<T>)
        {
            return TypeInfo<T>::Name();
        }
        else
        {
            return typeName<T>();
        }
    }

    inline constexpr DataType getIntegerType(bool isSigned, int sizeBytes)
    {
        if(isSigned)
        {
            switch(sizeBytes)
            {
            case 1:
                return DataType::Int8;
            case 2:
                return DataType::Int16;
            case 4:
                return DataType::Int32;
            case 8:
                return DataType::Int64;
            }
        }
        else
        {
            switch(sizeBytes)
            {
            case 1:
                return DataType::UInt8;
            case 2:
                return DataType::UInt16;
            case 4:
                return DataType::UInt32;
            case 8:
                return DataType::UInt64;
            }
        }

        auto prefix = isSigned ? "signed" : "unsigned";

        Throw<FatalError>(
            "No enumeration for ", prefix, " integer with size ", sizeBytes, " bytes.");

        // cppcheck doesn't seem to notice that Throw<>() is marked [[noreturn]] so it will
        // complain if this isn't here.
        return DataType::None;
    }

    template <>
    inline DataType fromString<DataType>(std::string const& str)
    {
        using myInt   = std::underlying_type_t<DataType>;
        auto maxValue = static_cast<myInt>(DataType::Count);
        for(myInt i = 0; i < maxValue; ++i)
        {
            auto        val     = static_cast<DataType>(i);
            std::string testStr = toString(val);

            if(std::equal(
                   str.begin(), str.end(), testStr.begin(), testStr.end(), [](auto a, auto b) {
                       return std::tolower(a) == std::tolower(b);
                   }))
                return val;
        }

        // Special cases
        std::string strCopy = str;
        std::transform(strCopy.begin(), strCopy.end(), strCopy.begin(), ::tolower);

        if(strCopy == "fp16")
        {
            return DataType::Half;
        }
        if(strCopy == "bf16")
        {
            return DataType::BFloat16;
        }

        Throw<FatalError>(
            "Invalid fromString: type name: ", typeName<DataType>(), ", string input: ", str);

        // Unreachable code
        return DataType::None;
    }

    inline constexpr VariableType::VariableType()
        : dataType()
    {
    }
    inline constexpr VariableType::VariableType(VariableType const& v)
        : dataType(v.dataType)
        , pointerType(v.pointerType)
    {
    }

    inline constexpr VariableType::VariableType(DataType d)
        : dataType(d)
        , pointerType(PointerType::Value)
    {
    }
    inline constexpr VariableType::VariableType(DataType d, PointerType p)
        : dataType(d)
        , pointerType(p)
    {
    }
    inline constexpr VariableType::VariableType(PointerType p)
        : dataType()
        , pointerType(p)
    {
    }

    inline bool VariableType::isPointer() const
    {
        return pointerType != PointerType::Value;
    }
    inline bool VariableType::isGlobalPointer() const
    {
        return pointerType == PointerType::PointerGlobal;
    }

    inline VariableType VariableType::getDereferencedType() const
    {
        return VariableType(dataType);
    }

    inline VariableType VariableType::getPointer() const
    {
        AssertFatal(pointerType == PointerType::Value, ShowValue(pointerType));
        return VariableType(dataType, PointerType::PointerGlobal);
    }

    inline DataType VariableType::getArithmeticType() const
    {
        if(pointerType == PointerType::Value)
            return dataType;

        return getIntegerType(false, getElementSize());
    }

    inline bool CompareVariableTypesPointersEqual::operator()(VariableType const& lhs,
                                                              VariableType const& rhs) const
    {
        if(lhs.pointerType < rhs.pointerType)
            return true;

        if(lhs.pointerType == rhs.pointerType && lhs.pointerType == PointerType::Value)
            return lhs.dataType < rhs.dataType;

        return false;
    }

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr VariableType BaseTypeInfo<T,
                                        T_DEnum,
                                        T_SegmentType,
                                        T_PEnum,
                                        T_Packing,
                                        T_RegCount,
                                        T_Bits,
                                        T_IsComplex,
                                        T_IsIntegral,
                                        T_IsSigned>::Var;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr VariableType BaseTypeInfo<T,
                                        T_DEnum,
                                        T_SegmentType,
                                        T_PEnum,
                                        T_Packing,
                                        T_RegCount,
                                        T_Bits,
                                        T_IsComplex,
                                        T_IsIntegral,
                                        T_IsSigned>::SegmentVariableType;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_Bits,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::ElementBytes;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_Bits,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::ElementBits;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_Bits,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::Packing;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr size_t BaseTypeInfo<T,
                                  T_DEnum,
                                  T_SegmentType,
                                  T_PEnum,
                                  T_Packing,
                                  T_RegCount,
                                  T_Bits,
                                  T_IsComplex,
                                  T_IsIntegral,
                                  T_IsSigned>::RegisterCount;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr bool BaseTypeInfo<T,
                                T_DEnum,
                                T_SegmentType,
                                T_PEnum,
                                T_Packing,
                                T_RegCount,
                                T_Bits,
                                T_IsComplex,
                                T_IsIntegral,
                                T_IsSigned>::IsComplex;

    template <typename T,
              DataType    T_DEnum,
              DataType    T_SegmentType,
              PointerType T_PEnum,
              int         T_Packing,
              int         T_RegCount,
              int         T_Bits,
              bool        T_IsComplex,
              bool        T_IsIntegral,
              bool        T_IsSigned>
    constexpr bool BaseTypeInfo<T,
                                T_DEnum,
                                T_SegmentType,
                                T_PEnum,
                                T_Packing,
                                T_RegCount,
                                T_Bits,
                                T_IsComplex,
                                T_IsIntegral,
                                T_IsSigned>::IsIntegral;

#define DeclareDefaultValueTypeInfo(dtype, enumVal)                         \
    template <>                                                             \
    struct TypeInfo<dtype> : public BaseTypeInfo<dtype,                     \
                                                 DataType::enumVal,         \
                                                 DataType::enumVal,         \
                                                 PointerType::Value,        \
                                                 1,                         \
                                                 1,                         \
                                                 sizeof(dtype) * 8,         \
                                                 false,                     \
                                                 std::is_integral_v<dtype>, \
                                                 std::is_signed_v<dtype>>   \
    {                                                                       \
    }

    DeclareDefaultValueTypeInfo(float, Float);

    DeclareDefaultValueTypeInfo(int8_t, Int8);
    DeclareDefaultValueTypeInfo(int16_t, Int16);
    DeclareDefaultValueTypeInfo(int32_t, Int32);

    DeclareDefaultValueTypeInfo(uint8_t, UInt8);
    DeclareDefaultValueTypeInfo(uint16_t, UInt16);
    DeclareDefaultValueTypeInfo(uint32_t, UInt32);

#undef DeclareDefaultValueTypeInfo

    template <>
    struct TypeInfo<uint64_t> : public BaseTypeInfo<uint64_t,
                                                    DataType::UInt64,
                                                    DataType::UInt64,
                                                    PointerType::Value,
                                                    1,
                                                    2,
                                                    64,
                                                    false,
                                                    true,
                                                    false>
    {
    };

    template <>
    struct TypeInfo<int64_t> : public BaseTypeInfo<int64_t,
                                                   DataType::Int64,
                                                   DataType::Int64,
                                                   PointerType::Value,
                                                   1,
                                                   2,
                                                   64,
                                                   false,
                                                   true,
                                                   true>
    {
    };

    template <>
    struct TypeInfo<double> : public BaseTypeInfo<double,
                                                  DataType::Double,
                                                  DataType::Double,
                                                  PointerType::Value,
                                                  1,
                                                  2,
                                                  64,
                                                  false,
                                                  false,
                                                  true>
    {
    };

    template <>
    struct TypeInfo<std::complex<float>> : public BaseTypeInfo<std::complex<float>,
                                                               DataType::ComplexFloat,
                                                               DataType::ComplexFloat,
                                                               PointerType::Value,
                                                               1,
                                                               2,
                                                               64,
                                                               true,
                                                               false,
                                                               true>
    {
    };

    template <>
    struct TypeInfo<std::complex<double>> : public BaseTypeInfo<std::complex<double>,
                                                                DataType::ComplexDouble,
                                                                DataType::ComplexDouble,
                                                                PointerType::Value,
                                                                1,
                                                                4,
                                                                128,
                                                                true,
                                                                false,
                                                                true>
    {
    };

    template <>
    struct TypeInfo<Int8x4> : public BaseTypeInfo<Int8x4,
                                                  DataType::Int8x4,
                                                  DataType::Int8,
                                                  PointerType::Value,
                                                  4,
                                                  1,
                                                  32,
                                                  false,
                                                  true,
                                                  true>
    {
    };

    template <>
    struct TypeInfo<UInt8x4> : public BaseTypeInfo<UInt8x4,
                                                   DataType::UInt8x4,
                                                   DataType::UInt8,
                                                   PointerType::Value,
                                                   4,
                                                   1,
                                                   32,
                                                   false,
                                                   true,
                                                   false>
    {
    };

    template <>
    struct TypeInfo<Half> : public BaseTypeInfo<Half,
                                                DataType::Half,
                                                DataType::Half,
                                                PointerType::Value,
                                                1,
                                                1,
                                                16,
                                                false,
                                                false,
                                                true>
    {
    };

    template <>
    struct TypeInfo<Halfx2> : public BaseTypeInfo<Halfx2,
                                                  DataType::Halfx2,
                                                  DataType::Half,
                                                  PointerType::Value,
                                                  2,
                                                  1,
                                                  32,
                                                  false,
                                                  false,
                                                  true>
    {
    };

    template <>
    struct TypeInfo<FP8> : public BaseTypeInfo<FP8,
                                               DataType::FP8,
                                               DataType::FP8,
                                               PointerType::Value,
                                               1,
                                               1,
                                               8,
                                               false,
                                               false,
                                               true>
    {
    };

    template <>
    struct TypeInfo<FP8x4> : public BaseTypeInfo<FP8x4,
                                                 DataType::FP8x4,
                                                 DataType::FP8,
                                                 PointerType::Value,
                                                 4,
                                                 1,
                                                 32,
                                                 false,
                                                 false,
                                                 true>
    {
    };

    template <>
    struct TypeInfo<BF8> : public BaseTypeInfo<BF8,
                                               DataType::BF8,
                                               DataType::BF8,
                                               PointerType::Value,
                                               1,
                                               1,
                                               8,
                                               false,
                                               false,
                                               true>
    {
    };

    template <>
    struct TypeInfo<BF8x4> : public BaseTypeInfo<BF8x4,
                                                 DataType::BF8x4,
                                                 DataType::BF8,
                                                 PointerType::Value,
                                                 4,
                                                 1,
                                                 32,
                                                 false,
                                                 false,
                                                 true>
    {
    };

    template <>
    struct TypeInfo<FP6> : public BaseTypeInfo<FP6,
                                               DataType::FP6,
                                               DataType::FP6,
                                               PointerType::Value,
                                               1,
                                               1,
                                               6,
                                               false,
                                               false,
                                               true>
    {
    };

    template <>
    struct TypeInfo<FP6x16> : public BaseTypeInfo<FP6x16,
                                                  DataType::FP6x16,
                                                  DataType::FP6,
                                                  PointerType::Value,
                                                  16,
                                                  3,
                                                  96,
                                                  false,
                                                  false,
                                                  false>
    {
    };

    template <>
    struct TypeInfo<BF6> : public BaseTypeInfo<BF6,
                                               DataType::BF6,
                                               DataType::BF6,
                                               PointerType::Value,
                                               1,
                                               1,
                                               6,
                                               false,
                                               false,
                                               true>
    {
    };

    template <>
    struct TypeInfo<BF6x16> : public BaseTypeInfo<BF6x16,
                                                  DataType::BF6x16,
                                                  DataType::BF6,
                                                  PointerType::Value,
                                                  16,
                                                  3,
                                                  96,
                                                  false,
                                                  false,
                                                  false>
    {
    };

    template <>
    struct TypeInfo<FP4> : public BaseTypeInfo<FP4,
                                               DataType::FP4,
                                               DataType::FP4,
                                               PointerType::Value,
                                               1,
                                               1,
                                               4,
                                               false,
                                               false,
                                               true>
    {
    };

    template <>
    struct TypeInfo<FP4x8> : public BaseTypeInfo<FP4x8,
                                                 DataType::FP4x8,
                                                 DataType::FP4,
                                                 PointerType::Value,
                                                 8,
                                                 1,
                                                 32,
                                                 false,
                                                 false,
                                                 false>
    {
    };

    template <>
    struct TypeInfo<BFloat16> : public BaseTypeInfo<BFloat16,
                                                    DataType::BFloat16,
                                                    DataType::BFloat16,
                                                    PointerType::Value,
                                                    1,
                                                    1,
                                                    16,
                                                    false,
                                                    false,
                                                    true>
    {
    };

    template <>
    struct TypeInfo<BFloat16x2> : public BaseTypeInfo<BFloat16x2,
                                                      DataType::BFloat16x2,
                                                      DataType::BFloat16,
                                                      PointerType::Value,
                                                      2,
                                                      1,
                                                      32,
                                                      false,
                                                      false,
                                                      true>
    {
    };

    template <>
    struct TypeInfo<Raw32> : public BaseTypeInfo<Raw32,
                                                 DataType::Raw32,
                                                 DataType::Raw32,
                                                 PointerType::Value,
                                                 1,
                                                 1,
                                                 32,
                                                 false,
                                                 true,
                                                 false>
    {
    };

    template <>
    struct TypeInfo<bool> : public BaseTypeInfo<bool,
                                                DataType::Bool,
                                                DataType::Bool,
                                                PointerType::Value,
                                                1,
                                                1,
                                                1,
                                                false,
                                                true,
                                                false>
    {
    };

    template <>
    struct TypeInfo<Bool32> : public BaseTypeInfo<Bool32,
                                                  DataType::Bool32,
                                                  DataType::Bool32,
                                                  PointerType::Value,
                                                  1,
                                                  1,
                                                  32,
                                                  false,
                                                  false,
                                                  false>
    {
    };

    template <>
    struct TypeInfo<Bool64> : public BaseTypeInfo<Bool64,
                                                  DataType::Bool64,
                                                  DataType::Bool64,
                                                  PointerType::Value,
                                                  1,
                                                  2,
                                                  64,
                                                  false,
                                                  false,
                                                  false>
    {
    };

    template <>
    struct TypeInfo<PointerLocal> : public BaseTypeInfo<PointerLocal,
                                                        DataType::None,
                                                        DataType::None,
                                                        PointerType::PointerLocal,
                                                        1,
                                                        1,
                                                        32,
                                                        false,
                                                        true,
                                                        false>
    {
    };

    template <>
    struct TypeInfo<PointerGlobal> : public BaseTypeInfo<PointerGlobal,
                                                         DataType::None,
                                                         DataType::None,
                                                         PointerType::PointerGlobal,
                                                         1,
                                                         2,
                                                         64,
                                                         false,
                                                         true,
                                                         false>
    {
    };

    template <>
    struct TypeInfo<E8M0> : public BaseTypeInfo<E8M0,
                                                DataType::E8M0,
                                                DataType::E8M0,
                                                PointerType::Value,
                                                1,
                                                1,
                                                8,
                                                false,
                                                true,
                                                false>
    {
    };

    template <>
    struct TypeInfo<E8M0x4> : public BaseTypeInfo<E8M0x4,
                                                  DataType::E8M0x4,
                                                  DataType::E8M0,
                                                  PointerType::Value,
                                                  4,
                                                  1,
                                                  32,
                                                  false,
                                                  false,
                                                  false>
    {
    };

    template <>
    struct TypeInfo<Buffer> : public BaseTypeInfo<Buffer,
                                                  DataType::None,
                                                  DataType::None,
                                                  PointerType::Buffer,
                                                  1,
                                                  4,
                                                  128,
                                                  false,
                                                  true,
                                                  false>
    {
    };

#define DeclareEnumTypeInfo(typeEnum, dtype)                         \
    template <>                                                      \
    struct EnumTypeInfo<DataType::typeEnum> : public TypeInfo<dtype> \
    {                                                                \
    }
    DeclareEnumTypeInfo(Float, float);
    DeclareEnumTypeInfo(Double, double);
    DeclareEnumTypeInfo(ComplexFloat, std::complex<float>);
    DeclareEnumTypeInfo(ComplexDouble, std::complex<double>);
    DeclareEnumTypeInfo(Half, Half);
    DeclareEnumTypeInfo(Halfx2, Halfx2);
    DeclareEnumTypeInfo(FP8, FP8);
    DeclareEnumTypeInfo(FP8x4, FP8x4);
    DeclareEnumTypeInfo(BF8, BF8);
    DeclareEnumTypeInfo(BF8x4, BF8x4);
    DeclareEnumTypeInfo(FP6, FP6);
    DeclareEnumTypeInfo(FP6x16, FP6x16);
    DeclareEnumTypeInfo(BF6, BF6);
    DeclareEnumTypeInfo(BF6x16, BF6x16);
    DeclareEnumTypeInfo(FP4, FP4);
    DeclareEnumTypeInfo(FP4x8, FP4x8);
    DeclareEnumTypeInfo(Int8x4, Int8x4);
    DeclareEnumTypeInfo(Int8, int8_t);
    DeclareEnumTypeInfo(Int16, int16_t);
    DeclareEnumTypeInfo(Int32, int32_t);
    DeclareEnumTypeInfo(Int64, int64_t);
    DeclareEnumTypeInfo(BFloat16, BFloat16);
    DeclareEnumTypeInfo(BFloat16x2, BFloat16x2);
    DeclareEnumTypeInfo(Raw32, Raw32);
    DeclareEnumTypeInfo(UInt8x4, UInt8x4);
    DeclareEnumTypeInfo(UInt8, uint8_t);
    DeclareEnumTypeInfo(UInt16, uint16_t);
    DeclareEnumTypeInfo(UInt32, uint32_t);
    DeclareEnumTypeInfo(UInt64, uint64_t);
    DeclareEnumTypeInfo(Bool, bool);
    DeclareEnumTypeInfo(Bool32, Bool32);
    DeclareEnumTypeInfo(Bool64, Bool64);
    DeclareEnumTypeInfo(E8M0, E8M0);
    DeclareEnumTypeInfo(E8M0x4, E8M0x4);

#undef DeclareEnumTypeInfo

}
