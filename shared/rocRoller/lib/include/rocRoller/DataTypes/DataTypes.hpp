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

#include <cassert>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include <rocRoller/DataTypes/DataTypes_BF6.hpp>
#include <rocRoller/DataTypes/DataTypes_BF8.hpp>
#include <rocRoller/DataTypes/DataTypes_BFloat16.hpp>
#include <rocRoller/DataTypes/DataTypes_E8M0.hpp>
#include <rocRoller/DataTypes/DataTypes_E8M0x4.hpp>
#include <rocRoller/DataTypes/DataTypes_FP4.hpp>
#include <rocRoller/DataTypes/DataTypes_FP6.hpp>
#include <rocRoller/DataTypes/DataTypes_FP8.hpp>
#include <rocRoller/DataTypes/DataTypes_Half.hpp>
#include <rocRoller/DataTypes/DataTypes_Int8.hpp>
#include <rocRoller/DataTypes/DataTypes_Int8x4.hpp>
#include <rocRoller/DataTypes/DataTypes_UInt8x4.hpp>

#include <rocRoller/GPUArchitecture/GPUArchitecture_fwd.hpp>

#include <rocRoller/InstructionValues/Register_fwd.hpp>

#include <rocRoller/Utilities/Comparison.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    /**
     * \ingroup rocRoller
     * \defgroup DataTypes Data Type Info
     *
     * @brief Definitions and metadata on supported data types.
     */

    enum class DataDirection
    {
        ReadOnly = 0,
        WriteOnly,
        ReadWrite,
        Count
    };

    std::string   toString(DataDirection dir);
    std::ostream& operator<<(std::ostream& stream, DataDirection dir);

    /**
     * \ingroup DataTypes
     * @{
     */

    /**
     * Data Type
     *
     * A value of `None` generically means any/unspecified/deferred.
     */
    enum class DataType : int
    {
        Float = 0, //< 32bit floating point
        Double, //< 64bit floating point
        ComplexFloat, //< Two 32bit floating point; real and imaginary
        ComplexDouble, //< Two 64bit floating point; real and imaginary
        Half, //< 16bit floating point (IEEE format)
        Halfx2, //< Two 16bit floating point; packed into 32bits
        BFloat16, //< 16bit floating point (brain-float format)
        BFloat16x2, //< Two bfloat values; packed into 32bits
        FP8, //< 8bit floating point (E4M3)
        FP8x4, //< Four 8bit floating point (E4M3); packed into 32bits
        BF8, //< 8bit floating point (E5M2)
        BF8x4, //< Four 8bit floating point (E5M2); packed into 32bits
        FP4, //< 4bit floating point (E2M1)
        FP4x8, //< Eight 4bit floating point (E2M1); packed into 32bits
        FP6, //< 6bit floating point in E2M3 format
        FP6x16, //< 16 6bit floating point in FP6 format; packed into 96bits
        BF6, //< 6bit floating point in E3M2 format
        BF6x16, //< 16 6bit floating point in BF6 format; packed into 96bits
        Int8x4, //< Four 8bit signed integers; packed into 32bits
        Int8, //< 8bit signed integer
        Int16, //< 16bit signed integer
        Int32, //< 32bit signed integer
        Int64, //< 64bit signed integer
        Raw32, //< Thirty-two bits
        UInt8x4, //< Four 8bit unsigned integers; packed into 32bits
        UInt8, //< 8bit unsigned integer
        UInt16, //< 16bit unsigned integer
        UInt32, //< 32bit unsigned integer
        UInt64, //< 64bit unsigned integer
        Bool, //< Single bit boolean (SCC)
        Bool32, //< Thirty-two booleans packed into 32bits.  Usually the result of a vector-comparison (VCC; On Wave32 VCC is a single Bool32).
        Bool64, //< Sixty-four booleans packed into 64bits.  Usually the result of a vector-comparison (VCC; On Wave64 VCC is a single Bool64).
        E8M0, //< 8bits scale type
        E8M0x4, //< Four 8bits scale type; packed into 32bits
        None, //< Represents: any, unknown/unspecified, or a deferred type.
        Count
    };

    std::string   toString(DataType d);
    std::string   TypeAbbrev(DataType d);
    std::ostream& operator<<(std::ostream& stream, DataType const& t);
    std::istream& operator>>(std::istream& stream, DataType& t);

    /**
     * Pointer Type
     */
    enum class PointerType : int
    {
        Value,
        PointerLocal,
        PointerGlobal,
        Buffer,

        Count,
        None = Count
    };

    /**
     * Memory location type; where a buffer is stored.
     */
    enum class MemoryType : int
    {
        Global = 0,
        LDS,
        AGPR,
        VGPR,
        WAVE,
        WAVE_LDS,
        WAVE_SPLIT,
        WAVE_Direct2LDS,
        WAVE_SWIZZLE,
        Literal,
        None,
        Count
    };

    std::string   toString(MemoryType m);
    std::ostream& operator<<(std::ostream& stream, MemoryType const& m);

    /**
     * Layout of wavetile for MFMA instructions.
     */
    enum class LayoutType : int
    {
        SCRATCH,
        MATRIX_A,
        MATRIX_B,
        MATRIX_ACCUMULATOR,
        None,
        Count
    };

    std::string   toString(LayoutType l);
    std::ostream& operator<<(std::ostream& stream, LayoutType l);

    enum class NaryArgument : int
    {
        DEST = 0,
        LHS,
        RHS,
        LHS_SCALE,
        RHS_SCALE,

        Count,
        None = Count
    };

    std::string   toString(NaryArgument n);
    std::ostream& operator<<(std::ostream& stream, NaryArgument n);

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

    // Case insensitive and with special cases
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

    /**
     * VariableType
     */
    struct VariableType
    {
        constexpr VariableType()
            : dataType()
        {
        }
        constexpr VariableType(VariableType const& v)
            : dataType(v.dataType)
            , pointerType(v.pointerType)
        {
        }
        // cppcheck-suppress noExplicitConstructor
        constexpr VariableType(DataType d)
            : dataType(d)
            , pointerType(PointerType::Value)
        {
        }
        constexpr VariableType(DataType d, PointerType p)
            : dataType(d)
            , pointerType(p)
        {
        }
        explicit constexpr VariableType(PointerType p)
            : dataType()
            , pointerType(p)
        {
        }

        DataType    dataType;
        PointerType pointerType = PointerType::Value;

        /**
         * @brief Gets element size in bytes.
         */
        size_t getElementSize() const;

        inline bool isPointer() const
        {
            return pointerType != PointerType::Value;
        }
        inline bool isGlobalPointer() const
        {
            return pointerType == PointerType::PointerGlobal;
        }

        inline VariableType getDereferencedType() const
        {
            return VariableType(dataType);
        }

        inline VariableType getPointer() const
        {
            AssertFatal(pointerType == PointerType::Value, ShowValue(pointerType));
            return VariableType(dataType, PointerType::PointerGlobal);
        }

        inline DataType getArithmeticType() const
        {
            if(pointerType == PointerType::Value)
                return dataType;

            return getIntegerType(false, getElementSize());
        }

        /**
         * Returns the register alignment for storing `count` values
        */
        int registerAlignment(Register::Type regType, int count, GPUArchitecture const& gpuArch);

        auto                  operator<=>(VariableType const&) const = default;
        inline constexpr bool operator==(const VariableType& rhs) const
        {
            return (dataType == rhs.dataType) && (pointerType == rhs.pointerType);
        }
        inline bool operator<(const VariableType& rhs) const
        {
            return pointerType < rhs.pointerType
                   || (pointerType == rhs.pointerType && pointerType == PointerType::Value
                       && dataType < rhs.dataType);
        }

        /**
         * Returns a VariableType representing the result of an arithmetic operation
         * between lhs and rhs. Will throw an exception for an invalid combination of inputs.
         *
         * Accepts up to one pointer value and one integral type, or two values of compatible
         * types.
         *
         * Does no conversion between different categories (float, integral, bool).
         */
        static VariableType Promote(VariableType lhs, VariableType rhs);
    };

    std::string   toString(PointerType const& p);
    std::ostream& operator<<(std::ostream& stream, PointerType const& p);

    std::string   toString(VariableType const& v);
    std::string   TypeAbbrev(VariableType const& v);
    std::ostream& operator<<(std::ostream& stream, VariableType const& v);

    /**
     * \ingroup DataTypes
     * \brief Runtime accessible data type metadata
     */
    struct DataTypeInfo
    {
        static DataTypeInfo const& Get(int index);
        static DataTypeInfo const& Get(DataType t);
        static DataTypeInfo const& Get(VariableType const& v);
        static DataTypeInfo const& Get(std::string const& str);

        /**
         * @brief Returns the not-segmented variable type.
         *
         * For example:
         * 1. If variableType == Half, this returns Halfx2.
         * 2. If variableType == Float, this returns {}.
         */
        std::optional<VariableType> packedVariableType() const;

        VariableType variableType;
        VariableType segmentVariableType;
        std::string  name;
        std::string  abbrev;

        unsigned int elementBytes;
        unsigned int elementBits;
        unsigned int registerCount;
        size_t       packing;
        size_t       alignment;

        bool isComplex;
        bool isIntegral;
        bool isSigned;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <typename T>
        static void registerTypeInfo();

        static void addInfoObject(DataTypeInfo const& info);

        static std::map<VariableType, DataTypeInfo> data;
        static std::map<std::string, VariableType>  typeNames;
    };

    /**
     * \ingroup DataTypes
     * \brief Compile-time accessible data type metadata.
     */
    template <typename T>
    struct TypeInfo
    {
    };

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
    struct BaseTypeInfo
    {
        using Type = T;

        constexpr static VariableType Var                 = VariableType(T_DEnum, T_PEnum);
        constexpr static VariableType SegmentVariableType = VariableType(T_SegmentType, T_PEnum);

        /// Bytes of one element.  May contain multiple segments.
        constexpr static size_t ElementBytes = sizeof(T);
        constexpr static size_t ElementBits  = T_Bits;
        /// Segments per element.
        constexpr static size_t Packing = T_Packing;

        constexpr static size_t Alignment     = alignof(T);
        constexpr static size_t RegisterCount = T_RegCount;

        constexpr static bool IsComplex  = T_IsComplex;
        constexpr static bool IsIntegral = T_IsIntegral;
        constexpr static bool IsSigned   = T_IsSigned;

        static inline std::string Name()
        {
            if(Var.pointerType == PointerType::Value)
                return toString(Var.dataType);
            else
                return toString(Var.pointerType);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Var);
        }
    };

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

    struct Halfx2 : public DistinctType<uint32_t, Halfx2>
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

    struct FP8x4 : public DistinctType<uint32_t, FP8x4>
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

    struct BF8x4 : public DistinctType<uint32_t, BF8x4>
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

    struct FP6x16
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
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

    struct BF6x16
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
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

    struct FP4x8 : public DistinctType<uint32_t, FP4x8>
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

    struct BFloat16x2 : public DistinctType<uint32_t, BFloat16x2>
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

    struct Raw32 : public DistinctType<uint32_t, Raw32>
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

    struct Bool32 : public DistinctType<uint32_t, Bool32>
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

    struct Bool64 : public DistinctType<uint64_t, Bool64>
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

    struct PointerLocal : public DistinctType<uint32_t, PointerLocal>
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

    struct PointerGlobal : public DistinctType<uint64_t, PointerGlobal>
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

    struct Buffer
    {
        uint32_t desc0;
        uint32_t desc1;
        uint32_t desc2;
        uint32_t desc3;
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

    template <DataType T_DataType>
    struct EnumTypeInfo
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

#undef DeclareEnumTypeInfo

    template <typename T>
    concept CArithmeticType = std::integral<T> || std::floating_point<T>;

    template <typename Result, typename T>
    concept CCanStaticCastTo = requires(T val) //
    {
        {
            static_cast<Result>(val)
            } -> std::same_as<Result>;
    };

    template <typename T>
    concept CHasTypeInfo = requires() //
    {
        {
            TypeInfo<T>::Name()
            } -> std::convertible_to<std::string>;
    };

    template <CArithmeticType T>
    using similar_integral_type = typename EnumTypeInfo<getIntegerType(
        std::is_signed_v<T>&& std::is_integral_v<T>, sizeof(T))>::Type;

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    template <>
    struct hash<rocRoller::VariableType>
    {
        inline size_t operator()(rocRoller::VariableType const& varType) const
        {
            return rocRoller::hash_combine(static_cast<size_t>(varType.dataType),
                                           static_cast<size_t>(varType.pointerType));
        }
    };
}
