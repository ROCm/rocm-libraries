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
#include <rocRoller/DataTypes/DataTypes_Scale_Utils.hpp>
#include <rocRoller/DataTypes/DataTypes_UInt8x4.hpp>

#include <rocRoller/GPUArchitecture/GPUArchitecture_fwd.hpp>

#include <rocRoller/InstructionValues/Register_fwd.hpp>

#include <rocRoller/Utilities/Comparison.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/LazySingleton.hpp>

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

    inline constexpr DataType getIntegerType(bool isSigned, int sizeBytes);

    // Case insensitive and with special cases
    template <>
    inline DataType fromString<DataType>(std::string const& str);

    /**
     * For types that we have given a name within rocRoller, this will return that.
     * For other types it will defer to the RTTI name.
     */
    template <typename T>
    std::string friendlyTypeName();

    /**
     * VariableType
     */
    struct VariableType
    {
        constexpr VariableType();
        constexpr VariableType(VariableType const& v);

        // cppcheck-suppress noExplicitConstructor
        constexpr VariableType(DataType d);
        constexpr VariableType(DataType d, PointerType p);
        explicit constexpr VariableType(PointerType p);

        DataType    dataType;
        PointerType pointerType = PointerType::Value;

        /**
         * @brief Gets element size in bytes.
         */
        size_t getElementSize() const;

        bool isPointer() const;
        bool isGlobalPointer() const;

        VariableType getDereferencedType() const;

        inline VariableType getPointer() const;

        inline DataType getArithmeticType() const;

        /**
         * Returns the register alignment for storing `count` values
        */
        int registerAlignment(Register::Type regType, int count, GPUArchitecture const& gpuArch);

        auto operator<=>(VariableType const&) const = default;

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

    /**
     * Allows DataTypeInfo to give correct info while only having a single entry
     * shared between pointer types.
     */
    struct CompareVariableTypesPointersEqual
    {
        bool operator()(VariableType const& lhs, VariableType const& rhs) const;
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
        class Data : public LazySingleton<Data>
        {
        public:
            Data();

            void registerAllTypeInfo();

            template <typename T>
            void registerTypeInfo();

            void addInfoObject(DataTypeInfo const& info);

            using Compare = CompareVariableTypesPointersEqual;

            std::map<VariableType, DataTypeInfo, Compare> const& data() const;
            std::map<std::string, VariableType> const&           typeNames() const;

        private:
            std::map<VariableType, DataTypeInfo, Compare> m_data;
            std::map<std::string, VariableType>           m_typeNames;
        };
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

    template <DataType T_DataType>
    struct EnumTypeInfo
    {
    };

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

    struct Halfx2 : public DistinctType<uint32_t, Halfx2>
    {
    };

    struct FP8x4 : public DistinctType<uint32_t, FP8x4>
    {
    };

    struct BF8x4 : public DistinctType<uint32_t, BF8x4>
    {
    };

    struct FP6x16
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
    };

    struct BF6x16
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
    };

    struct FP4x8 : public DistinctType<uint32_t, FP4x8>
    {
    };

    struct BFloat16x2 : public DistinctType<uint32_t, BFloat16x2>
    {
    };

    struct Raw32 : public DistinctType<uint32_t, Raw32>
    {
    };

    struct Bool32 : public DistinctType<uint32_t, Bool32>
    {
    };

    struct Bool64 : public DistinctType<uint64_t, Bool64>
    {
    };

    struct PointerLocal : public DistinctType<uint32_t, PointerLocal>
    {
    };

    struct PointerGlobal : public DistinctType<uint64_t, PointerGlobal>
    {
    };

    struct Buffer
    {
        uint32_t desc0;
        uint32_t desc1;
        uint32_t desc2;
        uint32_t desc3;
    };

}

#include "DataTypes_impl.hpp"

namespace rocRoller
{

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
