/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace Tensile
{
    /**
 * \ingroup Tensile
 * \defgroup ArithmeticUnits Arithmetic unit type Info
 *
 * @brief Definitions and metadata on supported arithmetic unit types.
 */

    /**
 * \ingroup ArithmeticUnits
 * @{
 */

    /**
 * Arithmetic Unit
 */
    enum class ArithmeticUnit : int
    {
        Any,
        MFMA,
        VALU,
        Count
    };

    std::string   ToString(ArithmeticUnit d);
    std::string   TypeAbbrev(ArithmeticUnit d);
    std::ostream& operator<<(std::ostream& stream, ArithmeticUnit const& t);
    std::istream& operator>>(std::istream& stream, ArithmeticUnit& t);

    /**
 * \ingroup ArithmeticUnits
 * \brief Runtime accessible arithmetic unit type metadata
 */
    struct ArithmeticUnitTypeInfo
    {
        static ArithmeticUnitTypeInfo const& Get(int index);
        static ArithmeticUnitTypeInfo const& Get(ArithmeticUnit t);
        static ArithmeticUnitTypeInfo const& Get(std::string const& str);

        ArithmeticUnit m_arithmeticUnit;
        std::string    name;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <ArithmeticUnit T_Enum>
        static void registerTypeInfo();

        static void addInfoObject(ArithmeticUnitTypeInfo const& info);

        static std::map<ArithmeticUnit, ArithmeticUnitTypeInfo> data;
        static std::map<std::string, ArithmeticUnit>            typeNames;
    };

    /**
 * \ingroup ArithmeticUnits
 * \brief Compile-time accessible arithmetic unit type metadata.
 */
    template <ArithmeticUnit T_Enum>
    struct ArithmeticUnitInfo;

    template <ArithmeticUnit T_Enum>
    struct BaseArithmeticUnitInfo
    {
        constexpr static ArithmeticUnit Enum = T_Enum;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <ArithmeticUnit T_Enum>
    constexpr ArithmeticUnit BaseArithmeticUnitInfo<T_Enum>::Enum;

    template <>
    struct ArithmeticUnitInfo<ArithmeticUnit::Any>
        : public BaseArithmeticUnitInfo<ArithmeticUnit::Any>
    {
    };
    template <>
    struct ArithmeticUnitInfo<ArithmeticUnit::MFMA>
        : public BaseArithmeticUnitInfo<ArithmeticUnit::MFMA>
    {
    };
    template <>
    struct ArithmeticUnitInfo<ArithmeticUnit::VALU>
        : public BaseArithmeticUnitInfo<ArithmeticUnit::VALU>
    {
    };

    /**
 * @}
 */
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::ArithmeticUnit>
    {
        inline size_t operator()(Tensile::ArithmeticUnit const& val) const
        {
            return hash<int>()(static_cast<int>(val));
        }
    };
} // namespace std
