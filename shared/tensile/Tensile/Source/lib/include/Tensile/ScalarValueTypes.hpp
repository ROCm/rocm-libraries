/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 * \defgroup ScalarValue Scalar value type Info
 *
 * @brief Definitions and metadata on scalar value enumerations.
 */

    /**
 * \ingroup ScalarValue
 * @{
 */

    /**
 * Scalar value
 */
    enum class ScalarValue : int
    {
        Any,
        One,
        NegativeOne,
        Count
    };

    std::string   ToString(ScalarValue d);
    std::string   TypeAbbrev(ScalarValue d);
    std::ostream& operator<<(std::ostream& stream, ScalarValue const& t);
    std::istream& operator>>(std::istream& stream, ScalarValue& t);

    /**
 * \ingroup ScalarValue
 * \brief Runtime accessible scalar value type metadata
 */
    struct ScalarValueTypeInfo
    {
        static ScalarValueTypeInfo const& Get(int index);
        static ScalarValueTypeInfo const& Get(ScalarValue t);
        static ScalarValueTypeInfo const& Get(std::string const& str);

        ScalarValue m_value;
        std::string name;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <ScalarValue T_Enum>
        static void registerTypeInfo();

        static void addInfoObject(ScalarValueTypeInfo const& info);

        static std::map<ScalarValue, ScalarValueTypeInfo> data;
        static std::map<std::string, ScalarValue>         typeNames;
    };

    /**
 * \ingroup ScalarValue
 * \brief Compile-time accessible scalar type metadata.
 */
    template <ScalarValue T_Enum>
    struct ScalarValueInfo;

    template <ScalarValue T_Enum>
    struct BaseScalarValueInfo
    {
        constexpr static ScalarValue Enum = T_Enum;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <ScalarValue T_Enum>
    constexpr ScalarValue BaseScalarValueInfo<T_Enum>::Enum;

    template <>
    struct ScalarValueInfo<ScalarValue::Any> : public BaseScalarValueInfo<ScalarValue::Any>
    {
    };
    template <>
    struct ScalarValueInfo<ScalarValue::One> : public BaseScalarValueInfo<ScalarValue::One>
    {
    };
    template <>
    struct ScalarValueInfo<ScalarValue::NegativeOne>
        : public BaseScalarValueInfo<ScalarValue::NegativeOne>
    {
    };

    /**
*  \ingroup ScalarValue
*  \brief Gets ScalarValue enum from value. Returns ScalarValue::Any if there is no match.
*/
    template <typename T>
    ScalarValue toScalarValueEnum(T value)
    {
        if(value == T(1))
            return ScalarValue::One;
        else if(value == T(-1))
            return ScalarValue::NegativeOne;
        else
            return ScalarValue::Any;
    }

    /**
 * @}
 */
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::ScalarValue>
    {
        inline size_t operator()(Tensile::ScalarValue const& val) const
        {
            return hash<int>()(static_cast<int>(val));
        }
    };
} // namespace std
