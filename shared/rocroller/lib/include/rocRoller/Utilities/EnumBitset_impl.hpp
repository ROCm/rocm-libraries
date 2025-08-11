/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 AMD ROCm(TM) Software
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

#include <sstream>

#include <rocRoller/Utilities/EnumBitset.hpp>

namespace rocRoller
{
    template <CCountedEnum Enum>
    inline constexpr size_t EnumBitset<Enum>::initialValue(std::initializer_list<Enum> items)
    {
        size_t rv = 0;

        for(auto const& entry : items)
        {
            // cppcheck-suppress useStlAlgorithm
            rv |= (1UL << static_cast<size_t>(entry));
        }

        return rv;
    }

    template <CCountedEnum Enum>
    inline constexpr EnumBitset<Enum>::EnumBitset(std::initializer_list<Enum> items)
        : Base(initialValue(items))
    {
    }

    template <CCountedEnum Enum>
    inline constexpr EnumBitset<Enum>::EnumBitset(Base const& val)
        : Base(val)
    {
    }

    template <CCountedEnum Enum>
    inline constexpr EnumBitset<Enum> EnumBitset<Enum>::All()
    {
        size_t rv = 0;

        for(size_t i = 0; i < Size; i++)
            rv |= (1UL << static_cast<size_t>(i));

        return EnumBitset<Enum>(rv);
    }

    template <CCountedEnum Enum>
    inline constexpr bool EnumBitset<Enum>::operator[](Enum val) const
    {
        return (*this)[static_cast<size_t>(val)];
    }

    template <CCountedEnum Enum>
    void EnumBitset<Enum>::set(Enum target, bool value)
    {
        std::bitset<static_cast<size_t>(Enum::Count)>::set(static_cast<size_t>(target), value);
    }

    template <CCountedEnum Enum>
    inline std::string toString(EnumBitset<Enum> const& bs)
    {
        using myInt = std::underlying_type_t<Enum>;

        std::ostringstream msg;

        for(myInt i = 0; i < static_cast<myInt>(Enum::Count); ++i)
        {
            auto val = static_cast<Enum>(i);
            bool set = bs[val];
            msg << toString(val) << ": " << std::boolalpha << set << std::endl;
        }

        return msg.str();
    }

    template <CCountedEnum Enum>
    std::ostream& operator<<(std::ostream& stream, EnumBitset<Enum> const& bs)
    {
        return stream << toString(bs);
    }

}
