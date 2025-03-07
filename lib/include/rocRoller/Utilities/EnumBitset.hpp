/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2022-2025 Advanced Micro Devices, Inc.
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

#include <bitset>

#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{
    /**
     * Bitset which uses an enum (class) as an indexer. Has the interface of std::bitset, and can be
     * indexed by the enum.
     *
     * All new interfaces are constexpr so it can be used in concepts and to limit template instantiation.
     *
     * - The enum must declare a Count entry.
     * - All other entries must be >= 0 and < Count.
     * - Count must be <= 64.
     */
    template <CCountedEnum Enum>
    class EnumBitset : public std::bitset<static_cast<size_t>(Enum::Count)>
    {
    public:
        static constexpr size_t Size = static_cast<size_t>(Enum::Count);
        static_assert(Size <= 64);

        using Base = std::bitset<Size>;
        using Base::Base;
        using Base::operator[];

        constexpr EnumBitset(std::initializer_list<Enum> items);
        constexpr EnumBitset(Base const& val);

        /**
         * Returns an object with all bits set.
         */
        constexpr static EnumBitset All();

        constexpr bool           operator[](Enum val) const;
        typename Base::reference operator[](Enum val);

    private:
        static constexpr size_t initialValue(std::initializer_list<Enum> items);
    };

    template <CCountedEnum Enum>
    std::string toString(EnumBitset<Enum> const& bs);

    template <CCountedEnum Enum>
    std::ostream& operator<<(std::ostream& stream, EnumBitset<Enum> const& bs);
}

#include <rocRoller/Utilities/EnumBitset_impl.hpp>
