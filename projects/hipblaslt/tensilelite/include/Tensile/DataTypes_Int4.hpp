// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tensilelitehost/export.h>

#include <cstdint>
#include <iostream>
#include <string>

#define TENSILE_USE_INT4

namespace TensileLite
{
    /**
     * Two signed 4-bit integers in one byte, element 0 in the low nibble.
     *
     * This is the in-memory type of w4a16 weights. It is deliberately plain
     * storage with no arithmetic operators: int4 is never a compute type here.
     * The kernel dequantizes it to bf16 with a per-K-group scale before the
     * MAC, and the CPU reference does the same, so everything downstream works
     * in float.
     */
    struct Int4x2
    {
        uint8_t data;

        Int4x2() = default;

        explicit Int4x2(int v0, int v1)
            : data(static_cast<uint8_t>((v0 & 0xF) | ((v1 & 0xF) << 4)))
        {
        }

        /// Raw nibble `idx` (0 or 1), no sign interpretation.
        inline uint8_t getNibble(size_t idx) const
        {
            return (idx & 1) ? (data >> 4) : (data & 0xF);
        }

        /// Sign-extend element `idx` (0 or 1) to a full int.
        inline int getElement(size_t idx) const
        {
            // Sign-extend the 4-bit two's-complement value.
            return static_cast<int>(getNibble(idx) ^ 0x8) - 8;
        }

        inline void setElement(size_t idx, int v)
        {
            if(idx & 1)
                data = static_cast<uint8_t>((data & 0x0F) | ((v & 0xF) << 4));
            else
                data = static_cast<uint8_t>((data & 0xF0) | (v & 0xF));
        }

        inline bool is_zero() const
        {
            return data == 0;
        }
        inline bool is_nan() const
        {
            return false;
        }
        inline bool is_inf() const
        {
            return false;
        }
    };

    static_assert(sizeof(Int4x2) == 1, "Int4x2 must be exactly one byte");

} // namespace TensileLite

namespace std
{
    inline std::string to_string(const TensileLite::Int4x2& a)
    {
        return std::to_string(a.getElement(0)) + " " + std::to_string(a.getElement(1));
    }

    inline ostream& operator<<(ostream& stream, const TensileLite::Int4x2 a)
    {
        return stream << a.getElement(0) << " " << a.getElement(1);
    }
} // namespace std
