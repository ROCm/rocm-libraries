/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc.
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

#ifdef TENSILE_USE_HIP
#include <hip/hip_runtime.h>
#endif

// comment out following macro to disable FP8/BF8 types
#define TENSILE_USE_FP8_BF8

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

namespace TensileLite
{

    struct tensile_hip_fp8_e4m3: public __hip_fp8_e4m3
    {
        using __hip_fp8_e4m3:: __hip_fp8_e4m3; // list base's constructor in derive's scope

        // constructor -> down cast
#if HIP_FP8_TYPE_OCP
        HIP_HOST_DEVICE tensile_hip_fp8_e4m3(const _Float16 f)
#else
        HIP_HOST tensile_hip_fp8_e4m3(const _Float16 f)
#endif
        : __hip_fp8_e4m3(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_OCP
        HIP_HOST_DEVICE operator _Float16() const
#else
        HIP_HOST operator _Float16() const
#endif
        {
            return _Float16(float(*this));
        }

    };


    typedef tensile_hip_fp8_e4m3 Float8;

    struct tensile_hip_fp8_e5m2: public __hip_fp8_e5m2
    {
        using __hip_fp8_e5m2:: __hip_fp8_e5m2;
#if HIP_FP8_TYPE_OCP
        HIP_HOST_DEVICE tensile_hip_fp8_e5m2(const _Float16 f)
#else
        HIP_HOST tensile_hip_fp8_e5m2(const _Float16 f)
#endif
        : __hip_fp8_e5m2(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_OCP
        HIP_HOST_DEVICE operator _Float16() const
#else
        HIP_HOST operator _Float16() const
#endif
        {
            return _Float16(float(*this));
        }
    };


    typedef tensile_hip_fp8_e5m2 BFloat8;

    struct tensile_hip_fp8_e4m3_fnuz: public __hip_fp8_e4m3_fnuz
    {
        using __hip_fp8_e4m3_fnuz:: __hip_fp8_e4m3_fnuz;
#if HIP_FP8_TYPE_FNUZ
        HIP_HOST_DEVICE tensile_hip_fp8_e4m3_fnuz(const _Float16 f)
#else
        HIP_HOST tensile_hip_fp8_e4m3_fnuz(const _Float16 f)
#endif
        : __hip_fp8_e4m3_fnuz(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_FNUZ
        HIP_HOST_DEVICE operator _Float16() const
#else
        HIP_HOST operator _Float16() const
#endif
        {
            return _Float16(float(*this));
        }
    };
    typedef tensile_hip_fp8_e4m3_fnuz Float8_fnuz;


    struct tensile_hip_fp8_e5m2_fnuz: public __hip_fp8_e5m2_fnuz
    {
        using __hip_fp8_e5m2_fnuz:: __hip_fp8_e5m2_fnuz;
#if HIP_FP8_TYPE_FNUZ
        HIP_HOST_DEVICE tensile_hip_fp8_e5m2_fnuz(const _Float16 f)
#else
        HIP_HOST tensile_hip_fp8_e5m2_fnuz(const _Float16 f)
#endif
        : __hip_fp8_e5m2_fnuz(reinterpret_cast<const __half &>(f)) {}

        // operator overloadiing -> upcast
#if HIP_FP8_TYPE_FNUZ
        HIP_HOST_DEVICE operator _Float16() const
#else
        HIP_HOST operator _Float16() const
#endif
        {
            return _Float16(float(*this));
        }
    };
    typedef tensile_hip_fp8_e5m2_fnuz BFloat8_fnuz;


    //  Other operator overloading
    inline std::ostream& operator<<(std::ostream& os, const Float8& f8)
    {
        os << static_cast<float>(f8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const BFloat8& bf8)
    {
        os << static_cast<float>(bf8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const Float8_fnuz& f8)
    {
        os << static_cast<float>(f8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const BFloat8_fnuz& bf8)
    {
        os << static_cast<float>(bf8);
        return os;
    }

    //
    inline Float8 operator+(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8 operator+(Float8 a, float b)
    {
        return static_cast<Float8>(static_cast<float>(a) + b);
    }
    inline Float8 operator+(float a, Float8 b)
    {
        return static_cast<Float8>(a + static_cast<float>(b));
    }
    inline BFloat8 operator+(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8 operator-(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline BFloat8 operator-(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) - static_cast<float>(b));
    }
    //  NOTE: It is not used in reference solution directly, we want to return float otherwise
    inline Float8 operator*(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline Float8 operator*(float a, Float8 b)
    {
        return static_cast<Float8>(a * static_cast<float>(b));
    }
    inline Float8 operator*(Float8 a, float b)
    {
        return static_cast<Float8>(static_cast<float>(a) * b);
    }
    inline BFloat8 operator*(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) * static_cast<float>(b));
    }

    inline Float8 operator/(Float8 a, Float8 b)
    {
        return static_cast<Float8>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline BFloat8 operator/(BFloat8 a, BFloat8 b)
    {
        return static_cast<BFloat8>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(Float8 a, Float8 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<(float a, Float8 b)
    {
        return a < static_cast<float>(b);
    }
    inline bool operator<(Float8 a, float b)
    {
        return static_cast<float>(a) < b;
    }
    inline bool operator<(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(Float8 a, Float8 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator<=(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(Float8 a, Float8 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator==(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(Float8 a, Float8 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator!=(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(Float8 a, Float8 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>(float a, Float8 b)
    {
        return a > static_cast<float>(b);
    }
    inline bool operator>(Float8 a, float b)
    {
        return static_cast<float>(a) > b;
    }
    inline bool operator>(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(Float8 a, Float8 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
    inline bool operator>=(BFloat8 a, BFloat8 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

//  FNUZ
    inline Float8_fnuz operator+(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8_fnuz operator+(Float8_fnuz a, float b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) + b);
    }
    inline Float8_fnuz operator+(float a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(a + static_cast<float>(b));
    }
    inline BFloat8_fnuz operator+(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline Float8_fnuz operator-(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline BFloat8_fnuz operator-(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) - static_cast<float>(b));
    }
    //  NOTE: It is not used in reference solution directly, we want to return float otherwise
    inline Float8_fnuz operator*(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline Float8_fnuz operator*(float a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(a * static_cast<float>(b));
    }
    inline Float8_fnuz operator*(Float8_fnuz a, float b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) * b);
    }
    inline BFloat8_fnuz operator*(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) * static_cast<float>(b));
    }

    inline Float8_fnuz operator/(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<Float8_fnuz>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline BFloat8_fnuz operator/(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<BFloat8_fnuz>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<(float a, Float8_fnuz b)
    {
        return a < static_cast<float>(b);
    }
    inline bool operator<(Float8_fnuz a, float b)
    {
        return static_cast<float>(a) < b;
    }
    inline bool operator<(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator<=(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator==(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator!=(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>(float a, Float8_fnuz b)
    {
        return a > static_cast<float>(b);
    }
    inline bool operator>(Float8_fnuz a, float b)
    {
        return static_cast<float>(a) > b;
    }
    inline bool operator>(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(Float8_fnuz a, Float8_fnuz b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
    inline bool operator>=(BFloat8_fnuz a, BFloat8_fnuz b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
} // end of namespace TensileLite

// dummy datatypes! TODO: removes these by redesigning the computeType
typedef struct Float8BFloat8{ uint8_t data;} Float8BFloat8;
typedef struct BFloat8Float8{ uint8_t data;} BFloat8Float8;
typedef struct Float8BFloat8_fnuz{ uint8_t data;} Float8BFloat8_fnuz;
typedef struct BFloat8Float8_fnuz{ uint8_t data;} BFloat8Float8_fnuz;


namespace std
{
    inline bool isinf(const TensileLite::Float8& a)
    {
        return false;
    }
    inline bool isinf(const TensileLite::BFloat8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7f) == 0x7c;
    }

    inline bool isnan(const TensileLite::Float8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7f) == 0x7f;
    }
    inline bool isnan(const TensileLite::BFloat8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7f) > 0x7c;
    }
    inline bool iszero(const TensileLite::Float8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7F) == 0x0;
    }
    inline bool iszero(const TensileLite::BFloat8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7F) == 0x0;
    }

    //TODO: better to & 0x7F
    inline TensileLite::Float8 abs(const TensileLite::Float8& a)
    {
        return TensileLite::Float8(std::abs(float(a)));
    }
    inline TensileLite::BFloat8 abs(const TensileLite::BFloat8& a)
    {
        return TensileLite::BFloat8(std::abs(float(a)));
    }

    inline TensileLite::Float8 sin(const TensileLite::Float8& a)
    {
        return TensileLite::Float8(std::sin(float(a)));
    }
    inline TensileLite::BFloat8 sin(const TensileLite::BFloat8& a)
    {
        return TensileLite::BFloat8(std::sin(float(a)));
    }

    inline TensileLite::Float8 cos(const TensileLite::Float8& a)
    {
        return TensileLite::Float8(std::cos(float(a)));
    }
    inline TensileLite::BFloat8 cos(const TensileLite::BFloat8& a)
    {
        return TensileLite::BFloat8(std::cos(float(a)));
    }

    inline std::string to_string(const TensileLite::Float8& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    inline std::string to_string(const TensileLite::BFloat8& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    // FNUZ
    inline bool isinf(const TensileLite::Float8_fnuz& a)
    {
        return false;
    }
    inline bool isinf(const TensileLite::BFloat8_fnuz& a)
    {
        return false;
    }

    inline bool isnan(const TensileLite::Float8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x80;
    }
    inline bool isnan(const TensileLite::BFloat8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x80;
    }
    inline bool iszero(const TensileLite::Float8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x0;  // NOTE: only +0 exists
    }
    inline bool iszero(const TensileLite::BFloat8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x0;  // NOTE: only +0 exists
    }

    inline TensileLite::Float8_fnuz abs(const TensileLite::Float8_fnuz& a)
    {
        return TensileLite::Float8_fnuz(std::abs(float(a)));
    }
    inline TensileLite::BFloat8_fnuz abs(const TensileLite::BFloat8_fnuz& a)
    {
        return TensileLite::BFloat8_fnuz(std::abs(float(a)));
    }

    inline TensileLite::Float8_fnuz sin(const TensileLite::Float8_fnuz& a)
    {
        return TensileLite::Float8_fnuz(std::sin(float(a)));
    }
    inline TensileLite::BFloat8_fnuz sin(const TensileLite::BFloat8_fnuz& a)
    {
        return TensileLite::BFloat8_fnuz(std::sin(float(a)));
    }

    inline TensileLite::Float8_fnuz cos(const TensileLite::Float8_fnuz& a)
    {
        return TensileLite::Float8_fnuz(std::cos(float(a)));
    }
    inline TensileLite::BFloat8_fnuz cos(const TensileLite::BFloat8_fnuz& a)
    {
        return TensileLite::BFloat8_fnuz(std::cos(float(a)));
    }

    inline std::string to_string(const TensileLite::Float8_fnuz& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    inline std::string to_string(const TensileLite::BFloat8_fnuz& a)
    {
        return std::to_string(static_cast<float>(a));
    }

} // namespace std
