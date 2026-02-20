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

//namespace Tensile
//{

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

        // copy constructor
        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3& operator=(const float& a)
        {
            __x = tensile_hip_fp8_e4m3(a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3& operator=(const double& a)
        {
            __x = tensile_hip_fp8_e4m3((float)a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3& operator=(const tensile_hip_fp8_e4m3& a)
        {
            __x = a.__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3& operator=(const _Float16& a)
        {
            __x = tensile_hip_fp8_e4m3(a).__x;
            return *this;
        }

        //  += operator
        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3& operator+=(const tensile_hip_fp8_e4m3& a)
        {
            __x = tensile_hip_fp8_e4m3(float(this->__x) + float(a.__x)).__x;
            return *this;
        }
    };


    typedef tensile_hip_fp8_e4m3 tensile_float8;

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
        // copy constructor
        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2& operator=(const float& a)
        {
            __x = tensile_hip_fp8_e5m2(a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2& operator=(const double& a)
        {
            __x = tensile_hip_fp8_e5m2((float)a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2& operator=(const tensile_hip_fp8_e5m2& a)
        {
            __x = a.__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2& operator=(const _Float16& a)
        {
            __x = tensile_hip_fp8_e5m2(a).__x;
            return *this;
        }

        //  += operator
        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2& operator+=(const tensile_hip_fp8_e5m2& a)
        {
            __x = tensile_hip_fp8_e5m2(float(this->__x) + float(a.__x)).__x;
            return *this;
        }
    };


    typedef tensile_hip_fp8_e5m2 tensile_bfloat8;

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
        // copy constructor
        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3_fnuz& operator=(const float& a)
        {
            __x = tensile_hip_fp8_e4m3_fnuz(a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3_fnuz& operator=(const double& a)
        {
            __x = tensile_hip_fp8_e4m3_fnuz((float)a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3_fnuz& operator=(const tensile_hip_fp8_e4m3_fnuz& a)
        {
            __x = a.__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3_fnuz& operator=(const _Float16& a)
        {
            __x = tensile_hip_fp8_e4m3_fnuz(a).__x;
            return *this;
        }

        //  += operator
        inline HIP_HOST_DEVICE tensile_hip_fp8_e4m3_fnuz& operator+=(const tensile_hip_fp8_e4m3_fnuz& a)
        {
            __x = tensile_hip_fp8_e4m3_fnuz(float(this->__x) + float(a.__x)).__x;
            return *this;
        }
    };
    typedef tensile_hip_fp8_e4m3_fnuz tensile_float8_fnuz;


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
        // copy constructor
        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2_fnuz& operator=(const float& a)
        {
            __x = tensile_hip_fp8_e5m2_fnuz(a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2_fnuz& operator=(const double& a)
        {
            __x = tensile_hip_fp8_e5m2_fnuz((float)a).__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2_fnuz& operator=(const tensile_hip_fp8_e5m2_fnuz& a)
        {
            __x = a.__x;
            return *this;
        }

        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2_fnuz& operator=(const _Float16& a)
        {
            __x = tensile_hip_fp8_e5m2_fnuz(a).__x;
            return *this;
        }

        //  += operator
        inline HIP_HOST_DEVICE tensile_hip_fp8_e5m2_fnuz& operator+=(const tensile_hip_fp8_e5m2_fnuz& a)
        {
            __x = tensile_hip_fp8_e5m2_fnuz(float(this->__x) + float(a.__x)).__x;
            return *this;
        }
    };
    typedef tensile_hip_fp8_e5m2_fnuz tensile_bfloat8_fnuz;


    //  Other operator overloading
    inline std::ostream& operator<<(std::ostream& os, const tensile_float8& f8)
    {
        os << static_cast<float>(f8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const tensile_bfloat8& bf8)
    {
        os << static_cast<float>(bf8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const tensile_float8_fnuz& f8)
    {
        os << static_cast<float>(f8);
        return os;
    }
    inline std::ostream& operator<<(std::ostream& os, const tensile_bfloat8_fnuz& bf8)
    {
        os << static_cast<float>(bf8);
        return os;
    }

    //
    inline tensile_float8 operator+(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<tensile_float8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline tensile_float8 operator+(tensile_float8 a, float b)
    {
        return static_cast<tensile_float8>(static_cast<float>(a) + b);
    }
    inline tensile_float8 operator+(float a, tensile_float8 b)
    {
        return static_cast<tensile_float8>(a + static_cast<float>(b));
    }
    inline tensile_bfloat8 operator+(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<tensile_bfloat8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline tensile_float8 operator-(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<tensile_float8>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline tensile_bfloat8 operator-(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<tensile_bfloat8>(static_cast<float>(a) - static_cast<float>(b));
    }
    //  NOTE: It is not used in reference solution directly, we want to return float otherwise
    inline tensile_float8 operator*(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<tensile_float8>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline tensile_float8 operator*(float a, tensile_float8 b)
    {
        return static_cast<tensile_float8>(a * static_cast<float>(b));
    }
    inline tensile_float8 operator*(tensile_float8 a, float b)
    {
        return static_cast<tensile_float8>(static_cast<float>(a) * b);
    }
    inline tensile_bfloat8 operator*(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<tensile_bfloat8>(static_cast<float>(a) * static_cast<float>(b));
    }

    inline tensile_float8 operator/(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<tensile_float8>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline tensile_bfloat8 operator/(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<tensile_bfloat8>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<(float a, tensile_float8 b)
    {
        return a < static_cast<float>(b);
    }
    inline bool operator<(tensile_float8 a, float b)
    {
        return static_cast<float>(a) < b;
    }
    inline bool operator<(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator<=(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator==(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator!=(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>(float a, tensile_float8 b)
    {
        return a > static_cast<float>(b);
    }
    inline bool operator>(tensile_float8 a, float b)
    {
        return static_cast<float>(a) > b;
    }
    inline bool operator>(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(tensile_float8 a, tensile_float8 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
    inline bool operator>=(tensile_bfloat8 a, tensile_bfloat8 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

//  FNUZ
    inline tensile_float8_fnuz operator+(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<tensile_float8_fnuz>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline tensile_float8_fnuz operator+(tensile_float8_fnuz a, float b)
    {
        return static_cast<tensile_float8_fnuz>(static_cast<float>(a) + b);
    }
    inline tensile_float8_fnuz operator+(float a, tensile_float8_fnuz b)
    {
        return static_cast<tensile_float8_fnuz>(a + static_cast<float>(b));
    }
    inline tensile_bfloat8_fnuz operator+(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<tensile_bfloat8_fnuz>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline tensile_float8_fnuz operator-(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<tensile_float8_fnuz>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline tensile_bfloat8_fnuz operator-(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<tensile_bfloat8_fnuz>(static_cast<float>(a) - static_cast<float>(b));
    }
    //  NOTE: It is not used in reference solution directly, we want to return float otherwise
    inline tensile_float8_fnuz operator*(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<tensile_float8_fnuz>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline tensile_float8_fnuz operator*(float a, tensile_float8_fnuz b)
    {
        return static_cast<tensile_float8_fnuz>(a * static_cast<float>(b));
    }
    inline tensile_float8_fnuz operator*(tensile_float8_fnuz a, float b)
    {
        return static_cast<tensile_float8_fnuz>(static_cast<float>(a) * b);
    }
    inline tensile_bfloat8_fnuz operator*(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<tensile_bfloat8_fnuz>(static_cast<float>(a) * static_cast<float>(b));
    }

    inline tensile_float8_fnuz operator/(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<tensile_float8_fnuz>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline tensile_bfloat8_fnuz operator/(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<tensile_bfloat8_fnuz>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<(float a, tensile_float8_fnuz b)
    {
        return a < static_cast<float>(b);
    }
    inline bool operator<(tensile_float8_fnuz a, float b)
    {
        return static_cast<float>(a) < b;
    }
    inline bool operator<(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator<=(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator==(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator!=(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>(float a, tensile_float8_fnuz b)
    {
        return a > static_cast<float>(b);
    }
    inline bool operator>(tensile_float8_fnuz a, float b)
    {
        return static_cast<float>(a) > b;
    }
    inline bool operator>(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(tensile_float8_fnuz a, tensile_float8_fnuz b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
    inline bool operator>=(tensile_bfloat8_fnuz a, tensile_bfloat8_fnuz b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }
//} // end of namespace Tensile

namespace std
{
    inline bool isinf(const tensile_float8& a)
    {
        return false;
    }
    inline bool isinf(const tensile_bfloat8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7f) == 0x7c;
    }

    inline bool isnan(const tensile_float8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7f) == 0x7f;
    }
    inline bool isnan(const tensile_bfloat8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7f) > 0x7c;
    }
    inline bool iszero(const tensile_float8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7F) == 0x0;
    }
    inline bool iszero(const tensile_bfloat8& a)
    {
        return (static_cast<unsigned char>(a) & 0x7F) == 0x0;
    }

    //TODO: better to & 0x7F
    inline tensile_float8 abs(const tensile_float8& a)
    {
        return tensile_float8(std::abs(float(a)));
    }
    inline tensile_bfloat8 abs(const tensile_bfloat8& a)
    {
        return tensile_bfloat8(std::abs(float(a)));
    }

    inline tensile_float8 sin(const tensile_float8& a)
    {
        return tensile_float8(std::sin(float(a)));
    }
    inline tensile_bfloat8 sin(const tensile_bfloat8& a)
    {
        return tensile_bfloat8(std::sin(float(a)));
    }

    inline tensile_float8 cos(const tensile_float8& a)
    {
        return tensile_float8(std::cos(float(a)));
    }
    inline tensile_bfloat8 cos(const tensile_bfloat8& a)
    {
        return tensile_bfloat8(std::cos(float(a)));
    }

    inline std::string to_string(const tensile_float8& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    inline std::string to_string(const tensile_bfloat8& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    // FNUZ
    inline bool isinf(const tensile_float8_fnuz& a)
    {
        return false;
    }
    inline bool isinf(const tensile_bfloat8_fnuz& a)
    {
        return false;
    }

    inline bool isnan(const tensile_float8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x80;
    }
    inline bool isnan(const tensile_bfloat8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x80;
    }
    inline bool iszero(const tensile_float8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x0;  // NOTE: only +0 exists
    }
    inline bool iszero(const tensile_bfloat8_fnuz& a)
    {
        return static_cast<unsigned char>(a) == 0x0;  // NOTE: only +0 exists
    }

    inline tensile_float8_fnuz abs(const tensile_float8_fnuz& a)
    {
        return tensile_float8_fnuz(std::abs(float(a)));
    }
    inline tensile_bfloat8_fnuz abs(const tensile_bfloat8_fnuz& a)
    {
        return tensile_bfloat8_fnuz(std::abs(float(a)));
    }

    inline tensile_float8_fnuz sin(const tensile_float8_fnuz& a)
    {
        return tensile_float8_fnuz(std::sin(float(a)));
    }
    inline tensile_bfloat8_fnuz sin(const tensile_bfloat8_fnuz& a)
    {
        return tensile_bfloat8_fnuz(std::sin(float(a)));
    }

    inline tensile_float8_fnuz cos(const tensile_float8_fnuz& a)
    {
        return tensile_float8_fnuz(std::cos(float(a)));
    }
    inline tensile_bfloat8_fnuz cos(const tensile_bfloat8_fnuz& a)
    {
        return tensile_bfloat8_fnuz(std::cos(float(a)));
    }

    inline std::string to_string(const tensile_float8_fnuz& a)
    {
        return std::to_string(static_cast<float>(a));
    }
    inline std::string to_string(const tensile_bfloat8_fnuz& a)
    {
        return std::to_string(static_cast<float>(a));
    }

} // namespace std
