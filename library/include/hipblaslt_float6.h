/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#ifndef _HIPBLASLT_FLOAT6_H_
#define _HIPBLASLT_FLOAT6_H_

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))
/*! \brief Struct to represent a 6 bit floating-point number. */

struct HIPBLASLT_EXPORT hipblaslt_f6
{
    uint8_t __x : 6;

    explicit hipblaslt_f6(uint8_t x)
        : __x(x)
    {
    }

    hipblaslt_f6() = default;

    operator _Float16() const
    {
        return _Float16(float(*this));
    }

    bool is_zero() const
    {
        return ((__x & 0x3F) == 0x00);
    }

    operator float() const
    {
        uint8_t index = __x & 0x3F; // Remove first two bits

        static constexpr std::array<float, 64> values = {
            /*  E2M3  */
            0.000f, /* 0 00 000 */
            0.125f, /* 0 00 001 */
            0.25f, /* 0 00 010 */
            0.375f, /* 0 00 011 */
            0.5f, /* 0 00 100 */
            0.625f, /* 0 00 101 */
            0.75f, /* 0 00 110 */
            0.875f, /* 0 00 111 */
            1.f, /* 0 01 000 */
            1.125f, /* 0 01 001 */
            1.25f, /* 0 01 010 */
            1.375f, /* 0 01 011 */
            1.5f, /* 0 01 100 */
            1.625f, /* 0 01 101 */
            1.75f, /* 0 01 110 */
            1.875f, /* 0 01 111 */
            2.f, /* 0 10 000 */
            2.25f, /* 0 10 001 */
            2.5f, /* 0 10 010 */
            2.75f, /* 0 10 011 */
            3.f, /* 0 10 100 */
            3.25f, /* 0 10 101 */
            3.5f, /* 0 10 110 */
            3.75f, /* 0 10 111 */
            4.f, /* 0 11 000 */
            4.5f, /* 0 11 001 */
            5.f, /* 0 11 010 */
            5.5f, /* 0 11 011 */
            6.f, /* 0 11 100 */
            6.5f, /* 0 11 101 */
            7.f, /* 0 11 110 */
            7.5f, /* 0 11 111 */

            -0.000f, /* 1 00 000 */
            -0.125f, /* 1 00 001 */
            -0.25f, /* 1 00 010 */
            -0.375f, /* 1 00 011 */
            -0.5f, /* 1 00 100 */
            -0.625f, /* 1 00 101 */
            -0.75f, /* 1 00 110 */
            -0.875f, /* 1 00 111 */
            -1.f, /* 1 01 000 */
            -1.125f, /* 1 01 001 */
            -1.25f, /* 1 01 010 */
            -1.375f, /* 1 01 011 */
            -1.5f, /* 1 01 100 */
            -1.625f, /* 1 01 101 */
            -1.75f, /* 1 01 110 */
            -1.875f, /* 1 01 111 */
            -2.f, /* 1 10 000 */
            -2.25f, /* 1 10 001 */
            -2.5f, /* 1 10 010 */
            -2.75f, /* 1 10 011 */
            -3.f, /* 1 10 100 */
            -3.25f, /* 1 10 101 */
            -3.5f, /* 1 10 110 */
            -3.75f, /* 1 10 111 */
            -4.f, /* 1 11 000 */
            -4.5f, /* 1 11 001 */
            -5.f, /* 1 11 010 */
            -5.5f, /* 1 11 011 */
            -6.f, /* 1 11 100 */
            -6.5f, /* 1 11 101 */
            -7.f, /* 1 11 110 */
            -7.5f, /* 1 11 111 */
        };

        return values[index];
    };
};

struct HIPBLASLT_EXPORT hipblaslt_bf6
{
    uint8_t __x : 6;

    hipblaslt_bf6() = default;

    explicit hipblaslt_bf6(uint8_t x)
        : __x(x)
    {
    }

    operator _Float16() const
    {
        return _Float16(float(*this));
    }

    bool is_zero() const
    {
        return ((__x & 0x3F) == 0x00);
    }

    operator float() const
    {
        uint8_t index = __x & 0x3F; // Remove first two bits

        static constexpr std::array<float, 64> values = {
            /*  E3M2  */
            0.00, /* 0 00 000 */
            0.0625, /* 0 00 001 */
            0.125, /* 0 00 010 */
            0.1875, /* 0 00 011 */
            0.25, /* 0 00 100 */
            0.3125, /* 0 00 101 */
            0.375, /* 0 00 110 */
            0.4375, /* 0 00 111 */
            0.50, /* 0 01 000 */
            0.6250, /* 0 01 001 */
            0.750, /* 0 01 010 */
            0.8750, /* 0 01 011 */
            1.00, /* 0 01 100 */
            1.2500, /* 0 01 101 */
            1.500, /* 0 01 110 */
            1.7500, /* 0 01 111 */
            2.00, /* 0 10 000 */
            2.5000, /* 0 10 001 */
            3.000, /* 0 10 010 */
            3.5000, /* 0 10 011 */
            4.00, /* 0 10 100 */
            5.0000, /* 0 10 101 */
            6.000, /* 0 10 110 */
            7.0000, /* 0 10 111 */
            8.00, /* 0 11 000 */
            10.0000, /* 0 11 001 */
            12.000, /* 0 11 010 */
            14.0000, /* 0 11 011 */
            16.00, /* 0 11 100 */
            20.0000, /* 0 11 101 */
            24.000, /* 0 11 110 */
            28.0000, /* 0 11 111 */

            -0.00, /* 1 00 000 */
            -0.0625, /* 1 00 001 */
            -0.125, /* 1 00 010 */
            -0.1875, /* 1 00 011 */
            -0.25, /* 1 00 100 */
            -0.3125, /* 1 00 101 */
            -0.375, /* 1 00 110 */
            -0.4375, /* 1 00 111 */
            -0.50, /* 1 01 000 */
            -0.6250, /* 1 01 001 */
            -0.750, /* 1 01 010 */
            -0.8750, /* 1 01 011 */
            -1.00, /* 1 01 100 */
            -1.2500, /* 1 01 101 */
            -1.500, /* 1 01 110 */
            -1.7500, /* 1 01 111 */
            -2.00, /* 1 10 000 */
            -2.5000, /* 1 10 001 */
            -3.000, /* 1 10 010 */
            -3.5000, /* 1 10 011 */
            -4.00, /* 1 10 100 */
            -5.0000, /* 1 10 101 */
            -6.000, /* 1 10 110 */
            -7.0000, /* 1 10 111 */
            -8.00, /* 1 11 000 */
            -10.0000, /* 1 11 001 */
            -12.000, /* 1 11 010 */
            -14.0000, /* 1 11 011 */
            -16.00, /* 1 11 100 */
            -20.0000, /* 1 11 101 */
            -24.000, /* 1 11 110 */
            -28.0000, /* 1 11 111 */
        };

        return values[index];
    };
};

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// TODO: HIP has implemented F6 (FP6/BF6) in the header below. However, currently there
//       is no direct use of F6 type, and we just create a new struct that provides
//       conversion to float. In the future, we should inherit HIP's F6 type to
//       provide full functionalties when needed.
// #include <hip/hip_ext_ocp.h>

struct HIPBLASLT_EXPORT hipblaslt_f6
{
    uint8_t __x : 6;

    explicit hipblaslt_f6(uint8_t x)
        : __x(x)
    {
    }

    HIP_HOST_DEVICE hipblaslt_f6() = default;

    HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this));
    }

    HIP_HOST_DEVICE bool is_zero() const
    {
        return ((__x & 0x3F) == 0x00);
    }

    operator float() const
    {
        uint8_t index = __x & 0x3F; // Remove first two bits

        static constexpr std::array<float, 64> values = {
            /*  E2M3  */
            0.000f, /* 0 00 000 */
            0.125f, /* 0 00 001 */
            0.25f, /* 0 00 010 */
            0.375f, /* 0 00 011 */
            0.5f, /* 0 00 100 */
            0.625f, /* 0 00 101 */
            0.75f, /* 0 00 110 */
            0.875f, /* 0 00 111 */
            1.f, /* 0 01 000 */
            1.125f, /* 0 01 001 */
            1.25f, /* 0 01 010 */
            1.375f, /* 0 01 011 */
            1.5f, /* 0 01 100 */
            1.625f, /* 0 01 101 */
            1.75f, /* 0 01 110 */
            1.875f, /* 0 01 111 */
            2.f, /* 0 10 000 */
            2.25f, /* 0 10 001 */
            2.5f, /* 0 10 010 */
            2.75f, /* 0 10 011 */
            3.f, /* 0 10 100 */
            3.25f, /* 0 10 101 */
            3.5f, /* 0 10 110 */
            3.75f, /* 0 10 111 */
            4.f, /* 0 11 000 */
            4.5f, /* 0 11 001 */
            5.f, /* 0 11 010 */
            5.5f, /* 0 11 011 */
            6.f, /* 0 11 100 */
            6.5f, /* 0 11 101 */
            7.f, /* 0 11 110 */
            7.5f, /* 0 11 111 */

            -0.000f, /* 1 00 000 */
            -0.125f, /* 1 00 001 */
            -0.25f, /* 1 00 010 */
            -0.375f, /* 1 00 011 */
            -0.5f, /* 1 00 100 */
            -0.625f, /* 1 00 101 */
            -0.75f, /* 1 00 110 */
            -0.875f, /* 1 00 111 */
            -1.f, /* 1 01 000 */
            -1.125f, /* 1 01 001 */
            -1.25f, /* 1 01 010 */
            -1.375f, /* 1 01 011 */
            -1.5f, /* 1 01 100 */
            -1.625f, /* 1 01 101 */
            -1.75f, /* 1 01 110 */
            -1.875f, /* 1 01 111 */
            -2.f, /* 1 10 000 */
            -2.25f, /* 1 10 001 */
            -2.5f, /* 1 10 010 */
            -2.75f, /* 1 10 011 */
            -3.f, /* 1 10 100 */
            -3.25f, /* 1 10 101 */
            -3.5f, /* 1 10 110 */
            -3.75f, /* 1 10 111 */
            -4.f, /* 1 11 000 */
            -4.5f, /* 1 11 001 */
            -5.f, /* 1 11 010 */
            -5.5f, /* 1 11 011 */
            -6.f, /* 1 11 100 */
            -6.5f, /* 1 11 101 */
            -7.f, /* 1 11 110 */
            -7.5f, /* 1 11 111 */
        };

        return values[index];
    };
};

// FIXME: this struct needs to inherit the bf6 implemented by HIP
struct HIPBLASLT_EXPORT hipblaslt_bf6
{
    uint8_t __x : 6;

    explicit hipblaslt_bf6(uint8_t x)
        : __x(x)
    {
    }

    HIP_HOST_DEVICE hipblaslt_bf6() = default;

    HIP_HOST_DEVICE operator _Float16() const
    {
        return _Float16(float(*this));
    }

    HIP_HOST_DEVICE bool is_zero() const
    {
        return ((__x & 0x3F) == 0x00);
    }

    operator float() const
    {
        uint8_t index = __x & 0x3F; // Remove first two bits

        static constexpr std::array<float, 64> values = {
            /*  E3M2  */
            0.00, /* 0 00 000 */
            0.0625, /* 0 00 001 */
            0.125, /* 0 00 010 */
            0.1875, /* 0 00 011 */
            0.25, /* 0 00 100 */
            0.3125, /* 0 00 101 */
            0.375, /* 0 00 110 */
            0.4375, /* 0 00 111 */
            0.50, /* 0 01 000 */
            0.6250, /* 0 01 001 */
            0.750, /* 0 01 010 */
            0.8750, /* 0 01 011 */
            1.00, /* 0 01 100 */
            1.2500, /* 0 01 101 */
            1.500, /* 0 01 110 */
            1.7500, /* 0 01 111 */
            2.00, /* 0 10 000 */
            2.5000, /* 0 10 001 */
            3.000, /* 0 10 010 */
            3.5000, /* 0 10 011 */
            4.00, /* 0 10 100 */
            5.0000, /* 0 10 101 */
            6.000, /* 0 10 110 */
            7.0000, /* 0 10 111 */
            8.00, /* 0 11 000 */
            10.0000, /* 0 11 001 */
            12.000, /* 0 11 010 */
            14.0000, /* 0 11 011 */
            16.00, /* 0 11 100 */
            20.0000, /* 0 11 101 */
            24.000, /* 0 11 110 */
            28.0000, /* 0 11 111 */

            -0.00, /* 1 00 000 */
            -0.0625, /* 1 00 001 */
            -0.125, /* 1 00 010 */
            -0.1875, /* 1 00 011 */
            -0.25, /* 1 00 100 */
            -0.3125, /* 1 00 101 */
            -0.375, /* 1 00 110 */
            -0.4375, /* 1 00 111 */
            -0.50, /* 1 01 000 */
            -0.6250, /* 1 01 001 */
            -0.750, /* 1 01 010 */
            -0.8750, /* 1 01 011 */
            -1.00, /* 1 01 100 */
            -1.2500, /* 1 01 101 */
            -1.500, /* 1 01 110 */
            -1.7500, /* 1 01 111 */
            -2.00, /* 1 10 000 */
            -2.5000, /* 1 10 001 */
            -3.000, /* 1 10 010 */
            -3.5000, /* 1 10 011 */
            -4.00, /* 1 10 100 */
            -5.0000, /* 1 10 101 */
            -6.000, /* 1 10 110 */
            -7.0000, /* 1 10 111 */
            -8.00, /* 1 11 000 */
            -10.0000, /* 1 11 001 */
            -12.000, /* 1 11 010 */
            -14.0000, /* 1 11 011 */
            -16.00, /* 1 11 100 */
            -20.0000, /* 1 11 101 */
            -24.000, /* 1 11 110 */
            -28.0000, /* 1 11 111 */
        };

        return values[index];
    };
};

inline __host__ __device__ float operator+(const float fa, hipblaslt_f6 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_f6 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_f6 a, hipblaslt_f6 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f6 a, hipblaslt_f6 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_f6 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f6 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(double a, hipblaslt_f6 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f6 a, double b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(int a, hipblaslt_f6 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f6 a, int b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator+(const float fa, hipblaslt_bf6 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_bf6 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_bf6 a, hipblaslt_bf6 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf6 a, hipblaslt_bf6 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_bf6 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf6 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(double a, hipblaslt_bf6 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf6 a, double b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(int a, hipblaslt_bf6 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_bf6 a, int b)
{
    return ((float)a * float(b));
}

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // _HIPBLASLT_FLOAT6_H_
