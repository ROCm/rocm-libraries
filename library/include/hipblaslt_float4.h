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

#ifndef _HIPBLASLT_FLOAT4_H_
#define _HIPBLASLT_FLOAT4_H_

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))
/*! \brief Struct to represent a 4 bit floating-point number. */

struct HIPBLASLT_EXPORT hipblaslt_f4
{
    uint8_t __x : 4;

    hipblaslt_f4() = default;

    explicit hipblaslt_f4(uint8_t x)
        : __x(x)
    {
    }

    operator _Float16() const
    {
        return _Float16(float(*this));
    }
    operator float() const
    {
        uint8_t                                val    = __x & 0x0F; // Remove first four bits
        static constexpr std::array<float, 16> values = {
            0.0, // 0000
            0.5, // 0001
            1.0, // 0010
            1.5, // 0011
            2.0, // 0100
            3.0, // 0101
            4.0, // 0110
            6.0, // 0111

            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        };

        return values[__x];
    };
};

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// TODO: HIP has implemented FP4 in the header below. However, currently there is no
//       direct use of FP4 type, and we just create a new struct that provides
//       conversion to float. In the future, we should inherit HIP's FP4 type to
//       provide full functionalties when needed.
// #include <hip/hip_ext_ocp.h>

struct HIPBLASLT_EXPORT hipblaslt_f4
{
    uint8_t __x : 4;

    hipblaslt_f4() = default;

    explicit hipblaslt_f4(uint8_t x)
        : __x(x)
    {
    }

    operator _Float16() const
    {
        return _Float16(float(*this));
    }
    operator float() const
    {
        uint8_t                                val    = __x & 0x0F; // Remove first four bits
        static constexpr std::array<float, 16> values = {
            0.0, // 0000
            0.5, // 0001
            1.0, // 0010
            1.5, // 0011
            2.0, // 0100
            3.0, // 0101
            4.0, // 0110
            6.0, // 0111

            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        };

        return values[__x];
    };
};

inline __host__ __device__ float operator+(const float fa, hipblaslt_f4 b)
{
    return (fa + float(b));
}

inline __host__ __device__ float operator+(hipblaslt_f4 a, const float fb)
{
    return (float(a) + fb);
}

inline __host__ __device__ float operator+(hipblaslt_f4 a, hipblaslt_f4 b)
{
    return (float(a) + float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f4 a, hipblaslt_f4 b)
{
    return float(a) * float(b);
}

inline __host__ __device__ float operator*(float a, hipblaslt_f4 b)
{
    return (a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f4 a, float b)
{
    return (float(a) * b);
}

inline __host__ __device__ float operator*(double a, hipblaslt_f4 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f4 a, double b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(int a, hipblaslt_f4 b)
{
    return ((float)a * float(b));
}

inline __host__ __device__ float operator*(hipblaslt_f4 a, int b)
{
    return ((float)a * float(b));
}

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // _HIPBLASLT_FLOAT4_H_
