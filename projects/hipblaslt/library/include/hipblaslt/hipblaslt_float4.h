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

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

#include <hip/hip_ext_ocp.h>

struct HIPBLASLT_EXPORT hipblaslt_f4x2
{
    enum class hip_f4_rounding_mode
    {
        standard,
        stochastic
    };

    uint8_t                 __x;
    static constexpr size_t packed_size = 2;

    hipblaslt_f4x2() = default;

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(_Float16             v0,
                                            _Float16             v1,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
    {
        __amd_fp16x2_storage_t f16x2;
        f16x2[0] = v0;
        f16x2[1] = v1;
        if(rm == hip_f4_rounding_mode::standard)
        {
            __x = __amd_cvt_fp16x2_to_fp4x2_scale(f16x2, __AMD_OCP_E2M1, 0);
        }
        else
        {
            __x = __amd_cvt_fp16x2_to_fp4x2_sr_scale(f16x2, __AMD_OCP_E2M1, rng, 0);
        }
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(float                v0,
                                            float                v1,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
    {
        __amd_floatx2_storage_t f32x2;
        f32x2[0] = v0;
        f32x2[1] = v1;
        if(rm == hip_f4_rounding_mode::standard)
        {
            __x = __amd_cvt_floatx2_to_fp4x2_scale(f32x2, __AMD_OCP_E2M1, 0);
        }
        else
        {
            __x = __amd_cvt_floatx2_to_fp4x2_sr_scale(f32x2, __AMD_OCP_E2M1, rng, 0);
        }
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(double               v0,
                                            double               v1,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
        : hipblaslt_f4x2(static_cast<float>(v0), static_cast<float>(v1), rm, rng)
    {
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(_Float16             val,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
        : hipblaslt_f4x2(val, val, rm, rng)
    {
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(float                val,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
        : hipblaslt_f4x2(val, val, rm, rng)
    {
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(double               val,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
        : hipblaslt_f4x2(val, val, rm, rng)
    {
    }

    float castElement(size_t idx) const
    {
        __amd_floatx2_storage_t fp32x2 = __amd_cvt_fp4x2_to_floatx2_scale(__x, __AMD_OCP_E2M1, 0);
        if(idx < packed_size)
            return fp32x2[idx];
        else
            return 0.0;
    };

    // only conver the first fp4 element to float16
    operator _Float16() const
    {
        return _Float16(float(*this));
    }

    // only conver the first fp4 element to float
    operator float() const
    {
        return castElement(0);
    }

    // assignment overloading only from the same types
    inline __host__ __device__ hipblaslt_f4x2& operator=(const hipblaslt_f4x2& a)
    {
        __x = a.__x;
        return *this;
    }
};

#endif
