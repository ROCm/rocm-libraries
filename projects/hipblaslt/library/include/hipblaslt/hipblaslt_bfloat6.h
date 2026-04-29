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

#ifndef _HIPBLASLT_BFLOAT6_H_
#define _HIPBLASLT_BFLOAT6_H_

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

#include <hip/hip_ext_ocp.h>

struct hipblaslt_bf6x16_storage
{
    uint8_t v0 : 6;
    uint8_t v1 : 6;
    uint8_t v2 : 6;
    uint8_t v3 : 6;
    uint8_t v4 : 6;
    uint8_t v5 : 6;
    uint8_t v6 : 6;
    uint8_t v7 : 6;
    uint8_t v8 : 6;
    uint8_t v9 : 6;
    uint8_t v10 : 6;
    uint8_t v11 : 6;
    uint8_t v12 : 6;
    uint8_t v13 : 6;
    uint8_t v14 : 6;
    uint8_t v15 : 6;
} __attribute__((packed));

struct HIPBLASLT_EXPORT hipblaslt_bf6x16
{
    enum class hip_bf6_rounding_mode
    {
        standard,
        stochastic
    };

    hipblaslt_bf6x16_storage data;
    constexpr static size_t  packed_size = 16;

    hipblaslt_bf6x16() = default;

    explicit HIP_HOST_DEVICE hipblaslt_bf6x16(_Float16              v0,
                                              _Float16              v1,
                                              _Float16              v2,
                                              _Float16              v3,
                                              _Float16              v4,
                                              _Float16              v5,
                                              _Float16              v6,
                                              _Float16              v7,
                                              _Float16              v8,
                                              _Float16              v9,
                                              _Float16              v10,
                                              _Float16              v11,
                                              _Float16              v12,
                                              _Float16              v13,
                                              _Float16              v14,
                                              _Float16              v15,
                                              hip_bf6_rounding_mode rm
                                              = hip_bf6_rounding_mode::standard,
                                              uint32_t rng = 0)
    {
        __amd_fp16x32_storage_t fp16x32{};
        fp16x32[0]  = v0;
        fp16x32[1]  = v1;
        fp16x32[2]  = v2;
        fp16x32[3]  = v3;
        fp16x32[4]  = v4;
        fp16x32[5]  = v5;
        fp16x32[6]  = v6;
        fp16x32[7]  = v7;
        fp16x32[8]  = v8;
        fp16x32[9]  = v9;
        fp16x32[10] = v10;
        fp16x32[11] = v11;
        fp16x32[12] = v12;
        fp16x32[13] = v13;
        fp16x32[14] = v14;
        fp16x32[15] = v15;

        union
        {
            hipblaslt_bf6x16_storage real[2];
            __amd_fp6x32_storage_t   fp6x32;
        } out = {};

        if(rm == hip_bf6_rounding_mode::standard)
        {
            out.fp6x32 = __amd_cvt_fp16x32_to_fp6x32_scale(fp16x32, __AMD_OCP_E3M2, 0);
            data       = out.real[0];
        }
        else
        {
            out.fp6x32 = __amd_cvt_fp16x32_to_fp6x32_sr_scale(fp16x32, __AMD_OCP_E3M2, rng, 0);
            data       = out.real[0];
        }
    }
    explicit HIP_HOST_DEVICE hipblaslt_bf6x16(float                 v0,
                                              float                 v1,
                                              float                 v2,
                                              float                 v3,
                                              float                 v4,
                                              float                 v5,
                                              float                 v6,
                                              float                 v7,
                                              float                 v8,
                                              float                 v9,
                                              float                 v10,
                                              float                 v11,
                                              float                 v12,
                                              float                 v13,
                                              float                 v14,
                                              float                 v15,
                                              hip_bf6_rounding_mode rm
                                              = hip_bf6_rounding_mode::standard,
                                              uint32_t rng = 0)
    {
        __amd_floatx16_storage_t fp32x16;

        fp32x16[0]  = v0;
        fp32x16[1]  = v1;
        fp32x16[2]  = v2;
        fp32x16[3]  = v3;
        fp32x16[4]  = v4;
        fp32x16[5]  = v5;
        fp32x16[6]  = v6;
        fp32x16[7]  = v7;
        fp32x16[8]  = v8;
        fp32x16[9]  = v9;
        fp32x16[10] = v10;
        fp32x16[11] = v11;
        fp32x16[12] = v12;
        fp32x16[13] = v13;
        fp32x16[14] = v14;
        fp32x16[15] = v15;

        union
        {
            __amd_floatx32_storage_t fp32x32;
            __amd_floatx16_storage_t fp32x16[2];
        } in = {};
        union
        {
            hipblaslt_bf6x16_storage real[2];
            __amd_fp6x32_storage_t   fp6x32;
        } out = {};
        in.fp32x16[0] = fp32x16;

        if(rm == hip_bf6_rounding_mode::standard)
        {
            out.fp6x32 = __amd_cvt_floatx32_to_fp6x32_scale(in.fp32x32, __AMD_OCP_E3M2, 0);
            data       = out.real[0];
        }
        else
        {
            out.fp6x32 = __amd_cvt_floatx32_to_fp6x32_sr_scale(in.fp32x32, __AMD_OCP_E3M2, rng, 0);
            data       = out.real[0];
        }
    }

    explicit HIP_HOST_DEVICE hipblaslt_bf6x16(double                v0,
                                              double                v1,
                                              double                v2,
                                              double                v3,
                                              double                v4,
                                              double                v5,
                                              double                v6,
                                              double                v7,
                                              double                v8,
                                              double                v9,
                                              double                v10,
                                              double                v11,
                                              double                v12,
                                              double                v13,
                                              double                v14,
                                              double                v15,
                                              hip_bf6_rounding_mode rm
                                              = hip_bf6_rounding_mode::standard,
                                              uint32_t rng = 0)
    {
        __amd_floatx16_storage_t fp32x16;

        fp32x16[0]  = static_cast<float>(v0);
        fp32x16[1]  = static_cast<float>(v1);
        fp32x16[2]  = static_cast<float>(v2);
        fp32x16[3]  = static_cast<float>(v3);
        fp32x16[4]  = static_cast<float>(v4);
        fp32x16[5]  = static_cast<float>(v5);
        fp32x16[6]  = static_cast<float>(v6);
        fp32x16[7]  = static_cast<float>(v7);
        fp32x16[8]  = static_cast<float>(v8);
        fp32x16[9]  = static_cast<float>(v9);
        fp32x16[10] = static_cast<float>(v10);
        fp32x16[11] = static_cast<float>(v11);
        fp32x16[12] = static_cast<float>(v12);
        fp32x16[13] = static_cast<float>(v13);
        fp32x16[14] = static_cast<float>(v14);
        fp32x16[15] = static_cast<float>(v15);

        union
        {
            __amd_floatx32_storage_t fp32x32;
            __amd_floatx16_storage_t fp32x16[2];
        } in = {};
        union
        {
            hipblaslt_bf6x16_storage real[2];
            __amd_fp6x32_storage_t   fp6x32;
        } out = {};
        in.fp32x16[0] = fp32x16;

        if(rm == hip_bf6_rounding_mode::standard)
        {
            out.fp6x32 = __amd_cvt_floatx32_to_fp6x32_scale(in.fp32x32, __AMD_OCP_E3M2, 0);
            data       = out.real[0];
        }
        else
        {
            out.fp6x32 = __amd_cvt_floatx32_to_fp6x32_sr_scale(in.fp32x32, __AMD_OCP_E3M2, rng, 0);
            data       = out.real[0];
        }
    }

    explicit HIP_HOST_DEVICE hipblaslt_bf6x16(_Float16              val,
                                              hip_bf6_rounding_mode rm
                                              = hip_bf6_rounding_mode::standard,
                                              uint32_t rng = 0)
        : hipblaslt_bf6x16(
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, rm, rng)
    {
    }

    explicit HIP_HOST_DEVICE hipblaslt_bf6x16(float                 val,
                                              hip_bf6_rounding_mode rm
                                              = hip_bf6_rounding_mode::standard,
                                              uint32_t rng = 0)
        : hipblaslt_bf6x16(
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, rm, rng)
    {
    }

    explicit HIP_HOST_DEVICE hipblaslt_bf6x16(double                val,
                                              hip_bf6_rounding_mode rm
                                              = hip_bf6_rounding_mode::standard,
                                              uint32_t rng = 0)
        : hipblaslt_bf6x16(
            val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, rm, rng)
    {
    }

    float castElement(size_t idx) const
    {
        // TODO: update below code if hip_ext_ocp.h supports __amd_cvt_fp6x16_to_floatx16_scale
        union
        {
            hipblaslt_bf6x16_storage real[2];
            __amd_fp6x32_storage_t   fp6x32;
        } in = {};
        in.real[0] = data;
        __amd_floatx32_storage_t fp32x32
            = __amd_cvt_fp6x32_to_floatx32_scale(in.fp6x32, __AMD_OCP_E3M2, 0);

        if(idx < packed_size)
            return fp32x32[idx];
        else
            return 0.0;
    };

    // only conver the first fp6 element to float16
    operator _Float16() const
    {
        return _Float16(float(*this));
    }

    // only conver the first fp6 element to float
    operator float() const
    {
        return castElement(0);
    };

    // assignment overloading only from the same types
    inline __host__ __device__ hipblaslt_bf6x16& operator=(const hipblaslt_bf6x16& a)
    {
        data = a.data;
        return *this;
    }
};

#endif
