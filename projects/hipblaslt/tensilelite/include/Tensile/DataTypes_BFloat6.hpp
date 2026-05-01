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

// Workaround: ROCm's amd_hip_ocp_host.hpp has a static_assert size mismatch
// (fp6x32_packed vs __amd_fp6x32_storage_t) in its host-fallback path, which
// is taken for all non-gfx950/gfx1250 device targets.
#include <Tensile/Macros.hpp>

#if (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__) || defined(__gfx1250__)) && !defined(WIN32) && !defined(_WIN32)
#define TENSILE_USE_BF6
#endif

#ifdef TENSILE_USE_BF6

#ifdef TENSILE_USE_HIP
#include <hip/hip_runtime.h>
#endif

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

#include <hip/hip_ext_ocp.h>

#include <cstdint>

TENSILE_HIDDEN_BEGIN

namespace TensileLite
{
    enum class hip_bf6_rounding_mode
    {
        standard,
        stochastic
    };

    struct BFloat6x16_Storage
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

    // data type
    struct BFloat6x16
    {
        BFloat6x16_Storage      data;
        constexpr static size_t packed_size = 16;

        // default constructor
        HIP_HOST_DEVICE BFloat6x16() = default;

        // constructor from float
        explicit HIP_HOST_DEVICE BFloat6x16(float                 val,
                                            hip_bf6_rounding_mode rm
                                            = hip_bf6_rounding_mode::standard,
                                            uint32_t rng = 0)
        {
            __amd_floatx16_storage_t f32x16;

            for(int i = 0; i < packed_size; i++)
                f32x16[i] = val;

            if(rm == hip_bf6_rounding_mode::standard)
            {
#ifdef HIPBLASLT_USE_HIP_FP6X16
                union
                {
                    BFloat6x16_Storage     real;
                    __amd_fp6x16_storage_t tmp;
                } cvt;
                cvt.tmp = __amd_cvt_floatx16_to_fp6x16_scale(f32x16, __AMD_OCP_E3M2, 0);
                data    = cvt.real;
#else
                union
                {
                    __amd_floatx32_storage_t fp32x32;
                    __amd_floatx16_storage_t fp32x16[2];
                } in;
                union
                {
                    BFloat6x16_Storage     real[2];
                    __amd_fp6x32_storage_t fp6x32;
                } out;
                in.fp32x16[0] = f32x16;
                out.fp6x32
                    = __amd_cvt_floatx32_to_fp6x32_scale(in.fp32x32, __AMD_OCP_E3M2, 0);
                data = out.real[0];
#endif
            }
            else
            {
                // TODO: update below code if hip_ext_ocp.h supports __amd_cvt_floatx16_to_fp6x16_sr_scale
                union
                {
                    __amd_floatx32_storage_t fp32x32;
                    __amd_floatx16_storage_t fp32x16[2];
                } in;
                union
                {
                    BFloat6x16_Storage     real[2];
                    __amd_fp6x32_storage_t fp6x32;
                } out;
                in.fp32x16[0] = f32x16;
                out.fp6x32
                    = __amd_cvt_floatx32_to_fp6x32_sr_scale(in.fp32x32, __AMD_OCP_E3M2, rng, 0);
                data = out.real[0];
            }
        }

        // constructor from float
        explicit HIP_HOST_DEVICE BFloat6x16(float                 v0,
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
            __amd_floatx16_storage_t f32x16;

            f32x16[0]  = v0;
            f32x16[1]  = v1;
            f32x16[2]  = v2;
            f32x16[3]  = v3;
            f32x16[4]  = v4;
            f32x16[5]  = v5;
            f32x16[6]  = v6;
            f32x16[7]  = v7;
            f32x16[8]  = v8;
            f32x16[9]  = v9;
            f32x16[10] = v10;
            f32x16[11] = v11;
            f32x16[12] = v12;
            f32x16[13] = v13;
            f32x16[14] = v14;
            f32x16[15] = v15;

            if(rm == hip_bf6_rounding_mode::standard)
            {
#ifdef HIPBLASLT_USE_HIP_FP6X16
                union
                {
                    BFloat6x16_Storage     real;
                    __amd_fp6x16_storage_t tmp;
                } cvt;
                cvt.tmp = __amd_cvt_floatx16_to_fp6x16_scale(f32x16, __AMD_OCP_E3M2, 0);
                data    = cvt.real;
#else
                union
                {
                    __amd_floatx32_storage_t fp32x32;
                    __amd_floatx16_storage_t fp32x16[2];
                } in;
                union
                {
                    BFloat6x16_Storage     real[2];
                    __amd_fp6x32_storage_t fp6x32;
                } out;
                in.fp32x16[0] = f32x16;
                out.fp6x32
                    = __amd_cvt_floatx32_to_fp6x32_scale(in.fp32x32, __AMD_OCP_E3M2, 0);
                data = out.real[0];
#endif
            }
            else
            {
                // TODO: update below code if hip_ext_ocp.h supports __amd_cvt_floatx16_to_fp6x16_sr_scale
                union
                {
                    __amd_floatx32_storage_t fp32x32;
                    __amd_floatx16_storage_t fp32x16[2];
                } in;
                union
                {
                    BFloat6x16_Storage     real[2];
                    __amd_fp6x32_storage_t fp6x32;
                } out;
                in.fp32x16[0] = f32x16;
                out.fp6x32
                    = __amd_cvt_floatx32_to_fp6x32_sr_scale(in.fp32x32, __AMD_OCP_E3M2, rng, 0);
                data = out.real[0];
            }
        }

        inline HIP_HOST_DEVICE float getElement(size_t idx) const
        {
            // TODO: update below code if hip_ext_ocp.h supports __amd_cvt_fp6x16_to_floatx16_scale
            __amd_floatx32_storage_t fp32x32;
            union
            {
                BFloat6x16_Storage     real[2];
                __amd_fp6x32_storage_t fp6x32;
            } in;
            in.real[0] = data;
            fp32x32    = __amd_cvt_fp6x32_to_floatx32_scale(in.fp6x32, __AMD_OCP_E3M2, 0);
            if(idx < packed_size)
                return fp32x32[idx];
            else
                return 0.0;
        }

        // check for zero
        inline HIP_HOST_DEVICE bool is_zero() const
        {
            return data.v0 == 0 && data.v1 == 0 && data.v2 == 0 && data.v3 == 0 && data.v4 == 0
                   && data.v5 == 0 && data.v6 == 0 && data.v7 == 0 && data.v8 == 0 && data.v9 == 0
                   && data.v10 == 0 && data.v11 == 0 && data.v12 == 0 && data.v13 == 0
                   && data.v14 == 0 && data.v15 == 0;
        }

        // check for nan
        inline HIP_HOST_DEVICE bool is_nan() const
        {
            return false;
        }

        // check for inf
        inline HIP_HOST_DEVICE bool is_inf() const
        {
            return false;
        }
    };
} // end of namespace TensileLite

namespace std
{
    inline std::string to_string(const TensileLite::BFloat6x16& a)
    {
        // TODO: update below code if hip_ext_ocp.h supports __amd_cvt_fp6x16_to_floatx16_scale
        union
        {
            TensileLite::BFloat6x16_Storage real[2];
            __amd_fp6x32_storage_t          fp6x32;
        } in;
        in.real[0]  = a.data;
        auto result = __amd_cvt_fp6x32_to_floatx32_scale(in.fp6x32, __AMD_OCP_E3M2, 0);

        std::string str = std::to_string(static_cast<float>(result[0]));

        for(int i = 1; i < a.packed_size; i++)
            str = str + " " + std::to_string(static_cast<float>(result[i]));

        return str;
    }

    inline ostream& operator<<(ostream& stream, const TensileLite::BFloat6x16 a)
    {
        return stream << to_string(a);
    }
} // namespace std

TENSILE_HIDDEN_END

#endif // TENSILE_USE_BF6
