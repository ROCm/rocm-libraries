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

#define TENSILE_USE_FP6

#ifdef TENSILE_USE_FP6

#ifdef _WIN32

#include <cstdint>

namespace TensileLite
{
    typedef struct Float6{ uint8_t data;} Float6;
} // end of namespace TensileLite

#else // _WIN32

#ifdef TENSILE_USE_HIP
#include <hip/hip_runtime.h>
#endif

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

#include <hip/hip_ext_ocp.h>

#include <cstdint>

namespace TensileLite
{
    enum class hip_f6_rounding_mode
    {
        standard,
        stochastic
    };

    struct Float6x32_Storage {
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
        uint8_t v16 : 6;
        uint8_t v17 : 6;
        uint8_t v18 : 6;
        uint8_t v19 : 6;
        uint8_t v20 : 6;
        uint8_t v21 : 6;
        uint8_t v22 : 6;
        uint8_t v23 : 6;
        uint8_t v24 : 6;
        uint8_t v25 : 6;
        uint8_t v26 : 6;
        uint8_t v27 : 6;
        uint8_t v28 : 6;
        uint8_t v29 : 6;
        uint8_t v30 : 6;
        uint8_t v31 : 6;
    } __attribute__((packed));

    // data type
    struct Float6x32
    {
        Float6x32_Storage data;

        // default constructor
        HIP_HOST_DEVICE Float6x32() = default;

        // constructor from float
        explicit HIP_HOST_DEVICE Float6x32(float                val,
                                           hip_f6_rounding_mode rm  = hip_f6_rounding_mode::standard,
                                           uint32_t             rng = 0)
        {
            union {
                Float6x32_Storage real;
                __amd_fp6x32_storage_t tmp;    
            } cvt;
            __amd_floatx32_storage_t f32x32;

            for (int i=0; i<32; i++)
                f32x32[i] = val;

            if (rm == hip_f6_rounding_mode::standard) {
                cvt.tmp = __amd_cvt_floatx32_to_fp6x32_scale(f32x32, __AMD_OCP_E2M3, 0);
                data = cvt.real;
            }
            else {
                cvt.tmp = __amd_cvt_floatx32_to_fp6x32_sr_scale(f32x32, __AMD_OCP_E2M3, rng, 0);
                data = cvt.real;
            }
        }

        // constructor from float
        explicit HIP_HOST_DEVICE Float6x32(float                v0,
                                           float                v1,
                                           float                v2,
                                           float                v3,
                                           float                v4,
                                           float                v5,
                                           float                v6,
                                           float                v7,
                                           float                v8,
                                           float                v9,
                                           float                v10,
                                           float                v11,
                                           float                v12,
                                           float                v13,
                                           float                v14,
                                           float                v15,
                                           float                v16,
                                           float                v17,
                                           float                v18,
                                           float                v19,
                                           float                v20,
                                           float                v21,
                                           float                v22,
                                           float                v23,
                                           float                v24,
                                           float                v25,
                                           float                v26,
                                           float                v27,
                                           float                v28,
                                           float                v29,
                                           float                v30,
                                           float                v31,
                                           hip_f6_rounding_mode rm  = hip_f6_rounding_mode::standard,
                                           uint32_t             rng = 0)
        {
            union {
                Float6x32_Storage real;
                __amd_fp6x32_storage_t tmp;    
            } cvt;
            __amd_floatx32_storage_t f32x32;

            f32x32[0] = v0;
            f32x32[1] = v1;
            f32x32[2] = v2;
            f32x32[3] = v3;
            f32x32[4] = v4;
            f32x32[5] = v5;
            f32x32[6] = v6;
            f32x32[7] = v7;
            f32x32[8] = v8;
            f32x32[9] = v9;
            f32x32[10] = v10;
            f32x32[11] = v11;
            f32x32[12] = v12;
            f32x32[13] = v13;
            f32x32[14] = v14;
            f32x32[15] = v15;
            f32x32[16] = v16;
            f32x32[17] = v17;
            f32x32[18] = v18;
            f32x32[19] = v19;
            f32x32[20] = v20;
            f32x32[21] = v21;
            f32x32[22] = v22;
            f32x32[23] = v23;
            f32x32[24] = v24;
            f32x32[25] = v25;
            f32x32[26] = v26;
            f32x32[27] = v27;
            f32x32[28] = v28;
            f32x32[29] = v29;
            f32x32[30] = v30;
            f32x32[31] = v31;

            if (rm == hip_f6_rounding_mode::standard) {
                cvt.tmp = __amd_cvt_floatx32_to_fp6x32_scale(f32x32, __AMD_OCP_E2M3, 0);
                data = cvt.real;
            }
            else {
                cvt.tmp = __amd_cvt_floatx32_to_fp6x32_sr_scale(f32x32, __AMD_OCP_E2M3, rng, 0);
                data = cvt.real;
            }
        }

        inline HIP_HOST_DEVICE float getElement(size_t idx) const
        {
            union {
                Float6x32_Storage real;
                __amd_fp6x32_storage_t tmp;
            } cvt;

            cvt.real = data;

            __amd_floatx32_storage_t fp32x32 = __amd_cvt_fp6x32_to_floatx32_scale(cvt.tmp, __AMD_OCP_E2M3, 0);
            if (idx < 32)
                return fp32x32[idx];
            else
                return 0.0;
        }

        // check for zero
        inline HIP_HOST_DEVICE bool is_zero() const
        {
            return data.v0 == 0 &&
                   data.v1 == 0 &&
                   data.v2 == 0 &&
                   data.v3 == 0 &&
                   data.v4 == 0 &&
                   data.v5 == 0 &&
                   data.v6 == 0 &&
                   data.v7 == 0 &&
                   data.v8 == 0 &&
                   data.v9 == 0 &&
                   data.v10 == 0 &&
                   data.v11 == 0 &&
                   data.v12 == 0 &&
                   data.v13 == 0 &&
                   data.v14 == 0 &&
                   data.v15 == 0 &&
                   data.v16 == 0 &&
                   data.v17 == 0 &&
                   data.v18 == 0 &&
                   data.v19 == 0 &&
                   data.v20 == 0 &&
                   data.v21 == 0 &&
                   data.v22 == 0 &&
                   data.v23 == 0 &&
                   data.v24 == 0 &&
                   data.v25 == 0 &&
                   data.v26 == 0 &&
                   data.v27 == 0 &&
                   data.v28 == 0 &&
                   data.v29 == 0 &&
                   data.v30 == 0 &&
                   data.v31 == 0;
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
    inline std::string to_string(const TensileLite::Float6x32& a)
    {
        union {
            TensileLite::Float6x32_Storage real;
            __amd_fp6x32_storage_t tmp;
        } cvt;

        cvt.real = a.data;

        auto result = __amd_cvt_fp6x32_to_floatx32_scale(cvt.tmp, __AMD_OCP_E2M3, 0);

        std::string str = std::to_string(static_cast<float>(result[0]));

        for (int i=1; i<32; i++)
          str = str + " " + std::to_string(static_cast<float>(result[i]));

        return str;
    }

    inline ostream& operator<<(ostream& stream, const TensileLite::Float6x32 a)
    {
        return stream << to_string(a);
    }
} // namespace std

#endif // _WIN32

#endif // TENSILE_USE_FP6
