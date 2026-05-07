/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc.
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

#include <Tensile/Macros.hpp>

#define TENSILE_USE_FP4

#ifdef TENSILE_USE_FP4

#ifdef _WIN32

namespace TensileLite
{
    typedef struct Float4{ uint8_t data;} Float4;
} // end of namespace TensileLite

#else // _WIN32

#ifdef TENSILE_USE_HIP
#include <hip/hip_runtime.h>
#endif

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

#include <hip/hip_ext_ocp.h>

TENSILE_HIDDEN_BEGIN

namespace TensileLite
{
    enum class hip_f4_rounding_mode
    {
        standard,
        stochastic
    };

    // data type
    struct Float4x2
    {
        __amd_fp4x2_storage_t data;

        // default constructor
        HIP_HOST_DEVICE Float4x2() = default;

        // constructor from float
        explicit HIP_HOST_DEVICE Float4x2(float                v0,
                                          float                v1,
                                          hip_f4_rounding_mode rm  = hip_f4_rounding_mode::standard,
                                          uint32_t             rng = 0)
        {
            __amd_floatx2_storage_t f32x2;
            f32x2[0] = v0;
            f32x2[1] = v1;
            if (rm == hip_f4_rounding_mode::standard) {
                data = __amd_cvt_floatx2_to_fp4x2_scale(f32x2, __AMD_OCP_E2M1, 0);
            }
            else {
                data = __amd_cvt_floatx2_to_fp4x2_sr_scale(f32x2, __AMD_OCP_E2M1, rng, 0);
            }
        }

        explicit HIP_HOST_DEVICE Float4x2(double               v0,
                                          double               v1,
                                          hip_f4_rounding_mode rm = hip_f4_rounding_mode::standard,
                                          uint32_t             rng = 0)
            : Float4x2(static_cast<float>(v0), static_cast<float>(v1), rm, rng)
        {
        }

        explicit HIP_HOST_DEVICE Float4x2(int                  v0,
                                          int                  v1,
                                          hip_f4_rounding_mode rm = hip_f4_rounding_mode::standard,
                                          uint32_t             rng = 0)
            : Float4x2((float)v0, (float)v1, rm, rng)
        {
        }
        explicit HIP_HOST_DEVICE Float4x2(size_t               v0,
                                          size_t               v1,
                                          hip_f4_rounding_mode rm = hip_f4_rounding_mode::standard,
                                          uint32_t             rng = 0)
            : Float4x2((float)v0, (float)v1, rm, rng)
        {
        }

        inline HIP_HOST_DEVICE float getElement(size_t idx) const
        {
            __amd_floatx2_storage_t fp32x2 = __amd_cvt_fp4x2_to_floatx2_scale(data, __AMD_OCP_E2M1, 0);
            switch(idx)
            {
            case 0:
                return fp32x2.x;
            case 1:
                return fp32x2.y;
            default:
                return 0.0;
            }
        }

        // check for zero
        inline HIP_HOST_DEVICE bool is_zero() const
        {
            return data == 0x00;
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
    inline std::string to_string(const TensileLite::Float4x2& a)
    {
        auto result = __amd_cvt_fp4x2_to_floatx2_scale(a.data, __AMD_OCP_E2M1, 0);
        return std::to_string(static_cast<float>(result[0])) + " " + std::to_string(static_cast<float>(result[1]));
    }

    inline ostream& operator<<(ostream& stream, const TensileLite::Float4x2 a)
    {
        auto result = __amd_cvt_fp4x2_to_floatx2_scale(a.data, __AMD_OCP_E2M1, 0);
        return stream << static_cast<float>(result[0]) << " " << static_cast<float>(result[1]);
    }
} // namespace std

TENSILE_HIDDEN_END

#endif // _WIN32

#endif // TENSILE_USE_FP4
