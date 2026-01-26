#ifndef _HIPBLASLT_FLOAT4_H_
#define _HIPBLASLT_FLOAT4_H_

#define HIP_HOST_DEVICE __host__ __device__
#define HIP_HOST __host__
#define HIP_DEVICE __device__

// TODO: remove defined after solving the compilation failed
#if !defined(__gfx1250__)
#include <hip/hip_ext_ocp.h>
#endif

struct HIPBLASLT_EXPORT hipblaslt_f4x2
{
    enum class hip_f4_rounding_mode
    {
        standard,
        stochastic
    };

    uint8_t __x;

    hipblaslt_f4x2() = default;

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(uint8_t x)
        : __x(x)
    {
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(_Float16             v0,
                                            _Float16             v1,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
    {
#if !defined(__gfx1250__)
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
#else
        __x = 0x00;
#endif
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(float                v0,
                                            float                v1,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
    {
#if !defined(__gfx1250__)
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
#else
        __x = 0x00;
#endif
    }

    explicit HIP_HOST_DEVICE hipblaslt_f4x2(double               v0,
                                            double               v1,
                                            hip_f4_rounding_mode rm
                                            = hip_f4_rounding_mode::standard,
                                            uint32_t rng = 0)
        : hipblaslt_f4x2(static_cast<float>(v0), static_cast<float>(v1), rm, rng)
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

    // assignment overloading only from the same types
    inline __host__ __device__ hipblaslt_f4x2& operator=(const hipblaslt_f4x2& a)
    {
        __x = a.__x;
        return *this;
    }
};

#endif