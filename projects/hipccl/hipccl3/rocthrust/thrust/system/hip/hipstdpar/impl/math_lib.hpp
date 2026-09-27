/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file thrust/system/hip/math_lib.hpp
 *  \brief Implementations for standard library math functions forwarded to HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#if __has_include(<__clang_hip_libdevice_declares.h>)
extern "C" {
inline __device__ __attribute__((used)) float __hipstdpar_acos_f32(float x)
{
    return __ocml_acos_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_acos_f64(double x)
{
    return __ocml_acos_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_acosh_f32(float x)
{
    return __ocml_acosh_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_acosh_f64(double x)
{
    return __ocml_acosh_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_asin_f32(float x)
{
    return __ocml_asin_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_asin_f64(double x)
{
    return __ocml_asin_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_asinh_f32(float x)
{
    return __ocml_asinh_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_asinh_f64(double x)
{
    return __ocml_asinh_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_atan_f32(float x)
{
    return __ocml_atan_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_atan_f64(double x)
{
    return __ocml_atan_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_atanh_f32(float x)
{
    return __ocml_atanh_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_atanh_f64(double x)
{
    return __ocml_atanh_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_atan2_f32(float x,
                                                                    float y)
{
    return __ocml_atan2_f32(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_atan2_f64(double x,
                                                                     double y)
{
    return __ocml_atan2_f64(x, y);
}

inline __device__ __attribute__((used)) float __hipstdpar_cbrt_f32(float x)
{
    return __ocml_cbrt_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_cbrt_f64(double x)
{
    return __ocml_cbrt_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_cos_f64(double x)
{
    return __ocml_cos_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_cosh_f32(float x)
{
    return __ocml_cosh_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_cosh_f64(double x)
{
    return __ocml_cosh_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_erf_f32(float x)
{
    return __ocml_erf_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_erf_f64(double x)
{
    return __ocml_erf_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_erfc_f32(float x)
{
    return __ocml_erfc_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_erfc_f64(double x)
{
    return __ocml_erfc_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_exp_f64(double x)
{
    return __ocml_exp_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_exp2_f64(double x)
{
    return __ocml_exp2_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_expm1_f32(float x)
{
    return __ocml_expm1_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_expm1_f64(double x)
{
    return __ocml_expm1_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_fdim_f32(float x,
                                                                   float y)
{
    return __ocml_fdim_f32(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_fdim_f64(double x,
                                                                    double y)
{
    return __ocml_fdim_f64(x, y);
}

inline __device__ __attribute__((used)) float __hipstdpar_hypot_f32(float x,
                                                                    float y)
{
    return __ocml_hypot_f32(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_hypot_f64(double x,
                                                                     double y)
{
    return __ocml_hypot_f64(x, y);
}

inline __device__ __attribute__((used)) int __hipstdpar_ilogb_f32(float x)
{
    return __ocml_ilogb_f32(x);
}

inline __device__ __attribute__((used)) int __hipstdpar_ilogb_f64(double x)
{
    return __ocml_ilogb_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_lgamma_f32(float x)
{
    return __ocml_lgamma_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_lgamma_f64(double x)
{
    return __ocml_lgamma_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_logb_f32(float x)
{
    return __ocml_logb_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_logb_f64(double x)
{
    return __ocml_logb_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_log_f64(double x)
{
    return __ocml_log_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_log1p_f32(float x)
{
    return __ocml_log1p_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_log1p_f64(double x)
{
    return __ocml_log1p_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_log10_f64(double x)
{
    return __ocml_log10_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_log2_f64(double x)
{
    return __ocml_log2_f64(x);
}

// The modf interface must match the LLVM intrinsic i.e. it returns a struct
inline __device__ __attribute__((used)) decltype(auto) __hipstdpar_modf_f32(float x)
{
    struct { float f; float i; } r{};
    r.f = __ocml_modf_f32(x, (__attribute__((opencl_private)) float*)&r.i);
    return r;
}

inline __device__ __attribute__((used)) decltype(auto) __hipstdpar_modf_f64(double x)
{
    struct { double f; double i; } r{};
    r.f = __ocml_modf_f64(x, (__attribute__((opencl_private)) double*)&r.i);
    return r;
}

inline __device__ __attribute__((used)) float __hipstdpar_nextafter_f32(float x,
                                                                        float y)
{
    return __ocml_nextafter_f32(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_nextafter_f64(double x,
                                                                         double y)
{
    return __ocml_nextafter_f64(x, y);
}

// No long double support on AMDGPU so for nexttoward we forward to nextafter
inline __device__ __attribute__((used)) float __hipstdpar_nexttoward_f32(float x,
                                                                         float y)
{
    return __ocml_nextafter_f32(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_nexttoward_f64(double x,
                                                                          double y)
{
    return __ocml_nextafter_f64(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_pow_f64(double x,
                                                                   double y)
{
    return __ocml_pow_f64(x, y);
}

inline __device__ __attribute__((used)) float __hipstdpar_remainder_f32(float x,
                                                                        float y)
{
    return __ocml_remainder_f32(x, y);
}

inline __device__ __attribute__((used)) double __hipstdpar_remainder_f64(double x,
                                                                         double y)
{
    return __ocml_remainder_f64(x, y);
}

inline __device__ __attribute__((used)) float __hipstdpar_remquo_f32(float x,
                                                                     float y,
                                                                     int* p)
{
    int t{};
    float r = __ocml_remquo_f32(x, y, (__attribute__((opencl_private)) int*)&t);
    *p = t;
    return r;
}

inline __device__ __attribute__((used)) double __hipstdpar_remquo_f64(double x,
                                                                      double y,
                                                                      int* p)
{
    int t{};
    double r = __ocml_remquo_f64(x, y, (__attribute__((opencl_private)) int*)&t);
    *p = t;
    return r;
}

inline __device__ __attribute__((used)) float __hipstdpar_scalbln_f32(float x,
                                                                      long e)
{
    return __ocml_scalb_f32(x, e);
}

inline __device__ __attribute__((used)) double __hipstdpar_scalbln_f64(double x,
                                                                       long e)
{
    return __ocml_scalb_f64(x, e);
}

inline __device__ __attribute__((used)) float __hipstdpar_scalbn_f32(float x,
                                                                     int e)
{
    return __ocml_scalbn_f32(x, e);
}

inline __device__ __attribute__((used)) double __hipstdpar_scalbn_f64(double x,
                                                                      int e)
{
    return __ocml_scalbn_f64(x, e);
}

inline __device__ __attribute__((used)) float __hipstdpar_tgamma_f32(float x)
{
    return __ocml_tgamma_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_tgamma_f64(double x)
{
    return __ocml_tgamma_f64(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_sin_f64(double x)
{
    return __ocml_sin_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_sinh_f32(float x)
{
    return __ocml_sinh_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_sinh_f64(double x)
{
    return __ocml_sinh_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_tan_f32(float x)
{
    return __ocml_tan_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_tan_f64(double x)
{
    return __ocml_tan_f64(x);
}

inline __device__ __attribute__((used)) float __hipstdpar_tanh_f32(float x)
{
    return __ocml_tanh_f32(x);
}

inline __device__ __attribute__((used)) double __hipstdpar_tanh_f64(double x)
{
    return __ocml_tanh_f64(x);
}
} // extern "C"
#else // __has_include(<__clang_hip_libdevice_declares.h>)
#    warning "ROCm Device Libs not available, please supply your own implementation of standard library math functions"
#endif
#else // __HIPSTDPAR__
#    error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
