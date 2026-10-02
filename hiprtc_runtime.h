#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
      
extern "C" {
  __attribute__((__visibility__("default")))
  __attribute__((weak))
  __attribute__((noreturn))
  __attribute__((device)) void __cxa_pure_virtual(void) {
    __builtin_trap();
  }
  __attribute__((__visibility__("default")))
  __attribute__((weak))
  __attribute__((noreturn))
  __attribute__((device)) void __cxa_deleted_virtual(void) {
    __builtin_trap();
  }
}
typedef long unsigned int __hip_size_t;
extern "C" {
__attribute__((device)) unsigned long long __ockl_dm_alloc(unsigned long long __size);
__attribute__((device)) void __ockl_dm_dealloc(unsigned long long __addr);
__attribute__((weak)) inline __attribute__((device)) void *malloc(__hip_size_t __size) {
  return (void *) __ockl_dm_alloc(__size);
}
__attribute__((weak)) inline __attribute__((device)) void free(void *__ptr) {
  __ockl_dm_dealloc((unsigned long long)__ptr);
}
}
typedef long unsigned int size_t;
extern "C" {
__attribute__((device)) __attribute__((const)) float __ocml_acos_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_acosh_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_asin_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_asinh_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_atan2_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_atan_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_atanh_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_cbrt_f32(float);
__attribute__((device)) float __ocml_cos_f32(float);
__attribute__((device)) float __ocml_native_cos_f32(float);
__attribute__((device)) __attribute__((pure)) __attribute__((device)) float __ocml_cosh_f32(float);
__attribute__((device)) float __ocml_cospi_f32(float);
__attribute__((device)) float __ocml_i0_f32(float);
__attribute__((device)) float __ocml_i1_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfc_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfcinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfcx_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erf_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_erfinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_expm1_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fdim_f32(float, float);
__attribute__((device)) __attribute__((const)) __attribute__((device)) float __ocml_fmod_f32(float,
                                                                   float);
__attribute__((device)) __attribute__((const)) float __ocml_hypot_f32(float, float);
__attribute__((device)) __attribute__((const)) int __ocml_ilogb_f32(float);
__attribute__((device)) float __ocml_j0_f32(float);
__attribute__((device)) float __ocml_j1_f32(float);
__attribute__((device)) float __ocml_lgamma_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_log1p_f32(float);
__attribute__((device)) float __ocml_modf_f32(float, __attribute__((opencl_private)) float *);
__attribute__((device)) __attribute__((const)) float __ocml_nextafter_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_len3_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_len4_f32(float, float, float,
                                                        float);
__attribute__((device)) __attribute__((pure)) float __ocml_ncdf_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_ncdfinv_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_pow_f32(float, float);
__attribute__((device)) __attribute__((pure)) float __ocml_pown_f32(float, int);
__attribute__((device)) __attribute__((pure)) float __ocml_rcbrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_remainder_f32(float, float);
__attribute__((device)) float __ocml_remquo_f32(float, float, __attribute__((opencl_private)) int *);
__attribute__((device)) __attribute__((const)) float __ocml_rhypot_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_rlen3_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_rlen4_f32(float, float, float,
                                                         float);
__attribute__((device)) __attribute__((pure)) float __ocml_rsqrt_f32(float);
__attribute__((device)) float __ocml_sincos_f32(float, __attribute__((opencl_private)) float *);
__attribute__((device)) float __ocml_sincospi_f32(float, __attribute__((opencl_private)) float *);
__attribute__((device)) float __ocml_sin_f32(float);
__attribute__((device)) float __ocml_native_sin_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_sinh_f32(float);
__attribute__((device)) float __ocml_sinpi_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_native_sqrt_f32(float);
__attribute__((device)) float __ocml_tan_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_tanh_f32(float);
__attribute__((device)) float __ocml_tgamma_f32(float);
__attribute__((device)) float __ocml_y0_f32(float);
__attribute__((device)) float __ocml_y1_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_add_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sub_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_mul_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rte_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtn_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtp_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_div_rtz_f32(float, float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rte_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtn_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtp_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_sqrt_rtz_f32(float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rte_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtn_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtp_f32(float, float, float);
__attribute__((device)) __attribute__((const)) float __ocml_fma_rtz_f32(float, float, float);
__attribute__((device)) __attribute__((const)) double __ocml_acos_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_acosh_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_asin_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_asinh_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_atan2_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_atan_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_atanh_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_cbrt_f64(double);
__attribute__((device)) double __ocml_cos_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_cosh_f64(double);
__attribute__((device)) double __ocml_cospi_f64(double);
__attribute__((device)) double __ocml_i0_f64(double);
__attribute__((device)) double __ocml_i1_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfc_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfcinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfcx_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erf_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_erfinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp10_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp2_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_exp_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_expm1_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fdim_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_fmod_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_hypot_f64(double, double);
__attribute__((device)) __attribute__((const)) int __ocml_ilogb_f64(double);
__attribute__((device)) double __ocml_j0_f64(double);
__attribute__((device)) double __ocml_j1_f64(double);
__attribute__((device)) double __ocml_lgamma_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log10_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log1p_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log2_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_log_f64(double);
__attribute__((device)) double __ocml_modf_f64(double, __attribute__((opencl_private)) double *);
__attribute__((device)) __attribute__((const)) double __ocml_nextafter_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_len3_f64(double, double,
                                                         double);
__attribute__((device)) __attribute__((const)) double __ocml_len4_f64(double, double, double,
                                                         double);
__attribute__((device)) __attribute__((pure)) double __ocml_ncdf_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_ncdfinv_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_pow_f64(double, double);
__attribute__((device)) __attribute__((pure)) double __ocml_pown_f64(double, int);
__attribute__((device)) __attribute__((pure)) double __ocml_rcbrt_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_remainder_f64(double, double);
__attribute__((device)) double __ocml_remquo_f64(double, double, __attribute__((opencl_private)) int *);
__attribute__((device)) __attribute__((const)) double __ocml_rhypot_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_rlen3_f64(double, double,
                                                          double);
__attribute__((device)) __attribute__((const)) double __ocml_rlen4_f64(double, double,
                                                          double, double);
__attribute__((device)) __attribute__((pure)) double __ocml_rsqrt_f64(double);
__attribute__((device)) double __ocml_sincos_f64(double, __attribute__((opencl_private)) double *);
__attribute__((device)) double __ocml_sincospi_f64(double, __attribute__((opencl_private)) double *);
__attribute__((device)) double __ocml_sin_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_sinh_f64(double);
__attribute__((device)) double __ocml_sinpi_f64(double);
__attribute__((device)) double __ocml_tan_f64(double);
__attribute__((device)) __attribute__((pure)) double __ocml_tanh_f64(double);
__attribute__((device)) double __ocml_tgamma_f64(double);
__attribute__((device)) double __ocml_y0_f64(double);
__attribute__((device)) double __ocml_y1_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_add_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sub_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_mul_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rte_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtn_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtp_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_div_rtz_f64(double, double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rte_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtn_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtp_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_sqrt_rtz_f64(double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rte_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtn_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtp_f64(double, double,
                                                            double);
__attribute__((device)) __attribute__((const)) double __ocml_fma_rtz_f64(double, double,
                                                            double);
__attribute__((device)) _Float16 __ocml_cos_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
__attribute__((device)) _Float16 __ocml_sin_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);
typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
typedef short __2i16 __attribute__((ext_vector_type(2)));
typedef bool __ockl_bool;
__attribute__((device)) __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b,
                                                     float c, __ockl_bool s);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_fabs_2f16(__2f16);
__attribute__((device)) __2f16 __ocml_cos_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
__attribute__((device)) __attribute__((const))
__2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
__attribute__((device)) __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
__attribute__((device)) __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
__attribute__((device)) __2f16 __ocml_sin_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);
__attribute__((device)) __attribute__((const)) __2f16 __ocml_pown_2f16(__2f16, __2i16);
__attribute__((device)) void __asan_poison_memory_region(const void *addr,
                                            long unsigned int size);
__attribute__((device)) void __asan_unpoison_memory_region(const void *addr,
                                              long unsigned int size);
__attribute__((device)) int __asan_address_is_poisoned(const void *addr);
__attribute__((device)) void *__asan_region_is_poisoned(void *beg, long unsigned int size);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_fma_f16(_Float16, _Float16,
                                                          _Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
__attribute__((device)) __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
__attribute__((device)) __attribute__((pure)) float __ocml_exp2_f32(float);
__attribute__((device)) __attribute__((pure)) float __ocml_exp_f32(float);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
__attribute__((device)) __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
}
static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa_base8(const char *__tagp __attribute__((nonnull))) {
  long unsigned int __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;
    if (__tmp >= '0' && __tmp <= '7')
      __r = (__r * 8u) + __tmp - '0';
    else
      return 0;
    ++__tagp;
  }
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa_base10(const char *__tagp __attribute__((nonnull))) {
  long unsigned int __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;
    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 10u) + __tmp - '0';
    else
      return 0;
    ++__tagp;
  }
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa_base16(const char *__tagp __attribute__((nonnull))) {
  long unsigned int __r = 0;
  while (*__tagp != '\0') {
    char __tmp = *__tagp;
    if (__tmp >= '0' && __tmp <= '9')
      __r = (__r * 16u) + __tmp - '0';
    else if (__tmp >= 'a' && __tmp <= 'f')
      __r = (__r * 16u) + __tmp - 'a' + 10;
    else if (__tmp >= 'A' && __tmp <= 'F')
      __r = (__r * 16u) + __tmp - 'A' + 10;
    else
      return 0;
    ++__tagp;
  }
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
long unsigned int __make_mantissa(const char *__tagp __attribute__((nonnull))) {
  if (*__tagp == '0') {
    ++__tagp;
    if (*__tagp == 'x' || *__tagp == 'X')
      return __make_mantissa_base16(__tagp);
    else
      return __make_mantissa_base8(__tagp);
  }
  return __make_mantissa_base10(__tagp);
}
static __attribute__((device)) inline __attribute__((always_inline))
float __cosf(float __x) { return __ocml_native_cos_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __exp10f(float __x) {
  const float __log2_10 = 0x1.a934f0p+1f;
  return __builtin_amdgcn_exp2f(__log2_10 * __x);
}
static __attribute__((device)) inline __attribute__((always_inline))
float __expf(float __x) {
  const float __log2_e = 0x1.715476p+0;
  return __builtin_amdgcn_exp2f(__log2_e * __x);
}
static __attribute__((device)) inline __attribute__((always_inline))
float __fadd_rn(float __x, float __y) { return __x + __y; }
static __attribute__((device)) inline __attribute__((always_inline))
float __fdiv_rn(float __x, float __y) { return __x / __y; }
static __attribute__((device)) inline __attribute__((always_inline))
float __fdividef(float __x, float __y) { return __x / __y; }
static __attribute__((device)) inline __attribute__((always_inline))
float __fmaf_rn(float __x, float __y, float __z) {
  return __builtin_fmaf(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
float __fmul_rn(float __x, float __y) { return __x * __y; }
static __attribute__((device)) inline __attribute__((always_inline))
float __frcp_rn(float __x) { return 1.0f / __x; }
static __attribute__((device)) inline __attribute__((always_inline))
float __frsqrt_rn(float __x) { return __builtin_amdgcn_rsqf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __fsqrt_rn(float __x) { return __ocml_native_sqrt_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __fsub_rn(float __x, float __y) { return __x - __y; }
static __attribute__((device)) inline __attribute__((always_inline))
float __log10f(float __x) { return __builtin_log10f(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __log2f(float __x) { return __builtin_amdgcn_logf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __logf(float __x) { return __builtin_logf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __powf(float __x, float __y) { return __ocml_pow_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float __saturatef(float __x) { return (__x < 0) ? 0 : ((__x > 1) ? 1 : __x); }
static __attribute__((device)) inline __attribute__((always_inline))
void __sincosf(float __x, float *__sinptr, float *__cosptr) {
  *__sinptr = __ocml_native_sin_f32(__x);
  *__cosptr = __ocml_native_cos_f32(__x);
}
static __attribute__((device)) inline __attribute__((always_inline))
float __sinf(float __x) { return __ocml_native_sin_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float __tanf(float __x) {
  return __sinf(__x) * __builtin_amdgcn_rcpf(__cosf(__x));
}
static __attribute__((device)) inline __attribute__((always_inline))
int abs(int __x) {
  return __builtin_abs(__x);
}
static __attribute__((device)) inline __attribute__((always_inline))
long labs(long __x) {
  return __builtin_labs(__x);
}
static __attribute__((device)) inline __attribute__((always_inline))
long long llabs(long long __x) {
  return __builtin_llabs(__x);
}
static __attribute__((device)) inline __attribute__((always_inline))
float acosf(float __x) { return __ocml_acos_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float acoshf(float __x) { return __ocml_acosh_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float asinf(float __x) { return __ocml_asin_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float asinhf(float __x) { return __ocml_asinh_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float atan2f(float __x, float __y) { return __ocml_atan2_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float atanf(float __x) { return __ocml_atan_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float atanhf(float __x) { return __ocml_atanh_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float cbrtf(float __x) { return __ocml_cbrt_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float ceilf(float __x) { return __builtin_ceilf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float copysignf(float __x, float __y) { return __builtin_copysignf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float cosf(float __x) { return __ocml_cos_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float coshf(float __x) { return __ocml_cosh_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float cospif(float __x) { return __ocml_cospi_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float cyl_bessel_i0f(float __x) { return __ocml_i0_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float cyl_bessel_i1f(float __x) { return __ocml_i1_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float erfcf(float __x) { return __ocml_erfc_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float erfcinvf(float __x) { return __ocml_erfcinv_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float erfcxf(float __x) { return __ocml_erfcx_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float erff(float __x) { return __ocml_erf_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float erfinvf(float __x) { return __ocml_erfinv_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float exp10f(float __x) { return __builtin_exp10f(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float exp2f(float __x) { return __builtin_exp2f(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float expf(float __x) { return __builtin_expf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float expm1f(float __x) { return __ocml_expm1_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float fabsf(float __x) { return __builtin_fabsf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float fdimf(float __x, float __y) { return __ocml_fdim_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float fdividef(float __x, float __y) { return __x / __y; }
static __attribute__((device)) inline __attribute__((always_inline))
float floorf(float __x) { return __builtin_floorf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float fmaf(float __x, float __y, float __z) {
  return __builtin_fmaf(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
float fmaxf(float __x, float __y) { return __builtin_fmaxf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float fminf(float __x, float __y) { return __builtin_fminf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float fmodf(float __x, float __y) { return __ocml_fmod_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float frexpf(float __x, int *__nptr) {
  return __builtin_frexpf(__x, __nptr);
}
static __attribute__((device)) inline __attribute__((always_inline))
float hypotf(float __x, float __y) { return __ocml_hypot_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
int ilogbf(float __x) { return __ocml_ilogb_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __finitef(float __x) { return __builtin_isfinite(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __isinff(float __x) { return __builtin_isinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __isnanf(float __x) { return __builtin_isnan(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float j0f(float __x) { return __ocml_j0_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float j1f(float __x) { return __ocml_j1_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float jnf(int __n, float __x) {
  if (__n == 0)
    return j0f(__x);
  if (__n == 1)
    return j1f(__x);
  float __x0 = j0f(__x);
  float __x1 = j1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    float __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}
static __attribute__((device)) inline __attribute__((always_inline))
float ldexpf(float __x, int __e) { return __builtin_amdgcn_ldexpf(__x, __e); }
static __attribute__((device)) inline __attribute__((always_inline))
float lgammaf(float __x) { return __ocml_lgamma_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long long int llrintf(float __x) { return __builtin_rintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long long int llroundf(float __x) { return __builtin_roundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float log10f(float __x) { return __builtin_log10f(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float log1pf(float __x) { return __ocml_log1p_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float log2f(float __x) { return __builtin_log2f(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float logbf(float __x) { return __builtin_logbf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float logf(float __x) { return __builtin_logf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long int lrintf(float __x) { return __builtin_rintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long int lroundf(float __x) { return __builtin_roundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float modff(float __x, float *__iptr) {
  float __tmp;
  float __r = __ocml_modf_f32(__x, (__attribute__((opencl_private)) float *)&__tmp);
  *__iptr = __tmp;
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
float nanf(const char *__tagp __attribute__((nonnull))) {
  union {
    float val;
    struct ieee_float {
      unsigned int mantissa : 22;
      unsigned int quiet : 1;
      unsigned int exponent : 8;
      unsigned int sign : 1;
    } bits;
  } __tmp;
  static_assert((sizeof(__tmp.val)) == (sizeof(__tmp.bits)), "");
  __tmp.bits.sign = 0u;
  __tmp.bits.exponent = ~0u;
  __tmp.bits.quiet = 1u;
  __tmp.bits.mantissa = __make_mantissa(__tagp);
  return __tmp.val;
}
static __attribute__((device)) inline __attribute__((always_inline))
float nearbyintf(float __x) { return __builtin_nearbyintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float nextafterf(float __x, float __y) {
  return __ocml_nextafter_f32(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline))
float norm3df(float __x, float __y, float __z) {
  return __ocml_len3_f32(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
float norm4df(float __x, float __y, float __z, float __w) {
  return __ocml_len4_f32(__x, __y, __z, __w);
}
static __attribute__((device)) inline __attribute__((always_inline))
float normcdff(float __x) { return __ocml_ncdf_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float normcdfinvf(float __x) { return __ocml_ncdfinv_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float normf(int __dim,
            const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __builtin_sqrtf(__r);
}
static __attribute__((device)) inline __attribute__((always_inline))
float powf(float __x, float __y) { return __ocml_pow_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float powif(float __x, int __y) { return __ocml_pown_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float rcbrtf(float __x) { return __ocml_rcbrt_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float remainderf(float __x, float __y) {
  return __ocml_remainder_f32(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline))
float remquof(float __x, float __y, int *__quo) {
  int __tmp;
  float __r = __ocml_remquo_f32(__x, __y, (__attribute__((opencl_private)) int *)&__tmp);
  *__quo = __tmp;
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
float rhypotf(float __x, float __y) { return __ocml_rhypot_f32(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float rintf(float __x) { return __builtin_rintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float rnorm3df(float __x, float __y, float __z) {
  return __ocml_rlen3_f32(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
float rnorm4df(float __x, float __y, float __z, float __w) {
  return __ocml_rlen4_f32(__x, __y, __z, __w);
}
static __attribute__((device)) inline __attribute__((always_inline))
float rnormf(int __dim,
             const float *__a) {
  float __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __ocml_rsqrt_f32(__r);
}
static __attribute__((device)) inline __attribute__((always_inline))
float roundf(float __x) { return __builtin_roundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float rsqrtf(float __x) { return __ocml_rsqrt_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float scalblnf(float __x, long int __n) {
  if (__n > 9223372036854775807L)
    __n = 9223372036854775807L;
  else if (__n < (-2147483647 - 1))
    __n = (-2147483647 - 1);
  return __builtin_ldexpf(__x, (int)__n);
}
static __attribute__((device)) inline __attribute__((always_inline))
float scalbnf(float __x, int __n) { return __builtin_amdgcn_ldexpf(__x, __n); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __signbitf(float __x) { return __builtin_signbitf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
void sincosf(float __x, float *__sinptr, float *__cosptr) {
  float __tmp;
  *__sinptr = __ocml_sincos_f32(__x, (__attribute__((opencl_private)) float *)&__tmp);
  *__cosptr = __tmp;
}
static __attribute__((device)) inline __attribute__((always_inline))
void sincospif(float __x, float *__sinptr, float *__cosptr) {
  float __tmp;
  *__sinptr = __ocml_sincospi_f32(__x, (__attribute__((opencl_private)) float *)&__tmp);
  *__cosptr = __tmp;
}
static __attribute__((device)) inline __attribute__((always_inline))
float sinf(float __x) { return __ocml_sin_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float sinhf(float __x) { return __ocml_sinh_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float sinpif(float __x) { return __ocml_sinpi_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float sqrtf(float __x) { return __builtin_sqrtf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float tanf(float __x) { return __ocml_tan_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float tanhf(float __x) { return __ocml_tanh_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float tgammaf(float __x) { return __ocml_tgamma_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float truncf(float __x) { return __builtin_truncf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float y0f(float __x) { return __ocml_y0_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float y1f(float __x) { return __ocml_y1_f32(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
float ynf(int __n, float __x) {
  if (__n == 0)
    return y0f(__x);
  if (__n == 1)
    return y1f(__x);
  float __x0 = y0f(__x);
  float __x1 = y1f(__x);
  for (int __i = 1; __i < __n; ++__i) {
    float __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}
static __attribute__((device)) inline __attribute__((always_inline))
double acos(double __x) { return __ocml_acos_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double acosh(double __x) { return __ocml_acosh_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double asin(double __x) { return __ocml_asin_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double asinh(double __x) { return __ocml_asinh_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double atan(double __x) { return __ocml_atan_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double atan2(double __x, double __y) { return __ocml_atan2_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double atanh(double __x) { return __ocml_atanh_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double cbrt(double __x) { return __ocml_cbrt_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double ceil(double __x) { return __builtin_ceil(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double copysign(double __x, double __y) {
  return __builtin_copysign(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline))
double cos(double __x) { return __ocml_cos_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double cosh(double __x) { return __ocml_cosh_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double cospi(double __x) { return __ocml_cospi_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double cyl_bessel_i0(double __x) { return __ocml_i0_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double cyl_bessel_i1(double __x) { return __ocml_i1_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double erf(double __x) { return __ocml_erf_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double erfc(double __x) { return __ocml_erfc_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double erfcinv(double __x) { return __ocml_erfcinv_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double erfcx(double __x) { return __ocml_erfcx_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double erfinv(double __x) { return __ocml_erfinv_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double exp(double __x) { return __builtin_exp(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double exp10(double __x) { return __builtin_exp10(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double exp2(double __x) { return __builtin_exp2(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double expm1(double __x) { return __ocml_expm1_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double fabs(double __x) { return __builtin_fabs(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double fdim(double __x, double __y) { return __ocml_fdim_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double floor(double __x) { return __builtin_floor(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double fma(double __x, double __y, double __z) {
  return __builtin_fma(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
double fmax(double __x, double __y) { return __builtin_fmax(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double fmin(double __x, double __y) { return __builtin_fmin(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double fmod(double __x, double __y) { return __ocml_fmod_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double frexp(double __x, int *__nptr) {
  return __builtin_frexp(__x, __nptr);
}
static __attribute__((device)) inline __attribute__((always_inline))
double hypot(double __x, double __y) { return __ocml_hypot_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
int ilogb(double __x) { return __ocml_ilogb_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __finite(double __x) { return __builtin_isfinite(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __isinf(double __x) { return __builtin_isinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __isnan(double __x) { return __builtin_isnan(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double j0(double __x) { return __ocml_j0_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double j1(double __x) { return __ocml_j1_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double jn(int __n, double __x) {
  if (__n == 0)
    return j0(__x);
  if (__n == 1)
    return j1(__x);
  double __x0 = j0(__x);
  double __x1 = j1(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}
static __attribute__((device)) inline __attribute__((always_inline))
double ldexp(double __x, int __e) { return __builtin_amdgcn_ldexp(__x, __e); }
static __attribute__((device)) inline __attribute__((always_inline))
double lgamma(double __x) { return __ocml_lgamma_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long long int llrint(double __x) { return __builtin_rint(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long long int llround(double __x) { return __builtin_round(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double log(double __x) { return __ocml_log_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double log10(double __x) { return __ocml_log10_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double log1p(double __x) { return __ocml_log1p_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double log2(double __x) { return __ocml_log2_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double logb(double __x) { return __builtin_logb(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long int lrint(double __x) { return __builtin_rint(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
long int lround(double __x) { return __builtin_round(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double modf(double __x, double *__iptr) {
  double __tmp;
  double __r = __ocml_modf_f64(__x, (__attribute__((opencl_private)) double *)&__tmp);
  *__iptr = __tmp;
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
double nan(const char *__tagp) {
  union {
    double val;
    struct ieee_double {
      long unsigned int mantissa : 51;
      unsigned int quiet : 1;
      unsigned int exponent : 11;
      unsigned int sign : 1;
    } bits;
  } __tmp;
  static_assert((sizeof(__tmp.val)) == (sizeof(__tmp.bits)), "");
  __tmp.bits.sign = 0u;
  __tmp.bits.exponent = ~0u;
  __tmp.bits.quiet = 1u;
  __tmp.bits.mantissa = __make_mantissa(__tagp);
  return __tmp.val;
}
static __attribute__((device)) inline __attribute__((always_inline))
double nearbyint(double __x) { return __builtin_nearbyint(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double nextafter(double __x, double __y) {
  return __ocml_nextafter_f64(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline))
double norm(int __dim,
            const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __builtin_sqrt(__r);
}
static __attribute__((device)) inline __attribute__((always_inline))
double norm3d(double __x, double __y, double __z) {
  return __ocml_len3_f64(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
double norm4d(double __x, double __y, double __z, double __w) {
  return __ocml_len4_f64(__x, __y, __z, __w);
}
static __attribute__((device)) inline __attribute__((always_inline))
double normcdf(double __x) { return __ocml_ncdf_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double normcdfinv(double __x) { return __ocml_ncdfinv_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double pow(double __x, double __y) { return __ocml_pow_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double powi(double __x, int __y) { return __ocml_pown_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double rcbrt(double __x) { return __ocml_rcbrt_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double remainder(double __x, double __y) {
  return __ocml_remainder_f64(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline))
double remquo(double __x, double __y, int *__quo) {
  int __tmp;
  double __r = __ocml_remquo_f64(__x, __y, (__attribute__((opencl_private)) int *)&__tmp);
  *__quo = __tmp;
  return __r;
}
static __attribute__((device)) inline __attribute__((always_inline))
double rhypot(double __x, double __y) { return __ocml_rhypot_f64(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double rint(double __x) { return __builtin_rint(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double rnorm(int __dim,
             const double *__a) {
  double __r = 0;
  while (__dim--) {
    __r += __a[0] * __a[0];
    ++__a;
  }
  return __ocml_rsqrt_f64(__r);
}
static __attribute__((device)) inline __attribute__((always_inline))
double rnorm3d(double __x, double __y, double __z) {
  return __ocml_rlen3_f64(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline))
double rnorm4d(double __x, double __y, double __z, double __w) {
  return __ocml_rlen4_f64(__x, __y, __z, __w);
}
static __attribute__((device)) inline __attribute__((always_inline))
double round(double __x) { return __builtin_round(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double rsqrt(double __x) { return __ocml_rsqrt_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double scalbln(double __x, long int __n) {
  if (__n > 9223372036854775807L)
    __n = 9223372036854775807L;
  else if (__n < (-2147483647 - 1))
    __n = (-2147483647 - 1);
  return __builtin_ldexp(__x, (int)__n);
}
static __attribute__((device)) inline __attribute__((always_inline))
double scalbn(double __x, int __n) { return __builtin_amdgcn_ldexp(__x, __n); }
static __attribute__((device)) inline __attribute__((always_inline))
bool __signbit(double __x) { return __builtin_signbit(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double sin(double __x) { return __ocml_sin_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
void sincos(double __x, double *__sinptr, double *__cosptr) {
  double __tmp;
  *__sinptr = __ocml_sincos_f64(__x, (__attribute__((opencl_private)) double *)&__tmp);
  *__cosptr = __tmp;
}
static __attribute__((device)) inline __attribute__((always_inline))
void sincospi(double __x, double *__sinptr, double *__cosptr) {
  double __tmp;
  *__sinptr = __ocml_sincospi_f64(__x, (__attribute__((opencl_private)) double *)&__tmp);
  *__cosptr = __tmp;
}
static __attribute__((device)) inline __attribute__((always_inline))
double sinh(double __x) { return __ocml_sinh_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double sinpi(double __x) { return __ocml_sinpi_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double sqrt(double __x) { return __builtin_sqrt(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double tan(double __x) { return __ocml_tan_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double tanh(double __x) { return __ocml_tanh_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double tgamma(double __x) { return __ocml_tgamma_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double trunc(double __x) { return __builtin_trunc(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double y0(double __x) { return __ocml_y0_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double y1(double __x) { return __ocml_y1_f64(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double yn(int __n, double __x) {
  if (__n == 0)
    return y0(__x);
  if (__n == 1)
    return y1(__x);
  double __x0 = y0(__x);
  double __x1 = y1(__x);
  for (int __i = 1; __i < __n; ++__i) {
    double __x2 = (2 * __i) / __x * __x1 - __x0;
    __x0 = __x1;
    __x1 = __x2;
  }
  return __x1;
}
static __attribute__((device)) inline __attribute__((always_inline))
double __dadd_rn(double __x, double __y) { return __x + __y; }
static __attribute__((device)) inline __attribute__((always_inline))
double __ddiv_rn(double __x, double __y) { return __x / __y; }
static __attribute__((device)) inline __attribute__((always_inline))
double __dmul_rn(double __x, double __y) { return __x * __y; }
static __attribute__((device)) inline __attribute__((always_inline))
double __drcp_rn(double __x) { return 1.0 / __x; }
static __attribute__((device)) inline __attribute__((always_inline))
double __dsqrt_rn(double __x) { return __builtin_sqrt(__x); }
static __attribute__((device)) inline __attribute__((always_inline))
double __dsub_rn(double __x, double __y) { return __x - __y; }
static __attribute__((device)) inline __attribute__((always_inline))
double __fma_rn(double __x, double __y, double __z) {
  return __builtin_fma(__x, __y, __z);
}
template <class T> static __attribute__((device)) inline __attribute__((always_inline)) T min(T __arg1, T __arg2) {
  return (__arg1 < __arg2) ? __arg1 : __arg2;
}
template <class T> static __attribute__((device)) inline __attribute__((always_inline)) T max(T __arg1, T __arg2) {
  return (__arg1 > __arg2) ? __arg1 : __arg2;
}
static __attribute__((device)) inline __attribute__((always_inline)) int min(int __arg1, int __arg2) {
  return (__arg1 < __arg2) ? __arg1 : __arg2;
}
static __attribute__((device)) inline __attribute__((always_inline)) int max(int __arg1, int __arg2) {
  return (__arg1 > __arg2) ? __arg1 : __arg2;
}
static __attribute__((device)) inline __attribute__((always_inline))
float max(float __x, float __y) { return __builtin_fmaxf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double max(double __x, double __y) { return __builtin_fmax(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
float min(float __x, float __y) { return __builtin_fminf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline))
double min(double __x, double __y) { return __builtin_fmin(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) double abs(double __x) { return ::fabs(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float abs(float __x) { return ::fabsf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long long abs(long long __n) { return ::llabs(__n); }
static __attribute__((device)) inline __attribute__((always_inline)) long abs(long __n) { return ::labs(__n); }
static __attribute__((device)) inline __attribute__((always_inline)) float fma(float __x, float __y, float __z) {
  return ::fmaf(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline)) float frexp(float __arg, int *__exp) {
  return ::frexpf(__arg, __exp);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isinf(float __x) { return ::__isinff(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isinf(double __x) { return ::__isinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isfinite(float __x) { return ::__finitef(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isfinite(double __x) { return ::__finite(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isnan(float __x) { return ::__isnanf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isnan(double __x) { return ::__isnan(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreater(float __x, float __y) {
  return __builtin_isgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreater(double __x, double __y) {
  return __builtin_isgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreaterequal(float __x, float __y) {
  return __builtin_isgreaterequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isgreaterequal(double __x, double __y) {
  return __builtin_isgreaterequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isless(float __x, float __y) {
  return __builtin_isless(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isless(double __x, double __y) {
  return __builtin_isless(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessequal(float __x, float __y) {
  return __builtin_islessequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessequal(double __x, double __y) {
  return __builtin_islessequal(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessgreater(float __x, float __y) {
  return __builtin_islessgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool islessgreater(double __x, double __y) {
  return __builtin_islessgreater(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isnormal(float __x) {
  return __builtin_isnormal(__x);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isnormal(double __x) {
  return __builtin_isnormal(__x);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isunordered(float __x, float __y) {
  return __builtin_isunordered(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool isunordered(double __x, double __y) {
  return __builtin_isunordered(__x, __y);
}
static __attribute__((device)) inline __attribute__((always_inline)) float modf(float __x, float *__iptr) {
  return ::modff(__x, __iptr);
}
static __attribute__((device)) inline __attribute__((always_inline)) float pow(float __base, int __iexp) {
  return ::powif(__base, __iexp);
}
static __attribute__((device)) inline __attribute__((always_inline)) double pow(double __base, int __iexp) {
  return ::powi(__base, __iexp);
}
static __attribute__((device)) inline __attribute__((always_inline)) float remquo(float __x, float __y, int *__quo) {
  return ::remquof(__x, __y, __quo);
}
static __attribute__((device)) inline __attribute__((always_inline)) float scalbln(float __x, long int __n) {
  return ::scalblnf(__x, __n);
}
static __attribute__((device)) inline __attribute__((always_inline)) bool signbit(float __x) { return ::__signbitf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) bool signbit(double __x) { return ::__signbit(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) _Float16 fma(_Float16 __x, _Float16 __y,
                                      _Float16 __z) {
  return __builtin_fmaf16(__x, __y, __z);
}
static __attribute__((device)) inline __attribute__((always_inline)) _Float16 pow(_Float16 __base, int __iexp) {
  return __ocml_pown_f16(__base, __iexp);
}
static __attribute__((device)) inline __attribute__((always_inline)) float acos(float __x) { return acosf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float acosh(float __x) { return acoshf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float asin(float __x) { return asinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float asinh(float __x) { return asinhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float atan(float __x) { return atanf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float atan2(float __x, float __y) { return atan2f(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float atanh(float __x) { return atanhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float cbrt(float __x) { return cbrtf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float ceil(float __x) { return ceilf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float copysign(float __x, float __y) { return copysignf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float cos(float __x) { return cosf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float cosh(float __x) { return coshf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float erf(float __x) { return erff(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float erfc(float __x) { return erfcf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float exp(float __x) { return expf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float exp2(float __x) { return exp2f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float expm1(float __x) { return expm1f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float fabs(float __x) { return fabsf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float fdim(float __x, float __y) { return fdimf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float floor(float __x) { return floorf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float fmax(float __x, float __y) { return fmaxf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float fmin(float __x, float __y) { return fminf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float fmod(float __x, float __y) { return fmodf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float hypot(float __x, float __y) { return hypotf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) int ilogb(float __x) { return ilogbf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float ldexp(float __x, int __y) { return ldexpf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float lgamma(float __x) { return lgammaf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log(float __x) { return logf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log10(float __x) { return log10f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log1p(float __x) { return log1pf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float log2(float __x) { return log2f(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float logb(float __x) { return logbf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long long llrint(float __x) { return llrintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long long llround(float __x) { return llroundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long lrint(float __x) { return lrintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) long lround(float __x) { return lroundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float nearbyint(float __x) { return nearbyintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float nextafter(float __x, float __y) { return nextafterf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float pow(float __x, float __y) { return powf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float remainder(float __x, float __y) { return remainderf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float rint(float __x) { return rintf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float round(float __x) { return roundf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float scalbn(float __x, int __y) { return scalbnf(__x, __y); }
static __attribute__((device)) inline __attribute__((always_inline)) float sin(float __x) { return sinf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float sinh(float __x) { return sinhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float sqrt(float __x) { return sqrtf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float tan(float __x) { return tanf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float tanh(float __x) { return tanhf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float tgamma(float __x) { return tgammaf(__x); }
static __attribute__((device)) inline __attribute__((always_inline)) float trunc(float __x) { return truncf(__x); }
template <bool __B, class __T = void> struct __hip_enable_if {};
template <class __T> struct __hip_enable_if<true, __T> { typedef __T type; };
namespace __hip {
template <class _Tp> struct is_integral {
  enum { value = 0 };
};
template <> struct is_integral<bool> {
  enum { value = 1 };
};
template <> struct is_integral<char> {
  enum { value = 1 };
};
template <> struct is_integral<signed char> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned char> {
  enum { value = 1 };
};
template <> struct is_integral<wchar_t> {
  enum { value = 1 };
};
template <> struct is_integral<short> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned short> {
  enum { value = 1 };
};
template <> struct is_integral<int> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned int> {
  enum { value = 1 };
};
template <> struct is_integral<long> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned long> {
  enum { value = 1 };
};
template <> struct is_integral<long long> {
  enum { value = 1 };
};
template <> struct is_integral<unsigned long long> {
  enum { value = 1 };
};
template <class _Tp> struct is_arithmetic {
  enum { value = 0 };
};
template <> struct is_arithmetic<bool> {
  enum { value = 1 };
};
template <> struct is_arithmetic<char> {
  enum { value = 1 };
};
template <> struct is_arithmetic<signed char> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned char> {
  enum { value = 1 };
};
template <> struct is_arithmetic<wchar_t> {
  enum { value = 1 };
};
template <> struct is_arithmetic<short> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned short> {
  enum { value = 1 };
};
template <> struct is_arithmetic<int> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned int> {
  enum { value = 1 };
};
template <> struct is_arithmetic<long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<long long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<unsigned long long> {
  enum { value = 1 };
};
template <> struct is_arithmetic<float> {
  enum { value = 1 };
};
template <> struct is_arithmetic<double> {
  enum { value = 1 };
};
struct true_type {
  static const __attribute__((constant)) bool value = true;
};
struct false_type {
  static const __attribute__((constant)) bool value = false;
};
template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};
template <typename __T> struct add_rvalue_reference { typedef __T &&type; };
template <typename __T> typename add_rvalue_reference<__T>::type declval();
template <class _Tp> struct __numeric_type {
  static void __test(...);
  static _Float16 __test(_Float16);
  static float __test(float);
  static double __test(char);
  static double __test(int);
  static double __test(unsigned);
  static double __test(long);
  static double __test(unsigned long);
  static double __test(long long);
  static double __test(unsigned long long);
  static double __test(double);
  static double __test(long double);
  template <typename _U>
  static auto __test_impl(int) -> decltype(__test(declval<_U>()));
  template <typename _U> static void __test_impl(...);
  typedef decltype(__test_impl<_Tp>(0)) type;
  static const bool value = !is_same<type, void>::value;
};
template <> struct __numeric_type<void> { static const bool value = true; };
template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&__numeric_type<_A2>::value
              &&__numeric_type<_A3>::value>
class __promote_imp {
public:
  static const bool value = false;
};
template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
  typedef typename __promote_imp<_A3>::type __type3;
public:
  typedef decltype(__type1() + __type2() + __type3()) type;
  static const bool value = true;
};
template <class _A1, class _A2> class __promote_imp<_A1, _A2, void, true> {
private:
  typedef typename __promote_imp<_A1>::type __type1;
  typedef typename __promote_imp<_A2>::type __type2;
public:
  typedef decltype(__type1() + __type2()) type;
  static const bool value = true;
};
template <class _A1> class __promote_imp<_A1, void, void, true> {
public:
  typedef typename __numeric_type<_A1>::type type;
  static const bool value = true;
};
template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};
}
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type acos(__T __x) { return ::acos((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type acosh(__T __x) { return ::acosh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type asin(__T __x) { return ::asin((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type asinh(__T __x) { return ::asinh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type atan(__T __x) { return ::atan((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type atan2(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return atan2((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type atanh(__T __x) { return ::atanh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type cbrt(__T __x) { return ::cbrt((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type ceil(__T __x) { return ::ceil((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type copysign(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return copysign((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type cos(__T __x) { return ::cos((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type cosh(__T __x) { return ::cosh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type erf(__T __x) { return ::erf((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type erfc(__T __x) { return ::erfc((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type exp(__T __x) { return ::exp((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type exp2(__T __x) { return ::exp2((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type expm1(__T __x) { return ::expm1((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type fabs(__T __x) { return ::fabs((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type fdim(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return fdim((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type floor(__T __x) { return ::floor((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type fmax(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return fmax((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type fmin(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return fmin((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type fmod(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return fmod((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type hypot(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return hypot((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, int>::type ilogb(__T __x) { return ::ilogb((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isfinite(__T __x) { return ::isfinite((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, bool>::type isgreater(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return isgreater((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, bool>::type isgreaterequal(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return isgreaterequal((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isinf(__T __x) { return ::isinf((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, bool>::type isless(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return isless((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, bool>::type islessequal(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return islessequal((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, bool>::type islessgreater(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return islessgreater((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isnan(__T __x) { return ::isnan((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type isnormal(__T __x) { return ::isnormal((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, bool>::type isunordered(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return isunordered((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type lgamma(__T __x) { return ::lgamma((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log(__T __x) { return ::log((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log10(__T __x) { return ::log10((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log1p(__T __x) { return ::log1p((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type log2(__T __x) { return ::log2((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type logb(__T __x) { return ::logb((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long long>::type llrint(__T __x) { return ::llrint((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long long>::type llround(__T __x) { return ::llround((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long>::type lrint(__T __x) { return ::lrint((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, long>::type lround(__T __x) { return ::lround((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type nearbyint(__T __x) { return ::nearbyint((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type nextafter(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return nextafter((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type pow(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return pow((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type remainder(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return remainder((__arg_type)__x, (__arg_type)__y); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type rint(__T __x) { return ::rint((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type round(__T __x) { return ::round((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, bool>::type signbit(__T __x) { return ::signbit((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type sin(__T __x) { return ::sin((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type sinh(__T __x) { return ::sinh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type sqrt(__T __x) { return ::sqrt((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type tan(__T __x) { return ::tan((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type tanh(__T __x) { return ::tanh((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type tgamma(__T __x) { return ::tgamma((double)__x); }
template <typename __T> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type trunc(__T __x) { return ::trunc((double)__x); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type max(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return max((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2> static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<__hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value, double>::type min(__T1 __x, __T2 __y) { typedef typename __hip::__promote<__T1, __T2>::type __arg_type; return min((__arg_type)__x, (__arg_type)__y); }
template <typename __T1, typename __T2, typename __T3>
static __attribute__((device)) inline __attribute__((always_inline)) typename __hip_enable_if<
    __hip::is_arithmetic<__T1>::value && __hip::is_arithmetic<__T2>::value &&
        __hip::is_arithmetic<__T3>::value,
    typename __hip::__promote<__T1, __T2, __T3>::type>::type
fma(__T1 __x, __T2 __y, __T3 __z) {
  typedef typename __hip::__promote<__T1, __T2, __T3>::type __result_type;
  return ::fma((__result_type)__x, (__result_type)__y, (__result_type)__z);
}
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    frexp(__T __x, int *__exp) {
  return ::frexp((double)__x, __exp);
}
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    ldexp(__T __x, int __exp) {
  return ::ldexp((double)__x, __exp);
}
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    modf(__T __x, double *__exp) {
  return ::modf((double)__x, __exp);
}
template <typename __T1, typename __T2>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_arithmetic<__T1>::value &&
                                 __hip::is_arithmetic<__T2>::value,
                             typename __hip::__promote<__T1, __T2>::type>::type
    remquo(__T1 __x, __T2 __y, int *__quo) {
  typedef typename __hip::__promote<__T1, __T2>::type __result_type;
  return ::remquo((__result_type)__x, (__result_type)__y, __quo);
}
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    scalbln(__T __x, long int __exp) {
  return ::scalbln((double)__x, __exp);
}
template <typename __T>
static __attribute__((device)) inline __attribute__((always_inline))
    typename __hip_enable_if<__hip::is_integral<__T>::value, double>::type
    scalbn(__T __x, int __exp) {
  return ::scalbn((double)__x, __exp);
}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#pragma clang diagnostic pop
extern "C" {
const char* amd_dbgapi_get_build_name();
const char* amd_dbgapi_get_git_hash();
size_t amd_dbgapi_get_build_id();
}
namespace __hip_internal {
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
typedef unsigned long size_t;
template <class _Tp, _Tp __v> struct integral_constant {
  static constexpr const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
  constexpr operator value_type() const { return value; }
  constexpr value_type operator()() const { return value; }
};
template <class _Tp, _Tp __v> constexpr const _Tp integral_constant<_Tp, __v>::value;
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;
template <bool B> using bool_constant = integral_constant<bool, B>;
typedef bool_constant<true> true_type;
typedef bool_constant<false> false_type;
template <bool __B, class __T = void> struct enable_if {};
template <class __T> struct enable_if<true, __T> {
  typedef __T type;
};
template <bool _B> struct true_or_false_type : public false_type {};
template <> struct true_or_false_type<true> : public true_type {};
template <class _Tp> struct is_integral : public false_type {};
template <> struct is_integral<bool> : public true_type {};
template <> struct is_integral<char> : public true_type {};
template <> struct is_integral<signed char> : public true_type {};
template <> struct is_integral<unsigned char> : public true_type {};
template <> struct is_integral<wchar_t> : public true_type {};
template <> struct is_integral<short> : public true_type {};
template <> struct is_integral<unsigned short> : public true_type {};
template <> struct is_integral<int> : public true_type {};
template <> struct is_integral<unsigned int> : public true_type {};
template <> struct is_integral<long> : public true_type {};
template <> struct is_integral<unsigned long> : public true_type {};
template <> struct is_integral<long long> : public true_type {};
template <> struct is_integral<unsigned long long> : public true_type {};
template <class _Tp> struct is_arithmetic : public false_type {};
template <> struct is_arithmetic<bool> : public true_type {};
template <> struct is_arithmetic<char> : public true_type {};
template <> struct is_arithmetic<signed char> : public true_type {};
template <> struct is_arithmetic<unsigned char> : public true_type {};
template <> struct is_arithmetic<wchar_t> : public true_type {};
template <> struct is_arithmetic<short> : public true_type {};
template <> struct is_arithmetic<unsigned short> : public true_type {};
template <> struct is_arithmetic<int> : public true_type {};
template <> struct is_arithmetic<unsigned int> : public true_type {};
template <> struct is_arithmetic<long> : public true_type {};
template <> struct is_arithmetic<unsigned long> : public true_type {};
template <> struct is_arithmetic<long long> : public true_type {};
template <> struct is_arithmetic<unsigned long long> : public true_type {};
template <> struct is_arithmetic<float> : public true_type {};
template <> struct is_arithmetic<double> : public true_type {};
template <typename _Tp> struct is_floating_point : public false_type {};
template <> struct is_floating_point<float> : public true_type {};
template <> struct is_floating_point<double> : public true_type {};
template <> struct is_floating_point<long double> : public true_type {};
template <typename __T, typename __U> struct is_same : public false_type {};
template <typename __T> struct is_same<__T, __T> : public true_type {};
template <typename _Tp, bool = is_arithmetic<_Tp>::value> struct is_signed : public false_type {};
template <typename _Tp> struct is_signed<_Tp, true> : public true_or_false_type<_Tp(-1) < _Tp(0)> {
};
template< class... >
using void_t = void;
template <typename _Tp> struct numeric_limits {
  static constexpr bool is_specialized = false;
  static constexpr _Tp min() noexcept { return _Tp(); }
  static constexpr _Tp max() noexcept { return _Tp(); }
  static constexpr _Tp lowest() noexcept { return _Tp(); }
};
template <size_t _NumBytes> struct __hip_internal_unsigned_of_size;
template <> struct __hip_internal_unsigned_of_size<1> {
  using type = unsigned char;
};
template <> struct __hip_internal_unsigned_of_size<2> {
  using type = unsigned short;
};
template <> struct __hip_internal_unsigned_of_size<4> {
  using type = unsigned int;
};
template <> struct __hip_internal_unsigned_of_size<8> {
  using type = unsigned long long;
};
template <typename _Type>
constexpr _Type __hip_internal_unsigned_max() noexcept {
  return static_cast<_Type>(~static_cast<_Type>(0));
}
template <typename _Type>
constexpr _Type __hip_internal_signed_max() noexcept {
  using _UnsignedType = typename __hip_internal_unsigned_of_size<sizeof(_Type)>::type;
  return static_cast<_Type>(__hip_internal_unsigned_max<_UnsignedType>() >> 1);
}
template <typename _Type>
constexpr _Type __hip_internal_signed_min() noexcept {
  return static_cast<_Type>(-__hip_internal_signed_max<_Type>() - 1);
}
template <> struct numeric_limits<signed char> { static constexpr bool is_specialized = true; static constexpr bool is_signed = true; static constexpr signed char min() noexcept { return __hip_internal_signed_min<signed char>(); } static constexpr signed char max() noexcept { return __hip_internal_signed_max<signed char>(); } static constexpr signed char lowest() noexcept { return min(); } };
template <> struct numeric_limits<unsigned char> { static constexpr bool is_specialized = true; static constexpr bool is_signed = false; static constexpr unsigned char min() noexcept { return static_cast<unsigned char>(0); } static constexpr unsigned char max() noexcept { return __hip_internal_unsigned_max<unsigned char>(); } static constexpr unsigned char lowest() noexcept { return min(); } };
template <> struct numeric_limits<short> { static constexpr bool is_specialized = true; static constexpr bool is_signed = true; static constexpr short min() noexcept { return __hip_internal_signed_min<short>(); } static constexpr short max() noexcept { return __hip_internal_signed_max<short>(); } static constexpr short lowest() noexcept { return min(); } };
template <> struct numeric_limits<unsigned short> { static constexpr bool is_specialized = true; static constexpr bool is_signed = false; static constexpr unsigned short min() noexcept { return static_cast<unsigned short>(0); } static constexpr unsigned short max() noexcept { return __hip_internal_unsigned_max<unsigned short>(); } static constexpr unsigned short lowest() noexcept { return min(); } };
template <> struct numeric_limits<int> { static constexpr bool is_specialized = true; static constexpr bool is_signed = true; static constexpr int min() noexcept { return __hip_internal_signed_min<int>(); } static constexpr int max() noexcept { return __hip_internal_signed_max<int>(); } static constexpr int lowest() noexcept { return min(); } };
template <> struct numeric_limits<unsigned int> { static constexpr bool is_specialized = true; static constexpr bool is_signed = false; static constexpr unsigned int min() noexcept { return static_cast<unsigned int>(0); } static constexpr unsigned int max() noexcept { return __hip_internal_unsigned_max<unsigned int>(); } static constexpr unsigned int lowest() noexcept { return min(); } };
template <> struct numeric_limits<long> { static constexpr bool is_specialized = true; static constexpr bool is_signed = true; static constexpr long min() noexcept { return __hip_internal_signed_min<long>(); } static constexpr long max() noexcept { return __hip_internal_signed_max<long>(); } static constexpr long lowest() noexcept { return min(); } };
template <> struct numeric_limits<unsigned long> { static constexpr bool is_specialized = true; static constexpr bool is_signed = false; static constexpr unsigned long min() noexcept { return static_cast<unsigned long>(0); } static constexpr unsigned long max() noexcept { return __hip_internal_unsigned_max<unsigned long>(); } static constexpr unsigned long lowest() noexcept { return min(); } };
template <> struct numeric_limits<long long> { static constexpr bool is_specialized = true; static constexpr bool is_signed = true; static constexpr long long min() noexcept { return __hip_internal_signed_min<long long>(); } static constexpr long long max() noexcept { return __hip_internal_signed_max<long long>(); } static constexpr long long lowest() noexcept { return min(); } };
template <> struct numeric_limits<unsigned long long> { static constexpr bool is_specialized = true; static constexpr bool is_signed = false; static constexpr unsigned long long min() noexcept { return static_cast<unsigned long long>(0); } static constexpr unsigned long long max() noexcept { return __hip_internal_unsigned_max<unsigned long long>(); } static constexpr unsigned long long lowest() noexcept { return min(); } };
template <> struct numeric_limits<char> {
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = __hip_internal::is_signed<char>::value;
  static constexpr char min() noexcept {
    return is_signed ? __hip_internal_signed_min<char>() : static_cast<char>(0);
  }
  static constexpr char max() noexcept {
    return is_signed ? __hip_internal_signed_max<char>()
                     : static_cast<char>(__hip_internal_unsigned_max<unsigned char>());
  }
  static constexpr char lowest() noexcept { return min(); }
};
template <class T> auto test_returnable(int)
    -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});
template <class> auto test_returnable(...) -> false_type;
template <class T> struct type_identity {
  using type = T;
};
template <class T>
auto try_add_lvalue_reference(int) -> type_identity<T&>;
template <class T>
auto try_add_lvalue_reference(...) -> type_identity<T>;
template <class T> auto try_add_rvalue_reference(int) -> type_identity<T&&>;
template <class T> auto try_add_rvalue_reference(...) -> type_identity<T>;
template <class T> struct add_lvalue_reference : decltype(try_add_lvalue_reference<T>(0)) {};
template <class T> struct add_rvalue_reference : decltype(try_add_rvalue_reference<T>(0)) {};
template <typename T> typename add_rvalue_reference<T>::type declval() noexcept;
template <class From, class To> auto test_implicitly_convertible(int)
    -> decltype(void(declval<void (&)(To)>()(declval<From>())), true_type{});
template <class, class> auto test_implicitly_convertible(...) -> false_type;
template <class T> struct remove_cv {
  typedef T type;
};
template <class T> struct remove_cv<const T> {
  typedef T type;
};
template <class T> struct remove_cv<volatile T> {
  typedef T type;
};
template <class T> struct remove_cv<const volatile T> {
  typedef T type;
};
template <typename T>
struct remove_reference
{ using type = T; };
template <typename T>
struct remove_reference<T&>
{ using type = T; };
template <typename T>
struct remove_reference<T&&>
{ using type = T; };
template <typename T>
struct remove_cvref {
  using type = typename remove_cv<typename remove_reference<T>::type>::type;
};
template <class T> struct is_void : public is_same<void, typename remove_cv<T>::type> {};
template <class From, class To> struct is_convertible
    : public integral_constant<bool, (decltype(test_returnable<To>(0))::value &&
                                      decltype(test_implicitly_convertible<From, To>(0))::value) ||
                                         (is_void<From>::value && is_void<To>::value)> {};
template <typename _CharT> struct char_traits;
template <typename _CharT, typename _Traits = char_traits<_CharT>> class basic_istream;
template <typename _CharT, typename _Traits = char_traits<_CharT>> class basic_ostream;
typedef basic_istream<char> istream;
typedef basic_ostream<char> ostream;
template <typename _Tp> struct is_standard_layout
    : public integral_constant<bool, __is_standard_layout(_Tp)> {};
template <typename _Tp> struct is_trivial : public integral_constant<bool, __is_trivial(_Tp)> {};
template <typename _Tp> struct is_trivially_copyable :
  public integral_constant<bool, __is_trivially_copyable(_Tp)> {};
template <bool B, class T, class F> struct conditional {
  using type = T;
};
template <class T, class F> struct conditional<false, T, F> {
  using type = F;
};
template <class T> struct alignment_of : integral_constant<size_t, alignof(T)> {};
template <typename T, T... Ints> struct integer_sequence {
  using value_type = T;
  static constexpr size_t size() noexcept { return sizeof...(Ints); }
};
template <size_t... Ints> using index_sequence = integer_sequence<size_t, Ints...>;
template <size_t _hip_N, size_t... Ints> struct make_index_sequence_impl
    : make_index_sequence_impl<_hip_N - 1, _hip_N - 1, Ints...> {};
template <size_t... Ints> struct make_index_sequence_impl<0, Ints...> {
  using type = index_sequence<Ints...>;
};
template <size_t _hip_N> using make_index_sequence =
    typename make_index_sequence_impl<_hip_N>::type;
template <size_t... Ints>
constexpr index_sequence<Ints...> make_index_sequence_value(index_sequence<Ints...>) {
  return {};
}
template <typename T>
struct NumericLimits;
template <>
struct NumericLimits<int> {
    static constexpr int maximum() { return 0x7FFFFFFF; }
    static constexpr int minimum() { return ~0x7FFFFFFF; }
};
template <>
struct NumericLimits<unsigned int> {
    static constexpr unsigned int maximum() { return 0xFFFFFFFFu; }
    static constexpr unsigned int minimum() { return 0u; }
};
template <>
struct NumericLimits<long long> {
    static constexpr long long maximum() { return 0x7FFFFFFFFFFFFFFFLL; }
    static constexpr long long minimum() { return ~0x7FFFFFFFFFFFFFFFLL; }
};
template <>
struct NumericLimits<unsigned long long> {
    static constexpr unsigned long long maximum() { return 0xFFFFFFFFFFFFFFFFull; }
    static constexpr unsigned long long minimum() { return 0ull; }
};
template <>
struct NumericLimits<float> {
  static constexpr float maximum() { return __builtin_bit_cast(float, 0x7f800000); }
  static constexpr float minimum() { return -maximum(); }
  static constexpr float quiet_NaN() { return __builtin_nanf(""); }
  static constexpr float infinity() { return __builtin_huge_valf(); }
};
template <>
struct NumericLimits<double> {
  static constexpr double maximum() { return __builtin_bit_cast(double, 0x7FF0000000000000LL); }
  static constexpr double minimum() { return -maximum(); }
};
inline constexpr float copysign(float x, float y) { return __builtin_copysignf(x, y); }
}
typedef __hip_internal::uint8_t __hip_uint8_t;
typedef __hip_internal::uint16_t __hip_uint16_t;
typedef __hip_internal::uint32_t __hip_uint32_t;
typedef __hip_internal::uint64_t __hip_uint64_t;
typedef __hip_internal::int8_t __hip_int8_t;
typedef __hip_internal::int16_t __hip_int16_t;
typedef __hip_internal::int32_t __hip_int32_t;
typedef __hip_internal::int64_t __hip_int64_t;
template <typename T, unsigned int n> struct HIP_vector_base;
template <typename T, unsigned int rank> struct HIP_vector_type;
namespace hip_impl {
template <typename T, unsigned int n> __attribute__((always_inline)) __attribute__((device))
    typename HIP_vector_base<T, n>::Native_vec_*
    get_native_pointer(HIP_vector_base<T, n>& base_vec) {
  static_assert(sizeof(base_vec) == sizeof(typename HIP_vector_base<T, n>::Native_vec_));
  static_assert(
      (__hip_internal::alignment_of<HIP_vector_base<T, n>>::value %
       __hip_internal::alignment_of<typename HIP_vector_base<T, n>::Native_vec_>::value) == 0);
  return reinterpret_cast<typename HIP_vector_base<T, n>::Native_vec_*>(&base_vec);
};
template <typename T, unsigned int n>
__attribute__((always_inline)) __attribute__((device)) const typename HIP_vector_base<T, n>::Native_vec_*
get_native_pointer(const HIP_vector_base<T, n>& base_vec) {
  static_assert(sizeof(base_vec) == sizeof(typename HIP_vector_base<T, n>::Native_vec_));
  static_assert(
      (__hip_internal::alignment_of<HIP_vector_base<T, n>>::value %
       __hip_internal::alignment_of<typename HIP_vector_base<T, n>::Native_vec_>::value) == 0);
  return reinterpret_cast<const typename HIP_vector_base<T, n>::Native_vec_*>(&base_vec);
};
}
template <typename T, unsigned int n> __attribute__((always_inline)) __attribute__((device))
    typename HIP_vector_base<T, n>::Native_vec_&
    get_native_vector(HIP_vector_base<T, n>& base_vec) {
  return *hip_impl::get_native_pointer(base_vec);
};
template <typename T, unsigned int n>
__attribute__((always_inline)) __attribute__((device)) const typename HIP_vector_base<T, n>::Native_vec_&
get_native_vector(const HIP_vector_base<T, n>& base_vec) {
  return *hip_impl::get_native_pointer(base_vec);
};
template <typename T> struct HIP_vector_base<T, 1> {
  using Native_vec_ = T __attribute__((ext_vector_type(1)));
  T x;
  using value_type = T;
  __attribute__((device))
  HIP_vector_base() = default;
  __attribute__((device))
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __attribute__((device))
  explicit constexpr HIP_vector_base(T x_) : x(x_) {}
  __attribute__((device))
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __attribute__((device))
  ~HIP_vector_base() = default;
  __attribute__((device))
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
};
template <typename T> struct alignas(2 * sizeof(T)) HIP_vector_base<T, 2> {
  using Native_vec_ = T __attribute__((ext_vector_type(2)));
  T x, y;
  using value_type = T;
  __attribute__((device))
  HIP_vector_base() = default;
  __attribute__((device))
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __attribute__((device))
  constexpr HIP_vector_base(T x_, T y_ = T()) : x(x_), y(y_) {}
  __attribute__((device))
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __attribute__((device))
  ~HIP_vector_base() = default;
  __attribute__((device))
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
};
template <typename T> struct HIP_vector_base<T, 3> {
  struct Native_vec_ {
    T d[3];
    __attribute__((device))
    Native_vec_() = default;
    __attribute__((device))
    explicit constexpr Native_vec_(T x_) noexcept : d{x_, x_, x_} {}
    __attribute__((device))
    constexpr Native_vec_(T x_, T y_, T z_) noexcept : d{x_, y_, z_} {}
    __attribute__((device))
    constexpr Native_vec_(const Native_vec_&) = default;
    __attribute__((device))
    constexpr Native_vec_(Native_vec_&&) = default;
    __attribute__((device))
    ~Native_vec_() = default;
    __attribute__((device))
    Native_vec_& operator=(const Native_vec_&) = default;
    __attribute__((device))
    Native_vec_& operator=(Native_vec_&&) = default;
    __attribute__((device))
    T& operator[](unsigned int idx) noexcept { return d[idx]; }
    __attribute__((device))
    T operator[](unsigned int idx) const noexcept { return d[idx]; }
    __attribute__((device))
    Native_vec_& operator+=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] += x_.d[i];
      return *this;
    }
    __attribute__((device))
    Native_vec_& operator-=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] -= x_.d[i];
      return *this;
    }
    __attribute__((device))
    Native_vec_& operator*=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] *= x_.d[i];
      return *this;
    }
    __attribute__((device))
    Native_vec_& operator/=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] /= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_signed<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_ operator-() const noexcept {
      auto r{*this};
      for (auto&& x : r.d) x = -x;
      return r;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_ operator~() const noexcept {
      auto r{*this};
      for (auto&& x : r.d) x = ~x;
      return r;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_& operator%=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] %= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_& operator^=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] ^= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_& operator|=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] |= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_& operator&=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] &= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_& operator>>=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] >>= x_.d[i];
      return *this;
    }
    template <typename U = T,
              typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
    __attribute__((device)) Native_vec_& operator<<=(const Native_vec_& x_) noexcept {
      for (auto i = 0u; i != 3u; ++i) d[i] <<= x_.d[i];
      return *this;
    }
    using Vec3_cmp = int __attribute__((vector_size(4 * sizeof(int))));
    __attribute__((device))
    Vec3_cmp operator==(const Native_vec_& x_) const noexcept {
      return Vec3_cmp{d[0] == x_.d[0], d[1] == x_.d[1], d[2] == x_.d[2]};
    }
  };
  T x, y, z;
  using value_type = T;
  __attribute__((device))
  HIP_vector_base() = default;
  __attribute__((device))
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __attribute__((device))
  constexpr HIP_vector_base(T x_, T y_ = T(), T z_ = T()) : x(x_), y(y_), z(z_) {};
  __attribute__((device))
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __attribute__((device))
  ~HIP_vector_base() = default;
  __attribute__((device))
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
  __attribute__((device))
  HIP_vector_base& operator=(HIP_vector_base&&) = default;
};
template <typename T> struct alignas(4 * sizeof(T)) HIP_vector_base<T, 4> {
  using Native_vec_ = T __attribute__((ext_vector_type(4)));
  T x, y, z, w;
  using value_type = T;
  __attribute__((device))
  HIP_vector_base() = default;
  __attribute__((device))
  constexpr HIP_vector_base(const HIP_vector_base&) = default;
  __attribute__((device))
  constexpr HIP_vector_base(T x_, T y_ = T(), T z_ = T(), T w_ = T())
      : x(x_), y(y_), z(z_), w(w_) {};
  __attribute__((device))
  constexpr HIP_vector_base(HIP_vector_base&&) = default;
  __attribute__((device))
  ~HIP_vector_base() = default;
  __attribute__((device))
  HIP_vector_base& operator=(const HIP_vector_base&) = default;
};
template <typename T, size_t rank, size_t... indices>
constexpr inline __attribute__((device)) HIP_vector_type<T, rank> make_vector_type_impl(
    T val, __hip_internal::index_sequence<indices...>) noexcept {
  return HIP_vector_type<T, rank>{((void)indices, val)...};
}
template <typename T, unsigned int rank>
constexpr inline __attribute__((device)) HIP_vector_type<T, rank> make_vector_type(T val) {
  return make_vector_type_impl<T, rank>(
      val, __hip_internal::make_index_sequence_value(__hip_internal::make_index_sequence<rank>{}));
}
template <typename T, unsigned int rank> struct HIP_vector_type : public HIP_vector_base<T, rank> {
  using typename HIP_vector_base<T, rank>::Native_vec_;
  __attribute__((device))
  HIP_vector_type() = default;
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>::value>::type* = nullptr>
  __attribute__((device)) explicit constexpr HIP_vector_type(U x_) noexcept
      : HIP_vector_base<T, rank>{static_cast<T>(x_)} {}
  template <
      typename... Us,
      typename __hip_internal::enable_if<(rank > 1) && sizeof...(Us) == rank>::type* = nullptr>
  __attribute__((device)) constexpr HIP_vector_type(Us... xs) noexcept
      : HIP_vector_base<T, rank>{static_cast<T>(xs)...} {}
  __attribute__((device))
  constexpr HIP_vector_type(const HIP_vector_type&) = default;
  __attribute__((device))
  constexpr HIP_vector_type(HIP_vector_type&&) = default;
  __attribute__((device))
  ~HIP_vector_type() = default;
  __attribute__((device))
  HIP_vector_type& operator=(const HIP_vector_type&) = default;
  __attribute__((device))
  HIP_vector_type& operator=(HIP_vector_type&&) = default;
  __attribute__((device))
  T& operator[](size_t idx) noexcept { return reinterpret_cast<T*>(this)[idx]; }
  __attribute__((device))
  const T& operator[](size_t idx) const noexcept { return reinterpret_cast<const T*>(this)[idx]; }
  __attribute__((device))
  HIP_vector_type& operator++() noexcept {
    HIP_vector_type unity = make_vector_type<T, rank>(1);
    return *this += unity;
  }
  __attribute__((device))
  HIP_vector_type operator++(int) noexcept {
    auto tmp(*this);
    ++*this;
    return tmp;
  }
  __attribute__((device))
  HIP_vector_type& operator--() noexcept {
    HIP_vector_type unity = make_vector_type<T, rank>(1);
    return *this -= unity;
  }
  __attribute__((device))
  HIP_vector_type operator--(int) noexcept {
    auto tmp(*this);
    --*this;
    return tmp;
  }
  __attribute__((device)) HIP_vector_type& operator+=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) += get_native_vector(x);
    return *this;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator+=(U x) noexcept {
    return *this += make_vector_type<T, rank>(x);
  }
  __attribute__((device)) HIP_vector_type& operator-=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) -= get_native_vector(x);
    return *this;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator-=(U x) noexcept {
    return *this -= make_vector_type<T, rank>(x);
  }
  __attribute__((device)) HIP_vector_type& operator*=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) *= get_native_vector(x);
    return *this;
  }
  friend __attribute__((device)) inline constexpr HIP_vector_type operator*(
      HIP_vector_type x, const HIP_vector_type& y) noexcept {
    return HIP_vector_type{x} *= y;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator*=(U x) noexcept {
    return *this *= make_vector_type<T, rank>(x);
  }
  friend __attribute__((device)) inline constexpr HIP_vector_type operator/(
      HIP_vector_type x, const HIP_vector_type& y) noexcept {
    return HIP_vector_type{x} /= y;
  }
  __attribute__((device)) HIP_vector_type& operator/=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) /= get_native_vector(x);
    return *this;
  }
  template <typename U, typename __hip_internal::enable_if<
                            __hip_internal::is_convertible<U, T>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator/=(U x) noexcept {
    return *this /= make_vector_type<T, rank>(x);
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_signed<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type operator-() const noexcept {
    auto tmp(*this);
    get_native_vector(tmp) = -get_native_vector(tmp);
    return tmp;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type operator~() const noexcept {
    HIP_vector_type r{*this};
    get_native_vector(r) = ~get_native_vector(r);
    return r;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator%=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) %= get_native_vector(x);
    return *this;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator^=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) ^= get_native_vector(x);
    return *this;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator|=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) |= get_native_vector(x);
    return *this;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator&=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) &= get_native_vector(x);
    return *this;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator>>=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) >>= get_native_vector(x);
    return *this;
  }
  template <typename U = T,
            typename __hip_internal::enable_if<__hip_internal::is_integral<U>{}>::type* = nullptr>
  __attribute__((device)) HIP_vector_type& operator<<=(const HIP_vector_type& x) noexcept {
    get_native_vector(*this) <<= get_native_vector(x);
    return *this;
  }
};
template <typename T, unsigned int n>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator+(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} += y;
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator+(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} += make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator+(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) += y;
}
template <typename T, unsigned int n>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator-(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} -= y;
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator-(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} -= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator-(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) -= y;
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator*(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} *= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator*(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) *= y;
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator/(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} /= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator/(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) /= y;
}
template <typename T, unsigned int n> __attribute__((device)) inline
    bool
    operator==(const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  bool isTrue = true;
  const auto& native_x = get_native_vector(x);
  const auto& native_y = get_native_vector(y);
  for (unsigned int i = 0; i < n; ++i) {
    isTrue = (isTrue && (native_x[i] == native_y[i]));
  }
  return isTrue;
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr bool operator==(const HIP_vector_type<T, n>& x, U y) noexcept {
  return x == make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr bool operator==(U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) == y;
}
template <typename T, unsigned int n>
__attribute__((device)) inline constexpr bool operator!=(const HIP_vector_type<T, n>& x,
                                                 const HIP_vector_type<T, n>& y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr bool operator!=(const HIP_vector_type<T, n>& x, U y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n, typename U>
__attribute__((device)) inline constexpr bool operator!=(U x, const HIP_vector_type<T, n>& y) noexcept {
  return !(x == y);
}
template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator%(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} %= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator%(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} %= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator%(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) %= y;
}
template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator^(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} ^= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator^(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} ^= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator^(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) ^= y;
}
template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator|(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} |= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator|(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} |= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator|(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) |= y;
}
template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator&(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} &= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator&(const HIP_vector_type<T, n>& x,
                                                                 U y) noexcept {
  return HIP_vector_type<T, n>{x} &= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator&(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) &= y;
}
template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator>>(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} >>= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator>>(const HIP_vector_type<T, n>& x,
                                                                  U y) noexcept {
  return HIP_vector_type<T, n>{x} >>= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator>>(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) >>= y;
}
template <typename T, unsigned int n,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator<<(
    const HIP_vector_type<T, n>& x, const HIP_vector_type<T, n>& y) noexcept {
  return HIP_vector_type<T, n>{x} <<= y;
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator<<(const HIP_vector_type<T, n>& x,
                                                                  U y) noexcept {
  return HIP_vector_type<T, n>{x} <<= make_vector_type<T, n>(y);
}
template <typename T, unsigned int n, typename U,
          typename __hip_internal::enable_if<__hip_internal::is_arithmetic<U>::value>::type,
          typename __hip_internal::enable_if<__hip_internal::is_integral<T>{}>* = nullptr>
__attribute__((device)) inline constexpr HIP_vector_type<T, n> operator<<(
    U x, const HIP_vector_type<T, n>& y) noexcept {
  return make_vector_type<T, n>(x) <<= y;
}
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
inline __attribute__((always_inline)) __attribute__((device))
    typename __hip_internal::enable_if<(rankT == 1 && rankU >= 1),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x));
};
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
inline __attribute__((always_inline)) __attribute__((device))
    typename __hip_internal::enable_if<(rankT == 2 && rankU == 1),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(0));
};
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
inline __attribute__((always_inline)) __attribute__((device))
    typename __hip_internal::enable_if<(rankT == 2 && rankU >= 2),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y));
};
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
inline __attribute__((always_inline)) __attribute__((device))
    typename __hip_internal::enable_if<(rankT == 4 && rankU == 1),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(0), static_cast<T>(0),
                                   static_cast<T>(0));
};
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
inline __attribute__((always_inline)) __attribute__((device))
    typename __hip_internal::enable_if<(rankT == 4 && rankU == 2),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y), static_cast<T>(0),
                                   static_cast<T>(0));
};
template <typename T, unsigned int rankT, typename U, unsigned int rankU>
inline __attribute__((always_inline)) __attribute__((device))
    typename __hip_internal::enable_if<(rankT == 4 && rankU == 4),
                                       const HIP_vector_type<T, rankT>>::type
    __hipMapVector(const HIP_vector_type<U, rankU>& u) {
  return HIP_vector_type<T, rankT>(static_cast<T>(u.x), static_cast<T>(u.y), static_cast<T>(u.z),
                                   static_cast<T>(u.w));
};
using uchar1 = HIP_vector_type<unsigned char, 1>; using uchar2 = HIP_vector_type<unsigned char, 2>; using uchar3 = HIP_vector_type<unsigned char, 3>; using uchar4 = HIP_vector_type<unsigned char, 4>;;
using char1 = HIP_vector_type<char, 1>; using char2 = HIP_vector_type<char, 2>; using char3 = HIP_vector_type<char, 3>; using char4 = HIP_vector_type<char, 4>;;
using ushort1 = HIP_vector_type<unsigned short, 1>; using ushort2 = HIP_vector_type<unsigned short, 2>; using ushort3 = HIP_vector_type<unsigned short, 3>; using ushort4 = HIP_vector_type<unsigned short, 4>;;
using short1 = HIP_vector_type<short, 1>; using short2 = HIP_vector_type<short, 2>; using short3 = HIP_vector_type<short, 3>; using short4 = HIP_vector_type<short, 4>;;
using uint1 = HIP_vector_type<unsigned int, 1>; using uint2 = HIP_vector_type<unsigned int, 2>; using uint3 = HIP_vector_type<unsigned int, 3>; using uint4 = HIP_vector_type<unsigned int, 4>;;
using int1 = HIP_vector_type<int, 1>; using int2 = HIP_vector_type<int, 2>; using int3 = HIP_vector_type<int, 3>; using int4 = HIP_vector_type<int, 4>;;
using ulong1 = HIP_vector_type<unsigned long, 1>; using ulong2 = HIP_vector_type<unsigned long, 2>; using ulong3 = HIP_vector_type<unsigned long, 3>; using ulong4 = HIP_vector_type<unsigned long, 4>;;
using long1 = HIP_vector_type<long, 1>; using long2 = HIP_vector_type<long, 2>; using long3 = HIP_vector_type<long, 3>; using long4 = HIP_vector_type<long, 4>;;
using ulonglong1 = HIP_vector_type<unsigned long long, 1>; using ulonglong2 = HIP_vector_type<unsigned long long, 2>; using ulonglong3 = HIP_vector_type<unsigned long long, 3>; using ulonglong4 = HIP_vector_type<unsigned long long, 4>;;
using longlong1 = HIP_vector_type<long long, 1>; using longlong2 = HIP_vector_type<long long, 2>; using longlong3 = HIP_vector_type<long long, 3>; using longlong4 = HIP_vector_type<long long, 4>;;
using float1 = HIP_vector_type<float, 1>; using float2 = HIP_vector_type<float, 2>; using float3 = HIP_vector_type<float, 3>; using float4 = HIP_vector_type<float, 4>;;
using double1 = HIP_vector_type<double, 1>; using double2 = HIP_vector_type<double, 2>; using double3 = HIP_vector_type<double, 3>; using double4 = HIP_vector_type<double, 4>;;
static inline __attribute__((device)) uchar1 make_uchar1(unsigned char x) { uchar1 r{x}; return r; };
static inline __attribute__((device)) uchar2 make_uchar2(unsigned char x, unsigned char y) { uchar2 r{x, y}; return r; };
static inline __attribute__((device)) uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z) { uchar3 r{x, y, z}; return r; };
static inline __attribute__((device)) uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w) { uchar4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) char1 make_char1(signed char x) { char1 r{x}; return r; };
static inline __attribute__((device)) char2 make_char2(signed char x, signed char y) { char2 r{x, y}; return r; };
static inline __attribute__((device)) char3 make_char3(signed char x, signed char y, signed char z) { char3 r{x, y, z}; return r; };
static inline __attribute__((device)) char4 make_char4(signed char x, signed char y, signed char z, signed char w) { char4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) ushort1 make_ushort1(unsigned short x) { ushort1 r{x}; return r; };
static inline __attribute__((device)) ushort2 make_ushort2(unsigned short x, unsigned short y) { ushort2 r{x, y}; return r; };
static inline __attribute__((device)) ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z) { ushort3 r{x, y, z}; return r; };
static inline __attribute__((device)) ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w) { ushort4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) short1 make_short1(signed short x) { short1 r{x}; return r; };
static inline __attribute__((device)) short2 make_short2(signed short x, signed short y) { short2 r{x, y}; return r; };
static inline __attribute__((device)) short3 make_short3(signed short x, signed short y, signed short z) { short3 r{x, y, z}; return r; };
static inline __attribute__((device)) short4 make_short4(signed short x, signed short y, signed short z, signed short w) { short4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) uint1 make_uint1(unsigned int x) { uint1 r{x}; return r; };
static inline __attribute__((device)) uint2 make_uint2(unsigned int x, unsigned int y) { uint2 r{x, y}; return r; };
static inline __attribute__((device)) uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { uint3 r{x, y, z}; return r; };
static inline __attribute__((device)) uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w) { uint4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) int1 make_int1(signed int x) { int1 r{x}; return r; };
static inline __attribute__((device)) int2 make_int2(signed int x, signed int y) { int2 r{x, y}; return r; };
static inline __attribute__((device)) int3 make_int3(signed int x, signed int y, signed int z) { int3 r{x, y, z}; return r; };
static inline __attribute__((device)) int4 make_int4(signed int x, signed int y, signed int z, signed int w) { int4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) float1 make_float1(float x) { float1 r{x}; return r; };
static inline __attribute__((device)) float2 make_float2(float x, float y) { float2 r{x, y}; return r; };
static inline __attribute__((device)) float3 make_float3(float x, float y, float z) { float3 r{x, y, z}; return r; };
static inline __attribute__((device)) float4 make_float4(float x, float y, float z, float w) { float4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) double1 make_double1(double x) { double1 r{x}; return r; };
static inline __attribute__((device)) double2 make_double2(double x, double y) { double2 r{x, y}; return r; };
static inline __attribute__((device)) double3 make_double3(double x, double y, double z) { double3 r{x, y, z}; return r; };
static inline __attribute__((device)) double4 make_double4(double x, double y, double z, double w) { double4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) ulong1 make_ulong1(unsigned long x) { ulong1 r{x}; return r; };
static inline __attribute__((device)) ulong2 make_ulong2(unsigned long x, unsigned long y) { ulong2 r{x, y}; return r; };
static inline __attribute__((device)) ulong3 make_ulong3(unsigned long x, unsigned long y, unsigned long z) { ulong3 r{x, y, z}; return r; };
static inline __attribute__((device)) ulong4 make_ulong4(unsigned long x, unsigned long y, unsigned long z, unsigned long w) { ulong4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) long1 make_long1(signed long x) { long1 r{x}; return r; };
static inline __attribute__((device)) long2 make_long2(signed long x, signed long y) { long2 r{x, y}; return r; };
static inline __attribute__((device)) long3 make_long3(signed long x, signed long y, signed long z) { long3 r{x, y, z}; return r; };
static inline __attribute__((device)) long4 make_long4(signed long x, signed long y, signed long z, signed long w) { long4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) ulonglong1 make_ulonglong1(unsigned long long x) { ulonglong1 r{x}; return r; };
static inline __attribute__((device)) ulonglong2 make_ulonglong2(unsigned long long x, unsigned long long y) { ulonglong2 r{x, y}; return r; };
static inline __attribute__((device)) ulonglong3 make_ulonglong3(unsigned long long x, unsigned long long y, unsigned long long z) { ulonglong3 r{x, y, z}; return r; };
static inline __attribute__((device)) ulonglong4 make_ulonglong4(unsigned long long x, unsigned long long y, unsigned long long z, unsigned long long w) { ulonglong4 r{x, y, z, w}; return r; };
static inline __attribute__((device)) longlong1 make_longlong1(signed long long x) { longlong1 r{x}; return r; };
static inline __attribute__((device)) longlong2 make_longlong2(signed long long x, signed long long y) { longlong2 r{x, y}; return r; };
static inline __attribute__((device)) longlong3 make_longlong3(signed long long x, signed long long y, signed long long z) { longlong3 r{x, y, z}; return r; };
static inline __attribute__((device)) longlong4 make_longlong4(signed long long x, signed long long y, signed long long z, signed long long w) { longlong4 r{x, y, z, w}; return r; };
__attribute__((device)) inline static char __ldg(const char* ptr) { return *ptr; }
__attribute__((device)) inline static char2 __ldg(const char2* ptr) { return *ptr; }
__attribute__((device)) inline static char4 __ldg(const char4* ptr) { return *ptr; }
__attribute__((device)) inline static signed char __ldg(const signed char* ptr) { return ptr[0]; }
__attribute__((device)) inline static unsigned char __ldg(const unsigned char* ptr) { return ptr[0]; }
__attribute__((device)) inline static short __ldg(const short* ptr) { return ptr[0]; }
__attribute__((device)) inline static short2 __ldg(const short2* ptr) { return ptr[0]; }
__attribute__((device)) inline static short4 __ldg(const short4* ptr) { return ptr[0]; }
__attribute__((device)) inline static unsigned short __ldg(const unsigned short* ptr) { return ptr[0]; }
__attribute__((device)) inline static int __ldg(const int* ptr) { return ptr[0]; }
__attribute__((device)) inline static int2 __ldg(const int2* ptr) { return ptr[0]; }
__attribute__((device)) inline static int4 __ldg(const int4* ptr) { return ptr[0]; }
__attribute__((device)) inline static unsigned int __ldg(const unsigned int* ptr) { return ptr[0]; }
__attribute__((device)) inline static long __ldg(const long* ptr) { return ptr[0]; }
__attribute__((device)) inline static unsigned long __ldg(const unsigned long* ptr) { return ptr[0]; }
__attribute__((device)) inline static long long __ldg(const long long* ptr) { return ptr[0]; }
__attribute__((device)) inline static longlong2 __ldg(const longlong2* ptr) { return ptr[0]; }
__attribute__((device)) inline static unsigned long long __ldg(const unsigned long long* ptr) { return ptr[0]; }
__attribute__((device)) inline static uchar2 __ldg(const uchar2* ptr) { return ptr[0]; }
__attribute__((device)) inline static uchar4 __ldg(const uchar4* ptr) { return ptr[0]; }
__attribute__((device)) inline static ushort2 __ldg(const ushort2* ptr) { return ptr[0]; }
__attribute__((device)) inline static uint2 __ldg(const uint2* ptr) { return ptr[0]; }
__attribute__((device)) inline static uint4 __ldg(const uint4* ptr) { return ptr[0]; }
__attribute__((device)) inline static ulonglong2 __ldg(const ulonglong2* ptr) { return ptr[0]; }
__attribute__((device)) inline static float __ldg(const float* ptr) { return ptr[0]; }
__attribute__((device)) inline static float2 __ldg(const float2* ptr) { return ptr[0]; }
__attribute__((device)) inline static float4 __ldg(const float4* ptr) { return ptr[0]; }
__attribute__((device)) inline static double __ldg(const double* ptr) { return ptr[0]; }
__attribute__((device)) inline static double2 __ldg(const double2* ptr) { return ptr[0]; }
typedef struct dim3 {
  __hip_uint32_t x;
  __hip_uint32_t y;
  __hip_uint32_t z;
  constexpr __attribute__((device)) dim3(__hip_uint32_t _x = 1, __hip_uint32_t _y = 1, __hip_uint32_t _z = 1)
      : x(_x), y(_y), z(_z) {};
} dim3;
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_thread_idx_x() { return __builtin_amdgcn_workitem_id_x(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_thread_idx_y() { return __builtin_amdgcn_workitem_id_y(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_thread_idx_z() { return __builtin_amdgcn_workitem_id_z(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_block_idx_x() { return __builtin_amdgcn_workgroup_id_x(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_block_idx_y() { return __builtin_amdgcn_workgroup_id_y(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_block_idx_z() { return __builtin_amdgcn_workgroup_id_z(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_block_dim_x() { return __builtin_amdgcn_workgroup_size_x(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_block_dim_y() { return __builtin_amdgcn_workgroup_size_y(); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_block_dim_z() { return __builtin_amdgcn_workgroup_size_z(); }
extern "C" __attribute__((device)) __attribute__((const)) size_t __ockl_get_num_groups(unsigned int);
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_grid_dim_x() { return __ockl_get_num_groups(0); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_grid_dim_y() { return __ockl_get_num_groups(1); }
static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __hip_get_grid_dim_z() { return __ockl_get_num_groups(2); }
struct __hip_builtin_threadIdx_t {
  __declspec(property(get = __get_x)) unsigned int x; __attribute__((internal_linkage)) static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_x(void) { return __hip_get_thread_idx_x(); };
  __declspec(property(get = __get_y)) unsigned int y; __attribute__((internal_linkage)) static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_y(void) { return __hip_get_thread_idx_y(); };
  __declspec(property(get = __get_z)) unsigned int z; __attribute__((internal_linkage)) static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_z(void) { return __hip_get_thread_idx_z(); };
  __attribute__((device)) operator dim3() const { return dim3(x, y, z); }
};
struct __hip_builtin_blockIdx_t {
  __declspec(property(get = __get_x)) unsigned int x; static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_x(void) { return __hip_get_block_idx_x(); };
  __declspec(property(get = __get_y)) unsigned int y; static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_y(void) { return __hip_get_block_idx_y(); };
  __declspec(property(get = __get_z)) unsigned int z; static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_z(void) { return __hip_get_block_idx_z(); };
  __attribute__((device)) operator dim3() const { return dim3(x, y, z); }
};
struct __hip_builtin_blockDim_t {
  __declspec(property(get = __get_x)) unsigned int x; __attribute__((internal_linkage)) static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_x(void) { return __hip_get_block_dim_x(); };
  __declspec(property(get = __get_y)) unsigned int y; __attribute__((internal_linkage)) static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_y(void) { return __hip_get_block_dim_y(); };
  __declspec(property(get = __get_z)) unsigned int z; __attribute__((internal_linkage)) static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_z(void) { return __hip_get_block_dim_z(); };
  __attribute__((device)) operator dim3() const { return dim3(x, y, z); }
};
struct __hip_builtin_gridDim_t {
  __declspec(property(get = __get_x)) unsigned int x; static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_x(void) { return __hip_get_grid_dim_x(); };
  __declspec(property(get = __get_y)) unsigned int y; static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_y(void) { return __hip_get_grid_dim_y(); };
  __declspec(property(get = __get_z)) unsigned int z; static __attribute__((device)) inline __attribute__((always_inline)) unsigned int __get_z(void) { return __hip_get_grid_dim_z(); };
  __attribute__((device)) operator dim3() const { return dim3(x, y, z); }
};
extern const __attribute__((device)) __attribute__((weak)) __hip_builtin_threadIdx_t threadIdx;
extern const __attribute__((device)) __attribute__((weak)) __hip_builtin_blockIdx_t blockIdx;
extern const __attribute__((device)) __attribute__((weak)) __hip_builtin_blockDim_t blockDim;
extern const __attribute__((device)) __attribute__((weak)) __hip_builtin_gridDim_t gridDim;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
struct hip_bfloat16 {
  __hip_uint16_t data;
  enum truncate_t { truncate };
  __attribute__((device)) hip_bfloat16() = default;
  explicit __attribute__((device)) hip_bfloat16(float f) : data(float_to_bfloat16(f)) {}
  explicit __attribute__((device)) hip_bfloat16(float f, truncate_t)
      : data(truncate_float_to_bfloat16(f)) {}
  __attribute__((device)) operator float() const {
    union {
      __hip_uint32_t int32;
      float fp32;
    } u = {__hip_uint32_t(data) << 16};
    return u.fp32;
  }
  __attribute__((device)) hip_bfloat16& operator=(const float& f) {
    data = float_to_bfloat16(f);
    return *this;
  }
  static __attribute__((device)) hip_bfloat16 round_to_bfloat16(float f) {
    hip_bfloat16 output;
    output.data = float_to_bfloat16(f);
    return output;
  }
  static __attribute__((device)) hip_bfloat16 round_to_bfloat16(float f, truncate_t) {
    hip_bfloat16 output;
    output.data = truncate_float_to_bfloat16(f);
    return output;
  }
 private:
  static __attribute__((device)) __hip_uint16_t float_to_bfloat16(float f) {
    union {
      float fp32;
      __hip_uint32_t int32;
    } u = {f};
    if (~u.int32 & 0x7f800000) {
      u.int32 += 0x7fff + ((u.int32 >> 16) & 1);
    } else if (u.int32 & 0xffff) {
      u.int32 |= 0x10000;
    }
    return __hip_uint16_t(u.int32 >> 16);
  }
  static __attribute__((device)) __hip_uint16_t truncate_float_to_bfloat16(float f) {
    union {
      float fp32;
      __hip_uint32_t int32;
    } u = {f};
    return __hip_uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
  }
};
#pragma clang diagnostic pop
typedef struct {
  __hip_uint16_t data;
} hip_bfloat16_public;
static_assert(__hip_internal::is_standard_layout<hip_bfloat16>{},
              "hip_bfloat16 is not a standard layout type, and thus is "
              "incompatible with C.");
static_assert(__hip_internal::is_trivial<hip_bfloat16>{},
              "hip_bfloat16 is not a trivial type, and thus is "
              "incompatible with C.");
inline __attribute__((device)) hip_bfloat16 operator+(hip_bfloat16 a) { return a; }
inline __attribute__((device)) hip_bfloat16 operator-(hip_bfloat16 a) {
  a.data ^= 0x8000;
  return a;
}
inline __attribute__((device)) hip_bfloat16 operator+(hip_bfloat16 a, hip_bfloat16 b) {
  return hip_bfloat16(float(a) + float(b));
}
inline __attribute__((device)) hip_bfloat16 operator-(hip_bfloat16 a, hip_bfloat16 b) {
  return hip_bfloat16(float(a) - float(b));
}
inline __attribute__((device)) hip_bfloat16 operator*(hip_bfloat16 a, hip_bfloat16 b) {
  return hip_bfloat16(float(a) * float(b));
}
inline __attribute__((device)) hip_bfloat16 operator/(hip_bfloat16 a, hip_bfloat16 b) {
  return hip_bfloat16(float(a) / float(b));
}
inline __attribute__((device)) bool operator<(hip_bfloat16 a, hip_bfloat16 b) {
  return float(a) < float(b);
}
inline __attribute__((device)) bool operator==(hip_bfloat16 a, hip_bfloat16 b) {
  return float(a) == float(b);
}
inline __attribute__((device)) bool operator>(hip_bfloat16 a, hip_bfloat16 b) { return b < a; }
inline __attribute__((device)) bool operator<=(hip_bfloat16 a, hip_bfloat16 b) { return !(a > b); }
inline __attribute__((device)) bool operator!=(hip_bfloat16 a, hip_bfloat16 b) { return !(a == b); }
inline __attribute__((device)) bool operator>=(hip_bfloat16 a, hip_bfloat16 b) { return !(a < b); }
inline __attribute__((device)) hip_bfloat16& operator+=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a + b;
}
inline __attribute__((device)) hip_bfloat16& operator-=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a - b;
}
inline __attribute__((device)) hip_bfloat16& operator*=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a * b;
}
inline __attribute__((device)) hip_bfloat16& operator/=(hip_bfloat16& a, hip_bfloat16 b) {
  return a = a / b;
}
inline __attribute__((device)) hip_bfloat16& operator++(hip_bfloat16& a) { return a += hip_bfloat16(1.0f); }
inline __attribute__((device)) hip_bfloat16& operator--(hip_bfloat16& a) { return a -= hip_bfloat16(1.0f); }
inline __attribute__((device)) hip_bfloat16 operator++(hip_bfloat16& a, int) {
  hip_bfloat16 orig = a;
  ++a;
  return orig;
}
inline __attribute__((device)) hip_bfloat16 operator--(hip_bfloat16& a, int) {
  hip_bfloat16 orig = a;
  --a;
  return orig;
}
namespace std {
constexpr __attribute__((device)) bool isinf(hip_bfloat16 a) {
  return !(~a.data & 0x7f80) && !(a.data & 0x7f);
}
constexpr __attribute__((device)) bool isnan(hip_bfloat16 a) {
  return !(~a.data & 0x7f80) && +(a.data & 0x7f);
}
constexpr __attribute__((device)) bool iszero(hip_bfloat16 a) { return !(a.data & 0x7fff); }
}

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __constant__ __attribute__((constant))
#define __shared__ __attribute__((shared))
#define __align__(x) __attribute__((aligned(x)))
#if !defined(__has_feature) || !__has_feature(cuda_noinline_keyword)
#define __noinline__ __attribute__((noinline))
#endif
#define __forceinline__ inline __attribute__((always_inline))
#if __HIP_NO_IMAGE_SUPPORT
#define __hip_img_chk__ __attribute__((unavailable("The image/texture API not supported on the device")))
#else
#define __hip_img_chk__
#endif
#define launch_bounds_impl0(requiredMaxThreadsPerBlock)                                       \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock)))
#define launch_bounds_impl1(requiredMaxThreadsPerBlock, minBlocksPerMultiprocessor)           \
    __attribute__((amdgpu_flat_work_group_size(1, requiredMaxThreadsPerBlock),                \
                   amdgpu_waves_per_eu(minBlocksPerMultiprocessor)))
#define select_impl_(_1, _2, impl_, ...) impl_
#define __launch_bounds__(...)                                                                \
    select_impl_(__VA_ARGS__, launch_bounds_impl1, launch_bounds_impl0)(__VA_ARGS__)           
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define _HIP_BFLOAT16_H_
#define HIP_INCLUDE_HIP_MATH_FUNCTIONS_H
#define HIP_INCLUDE_HIP_HIP_VECTOR_TYPES_H
#if !__HIP_NO_STD_DEFS__
#if defined(__HIPRTC_PTRDIFF_T_IS_LONG_LONG__) && __HIPRTC_PTRDIFF_T_IS_LONG_LONG__==1
typedef long long ptrdiff_t;
#else
typedef __PTRDIFF_TYPE__ ptrdiff_t;
#endif
typedef long clock_t;
namespace std {
using ::ptrdiff_t;
using ::clock_t;
}
#endif // __HIP_NO_STD_DEFS__
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_HIP_COMMON_H
#define HIP_INCLUDE_HIP_HIP_COMMON_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#endif
// Common code included at start of every hip file.
// Auto enable __HIP_PLATFORM_AMD__ if compiling on AMD platform
// Other compiler (GCC,ICC,etc) need to set one of these macros explicitly
#if defined(__clang__) && defined(__HIP__)
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#endif  // defined(__clang__) && defined(__HIP__)

// Auto enable __HIP_PLATFORM_NVIDIA__ if compiling with NVIDIA platform
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__) && !defined(__HIP__))
#ifndef __HIP_PLATFORM_NVIDIA__
#define __HIP_PLATFORM_NVIDIA__
#endif

#ifdef __CUDACC__
#define __HIPCC__
#endif

#endif  //__NVCC__

// Auto enable __HIP_DEVICE_COMPILE__ if compiled in HCC or NVCC device path
#if (defined(__HCC_ACCELERATOR__) && __HCC_ACCELERATOR__ != 0) ||                                  \
    (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0)
#define __HIP_DEVICE_COMPILE__ 1
#endif

#ifdef __GNUC__
#define HIP_PUBLIC_API __attribute__((visibility("default")))
#define HIP_INTERNAL_EXPORTED_API __attribute__((visibility("default")))
#else
#define HIP_PUBLIC_API
#define HIP_INTERNAL_EXPORTED_API
#endif

#if __HIP_DEVICE_COMPILE__ == 0
// 32-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT32_ATOMICS__ (0)
#define __HIP_ARCH_HAS_GLOBAL_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_SHARED_INT32_ATOMICS__ (0)
#define __HIP_ARCH_HAS_SHARED_FLOAT_ATOMIC_EXCH__ (0)
#define __HIP_ARCH_HAS_FLOAT_ATOMIC_ADD__ (0)

// 64-bit Atomics
#define __HIP_ARCH_HAS_GLOBAL_INT64_ATOMICS__ (0)
#define __HIP_ARCH_HAS_SHARED_INT64_ATOMICS__ (0)

// Doubles
#define __HIP_ARCH_HAS_DOUBLES__ (0)

// Warp cross-lane operations
#define __HIP_ARCH_HAS_WARP_VOTE__ (0)
#define __HIP_ARCH_HAS_WARP_BALLOT__ (0)
#define __HIP_ARCH_HAS_WARP_SHUFFLE__ (0)
#define __HIP_ARCH_HAS_WARP_FUNNEL_SHIFT__ (0)

// Sync
#define __HIP_ARCH_HAS_THREAD_FENCE_SYSTEM__ (0)
#define __HIP_ARCH_HAS_SYNC_THREAD_EXT__ (0)

// Misc
#define __HIP_ARCH_HAS_SURFACE_FUNCS__ (0)
#define __HIP_ARCH_HAS_3DGRID__ (0)
#define __HIP_ARCH_HAS_DYNAMIC_PARALLEL__ (0)
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  amd_detail/device_library_decls.h
 *  @brief Contains declarations for types and functions in device library.
 *         Uses __hip_int64_t and __hip_uint64_t instead of long, long long, unsigned
 *         long and unsigned long long types for device library API
 *         declarations.
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_LIBRARY_DECLS_H

#if !defined(__HIPCC_RTC__)
#include "hip/amd_detail/host_defines.h"
#if __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#endif

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned long long ullong;

extern "C" __device__ __attribute__((const)) bool __ockl_wfany_i32(int);
extern "C" __device__ __attribute__((const)) bool __ockl_wfall_i32(int);
extern "C" __device__ uint __ockl_activelane_u32(void);

extern "C" __device__ __attribute__((const)) uint __ockl_mul24_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul24_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_mul_hi_u32(uint, uint);
extern "C" __device__ __attribute__((const)) int __ockl_mul_hi_i32(int, int);
extern "C" __device__ __attribute__((const)) uint __ockl_sadd_u32(uint, uint, uint);

extern "C" __device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
extern "C" __device__ __attribute__((const)) float __ocml_fmax_f32(float, float);

extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_f64(double);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_f64(double);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_f64(double);

extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
extern "C" __device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);

extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_s32(int);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_u32(__hip_uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_u32(__hip_uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_u32(__hip_uint32_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_s64(__hip_int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_s64(__hip_int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_s64(__hip_int64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtn_f32_u64(__hip_uint64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtp_f32_u64(__hip_uint64_t);
extern "C" __device__ __attribute__((const)) float __ocml_cvtrtz_f32_u64(__hip_uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtn_f64_s64(__hip_int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtp_f64_s64(__hip_int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtz_f64_s64(__hip_int64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtn_f64_u64(__hip_uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtp_f64_u64(__hip_uint64_t);
extern "C" __device__ __attribute__((const)) double __ocml_cvtrtz_f64_u64(__hip_uint64_t);

extern "C" __device__ __attribute__((convergent)) void __ockl_gws_barrier(uint nwm1, uint rid);

extern "C" __device__ __attribute__((const)) __hip_uint32_t __ockl_lane_u32();
extern "C" __device__ __attribute__((const)) int __ockl_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_grid_sync(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_num_grids(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_grid_rank(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_size(void);
extern "C" __device__ __attribute__((const)) uint __ockl_multi_grid_thread_rank(void);
extern "C" __device__ __attribute__((const)) int __ockl_multi_grid_is_valid(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_multi_grid_sync(void);
extern "C" __device__ __attribute__((const)) uint __ockl_grid_bar_arrive(void);
extern "C" __device__ __attribute__((convergent)) void __ockl_grid_bar_wait(uint);

extern "C" __device__ void __ockl_atomic_add_noret_f32(float*, float);

extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_add_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_and_i32(int a);
extern "C" __device__ __attribute__((convergent)) int __ockl_wgred_or_i32(int a);

extern "C" __device__ __hip_uint64_t __ockl_fprintf_stderr_begin();
extern "C" __device__ __hip_uint64_t __ockl_fprintf_append_args(
    __hip_uint64_t msg_desc, __hip_uint32_t num_args, __hip_uint64_t value0, __hip_uint64_t value1,
    __hip_uint64_t value2, __hip_uint64_t value3, __hip_uint64_t value4, __hip_uint64_t value5,
    __hip_uint64_t value6, __hip_uint32_t is_last);
extern "C" __device__ __hip_uint64_t __ockl_fprintf_append_string_n(__hip_uint64_t msg_desc,
                                                                    const char* data,
                                                                    __hip_uint64_t length,
                                                                    __hip_uint32_t is_last);

// Using hip.amdgcn.bc - sync threads
#define __CLK_LOCAL_MEM_FENCE  0x01
#define __CLK_GLOBAL_MEM_FENCE 0x02
typedef unsigned __cl_mem_fence_flags;

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(__clang__) && defined(__HIP__)

// abort
extern "C" __device__ inline __attribute__((weak)) void abort() { __builtin_trap(); }

// The noinline attribute helps encapsulate the printf expansion,
// which otherwise has a performance impact just by increasing the
// size of the calling function. Additionally, the weak attribute
// allows the function to exist as a global although its definition is
// included in every compilation unit.
#if defined(_WIN32) || defined(_WIN64)
extern "C" __device__ __attribute__((noinline)) __attribute__((weak)) void _wassert(
    const wchar_t* _msg, const wchar_t* _file, unsigned _line) {
  // FIXME: Need `wchar_t` support to generate assertion message.
  __builtin_trap();
}
#else /* defined(_WIN32) || defined(_WIN64) */
extern "C" __device__ __attribute__((noinline)) __attribute__((weak)) void __assert_fail(
    const char* assertion, const char* file, unsigned int line, const char* function) {
  static const char fmt[] = "%s:%u: %s: Device-side assertion `%s' failed.\n";

  // strlen is not available as a built-in yet, so we create our own
  // loop in a macro. With a string literal argument, the compiler
  // usually manages to replace the loop with a constant.
  //
  // The macro does not check for null pointer, since all the string
  // arguments are defined to be constant literals when called from
  // the assert() macro.
  //
  // NOTE: The loop below includes the null terminator in the length
  // as required by append_string_n().
#define __hip_get_string_length(LEN, STR)                                                          \
  do {                                                                                             \
    const char* tmp = STR;                                                                         \
    while (*tmp++);                                                                                \
    LEN = tmp - STR;                                                                               \
  } while (0)

  auto msg = __ockl_fprintf_stderr_begin();
  int len = 0;
  __hip_get_string_length(len, fmt);
  msg = __ockl_fprintf_append_string_n(msg, fmt, len, 0);
  __hip_get_string_length(len, file);
  msg = __ockl_fprintf_append_string_n(msg, file, len, 0);
  msg = __ockl_fprintf_append_args(msg, 1, line, 0, 0, 0, 0, 0, 0, 0);
  __hip_get_string_length(len, function);
  msg = __ockl_fprintf_append_string_n(msg, function, len, 0);
  __hip_get_string_length(len, assertion);
  __ockl_fprintf_append_string_n(msg, assertion, len, /* is_last = */ 1);

#undef __hip_get_string_length

  __builtin_trap();
}

extern "C" __device__ __attribute__((noinline)) __attribute__((weak)) void __assertfail() {
  // ignore all the args for now.
  __builtin_trap();
}
#endif /* defined(_WIN32) || defined(_WIN64) */

#if defined(NDEBUG)
#define __hip_assert(COND)
#else
#define __hip_assert(COND)                                                                         \
  do {                                                                                             \
    if (!(COND)) __builtin_trap();                                                                 \
  } while (0)
#endif

#endif  // defined(__clang__) and defined(__HIP__)
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_LIBRARY_TYPES_H
#define HIP_INCLUDE_HIP_LIBRARY_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

typedef enum hipDataType {
  HIP_R_32F = 0,
  HIP_R_64F = 1,
  HIP_R_16F = 2,
  HIP_R_8I = 3,
  HIP_C_32F = 4,
  HIP_C_64F = 5,
  HIP_C_16F = 6,
  HIP_C_8I = 7,
  HIP_R_8U = 8,
  HIP_C_8U = 9,
  HIP_R_32I = 10,
  HIP_C_32I = 11,
  HIP_R_32U = 12,
  HIP_C_32U = 13,
  HIP_R_16BF = 14,
  HIP_C_16BF = 15,
  HIP_R_4I = 16,
  HIP_C_4I = 17,
  HIP_R_4U = 18,
  HIP_C_4U = 19,
  HIP_R_16I = 20,
  HIP_C_16I = 21,
  HIP_R_16U = 22,
  HIP_C_16U = 23,
  HIP_R_64I = 24,
  HIP_C_64I = 25,
  HIP_R_64U = 26,
  HIP_C_64U = 27,
  HIP_R_8F_E4M3 = 28,
  HIP_R_8F_E5M2 = 29,
  HIP_R_8F_UE8M0 = 30,
  HIP_R_6F_E2M3 = 31,
  HIP_R_6F_E3M2 = 32,
  HIP_R_4F_E2M1 = 33,
  // HIP specific Data Types
  HIP_R_8F_E4M3_FNUZ = 1000,
  HIP_R_8F_E5M2_FNUZ = 1001,
} hipDataType;

typedef enum hipLibraryPropertyType {
  HIP_LIBRARY_MAJOR_VERSION,
  HIP_LIBRARY_MINOR_VERSION,
  HIP_LIBRARY_PATCH_LEVEL
} hipLibraryPropertyType;

#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "library_types.h"
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_DRIVER_TYPES_H
#define HIP_INCLUDE_HIP_DRIVER_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#if __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>  // size_t
#endif
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "driver_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

/**
 *  @defgroup DriverTypes Driver Types
 *  @{
 *  This section describes the driver data types.
 *
 */

typedef void* hipDeviceptr_t;
/**
 * HIP channel format kinds
 */
typedef enum hipChannelFormatKind {
  hipChannelFormatKindSigned = 0,    ///< Signed channel format
  hipChannelFormatKindUnsigned = 1,  ///< Unsigned channel format
  hipChannelFormatKindFloat = 2,     ///< Float channel format
  hipChannelFormatKindNone = 3       ///< No channel format
} hipChannelFormatKind;
/**
 * HIP channel format descriptor
 */
typedef struct hipChannelFormatDesc {
  int x;
  int y;
  int z;
  int w;
  enum hipChannelFormatKind f;  ///< Channel format kind
} hipChannelFormatDesc;
/** @brief The hipTexRefSetArray function flags parameter override format value*/
#define HIP_TRSA_OVERRIDE_FORMAT 0x01
/** @brief The hipTexRefSetFlags function flags parameter read as integer value*/
#define HIP_TRSF_READ_AS_INTEGER 0x01
/** @brief The hipTexRefSetFlags function flags parameter normalized coordinate value*/
#define HIP_TRSF_NORMALIZED_COORDINATES 0x02
/** @brief The hipTexRefSetFlags function flags parameter srgb value*/
#define HIP_TRSF_SRGB 0x10

typedef struct hipArray* hipArray_t;
typedef const struct hipArray* hipArray_const_t;
/**
 * HIP array format
 */
typedef enum hipArray_Format {
  HIP_AD_FORMAT_UNSIGNED_INT8 = 0x01,   ///< Unsigned 8-bit array format
  HIP_AD_FORMAT_UNSIGNED_INT16 = 0x02,  ///< Unsigned 16-bit array format
  HIP_AD_FORMAT_UNSIGNED_INT32 = 0x03,  ///< Unsigned 32-bit array format
  HIP_AD_FORMAT_SIGNED_INT8 = 0x08,     ///< Signed 8-bit array format
  HIP_AD_FORMAT_SIGNED_INT16 = 0x09,    ///< Signed 16-bit array format
  HIP_AD_FORMAT_SIGNED_INT32 = 0x0a,    ///< Signed 32-bit array format
  HIP_AD_FORMAT_HALF = 0x10,            ///< Half array format
  HIP_AD_FORMAT_FLOAT = 0x20            ///< Float array format
} hipArray_Format;
/**
 * HIP array descriptor
 */
typedef struct HIP_ARRAY_DESCRIPTOR {
  size_t Width;                 ///< Width of the array
  size_t Height;                ///< Height of the array
  enum hipArray_Format Format;  ///< Format of the array
  unsigned int NumChannels;     ///< Number of channels of the array
} HIP_ARRAY_DESCRIPTOR;

/**
 * HIP 3D array descriptor
 */
typedef struct HIP_ARRAY3D_DESCRIPTOR {
  size_t Width;                 ///< Width of the array
  size_t Height;                ///< Height of the array
  size_t Depth;                 ///< Depth of the array
  enum hipArray_Format Format;  ///< Format of the array
  unsigned int NumChannels;     ///< Number of channels of the array
  unsigned int Flags;           ///< Flags of the array
} HIP_ARRAY3D_DESCRIPTOR;
#if !defined(__HIPCC_RTC__)
/**
 * HIP 2D memory copy parameters
 */
typedef struct hip_Memcpy2D {
  size_t srcXInBytes;           ///< Source width in bytes
  size_t srcY;                  ///< Source height
  hipMemoryType srcMemoryType;  ///< Source memory type
  const void* srcHost;          ///< Source pointer
  hipDeviceptr_t srcDevice;     ///< Source device
  hipArray_t srcArray;          ///< Source array
  size_t srcPitch;              ///< Source pitch
  size_t dstXInBytes;           ///< Destination width in bytes
  size_t dstY;                  ///< Destination height
  hipMemoryType dstMemoryType;  ///< Destination memory type
  void* dstHost;                ///< Destination pointer
  hipDeviceptr_t dstDevice;     ///< Destination device
  hipArray_t dstArray;          ///< Destination array
  size_t dstPitch;              ///< Destination pitch
  size_t WidthInBytes;          ///< Width in bytes of the 2D memory copy
  size_t Height;                ///< Height of the 2D memory copy
} hip_Memcpy2D;
#endif  // !defined(__HIPCC_RTC__)
/**
 * HIP mipmapped array
 */
typedef struct hipMipmappedArray {
  void* data;                        ///< Data pointer of the mipmapped array
  struct hipChannelFormatDesc desc;  ///< Description of the mipmapped array
  unsigned int type;                 ///< Type of the mipmapped array
  unsigned int width;                ///< Width of the mipmapped array
  unsigned int height;               ///< Height of the mipmapped array
  unsigned int depth;                ///< Depth of the mipmapped array
  unsigned int min_mipmap_level;     ///< Minimum level of the mipmapped array
  unsigned int max_mipmap_level;     ///< Maximum level of the mipmapped array
  unsigned int flags;                ///< Flags of the mipmapped array
  enum hipArray_Format format;       ///< Format of the mipmapped array
  unsigned int num_channels;         ///< Number of channels of the mipmapped array
} hipMipmappedArray;
/**
 * HIP mipmapped array pointer
 */
typedef struct hipMipmappedArray* hipMipmappedArray_t;
typedef hipMipmappedArray_t hipmipmappedArray;
typedef const struct hipMipmappedArray* hipMipmappedArray_const_t;
/**
 * HIP resource types
 */
typedef enum hipResourceType {
  hipResourceTypeArray = 0x00,           ///< Array resource
  hipResourceTypeMipmappedArray = 0x01,  ///< Mipmapped array resource
  hipResourceTypeLinear = 0x02,          ///< Linear resource
  hipResourceTypePitch2D = 0x03          ///< Pitch 2D resource
} hipResourceType;
typedef enum HIPresourcetype_enum {
  HIP_RESOURCE_TYPE_ARRAY = 0x00,            ///< Array resource
  HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01,  ///< Mipmapped array resource
  HIP_RESOURCE_TYPE_LINEAR = 0x02,           ///< Linear resource
  HIP_RESOURCE_TYPE_PITCH2D = 0x03           ///< Pitch 2D resource
} HIPresourcetype,
    hipResourcetype;
/**
 * HIP texture address modes
 */
typedef enum HIPaddress_mode_enum {
  HIP_TR_ADDRESS_MODE_WRAP = 0,    ///< Wrap address mode
  HIP_TR_ADDRESS_MODE_CLAMP = 1,   ///< Clamp address mode
  HIP_TR_ADDRESS_MODE_MIRROR = 2,  ///< Mirror address mode
  HIP_TR_ADDRESS_MODE_BORDER = 3   ///< Border address mode
} HIPaddress_mode;
/**
 * HIP filter modes
 */
typedef enum HIPfilter_mode_enum {
  HIP_TR_FILTER_MODE_POINT = 0,  ///< Filter mode point
  HIP_TR_FILTER_MODE_LINEAR = 1  ///< Filter mode linear
} HIPfilter_mode;
/**
 * HIP texture descriptor
 */
typedef struct HIP_TEXTURE_DESC_st {
  HIPaddress_mode addressMode[3];   ///< Address modes
  HIPfilter_mode filterMode;        ///< Filter mode
  unsigned int flags;               ///< Flags
  unsigned int maxAnisotropy;       ///< Maximum anisotropy ratio
  HIPfilter_mode mipmapFilterMode;  ///< Mipmap filter mode
  float mipmapLevelBias;            ///< Mipmap level bias
  float minMipmapLevelClamp;        ///< Mipmap minimum level clamp
  float maxMipmapLevelClamp;        ///< Mipmap maximum level clamp
  float borderColor[4];             ///< Border Color
  int reserved[12];
} HIP_TEXTURE_DESC;
/**
 * HIP texture resource view formats
 */
typedef enum hipResourceViewFormat {
  hipResViewFormatNone = 0x00,  ///< No resource view format (use underlying resource format)
  hipResViewFormatUnsignedChar1 = 0x01,              ///< 1 channel, unsigned 8-bit integers
  hipResViewFormatUnsignedChar2 = 0x02,              ///< 2 channels, unsigned 8-bit integers
  hipResViewFormatUnsignedChar4 = 0x03,              ///< 4 channels, unsigned 8-bit integers
  hipResViewFormatSignedChar1 = 0x04,                ///< 1 channel, signed 8-bit integers
  hipResViewFormatSignedChar2 = 0x05,                ///< 2 channels, signed 8-bit integers
  hipResViewFormatSignedChar4 = 0x06,                ///< 4 channels, signed 8-bit integers
  hipResViewFormatUnsignedShort1 = 0x07,             ///< 1 channel, unsigned 16-bit integers
  hipResViewFormatUnsignedShort2 = 0x08,             ///< 2 channels, unsigned 16-bit integers
  hipResViewFormatUnsignedShort4 = 0x09,             ///< 4 channels, unsigned 16-bit integers
  hipResViewFormatSignedShort1 = 0x0a,               ///< 1 channel, signed 16-bit integers
  hipResViewFormatSignedShort2 = 0x0b,               ///< 2 channels, signed 16-bit integers
  hipResViewFormatSignedShort4 = 0x0c,               ///< 4 channels, signed 16-bit integers
  hipResViewFormatUnsignedInt1 = 0x0d,               ///< 1 channel, unsigned 32-bit integers
  hipResViewFormatUnsignedInt2 = 0x0e,               ///< 2 channels, unsigned 32-bit integers
  hipResViewFormatUnsignedInt4 = 0x0f,               ///< 4 channels, unsigned 32-bit integers
  hipResViewFormatSignedInt1 = 0x10,                 ///< 1 channel, signed 32-bit integers
  hipResViewFormatSignedInt2 = 0x11,                 ///< 2 channels, signed 32-bit integers
  hipResViewFormatSignedInt4 = 0x12,                 ///< 4 channels, signed 32-bit integers
  hipResViewFormatHalf1 = 0x13,                      ///< 1 channel, 16-bit floating point
  hipResViewFormatHalf2 = 0x14,                      ///< 2 channels, 16-bit floating point
  hipResViewFormatHalf4 = 0x15,                      ///< 4 channels, 16-bit floating point
  hipResViewFormatFloat1 = 0x16,                     ///< 1 channel, 32-bit floating point
  hipResViewFormatFloat2 = 0x17,                     ///< 2 channels, 32-bit floating point
  hipResViewFormatFloat4 = 0x18,                     ///< 4 channels, 32-bit floating point
  hipResViewFormatUnsignedBlockCompressed1 = 0x19,   ///< Block-compressed 1
  hipResViewFormatUnsignedBlockCompressed2 = 0x1a,   ///< Block-compressed 2
  hipResViewFormatUnsignedBlockCompressed3 = 0x1b,   ///< Block-compressed 3
  hipResViewFormatUnsignedBlockCompressed4 = 0x1c,   ///< Block-compressed 4 unsigned
  hipResViewFormatSignedBlockCompressed4 = 0x1d,     ///< Block-compressed 4 signed
  hipResViewFormatUnsignedBlockCompressed5 = 0x1e,   ///< Block-compressed 5 unsigned
  hipResViewFormatSignedBlockCompressed5 = 0x1f,     ///< Block-compressed 5 signed
  hipResViewFormatUnsignedBlockCompressed6H = 0x20,  ///< Block-compressed 6 unsigned half-float
  hipResViewFormatSignedBlockCompressed6H = 0x21,    ///< Block-compressed 6 signed half-float
  hipResViewFormatUnsignedBlockCompressed7 = 0x22    ///< Block-compressed 7
} hipResourceViewFormat;
/**
 * HIP texture resource view formats
 */
typedef enum HIPresourceViewFormat_enum {
  HIP_RES_VIEW_FORMAT_NONE = 0x00,  ///< No resource view format (use underlying resource format)
  HIP_RES_VIEW_FORMAT_UINT_1X8 = 0x01,       ///< 1 channel, unsigned 8-bit integers
  HIP_RES_VIEW_FORMAT_UINT_2X8 = 0x02,       ///< 2 channels, unsigned 8-bit integers
  HIP_RES_VIEW_FORMAT_UINT_4X8 = 0x03,       ///< 4 channels, unsigned 8-bit integers
  HIP_RES_VIEW_FORMAT_SINT_1X8 = 0x04,       ///< 1 channel, signed 8-bit integers
  HIP_RES_VIEW_FORMAT_SINT_2X8 = 0x05,       ///< 2 channels, signed 8-bit integers
  HIP_RES_VIEW_FORMAT_SINT_4X8 = 0x06,       ///< 4 channels, signed 8-bit integers
  HIP_RES_VIEW_FORMAT_UINT_1X16 = 0x07,      ///< 1 channel, unsigned 16-bit integers
  HIP_RES_VIEW_FORMAT_UINT_2X16 = 0x08,      ///< 2 channels, unsigned 16-bit integers
  HIP_RES_VIEW_FORMAT_UINT_4X16 = 0x09,      ///< 4 channels, unsigned 16-bit integers
  HIP_RES_VIEW_FORMAT_SINT_1X16 = 0x0a,      ///< 1 channel, signed 16-bit integers
  HIP_RES_VIEW_FORMAT_SINT_2X16 = 0x0b,      ///< 2 channels, signed 16-bit integers
  HIP_RES_VIEW_FORMAT_SINT_4X16 = 0x0c,      ///< 4 channels, signed 16-bit integers
  HIP_RES_VIEW_FORMAT_UINT_1X32 = 0x0d,      ///< 1 channel, unsigned 32-bit integers
  HIP_RES_VIEW_FORMAT_UINT_2X32 = 0x0e,      ///< 2 channels, unsigned 32-bit integers
  HIP_RES_VIEW_FORMAT_UINT_4X32 = 0x0f,      ///< 4 channels, unsigned 32-bit integers
  HIP_RES_VIEW_FORMAT_SINT_1X32 = 0x10,      ///< 1 channel, signed 32-bit integers
  HIP_RES_VIEW_FORMAT_SINT_2X32 = 0x11,      ///< 2 channels, signed 32-bit integers
  HIP_RES_VIEW_FORMAT_SINT_4X32 = 0x12,      ///< 4 channels, signed 32-bit integers
  HIP_RES_VIEW_FORMAT_FLOAT_1X16 = 0x13,     ///< 1 channel, 16-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_2X16 = 0x14,     ///< 2 channels, 16-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_4X16 = 0x15,     ///< 4 channels, 16-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_1X32 = 0x16,     ///< 1 channel, 32-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_2X32 = 0x17,     ///< 2 channels, 32-bit floating point
  HIP_RES_VIEW_FORMAT_FLOAT_4X32 = 0x18,     ///< 4 channels, 32-bit floating point
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC1 = 0x19,   ///< Block-compressed 1
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC2 = 0x1a,   ///< Block-compressed 2
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC3 = 0x1b,   ///< Block-compressed 3
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC4 = 0x1c,   ///< Block-compressed 4 unsigned
  HIP_RES_VIEW_FORMAT_SIGNED_BC4 = 0x1d,     ///< Block-compressed 4 signed
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC5 = 0x1e,   ///< Block-compressed 5 unsigned
  HIP_RES_VIEW_FORMAT_SIGNED_BC5 = 0x1f,     ///< Block-compressed 5 signed
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC6H = 0x20,  ///< Block-compressed 6 unsigned half-float
  HIP_RES_VIEW_FORMAT_SIGNED_BC6H = 0x21,    ///< Block-compressed 6 signed half-float
  HIP_RES_VIEW_FORMAT_UNSIGNED_BC7 = 0x22    ///< Block-compressed 7
} HIPresourceViewFormat;
/**
 * HIP resource descriptor
 */
typedef struct hipResourceDesc {
  enum hipResourceType resType;  ///< Resource type
  union {
    struct {
      hipArray_t array;  ///< HIP array
    } array;
    struct {
      hipMipmappedArray_t mipmap;  ///< HIP mipmapped array
    } mipmap;
    struct {
      void* devPtr;                      ///< Device pointer
      struct hipChannelFormatDesc desc;  ///< Channel format description
      size_t sizeInBytes;                ///< Size in bytes
    } linear;
    struct {
      void* devPtr;                      ///< Device pointer
      struct hipChannelFormatDesc desc;  ///< Channel format description
      size_t width;                      ///< Width of the array in elements
      size_t height;                     ///< Height of the array in elements
      size_t pitchInBytes;               ///< Pitch between two rows in bytes
    } pitch2D;
  } res;
} hipResourceDesc;

/**
 * HIP resource view descriptor struct
 */
typedef struct HIP_RESOURCE_DESC_st {
  HIPresourcetype resType;  ///< Resource type
  union {
    struct {
      hipArray_t hArray;  ///< HIP array
    } array;
    struct {
      hipMipmappedArray_t hMipmappedArray;  ///< HIP mipmapped array
    } mipmap;
    struct {
      hipDeviceptr_t devPtr;     ///< Device pointer
      hipArray_Format format;    ///< Array format
      unsigned int numChannels;  ///< Channels per array element
      size_t sizeInBytes;        ///< Size in bytes
    } linear;
    struct {
      hipDeviceptr_t devPtr;     ///< Device pointer
      hipArray_Format format;    ///< Array format
      unsigned int numChannels;  ///< Channels per array element
      size_t width;              ///< Width of the array in elements
      size_t height;             ///< Height of the array in elements
      size_t pitchInBytes;       ///< Pitch between two rows in bytes
    } pitch2D;
    struct {
      int reserved[32];
    } reserved;
  } res;
  unsigned int flags;  ///< Flags (must be zero)
} HIP_RESOURCE_DESC;
/**
 * HIP resource view descriptor
 */
struct hipResourceViewDesc {
  enum hipResourceViewFormat format;  ///< Resource view format
  size_t width;                       ///< Width of the resource view
  size_t height;                      ///< Height of the resource view
  size_t depth;                       ///< Depth of the resource view
  unsigned int firstMipmapLevel;      ///< First defined mipmap level
  unsigned int lastMipmapLevel;       ///< Last defined mipmap level
  unsigned int firstLayer;            ///< First layer index
  unsigned int lastLayer;             ///< Last layer index
};
/**
 * Resource view descriptor
 */
typedef struct HIP_RESOURCE_VIEW_DESC_st {
  HIPresourceViewFormat format;   ///< Resource view format
  size_t width;                   ///< Width of the resource view
  size_t height;                  ///< Height of the resource view
  size_t depth;                   ///< Depth of the resource view
  unsigned int firstMipmapLevel;  ///< First defined mipmap level
  unsigned int lastMipmapLevel;   ///< Last defined mipmap level
  unsigned int firstLayer;        ///< First layer index
  unsigned int lastLayer;         ///< Last layer index
  unsigned int reserved[16];
} HIP_RESOURCE_VIEW_DESC;
/**
 * Memory copy types
 */
#if !defined(__HIPCC_RTC__)
typedef enum hipMemcpyKind {
  hipMemcpyHostToHost = 0,            ///< Host-to-Host Copy
  hipMemcpyHostToDevice = 1,          ///< Host-to-Device Copy
  hipMemcpyDeviceToHost = 2,          ///< Device-to-Host Copy
  hipMemcpyDeviceToDevice = 3,        ///< Device-to-Device Copy
  hipMemcpyDefault = 4,               ///< Runtime will automatically determine
                                      ///< copy-kind based on virtual addresses.
  hipMemcpyDeviceToDeviceNoCU = 1024  ///< Device-to-Device Copy without using compute units
} hipMemcpyKind;
/**
 * HIP pithed pointer
 */
typedef struct hipPitchedPtr {
  void* ptr;     ///< Pointer to the allocated memory
  size_t pitch;  ///< Pitch in bytes
  size_t xsize;  ///< Logical size of the first dimension of allocation in elements
  size_t ysize;  ///< Logical size of the second dimension of allocation in elements
} hipPitchedPtr;
/**
 * HIP extent
 */
typedef struct hipExtent {
  size_t width;  // Width in elements when referring to array memory, in bytes when referring to
                 // linear memory
  size_t height;
  size_t depth;
} hipExtent;
/**
 *  HIP position
 */
typedef struct hipPos {
  size_t x;  ///< X coordinate
  size_t y;  ///< Y coordinate
  size_t z;  ///< Z coordinate
} hipPos;
/**
 * HIP 3D memory copy parameters
 */
typedef struct hipMemcpy3DParms {
  hipArray_t srcArray;          ///< Source array
  struct hipPos srcPos;         ///< Source position
  struct hipPitchedPtr srcPtr;  ///< Source pointer
  hipArray_t dstArray;          ///< Destination array
  struct hipPos dstPos;         ///< Destination position
  struct hipPitchedPtr dstPtr;  ///< Destination pointer
  struct hipExtent extent;      ///< Extent of 3D memory copy
  enum hipMemcpyKind kind;      ///< Kind of 3D memory copy
} hipMemcpy3DParms;
/**
 * HIP 3D memory copy
 */
typedef struct HIP_MEMCPY3D {
  size_t srcXInBytes;           ///< Source X in bytes
  size_t srcY;                  ///< Source Y
  size_t srcZ;                  ///< Source Z
  size_t srcLOD;                ///< Source LOD
  hipMemoryType srcMemoryType;  ///< Source memory type
  const void* srcHost;          ///< Source host pointer
  hipDeviceptr_t srcDevice;     ///< Source device
  hipArray_t srcArray;          ///< Source array
  size_t srcPitch;              ///< Source pitch
  size_t srcHeight;             ///< Source height
  size_t dstXInBytes;           ///< Destination X in bytes
  size_t dstY;                  ///< Destination Y
  size_t dstZ;                  ///< Destination Z
  size_t dstLOD;                ///< Destination LOD
  hipMemoryType dstMemoryType;  ///< Destination memory type
  void* dstHost;                ///< Destination host pointer
  hipDeviceptr_t dstDevice;     ///< Destination device
  hipArray_t dstArray;          ///< Destination array
  size_t dstPitch;              ///< Destination pitch
  size_t dstHeight;             ///< Destination height
  size_t WidthInBytes;          ///< Width in bytes of 3D memory copy
  size_t Height;                ///< Height in bytes of 3D memory copy
  size_t Depth;                 ///< Depth in bytes of 3D memory copy
} HIP_MEMCPY3D;
/**
 * Specifies the type of location
 */
typedef enum hipMemLocationType {
  hipMemLocationTypeInvalid = 0,
  hipMemLocationTypeNone = 0,
  hipMemLocationTypeDevice = 1,    ///< Device location, thus it's HIP device ID
  hipMemLocationTypeHost = 2,      ///< Host location, id is ignored
  hipMemLocationTypeHostNuma = 3,  ///< Host NUMA node location, id is host NUMA node id
  hipMemLocationTypeHostNumaCurrent =
      4  ///< Host NUMA node closest to current thread’s CPU, id is ignored
} hipMemLocationType;
/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = @p hipMemLocationTypeDevice and set id = the gpu's device ID
 */
typedef struct hipMemLocation {
  hipMemLocationType type;  ///< Specifies the location type, which describes the meaning of id
  int id;                   ///< Identifier for the provided location type @p hipMemLocationType
} hipMemLocation;

/**
 * Flags to specify for copies within a batch. Used with hipMemcpyBatchAsync
 */
typedef enum hipMemcpyFlags {
  hipMemcpyFlagDefault = 0x0,                  ///< Default flag
  hipMemcpyFlagPreferOverlapWithCompute = 0x1, ///< Tries to overlap copy with compute work.
  hipMemcpyFlagExtPreferCE = 0x100,            ///< Prefer copy engine over compute engine.
  hipMemcpyFlagExtOpSwap = 0x200,              ///< Swap contents of src and dst.
  hipMemcpyFlagExtOpIndirectSrc = 0x400,       ///< The src pointer holds the address of the real
                                               ///< source pointer, read when the copy runs rather
                                               ///< than when it is submitted.
  hipMemcpyFlagExtOpIndirectDst = 0x800        ///< Same as IndirectSrc, but for the dst pointer.
} hipMemcpyFlags;

/**
 * Flags to specify order in which source pointer is accessed by Batch memcpy
 */
typedef enum hipMemcpySrcAccessOrder {
  hipMemcpySrcAccessOrderInvalid = 0x0,  ///< Default Invalid.
  hipMemcpySrcAccessOrderStream = 0x1,   ///< Access to source pointer must be in stream order.
  hipMemcpySrcAccessOrderDuringApiCall =
      0x2,  ///< Access to source pointer can be out of stream order and all accesses must be
            ///< complete before API call returns.
  hipMemcpySrcAccessOrderAny =
      0x3,  ///< Access to the source pointer can be out of stream order and the accesses can happen
            ///< even after the API call return.
  hipMemcpySrcAccessOrderMax = 0x7FFFFFFF
} hipMemcpySrcAccessOrder;

/**
 * Attributes for copies within a batch.
 */
typedef struct hipMemcpyAttributes {
  hipMemcpySrcAccessOrder
      srcAccessOrder;  ///< Source access ordering to be observed for copies with this attribute.
  hipMemLocation srcLocHint;  ///< Location hint for src operand.
  hipMemLocation dstLocHint;  ///< Location hint for destination operand.
  unsigned int flags;         ///< Additional Flags for copies. See hipMemcpyFlags.
} hipMemcpyAttributes;
/**
 * Operand types for individual copies within a batch
 */
typedef enum hipMemcpy3DOperandType {
  hipMemcpyOperandTypePointer = 0x1,  ///< Mempcy operand is a valid pointer.
  hipMemcpyOperandTypeArray = 0x2,    ///< Memcpy operand is a valid hipArray.
  hipMemcpyOperandTypeMax = 0x7FFFFFFF
} hipMemcpy3DOperandType;

/**
 * Struct representing offset into a hipArray_t in elements.
 */
typedef struct hipOffset3D {
  size_t x;
  size_t y;
  size_t z;
} hipOffset3D;
/**
 *  Struct representing an operand for copy with hipMemcpy3DBatchAsync.
 */
typedef struct hipMemcpy3DOperand {
  hipMemcpy3DOperandType type;
  union {
    struct {
      void* ptr;
      size_t rowLength;        ///< Length of each row in elements.
      size_t layerHeight;      ///< Height of each layer in elements.
      hipMemLocation locHint;  ///< Location Hint for the operand.
    } ptr;
    struct {
      hipArray_t array;    ///< Array struct for hipMemcpyOperandTypeArray.
      hipOffset3D offset;  ///< Offset into array in elements.
    } array;
  } op;
} hipMemcpy3DOperand;

/**
 * HIP 3D Batch Op
 */
typedef struct hipMemcpy3DBatchOp {
  hipMemcpy3DOperand src;
  hipMemcpy3DOperand dst;
  hipExtent extent;
  hipMemcpySrcAccessOrder srcAccessOrder;
  unsigned int flags;
} hipMemcpy3DBatchOp;

typedef struct hipMemcpy3DPeerParms {
  hipArray_t srcArray;   ///< Source memory address
  hipPos srcPos;         ///< Source position offset
  hipPitchedPtr srcPtr;  ///< Pitched source memory address
  int srcDevice;         ///< Source device
  hipArray_t dstArray;   ///< Destination memory address
  hipPos dstPos;         ///< Destination position offset
  hipPitchedPtr dstPtr;  ///< Pitched destination memory address
  int dstDevice;         ///< Destination device
  hipExtent extent;      ///< Requested memory copy size
} hipMemcpy3DPeerParms;

/**
 * @brief Make hipPitchedPtr
 *
 * @param [in] d Pointer to the allocated memory
 * @param [in] p Pitch in bytes
 * @param [in] xsz Logical size of the first dimension of allocation in elements
 * @param [in] ysz Logical size of the second dimension of allocation in elements
 *
 * @returns The created hipPitchedPtr
 */
static inline struct hipPitchedPtr make_hipPitchedPtr(void* d, size_t p, size_t xsz, size_t ysz) {
  struct hipPitchedPtr s;
  s.ptr = d;
  s.pitch = p;
  s.xsize = xsz;
  s.ysize = ysz;
  return s;
}
/**
 * @brief Make hipPos struct
 *
 * @param [in] x X coordinate of the new hipPos
 * @param [in] y Y coordinate of the new hipPos
 * @param [in] z Z coordinate of the new hipPos
 *
 * @returns The created hipPos struct
 */
static inline struct hipPos make_hipPos(size_t x, size_t y, size_t z) {
  struct hipPos p;
  p.x = x;
  p.y = y;
  p.z = z;
  return p;
}
/**
 * @brief Make hipExtent struct
 *
 * @param [in] w Width of the new hipExtent
 * @param [in] h Height of the new hipExtent
 * @param [in] d Depth of the new hipExtent
 *
 * @returns The created hipExtent struct
 */
static inline struct hipExtent make_hipExtent(size_t w, size_t h, size_t d) {
  struct hipExtent e;
  e.width = w;
  e.height = h;
  e.depth = d;
  return e;
}

typedef enum hipFunction_attribute {
  HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,                 ///< The maximum number of threads per block. Depends
                                                            ///< on function and device.
  HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,                     ///< The statically allocated shared memory size in bytes
                                                            ///< per block required by the function.
  HIP_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,                      ///< The user-allocated constant memory by the function in
                                                            ///< bytes.
  HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,                      ///< The local memory usage of each thread by this function
                                                            ///< in bytes.
  HIP_FUNC_ATTRIBUTE_NUM_REGS,                              ///< The number of registers used by each thread of this function.
  HIP_FUNC_ATTRIBUTE_PTX_VERSION,                           ///< PTX version
  HIP_FUNC_ATTRIBUTE_BINARY_VERSION,                        ///< Binary version
  HIP_FUNC_ATTRIBUTE_CACHE_MODE_CA,                         ///< Cache mode
  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,         ///< The maximum dynamic shared memory per
                                                            ///< block for this function in bytes.
  HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,      ///< The shared memory carveout preference
                                                            ///< in percent of the maximum shared
                                                            ///< memory.
  HIP_FUNC_ATTRIBUTE_CLUSTER_DIM_MUST_BE_SET,               ///< the kernel must launch with a valid cluster size specified.
  HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,                ///< The required cluster width in blocks
  HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,               ///< The required cluster height in blocks
  HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,                ///< The required cluster depth in blocks
  HIP_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,     ///< Is the function allowed to launch with non-portable cluster size.
  HIP_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,  ///< The block scheduling policy of a function.
  HIP_FUNC_ATTRIBUTE_MAX
} hipFunction_attribute;

typedef enum hipPointer_attribute {
  HIP_POINTER_ATTRIBUTE_CONTEXT = 1,     ///< The context on which a pointer was allocated
                                         ///< @warning This attribute is not supported in HIP
  HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,     ///< memory type describing the location of a pointer
  HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,  ///< address at which the pointer is allocated on the
                                         ///< device
  HIP_POINTER_ATTRIBUTE_HOST_POINTER,    ///< address at which the pointer is allocated on the host
  HIP_POINTER_ATTRIBUTE_P2P_TOKENS,      ///< A pair of tokens for use with Linux kernel interface
                                         ///< @warning This attribute is not supported in HIP
  HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,     ///< Synchronize every synchronous memory operation
                                         ///< initiated on this region
  HIP_POINTER_ATTRIBUTE_BUFFER_ID,       ///< Unique ID for an allocated memory region
  HIP_POINTER_ATTRIBUTE_IS_MANAGED,      ///< Indicates if the pointer points to managed memory
  HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,  ///< device ordinal of a device on which a pointer
                                         ///< was allocated or registered
  HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE,  ///< if this pointer maps to an allocation
                                                    ///< that is suitable for hipIpcGetMemHandle
                                                    ///< @warning This attribute is not supported in
                                                    ///< HIP
  HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,           ///< Starting address for this requested pointer
  HIP_POINTER_ATTRIBUTE_RANGE_SIZE,  ///< Size of the address range for this requested pointer
  HIP_POINTER_ATTRIBUTE_MAPPED,      ///< tells if this pointer is in a valid address range
                                     ///< that is mapped to a backing allocation
  HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,  ///< Bitmask of allowed hipmemAllocationHandleType
                                               ///< for this allocation @warning This attribute is
                                               ///< not supported in HIP
  HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,  ///< returns if the memory referenced by
                                                     ///< this pointer can be used with the
                                                     ///< GPUDirect RDMA API
                                                     ///< @warning This attribute is not supported
                                                     ///< in HIP
  HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,   ///< Returns the access flags the device associated with
                                        ///< for the corresponding memory referenced by the ptr
  HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE  ///< Returns the mempool handle for the allocation if
                                        ///< it was allocated from a mempool
                                        ///< @warning This attribute is not supported in HIP
} hipPointer_attribute;

// doxygen end DriverTypes
/**
 * @}
 */

#endif  // !defined(__HIPCC_RTC__)
#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif
#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  surface_types.h
 *  @brief Defines surface types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_SURFACE_TYPES_H
#define HIP_INCLUDE_HIP_SURFACE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/driver_types.h>
#endif

/**
 * An opaque value that represents a hip surface object
 */
struct __hip_surface;
typedef struct __hip_surface* hipSurfaceObject_t;

/**
 * hip surface reference
 */
struct surfaceReference {
  hipSurfaceObject_t surfaceObject;
};

/**
 * hip surface boundary modes
 */
enum hipSurfaceBoundaryMode {
  hipBoundaryModeZero = 0,
  hipBoundaryModeTrap = 1,
  hipBoundaryModeClamp = 2
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* !HIP_INCLUDE_HIP_SURFACE_TYPES_H */
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#include <hip/driver_types.h>
#include <hip/amd_detail/amd_hip_vector_types.h>
#endif

#ifdef __cplusplus

extern "C" HIP_PUBLIC_API hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                                    hipChannelFormatKind f);

static inline hipChannelFormatDesc hipCreateChannelDescHalf() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf1() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf2() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindFloat);
}

static inline hipChannelFormatDesc hipCreateChannelDescHalf4() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindFloat);
}

template <typename T> static inline hipChannelFormatDesc hipCreateChannelDesc() {
  return hipCreateChannelDesc(0, 0, 0, 0, hipChannelFormatKindNone);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<char>() {
  int e = (int)sizeof(char) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<signed char>() {
  int e = (int)sizeof(signed char) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<unsigned char>() {
  int e = (int)sizeof(unsigned char) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<uchar1>() {
  int e = (int)sizeof(unsigned char) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<char1>() {
  int e = (int)sizeof(signed char) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<uchar2>() {
  int e = (int)sizeof(unsigned char) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<char2>() {
  int e = (int)sizeof(signed char) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__  // vector3 is the same as vector4
template <> inline hipChannelFormatDesc hipCreateChannelDesc<uchar3>() {
  int e = (int)sizeof(unsigned char) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<char3>() {
  int e = (int)sizeof(signed char) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <> inline hipChannelFormatDesc hipCreateChannelDesc<uchar4>() {
  int e = (int)sizeof(unsigned char) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<char4>() {
  int e = (int)sizeof(signed char) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<unsigned short>() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<signed short>() {
  int e = (int)sizeof(signed short) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<ushort1>() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<short1>() {
  int e = (int)sizeof(signed short) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<ushort2>() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<short2>() {
  int e = (int)sizeof(signed short) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <> inline hipChannelFormatDesc hipCreateChannelDesc<ushort3>() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<short3>() {
  int e = (int)sizeof(signed short) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <> inline hipChannelFormatDesc hipCreateChannelDesc<ushort4>() {
  int e = (int)sizeof(unsigned short) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<short4>() {
  int e = (int)sizeof(signed short) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<unsigned int>() {
  int e = (int)sizeof(unsigned int) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<signed int>() {
  int e = (int)sizeof(signed int) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<uint1>() {
  int e = (int)sizeof(unsigned int) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<int1>() {
  int e = (int)sizeof(signed int) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<uint2>() {
  int e = (int)sizeof(unsigned int) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<int2>() {
  int e = (int)sizeof(signed int) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <> inline hipChannelFormatDesc hipCreateChannelDesc<uint3>() {
  int e = (int)sizeof(unsigned int) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<int3>() {
  int e = (int)sizeof(signed int) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <> inline hipChannelFormatDesc hipCreateChannelDesc<uint4>() {
  int e = (int)sizeof(unsigned int) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<int4>() {
  int e = (int)sizeof(signed int) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<float>() {
  int e = (int)sizeof(float) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<float1>() {
  int e = (int)sizeof(float) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindFloat);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<float2>() {
  int e = (int)sizeof(float) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindFloat);
}

#ifndef __GNUC__
template <> inline hipChannelFormatDesc hipCreateChannelDesc<float3>() {
  int e = (int)sizeof(float) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindFloat);
}
#endif

template <> inline hipChannelFormatDesc hipCreateChannelDesc<float4>() {
  int e = (int)sizeof(float) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindFloat);
}

#if !defined(__LP64__)

template <> inline hipChannelFormatDesc hipCreateChannelDesc<unsigned long>() {
  int e = (int)sizeof(unsigned long) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<signed long>() {
  int e = (int)sizeof(signed long) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<ulong1>() {
  int e = (int)sizeof(unsigned long) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<long1>() {
  int e = (int)sizeof(signed long) * 8;
  return hipCreateChannelDesc(e, 0, 0, 0, hipChannelFormatKindSigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<ulong2>() {
  int e = (int)sizeof(unsigned long) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<long2>() {
  int e = (int)sizeof(signed long) * 8;
  return hipCreateChannelDesc(e, e, 0, 0, hipChannelFormatKindSigned);
}

#ifndef __GNUC__
template <> inline hipChannelFormatDesc hipCreateChannelDesc<ulong3>() {
  int e = (int)sizeof(unsigned long) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<long3>() {
  int e = (int)sizeof(signed long) * 8;
  return hipCreateChannelDesc(e, e, e, 0, hipChannelFormatKindSigned);
}
#endif

template <> inline hipChannelFormatDesc hipCreateChannelDesc<ulong4>() {
  int e = (int)sizeof(unsigned long) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindUnsigned);
}

template <> inline hipChannelFormatDesc hipCreateChannelDesc<long4>() {
  int e = (int)sizeof(signed long) * 8;
  return hipCreateChannelDesc(e, e, e, e, hipChannelFormatKindSigned);
}
#endif /* !__LP64__ */

#else

struct hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                 enum hipChannelFormatKind f);

#endif /* __cplusplus */

#endif /* !HIP_INCLUDE_HIP_AMD_DETAIL_CHANNEL_DESCRIPTOR_H */
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_TEXTURE_TYPES_H
#define HIP_INCLUDE_HIP_TEXTURE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#pragma clang diagnostic ignored "-Wc++98-compat"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "texture_types.h"
#elif defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if !defined(__HIPCC_RTC__)
#include <hip/channel_descriptor.h>
#include <hip/driver_types.h>
#endif  // !defined(__HIPCC_RTC__)

#define hipTextureType1D 0x01
#define hipTextureType2D 0x02
#define hipTextureType3D 0x03
#define hipTextureTypeCubemap 0x0C
#define hipTextureType1DLayered 0xF1
#define hipTextureType2DLayered 0xF2
#define hipTextureTypeCubemapLayered 0xFC

/**
 * Should be same as HSA_IMAGE_OBJECT_SIZE_DWORD/HSA_SAMPLER_OBJECT_SIZE_DWORD
 */
#define HIP_IMAGE_OBJECT_SIZE_DWORD 12
#define HIP_SAMPLER_OBJECT_SIZE_DWORD 8
#define HIP_SAMPLER_OBJECT_OFFSET_DWORD HIP_IMAGE_OBJECT_SIZE_DWORD
#define HIP_TEXTURE_OBJECT_SIZE_DWORD (HIP_IMAGE_OBJECT_SIZE_DWORD + HIP_SAMPLER_OBJECT_SIZE_DWORD)

/**
 * An opaque value that represents a hip texture object
 */
struct __hip_texture;
typedef struct __hip_texture* hipTextureObject_t;

/**
 * hip texture address modes
 */
enum hipTextureAddressMode {
  hipAddressModeWrap = 0,
  hipAddressModeClamp = 1,
  hipAddressModeMirror = 2,
  hipAddressModeBorder = 3
};

/**
 * hip texture filter modes
 */
enum hipTextureFilterMode { hipFilterModePoint = 0, hipFilterModeLinear = 1 };

/**
 * hip texture read modes
 */
enum hipTextureReadMode { hipReadModeElementType = 0, hipReadModeNormalizedFloat = 1 };

/**
 * hip texture reference
 */
typedef struct textureReference {
  int normalized;
  enum hipTextureReadMode readMode;  // used only for driver API's
  enum hipTextureFilterMode filterMode;
  enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
  struct hipChannelFormatDesc channelDesc;
  int sRGB;                    // Perform sRGB->linear conversion during texture read
  unsigned int maxAnisotropy;  // Limit to the anisotropy ratio
  enum hipTextureFilterMode mipmapFilterMode;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;

  hipTextureObject_t textureObject;
  int numChannels;
  enum hipArray_Format format;
} textureReference;

/**
 * hip texture descriptor
 */
typedef struct hipTextureDesc {
  enum hipTextureAddressMode addressMode[3];  // Texture address mode for up to 3 dimensions
  enum hipTextureFilterMode filterMode;
  enum hipTextureReadMode readMode;
  int sRGB;  // Perform sRGB->linear conversion during texture read
  float borderColor[4];
  int normalizedCoords;
  unsigned int maxAnisotropy;
  enum hipTextureFilterMode mipmapFilterMode;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
} hipTextureDesc;

#if __cplusplus

/*******************************************************************************
 *                                                                              *
 *                                                                              *
 *                                                                              *
 *******************************************************************************/
#if __HIP__
#define __HIP_TEXTURE_ATTRIB __attribute__((device_builtin_texture_type))
#else
#define __HIP_TEXTURE_ATTRIB
#endif

typedef textureReference* hipTexRef;

template <class T, int texType = hipTextureType1D,
          enum hipTextureReadMode mode = hipReadModeElementType>
struct __HIP_TEXTURE_ATTRIB texture : public textureReference {
  texture(int norm = 0, enum hipTextureFilterMode fMode = hipFilterModePoint,
          enum hipTextureAddressMode aMode = hipAddressModeClamp) {
    normalized = norm;
    readMode = mode;
    filterMode = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc = hipCreateChannelDesc<T>();
    sRGB = 0;
    textureObject = nullptr;
    maxAnisotropy = 0;
    mipmapLevelBias = 0;
    minMipmapLevelClamp = 0;
    maxMipmapLevelClamp = 0;
  }

  texture(int norm, enum hipTextureFilterMode fMode, enum hipTextureAddressMode aMode,
          struct hipChannelFormatDesc desc) {
    normalized = norm;
    readMode = mode;
    filterMode = fMode;
    addressMode[0] = aMode;
    addressMode[1] = aMode;
    addressMode[2] = aMode;
    channelDesc = desc;
    sRGB = 0;
    textureObject = nullptr;
    maxAnisotropy = 0;
    mipmapLevelBias = 0;
    minMipmapLevelClamp = 0;
    maxMipmapLevelClamp = 0;
  }
};

#endif /* __cplusplus */

#else
#error ("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#endif

extern "C" {

#define ADDRESS_SPACE_CONSTANT __attribute__((address_space(4)))

__device__ float4::Native_vec_ __ockl_image_load_1D(unsigned int ADDRESS_SPACE_CONSTANT* i, int c);

__device__ float4::Native_vec_ __ockl_image_load_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i, int c);

__device__ float4::Native_vec_ __ockl_image_load_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                     int2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                    int2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                     int4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_3D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                    int4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_load_CM(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                    int2::Native_vec_ c, int f);

__device__ float4::Native_vec_ __ockl_image_load_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                     int4::Native_vec_ c, int f);

__device__ float4::Native_vec_ __ockl_image_load_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        int c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                         int2::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        int2::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                         int4::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        int4::Native_vec_ c, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        int2::Native_vec_ c, int f, int l);

__device__ float4::Native_vec_ __ockl_image_load_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                         int4::Native_vec_ c, int f, int l);

__device__ void __ockl_image_store_1D(unsigned int ADDRESS_SPACE_CONSTANT* i, int c,
                                      float4::Native_vec_ p);

__device__ void __ockl_image_store_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i, int2::Native_vec_ c,
                                       float4::Native_vec_ p);

__device__ void __ockl_image_store_2D(unsigned int ADDRESS_SPACE_CONSTANT* i, int2::Native_vec_ c,
                                      float4::Native_vec_ p);

__device__ void __ockl_image_store_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i, int4::Native_vec_ c,
                                       float4::Native_vec_ p);

__device__ void __ockl_image_store_3D(unsigned int ADDRESS_SPACE_CONSTANT* i, int4::Native_vec_ c,
                                      float4::Native_vec_ p);

__device__ void __ockl_image_store_CM(unsigned int ADDRESS_SPACE_CONSTANT* i, int2::Native_vec_ c,
                                      int f, float4::Native_vec_ p);

__device__ void __ockl_image_store_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i, int4::Native_vec_ c,
                                       int f, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT* i, int c, int l,
                                          float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                           int2::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                          int2::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                           int4::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                          int4::Native_vec_ c, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                          int2::Native_vec_ c, int f, int l, float4::Native_vec_ p);

__device__ void __ockl_image_store_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                           int4::Native_vec_ c, int f, int l,
                                           float4::Native_vec_ p);

__device__ float4::Native_vec_ __ockl_image_sample_1D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                      unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                      float c);

__device__ float4::Native_vec_ __ockl_image_sample_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                       unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                       float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                      unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                      float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                       unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                       float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_3D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                      unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                      float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_CM(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                      unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                      float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                       unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                       float4::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_sample_grad_1D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                           unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                           float c, float dx, float dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                            unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                            float2::Native_vec_ c, float dx,
                                                            float dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                           unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                           float2::Native_vec_ c,
                                                           float2::Native_vec_ dx,
                                                           float2::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                            unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                            float4::Native_vec_ c,
                                                            float2::Native_vec_ dx,
                                                            float2::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_grad_3D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                           unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                           float4::Native_vec_ c,
                                                           float4::Native_vec_ dx,
                                                           float4::Native_vec_ dy);

__device__ float4::Native_vec_ __ockl_image_sample_lod_1D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                          unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                          float c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                           unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                           float2::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                          unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                          float2::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                           unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                           float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_3D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                          unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                          float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_CM(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                          unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                          float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_sample_lod_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                           unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                           float4::Native_vec_ c, float l);

__device__ float4::Native_vec_ __ockl_image_gather4r_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                        float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4g_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                        float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4b_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                        float2::Native_vec_ c);

__device__ float4::Native_vec_ __ockl_image_gather4a_2D(unsigned int ADDRESS_SPACE_CONSTANT* i,
                                                        unsigned int ADDRESS_SPACE_CONSTANT* s,
                                                        float2::Native_vec_ c);

__device__ int __ockl_image_channel_data_type_1D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Dad(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_2Dd(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_3D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_CM(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_data_type_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_1Db(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Da(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Dad(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_2Dd(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_3D(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_CM(unsigned int ADDRESS_SPACE_CONSTANT* i);

__device__ int __ockl_image_channel_order_CMa(unsigned int ADDRESS_SPACE_CONSTANT* i);
}
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#include <hip/hip_texture_types.h>
#include <hip/amd_detail/ockl_image.h>
#include <type_traits>
#endif  // !defined(__HIPCC_RTC__)

#define TEXTURE_PARAMETERS_INIT                                                                    \
  unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)t.textureObject;  \
  unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;                    \
  (void)s;

template <typename T> struct __hip_is_tex_surf_scalar_channel_type {
  static constexpr bool value =
      __hip_internal::is_same<T, char>::value || __hip_internal::is_same<T, unsigned char>::value ||
      __hip_internal::is_same<T, short>::value ||
      __hip_internal::is_same<T, unsigned short>::value || __hip_internal::is_same<T, int>::value ||
      __hip_internal::is_same<T, unsigned int>::value || __hip_internal::is_same<T, float>::value;
};

template <typename T> struct __hip_is_tex_surf_channel_type {
  static constexpr bool value = __hip_is_tex_surf_scalar_channel_type<T>::value;
};

template <typename T, unsigned int rank>
struct __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>> {
  static constexpr bool value = __hip_is_tex_surf_scalar_channel_type<T>::value &&
                                ((rank == 1) || (rank == 2) || (rank == 4));
};

template <typename T> struct __hip_is_tex_normalized_channel_type {
  static constexpr bool value =
      __hip_internal::is_same<T, char>::value || __hip_internal::is_same<T, unsigned char>::value ||
      __hip_internal::is_same<T, short>::value || __hip_internal::is_same<T, unsigned short>::value;
};

template <typename T, unsigned int rank>
struct __hip_is_tex_normalized_channel_type<HIP_vector_type<T, rank>> {
  static constexpr bool value =
      __hip_is_tex_normalized_channel_type<T>::value && ((rank == 1) || (rank == 2) || (rank == 4));
};

template <typename T, hipTextureReadMode readMode, typename Enable = void> struct __hip_tex_ret {
  static_assert(__hip_internal::is_same<Enable, void>::value, "Invalid channel type!");
};

/*
 * Map from device function return U to scalar texture type T
 */
template <typename T, typename U> __forceinline__ __device__
    typename __hip_internal::enable_if<__hip_is_tex_surf_scalar_channel_type<T>::value,
                                       const T>::type
    __hipMapFrom(const U& u) {
  if constexpr (sizeof(T) < sizeof(float)) {
    union {
      U u;
      int i;
    } d = {u};
    return static_cast<T>(d.i);
  } else {  // sizeof(T) == sizeof(float)
    union {
      U u;
      T t;
    } d = {u};
    return d.t;
  }
}

/*
 * Map from device function return U to vector texture type T
 */
template <typename T, typename U> __forceinline__ __device__ typename __hip_internal::enable_if<
    __hip_is_tex_surf_scalar_channel_type<typename T::value_type>::value, const T>::type
__hipMapFrom(const U& u) {
  if constexpr (sizeof(typename T::value_type) < sizeof(float)) {
    union {
      U u;
      int4 i4;
    } d = {u};
    return __hipMapVector<typename T::value_type, sizeof(T) / sizeof(typename T::value_type)>(d.i4);
  } else {  // sizeof(typename T::value_type) == sizeof(float)
    union {
      U u;
      T t;
    } d = {u};
    return d.t;
  }
}

/*
 * Map from scalar texture type T to device function input U
 */
template <typename U, typename T> __forceinline__ __device__
    typename __hip_internal::enable_if<__hip_is_tex_surf_scalar_channel_type<T>::value,
                                       const U>::type
    __hipMapTo(const T& t) {
  if constexpr (sizeof(T) < sizeof(float)) {
    union {
      U u;
      int i;
    } d = {0};
    d.i = static_cast<int>(t);
    return d.u;
  } else {  // sizeof(T) == sizeof(float)
    union {
      U u;
      T t;
    } d = {0};
    d.t = t;
    return d.u;
  }
}

/*
 * Map from vector texture type T to device function input U
 */
template <typename U, typename T> __forceinline__ __device__ typename __hip_internal::enable_if<
    __hip_is_tex_surf_scalar_channel_type<typename T::value_type>::value, const U>::type
__hipMapTo(const T& t) {
  if constexpr (sizeof(typename T::value_type) < sizeof(float)) {
    union {
      U u;
      int4 i4;
    } d = {0};
    d.i4 = __hipMapVector<int, 4>(t);
    return d.u;
  } else {  // sizeof(typename T::value_type) == sizeof(float)
    union {
      U u;
      T t;
    } d = {0};
    d.t = t;
    return d.u;
  }
}

template <typename T, hipTextureReadMode readMode> using __hip_tex_ret_t =
    typename __hip_tex_ret<T, readMode, bool>::type;

template <typename T> struct __hip_tex_ret<
    T, hipReadModeElementType,
    typename __hip_internal::enable_if<__hip_is_tex_surf_channel_type<T>::value, bool>::type> {
  using type = T;
};

template <typename T, unsigned int rank> struct __hip_tex_ret<
    HIP_vector_type<T, rank>, hipReadModeElementType,
    typename __hip_internal::enable_if<
        __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>::value, bool>::type> {
  using type = HIP_vector_type<__hip_tex_ret_t<T, hipReadModeElementType>, rank>;
};

template <typename T>
struct __hip_tex_ret<T, hipReadModeNormalizedFloat,
                     typename __hip_internal::enable_if<
                         __hip_is_tex_normalized_channel_type<T>::value, bool>::type> {
  using type = float;
};

template <typename T, unsigned int rank> struct __hip_tex_ret<
    HIP_vector_type<T, rank>, hipReadModeNormalizedFloat,
    typename __hip_internal::enable_if<
        __hip_is_tex_normalized_channel_type<HIP_vector_type<T, rank>>::value, bool>::type> {
  using type = HIP_vector_type<__hip_tex_ret_t<T, hipReadModeNormalizedFloat>, rank>;
};


template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1Dfetch(
    texture<T, hipTextureType1D, readMode> t, int x) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_load_1Db(i, x);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1D(
    texture<T, hipTextureType1D, readMode> t, float x) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_sample_1D(i, s, x);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2D(
    texture<T, hipTextureType2D, readMode> t, float x, float y) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_2D(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayered(
    texture<T, hipTextureType1DLayered, readMode> t, float x, int layer) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_1Da(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayered(
    texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_2Da(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3D(
    texture<T, hipTextureType3D, readMode> t, float x, float y, float z) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_3D(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemap(
    texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_CM(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLod(
    texture<T, hipTextureType1D, readMode> t, float x, float level) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_sample_lod_1D(i, s, x, level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLod(
    texture<T, hipTextureType2D, readMode> t, float x, float y, float level) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_lod_2D(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayeredLod(
    texture<T, hipTextureType1DLayered, readMode> t, float x, int layer, float level) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_lod_1Da(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayeredLod(
    texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer, float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_lod_2Da(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3DLod(
    texture<T, hipTextureType3D, readMode> t, float x, float y, float z, float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_lod_3D(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLod(
    texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z, float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_lod_CM(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayered(
    texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, layer};
  auto tmp = __ockl_image_sample_CMa(i, s, get_native_vector(coords));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapLayeredLod(
    texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y, float z, int layer,
    float level) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, layer};
  auto tmp = __ockl_image_sample_lod_CMa(i, s, get_native_vector(coords), level);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> texCubemapGrad(
    texture<T, hipTextureTypeCubemap, readMode> t, float x, float y, float z, float4 dPdx,
    float4 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  (void)x;
  (void)y;
  (void)z;
  (void)dPdx;
  (void)dPdy;
  // TODO missing in device libs.
  // auto tmp = __ockl_image_sample_grad_CM(i, s, get_native_vector(float4(x, y, z, 0.0f)),
  // get_native_vector(float4(dPdx.x, dPdx.y, dPdx.z, 0.0f)), get_native_vector(float4(dPdy.x,
  // dPdy.y, dPdy.z, 0.0f))); return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
  return {};
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode>
texCubemapLayeredGrad(texture<T, hipTextureTypeCubemapLayered, readMode> t, float x, float y,
                      float z, int layer, float4 dPdx, float4 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  (void)x;
  (void)y;
  (void)z;
  (void)layer;
  (void)dPdx;
  (void)dPdy;
  // TODO missing in device libs.
  // auto tmp = __ockl_image_sample_grad_CMa(i, s, get_native_vector(float4(x, y, z, layer)),
  // get_native_vector(float4(dPdx.x, dPdx.y, dPdx.z, 0.0f)), get_native_vector(float4(dPdy.x,
  // dPdy.y, dPdy.z, 0.0f))); return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
  return {};
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DGrad(
    texture<T, hipTextureType1D, readMode> t, float x, float dPdx, float dPdy) {
  TEXTURE_PARAMETERS_INIT;
  auto tmp = __ockl_image_sample_grad_1D(i, s, x, dPdx, dPdy);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DGrad(
    texture<T, hipTextureType2D, readMode> t, float x, float y, float2 dPdx, float2 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_grad_2D(i, s, get_native_vector(coords), get_native_vector(dPdx),
                                         get_native_vector(dPdy));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex1DLayeredGrad(
    texture<T, hipTextureType1DLayered, readMode> t, float x, int layer, float dPdx, float dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_grad_1Da(i, s, get_native_vector(coords), dPdx, dPdy);
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex2DLayeredGrad(
    texture<T, hipTextureType2DLayered, readMode> t, float x, float y, int layer, float2 dPdx,
    float2 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_grad_2Da(i, s, get_native_vector(coords), get_native_vector(dPdx),
                                          get_native_vector(dPdy));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex_ret_t<T, readMode> tex3DGrad(
    texture<T, hipTextureType3D, readMode> t, float x, float y, float z, float4 dPdx, float4 dPdy) {
  TEXTURE_PARAMETERS_INIT;
  float4 coords{x, y, z, 0.0f};
  float4 gradx{dPdx.x, dPdx.y, dPdx.z, 0.0f};
  float4 grady{dPdy.x, dPdy.y, dPdy.z, 0.0f};
  auto tmp = __ockl_image_sample_grad_3D(i, s, get_native_vector(coords), get_native_vector(gradx),
                                         get_native_vector(grady));
  return __hipMapFrom<__hip_tex_ret_t<T, readMode>>(tmp);
}

template <typename T, hipTextureReadMode readMode, typename Enable = void>
struct __hip_tex2dgather_ret {
  static_assert(__hip_internal::is_same<Enable, void>::value, "Invalid channel type!");
};

template <typename T, hipTextureReadMode readMode> using __hip_tex2dgather_ret_t =
    typename __hip_tex2dgather_ret<T, readMode, bool>::type;

template <typename T> struct __hip_tex2dgather_ret<
    T, hipReadModeElementType,
    typename __hip_internal::enable_if<__hip_is_tex_surf_channel_type<T>::value, bool>::type> {
  using type = HIP_vector_type<T, 4>;
};

template <typename T, unsigned int rank> struct __hip_tex2dgather_ret<
    HIP_vector_type<T, rank>, hipReadModeElementType,
    typename __hip_internal::enable_if<
        __hip_is_tex_surf_channel_type<HIP_vector_type<T, rank>>::value, bool>::type> {
  using type = HIP_vector_type<T, 4>;
};

template <typename T>
struct __hip_tex2dgather_ret<T, hipReadModeNormalizedFloat,
                             typename __hip_internal::enable_if<
                                 __hip_is_tex_normalized_channel_type<T>::value, bool>::type> {
  using type = float4;
};

template <typename T, hipTextureReadMode readMode>
static __forceinline__ __device__ __hip_img_chk__ __hip_tex2dgather_ret_t<T, readMode> tex2Dgather(
    texture<T, hipTextureType2D, readMode> t, float x, float y, int comp = 0) {
  TEXTURE_PARAMETERS_INIT;
  float2 coords{x, y};
  switch (comp) {
    case 1: {
      auto tmp = __ockl_image_gather4g_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    case 2: {
      auto tmp = __ockl_image_gather4b_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    case 3: {
      auto tmp = __ockl_image_gather4a_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
    default: {
      auto tmp = __ockl_image_gather4r_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<__hip_tex2dgather_ret_t<T, readMode>>(tmp);
    }
  }
  return {};
}

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/hip_vector_types.h>
#include <hip/hip_texture_types.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/ockl_image.h>
#include <type_traits>
#endif  // !defined(__HIPCC_RTC__)

#define TEXTURE_OBJECT_PARAMETERS_INIT                                                             \
  unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)textureObject;    \
  unsigned int ADDRESS_SPACE_CONSTANT* s = i + HIP_SAMPLER_OBJECT_OFFSET_DWORD;                    \
  (void)s;

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1Dfetch(hipTextureObject_t textureObject, int x) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  auto tmp = __ockl_image_load_1Db(i, x);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1Dfetch(T* ptr, hipTextureObject_t textureObject, int x) {
  *ptr = tex1Dfetch<T>(textureObject, x);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1D(hipTextureObject_t textureObject, float x) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  auto tmp = __ockl_image_sample_1D(i, s, x);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1D(T* ptr, hipTextureObject_t textureObject, float x) {
  *ptr = tex1D<T>(textureObject, x);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2D(hipTextureObject_t textureObject, float x, float y) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_2D(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2D(T* ptr, hipTextureObject_t textureObject, float x,
                                             float y) {
  *ptr = tex2D<T>(textureObject, x, y);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3D(hipTextureObject_t textureObject, float x, float y,
                                          float z) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_3D(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3D(T* ptr, hipTextureObject_t textureObject, float x,
                                             float y, float z) {
  *ptr = tex3D<T>(textureObject, x, y, z);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayered(hipTextureObject_t textureObject, float x,
                                                 int layer) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_1Da(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayered(T* ptr, hipTextureObject_t textureObject,
                                                    float x, int layer) {
  *ptr = tex1DLayered<T>(textureObject, x, layer);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayered(hipTextureObject_t textureObject, float x, float y,
                                                 int layer) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_2Da(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayered(T* ptr, hipTextureObject_t textureObject,
                                                    float x, float y, int layer) {
  *ptr = tex1DLayered<T>(textureObject, x, y, layer);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemap(hipTextureObject_t textureObject, float x, float y,
                                               float z) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_CM(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemap(T* ptr, hipTextureObject_t textureObject, float x,
                                                  float y, float z) {
  *ptr = texCubemap<T>(textureObject, x, y, z);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayered(hipTextureObject_t textureObject, float x,
                                                      float y, float z, int layer) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, z, layer};
  auto tmp = __ockl_image_sample_CMa(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayered(T* ptr, hipTextureObject_t textureObject,
                                                         float x, float y, float z, int layer) {
  *ptr = texCubemapLayered<T>(textureObject, x, y, z, layer);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2Dgather(hipTextureObject_t textureObject, float x, float y,
                                                int comp = 0) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float2 coords{x, y};
  switch (comp) {
    case 1: {
      auto tmp = __ockl_image_gather4r_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<T>(tmp);
      break;
    }
    case 2: {
      auto tmp = __ockl_image_gather4g_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<T>(tmp);
      break;
    }
    case 3: {
      auto tmp = __ockl_image_gather4b_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<T>(tmp);
      break;
    }
    default: {
      auto tmp = __ockl_image_gather4a_2D(i, s, get_native_vector(coords));
      return __hipMapFrom<T>(tmp);
      break;
    }
  }
  return {};
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2Dgather(T* ptr, hipTextureObject_t textureObject,
                                                   float x, float y, int comp = 0) {
  *ptr = texCubemapLayered<T>(textureObject, x, y, comp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLod(hipTextureObject_t textureObject, float x,
                                             float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  auto tmp = __ockl_image_sample_lod_1D(i, s, x, level);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLod(T* ptr, hipTextureObject_t textureObject, float x,
                                                float level) {
  *ptr = tex1DLod<T>(textureObject, x, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLod(hipTextureObject_t textureObject, float x, float y,
                                             float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_lod_2D(i, s, get_native_vector(coords), level);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLod(T* ptr, hipTextureObject_t textureObject, float x,
                                                float y, float level) {
  *ptr = tex2DLod<T>(textureObject, x, y, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3DLod(hipTextureObject_t textureObject, float x, float y,
                                             float z, float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_lod_3D(i, s, get_native_vector(coords), level);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3DLod(T* ptr, hipTextureObject_t textureObject, float x,
                                                float y, float z, float level) {
  *ptr = tex3DLod<T>(textureObject, x, y, z, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayeredLod(hipTextureObject_t textureObject, float x,
                                                    int layer, float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT;
  (void)level;
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_1Da(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayeredLod(T* ptr, hipTextureObject_t textureObject,
                                                       float x, int layer, float level) {
  *ptr = tex1DLayeredLod<T>(textureObject, x, layer, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayeredLod(hipTextureObject_t textureObject, float x,
                                                    float y, int layer, float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT;
  (void)level;
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_2Da(i, s, get_native_vector(coords));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayeredLod(T* ptr, hipTextureObject_t textureObject,
                                                       float x, float y, int layer, float level) {
  *ptr = tex2DLayeredLod<T>(textureObject, x, y, layer, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLod(hipTextureObject_t textureObject, float x,
                                                  float y, float z, float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, z, 0.0f};
  auto tmp = __ockl_image_sample_lod_CM(i, s, get_native_vector(coords), level);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLod(T* ptr, hipTextureObject_t textureObject,
                                                     float x, float y, float z, float level) {
  *ptr = texCubemapLod<T>(textureObject, x, y, z, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapGrad(hipTextureObject_t textureObject, float x,
                                                   float y, float z, float4 dPdx, float4 dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT;
  (void)x;
  (void)y;
  (void)z;
  (void)dPdx;
  (void)dPdy;
  // TODO missing in device libs.
  // auto tmp = __ockl_image_sample_grad_CM(i, s, get_native_vector(float4(x, y, z, 0.0f)),
  // get_native_vector(float4(dPdx.x, dPdx.y, dPdx.z, 0.0f)), get_native_vector(float4(dPdy.x,
  // dPdy.y, dPdy.z, 0.0f))); return __hipMapFrom<T>(tmp);
  return {};
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapGrad(T* ptr, hipTextureObject_t textureObject,
                                                      float x, float y, float z, float4 dPdx,
                                                      float4 dPdy) {
  *ptr = texCubemapGrad<T>(textureObject, x, y, z, dPdx, dPdy);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayeredLod(hipTextureObject_t textureObject, float x,
                                                         float y, float z, int layer, float level) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, z, layer};
  auto tmp = __ockl_image_sample_lod_CMa(i, s, get_native_vector(coords), level);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayeredLod(T* ptr,
                                                            hipTextureObject_t textureObject,
                                                            float x, float y, float z, int layer,
                                                            float level) {
  *ptr = texCubemapLayeredLod<T>(textureObject, x, y, z, layer, level);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DGrad(hipTextureObject_t textureObject, float x, float dPdx,
                                              float dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  auto tmp = __ockl_image_sample_grad_1D(i, s, x, dPdx, dPdy);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DGrad(T* ptr, hipTextureObject_t textureObject, float x,
                                                 float dPdx, float dPdy) {
  *ptr = tex1DGrad<T>(textureObject, x, dPdx, dPdy);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DGrad(hipTextureObject_t textureObject, float x, float y,
                                              float2 dPdx, float2 dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float2 coords{x, y};
  auto tmp = __ockl_image_sample_grad_2D(i, s, get_native_vector(coords), get_native_vector(dPdx),
                                         get_native_vector(dPdy));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DGrad(T* ptr, hipTextureObject_t textureObject, float x,
                                                 float y, float2 dPdx, float2 dPdy) {
  *ptr = tex2DGrad<T>(textureObject, x, y, dPdx, dPdy);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex3DGrad(hipTextureObject_t textureObject, float x, float y,
                                              float z, float4 dPdx, float4 dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT;
  (void)dPdx;
  float4 coords{x, y, z, 0.0f};
  float4 gradx{dPdy.x, dPdy.y, dPdy.z, 0.0f};
  float4 grady{dPdy.x, dPdy.y, dPdy.z, 0.0f};
  auto tmp = __ockl_image_sample_grad_3D(i, s, get_native_vector(coords), get_native_vector(gradx),
                                         get_native_vector(grady));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex3DGrad(T* ptr, hipTextureObject_t textureObject, float x,
                                                 float y, float z, float4 dPdx, float4 dPdy) {
  *ptr = tex3DGrad<T>(textureObject, x, y, z, dPdx, dPdy);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex1DLayeredGrad(hipTextureObject_t textureObject, float x,
                                                     int layer, float dPdx, float dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float2 coords{x, layer};
  auto tmp = __ockl_image_sample_grad_1Da(i, s, get_native_vector(coords), dPdx, dPdy);
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex1DLayeredGrad(T* ptr, hipTextureObject_t textureObject,
                                                        float x, int layer, float dPdx,
                                                        float dPdy) {
  *ptr = tex1DLayeredGrad<T>(textureObject, x, layer, dPdx, dPdy);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T tex2DLayeredGrad(hipTextureObject_t textureObject, float x,
                                                     float y, int layer, float2 dPdx, float2 dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT
  float4 coords{x, y, layer, 0.0f};
  auto tmp = __ockl_image_sample_grad_2Da(i, s, get_native_vector(coords), get_native_vector(dPdx),
                                          get_native_vector(dPdy));
  return __hipMapFrom<T>(tmp);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void tex2DLayeredGrad(T* ptr, hipTextureObject_t textureObject,
                                                        float x, float y, int layer, float2 dPdx,
                                                        float2 dPdy) {
  *ptr = tex2DLayeredGrad<T>(textureObject, x, y, layer, dPdx, dPdy);
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ T texCubemapLayeredGrad(hipTextureObject_t textureObject, float x,
                                                          float y, float z, int layer, float4 dPdx,
                                                          float4 dPdy) {
  TEXTURE_OBJECT_PARAMETERS_INIT;
  (void)x;
  (void)y;
  (void)z;
  (void)layer;
  (void)dPdx;
  (void)dPdy;
  // TODO missing in device libs.
  // auto tmp = __ockl_image_sample_grad_CMa(i, s, get_native_vector(float4(x, y, z, layer)),
  // get_native_vector(float4(dPdx.x, dPdx.y, dPdx.z, 0.0f)), get_native_vector(float4(dPdy.x,
  // dPdy.y, dPdy.z, 0.0f))); return __hipMapFrom<T>(tmp);
  return {};
}

template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void texCubemapLayeredGrad(T* ptr,
                                                             hipTextureObject_t textureObject,
                                                             float x, float y, float z, int layer,
                                                             float4 dPdx, float4 dPdy) {
  *ptr = texCubemapLayeredGrad<T>(textureObject, x, y, z, layer, dPdx, dPdy);
}

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_SURFACE_FUNCTIONS_H

#if defined(__cplusplus)

#if !defined(__HIPCC_RTC__)
#include <hip/surface_types.h>
#include <hip/hip_vector_types.h>
#include <hip/amd_detail/texture_fetch_functions.h>
#include <hip/amd_detail/ockl_image.h>
#endif

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
#endif

#define __HIP_SURFACE_OBJECT_PARAMETERS_INIT                                                       \
  unsigned int ADDRESS_SPACE_CONSTANT* i = (unsigned int ADDRESS_SPACE_CONSTANT*)surfObj;

/**
 *  @defgroup SurfaceAPI Surface API
 *  @{
 */

// CUDA is using byte address, need map to pixel address for HIP
static __HOST_DEVICE__ __forceinline__ int __hipGetPixelAddr(int x, int format, int order) {
  /*
  * use below format index to generate format LUT
    typedef enum {
      HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = 4,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 6,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = 7,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 8,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 9,
      HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 10,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 11,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 12,
      HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 13,
      HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 14,
      HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = 15
    } hsa_ext_image_channel_type_t;
  */
  static const int FormatLUT[] = {0, 1, 0, 1, 3, 1, 1, 1, 0, 1, 2, 0, 1, 2, 1, 2};
  x = FormatLUT[format] == 3 ? x / FormatLUT[format] : x >> FormatLUT[format];

  /*
  * use below order index to generate order LUT
    typedef enum {
      HSA_EXT_IMAGE_CHANNEL_ORDER_A = 0,
      HSA_EXT_IMAGE_CHANNEL_ORDER_R = 1,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RX = 2,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RG = 3,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = 4,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RA = 5,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = 6,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = 7,
      HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = 8,
      HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = 9,
      HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = 10,
      HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = 11,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = 12,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = 13,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = 14,
      HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = 15,
      HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = 16,
      HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = 17,
      HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = 18,
      HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = 19
    } hsa_ext_image_channel_order_t;
  */
  static const int OrderLUT[] = {0, 0, 1, 1, 3, 1, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 0, 0, 0, 0};
  return x = OrderLUT[order] == 3 ? x / OrderLUT[order] : x >> OrderLUT[order];
}

/** \brief Reads the value at coordinate x from the one-dimensional surface.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The coordinate where the value will be read out.
 *  \param boundaryMode [in] The boundary mode is currently ignored.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1Dread(T* data, hipSurfaceObject_t surfObj, int x,
                                                  int boundaryMode = hipBoundaryModeZero) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT;
  (void)boundaryMode;
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __ockl_image_load_1D(i, x);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the one-dimensional surface at coordinate x.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The coordinate where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1Dwrite(T data, hipSurfaceObject_t surfObj, int x) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_1D(i, x, tmp);
}


/** \brief Reads the value from the two-dimensional surface at coordinate x, y.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2Dread(T* data, hipSurfaceObject_t surfObj, int x,
                                                  int y) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_2D(i, get_native_vector(coords));
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the two-dimensional surface at coordinate
 *         x, y.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2Dwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                   int y) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_2D(i, get_native_vector(coords), tmp);
}

/** \brief Reads the value from the three-dimensional surface at coordinate
 *         x, y, z.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param z [in] The z coordinate where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf3Dread(T* data, hipSurfaceObject_t surfObj, int x, int y,
                                                  int z) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_3D(i), __ockl_image_channel_order_3D(i));
  int4 coords{x, y, z, 0};
  auto tmp = __ockl_image_load_3D(i, get_native_vector(coords));
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the three-dimensional surface at coordinate
 *         x, y, z.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param z [in] The z coordinate where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf3Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y,
                                                   int z) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_3D(i), __ockl_image_channel_order_3D(i));
  int4 coords{x, y, z, 0};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_3D(i, get_native_vector(coords), tmp);
}

/** \brief Reads the value from the one-dimensional layered surface at
 *         coordinate x and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The coordinate where the value will be read out.
 *  \param layer [in] The layer index where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1DLayeredread(T* data, hipSurfaceObject_t surfObj, int x,
                                                         int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __ockl_image_load_lod_1D(i, x, layer);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the one-dimensional layered surface at
 *         coordinate x and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf1DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                          int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_1D(i), __ockl_image_channel_order_1D(i));
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_lod_1D(i, x, layer, tmp);
}

/** \brief Reads the value from the two-dimensional layered surface at
 *         coordinate x, y and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param layer [in] The layer index where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2DLayeredread(T* data, hipSurfaceObject_t surfObj, int x,
                                                         int y, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_lod_2D(i, get_native_vector(coords), layer);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the two-dimensional layered surface at
 *         coordinate x, y and layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surf2DLayeredwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                          int y, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_lod_2D(i, get_native_vector(coords), layer, tmp);
}

/** \brief Reads the value from the cubemap surface at coordinate x, y and
 *         face index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param face [in] The face index where the value will be read out.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapread(T* data, hipSurfaceObject_t surfObj, int x,
                                                       int y, int face) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_CM(i, get_native_vector(coords), face);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the cubemap surface at coordinate x, y and
 *         face index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value is written to surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param face [in] The face index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapwrite(T data, hipSurfaceObject_t surfObj, int x,
                                                        int y, int face) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_CM(i, get_native_vector(coords), face, tmp);
}

/** \brief Reads the value from the layered cubemap surface at coordinate x, y
 *         and face, layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [out] The T type result is stored in this pointer.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the value will be read out.
 *  \param y [in] The y coordinate where the value will be read out.
 *  \param face [in] The face index where the value will be read out.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapLayeredread(T* data, hipSurfaceObject_t surfObj,
                                                              int x, int y, int face, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __ockl_image_load_lod_CM(i, get_native_vector(coords), face, layer);
  *data = __hipMapFrom<T>(tmp);
}

/** \brief Writes the value data to the layered cubemap surface at coordinate
 *         x, y and face, layer index.
 *
 *  \tparam T The data type of the surface.
 *  \param data [in] The T type value to write to the surface.
 *  \param surfObj [in] The surface descriptor.
 *  \param x [in] The x coordinate where the data will be written.
 *  \param y [in] The y coordinate where the data will be written.
 *  \param face [in] The face index where the data will be written.
 *  \param layer [in] The layer index where the data will be written.
 */
template <typename T, typename __hip_internal::enable_if<
                          __hip_is_tex_surf_channel_type<T>::value>::type* = nullptr>
static __device__ __hip_img_chk__ void surfCubemapLayeredwrite(T* data, hipSurfaceObject_t surfObj,
                                                               int x, int y, int face, int layer) {
  __HIP_SURFACE_OBJECT_PARAMETERS_INIT
  x = __hipGetPixelAddr(x, __ockl_image_channel_data_type_2D(i), __ockl_image_channel_order_2D(i));
  int2 coords{x, y};
  auto tmp = __hipMapTo<float4::Native_vec_>(data);
  __ockl_image_store_lod_CM(i, get_native_vector(coords), face, layer, tmp);
}

// Doxygen end group SurfaceAPI
/**
 * @}
 */

#endif

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* The header defines complex numbers and related functions*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H

#if !defined(__HIPCC_RTC__)
#include "hip/amd_detail/amd_hip_vector_types.h"
#endif

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
// TODO: Clang has a bug which allows device functions to call std functions
// when std functions are introduced into default namespace by using statement.
// math.h may be included after this bug is fixed.
#if __cplusplus
#include <cmath>
#else
#include "math.h"
#endif
#endif  // !defined(__HIPCC_RTC__)

typedef float2 hipFloatComplex;

__HOST_DEVICE__ static inline float hipCrealf(hipFloatComplex z) { return z.x; }

__HOST_DEVICE__ static inline float hipCimagf(hipFloatComplex z) { return z.y; }

__HOST_DEVICE__ static inline hipFloatComplex make_hipFloatComplex(float a, float b) {
  hipFloatComplex z;
  z.x = a;
  z.y = b;
  return z;
}

__HOST_DEVICE__ static inline hipFloatComplex hipConjf(hipFloatComplex z) {
  hipFloatComplex ret;
  ret.x = z.x;
  ret.y = -z.y;
  return ret;
}

__HOST_DEVICE__ static inline float hipCsqabsf(hipFloatComplex z) { return z.x * z.x + z.y * z.y; }

__HOST_DEVICE__ static inline hipFloatComplex hipCaddf(hipFloatComplex p, hipFloatComplex q) {
  return make_hipFloatComplex(p.x + q.x, p.y + q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCsubf(hipFloatComplex p, hipFloatComplex q) {
  return make_hipFloatComplex(p.x - q.x, p.y - q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCmulf(hipFloatComplex p, hipFloatComplex q) {
  return make_hipFloatComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipCdivf(hipFloatComplex p, hipFloatComplex q) {
  float sqabs = hipCsqabsf(q);
  hipFloatComplex ret;
  ret.x = (p.x * q.x + p.y * q.y) / sqabs;
  ret.y = (p.y * q.x - p.x * q.y) / sqabs;
  return ret;
}

__HOST_DEVICE__ static inline float hipCabsf(hipFloatComplex z) { return sqrtf(hipCsqabsf(z)); }


typedef double2 hipDoubleComplex;

__HOST_DEVICE__ static inline double hipCreal(hipDoubleComplex z) { return z.x; }

__HOST_DEVICE__ static inline double hipCimag(hipDoubleComplex z) { return z.y; }

__HOST_DEVICE__ static inline hipDoubleComplex make_hipDoubleComplex(double a, double b) {
  hipDoubleComplex z;
  z.x = a;
  z.y = b;
  return z;
}

__HOST_DEVICE__ static inline hipDoubleComplex hipConj(hipDoubleComplex z) {
  hipDoubleComplex ret;
  ret.x = z.x;
  ret.y = -z.y;
  return ret;
}

__HOST_DEVICE__ static inline double hipCsqabs(hipDoubleComplex z) { return z.x * z.x + z.y * z.y; }

__HOST_DEVICE__ static inline hipDoubleComplex hipCadd(hipDoubleComplex p, hipDoubleComplex q) {
  return make_hipDoubleComplex(p.x + q.x, p.y + q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCsub(hipDoubleComplex p, hipDoubleComplex q) {
  return make_hipDoubleComplex(p.x - q.x, p.y - q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCmul(hipDoubleComplex p, hipDoubleComplex q) {
  return make_hipDoubleComplex(p.x * q.x - p.y * q.y, p.y * q.x + p.x * q.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCdiv(hipDoubleComplex p, hipDoubleComplex q) {
  double sqabs = hipCsqabs(q);
  hipDoubleComplex ret;
  ret.x = (p.x * q.x + p.y * q.y) / sqabs;
  ret.y = (p.y * q.x - p.x * q.y) / sqabs;
  return ret;
}

__HOST_DEVICE__ static inline double hipCabs(hipDoubleComplex z) { return sqrt(hipCsqabs(z)); }

typedef hipFloatComplex hipComplex;

__HOST_DEVICE__ static inline hipComplex make_hipComplex(float x, float y) {
  return make_hipFloatComplex(x, y);
}

__HOST_DEVICE__ static inline hipFloatComplex hipComplexDoubleToFloat(hipDoubleComplex z) {
  return make_hipFloatComplex((float)z.x, (float)z.y);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipComplexFloatToDouble(hipFloatComplex z) {
  return make_hipDoubleComplex((double)z.x, (double)z.y);
}

__HOST_DEVICE__ static inline hipComplex hipCfmaf(hipComplex p, hipComplex q, hipComplex r) {
  float real = (p.x * q.x) + r.x;
  float imag = (q.x * p.y) + r.y;

  real = -(p.y * q.y) + real;
  imag = (p.x * q.y) + imag;

  return make_hipComplex(real, imag);
}

__HOST_DEVICE__ static inline hipDoubleComplex hipCfma(hipDoubleComplex p, hipDoubleComplex q,
                                                       hipDoubleComplex r) {
  double real = (p.x * q.x) + r.x;
  double imag = (q.x * p.y) + r.y;

  real = -(p.y * q.y) + real;
  imag = (p.x * q.y) + imag;

  return make_hipDoubleComplex(real, imag);
}

#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COMPLEX_H
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef AMD_HIP_MATH_CONSTANTS_H
#define AMD_HIP_MATH_CONSTANTS_H

// single precision constants
#define HIP_INF_F __int_as_float(0x7f800000U)
#define HIP_NAN_F __int_as_float(0x7fffffffU)
#define HIP_MIN_DENORM_F __int_as_float(0x00000001U)
#define HIP_MAX_NORMAL_F __int_as_float(0x7f7fffffU)
#define HIP_NEG_ZERO_F __int_as_float(0x80000000U)
#define HIP_ZERO_F 0.0F
#define HIP_ONE_F 1.0F
#define HIP_SQRT_HALF_F 0.707106781F
#define HIP_SQRT_HALF_HI_F 0.707106781F
#define HIP_SQRT_HALF_LO_F 1.210161749e-08F
#define HIP_SQRT_TWO_F 1.414213562F
#define HIP_THIRD_F 0.333333333F
#define HIP_PIO4_F 0.785398163F
#define HIP_PIO2_F 1.570796327F
#define HIP_3PIO4_F 2.356194490F
#define HIP_2_OVER_PI_F 0.636619772F
#define HIP_SQRT_2_OVER_PI_F 0.797884561F
#define HIP_PI_F 3.141592654F
#define HIP_L2E_F 1.442695041F
#define HIP_L2T_F 3.321928094F
#define HIP_LG2_F 0.301029996F
#define HIP_LGE_F 0.434294482F
#define HIP_LN2_F 0.693147181F
#define HIP_LNT_F 2.302585093F
#define HIP_LNPI_F 1.144729886F
#define HIP_TWO_TO_M126_F 1.175494351e-38F
#define HIP_TWO_TO_126_F 8.507059173e37F
#define HIP_NORM_HUGE_F 3.402823466e38F
#define HIP_TWO_TO_23_F 8388608.0F
#define HIP_TWO_TO_24_F 16777216.0F
#define HIP_TWO_TO_31_F 2147483648.0F
#define HIP_TWO_TO_32_F 4294967296.0F
#define HIP_REMQUO_BITS_F 3U
#define HIP_REMQUO_MASK_F (~((~0U) << HIP_REMQUO_BITS_F))
#define HIP_TRIG_PLOSS_F 105615.0F

// double precision constants
#define HIP_INF __longlong_as_double(0x7ff0000000000000ULL)
#define HIP_NAN __longlong_as_double(0xfff8000000000000ULL)
#define HIP_NEG_ZERO __longlong_as_double(0x8000000000000000ULL)
#define HIP_MIN_DENORM __longlong_as_double(0x0000000000000001ULL)
#define HIP_ZERO 0.0
#define HIP_ONE 1.0
#define HIP_SQRT_TWO 1.4142135623730951e+0
#define HIP_SQRT_HALF 7.0710678118654757e-1
#define HIP_SQRT_HALF_HI 7.0710678118654757e-1
#define HIP_SQRT_HALF_LO (-4.8336466567264567e-17)
#define HIP_THIRD 3.3333333333333333e-1
#define HIP_TWOTHIRD 6.6666666666666667e-1
#define HIP_PIO4 7.8539816339744828e-1
#define HIP_PIO4_HI 7.8539816339744828e-1
#define HIP_PIO4_LO 3.0616169978683830e-17
#define HIP_PIO2 1.5707963267948966e+0
#define HIP_PIO2_HI 1.5707963267948966e+0
#define HIP_PIO2_LO 6.1232339957367660e-17
#define HIP_3PIO4 2.3561944901923448e+0
#define HIP_2_OVER_PI 6.3661977236758138e-1
#define HIP_PI 3.1415926535897931e+0
#define HIP_PI_HI 3.1415926535897931e+0
#define HIP_PI_LO 1.2246467991473532e-16
#define HIP_SQRT_2PI 2.5066282746310007e+0
#define HIP_SQRT_2PI_HI 2.5066282746310007e+0
#define HIP_SQRT_2PI_LO (-1.8328579980459167e-16)
#define HIP_SQRT_PIO2 1.2533141373155003e+0
#define HIP_SQRT_PIO2_HI 1.2533141373155003e+0
#define HIP_SQRT_PIO2_LO (-9.1642899902295834e-17)
#define HIP_SQRT_2OPI 7.9788456080286536e-1
#define HIP_L2E 1.4426950408889634e+0
#define HIP_L2E_HI 1.4426950408889634e+0
#define HIP_L2E_LO 2.0355273740931033e-17
#define HIP_L2T 3.3219280948873622e+0
#define HIP_LG2 3.0102999566398120e-1
#define HIP_LG2_HI 3.0102999566398120e-1
#define HIP_LG2_LO (-2.8037281277851704e-18)
#define HIP_LGE 4.3429448190325182e-1
#define HIP_LGE_HI 4.3429448190325182e-1
#define HIP_LGE_LO 1.09831965021676510e-17
#define HIP_LN2 6.9314718055994529e-1
#define HIP_LN2_HI 6.9314718055994529e-1
#define HIP_LN2_LO 2.3190468138462996e-17
#define HIP_LNT 2.3025850929940459e+0
#define HIP_LNT_HI 2.3025850929940459e+0
#define HIP_LNT_LO (-2.1707562233822494e-16)
#define HIP_LNPI 1.1447298858494002e+0
#define HIP_LN2_X_1024 7.0978271289338397e+2
#define HIP_LN2_X_1025 7.1047586007394398e+2
#define HIP_LN2_X_1075 7.4513321910194122e+2
#define HIP_LG2_X_1024 3.0825471555991675e+2
#define HIP_LG2_X_1075 3.2360724533877976e+2
#define HIP_TWO_TO_23 8388608.0
#define HIP_TWO_TO_52 4503599627370496.0
#define HIP_TWO_TO_53 9007199254740992.0
#define HIP_TWO_TO_54 18014398509481984.0
#define HIP_TWO_TO_M54 5.5511151231257827e-17
#define HIP_TWO_TO_M1022 2.22507385850720140e-308
#define HIP_TRIG_PLOSS 2147483648.0
#define HIP_DBL2INT_CVT 6755399441055744.0

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#include "amd_hip_vector_types.h"  // For Native_vec_
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__device__ __attribute__((const)) int __ockl_sdot2(HIP_vector_base<short, 2>::Native_vec_,
                                                   HIP_vector_base<short, 2>::Native_vec_, int,
                                                   bool);

__device__ __attribute__((const)) unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_, unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot4(HIP_vector_base<char, 4>::Native_vec_,
                                                   HIP_vector_base<char, 4>::Native_vec_, int,
                                                   bool);

__device__ __attribute__((const)) unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_, HIP_vector_base<unsigned char, 4>::Native_vec_,
    unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot8(int, int, int, bool);

__device__ __attribute__((const)) unsigned int __ockl_udot8(unsigned int, unsigned int,
                                                            unsigned int, bool);
#endif

// BEGIN FLOAT
__device__ __attribute__((const)) float __ocml_acos_f32(float);
__device__ __attribute__((pure)) float __ocml_acosh_f32(float);
__device__ __attribute__((const)) float __ocml_asin_f32(float);
__device__ __attribute__((pure)) float __ocml_asinh_f32(float);
__device__ __attribute__((const)) float __ocml_atan2_f32(float, float);
__device__ __attribute__((const)) float __ocml_atan_f32(float);
__device__ __attribute__((pure)) float __ocml_atanh_f32(float);
__device__ __attribute__((pure)) float __ocml_cbrt_f32(float);
__device__ __attribute__((const)) float __ocml_ceil_f32(float);
__device__ __attribute__((const)) __device__ float __ocml_copysign_f32(float, float);
__device__ float __ocml_cos_f32(float);
__device__ float __ocml_native_cos_f32(float);
__device__ __attribute__((pure)) __device__ float __ocml_cosh_f32(float);
__device__ float __ocml_cospi_f32(float);
__device__ float __ocml_i0_f32(float);
__device__ float __ocml_i1_f32(float);
__device__ __attribute__((pure)) float __ocml_erfc_f32(float);
__device__ __attribute__((pure)) float __ocml_erfcinv_f32(float);
__device__ __attribute__((pure)) float __ocml_erfcx_f32(float);
__device__ __attribute__((pure)) float __ocml_erf_f32(float);
__device__ __attribute__((pure)) float __ocml_erfinv_f32(float);
__device__ __attribute__((pure)) float __ocml_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_exp2_f32(float);
__device__ __attribute__((pure)) float __ocml_exp_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp_f32(float);
__device__ __attribute__((pure)) float __ocml_expm1_f32(float);
__device__ __attribute__((const)) float __ocml_fabs_f32(float);
__device__ __attribute__((const)) float __ocml_fdim_f32(float, float);
__device__ __attribute__((const)) float __ocml_floor_f32(float);
__device__ __attribute__((const)) float __ocml_fma_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fmax_f32(float, float);
__device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
__device__ __attribute__((const)) __device__ float __ocml_fmod_f32(float, float);
__device__ float __ocml_frexp_f32(float, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) float __ocml_hypot_f32(float, float);
__device__ __attribute__((const)) int __ocml_ilogb_f32(float);
__device__ __attribute__((const)) int __ocml_isfinite_f32(float);
__device__ __attribute__((const)) int __ocml_isinf_f32(float);
__device__ __attribute__((const)) int __ocml_isnan_f32(float);
__device__ float __ocml_j0_f32(float);
__device__ float __ocml_j1_f32(float);
__device__ __attribute__((const)) float __ocml_ldexp_f32(float, int);
__device__ float __ocml_lgamma_f32(float);
__device__ __attribute__((pure)) float __ocml_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_log1p_f32(float);
__device__ __attribute__((pure)) float __ocml_log2_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log2_f32(float);
__device__ __attribute__((const)) float __ocml_logb_f32(float);
__device__ __attribute__((pure)) float __ocml_log_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log_f32(float);
__device__ float __ocml_modf_f32(float, __attribute__((address_space(5))) float*);
__device__ __attribute__((const)) float __ocml_nearbyint_f32(float);
__device__ __attribute__((const)) float __ocml_nextafter_f32(float, float);
__device__ __attribute__((const)) float __ocml_len3_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_len4_f32(float, float, float, float);
__device__ __attribute__((pure)) float __ocml_ncdf_f32(float);
__device__ __attribute__((pure)) float __ocml_ncdfinv_f32(float);
__device__ __attribute__((pure)) float __ocml_pow_f32(float, float);
__device__ __attribute__((pure)) float __ocml_pown_f32(float, int);
__device__ __attribute__((pure)) float __ocml_rcbrt_f32(float);
__device__ __attribute__((const)) float __ocml_remainder_f32(float, float);
__device__ float __ocml_remquo_f32(float, float, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) float __ocml_rhypot_f32(float, float);
__device__ __attribute__((const)) float __ocml_rint_f32(float);
__device__ __attribute__((const)) float __ocml_rlen3_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_rlen4_f32(float, float, float, float);
__device__ __attribute__((const)) float __ocml_round_f32(float);
__device__ __attribute__((pure)) float __ocml_rsqrt_f32(float);
__device__ __attribute__((const)) float __ocml_scalb_f32(float, float);
__device__ __attribute__((const)) float __ocml_scalbn_f32(float, int);
__device__ __attribute__((const)) int __ocml_signbit_f32(float);
__device__ float __ocml_sincos_f32(float, __attribute__((address_space(5))) float*);
__device__ float __ocml_sincospi_f32(float, __attribute__((address_space(5))) float*);
__device__ float __ocml_sin_f32(float);
__device__ float __ocml_native_sin_f32(float);
__device__ __attribute__((pure)) float __ocml_sinh_f32(float);
__device__ float __ocml_sinpi_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_f32(float);
__device__ __attribute__((const)) float __ocml_native_sqrt_f32(float);
__device__ float __ocml_tan_f32(float);
__device__ __attribute__((pure)) float __ocml_tanh_f32(float);
__device__ float __ocml_tgamma_f32(float);
__device__ __attribute__((const)) float __ocml_trunc_f32(float);
__device__ float __ocml_y0_f32(float);
__device__ float __ocml_y1_f32(float);

// BEGIN INTRINSICS
__device__ __attribute__((const)) float __ocml_add_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_sqrt_rte_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtn_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtp_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtz_f32(float);
__device__ __attribute__((const)) float __ocml_fma_rte_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtn_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtp_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtz_f32(float, float, float);
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__device__ __attribute__((const)) double __ocml_acos_f64(double);
__device__ __attribute__((pure)) double __ocml_acosh_f64(double);
__device__ __attribute__((const)) double __ocml_asin_f64(double);
__device__ __attribute__((pure)) double __ocml_asinh_f64(double);
__device__ __attribute__((const)) double __ocml_atan2_f64(double, double);
__device__ __attribute__((const)) double __ocml_atan_f64(double);
__device__ __attribute__((pure)) double __ocml_atanh_f64(double);
__device__ __attribute__((pure)) double __ocml_cbrt_f64(double);
__device__ __attribute__((const)) double __ocml_ceil_f64(double);
__device__ __attribute__((const)) double __ocml_copysign_f64(double, double);
__device__ double __ocml_cos_f64(double);
__device__ __attribute__((pure)) double __ocml_cosh_f64(double);
__device__ double __ocml_cospi_f64(double);
__device__ double __ocml_i0_f64(double);
__device__ double __ocml_i1_f64(double);
__device__ __attribute__((pure)) double __ocml_erfc_f64(double);
__device__ __attribute__((pure)) double __ocml_erfcinv_f64(double);
__device__ __attribute__((pure)) double __ocml_erfcx_f64(double);
__device__ __attribute__((pure)) double __ocml_erf_f64(double);
__device__ __attribute__((pure)) double __ocml_erfinv_f64(double);
__device__ __attribute__((pure)) double __ocml_exp10_f64(double);
__device__ __attribute__((pure)) double __ocml_exp2_f64(double);
__device__ __attribute__((pure)) double __ocml_exp_f64(double);
__device__ __attribute__((pure)) double __ocml_expm1_f64(double);
__device__ __attribute__((const)) double __ocml_fabs_f64(double);
__device__ __attribute__((const)) double __ocml_fdim_f64(double, double);
__device__ __attribute__((const)) double __ocml_floor_f64(double);
__device__ __attribute__((const)) double __ocml_fma_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fmax_f64(double, double);
__device__ __attribute__((const)) double __ocml_fmin_f64(double, double);
__device__ __attribute__((const)) double __ocml_fmod_f64(double, double);
__device__ double __ocml_frexp_f64(double, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) double __ocml_hypot_f64(double, double);
__device__ __attribute__((const)) int __ocml_ilogb_f64(double);
__device__ __attribute__((const)) int __ocml_isfinite_f64(double);
__device__ __attribute__((const)) int __ocml_isinf_f64(double);
__device__ __attribute__((const)) int __ocml_isnan_f64(double);
__device__ double __ocml_j0_f64(double);
__device__ double __ocml_j1_f64(double);
__device__ __attribute__((const)) double __ocml_ldexp_f64(double, int);
__device__ double __ocml_lgamma_f64(double);
__device__ __attribute__((pure)) double __ocml_log10_f64(double);
__device__ __attribute__((pure)) double __ocml_log1p_f64(double);
__device__ __attribute__((pure)) double __ocml_log2_f64(double);
__device__ __attribute__((const)) double __ocml_logb_f64(double);
__device__ __attribute__((pure)) double __ocml_log_f64(double);
__device__ double __ocml_modf_f64(double, __attribute__((address_space(5))) double*);
__device__ __attribute__((const)) double __ocml_nearbyint_f64(double);
__device__ __attribute__((const)) double __ocml_nextafter_f64(double, double);
__device__ __attribute__((const)) double __ocml_len3_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_len4_f64(double, double, double, double);
__device__ __attribute__((pure)) double __ocml_ncdf_f64(double);
__device__ __attribute__((pure)) double __ocml_ncdfinv_f64(double);
__device__ __attribute__((pure)) double __ocml_pow_f64(double, double);
__device__ __attribute__((pure)) double __ocml_pown_f64(double, int);
__device__ __attribute__((pure)) double __ocml_rcbrt_f64(double);
__device__ __attribute__((const)) double __ocml_remainder_f64(double, double);
__device__ double __ocml_remquo_f64(double, double, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) double __ocml_rhypot_f64(double, double);
__device__ __attribute__((const)) double __ocml_rint_f64(double);
__device__ __attribute__((const)) double __ocml_rlen3_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_rlen4_f64(double, double, double, double);
__device__ __attribute__((const)) double __ocml_round_f64(double);
__device__ __attribute__((pure)) double __ocml_rsqrt_f64(double);
__device__ __attribute__((const)) double __ocml_scalb_f64(double, double);
__device__ __attribute__((const)) double __ocml_scalbn_f64(double, int);
__device__ __attribute__((const)) int __ocml_signbit_f64(double);
__device__ double __ocml_sincos_f64(double, __attribute__((address_space(5))) double*);
__device__ double __ocml_sincospi_f64(double, __attribute__((address_space(5))) double*);
__device__ double __ocml_sin_f64(double);
__device__ __attribute__((pure)) double __ocml_sinh_f64(double);
__device__ double __ocml_sinpi_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_f64(double);
__device__ double __ocml_tan_f64(double);
__device__ __attribute__((pure)) double __ocml_tanh_f64(double);
__device__ double __ocml_tgamma_f64(double);
__device__ __attribute__((const)) double __ocml_trunc_f64(double);
__device__ double __ocml_y0_f64(double);
__device__ double __ocml_y1_f64(double);

// BEGIN INTRINSICS
__device__ __attribute__((const)) double __ocml_add_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_sqrt_rte_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtn_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtp_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtz_f64(double);
__device__ __attribute__((const)) double __ocml_fma_rte_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fma_rtn_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fma_rtp_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fma_rtz_f64(double, double, double);
// END INTRINSICS
// END DOUBLE

#if defined(__cplusplus)
}  // extern "C"
#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_WARP_FUNCTIONS_H

#if !defined(__HIPCC_RTC__)
#include "device_library_decls.h"  // ockl warp functions
#endif                             // !defined(__HIPCC_RTC__)

#if defined(__has_attribute) && __has_attribute(maybe_undef)
#define MAYBE_UNDEF __attribute__((maybe_undef))
#else
#define MAYBE_UNDEF
#endif

__device__ static inline unsigned __hip_ds_bpermute(int index, unsigned src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)
              ? __builtin_amdgcn_ds_bpermute(index, tmp.i)
              : 0;
  return tmp.u;
}

__device__ static inline float __hip_ds_bpermutef(int index, float src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)
              ? __builtin_amdgcn_ds_bpermute(index, tmp.i)
              : 0;
  return tmp.f;
}

__device__ static inline unsigned __hip_ds_permute(int index, unsigned src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_permute)
              ? __builtin_amdgcn_ds_permute(index, tmp.i)
              : 0;
  return tmp.u;
}

__device__ static inline float __hip_ds_permutef(int index, float src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_permute)
              ? __builtin_amdgcn_ds_permute(index, tmp.i)
              : 0;
  return tmp.f;
}

#define __hip_ds_swizzle(src, pattern) __hip_ds_swizzle_N<(pattern)>((src))
#define __hip_ds_swizzlef(src, pattern) __hip_ds_swizzlef_N<(pattern)>((src))

template <int pattern> __device__ static inline unsigned __hip_ds_swizzle_N(unsigned int src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_swizzle)
              ? __builtin_amdgcn_ds_swizzle(tmp.i, pattern)
              : 0;
  return tmp.u;
}

template <int pattern> __device__ static inline float __hip_ds_swizzlef_N(float src) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = src;
  tmp.i = __builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_swizzle)
              ? __builtin_amdgcn_ds_swizzle(tmp.i, pattern)
              : 0;
  return tmp.f;
}

#define __hip_move_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)                             \
  __hip_move_dpp_N<(dpp_ctrl), (row_mask), (bank_mask), (bound_ctrl)>((src))

template <int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl>
__device__ static inline int __hip_move_dpp_N(int src) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mov_dpp))
    return __builtin_amdgcn_mov_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
  __builtin_trap();
}

inline __device__ const struct {
  __device__ __attribute__((always_inline, const)) operator int() const noexcept {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_wavefrontsize))
      return __builtin_amdgcn_wavefrontsize();
    __builtin_trap();
  }
} warpSize{};

// warp vote function __all __any __ballot
__device__ inline int __all(int predicate) { return __ockl_wfall_i32(predicate); }

__device__ inline int __any(int predicate) { return __ockl_wfany_i32(predicate); }

__device__ inline unsigned long long int __ballot(int predicate) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ballot_w64))
    return __builtin_amdgcn_ballot_w64(predicate);
  __builtin_trap();
}

__device__ inline unsigned long long int __ballot64(int predicate) { return __ballot(predicate); }

// See amd_warp_sync_functions.h for an explanation of this preprocessor flag.
#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
// Since threads in a wave do not make independent progress, __activemask()
// always returns the exact active mask, i.e, all active threads in the wave.
__device__ inline unsigned long long __activemask() { return __ballot(true); }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS

__device__ static inline unsigned int __lane_id() {
  if (static_cast<int>(warpSize) == 32) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_lo))
      return __builtin_amdgcn_mbcnt_lo(-1, 0);
    __builtin_trap();
  }
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_hi))
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_lo))
      return __builtin_amdgcn_mbcnt_hi(-1, __builtin_amdgcn_mbcnt_lo(-1, 0));
  __builtin_trap();
}

__device__ inline int __shfl(MAYBE_UNDEF int var, int src_lane, int width = warpSize) {
  int self = __lane_id();
  int index = (src_lane & (width - 1)) + (self & ~(width - 1));
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl(MAYBE_UNDEF unsigned int var, int src_lane,
                                      int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.u;
}
__device__ inline float __shfl(MAYBE_UNDEF float var, int src_lane, int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.f;
}
__device__ inline double __shfl(MAYBE_UNDEF double var, int src_lane, int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl(MAYBE_UNDEF long var, int src_lane, int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl(static_cast<int>(var), src_lane, width));
#endif
}
__device__ inline unsigned long __shfl(MAYBE_UNDEF unsigned long var, int src_lane,
                                       int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl(static_cast<unsigned int>(var), src_lane, width));
#endif
}
__device__ inline long long __shfl(MAYBE_UNDEF long long var, int src_lane, int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline unsigned long long __shfl(MAYBE_UNDEF unsigned long long var, int src_lane,
                                            int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl(tmp[0], src_lane, width);
  tmp[1] = __shfl(tmp[1], src_lane, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline int __shfl_up(MAYBE_UNDEF int var, unsigned int lane_delta,
                                int width = warpSize) {
  int self = __lane_id();
  int index = self - lane_delta;
  index = (index < (self & ~(width - 1))) ? self : index;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl_up(MAYBE_UNDEF unsigned int var, unsigned int lane_delta,
                                         int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.u;
}
__device__ inline float __shfl_up(MAYBE_UNDEF float var, unsigned int lane_delta,
                                  int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.f;
}
__device__ inline double __shfl_up(MAYBE_UNDEF double var, unsigned int lane_delta,
                                   int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl_up(MAYBE_UNDEF long var, unsigned int lane_delta,
                                 int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl_up(static_cast<int>(var), lane_delta, width));
#endif
}

__device__ inline unsigned long __shfl_up(MAYBE_UNDEF unsigned long var, unsigned int lane_delta,
                                          int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl_up(static_cast<unsigned int>(var), lane_delta, width));
#endif
}

__device__ inline long long __shfl_up(MAYBE_UNDEF long long var, unsigned int lane_delta,
                                      int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");
  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline unsigned long long __shfl_up(MAYBE_UNDEF unsigned long long var,
                                               unsigned int lane_delta, int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");
  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_up(tmp[0], lane_delta, width);
  tmp[1] = __shfl_up(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline int __shfl_down(MAYBE_UNDEF int var, unsigned int lane_delta,
                                  int width = warpSize) {
  int self = __lane_id();
  int index = self + lane_delta;
  index = (int)((self & (width - 1)) + lane_delta) >= width ? self : index;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl_down(MAYBE_UNDEF unsigned int var, unsigned int lane_delta,
                                           int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.u;
}
__device__ inline float __shfl_down(MAYBE_UNDEF float var, unsigned int lane_delta,
                                    int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.f;
}
__device__ inline double __shfl_down(MAYBE_UNDEF double var, unsigned int lane_delta,
                                     int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl_down(MAYBE_UNDEF long var, unsigned int lane_delta,
                                   int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl_down(static_cast<int>(var), lane_delta, width));
#endif
}
__device__ inline unsigned long __shfl_down(MAYBE_UNDEF unsigned long var, unsigned int lane_delta,
                                            int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl_down(static_cast<unsigned int>(var), lane_delta, width));
#endif
}
__device__ inline long long __shfl_down(MAYBE_UNDEF long long var, unsigned int lane_delta,
                                        int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");
  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline unsigned long long __shfl_down(MAYBE_UNDEF unsigned long long var,
                                                 unsigned int lane_delta, int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");
  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_down(tmp[0], lane_delta, width);
  tmp[1] = __shfl_down(tmp[1], lane_delta, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

__device__ inline int __shfl_xor(MAYBE_UNDEF int var, int lane_mask, int width = warpSize) {
  int self = __lane_id();
  int index = self ^ lane_mask;
  index = index >= ((self + width) & ~(width - 1)) ? self : index;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_bpermute)) {
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    return __builtin_amdgcn_wave_shuffle(var, index);
#else
    return __builtin_amdgcn_ds_bpermute(index << 2, var);
#endif
  }
  __builtin_trap();
}
__device__ inline unsigned int __shfl_xor(MAYBE_UNDEF unsigned int var, int lane_mask,
                                          int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.u = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.u;
}
__device__ inline float __shfl_xor(MAYBE_UNDEF float var, int lane_mask, int width = warpSize) {
  union {
    int i;
    unsigned u;
    float f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.f;
}
__device__ inline double __shfl_xor(MAYBE_UNDEF double var, int lane_mask, int width = warpSize) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline long __shfl_xor(MAYBE_UNDEF long var, int lane_mask, int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(long) == 2 * sizeof(int), "");
  static_assert(sizeof(long) == sizeof(__hip_uint64_t), "");

  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(long) == sizeof(int), "");
  return static_cast<long>(__shfl_xor(static_cast<int>(var), lane_mask, width));
#endif
}
__device__ inline unsigned long __shfl_xor(MAYBE_UNDEF unsigned long var, int lane_mask,
                                           int width = warpSize) {
#ifndef _MSC_VER
  static_assert(sizeof(unsigned long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long) == sizeof(__hip_uint64_t), "");

  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
#else
  static_assert(sizeof(unsigned long) == sizeof(unsigned int), "");
  return static_cast<unsigned long>(__shfl_xor(static_cast<unsigned int>(var), lane_mask, width));
#endif
}
__device__ inline long long __shfl_xor(MAYBE_UNDEF long long var, int lane_mask,
                                       int width = warpSize) {
  static_assert(sizeof(long long) == 2 * sizeof(int), "");
  static_assert(sizeof(long long) == sizeof(__hip_uint64_t), "");
  int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}
__device__ inline unsigned long long __shfl_xor(MAYBE_UNDEF unsigned long long var, int lane_mask,
                                                int width = warpSize) {
  static_assert(sizeof(unsigned long long) == 2 * sizeof(unsigned int), "");
  static_assert(sizeof(unsigned long long) == sizeof(__hip_uint64_t), "");
  unsigned int tmp[2];
  __builtin_memcpy(tmp, &var, sizeof(tmp));
  tmp[0] = __shfl_xor(tmp[0], lane_mask, width);
  tmp[1] = __shfl_xor(tmp[1], lane_mask, width);
  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(tmp[1]) << 32ull) | static_cast<__hip_uint32_t>(tmp[0]);
  unsigned long long tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));
  return tmp1;
}

#endif
/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_FUNCTIONS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_DEVICE_FUNCTIONS_H

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#include <hip/amd_detail/device_library_decls.h>
#include <hip/amd_detail/hip_assert.h>
#include "host_defines.h"
#include "math_fwd.h"
#include <hip/hip_runtime_api.h>
#include <stddef.h>
#include <hip/hip_vector_types.h>
#endif  // !defined(__HIPCC_RTC__)

#if defined(__clang__) && defined(__HIP__)
extern "C" __device__ int printf(const char* fmt, ...);
#else
template <typename... All> static inline __device__ void printf(const char* format, All... all) {}
#endif

/*
Integer Intrinsics
*/

// integer intrinsic function __poc __clz __ffs __brev
__device__ static inline unsigned int __popc(unsigned int input) {
  return __builtin_popcount(input);
}
__device__ static inline unsigned int __popcll(unsigned long long int input) {
  return __builtin_popcountll(input);
}

__device__ static inline int __clz(int input) {
  return input == 0u ? 32 : __builtin_clz((uint)input);
}

__device__ static inline int __clzll(long long int input) {
  return input == 0u ? 64 : __builtin_clzll((__hip_uint64_t)input);
}

__device__ static inline int __ffs(unsigned int input) {
  return (input == 0 ? -1 : __builtin_ctz(input)) + 1;
}

__device__ static inline int __ffsll(unsigned long long int input) {
  return (input == 0 ? -1 : __builtin_ctzll(input)) + 1;
}

__device__ static inline int __ffs(int input) {
  return (input == 0 ? -1 : __builtin_ctz(input)) + 1;
}

__device__ static inline int __ffsll(long long int input) {
  return (input == 0 ? -1 : __builtin_ctzll(input)) + 1;
}

// Given a 32/64-bit value exec mask and an integer value base (between 0 and WAVEFRONT_SIZE),
// find the n-th (given by offset) set bit in the exec mask from the base bit, and return the bit
// position. If not found, return -1.
__device__ static __hip_int32_t __fns64(__hip_uint64_t mask, __hip_uint32_t base,
                                        __hip_int32_t offset) {
  __hip_uint64_t temp_mask = mask;
  __hip_int32_t temp_offset = offset;

  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  } else if (offset < 0) {
    temp_mask = __builtin_bitreverse64(mask);
    base = 63 - base;
    temp_offset = -offset;
  }

  temp_mask = temp_mask & ((~0ULL) << base);
  if (__builtin_popcountll(temp_mask) < temp_offset) return -1;
  __hip_int32_t total = 0;
  for (int i = 0x20; i > 0; i >>= 1) {
    __hip_uint64_t temp_mask_lo = temp_mask & ((1ULL << i) - 1);
    __hip_int32_t pcnt = __builtin_popcountll(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    } else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 63 - total;
  else
    return total;
}

__device__ static __hip_int32_t __fns32(__hip_uint64_t mask, __hip_uint32_t base,
                                        __hip_int32_t offset) {
  __hip_uint32_t temp_mask = mask;
  __hip_int32_t temp_offset = offset;
  if (offset == 0) {
    temp_mask &= (1 << base);
    temp_offset = 1;
  } else if (offset < 0) {
    temp_mask = __builtin_bitreverse32(mask);
    base = 31 - base;
    temp_offset = -offset;
  }
  temp_mask = temp_mask & ((~0U) << base);
  if (__builtin_popcount(temp_mask) < temp_offset) return -1;
  __hip_int32_t total = 0;
  for (int i = 0x10; i > 0; i >>= 1) {
    __hip_uint32_t temp_mask_lo = temp_mask & ((1U << i) - 1);
    __hip_int32_t pcnt = __builtin_popcount(temp_mask_lo);
    if (pcnt < temp_offset) {
      temp_mask = temp_mask >> i;
      temp_offset -= pcnt;
      total += i;
    } else {
      temp_mask = temp_mask_lo;
    }
  }
  if (offset < 0)
    return 31 - total;
  else
    return total;
}

// Wrapper around __fns32() to make porting from CUDA easier
__device__ static __hip_int32_t __fns(unsigned int mask, unsigned int base, int offset) {
  return __fns32(mask, base, offset);
}

__device__ static inline unsigned int __brev(unsigned int input) {
  return __builtin_bitreverse32(input);
}

__device__ static inline unsigned long long int __brevll(unsigned long long int input) {
  return __builtin_bitreverse64(input);
}

__device__ static inline unsigned int __lastbit_u32_u64(__hip_uint64_t input) {
  return input == 0 ? -1 : __builtin_ctzl(input);
}

__device__ static inline unsigned int __bitextract_u32(unsigned int src0, unsigned int src1,
                                                       unsigned int src2) {
  __hip_uint32_t offset = src1 & 31;
  __hip_uint32_t width = src2 & 31;
  return width == 0 ? 0 : (src0 << (32 - offset - width)) >> (32 - width);
}

__device__ static inline __hip_uint64_t __bitextract_u64(__hip_uint64_t src0, unsigned int src1,
                                                         unsigned int src2) {
  __hip_uint64_t offset = src1 & 63;
  __hip_uint64_t width = src2 & 63;
  return width == 0 ? 0 : (src0 << (64 - offset - width)) >> (64 - width);
}

__device__ static inline unsigned int __bitinsert_u32(unsigned int src0, unsigned int src1,
                                                      unsigned int src2, unsigned int src3) {
  __hip_uint32_t offset = src2 & 31;
  __hip_uint32_t width = src3 & 31;
  __hip_uint32_t mask = (1 << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__device__ static inline __hip_uint64_t __bitinsert_u64(__hip_uint64_t src0, __hip_uint64_t src1,
                                                        unsigned int src2, unsigned int src3) {
  __hip_uint64_t offset = src2 & 63;
  __hip_uint64_t width = src3 & 63;
  __hip_uint64_t mask = (1ULL << width) - 1;
  return ((src0 & ~(mask << offset)) | ((src1 & mask) << offset));
}

__device__ inline unsigned int __funnelshift_l(unsigned int lo, unsigned int hi,
                                               unsigned int shift) {
  __hip_uint32_t mask_shift = shift & 31;
  if (mask_shift == 0) return hi;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_alignbit))
    return __builtin_amdgcn_alignbit(hi, lo, 32 - mask_shift);
  __builtin_trap();
}

__device__ inline unsigned int __funnelshift_lc(unsigned int lo, unsigned int hi,
                                                unsigned int shift) {
  __hip_uint32_t min_shift = shift >= 32 ? 32 : shift;
  if (min_shift == 0) return hi;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_alignbit))
    return __builtin_amdgcn_alignbit(hi, lo, 32 - min_shift);
  __builtin_trap();
}

__device__ inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi,
                                               unsigned int shift) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_alignbit))
    return __builtin_amdgcn_alignbit(hi, lo, shift);
  __builtin_trap();
}

__device__ inline unsigned int __funnelshift_rc(unsigned int lo, unsigned int hi,
                                                unsigned int shift) {
  if (shift >= 32) return hi;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_alignbit))
    return __builtin_amdgcn_alignbit(hi, lo, shift);
  __builtin_trap();
}

__device__ static unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s);
__device__ static int __hadd(int x, int y);
__device__ static int __mul24(int x, int y);
__device__ static long long int __mul64hi(long long int x, long long int y);
__device__ static int __mulhi(int x, int y);
__device__ static int __rhadd(int x, int y);
__device__ static unsigned int __sad(int x, int y, unsigned int z);
__device__ static unsigned int __uhadd(unsigned int x, unsigned int y);
__device__ static int __umul24(unsigned int x, unsigned int y);
__device__ static unsigned long long int __umul64hi(unsigned long long int x,
                                                    unsigned long long int y);
__device__ static unsigned int __umulhi(unsigned int x, unsigned int y);
__device__ static unsigned int __urhadd(unsigned int x, unsigned int y);
__device__ static unsigned int __usad(unsigned int x, unsigned int y, unsigned int z);

struct ucharHolder {
  union {
    unsigned char c[4];
    unsigned int ui;
  };
} __attribute__((aligned(4)));

struct uchar2Holder {
  union {
    unsigned int ui[2];
    unsigned char c[8];
  };
} __attribute__((aligned(8)));

__device__ static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
  struct uchar2Holder cHoldVal;
  struct ucharHolder cHoldKey;
  cHoldKey.ui = s;
  cHoldVal.ui[0] = x;
  cHoldVal.ui[1] = y;
  unsigned int result;
  result = cHoldVal.c[cHoldKey.c[0] & 0x07];
  result += (cHoldVal.c[(cHoldKey.c[0] & 0x70) >> 4] << 8);
  result += (cHoldVal.c[cHoldKey.c[1] & 0x07] << 16);
  result += (cHoldVal.c[(cHoldKey.c[1] & 0x70) >> 4] << 24);
  return result;
}

__device__ static inline int __hadd(int x, int y) { return ((long long)x + (long long)y) >> 1; }

__device__ static inline int __mul24(int x, int y) { return __ockl_mul24_i32(x, y); }

__device__ static inline long long __mul64hi(long long int x, long long int y) {
  unsigned long long x0 = (unsigned long long)x & 0xffffffffUL;
  long long x1 = x >> 32;
  unsigned long long y0 = (unsigned long long)y & 0xffffffffUL;
  long long y1 = y >> 32;
  unsigned long long z0 = x0 * y0;
  long long t = x1 * y0 + (z0 >> 32);
  long long z1 = t & 0xffffffffL;
  long long z2 = t >> 32;
  z1 = x0 * y1 + z1;
  return x1 * y1 + z2 + (z1 >> 32);
}

__device__ static inline int __mulhi(int x, int y) { return (int)(((__hip_int64_t)x * (__hip_int64_t)y) >> 32); }

__device__ static inline int __rhadd(int x, int y) {
  return ((__hip_int64_t)x + (__hip_int64_t)y + 1) >> 1;
}

__device__ static inline unsigned int __sad(int x, int y, unsigned int z) {
  return x > y ? x - y + z : y - x + z;
}

__device__ static inline unsigned int __uhadd(unsigned int x, unsigned int y) {
  return ((unsigned long long)x + (unsigned long long)y) >> 1;
}

__device__ static inline int __umul24(unsigned int x, unsigned int y) {
  return __ockl_mul24_u32(x, y);
}

__device__ static inline unsigned long long __umul64hi(unsigned long long int x,
                                                       unsigned long long int y) {
  unsigned long long x0 = x & 0xffffffffUL;
  unsigned long long x1 = x >> 32;
  unsigned long long y0 = y & 0xffffffffUL;
  unsigned long long y1 = y >> 32;
  unsigned long long z0 = x0 * y0;
  unsigned long long t = x1 * y0 + (z0 >> 32);
  unsigned long long z1 = t & 0xffffffffUL;
  unsigned long long z2 = t >> 32;
  z1 = x0 * y1 + z1;
  return x1 * y1 + z2 + (z1 >> 32);
}

__device__ static inline unsigned int __umulhi(unsigned int x, unsigned int y) {
  return (unsigned int)(((__hip_uint64_t)x * (__hip_uint64_t)y) >> 32);
}

__device__ static inline unsigned int __urhadd(unsigned int x, unsigned int y) {
  return ((unsigned long long)x + (unsigned long long)y + 1) >> 1;
}

__device__ static inline unsigned int __usad(unsigned int x, unsigned int y, unsigned int z) {
  return __ockl_sadd_u32(x, y, z);
}

__device__ static inline unsigned int __mbcnt_lo(unsigned int x, unsigned int y) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_lo))
    return __builtin_amdgcn_mbcnt_lo(x, y);
  __builtin_trap();
};

__device__ static inline unsigned int __mbcnt_hi(unsigned int x, unsigned int y) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_hi))
    return __builtin_amdgcn_mbcnt_hi(x, y);
  __builtin_trap();
};

/*
HIP specific device functions
*/

#if !defined(__HIPCC_RTC__)
#include "amd_warp_functions.h"
#include "amd_warp_sync_functions.h"
#endif

#define MASK1 0x00ff00ff
#define MASK2 0xff00ff00

__device__ static inline char4 __hip_hc_add8pk(char4 in1, char4 in2) {
  char4 out;
  unsigned one1 = in1.w & MASK1;
  unsigned one2 = in2.w & MASK1;
  out.w = (one1 + one2) & MASK1;
  one1 = in1.w & MASK2;
  one2 = in2.w & MASK2;
  out.w = out.w | ((one1 + one2) & MASK2);
  return out;
}

__device__ static inline char4 __hip_hc_sub8pk(char4 in1, char4 in2) {
  char4 out;
  unsigned one1 = in1.w & MASK1;
  unsigned one2 = in2.w & MASK1;
  out.w = (one1 - one2) & MASK1;
  one1 = in1.w & MASK2;
  one2 = in2.w & MASK2;
  out.w = out.w | ((one1 - one2) & MASK2);
  return out;
}

__device__ static inline char4 __hip_hc_mul8pk(char4 in1, char4 in2) {
  char4 out;
  unsigned one1 = in1.w & MASK1;
  unsigned one2 = in2.w & MASK1;
  out.w = (one1 * one2) & MASK1;
  one1 = in1.w & MASK2;
  one2 = in2.w & MASK2;
  out.w = out.w | ((one1 * one2) & MASK2);
  return out;
}

__device__ static inline float __double2float_rd(double x) { return __ocml_cvtrtn_f32_f64(x); }
__device__ static inline float __double2float_rn(double x) { return x; }
__device__ static inline float __double2float_ru(double x) { return __ocml_cvtrtp_f32_f64(x); }
__device__ static inline float __double2float_rz(double x) { return __ocml_cvtrtz_f32_f64(x); }

__device__ static inline int __double2hiint(double x) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");

  int tmp[2];
  __builtin_memcpy(tmp, &x, sizeof(tmp));

  return tmp[1];
}
__device__ static inline int __double2loint(double x) {
  static_assert(sizeof(double) == 2 * sizeof(int), "");

  int tmp[2];
  __builtin_memcpy(tmp, &x, sizeof(tmp));

  return tmp[0];
}

__device__ static inline int __double2int_rd(double x) {
  return (int)__builtin_elementwise_floor(x);
}
__device__ static inline int __double2int_rn(double x) {
  return (int)__builtin_elementwise_rint(x);
}
__device__ static inline int __double2int_ru(double x) {
  return (int)__builtin_elementwise_ceil(x);
}
__device__ static inline int __double2int_rz(double x) { return (int)x; }

__device__ static inline long long int __double2ll_rd(double x) {
  return (long long)__builtin_elementwise_floor(x);
}
__device__ static inline long long int __double2ll_rn(double x) {
  return (long long)__builtin_elementwise_rint(x);
}
__device__ static inline long long int __double2ll_ru(double x) {
  return (long long)__builtin_elementwise_ceil(x);
}
__device__ static inline long long int __double2ll_rz(double x) { return (long long)x; }

__device__ static inline unsigned int __double2uint_rd(double x) {
  return (unsigned int)__builtin_elementwise_floor(x);
}
__device__ static inline unsigned int __double2uint_rn(double x) {
  return (unsigned int)__builtin_elementwise_rint(x);
}
__device__ static inline unsigned int __double2uint_ru(double x) {
  return (unsigned int)__builtin_elementwise_ceil(x);
}
__device__ static inline unsigned int __double2uint_rz(double x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __double2ull_rd(double x) {
  return (unsigned long long int)__builtin_elementwise_floor(x);
}
__device__ static inline unsigned long long int __double2ull_rn(double x) {
  return (unsigned long long int)__builtin_elementwise_rint(x);
}
__device__ static inline unsigned long long int __double2ull_ru(double x) {
  return (unsigned long long int)__builtin_elementwise_ceil(x);
}
__device__ static inline unsigned long long int __double2ull_rz(double x) {
  return (unsigned long long int)x;
}
__device__ static inline long long int __double_as_longlong(double x) {
  static_assert(sizeof(long long) == sizeof(double), "");

  long long tmp;
  __builtin_memcpy(&tmp, &x, sizeof(tmp));

  return tmp;
}

/*
__device__ unsigned short __float2half_rn(float x);
__device__ float __half2float(unsigned short);

The above device function are not a valid .
Use
__device__ __half __float2half_rn(float x);
__device__ float __half2float(__half);
from hip_fp16.h

CUDA implements half as unsigned short whereas, HIP doesn't.

*/

__device__ static inline int __float2int_rd(float x) { return (int)__builtin_elementwise_floor(x); }
__device__ static inline int __float2int_rn(float x) { return (int)__builtin_elementwise_rint(x); }
__device__ static inline int __float2int_ru(float x) { return (int)__builtin_elementwise_ceil(x); }
__device__ static inline int __float2int_rz(float x) { return (int)__builtin_elementwise_trunc(x); }

__device__ static inline long long int __float2ll_rd(float x) {
  return (long long int)__builtin_elementwise_floor(x);
}
__device__ static inline long long int __float2ll_rn(float x) {
  return (long long int)__builtin_elementwise_rint(x);
}
__device__ static inline long long int __float2ll_ru(float x) {
  return (long long int)__builtin_elementwise_ceil(x);
}
__device__ static inline long long int __float2ll_rz(float x) { return (long long int)x; }

__device__ static inline unsigned int __float2uint_rd(float x) {
  return (unsigned int)__builtin_elementwise_floor(x);
}
__device__ static inline unsigned int __float2uint_rn(float x) {
  return (unsigned int)__builtin_elementwise_rint(x);
}
__device__ static inline unsigned int __float2uint_ru(float x) {
  return (unsigned int)__builtin_elementwise_ceil(x);
}
__device__ static inline unsigned int __float2uint_rz(float x) { return (unsigned int)x; }

__device__ static inline unsigned long long int __float2ull_rd(float x) {
  return (unsigned long long int)__builtin_elementwise_floor(x);
}
__device__ static inline unsigned long long int __float2ull_rn(float x) {
  return (unsigned long long int)__builtin_elementwise_rint(x);
}
__device__ static inline unsigned long long int __float2ull_ru(float x) {
  return (unsigned long long int)__builtin_elementwise_ceil(x);
}
__device__ static inline unsigned long long int __float2ull_rz(float x) {
  return (unsigned long long int)x;
}

__device__ static inline int __float_as_int(float x) {
  static_assert(sizeof(int) == sizeof(float), "");

  int tmp;
  __builtin_memcpy(&tmp, &x, sizeof(tmp));

  return tmp;
}

__device__ static inline unsigned int __float_as_uint(float x) {
  static_assert(sizeof(unsigned int) == sizeof(float), "");

  unsigned int tmp;
  __builtin_memcpy(&tmp, &x, sizeof(tmp));

  return tmp;
}

__device__ static inline double __hiloint2double(int hi, int lo) {
  static_assert(sizeof(double) == sizeof(__hip_uint64_t), "");

  __hip_uint64_t tmp0 =
      (static_cast<__hip_uint64_t>(hi) << 32ull) | static_cast<__hip_uint32_t>(lo);
  double tmp1;
  __builtin_memcpy(&tmp1, &tmp0, sizeof(tmp0));

  return tmp1;
}

__device__ static inline double __int2double_rn(int x) { return (double)x; }

__device__ static inline float __int2float_rd(int x) { return __ocml_cvtrtn_f32_s32(x); }
__device__ static inline float __int2float_rn(int x) { return (float)x; }
__device__ static inline float __int2float_ru(int x) { return __ocml_cvtrtp_f32_s32(x); }
__device__ static inline float __int2float_rz(int x) { return __ocml_cvtrtz_f32_s32(x); }

__device__ static inline float __int_as_float(int x) {
  static_assert(sizeof(float) == sizeof(int), "");

  float tmp;
  __builtin_memcpy(&tmp, &x, sizeof(tmp));

  return tmp;
}

__device__ static inline double __ll2double_rd(long long int x) { return __ocml_cvtrtn_f64_s64(x); }
__device__ static inline double __ll2double_rn(long long int x) { return (double)x; }
__device__ static inline double __ll2double_ru(long long int x) { return __ocml_cvtrtp_f64_s64(x); }
__device__ static inline double __ll2double_rz(long long int x) { return __ocml_cvtrtz_f64_s64(x); }

__device__ static inline float __ll2float_rd(long long int x) { return __ocml_cvtrtn_f32_s64(x); }
__device__ static inline float __ll2float_rn(long long int x) { return (float)x; }
__device__ static inline float __ll2float_ru(long long int x) { return __ocml_cvtrtp_f32_s64(x); }
__device__ static inline float __ll2float_rz(long long int x) { return __ocml_cvtrtz_f32_s64(x); }

__device__ static inline double __longlong_as_double(long long int x) {
  static_assert(sizeof(double) == sizeof(long long), "");

  double tmp;
  __builtin_memcpy(&tmp, &x, sizeof(tmp));

  return tmp;
}

__device__ static inline double __uint2double_rn(unsigned int x) { return (double)x; }

__device__ static inline float __uint2float_rd(unsigned int x) { return __ocml_cvtrtn_f32_u32(x); }
__device__ static inline float __uint2float_rn(unsigned int x) { return (float)x; }
__device__ static inline float __uint2float_ru(unsigned int x) { return __ocml_cvtrtp_f32_u32(x); }
__device__ static inline float __uint2float_rz(unsigned int x) { return __ocml_cvtrtz_f32_u32(x); }

__device__ static inline float __uint_as_float(unsigned int x) {
  static_assert(sizeof(float) == sizeof(unsigned int), "");

  float tmp;
  __builtin_memcpy(&tmp, &x, sizeof(tmp));

  return tmp;
}

__device__ static inline double __ull2double_rd(unsigned long long int x) {
  return __ocml_cvtrtn_f64_u64(x);
}
__device__ static inline double __ull2double_rn(unsigned long long int x) { return (double)x; }
__device__ static inline double __ull2double_ru(unsigned long long int x) {
  return __ocml_cvtrtp_f64_u64(x);
}
__device__ static inline double __ull2double_rz(unsigned long long int x) {
  return __ocml_cvtrtz_f64_u64(x);
}

__device__ static inline float __ull2float_rd(unsigned long long int x) {
  return __ocml_cvtrtn_f32_u64(x);
}
__device__ static inline float __ull2float_rn(unsigned long long int x) { return (float)x; }
__device__ static inline float __ull2float_ru(unsigned long long int x) {
  return __ocml_cvtrtp_f32_u64(x);
}
__device__ static inline float __ull2float_rz(unsigned long long int x) {
  return __ocml_cvtrtz_f32_u64(x);
}

#if defined(__clang__) && defined(__HIP__)

// Clock functions
__device__ long long int __clock64();
__device__ long long int __clock();
__device__ long long int clock64();
__device__ long long int clock();
__device__ long long int wall_clock64();
// hip.amdgcn.bc - named sync
__device__ void __named_sync();

#ifdef __HIP_DEVICE_COMPILE__

// Clock function to return GPU core cycle count.
// GPU can change its core clock frequency at runtime. The maximum frequency can be queried
// through hipDeviceAttributeClockRate attribute.
__device__ inline __attribute((always_inline)) long long int __clock64() {
  return (long long int)__builtin_readcyclecounter();
}

__device__ inline __attribute((always_inline)) long long int __clock() { return __clock64(); }

// Clock function to return wall clock count at a constant frequency that can be queried
// through hipDeviceAttributeWallClockRate attribute.
__device__ inline __attribute__((always_inline)) long long int wall_clock64() {
  return (long long int)__builtin_readsteadycounter();
}

__device__ inline __attribute__((always_inline)) long long int clock64() { return __clock64(); }

__device__ inline __attribute__((always_inline)) long long int clock() { return __clock(); }

// hip.amdgcn.bc - named sync
__device__ inline void __named_sync() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier)) __builtin_amdgcn_s_barrier();
}

#endif  // __HIP_DEVICE_COMPILE__

// hip.amdgcn.bc - lanemask
__device__ inline __hip_uint64_t __lanemask_gt() {
  __hip_uint32_t lane = __lane_id();
  if (lane == 63) return 0;
  __hip_uint64_t ballot = __ballot64(1);
  __hip_uint64_t mask = (~((__hip_uint64_t)0)) << (lane + 1);
  return mask & ballot;
}

__device__ inline __hip_uint64_t __lanemask_lt() {
  __hip_uint32_t lane = __lane_id();
  __hip_int64_t ballot = __ballot64(1);
  __hip_uint64_t mask = ((__hip_uint64_t)1 << lane) - (__hip_uint64_t)1;
  return mask & ballot;
}

__device__ inline __hip_uint64_t __lanemask_eq() {
  __hip_uint32_t lane = __lane_id();
  __hip_int64_t mask = ((__hip_uint64_t)1 << lane);
  return mask;
}

// Memory Fence Functions
__device__ inline static void __threadfence() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "agent");
}

__device__ inline static void __threadfence_block() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "workgroup");
}

__device__ inline static void __threadfence_system() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}
__device__ inline static void __work_group_barrier(__cl_mem_fence_flags flags) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence) &&
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier)) {
    if (flags == (__CLK_GLOBAL_MEM_FENCE | __CLK_LOCAL_MEM_FENCE)) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    } else if (flags & (__CLK_GLOBAL_MEM_FENCE)) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "global");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "global");
    } else if (flags & (__CLK_LOCAL_MEM_FENCE)) {
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
      __builtin_amdgcn_s_barrier();
      __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
    } else {
      __builtin_amdgcn_s_barrier();
    }
  }
}

__device__ inline static void __barrier(int n) { __work_group_barrier((__cl_mem_fence_flags)n); }

__device__ inline __attribute__((convergent)) void __syncthreads() {
  __barrier(__CLK_GLOBAL_MEM_FENCE | __CLK_LOCAL_MEM_FENCE);
}

__device__ inline __attribute__((convergent)) int __syncthreads_count(int predicate) {
  return __ockl_wgred_add_i32(!!predicate);
}

__device__ inline __attribute__((convergent)) int __syncthreads_and(int predicate) {
  return __ockl_wgred_and_i32(!!predicate);
}

__device__ inline __attribute__((convergent)) int __syncthreads_or(int predicate) {
  return __ockl_wgred_or_i32(!!predicate);
}

// hip.amdgcn.bc - device routine
/*
  HW_ID Register bit structure for RDNA2 & RDNA3
  WAVE_ID     4:0     Wave id within the SIMD.
  SIMD_ID     9:8     SIMD_ID within the WGP: [0] = row, [1] = column.
  WGP_ID      13:10   Physical WGP ID.
  SA_ID       16      Shader Array ID
  SE_ID       20:18   Shader Engine the wave is assigned to for gfx11
  SE_ID       19:18   Shader Engine the wave is assigned to for gfx10
  DP_RATE     31:29   Number of double-precision float units per SIMD

  HW_ID Register bit structure for GCN and CDNA
  WAVE_ID     3:0     Wave buffer slot number. 0-9.
  SIMD_ID     5:4     SIMD which the wave is assigned to within the CU.
  PIPE_ID     7:6     Pipeline from which the wave was dispatched.
  CU_ID       11:8    Compute Unit the wave is assigned to.
  SH_ID       12      Shader Array (within an SE) the wave is assigned to.
  SE_ID       15:13   Shader Engine the wave is assigned to for gfx908, gfx90a
              14:13   Shader Engine the wave is assigned to for 942
  TG_ID       19:16   Thread-group ID
  VM_ID       23:20   Virtual Memory ID
  QUEUE_ID    26:24   Queue from which this wave was dispatched.
  STATE_ID    29:27   State ID (graphics only, not compute).
  ME_ID       31:30   Micro-engine ID.

  XCC_ID Register bit structure for 942/950
  XCC_ID      3:0     XCC the wave is assigned to.
 */

#if (defined (__GFX10__) || defined (__GFX11__) || defined(__GFX12__))
  #define HW_ID               23
  #define HW_ID_WGP_ID_OFFSET 10
  #define HW_ID_WGP_ID_SIZE    4
  #if (defined(__AMDGCN_CUMODE__))
    #define HW_ID_CU_ID_OFFSET  8
    #define HW_ID_CU_ID_SIZE    1
  #endif
  #define HW_ID_SA_ID_OFFSET  16
  #define HW_ID_SA_ID_SIZE     1
#else
  #define HW_ID               4
  #define HW_ID_CU_ID_SIZE    4
  #define HW_ID_CU_ID_OFFSET  8
#endif

#if (defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__GFX11__))
  #define HW_ID_SE_ID_SIZE    3
#elif (defined(__GFX12__))
  #define SE_HW_ID_SE_ID_SIZE 4
#else //4 SEs/XCC for gfx940-942
  #define HW_ID_SE_ID_SIZE    2
#endif
#if (defined(__GFX10__) || defined(__GFX11__))
  #define HW_ID_SE_ID_OFFSET  18
#elif defined(__GFX12__)
  #define RTN_GET_SE_HW_ID       0x87
  #define SE_HW_ID_SE_ID_OFFSET  0
#else
#define HW_ID_SE_ID_OFFSET 13
#endif

#if (defined(__gfx942__) || defined(__gfx950__))
#define __gfx94plus_clr__
#define XCC_ID 20
#define XCC_ID_XCC_ID_SIZE 4
#define XCC_ID_XCC_ID_OFFSET 0
#endif

#if !defined(__HIP_NO_IMAGE_SUPPORT) && defined(__gfx94plus_clr__)
  #define __HIP_NO_IMAGE_SUPPORT   1
#endif

/*
   Encoding of parameter bitmask
   HW_ID        5:0     HW_ID
   OFFSET       10:6    Range: 0..31
   SIZE         15:11   Range: 1..32
 */

#define GETREG_IMMED(SZ, OFF, REG) (((SZ) << 11) | ((OFF) << 6) | (REG))

__device__ inline unsigned assemble_smid(unsigned xcc, unsigned se, unsigned sa,
                                         unsigned wgp, unsigned cu,
                                         unsigned se_bits, unsigned sa_bits,
                                         unsigned wgp_bits, unsigned cu_bits)
{
  unsigned temp = xcc;
  temp = (temp << se_bits) | se;
  temp = (temp << sa_bits) | sa;
  temp = (temp << wgp_bits) | wgp;
  temp = (temp << cu_bits) | cu;
  return temp;
}

/*
  __smid returns the wave's assigned Compute Unit and Shader Engine.
  The Compute Unit, CU_ID returned in bits 3:0, and Shader Engine, SE_ID in bits 5:4.
  Note: the results vary over time.
  SZ minus 1 since SIZE is 1-based.
*/
__device__
inline
unsigned __smid(void)
{
  unsigned xcc{}, se{}, sa{}, wgp{}, cu{};
  unsigned se_bits{}, sa_bits{}, wgp_bits{}, cu_bits{};

  #if defined(__GFX12__)
    unsigned msg = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sendmsg_rtn)
                       ? __builtin_amdgcn_s_sendmsg_rtn(RTN_GET_SE_HW_ID)
                       : 0u;
    se       = (msg >> SE_HW_ID_SE_ID_OFFSET) & ((1 << SE_HW_ID_SE_ID_SIZE) - 1);
    se_bits  = SE_HW_ID_SE_ID_SIZE;

    unsigned hw = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
                      ? __builtin_amdgcn_s_getreg(GETREG_IMMED(31, 0, HW_ID))
                      : 0u;
    wgp      = (hw >> HW_ID_WGP_ID_OFFSET) & ((1 << HW_ID_WGP_ID_SIZE) - 1);
    sa       = (hw >> HW_ID_SA_ID_OFFSET) & ((1 << HW_ID_SA_ID_SIZE) - 1);
    wgp_bits = HW_ID_WGP_ID_SIZE;
    sa_bits  = HW_ID_SA_ID_SIZE;

  #elif defined(__GFX10__) || defined(__GFX11__)
    se   = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(HW_ID_SE_ID_SIZE - 1, HW_ID_SE_ID_OFFSET, HW_ID))
               : 0u;
    sa   = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(HW_ID_SA_ID_SIZE - 1, HW_ID_SA_ID_OFFSET, HW_ID))
               : 0u;
    wgp  = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(HW_ID_WGP_ID_SIZE - 1, HW_ID_WGP_ID_OFFSET, HW_ID))
               : 0u;
    se_bits  = HW_ID_SE_ID_SIZE;
    sa_bits  = HW_ID_SA_ID_SIZE;
    wgp_bits = HW_ID_WGP_ID_SIZE;
    #if defined(__AMDGCN_CUMODE__)
      cu = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID))
               : 0u;
      cu_bits = HW_ID_CU_ID_SIZE;
    #endif

  #elif defined(__gfx94plus_clr__)
    se   = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(HW_ID_SE_ID_SIZE - 1, HW_ID_SE_ID_OFFSET, HW_ID))
               : 0u;
    xcc  = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(XCC_ID_XCC_ID_SIZE - 1, XCC_ID_XCC_ID_OFFSET, XCC_ID))
               : 0u;
    cu   = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
               ? __builtin_amdgcn_s_getreg(
                     GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID))
               : 0u;
    se_bits  = HW_ID_SE_ID_SIZE;
    cu_bits  = HW_ID_CU_ID_SIZE;

  #else
    se = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
             ? __builtin_amdgcn_s_getreg(
                   GETREG_IMMED(HW_ID_SE_ID_SIZE - 1, HW_ID_SE_ID_OFFSET, HW_ID))
             : 0u;
    cu = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
             ? __builtin_amdgcn_s_getreg(
                   GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID))
             : 0u;
    se_bits = HW_ID_SE_ID_SIZE;
    cu_bits = HW_ID_CU_ID_SIZE;
  #endif

  return assemble_smid(xcc, se, sa, wgp, cu, se_bits, sa_bits, wgp_bits, cu_bits);
}

/**
 * Map HIP_DYNAMIC_SHARED to "extern __shared__" for compatibility with old HIP applications
 * To be removed in a future release.
 */
#define HIP_DYNAMIC_SHARED(type, var) extern __shared__ type var[];
#define HIP_DYNAMIC_SHARED_ATTRIBUTE

#endif  // defined(__clang__) && defined(__HIP__)

// rely on `__builtin_* functions for memcpy/memset
static inline __device__ void* __hip_hc_memcpy(void* dst, const void* src, size_t size) {
  return __builtin_memcpy(dst, src, size);
}

// change the value from unsigned char to int, what a builtin expects.
static inline __device__ void* __hip_hc_memset(void* dst, int val, size_t size) {
  return __builtin_memset(dst, val, size);
}

#ifndef __OPENMP_AMDGCN__
static inline __device__ void* memcpy(void* dst, const void* src, size_t size) {
  return __hip_hc_memcpy(dst, src, size);
}

static inline __device__ void* memset(void* ptr, int val, size_t size) {
  return __hip_hc_memset(ptr, val, size);
}
#endif  // !__OPENMP_AMDGCN__

#define HIP_IMPL_GENERATE_SCAN_FUNC(OP, TYPE_ALIAS, TYPE) \
  extern "C" __device__ __attribute__((const)) TYPE __ockl_wfscan_ ## OP ## _ ## TYPE_ALIAS(TYPE, bool);\
\
    template <bool Inclusive>\
    __device__ __forceinline__ TYPE scan_ ## OP(TYPE val)\
    {\
      return __ockl_wfscan_ ## OP ## _ ## TYPE_ALIAS(val, Inclusive);\
    }

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// Warp sync builtins (with explicit mask argument) introduced in ROCm 6.2 as a
// preview to allow end-users to adapt to the new interface involving 64-bit
// masks. These are enabled by default, and can be disabled by setting the macro
// "HIP_DISABLE_WARP_SYNC_BUILTINS". This arrangement also applies to the
// __activemask() builtin defined in amd_warp_functions.h.
#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

#if !defined(__HIPCC_RTC__)
#include "amd_warp_functions.h"
#include "amd_device_functions.h"
#include "hip_assert.h"
#include <functional>
#include <algorithm>
#endif

#pragma push_macro("MAYBE_UNDEF")
#if defined(__has_attribute) && __has_attribute(maybe_undef)
#define MAYBE_UNDEF __attribute__((maybe_undef))
#else
#define MAYBE_UNDEF
#endif

extern "C" __device__ __attribute__((const)) int __ockl_wfred_add_i32(int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_add_u32(unsigned int);
extern "C" __device__ __attribute__((const)) int __ockl_wfred_min_i32(int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_min_u32(unsigned int);
extern "C" __device__ __attribute__((const)) int __ockl_wfred_max_i32(int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_max_u32(unsigned int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_and_u32(unsigned int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_or_u32(unsigned int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_xor_u32(unsigned int);

#ifdef HIP_ENABLE_EXTRA_WARP_SYNC_TYPES
// this macro enable types that are not in CUDA
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_add_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_add_u64(
    unsigned long long);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_add_f32(float);
extern "C" __device__ __attribute__((const)) double __ockl_wfred_add_f64(double);

extern "C" __device__ __attribute__((const)) long long __ockl_wfred_min_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_min_u64(
    unsigned long long);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_min_f32(float);
extern "C" __device__ __attribute__((const)) double __ockl_wfred_min_f64(double);

extern "C" __device__ __attribute__((const)) long long __ockl_wfred_max_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_max_u64(
    unsigned long long);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_max_f32(float);
extern "C" __device__ __attribute__((const)) double __ockl_wfred_max_f64(double);

extern "C" __device__ __attribute__((const)) int __ockl_wfred_and_i32(int);
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_and_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_and_u64(
    unsigned long long);

extern "C" __device__ __attribute__((const)) int __ockl_wfred_or_i32(int);
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_or_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_or_u64(
    unsigned long long);

extern "C" __device__ __attribute__((const)) int __ockl_wfred_xor_i32(int);
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_xor_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_xor_u64(
    unsigned long long);

#endif

template <typename T> __device__ inline T __hip_readfirstlane(T val) {
  // In theory, behaviour is undefined when reading from a union member other
  // than the member that was last assigned to, but it works in practice because
  // we rely on the compiler to do the reasonable thing.
  union {
    unsigned long long l;
    T d;
  } u;
  u.d = val;
  // NOTE: The builtin returns int, so we first cast it to unsigned int and only
  // then extend it to 64 bits.
  unsigned long long lower = (unsigned)(
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_readfirstlane)
          ? __builtin_amdgcn_readfirstlane(u.l)
          : 0);
  unsigned long long upper = (unsigned)(
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_readfirstlane)
          ? __builtin_amdgcn_readfirstlane(u.l >> 32)
          : 0);
  u.l = (upper << 32) | lower;
  return u.d;
}

// When compiling for wave32 mode, ignore the upper half of the 64-bit mask.
#define __hip_adjust_mask_for_wave32(MASK)                                                         \
  do {                                                                                             \
    if (static_cast<int>(warpSize) == 32) MASK &= 0xFFFFFFFF;                                      \
  } while (0)

// We use a macro to expand each builtin into a waterfall that implements the
// mask semantics:
//
// 1. The mask argument may be divergent.
// 2. Each active thread must have its own bit set in its own mask value.
// 3. For a given mask value, all threads that are mentioned in the mask must
//    execute the same static instance of the builtin with the same mask.
// 4. The union of all mask values supplied at a static instance must be equal
//    to the activemask at the program point.
//
// Thus, the mask argument partitions the set of currently active threads in the
// wave into disjoint subsets that cover all active threads.
//
// Implementation notes:
// ---------------------
//
// We implement this as a waterfall loop that executes the builtin for each
// subset separately. The return value is a divergent value across the active
// threads. The value for inactive threads is defined by each builtin
// separately.
//
// As long as every mask value is non-zero, we don't need to check if a lane
// specifies itself in the mask; that is done by the later assertion where all
// chosen lanes must be in the chosen mask.

#define __hip_check_mask(MASK)                                                                     \
  do {                                                                                             \
    __hip_assert(MASK && "mask must be non-zero");                                                 \
    bool done = false;                                                                             \
    while (__any(!done)) {                                                                         \
      if (!done) {                                                                                 \
        auto chosen_mask = __hip_readfirstlane(MASK);                                              \
        if (MASK == chosen_mask) {                                                                 \
          __hip_assert(MASK == __ballot(true) &&                                                   \
                       "all threads specified in the mask"                                         \
                       " must execute the same operation with the same mask");                     \
          done = true;                                                                             \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } while (0)

#define __hip_do_sync(RETVAL, FUNC, MASK, ...)                                                     \
  do {                                                                                             \
    __hip_assert(MASK && "mask must be non-zero");                                                 \
    bool done = false;                                                                             \
    while (__any(!done)) {                                                                         \
      if (!done) {                                                                                 \
        auto chosen_mask = __hip_readfirstlane(MASK);                                              \
        if (MASK == chosen_mask) {                                                                 \
          __hip_assert(MASK == __ballot(true) &&                                                   \
                       "all threads specified in the mask"                                         \
                       " must execute the same operation with the same mask");                     \
          RETVAL = FUNC(__VA_ARGS__);                                                              \
          done = true;                                                                             \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } while (0)

__device__ inline void __syncwarp() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_wave_barrier))
    __builtin_amdgcn_wave_barrier();
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template <typename MaskT> __device__ inline void __syncwarp(MaskT mask) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_check_mask(mask);
  return __syncwarp();
}

// __all_sync, __any_sync, __ballot_sync

template <typename MaskT>
__device__ inline unsigned long long __ballot_sync(MaskT mask, int predicate) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __ballot(predicate) & mask;
}

template <typename MaskT> __device__ inline int __all_sync(MaskT mask, int predicate) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  return __ballot_sync(mask, predicate) == mask;
}

template <typename MaskT> __device__ inline int __any_sync(MaskT mask, int predicate) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  return __ballot_sync(mask, predicate) != 0;
}

// __match_any, __match_all and sync variants

template <typename T> __device__ inline unsigned long long __match_any(T value) {
  static_assert(
      (__hip_internal::is_integral<T>::value || __hip_internal::is_floating_point<T>::value) &&
          (sizeof(T) == 4 || sizeof(T) == 8),
      "T can be int, unsigned int, long, unsigned long, long long, unsigned "
      "long long, float or double.");
  bool done = false;
  unsigned long long retval = 0;

  while (__any(!done)) {
    if (!done) {
      T chosen = __hip_readfirstlane(value);
      if (chosen == value) {
        retval = __activemask();
        done = true;
      }
    }
  }

  return retval;
}

template <typename MaskT, typename T>
__device__ inline unsigned long long __match_any_sync(MaskT mask, T value) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __match_any(value) & mask;
}

template <typename T> __device__ inline unsigned long long __match_all(T value, int* pred) {
  static_assert(
      (__hip_internal::is_integral<T>::value || __hip_internal::is_floating_point<T>::value) &&
          (sizeof(T) == 4 || sizeof(T) == 8),
      "T can be int, unsigned int, long, unsigned long, long long, unsigned "
      "long long, float or double.");
  T first = __hip_readfirstlane(value);
  if (__all(first == value)) {
    *pred = true;
    return __activemask();
  } else {
    *pred = false;
    return 0;
  }
}

template <typename MaskT, typename T>
__device__ inline unsigned long long __match_all_sync(MaskT mask, T value, int* pred) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  MaskT retval = 0;
  __hip_adjust_mask_for_wave32(mask);
  __hip_do_sync(retval, __match_all, mask, value, pred);
  return retval;
}

// various variants of shfl

template <typename MaskT, typename T>
__device__ inline T __shfl_sync(MaskT mask, MAYBE_UNDEF T var, int srcLane, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl(var, srcLane, width);
}

template <typename MaskT, typename T>
__device__ inline T __shfl_up_sync(MaskT mask, MAYBE_UNDEF T var, unsigned int delta, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_up(var, delta, width);
}

template <typename MaskT, typename T>
__device__ inline T __shfl_down_sync(MaskT mask, MAYBE_UNDEF T var, unsigned int delta, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_down(var, delta, width);
}

template <typename MaskT, typename T>
__device__ inline T __shfl_xor_sync(MaskT mask, MAYBE_UNDEF T var, int laneMask, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_xor(var, laneMask, width);
}

template <typename MaskT, typename T, typename BinaryOp, typename WfReduce>
__device__ inline T __reduce_op_sync(MaskT mask, T val, BinaryOp op, WfReduce wfReduce) {
  static constexpr bool isPrimitiveType = !__hip_internal::is_same<WfReduce, decltype(nullptr)>::value;
  using permuteType =
      typename __hip_internal::conditional<isPrimitiveType && (sizeof(T) == 4 || sizeof(T) == 2), T, unsigned int>::type;

  // the number of backward permutes will be: size(T) / 4 rounded up
  static constexpr int kNumOfPermutes = (sizeof(T) < 4)?
                                        1 :
                                        (sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
  static constexpr auto kMaskNumBits = sizeof(MaskT) * 8;
  static constexpr auto alignment = alignof(T) <= 4? 4 : alignof(T);
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  unsigned int laneId;
  unsigned int maskIdx;
  // next bit to aggregate with
  int nextBit;

  // if doing the binary reduction tree, this will increase by two in every iteration
  int modulo = 1;
  int leadingZeroes = __clzll(mask);
  int firstLane;
  int lastLane = kMaskNumBits - leadingZeroes - 1;
  int maskNumBits;
  int numIterations;

  // unsigned int[N] is used in some cases, e.g. when T is wider than 32-bit
  using ResultType = typename __hip_internal::conditional<
                       isPrimitiveType && (sizeof(T) == 4 || sizeof(T) == 2),
                       permuteType, permuteType[kNumOfPermutes]>::type;
  alignas(alignment) ResultType result;
  alignas(alignment) ResultType permuteResult;
  auto backwardPermute = [](int index, permuteType arg) {
    if constexpr (__hip_internal::is_floating_point<T>::value &&
                  sizeof(T) <= 4) {
      return __hip_ds_bpermutef(index, arg);
    } else {
      return __hip_ds_bpermute(index, arg);
    }
  };

  __hip_check_mask(mask);
  maskNumBits = __popcll(mask);

#ifdef __OPTIMIZE__  // at the time of this writing the ockl wfred functions do not compile when
                     // using -O0
  if constexpr (!__hip_internal::is_same<WfReduce, decltype(nullptr)>::value){
    if (maskNumBits == lastLane + 1)
      // this means the mask "does not have holes", and starts from 0; we can use a specific intrinsic
      // to calculate the aggregated result
      return wfReduce(val);
  }
#endif

  firstLane = __builtin_ctzll(mask);
  laneId = __lane_id();
  nextBit = laneId;
  // the number of iterations needs to be at least log2(number of bits on)
  numIterations = sizeof(int) * 8 - __clz(maskNumBits);

  if (!(maskNumBits & (maskNumBits - 1)))
    // the number of bits in the mask is a power of 2
    numIterations -= 1;

  maskIdx = __popcll(((1ul << laneId) - 1) & mask);
  mask >>= laneId;
  mask >>= 1ul;

  if constexpr (isPrimitiveType && (sizeof(T) == 2 || sizeof(T) == 4)) { 
    result = val;
  } else {
    __builtin_memcpy(result, &val, sizeof(T));
  }

  // add the values from the lanes using a reduction tree (first the threads with even-numbered
  // lanes, then multiples of 4, then 8, ...
  while (numIterations) {
    int offset = modulo >> 1;
    int increment = modulo - offset;
    int nextPos = maskIdx + offset + increment;
    bool insideLanes = nextPos < maskNumBits;

    if (insideLanes) {
      int next;

      // find the position to aggregate with; although we could just call fns64() that will probably
      // be very slow when called multiple times in this for loop; this is equivalent
      for (int i = 0; i < increment; i++) {
        next = __builtin_ctzll(mask) + 1;
        mask >>= next;
        nextBit += next;
      }
    }

    if constexpr (!isPrimitiveType) {
      for (int i = 0; i < kNumOfPermutes; i++) {
        permuteResult[i] = backwardPermute(nextBit << 2, result[i]);
      }
    } else if constexpr (sizeof(T) == 2) {
      union {
        int i;
        T f;
      } tmp;

      tmp.f = result;
      tmp.i = __hip_ds_bpermute(nextBit << 2, tmp.i);
      permuteResult = tmp.f;
    } else if constexpr (sizeof(T) == 4) {
      auto bPermuteResult = backwardPermute(nextBit << 2, result);
      __builtin_memcpy(&permuteResult, &bPermuteResult, sizeof(result));
    } else {
      // ds_bpermute only deals with 32-bit sizes, so for other sizes
      // we need to call the permute multiple times
      permuteResult[0] = backwardPermute(nextBit << 2, result[0]);
      permuteResult[1] = backwardPermute(nextBit << 2, result[1]);
    }

    if (insideLanes) {
      if constexpr (__hip_internal::is_same<WfReduce, decltype(nullptr)>::value) {
        T toReturn;
        toReturn = op(*reinterpret_cast<T*>(result), *reinterpret_cast<T*>(permuteResult));
        __builtin_memcpy(result, &toReturn, sizeof(T));
      } else if constexpr (sizeof(T) == 4 || sizeof(T) == 2)
        result = op(result, permuteResult);
      else if constexpr (sizeof(T) == 8) {
        T tmp;
        unsigned long long rhs =
            (static_cast<unsigned long long>(permuteResult[1]) << 32) | permuteResult[0];

        __builtin_memcpy(&tmp, result, sizeof(result));
        tmp = op(tmp, *reinterpret_cast<T*>(&rhs));
        __builtin_memcpy(result, &tmp, sizeof(result));
      }
    }

    modulo <<= 1;
    numIterations--;
  }

  // get result from the first active lane
  if constexpr (!isPrimitiveType) {
    // user defined types
    for (int i = 0; i < kNumOfPermutes; i++) {
      permuteResult[i] = backwardPermute(firstLane << 2, result[i]);
    }

    return *reinterpret_cast<T*>(permuteResult);
  } else if constexpr (sizeof(T) == 2) {
    union {
      int i;
      T f;
    } tmp;
    tmp.f = result;
    tmp.i = __hip_ds_bpermute(firstLane << 2, tmp.i);
    return tmp.f;
  } else if constexpr (sizeof(T) == 4) {
    return backwardPermute(firstLane << 2, result);
  } else {
    auto tmp = (static_cast<unsigned long long>(backwardPermute(firstLane << 2, result[1])) << 32) |
               static_cast<unsigned int>(backwardPermute(firstLane << 2, result[0]));
    return *reinterpret_cast<T*>(&tmp);
  }
}

// Compiler now provides builtins to optimized code for reduce device functions
// Checking for one, they all were added at the same time.
// Quite a lot of our work should be offloaded now, all we need to do is validations
#if __has_builtin(__builtin_amdgcn_wave_reduce_add_u32)
#define HIP_WARP_REDUCE_COMPILER_BUILTINS

template <
    typename MaskT, typename ValueT,
    typename __hip_internal::enable_if<__hip_internal::is_same<ValueT, int>::value ||
                                           __hip_internal::is_same<ValueT, unsigned int>::value,
                                       int>::type = 0>
static __device__ ValueT __reduce_add_sync_builtins_variant(MaskT mask, ValueT val) {
  // Mask size checks
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");

  // Adjust mask for 32 warp size
  __hip_adjust_mask_for_wave32(mask);

  unsigned int ret{};
  __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_add_u32, mask, val, 0);
  // the builtin is unsigned but it should work fine for signed as well
  return static_cast<ValueT>(ret);
}

template <
    typename MaskT, typename ValueT,
    typename __hip_internal::enable_if<__hip_internal::is_same<ValueT, int>::value ||
                                           __hip_internal::is_same<ValueT, unsigned int>::value,
                                       int>::type = 0>
static __device__ ValueT __reduce_min_sync_builtins_variant(MaskT mask, ValueT val) {
  // Mask size checks
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");

  // Adjust mask for 32 warp size
  __hip_adjust_mask_for_wave32(mask);

  ValueT ret{};
  if constexpr (__hip_internal::is_same<ValueT, int>::value) {
    __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_min_i32, mask, val, 0);
  } else {
    __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_min_u32, mask, val, 0);
  }
  return ret;
}

template <
    typename MaskT, typename ValueT,
    typename __hip_internal::enable_if<__hip_internal::is_same<ValueT, int>::value ||
                                           __hip_internal::is_same<ValueT, unsigned int>::value,
                                       int>::type = 0>
static __device__ ValueT __reduce_max_sync_builtins_variant(MaskT mask, ValueT val) {
  // Mask size checks
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");

  // Adjust mask for 32 warp size
  __hip_adjust_mask_for_wave32(mask);

  ValueT ret{};
  if constexpr (__hip_internal::is_same<ValueT, int>::value) {
    __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_max_i32, mask, val, 0);
  } else {
    __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_max_u32, mask, val, 0);
  }
  return ret;
}


template <typename MaskT, typename ValueT,
          typename __hip_internal::enable_if<__hip_internal::is_same<ValueT, unsigned int>::value,
                                             int>::type = 0>
static __device__ ValueT __reduce_and_sync_builtins_variant(MaskT mask, ValueT val) {
  // Mask size checks
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");

  // Adjust mask for 32 warp size
  __hip_adjust_mask_for_wave32(mask);

  ValueT ret{};
  __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_and_b32, mask, val, 0);
  return ret;
}

template <typename MaskT, typename ValueT,
          typename __hip_internal::enable_if<__hip_internal::is_same<ValueT, unsigned int>::value,
                                             int>::type = 0>
static __device__ ValueT __reduce_or_sync_builtins_variant(MaskT mask, ValueT val) {
  // Mask size checks
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");

  // Adjust mask for 32 warp size
  __hip_adjust_mask_for_wave32(mask);

  ValueT ret{};
  __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_or_b32, mask, val, 0);
  return ret;
}

template <typename MaskT, typename ValueT,
          typename __hip_internal::enable_if<__hip_internal::is_same<ValueT, unsigned int>::value,
                                             int>::type = 0>
static __device__ ValueT __reduce_xor_sync_builtins_variant(MaskT mask, ValueT val) {
  // Mask size checks
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");

  // Adjust mask for 32 warp size
  __hip_adjust_mask_for_wave32(mask);

  ValueT ret{};
  __hip_do_sync(ret, __builtin_amdgcn_wave_reduce_xor_b32, mask, val, 0);
  return ret;
}
#endif

template <typename MaskT> __device__ inline int __reduce_add_sync(MaskT mask, int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_add_sync_builtins_variant(mask, val);
#else
  // although C++ has std::plus and other functors, we do not use them because
  // they are in the header <functional> and they were causing problems with hipRTC
  // at this time
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT>
__device__ inline unsigned int __reduce_add_sync(MaskT mask, unsigned int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_add_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT> __device__ inline int __reduce_min_sync(MaskT mask, int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_min_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT>
__device__ inline unsigned int __reduce_min_sync(MaskT mask, unsigned int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_min_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT> __device__ inline int __reduce_max_sync(MaskT mask, int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_max_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT>
__device__ inline unsigned int __reduce_max_sync(MaskT mask, unsigned int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_max_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT>
__device__ inline unsigned int __reduce_or_sync(MaskT mask, unsigned int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_or_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs | rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT>
__device__ inline unsigned int __reduce_and_sync(MaskT mask, unsigned int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_and_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs & rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

template <typename MaskT>
__device__ inline unsigned int __reduce_xor_sync(MaskT mask, unsigned int val) {
#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
  return __reduce_xor_sync_builtins_variant(mask, val);
#else
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs ^ rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
#endif
}

#ifdef HIP_ENABLE_EXTRA_WARP_SYNC_TYPES
template <typename MaskT> __device__ inline long long __reduce_add_sync(MaskT mask, long long val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_add_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline float __reduce_add_sync(MaskT mask, float val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_f32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline double __reduce_add_sync(MaskT mask, double val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_f64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_min_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_min_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline float __reduce_min_sync(MaskT mask, float val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_f32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline double __reduce_min_sync(MaskT mask, double val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_f64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_max_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_max_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline float __reduce_max_sync(MaskT mask, float val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_f32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline double __reduce_max_sync(MaskT mask, double val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_f64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_and_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs & rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_and_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs & rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_and_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs & rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_or_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs | rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_or_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs | rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_or_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs | rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_xor_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs ^ rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_xor_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs ^ rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_xor_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs ^ rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

#undef __hip_do_sync
#undef __hip_check_mask
#undef __hip_adjust_mask_for_wave32

#endif  // HIP_ENABLE_EXTRA_WARP_SYNC_TYPES

#if defined(HIP_WARP_REDUCE_COMPILER_BUILTINS)
#undef HIP_WARP_REDUCE_COMPILER_BUILTINS
#endif

#pragma pop_macro("MAYBE_UNDEF")

#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  amd_detail/hip_cooperative_groups_helper.h
 *
 *  @brief Device side implementation of cooperative group feature.
 *
 *  Defines helper constructs and APIs which aid the types and device API
 *  wrappers defined within `amd_detail/hip_cooperative_groups.h`.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h>  // threadId, blockId
#include <hip/amd_detail/amd_device_functions.h>
#endif
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

#if !defined(__CG_QUALIFIER__)
#define __CG_QUALIFIER__ __device__ __forceinline__
#endif

#if !defined(__CG_STATIC_QUALIFIER__)
#define __CG_STATIC_QUALIFIER__ __device__ static __forceinline__
#endif

#if !defined(_CG_STATIC_CONST_DECL_)
#define _CG_STATIC_CONST_DECL_ static constexpr
#endif

using lane_mask = unsigned long long int;
namespace cooperative_groups {

/* Global scope */
template <unsigned int size> using is_power_of_2 =
    __hip_internal::integral_constant<bool, (size & (size - 1)) == 0>;

template <unsigned int size> using is_valid_wavefront =
    __hip_internal::integral_constant<bool, size <= 64>;

template <unsigned int size> using is_valid_tile_size =
    __hip_internal::integral_constant<bool, is_power_of_2<size>::value &&
                                                is_valid_wavefront<size>::value>;

template <typename T> using is_valid_type =
    __hip_internal::integral_constant<bool, __hip_internal::is_integral<T>::value ||
                                                __hip_internal::is_floating_point<T>::value>;

namespace internal {

/**
 * @brief Enums representing different cooperative group types
 * @note  This enum is only applicable on Linux.
 *
 */
typedef enum {
  cg_invalid,
  cg_multi_grid,
  cg_grid,
  cg_workgroup,
  cg_tiled_group,
  cg_coalesced_group
} group_type;
/**
 *  @ingroup CooperativeG
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  @note  Cooperative groups feature is implemented on Linux, under developement
 *  on Windows.
 *
 */
namespace helper {
/**
 * @brief Create output mask from input_mask at places where base_mask is set
 *
 * Example: base_mask = 0101'0101, input_mask = 1111'0000
 * Output mask: 1100
 * Explaination:
 *           | | | |  | | | |
 * base:    0|1|0|1|'0|1|0|1|   // Which bits are set
 * input:   1|1|1|1|'0|0|0|0|   // Which values are picked
 *           | | | |  | | | |
 * output:    1   1    0   0
 */
__CG_STATIC_QUALIFIER__ unsigned long long adjust_mask(unsigned long long base_mask,
                                                       unsigned long long input_mask) {
  unsigned long long out = 0;
  for (unsigned int i = 0, index = 0; i < warpSize; i++) {
    auto lane_active = base_mask & (1ull << i);
    if (lane_active) {
      auto result = input_mask & (1ull << i);
      out |= ((result ? 1ull : 0ull) << index);
      index++;
    }
  }
  return out;
}
}  // namespace helper
/**
 *
 * @brief  Functionalities related to multi-grid cooperative group type
 * @note  The following cooperative groups functions are only applicable on Linux.
 *
 */
namespace multi_grid {

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_grids() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_num_grids());
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t grid_rank() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_grid_rank());
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_size());
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
  return static_cast<__hip_uint32_t>(__ockl_multi_grid_thread_rank());
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_multi_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_multi_grid_sync(); }

}  // namespace multi_grid

/**
 *  @brief Functionalities related to grid cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace grid {

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
  return static_cast<__hip_uint32_t>((blockDim.z * gridDim.z) * (blockDim.y * gridDim.y) *
                                     (blockDim.x * gridDim.x));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
  // Compute global id of the workgroup to which the current thread belongs to
  __hip_uint32_t blkIdx = static_cast<__hip_uint32_t>((blockIdx.z * gridDim.y * gridDim.x) +
                                                      (blockIdx.y * gridDim.x) + (blockIdx.x));

  // Compute total number of threads being passed to reach current workgroup
  // within grid
  __hip_uint32_t num_threads_till_current_workgroup =
      static_cast<__hip_uint32_t>(blkIdx * (blockDim.x * blockDim.y * blockDim.z));

  // Compute thread local rank within current workgroup
  __hip_uint32_t local_thread_rank = static_cast<__hip_uint32_t>(
      (threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + (threadIdx.x));

  return (num_threads_till_current_workgroup + local_thread_rank);
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t block_rank() {
  return static_cast<__hip_uint32_t>((blockIdx.z * gridDim.y * gridDim.x) +
                                     (blockIdx.y * gridDim.x) + (blockIdx.x));
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_grid_sync(); }

__CG_STATIC_QUALIFIER__ dim3 grid_dim() {
  return (dim3(static_cast<__hip_uint32_t>(gridDim.x), static_cast<__hip_uint32_t>(gridDim.y),
               static_cast<__hip_uint32_t>(gridDim.z)));
}

__CG_STATIC_QUALIFIER__ unsigned int barrier_arrive() { return __ockl_grid_bar_arrive(); }

__CG_STATIC_QUALIFIER__ unsigned int barrier_signal() { return __ockl_grid_bar_arrive(); }

__CG_STATIC_QUALIFIER__ void barrier_wait(unsigned int s) { __ockl_grid_bar_wait(s); }
}  // namespace grid

/**
 *  @brief Functionalities related to `workgroup` (thread_block in CUDA terminology)
 *  cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace workgroup {

__CG_STATIC_QUALIFIER__ dim3 group_index() {
  return (dim3(static_cast<__hip_uint32_t>(blockIdx.x), static_cast<__hip_uint32_t>(blockIdx.y),
               static_cast<__hip_uint32_t>(blockIdx.z)));
}

__CG_STATIC_QUALIFIER__ dim3 thread_index() {
  return (dim3(static_cast<__hip_uint32_t>(threadIdx.x), static_cast<__hip_uint32_t>(threadIdx.y),
               static_cast<__hip_uint32_t>(threadIdx.z)));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
  return (static_cast<__hip_uint32_t>(blockDim.x * blockDim.y * blockDim.z));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
  return (static_cast<__hip_uint32_t>((threadIdx.z * blockDim.y * blockDim.x) +
                                      (threadIdx.y * blockDim.x) + (threadIdx.x)));
}

__CG_STATIC_QUALIFIER__ __hip_uint32_t block_rank() {
  return (static_cast<__hip_uint32_t>((blockIdx.z * gridDim.x * gridDim.y) +
                                      (blockIdx.y * gridDim.x) + (blockIdx.x)));
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return true; }

__CG_STATIC_QUALIFIER__ void sync() { __syncthreads(); }

__CG_STATIC_QUALIFIER__ dim3 block_dim() {
  return (dim3(static_cast<__hip_uint32_t>(blockDim.x), static_cast<__hip_uint32_t>(blockDim.y),
               static_cast<__hip_uint32_t>(blockDim.z)));
}

__CG_STATIC_QUALIFIER__ void barrier_arrive() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_signal))
    __builtin_amdgcn_s_barrier_signal(-1);  // -1 is workgroup barriers
}

__CG_STATIC_QUALIFIER__ void barrier_wait() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_wait))
    __builtin_amdgcn_s_barrier_wait(-1);
  else if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier))
    __builtin_amdgcn_s_barrier();
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}
}  // namespace workgroup

namespace tiled_group {

// enforce ordering for memory instructions
__CG_STATIC_QUALIFIER__ void sync() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
}

}  // namespace tiled_group

namespace coalesced_group {

// enforce ordering for memory instructions
__CG_STATIC_QUALIFIER__ void sync() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "wavefront");
}

// Masked bit count
//
// For each thread, this function returns the number of active threads which
// have i-th bit of x set and come before the current thread.
__CG_STATIC_QUALIFIER__ unsigned int masked_bit_count(lane_mask x, unsigned int add = 0) {
  unsigned int counter = 0;
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_lo)) __builtin_trap();
  if (static_cast<int>(warpSize) == 32) {
    counter = __builtin_amdgcn_mbcnt_lo(static_cast<unsigned int>(x), add);
  } else {
    unsigned int lo = static_cast<unsigned int>(x & 0xFFFFFFFF);
    unsigned int hi = static_cast<unsigned int>((x >> 32) & 0xFFFFFFFF);
    counter = __builtin_amdgcn_mbcnt_lo(lo, add);
    counter = __builtin_amdgcn_is_invocable(__builtin_amdgcn_mbcnt_hi)
                  ? __builtin_amdgcn_mbcnt_hi(hi, counter)
                  : counter;
  }

  return counter;
}

}  // namespace coalesced_group

namespace cluster {
__CG_STATIC_QUALIFIER__ void sync() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_cluster_barrier))
    // Generates a signal + wait combination for cluster barrier
    __builtin_amdgcn_s_cluster_barrier();
  else if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier))
    __builtin_amdgcn_s_barrier();  // fallback to s_barrier if device does not support clusters
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");
}

__CG_STATIC_QUALIFIER__ void barrier_arrive() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_signal) &&
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_wait)) {
    bool isfirst = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_signal_isfirst)
                       ? __builtin_amdgcn_s_barrier_signal_isfirst(-1)  // -1 is workgroup barrier
                       : false;
    __builtin_amdgcn_s_barrier_wait(-1);

    if (isfirst) {
      // Signal the cluster barrier, -3 means user cluster barrier
      __builtin_amdgcn_s_barrier_signal(-3);
    }
  }
}

__CG_STATIC_QUALIFIER__ void barrier_wait() {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier_wait))
    // wait on the cluster barrier, -3 means user cluster barrier
    __builtin_amdgcn_s_barrier_wait(-3);
  else if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_barrier))
    __builtin_amdgcn_s_barrier();  // Fall back to s_barrier
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fence))
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");
}

__CG_STATIC_QUALIFIER__ dim3 block_index() {
  return dim3(__builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_id_x)
                  ? __builtin_amdgcn_cluster_workgroup_id_x()
                  : 0,
              __builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_id_y)
                  ? __builtin_amdgcn_cluster_workgroup_id_y()
                  : 0,
              __builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_id_z)
                  ? __builtin_amdgcn_cluster_workgroup_id_z()
                  : 0);
}

__CG_STATIC_QUALIFIER__ dim3 dim_blocks() {
  return dim3((__builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_max_id_x)
                   ? __builtin_amdgcn_cluster_workgroup_max_id_x()
                   : 0) +
                  1,
              (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_max_id_y)
                   ? __builtin_amdgcn_cluster_workgroup_max_id_y()
                   : 0) +
                  1,
              (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_max_id_z)
                   ? __builtin_amdgcn_cluster_workgroup_max_id_z()
                   : 0) +
                  1);
}

__CG_STATIC_QUALIFIER__ unsigned int block_rank() {
  auto idx = block_index();
  auto dim = dim_blocks();
  return idx.x + idx.y * dim.x + idx.z * dim.x * dim.y;
}

__CG_STATIC_QUALIFIER__ dim3 thread_index() {
  const dim3 blockIndex = block_index();
  return dim3(blockIndex.x * blockDim.x + threadIdx.x, blockIndex.y * blockDim.y + threadIdx.y,
              blockIndex.z * blockDim.z + threadIdx.z);
}

__CG_STATIC_QUALIFIER__ unsigned int num_blocks() {
  return (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cluster_workgroup_max_flat_id)
              ? __builtin_amdgcn_cluster_workgroup_max_flat_id()
              : 0) +
         1;
}

__CG_STATIC_QUALIFIER__ dim3 dim_threads() {
  const dim3 dimBlocks = dim_blocks();
  const unsigned int x = dimBlocks.x * blockDim.x;
  const unsigned int y = dimBlocks.y * blockDim.y;
  const unsigned int z = dimBlocks.z * blockDim.z;
  return dim3(x, y, z);
}

__CG_STATIC_QUALIFIER__ unsigned int num_threads() {
  auto d = dim_threads();
  return d.x * d.y * d.z;
}

__CG_STATIC_QUALIFIER__ unsigned int thread_rank() {
  return block_rank() * (blockDim.x * blockDim.y * blockDim.z) +
      ((threadIdx.z * blockDim.y * blockDim.x) + (threadIdx.y * blockDim.x) + threadIdx.x);
}

template <typename T> __CG_STATIC_QUALIFIER__ T* map_shared_rank(T* in, int rank) {
#if __has_builtin(__builtin_amdgcn_map_shared_rank)
  return (T*)(__builtin_amdgcn_is_invocable(__builtin_amdgcn_map_shared_rank)
                  ? __builtin_amdgcn_map_shared_rank((void*)in, rank)
                  : nullptr);
#else
  return nullptr;
#endif
}

__CG_STATIC_QUALIFIER__ unsigned int query_shared_rank(const void* in) {
#if __has_builtin(__builtin_amdgcn_query_shared_rank)
  if (!__builtin_amdgcn_is_invocable(__builtin_amdgcn_query_shared_rank)) __builtin_trap();
  return static_cast<unsigned int>(
      __builtin_amdgcn_query_shared_rank((__attribute__((address_space(11))) const void*)in));
#else
  return 0;
#endif
}
}  // namespace cluster
}  // namespace internal
}  // namespace cooperative_groups
/**
 *  @}
 */

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 *  @file  amd_detail/amd_hip_cooperative_groups.h
 *
 *  \brief Device side implementation of `Cooperative Group` feature.
 *
 *  Defines new types and device API wrappers related to `Cooperative Group`
 *  feature, which the programmer can directly use in his kernel(s) in order to
 *  make use of this feature.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/hip_cooperative_groups_helper.h>
#endif

namespace cooperative_groups {
  class thread_group;
namespace impl {
  template <typename TyGroup>
__CG_QUALIFIER__ unsigned long long groupMask(const TyGroup&);
}

/** \brief The base type of all cooperative group types.
 *
 *  \details Holds the key properties of a constructed cooperative group types
 *           object, like the group type, its size, etc.
 *
 *  \note  Cooperative groups feature is implemented on Linux, under development
 *         on Microsoft Windows.
 */
class thread_group {
  template <typename TyGroup>
  friend __CG_QUALIFIER__ unsigned long long cooperative_groups::impl::groupMask(const TyGroup&);
 protected:
  __hip_uint32_t _type;         //! Type of the thread_group.
  __hip_uint32_t _num_threads;  //! Total number of threads in the thread_group.
  __hip_uint64_t _mask;         //! Lanemask for coalesced and tiled partitioned group types,
                                //! LSB represents lane 0, and MSB represents lane 63
  //! Construct a thread group, and set thread group type and other essential
  //! thread group properties. This generic thread group is directly constructed
  //! only when the group is supposed to contain only the calling thread
  //! (through the API - `this_thread()`), and in all other cases, this thread
  //! group object is a sub-object of some other derived thread group object.
  __CG_QUALIFIER__ thread_group(internal::group_type type,
                                __hip_uint32_t num_threads = static_cast<__hip_uint64_t>(0),
                                __hip_uint64_t mask = static_cast<__hip_uint64_t>(0)) {
    _type = type;
    _num_threads = num_threads;
    _mask = mask;
  }

  struct _tiled_info {
    bool is_tiled;
    unsigned int num_threads;
    unsigned int meta_group_rank;
    unsigned int meta_group_size;
  };

  struct _coalesced_info {
    lane_mask member_mask;
    unsigned int num_threads;
    struct _tiled_info tiled_info;
  } coalesced_info;

  friend __CG_QUALIFIER__ thread_group this_thread();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend class thread_block;

 public:
  //! Total number of threads in the thread_group, and this serves the purpose
  //! for all derived cooperative group types because their `num_threads` is directly
  //! saved during the construction.
  __CG_QUALIFIER__ __hip_uint32_t num_threads() const { return _num_threads; }
  //! Total number of threads in the group (alias of num_threads())
  __CG_QUALIFIER__ __hip_uint32_t size() const { return num_threads(); }
  //! Returns the type of the group.
  __CG_QUALIFIER__ unsigned int cg_type() const { return _type; }
  //! Rank of the calling thread within [0, \link num_threads() num_threads() \endlink).
  __CG_QUALIFIER__ __hip_uint32_t thread_rank() const;
  //! Rank of the block in calling thread within [0, \link num_threads() num_threads() \endlink).
  __CG_QUALIFIER__ __hip_uint32_t block_rank() const;
  //! Returns true if the group has not violated any API constraints.
  __CG_QUALIFIER__ bool is_valid() const;

  /** \brief   Synchronizes the threads in the group.
   *
   *  \details Causes all threads in the group to wait at this synchronization point,
   *           and for all shared and global memory accesses by the threads to complete,
   *           before running synchronization. This guarantees the visibility of accessed data
   *           for all threads in the group.
   *
   * \note     There are potential read-after-write (RAW), write-after-read (WAR), or
   *           write-after-write (WAW) hazards, when threads in the group access the
   *           same addresses in shared or global memory. The data hazards can
   *           be avoided with synchronization of the group.
   */
  __CG_QUALIFIER__ void sync() const;
};
/**
 *  @defgroup CooperativeG Cooperative Groups
 *  @ingroup API
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  \note  Cooperative groups feature is implemented on Linux, under development
 *         on Microsoft Windows.
 *
 */

/** \brief   The multi-grid cooperative group type.
 *
 *  \details Represents an inter-device cooperative group type, where the
 *           participating threads within the group span across multiple
 *           devices, running the (same) kernel on these devices.
 *  \note    The multi-grid cooperative group type is implemented on Linux,
 *           under development on Microsoft Windows.
 */
class multi_grid_group : public thread_group {
  //! Only these friend functions are allowed to construct an object of this class
  //! and access its resources.
  friend __CG_QUALIFIER__ multi_grid_group this_multi_grid();

 protected:
  //! Construct multi-grid thread group (through the API this_multi_grid())
  explicit __CG_QUALIFIER__ multi_grid_group(__hip_uint32_t size)
      : thread_group(internal::cg_multi_grid, size) {}

 public:
  //! Number of invocations participating in this multi-grid group. In other
  //! words, the number of GPUs.
  __CG_QUALIFIER__ __hip_uint32_t num_grids() { return internal::multi_grid::num_grids(); }

  //! Rank of this invocation. In other words, an ID number within the range
  //! [0, num_grids()) of the GPU that kernel is running on.
  __CG_QUALIFIER__ __hip_uint32_t grid_rank() { return internal::multi_grid::grid_rank(); }
  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ __hip_uint32_t thread_rank() const {
    return internal::multi_grid::thread_rank();
  }
  //! @copydoc thread_group::is_valid
  __CG_QUALIFIER__ bool is_valid() const { return internal::multi_grid::is_valid(); }
  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::multi_grid::sync(); }
};

/** \addtogroup CooperativeGConstruct Construct functions of Cooperative groups
 * \ingroup CooperativeG
 *  @{ */

/** \brief   User-exposed API interface to construct grid cooperative group type
 *           object - `multi_grid_group`.
 *
 *  \details User is not allowed to construct an object of type
 *           `multi_grid_group` directly. Instead, they should construct it
 *           through this API function.
 *  \note    This multi-grid cooperative API type is implemented on Linux, under
 *           development on Microsoft Windows.
 */
__CG_QUALIFIER__ multi_grid_group this_multi_grid() {
  return multi_grid_group(internal::multi_grid::num_threads());
}
// Doxygen end group CooperativeGConstruct
/** @} */

/** \brief   The grid cooperative group type.
 *
 *  \details Represents an inter-workgroup cooperative group type, where the
 *           participating threads within the group spans across multiple
 *           workgroups running the (same) kernel on the same device.
 *  \note    This is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class grid_group : public thread_group {
  //! Only these friend functions are allowed to construct an object of this class
  //! and access its resources.
  friend __CG_QUALIFIER__ grid_group this_grid();

 protected:
  //! Construct grid thread group (through the API this_grid())
  explicit __CG_QUALIFIER__ grid_group(__hip_uint32_t size)
      : thread_group(internal::cg_grid, size) {}

 public:
  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ __hip_uint32_t thread_rank() const { return internal::grid::thread_rank(); }
  //! @copydoc thread_group::block_rank
  __CG_QUALIFIER__ __hip_uint32_t block_rank() const { return internal::grid::block_rank(); }
  //! @copydoc thread_group::is_valid
  __CG_QUALIFIER__ bool is_valid() const { return internal::grid::is_valid(); }
  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::grid::sync(); }
  __CG_QUALIFIER__ dim3 group_dim() const { return internal::grid::grid_dim(); }
  struct arrival_token {
    unsigned int signal;
  };
  //! Arrive at a barrier
  __CG_QUALIFIER__ arrival_token barrier_arrive() const {
    arrival_token t;
    t.signal = internal::grid::barrier_signal();
    return t;
  }
  //! Arrive at a barrier
  __CG_QUALIFIER__ void barrier_wait(arrival_token&& t) const {
    internal::grid::barrier_wait(t.signal);
  }
};

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API interface to construct grid cooperative group type
 *           object - `grid_group`.
 *
 *  \details User is not allowed to construct an object of type `grid_group`
 *           directly. Instead, they should construct it through this
 *           API function.
 *  \note    This function is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
__CG_QUALIFIER__ grid_group this_grid() { return grid_group(internal::grid::num_threads()); }

/** \brief   The workgroup (thread-block in CUDA terminology) cooperative group
 *           type.
 *
 *  \details Represents an intra-workgroup cooperative group type, where the
 *           participating threads within the group are the same threads that
 *           participated in the currently executing `workgroup`.
 *  \note    This function is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class thread_block : public thread_group {
  //! Only these friend functions are allowed to construct an object of thi
  //! class and access its resources
  friend __CG_QUALIFIER__ thread_block this_thread_block();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent,
                                                       unsigned int tile_size);

 protected:
  // Construct a workgroup thread group (through the API this_thread_block())
  explicit __CG_QUALIFIER__ thread_block(__hip_uint32_t size)
      : thread_group(internal::cg_workgroup, size) {}

  __CG_QUALIFIER__ thread_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);
    // Invalid tile size, assert
    if (!tile_size || (tile_size > warpSize) || !pow2) {
      __hip_assert(false && "invalid tile size");
    }

    auto block_size = num_threads();
    auto rank = thread_rank();
    auto partitions = (block_size + tile_size - 1) / tile_size;
    auto tail = (partitions * tile_size) - block_size;
    auto partition_size = tile_size - tail * (rank >= (partitions - 1) * tile_size);
    thread_group tiledGroup = thread_group(internal::cg_tiled_group, partition_size);

    tiledGroup.coalesced_info.tiled_info.num_threads = tile_size;
    tiledGroup.coalesced_info.tiled_info.is_tiled = true;
    tiledGroup.coalesced_info.tiled_info.meta_group_rank = rank / tile_size;
    tiledGroup.coalesced_info.tiled_info.meta_group_size = partitions;
    return tiledGroup;
  }

 public:
  //! Returns 3-dimensional block index within the grid.
  __CG_STATIC_QUALIFIER__ dim3 group_index() { return internal::workgroup::group_index(); }
  //! Returns 3-dimensional thread index within the block.
  __CG_STATIC_QUALIFIER__ dim3 thread_index() { return internal::workgroup::thread_index(); }
  //! @copydoc thread_group::thread_rank
  __CG_STATIC_QUALIFIER__ __hip_uint32_t thread_rank() {
    return internal::workgroup::thread_rank();
  }
  //! @copydoc thread_group::block_rank
  __CG_STATIC_QUALIFIER__ __hip_uint32_t block_rank() {
    return internal::workgroup::block_rank();
  }
  //! @copydoc thread_group::num_threads
  __CG_STATIC_QUALIFIER__ __hip_uint32_t num_threads() {
    return internal::workgroup::num_threads();
  }
  //! @copydoc thread_group::size
  __CG_STATIC_QUALIFIER__ __hip_uint32_t size() { return num_threads(); }
  //! @copydoc thread_group::is_valid
  __CG_STATIC_QUALIFIER__ bool is_valid() { return internal::workgroup::is_valid(); }
  //! @copydoc thread_group::sync
  __CG_STATIC_QUALIFIER__ void sync() { internal::workgroup::sync(); }
  //! Returns the group dimensions.
  __CG_QUALIFIER__ dim3 group_dim() { return internal::workgroup::block_dim(); }
  struct arrival_token {};
  //! Arrive at a barrier
  __CG_QUALIFIER__ arrival_token barrier_arrive() const {
    internal::workgroup::barrier_arrive();
    return arrival_token{};
  }
  //! Arrive at a barrier
  __CG_QUALIFIER__ void barrier_wait(arrival_token&&) const { internal::workgroup::barrier_wait(); }
};

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API interface to construct workgroup cooperative
 *           group type object - `thread_block`.
 *
 *  \details User is not allowed to construct an object of type `thread_block`
 *           directly. Instead, they should construct it through this API
 *           function.
 *  \note    This function is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
__CG_QUALIFIER__ thread_block this_thread_block() {
  return thread_block(internal::workgroup::num_threads());
}

/** \brief   The tiled_group cooperative group type
 *
 *  \details Represents one tiled thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 *  \note    This is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class tiled_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent,
                                                      unsigned int tile_size);

  __CG_QUALIFIER__ tiled_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || (tile_size > warpSize) || !pow2) {
      __hip_assert(false && "invalid tile size");
    }

    if (num_threads() <= tile_size) {
      return *this;
    }

    tiled_group tiledGroup = tiled_group(tile_size);
    tiledGroup.coalesced_info.tiled_info.is_tiled = true;
    return tiledGroup;
  }

 protected:
  explicit __CG_QUALIFIER__ tiled_group(unsigned int tileSize)
      : thread_group(internal::cg_tiled_group, tileSize) {
    coalesced_info.tiled_info.num_threads = tileSize;
    coalesced_info.tiled_info.is_tiled = true;
  }

 public:
  //! @copydoc thread_group::num_threads
  __CG_QUALIFIER__ unsigned int num_threads() const {
    return (coalesced_info.tiled_info.num_threads);
  }

  //! @copydoc thread_group::size
  __CG_QUALIFIER__ unsigned int size() const { return num_threads(); }

  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ unsigned int thread_rank() const {
    return (internal::workgroup::thread_rank() & (coalesced_info.tiled_info.num_threads - 1));
  }
  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::tiled_group::sync(); }
};

template <unsigned int size, class ParentCGTy> class thread_block_tile;

/** \brief   The coalesced_group cooperative group type
 *
 *  \details Represents an active thread group in a wavefront.
 *           This group type also supports sub-wave level intrinsics.
 *  \note    This is implemented on Linux and is under development
 *           on Microsoft Windows.
 */
class coalesced_group : public thread_group {
 private:
  friend __CG_QUALIFIER__ coalesced_group coalesced_threads();
  friend __CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent,
                                                       unsigned int tile_size);
  friend __CG_QUALIFIER__ coalesced_group tiled_partition(const coalesced_group& parent,
                                                          unsigned int tile_size);
  friend __CG_QUALIFIER__ coalesced_group binary_partition(const coalesced_group& cgrp, bool pred);

  template <unsigned int fsize, class fparent> friend __CG_QUALIFIER__ coalesced_group
  binary_partition(const thread_block_tile<fsize, fparent>& tgrp, bool pred);

  __CG_QUALIFIER__ coalesced_group new_tiled_group(unsigned int tile_size) const {
    const bool pow2 = ((tile_size & (tile_size - 1)) == 0);

    if (!tile_size || !pow2) {
      return coalesced_group(0);
    }

    // If a tiled group is passed to be partitioned further into a coalesced_group.
    // prepare a mask for further partitioning it so that it stays coalesced.
    if (coalesced_info.tiled_info.is_tiled) {
      unsigned int base_offset = (thread_rank() & (~(tile_size - 1)));
      unsigned int masklength =
          min(static_cast<unsigned int>(num_threads()) - base_offset, tile_size);
      lane_mask full_mask = (static_cast<int>(warpSize) == 32)
                                ? static_cast<lane_mask>((1u << 32) - 1)
                                : static_cast<lane_mask>(-1ull);
      lane_mask member_mask = full_mask >> (warpSize - masklength);

      member_mask <<= (__lane_id() & ~(tile_size - 1));
      coalesced_group coalesced_tile = coalesced_group(member_mask);
      coalesced_tile.coalesced_info.tiled_info.is_tiled = true;
      coalesced_tile.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
      coalesced_tile.coalesced_info.tiled_info.meta_group_size = num_threads() / tile_size;
      return coalesced_tile;
    }
    // Here the parent coalesced_group is not partitioned.
    else {
      lane_mask member_mask = 0;
      unsigned int tile_rank = 0;
      int lanes_to_skip = ((thread_rank()) / tile_size) * tile_size;

      for (unsigned int i = 0; i < warpSize; i++) {
        lane_mask active = coalesced_info.member_mask & (static_cast<lane_mask>(1) << i);
        // Make sure the lane is active
        if (active) {
          if (lanes_to_skip <= 0 && tile_rank < tile_size) {
            // Prepare a member_mask that is appropriate for a tile
            member_mask |= active;
            tile_rank++;
          }
          lanes_to_skip--;
        }
      }
      coalesced_group coalesced_tile = coalesced_group(member_mask);
      coalesced_tile.coalesced_info.tiled_info.meta_group_rank = thread_rank() / tile_size;
      coalesced_tile.coalesced_info.tiled_info.meta_group_size =
          (num_threads() + tile_size - 1) / tile_size;
      return coalesced_tile;
    }
    return coalesced_group(0);
  }

 protected:
  // Constructor
  explicit __CG_QUALIFIER__ coalesced_group(lane_mask member_mask)
      : thread_group(internal::cg_coalesced_group) {
    coalesced_info.member_mask = member_mask;  // Which threads are active
    coalesced_info.num_threads =
        __popcll(coalesced_info.member_mask);    // How many threads are active
    coalesced_info.tiled_info.is_tiled = false;  // Not a partitioned group
    coalesced_info.tiled_info.meta_group_rank = 0;
    coalesced_info.tiled_info.meta_group_size = 1;
  }

 public:
  //! @copydoc thread_group::num_threads
  __CG_QUALIFIER__ unsigned int num_threads() const { return coalesced_info.num_threads; }

  //! @copydoc thread_group::size
  __CG_QUALIFIER__ unsigned int size() const { return num_threads(); }

  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ unsigned int thread_rank() const {
    return internal::coalesced_group::masked_bit_count(coalesced_info.member_mask);
  }

  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync() const { internal::coalesced_group::sync(); }

  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size).
  __CG_QUALIFIER__ unsigned int meta_group_rank() const {
    return coalesced_info.tiled_info.meta_group_rank;
  }

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_QUALIFIER__ unsigned int meta_group_size() const {
    return coalesced_info.tiled_info.meta_group_size;
  }

  /** \brief Shuffle operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle operation is a direct copy of ``var`` from ``srcRank``
   *           thread ID of group.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy. Only the srcRank thread ID of
   *                  group is copied to other threads.
   *  \param srcRank [in] The source thread ID of the group for copy.
   */
  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    srcRank = srcRank % static_cast<int>(num_threads());

    int lane = (num_threads() == warpSize) ? srcRank
               : (static_cast<int>(warpSize) == 64)
                   ? __fns64(coalesced_info.member_mask, 0, (srcRank + 1))
                   : __fns32(coalesced_info.member_mask, 0, (srcRank + 1));

    return __shfl(var, lane, warpSize);
  }

  /** \brief Shuffle down operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle down operation is copy of ``var`` from thread with
   *           thread ID of group relative higher with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID + lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    // Note: The cuda implementation appears to use the remainder of lane_delta
    // and WARP_SIZE as the shift value rather than lane_delta itself.
    // This is not described in the documentation and is not done here.

    if (num_threads() == warpSize) {
      return __shfl_down(var, lane_delta, warpSize);
    }

    int lane;
    if (static_cast<int>(warpSize) == 64) {
      lane = __fns64(coalesced_info.member_mask, __lane_id(), lane_delta + 1);
    } else {
      lane = __fns32(coalesced_info.member_mask, __lane_id(), lane_delta + 1);
    }

    if (lane == -1) {
      lane = __lane_id();
    }

    return __shfl(var, lane, warpSize);
  }

  /** \brief Shuffle up operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle up operation is copy of ``var`` from thread with
   *           thread ID of group relative lower with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID - lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    // Note: The cuda implementation appears to use the remainder of lane_delta
    // and WARP_SIZE as the shift value rather than lane_delta itself.
    // This is not described in the documentation and is not done here.

    if (num_threads() == warpSize) {
      return __shfl_up(var, lane_delta, warpSize);
    }

    int lane;
    if (static_cast<int>(warpSize) == 64) {
      lane = __fns64(coalesced_info.member_mask, __lane_id(), -(lane_delta + 1));
    } else if (static_cast<int>(warpSize) == 32) {
      lane = __fns32(coalesced_info.member_mask, __lane_id(), -(lane_delta + 1));
    }

    if (lane == -1) {
      lane = __lane_id();
    }

    return __shfl(var, lane, warpSize);
  }
#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

  /** \brief Ballot function on group level.
   *
   *  \details Returns a bit mask with the Nth bit set to one if the specified
   *           predicate evaluates as true on the Nth thread.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ unsigned long long ballot(int pred) const {
    return internal::helper::adjust_mask(
        coalesced_info.member_mask,
        __ballot_sync<unsigned long long>(coalesced_info.member_mask, pred));
  }

  /** \brief Any function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for any threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int any(int pred) const {
    return __any_sync(static_cast<unsigned long long>(coalesced_info.member_mask), pred);
  }

  /** \brief All function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for all threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int all(int pred) const {
    return __all_sync(static_cast<unsigned long long>(coalesced_info.member_mask), pred);
  }

  /** \brief Match any function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if that thread has the same value in ``value`` as the
   *           caller thread.
   *
   *  \param value [in] The value to examine on the current thread in group.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_any(T value) const {
    return internal::helper::adjust_mask(
        coalesced_info.member_mask,
        __match_any_sync(static_cast<unsigned long long>(coalesced_info.member_mask), value));
  }

  /** \brief Match all function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if they all have the same value in ``value`` as the caller
   *           thread. The predicate ``pred`` is set to true if all
   *           participating threads have the same value in ``value``.
   *
   *  \param value [in] The value to examine on the current thread in group.
   *  \param pred [out] The predicate is set to true if all participating
   *                    threads in the thread group have the same value.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_all(T value, int& pred) const {
    return internal::helper::adjust_mask(
        coalesced_info.member_mask,
        __match_all_sync(static_cast<unsigned long long>(coalesced_info.member_mask), value,
                         &pred));
  }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
};

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API to create coalesced groups.
 *
 *  \details A collective operation that groups all active lanes into a new
 *           thread group.
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ coalesced_group coalesced_threads() {
  return cooperative_groups::coalesced_group(
      __builtin_amdgcn_is_invocable(__builtin_amdgcn_read_exec)
          ? __builtin_amdgcn_read_exec()
          : 0);
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/**
 *  Implementation of all publicly exposed base class APIs
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ __hip_uint32_t thread_group::thread_rank() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->thread_rank());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->thread_rank());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->thread_rank());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->thread_rank());
    }
    case internal::cg_coalesced_group: {
      return (static_cast<const coalesced_group*>(this)->thread_rank());
    }
    default: {
      __hip_assert(false && "invalid cooperative group type");
      return -1;
    }
  }
}

/**
 *  Implementation of all publicly exposed thread group API
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ bool thread_group::is_valid() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      return (static_cast<const multi_grid_group*>(this)->is_valid());
    }
    case internal::cg_grid: {
      return (static_cast<const grid_group*>(this)->is_valid());
    }
    case internal::cg_workgroup: {
      return (static_cast<const thread_block*>(this)->is_valid());
    }
    case internal::cg_tiled_group: {
      return (static_cast<const tiled_group*>(this)->is_valid());
    }
    case internal::cg_coalesced_group: {
      return (static_cast<const coalesced_group*>(this)->is_valid());
    }
    default: {
      __hip_assert(false && "invalid cooperative group type");
      return false;
    }
  }
}

/**
 *  Implementation of all publicly exposed thread group sync API
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
__CG_QUALIFIER__ void thread_group::sync() const {
  switch (this->_type) {
    case internal::cg_multi_grid: {
      static_cast<const multi_grid_group*>(this)->sync();
      break;
    }
    case internal::cg_grid: {
      static_cast<const grid_group*>(this)->sync();
      break;
    }
    case internal::cg_workgroup: {
      static_cast<const thread_block*>(this)->sync();
      break;
    }
    case internal::cg_tiled_group: {
      static_cast<const tiled_group*>(this)->sync();
      break;
    }
    case internal::cg_coalesced_group: {
      static_cast<const coalesced_group*>(this)->sync();
      break;
    }
    default: {
      __hip_assert(false && "invalid cooperative group type");
    }
  }
#if __has_builtin(__builtin_amdgcn_s_wait_asynccnt)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_wait_asynccnt))
    __builtin_amdgcn_s_wait_asynccnt(0);
#endif
}

#endif

/** \addtogroup CooperativeGAPI User-exposed API of Cooperative groups
 * \ingroup CooperativeG
 *  @{ */

/** \brief   Returns the size of the group.
 *
 *  \details Total number of threads in the thread group, and this serves the
 *           purpose for all derived cooperative group types because their
 *           `size` is directly saved during the construction.
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for size returns.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ __hip_uint32_t group_size(CGTy const& g) {
  return g.num_threads();
}

/** \brief   Returns the rank of thread of the group.
 *
 *  \details Rank of the calling thread within [0, \link num_threads() num_threads() \endlink).
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for rank returns.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ __hip_uint32_t thread_rank(CGTy const& g) {
  return g.thread_rank();
}

/** \brief   Returns true if the group has not violated any API constraints.
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for validity check.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ bool is_valid(CGTy const& g) { return g.is_valid(); }

/** \brief   Synchronizes the threads in the group.
 *
 *  \tparam CGTy  The cooperative group class template parameter.
 *  \param g [in] The cooperative group for synchronization.
 *
 *  \note    Implementation of publicly exposed `wrapper` API on top of basic
 *           cooperative group type APIs. This function is implemented on Linux
 *           and is under development on Microsoft Windows.
 */
template <class CGTy> __CG_QUALIFIER__ void sync(CGTy const& g) { g.sync(); }

// Doxygen end group CooperativeGAPI
/** @} */

/**
 * template class tile_base
 *  \note  This function is implemented on Linux and is under development
 *  on Microsoft Windows.
 */
template <unsigned int tileSize> class tile_base {
 protected:
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

 public:
  //! Rank of the thread within this tile
  _CG_STATIC_CONST_DECL_ unsigned int thread_rank() {
    return (internal::workgroup::thread_rank() & (numThreads - 1));
  }

  //! Number of threads within this tile
  __CG_STATIC_QUALIFIER__ unsigned int num_threads() { return numThreads; }

  //! Legacy member functions
  //! Number of threads within this tile (alias of num_threads())
  __CG_STATIC_QUALIFIER__ unsigned int size() { return num_threads(); }
};

/**
 * template class thread_block_tile_base
 *  \note  This class is implemented on Linux, under development
 *  on Microsoft Windows.
 */
template <unsigned int size> class thread_block_tile_base : public tile_base<size> {
  static_assert(is_valid_tile_size<size>::value,
                "Tile size is either not a power of 2 or greater than the wavefront size");
  using tile_base<size>::numThreads;

  template <unsigned int fsize, class fparent> friend __CG_QUALIFIER__ coalesced_group
  binary_partition(const thread_block_tile<fsize, fparent>& tgrp, bool pred);

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
  __CG_QUALIFIER__ unsigned long long build_mask() const {
    unsigned long long mask = ~0ull >> (64 - numThreads);
    // thread_rank() gives thread id from 0..thread launch size.
    return mask << (((internal::workgroup::thread_rank() % warpSize) / numThreads) * numThreads);
  }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS

 public:
  __CG_STATIC_QUALIFIER__ void sync() { internal::tiled_group::sync(); }

  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const {
    return (__shfl(var, srcRank, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const {
    return (__shfl_down(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const {
    return (__shfl_up(var, lane_delta, numThreads));
  }

  template <class T> __CG_QUALIFIER__ T shfl_xor(T var, unsigned int laneMask) const {
    return (__shfl_xor(var, laneMask, numThreads));
  }

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
  __CG_QUALIFIER__ unsigned long long ballot(int pred) const {
    const auto mask = build_mask();
    return internal::helper::adjust_mask(mask, __ballot_sync(mask, pred));
  }

  __CG_QUALIFIER__ int any(int pred) const { return __any_sync(build_mask(), pred); }

  __CG_QUALIFIER__ int all(int pred) const { return __all_sync(build_mask(), pred); }

  template <typename T> __CG_QUALIFIER__ unsigned long long match_any(T value) const {
    const auto mask = build_mask();
    return internal::helper::adjust_mask(mask, __match_any_sync(mask, value));
  }

  template <typename T> __CG_QUALIFIER__ unsigned long long match_all(T value, int& pred) const {
    const auto mask = build_mask();
    return internal::helper::adjust_mask(mask, __match_all_sync(mask, value, &pred));
  }
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
};

/** \brief   User exposed API that captures the state of the parent group pre-partition
 */
template <unsigned int tileSize, typename ParentCGTy> class parent_group_info {
 public:
  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size)
  __CG_STATIC_QUALIFIER__ unsigned int meta_group_rank() {
    return ParentCGTy::thread_rank() / tileSize;
  }

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_STATIC_QUALIFIER__ unsigned int meta_group_size() {
    return (ParentCGTy::num_threads() + tileSize - 1) / tileSize;
  }
};

/** \brief   Group type - thread_block_tile
 *
 *  \details  Represents one tile of thread group.
 *  \note     This type is implemented on Linux, under development
 *            on Microsoft Windows.
 */
template <unsigned int tileSize, class ParentCGTy> class thread_block_tile_type
    : public thread_block_tile_base<tileSize>,
      public tiled_group,
      public parent_group_info<tileSize, ParentCGTy> {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;
  typedef thread_block_tile_base<numThreads> tbtBase;

 protected:
  __CG_QUALIFIER__ thread_block_tile_type() : tiled_group(numThreads) {
    coalesced_info.tiled_info.num_threads = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
  }

  __CG_QUALIFIER__ thread_block_tile_type(unsigned int meta_group_rank,
                                          unsigned int meta_group_size)
      : tiled_group(numThreads) {
    coalesced_info.tiled_info.num_threads = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
    coalesced_info.tiled_info.meta_group_rank = meta_group_rank;
    coalesced_info.tiled_info.meta_group_size = meta_group_size;
  }

 public:
  using tbtBase::num_threads;
  using tbtBase::size;
  using tbtBase::sync;
  using tbtBase::thread_rank;
};

// Partial template specialization
template <unsigned int tileSize> class thread_block_tile_type<tileSize, void>
    : public thread_block_tile_base<tileSize>, public tiled_group {
  _CG_STATIC_CONST_DECL_ unsigned int numThreads = tileSize;

  typedef thread_block_tile_base<numThreads> tbtBase;

 protected:
  __CG_QUALIFIER__ thread_block_tile_type(unsigned int meta_group_rank,
                                          unsigned int meta_group_size)
      : tiled_group(numThreads) {
    coalesced_info.tiled_info.num_threads = numThreads;
    coalesced_info.tiled_info.is_tiled = true;
    coalesced_info.tiled_info.meta_group_rank = meta_group_rank;
    coalesced_info.tiled_info.meta_group_size = meta_group_size;
  }

 public:
  using tbtBase::num_threads;
  using tbtBase::size;
  using tbtBase::sync;
  using tbtBase::thread_rank;

  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size)
  __CG_QUALIFIER__ unsigned int meta_group_rank() const {
    return coalesced_info.tiled_info.meta_group_rank;
  }

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_QUALIFIER__ unsigned int meta_group_size() const {
    return coalesced_info.tiled_info.meta_group_size;
  }
  // Doxygen end group CooperativeG
  /**
   * @}
   */
};

__CG_QUALIFIER__ thread_group this_thread() {
  thread_group g(internal::group_type::cg_coalesced_group, 1, __ockl_activelane_u32());
  return g;
}

/** \ingroup CooperativeGConstruct
 *  \brief   User-exposed API to partition groups.
 *
 *  \details A collective operation that partitions the parent group into a
 *           one-dimensional, row-major, tiling of subgroups.
 */

__CG_QUALIFIER__ thread_group tiled_partition(const thread_group& parent, unsigned int tile_size) {
  if (parent.cg_type() == internal::cg_tiled_group) {
    const tiled_group* cg = static_cast<const tiled_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  } else if (parent.cg_type() == internal::cg_coalesced_group) {
    const coalesced_group* cg = static_cast<const coalesced_group*>(&parent);
    return cg->new_tiled_group(tile_size);
  } else {
    const thread_block* tb = static_cast<const thread_block*>(&parent);
    return tb->new_tiled_group(tile_size);
  }
}

// Thread block type overload
__CG_QUALIFIER__ thread_group tiled_partition(const thread_block& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

__CG_QUALIFIER__ tiled_group tiled_partition(const tiled_group& parent, unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

// If a coalesced group is passed to be partitioned, it should remain coalesced
__CG_QUALIFIER__ coalesced_group tiled_partition(const coalesced_group& parent,
                                                 unsigned int tile_size) {
  return (parent.new_tiled_group(tile_size));
}

namespace impl {
template <unsigned int size, class ParentCGTy> class thread_block_tile_internal;

template <unsigned int size, class ParentCGTy> class thread_block_tile_internal
    : public thread_block_tile_type<size, ParentCGTy> {
 protected:
  template <unsigned int tbtSize, class tbtParentT> __CG_QUALIFIER__ thread_block_tile_internal(
      const thread_block_tile_internal<tbtSize, tbtParentT>& g)
      : thread_block_tile_type<size, ParentCGTy>(g.meta_group_rank(), g.meta_group_size()) {}

  __CG_QUALIFIER__ thread_block_tile_internal(const thread_block& g)
      : thread_block_tile_type<size, ParentCGTy>() {}
};

// becomes to std::true_type if the group is tiled and has a size known at compile time
template <class TyGroup>
struct isTiledGroup : __hip_internal::false_type {
};

template <unsigned int N, class ParentCGTy>
struct isTiledGroup<cooperative_groups::thread_block_tile<N, ParentCGTy>>
  : __hip_internal::integral_constant<bool,
        (N == 1  || N == 2  || N == 4  || N == 8 ||
         N == 16 || N == 32 || N == 64)> {
};

// returns the size of tile_group provided it is known at compile time
template <class TyGroup>
struct tiledGroupSize : __hip_internal::integral_constant<int, 0> {

};
template <unsigned int N, class ParentCGTy>
struct tiledGroupSize<cooperative_groups::thread_block_tile<N, ParentCGTy>>
  : __hip_internal::integral_constant<int, N> {
};

template <class TyGroup>
struct isCoalescedGroup : __hip_internal::false_type {
};

template <>
struct isCoalescedGroup<cooperative_groups::coalesced_group> : __hip_internal::true_type {
};
}  // namespace impl

/** \brief    Group type - thread_block_tile
 *
 *  \details  Represents one tiled thread group in a wavefront.
 *            This group type also supports sub-wave level intrinsics.
 *
 *  \note     This type is implemented on Linux, under development
 *            on Microsoft Windows.
 */
template <unsigned int size, class ParentCGTy> class thread_block_tile
    : public impl::thread_block_tile_internal<size, ParentCGTy> {
 protected:
  __CG_QUALIFIER__ thread_block_tile(const ParentCGTy& g)
      : impl::thread_block_tile_internal<size, ParentCGTy>(g) {}

 public:
  __CG_QUALIFIER__ operator thread_block_tile<size, void>() const {
    return thread_block_tile<size, void>(*this);
  }

#ifdef DOXYGEN_SHOULD_INCLUDE_THIS

  //! @copydoc thread_group::thread_rank
  __CG_QUALIFIER__ unsigned int thread_rank() const;

  //! @copydoc thread_group::sync
  __CG_QUALIFIER__ void sync();

  //! Returns the linear rank of the group within the set of tiles partitioned
  //! from a parent group (bounded by meta_group_size)
  __CG_QUALIFIER__ unsigned int meta_group_rank() const;

  //! Returns the number of groups created when the parent group was partitioned.
  __CG_QUALIFIER__ unsigned int meta_group_size() const;

  /** \brief Shuffle operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle operation is a direct copy of ``var`` from ``srcRank``
   *           thread ID of group.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy. Only the srcRank thread ID of
   *                  group is copied to other threads.
   *  \param srcRank [in] The source thread ID of the group for copy.
   */
  template <class T> __CG_QUALIFIER__ T shfl(T var, int srcRank) const;

  /** \brief Shuffle down operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle down operation is copy of ``var`` from thread with
   *           thread ID of group relative higher with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID + lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_down(T var, unsigned int lane_delta) const;

  /** \brief Shuffle up operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle up operation is copy of ``var`` from thread with
   *           thread ID of group relative lower with ``lane_delta`` to caller
   *           thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param lane_delta [in] The lane_delta is the relative thread ID difference
   *                         between caller thread ID and source of copy thread
   *                         ID. sourceID = (threadID - lane_delta) % size()
   */
  template <class T> __CG_QUALIFIER__ T shfl_up(T var, unsigned int lane_delta) const;

  /** \brief Shuffle xor operation on group level.
   *
   *  \details Exchanging variables between threads without use of shared memory.
   *           Shuffle xor operation is copy of var from thread with thread ID
   *           of group based on laneMask XOR of the caller thread ID.
   *
   *  \tparam T The type can be a 32-bit integer or single-precision
   *            floating point.
   *  \param var [in] The source variable to copy.
   *  \param laneMask [in] The laneMask is the mask for XOR operation.
   *                       sourceID = threadID ^ laneMask
   */
  template <class T> __CG_QUALIFIER__ T shfl_xor(T var, unsigned int laneMask) const;

  /** \brief Ballot function on group level.
   *
   *  \details Returns a bit mask with the Nth bit set to one if the Nth thread
   *           predicate evaluates true.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ unsigned long long ballot(int pred) const;

  /** \brief Any function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for any threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int any(int pred) const;

  /** \brief All function on group level.
   *
   *  \details Returns non-zero if a predicate evaluates true for all threads.
   *
   *  \param pred [in] The predicate to evaluate on group threads.
   */
  __CG_QUALIFIER__ int all(int pred) const;

  /** \brief Match any function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if that thread has the same value in ``value`` as the
   *           caller thread.
   *
   *  \param value [in] The value to examine on the current thread in group.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_any(T value) const;

  /** \brief Match all function on group level.
   *
   *  \details Returns a bit mask containing a 1-bit for every participating
   *           thread if they all have the same value in ``value`` as the caller
   *           thread. The predicate ``pred`` is set to true if all
   *           participating threads have the same value in ``value``.
   *
   *  \param value [in] The value to examine on the current thread in group.
   *  \param pred [out] The predicate is set to true if all participating
   *                    threads in the thread group have the same value.
   */
  template <typename T> __CG_QUALIFIER__ unsigned long long match_all(T value, int& pred) const;

#endif
};

template <unsigned int size> class thread_block_tile<size, void>
    : public impl::thread_block_tile_internal<size, void> {
  template <unsigned int, class ParentCGTy> friend class thread_block_tile;

 protected:
 public:
  template <class ParentCGTy>
  __CG_QUALIFIER__ thread_block_tile(const thread_block_tile<size, ParentCGTy>& g)
      : impl::thread_block_tile_internal<size, void>(g) {}
};

template <unsigned int size, class ParentCGTy = void> class thread_block_tile;

namespace impl {
template <unsigned int size, class ParentCGTy> struct tiled_partition_internal;

template <unsigned int size> struct tiled_partition_internal<size, thread_block>
    : public thread_block_tile<size, thread_block> {
  __CG_QUALIFIER__ tiled_partition_internal(const thread_block& g)
      : thread_block_tile<size, thread_block>(g) {}
};

// ParentCGTy = thread_block_tile<ParentSize, GrandParentCGTy> specialization
template <unsigned int size, unsigned int ParentSize, class GrandParentCGTy>
struct tiled_partition_internal<size, thread_block_tile<ParentSize, GrandParentCGTy> >
    : public thread_block_tile<size, thread_block_tile<ParentSize, GrandParentCGTy> > {
  static_assert(size < ParentSize, "Sub tile size must be < parent tile size in tiled_partition");

  __CG_QUALIFIER__ tiled_partition_internal(const thread_block_tile<ParentSize, GrandParentCGTy>& g)
      : thread_block_tile<size, thread_block_tile<ParentSize, GrandParentCGTy> >(g) {}
};

}  // namespace impl

/** \ingroup CooperativeGConstruct
 *  \brief   Create a partition.
 *
 *  \details This constructs a templated class derived from thread_group. The
 *           template defines the tile size of the new thread group at compile
 *           time.
 *
 *  \tparam size       The new size of the partition.
 *  \tparam ParentCGTy The cooperative group class template parameter of the input group.
 *
 *  \param g [in] The coalesced group for split.
 */
template <unsigned int size, class ParentCGTy>
__CG_QUALIFIER__ thread_block_tile<size, ParentCGTy> tiled_partition(const ParentCGTy& g) {
  static_assert(is_valid_tile_size<size>::value,
                "Tiled partition with size > wavefront size. Currently not supported ");
  return impl::tiled_partition_internal<size, ParentCGTy>(g);
}

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

/** \ingroup CooperativeGConstruct
 *  \brief Binary partition.
 *
 *  \details This splits the input thread group into two partitions determined by predicate.
 *
 *  \param cgrp [in] The coalesced group for split.
 *  \param pred [in] The predicate used during the group split up.
 */
__CG_QUALIFIER__ coalesced_group binary_partition(const coalesced_group& cgrp, bool pred) {
  auto mask = __ballot_sync<unsigned long long>(cgrp.coalesced_info.member_mask, pred);

  if (pred) {
    return coalesced_group(mask);
  } else {
    return coalesced_group(cgrp.coalesced_info.member_mask ^ mask);
  }
}

/** \ingroup CooperativeGConstruct
 *  \brief Binary partition.
 *
 *  \details This splits the input thread group into two partitions determined by predicate.
 *
 *  \tparam size   The size of the input thread block tile group.
 *  \tparam parent The cooperative group class template parameter of the input group.
 *
 *  \param tgrp [in] The thread block tile group for split.
 *  \param pred [in] The predicate used during the group split up.
 */
template <unsigned int size, class parent>
__CG_QUALIFIER__ coalesced_group binary_partition(const thread_block_tile<size, parent>& tgrp,
                                                  bool pred) {
  auto mask = __ballot_sync<unsigned long long>(tgrp.build_mask(), pred);

  if (pred) {
    return coalesced_group(mask);
  } else {
    return coalesced_group(tgrp.build_mask() ^ mask);
  }
}


template <class T>
struct plus {
  __CG_QUALIFIER__ T operator()(T lhs, T rhs) const
  {
    return lhs + rhs;
  }
};

template <class T>
struct less {
  __CG_QUALIFIER__ T operator()(T lhs, T rhs) const
  {
    return lhs < rhs? lhs : rhs;
  }
};

template <class T>
struct greater {
  __CG_QUALIFIER__ T operator()(T lhs, T rhs) const
  {
    return lhs < rhs? rhs : lhs;
  }
};

template <class T>
struct bit_and {
  __CG_QUALIFIER__ T operator()(T lhs, T rhs) const
  {
    return lhs & rhs;
  }
};

template <class T>
struct bit_xor {
  __CG_QUALIFIER__ T operator()(T lhs, T rhs) const
  {
    return lhs ^ rhs;
  }
};

template <class T>
struct bit_or {
  __CG_QUALIFIER__ T operator()(T lhs, T rhs) const
  {
    return lhs | rhs;
  }
};

namespace impl {
// when instantiated with two parameter types, allows to know if there are the same, regardless
// of const/volatile qualifiers
template <typename T, typename U>
using is_param_type_same = __hip_internal::is_same<typename __hip_internal::remove_cvref<T>,
                                                   typename __hip_internal::remove_cvref<U>>;

template <class T, class Op>
struct isArithmeticFunc : __hip_internal::false_type {
};

template <class T>
struct isArithmeticFunc<T, cooperative_groups::plus<T>> : __hip_internal::true_type {
};

template <class T>
struct isArithmeticFunc<T, cooperative_groups::less<T>> : __hip_internal::true_type {
};

template <class T>
struct isArithmeticFunc<T, cooperative_groups::greater<T>> : __hip_internal::true_type {
};

template <class T, class Op>
struct isBooleanFunc : __hip_internal::false_type {
};

template <class T>
struct isBooleanFunc<T, cooperative_groups::bit_and<T>> : __hip_internal::true_type {
};

template <class T>
struct isBooleanFunc<T, cooperative_groups::bit_or<T>> : __hip_internal::true_type {
};

template <class T>
struct isBooleanFunc<T, cooperative_groups::bit_xor<T>> : __hip_internal::true_type {
};

// this is the value to return in exclusive_scan, for lane 0
template <class T, class Op>
struct CGIdentity {
  __CG_QUALIFIER__ T operator()()
  {
    T result = {};
    return result;
  }
};

template <class T>
struct CGIdentity<T, cooperative_groups::bit_and<T>> {
  __CG_QUALIFIER__ T operator()()
  {
    T result {};
    return ~result;
  }
};

template <class T>
struct CGIdentity<T, cooperative_groups::less<T>> {
  __CG_QUALIFIER__ T operator()()
  {
    // CUDA would return 0 in this case. But in our case we mimic what __ockl_wfscan_*
    // would do
    return __hip_internal::NumericLimits<T>::maximum();
  }
};

template <class T>
struct CGIdentity<T, cooperative_groups::greater<T>> {
  __CG_QUALIFIER__ T operator()()
  {
    return __hip_internal::NumericLimits<T>::minimum();
  }
};

// calculates the necessary warp mask for cooperative groups that support reduce(), or
// inclusive/exlcusive_scan()
template <typename TyGroup>
__CG_QUALIFIER__ unsigned long long groupMask(const TyGroup& group)
{
  unsigned long long mask = ~0ull;

  if constexpr (impl::isCoalescedGroup<TyGroup>::value) {
    mask = group.coalesced_info.member_mask;
  } else {
    // we cannot simply just use the __activemask() here, because more than one tile could have active
    // threads at a time; we need to mask away the threads that not part of this tile first
    mask >>= (64 - group.num_threads());
    mask <<= (((internal::workgroup::thread_rank() % warpSize) / group.num_threads()) * group.num_threads());
  }

  return mask;
}

// backward permute implementation for cooperative group operations, i.e.
// for up to 32 bytes (__hip_ds_bpermute() can only do 4 bytes at a time,
// this function calls it (or the floating point version) multiple times to
// implement it for bigger sizes
template <bool isPrimitiveType, class T, size_t NumPermutes, typename __hip_internal::enable_if<NumPermutes == 0, int>::type = 0>
__CG_QUALIFIER__ void bPermute(T&, T, int from)
{
}

// trivial case: the type fits within the permute size
template <bool IsPrimitiveType, class T, size_t NumPermutes, typename __hip_internal::enable_if<IsPrimitiveType && NumPermutes == 1, int>::type = 0>
__CG_QUALIFIER__ void bPermute(T& permuteResult, T result, int from)
{
  auto backwardPermute = [](int index, T arg) {
      if constexpr (__hip_internal::is_floating_point<T>::value &&
                    sizeof(T) <= 4) {
        return __hip_ds_bpermutef(index, arg);
      } else {
        return __hip_ds_bpermute(index, arg);
      }
    };

  if constexpr (sizeof(T) == 2) {
    union {
      int i;
      T f;
    } tmp;

    tmp.f = result;
    tmp.i = __hip_ds_bpermute(from << 2, tmp.i);
    permuteResult = tmp.f;
  } else if constexpr (sizeof(T) == 4) {
    auto bPermuteResult = backwardPermute(from << 2, result);
    __builtin_memcpy(&permuteResult, &bPermuteResult, sizeof(result));
  } else {
    static_assert(__hip_internal::is_void<T>::value, "Unexpected type");
  }
}

// Overload when we need multiple ds_permute, because one is not enough
template <bool IsPrimitiveType, class T, size_t NumPermutes>
__CG_QUALIFIER__ void bPermute(T permuteResult[NumPermutes], T result[NumPermutes], int from)
{
  // ds_bpermute only deals with 32-bit sizes, so for other sizes
  // we need to call the permute multiple times
  for (int i = 0; i < NumPermutes; i++) {
    permuteResult[i] = __hip_ds_bpermute(from << 2, result[i]);
  }
}

}  // namespace impl
#endif

/**
 * \brief Cluster group
 *
 * \note cluster can be 1D/2D/3D and have upto 15 workgroups in a cluster. Each cluster runs on a
 * separate WGP processor. Sizing, cluster size is in workgroups, workgroups size is in threads.
 */
class cluster_group {
  friend __CG_QUALIFIER__ cluster_group this_cluster();

  // Default constructor, hidden
  __CG_QUALIFIER__ cluster_group() {}

 public:
  using arrival_token = struct {};

  // Sync the cluster, equivalent to c.barrier_wait(c.barrier_arrive());
  __CG_STATIC_QUALIFIER__ void sync() { internal::cluster::sync(); }

  // Arrive on a cluster barrier, returns token that needs to be passed to barrier_wait
  __CG_STATIC_QUALIFIER__ arrival_token barrier_arrive() {
    // signal user cluster barrier
    internal::cluster::barrier_arrive();
    return arrival_token();
  }

  // Wait on arrival_token
  __CG_STATIC_QUALIFIER__ void barrier_wait(arrival_token&&) { internal::cluster::barrier_wait(); }

  // TODO: implement this when compiler work is done
  // block rank to which shared memory address belongs to
  // __CG_STATIC_QUALIFIER__ unsigned int query_shared_rank(const void* addr) {}
  // Obtain the address of shared memory variable of another block in the cluster
  // template <typename T> __CG_STATIC_QUALIFIER__ T* map_shared_rank(T* addr, int rank) {}

  // index of the calling block within cluster
  __CG_STATIC_QUALIFIER__ dim3 block_index() { return internal::cluster::block_index(); }

  // Rank of calling block within [0, num_blocks)
  __CG_STATIC_QUALIFIER__ unsigned int block_rank() { return internal::cluster::block_rank(); }

  // index of the calling thread within cluster
  __CG_STATIC_QUALIFIER__ dim3 thread_index() { return internal::cluster::thread_index(); }

  // Rank of calling thread within [0, num_threads)
  __CG_STATIC_QUALIFIER__ unsigned int thread_rank() { return internal::cluster::thread_rank(); }

  // Dimensions of launched cluster in unit of blocks
  __CG_STATIC_QUALIFIER__ dim3 dim_blocks() { return internal::cluster::dim_blocks(); }

  // total number of blocks in the group
  __CG_STATIC_QUALIFIER__ unsigned int num_blocks() { return internal::cluster::num_blocks(); }

  // Dimensions of launched cluster in unit of threads
  __CG_STATIC_QUALIFIER__ dim3 dim_threads() { return internal::cluster::dim_threads(); }

  // Total number of threads in the group
  __CG_STATIC_QUALIFIER__ unsigned int num_threads() { return internal::cluster::num_threads(); }

  // Get address of shared memory variable in another cluster
  template <typename T> __CG_STATIC_QUALIFIER__ T* map_shared_rank(T* in, int rank) {
    return internal::cluster::map_shared_rank<T>(in, rank);
  }

  // Return block rank of shared memory address
  __CG_STATIC_QUALIFIER__ unsigned int query_shared_rank(const void* in) {
    return internal::cluster::query_shared_rank(in);
  }

  // Alias of num_threads
  __CG_STATIC_QUALIFIER__ unsigned int size() { return num_threads(); }
};

/**
 * \brief get cluster group
 *
 * \return cluster_group
 */
__CG_QUALIFIER__ cluster_group this_cluster() {
  cluster_group cg;
  return cg;
}
}  // namespace cooperative_groups

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_H
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_REDUCE_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_REDUCE_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/hip_cooperative_groups_helper.h>
#include <hip/amd_detail/amd_hip_cooperative_groups.h>
#endif

#if !defined(__HIPCC_RTC__)
#include <type_traits>
#endif

namespace cooperative_groups {
namespace impl {
  template <typename T, typename = void>
  struct has_add : __hip_internal::false_type {
  };

  template <typename T>
  struct has_add<T,
                 __hip_internal::void_t<decltype(__reduce_add_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_min : __hip_internal::false_type {
  };

  template <typename T>
  struct has_min<T,
                 __hip_internal::void_t<decltype(__reduce_min_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_max : __hip_internal::false_type {
  };

  template <typename T>
  struct has_max<T,
                 __hip_internal::void_t<decltype(__reduce_max_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_and : __hip_internal::false_type {
  };

  template <typename T>
  struct has_and<T,
                 __hip_internal::void_t<decltype(__reduce_and_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_or : __hip_internal::false_type {
  };

  template <typename T>
  struct has_or<T,
                 __hip_internal::void_t<decltype(__reduce_or_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

  template <typename T, typename = void>
  struct has_xor : __hip_internal::false_type {
  };

  template <typename T>
  struct has_xor<T,
                 __hip_internal::void_t<decltype(__reduce_xor_sync<unsigned long long>(0ull, T {}))>
    > : __hip_internal::true_type {};

}

/** \ingroup CooperativeGAPI
  *
  * \brief   Returns the result of the reduction of the specified cooperative group using a
  *          specified functor
  * \details A collective operation that can be used with a thread_block_tile whose size is known
  *          at compile-time or a coalesced_group.
  *
  * \tparam  TyGroup  The cooperative group class template parameter.
  * \tparam  TyVal    The type of the value this thread contributes for the reduction.
  * \tparam  TyFn     The type of the function object
  *
  * \param group  The group to reduce.
  * \param val    The value this thread contributes to the reduction.
  * \param op     The function object whose operator() will be called.
  */
template <typename TyGroup, typename TyVal, typename TyFn>
__CG_QUALIFIER__ auto reduce(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
  using Op = typename __hip_internal::remove_cvref<TyFn>::type;
  using Val = typename __hip_internal::remove_cvref<TyVal>::type;
  static_assert(impl::is_param_type_same<Val, decltype(op(val, val))>::value, "Operator input and output types differ");
  static_assert(__hip_internal::is_trivially_copyable<Val>::value, "val must be trivially copyable");
  static_assert(sizeof(Val) <= 32, "Can only reduce values of size up to 32 bytes");

  if constexpr (!impl::isTiledGroup<TyGroup>::value && !impl::isCoalescedGroup<TyGroup>::value) {
    static_assert(__hip_internal::is_void<TyGroup>::value, "This group does not exclusively represent a tile");
  }

  unsigned long long mask = ~0ull;

  mask = impl::groupMask(group);

  if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value &&
                impl::has_add<Val>::value) {
    return __reduce_add_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::less<Val>>::value &&
                impl::has_min<Val>::value) {
    return __reduce_min_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::greater<Val>>::value &&
                impl::has_max<Val>::value) {
    return __reduce_max_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_and<Val>>::value &&
                impl::has_and<Val>::value) {
    return __reduce_and_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_or<Val>>::value &&
                impl::has_or<Val>::value) {
    return __reduce_or_sync(mask, val);
  } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_xor<Val>>::value &&
                impl::has_xor<Val>::value) {
    return __reduce_xor_sync(mask, val);
  } else {
    return __reduce_op_sync(mask, val, op, nullptr);
  }
}
}

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_REDUCE_H
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/hip_cooperative_groups_helper.h>
#include <hip/amd_detail/amd_hip_cooperative_groups.h>
#endif

#if !defined(__HIPCC_RTC__)
#include <type_traits>
#endif

namespace cooperative_groups {
namespace impl {
  // these functions allow to make use of C++ function overloads, instead of having to code
  // a big if-constexpr according to operand type
  HIP_IMPL_GENERATE_SCAN_FUNC(add, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(min, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(max, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(and, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(and, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(or, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(or, u32, unsigned int);

  HIP_IMPL_GENERATE_SCAN_FUNC(xor, i32, int);
  HIP_IMPL_GENERATE_SCAN_FUNC(xor, u32, unsigned int);

  // extra types. Unlike cg::reduce() which depends on reduce_*_sync() functions being defined with
  // HIP_ENABLE_EXTRA_WARP_SYNC_TYPES to be able to use the ockl intrinsics, for scan we always
  // define them here
  HIP_IMPL_GENERATE_SCAN_FUNC(add, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, u64, unsigned long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, f32, float);
  HIP_IMPL_GENERATE_SCAN_FUNC(add, f64, double);

  HIP_IMPL_GENERATE_SCAN_FUNC(min, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, u64, unsigned long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, f32, float);
  HIP_IMPL_GENERATE_SCAN_FUNC(min, f64, double);

  HIP_IMPL_GENERATE_SCAN_FUNC(max, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, u64, unsigned long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, f32, float);
  HIP_IMPL_GENERATE_SCAN_FUNC(max, f64, double);

  HIP_IMPL_GENERATE_SCAN_FUNC(and, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(and, u64, unsigned long long);

  HIP_IMPL_GENERATE_SCAN_FUNC(or, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(or, u64, unsigned long long);

  HIP_IMPL_GENERATE_SCAN_FUNC(xor, i64, long long);
  HIP_IMPL_GENERATE_SCAN_FUNC(xor, u64, unsigned long long);

  // not all types could be used with wfscan (e.g. user defined types), this predicate
  // indicates whether that is the case
  template <typename T, typename = void>
  struct has_arithmetic_scan : __hip_internal::false_type {
  };

  // all the arithmetic operations accept the same types, so we just based it on the overload being
  // present for scan_add
  template <typename T>
  struct has_arithmetic_scan<T,
                 __hip_internal::void_t<decltype(scan_add<false>(T {}))>
    > : __hip_internal::true_type {
  };

  template <typename T, typename = void>
  struct has_boolean_scan : __hip_internal::false_type {
  };

  // all the arithmetic operations accept the same types, so we just based it on the overload being
  // present for scan_and
  template <typename T>
  struct has_boolean_scan<T,
                 __hip_internal::void_t<decltype(scan_and<false>(T {}))>
    > : __hip_internal::true_type {
  };

  // given cooperative_groups template parameter, calls the right impl::scan function (that would contain __ockl_wfscan_*)
  // and that is overloaded by type
  template <class TyVal, class Op, bool Inclusive>
  __CG_QUALIFIER__ TyVal call_scan(const TyVal& val)
  {
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    if constexpr (__hip_internal::is_same<Op, cooperative_groups::plus<Val>>::value) {
      return impl::scan_add<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::less<Val>>::value) {
      return impl::scan_min<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::greater<Val>>::value) {
      return impl::scan_max<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_and<Val>>::value) {
      return impl::scan_and<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_or<Val>>::value) {
      return impl::scan_or<Inclusive>(val);
    } else if constexpr (__hip_internal::is_same<Op, cooperative_groups::bit_xor<Val>>::value) {
      return impl::scan_xor<Inclusive>(val);
    }
  }

  template <bool Inclusive, typename TyGroup, typename TyVal, typename TyFn>
  __CG_QUALIFIER__ auto scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val))
  {
    using Op = typename __hip_internal::remove_cvref<TyFn>::type;
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    constexpr bool isPrimitiveType = impl::has_arithmetic_scan<Val>::value;
    using permuteType = typename __hip_internal::conditional<isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), Val, unsigned int>::type;

    // the number of backward permutes will be: size(Val) / 4 rounded up
    static constexpr int kNumOfPermutes = (sizeof(Val) <= 4)?
                                          1 :
                                          (sizeof(Val) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
    static_assert(cooperative_groups::impl::is_param_type_same<Val, decltype(op(val, val))>::value, "Operator input and output types differ");
    static_assert(__hip_internal::is_trivially_copyable<Val>::value, "val must be trivially copyable");
    static_assert(sizeof(Val) <= 32, "scan only operate on values of size up to 32 bytes");

    if constexpr (!cooperative_groups::impl::isTiledGroup<TyGroup>::value && !cooperative_groups::impl::isCoalescedGroup<TyGroup>::value) {
      static_assert(__hip_internal::is_void<TyGroup>::value, "This group does not exclusively represent a tile");
    }

    unsigned int maskNumBits;
    int numIterations;
    // next bit to aggregate with
    unsigned int maskIdx;
    unsigned int laneId = __lane_id();
    int nextBit = laneId;
    unsigned long long mask = ~0ull;

    mask = impl::groupMask(group);
    maskNumBits = __popcll(mask);
    maskIdx = __popcll(((1ull << laneId) - 1) & mask);

    if (laneId) {
      mask <<= 64 - laneId;
    }

#ifdef __OPTIMIZE__  // at the time of this writing the ockl wfscan functions do not compile when
                      // using -O0
    if (impl::isTiledGroup<TyGroup>::value) {
      // for tiled_groups we know at compile time that whether we can call the ockl intrinsics or
      // not; if the block tile is actually the whole warp
      if (impl::tiledGroupSize<TyGroup>::value == warpSize) {
        if constexpr (impl::isArithmeticFunc<Val, Op>::value && impl::has_arithmetic_scan<Val>::value ||
                      impl::isBooleanFunc<Val, Op>::value && impl::has_boolean_scan<Val>::value) {
          return impl::call_scan<Val, Op, Inclusive>(val);
        }
      }
    } else if constexpr (impl::isCoalescedGroup<TyGroup>::value) {
      // for the coalesced_group case we do need to check at runtime, adding a slight overhead on
      // this branch
      if (maskNumBits == warpSize) {
        if constexpr (impl::isArithmeticFunc<Val, Op>::value && impl::has_arithmetic_scan<Val>::value ||
                      impl::isBooleanFunc<Val, Op>::value && impl::has_boolean_scan<Val>::value) {
          return impl::call_scan<Val, Op, Inclusive>(val);
        }
      }
    }
#endif

    // unsigned int[N] is used in some cases, e.g. when T is wider than 32-bit
    using ResultType = typename __hip_internal::conditional<
                         isPrimitiveType && (sizeof(Val) == 4 || sizeof(Val) == 2), permuteType,
                         permuteType[kNumOfPermutes]>::type;
    static constexpr int alignment = alignof(Val) <= 4? 4 : alignof(Val);
    alignas(alignment) ResultType result;
    alignas(alignment) ResultType permuteResult;

    if constexpr (isPrimitiveType && (sizeof(Val) == 2 || sizeof(Val) == 4)) {
      result = val;
    } else {
      __builtin_memcpy(result, &val, sizeof(Val));
    }

    // the number of iterations needs to be at least log2(number of bits on)
    numIterations = sizeof(int) * 8 - __clz(maskNumBits);

    if constexpr (impl::isTiledGroup<TyGroup>::value) {
      // the number of bits in the mask is always a power of 2 when
      // tiled blocks are used and in that case we need an iteration less
      numIterations -= 1;
    } else {
      // in the coalesced_threads case it depends, we are not sure whether
      // it is a power of 2, or not so we check
      if (!(maskNumBits & (maskNumBits - 1))) {
        numIterations -= 1;
      }
    }

    int modulo = 1;

    while (numIterations) {
      int offset = modulo >> 1;
      int increment = modulo - offset;
      int nextPos = maskIdx - offset - increment;
      bool insideLanes = nextPos >= 0;

      if (insideLanes) {
        int next;

        // find the position to aggregate with
        for (int i = 0; i < increment; i++) {
          next = __builtin_clzll(mask) + 1;
          mask <<= next;
          nextBit -= next;
        }
      }

      // clamp index; if out of bounds, the thread read its own value
      nextBit = (nextBit < (laneId & ~(warpSize - 1))) ? laneId : nextBit;
      bPermute<isPrimitiveType, permuteType, kNumOfPermutes>(permuteResult, result, nextBit);

      if (insideLanes) {
        if constexpr (!isPrimitiveType) {
          Val toReturn;
          toReturn = op(*reinterpret_cast<Val*>(permuteResult), *reinterpret_cast<Val*>(result));
        __builtin_memcpy(result, &toReturn, sizeof(Val));
        } else if constexpr (sizeof(Val) == 4 || sizeof(Val) == 2) {
          result = op(permuteResult, result);
        } else if constexpr (sizeof(Val) == 8) {
          Val tmp;
          unsigned long long rhs =
              (static_cast<unsigned long long>(permuteResult[1]) << 32) | permuteResult[0];
          __builtin_memcpy(&tmp, result, sizeof(Val));
          tmp = op(*reinterpret_cast<Val*>(&rhs), tmp);
          __builtin_memcpy(result, &tmp, sizeof(Val));
        }
      }

      modulo <<= 1;
      numIterations--;
    }

    if constexpr (Inclusive) {
      return *reinterpret_cast<Val*>(&result);
    } else {
      int nextBit = laneId;

      mask = impl::groupMask(group);

      if (laneId) {
        mask <<= 64 - laneId;
        nextBit -= mask? __builtin_clzll(mask) + 1 : 0;
      } else {
        mask = 0ull;
      }

      // clamp index; if out of bounds, the thread read its own value
      nextBit = (nextBit < (laneId & ~(warpSize - 1))) ? laneId : nextBit;
      bPermute<isPrimitiveType, permuteType, kNumOfPermutes>(permuteResult, result, nextBit);

      if (mask) {
        return *reinterpret_cast<Val*>(&permuteResult);
      } else {
        impl::CGIdentity<Val, Op> identity;

        return identity();
      }
    }
  }
}

  template <typename TyGroup, typename TyVal, typename TyFn>
  __CG_QUALIFIER__ auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op)
    -> decltype(op(val, val))
  {
    return impl::scan<true>(group, val, op);
  }

  template <typename TyGroup, typename TyVal, typename TyFn>
  __CG_QUALIFIER__ auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op)
    -> decltype(op(val, val))
  {
    return impl::scan<false>(group, val, op);
  }

  template <typename TyGroup, typename TyVal>
  __CG_QUALIFIER__ auto inclusive_scan(const TyGroup& group, TyVal&& val)
  {
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    return impl::scan<true>(group, val, cooperative_groups::plus<Val>());
  }

  template <typename TyGroup, typename TyVal>
  __CG_QUALIFIER__ auto exclusive_scan(const TyGroup& group, TyVal&& val)
  {
    using Val = typename __hip_internal::remove_cvref<TyVal>::type;

    return impl::scan<false>(group, val, cooperative_groups::plus<Val>());
  }
}
#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_SCAN_H
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef __cplusplus

#pragma push_macro("__HIP_ATOMICS_IGNORE_DENORMAL_MODE")
#if defined(__has_extension) && __has_extension(clang_atomic_attributes)
#define __HIP_ATOMICS_IGNORE_DENORMAL_MODE [[clang::atomic(ignore_denormal_mode)]]
#else
#define __HIP_ATOMICS_IGNORE_DENORMAL_MODE
#endif

/**
 * @brief Unsafe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 4 bytes aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and is not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicAdd(float* addr, float value) {
#if defined(__gfx90a__) && __has_builtin(__builtin_amdgcn_is_shared) &&                            \
    __has_builtin(__builtin_amdgcn_is_private) &&                                                  \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f32) &&                                          \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
  if (__builtin_amdgcn_is_shared((const __attribute__((address_space(0))) void*)addr)) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_ds_atomic_fadd_f32))
      return __builtin_amdgcn_ds_atomic_fadd_f32(addr, value);
    return 0.0f;
  } else if (__builtin_amdgcn_is_private((const __attribute__((address_space(0))) void*)addr)) {
    float temp = *addr;
    *addr = temp + value;
    return temp;
  } else {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_global_atomic_fadd_f32))
      return __builtin_amdgcn_global_atomic_fadd_f32(addr, value);
    return 0.0f;
  }
#elif __has_builtin(__hip_atomic_fetch_add)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe floating point rmw atomic max.
 *
 * Performs a relaxed read-modify-write floating point atomic max with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if greater.
 *
 * @note This operation is currently identical to that performed by
 * atomicMax and is included for completeness.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicMax(float* addr, float val) {
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    bool done = false;
    while (!done && value < val) {
      done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return value;
  }
#else
  unsigned int* uaddr = (unsigned int*)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false, __ATOMIC_RELAXED,
                                       __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
#endif
}

/**
 * @brief Unsafe floating point rmw atomic min.
 *
 * Performs a relaxed read-modify-write floating point atomic min with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if lesser.
 *
 * @note This operation is currently identical to that performed by
 * atomicMin and is included for completeness.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicMin(float* addr, float val) {
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    bool done = false;
    while (!done && value > val) {
      done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return value;
  }
#else
  unsigned int* uaddr = (unsigned int*)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false, __ATOMIC_RELAXED,
                                       __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
#endif
}

/**
 * @brief Unsafe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double unsafeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) && __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f64)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_f64))
    return __builtin_amdgcn_flat_atomic_fadd_f64(addr, value);
  return 0.0;
#elif __has_builtin(__hip_atomic_fetch_add)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe double precision rmw atomic max.
 *
 * Performs a relaxed read-modify-write double precision atomic max with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if greater.
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double unsafeAtomicMax(double* addr, double val) {
#if (defined(__gfx90a__) || defined(__gfx94plus_clr__)) &&                                         \
    __has_builtin(__builtin_amdgcn_flat_atomic_fmax_f64)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fmax_f64))
    return __builtin_amdgcn_flat_atomic_fmax_f64(addr, val);
  return 0.0;
#else
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    bool done = false;
    while (!done && value < val) {
      done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return value;
  }
#else
  unsigned long long* uaddr = (unsigned long long*)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
#endif
#endif
}

/**
 * @brief Unsafe double precision rmw atomic min.
 *
 * Performs a relaxed read-modify-write double precision atomic min with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if lesser.
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double unsafeAtomicMin(double* addr, double val) {
#if (defined(__gfx90a__) || defined(__gfx94plus_clr__)) &&                                         \
    __has_builtin(__builtin_amdgcn_flat_atomic_fmin_f64)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fmin_f64))
    return __builtin_amdgcn_flat_atomic_fmin_f64(addr, val);
  return 0.0;
#else
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    bool done = false;
    while (!done && value > val) {
      done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return value;
  }
#else
  unsigned long long* uaddr = (unsigned long long*)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
#endif
#endif
}

/**
 * @brief Safe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicAdd(float* addr, float value) {
#if defined(__gfx908__) || ((defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)) && \
                            !__has_builtin(__hip_atomic_fetch_add))
  // On gfx908, we can generate unsafe FP32 atomic add that does not follow all
  // IEEE rules when -munsafe-fp-atomics is passed. Do a CAS loop emulation instead.
  // On gfx90a, gfx942 and gfx950 if we do not have the __hip_atomic_fetch_add builtin, we
  // need to force a CAS loop here.
  float old_val;
#if __has_builtin(__hip_atomic_load)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#else   // !__has_builtin(__hip_atomic_load)
  old_val =
      __uint_as_float(__atomic_load_n(reinterpret_cast<unsigned int*>(addr), __ATOMIC_RELAXED));
#endif  // __has_builtin(__hip_atomic_load)
  float expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
      __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
#else   // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false, __ATOMIC_RELAXED,
                                __ATOMIC_RELAXED);
#endif  // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__float_as_uint(temp) != __float_as_uint(old_val));
  return old_val;
#elif defined(__gfx90a__)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#elif __has_builtin(__hip_atomic_fetch_add)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Safe floating point rmw atomic max.
 *
 * Performs a relaxed read-modify-write floating point atomic max with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if greater.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicMax(float* addr, float val) {
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    bool done = false;
    while (!done && value < val) {
      done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return value;
  }
#else
  unsigned int* uaddr = (unsigned int*)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false, __ATOMIC_RELAXED,
                                       __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
#endif
}

/**
 * @brief Safe floating point rmw atomic min.
 *
 * Performs a relaxed read-modify-write floating point atomic min with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if lesser.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicMin(float* addr, float val) {
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    bool done = false;
    while (!done && value > val) {
      done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return value;
  }
#else
  unsigned int* uaddr = (unsigned int*)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false, __ATOMIC_RELAXED,
                                       __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
#endif
}

/**
 * @brief Safe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double safeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) && __has_builtin(__hip_atomic_fetch_add)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#elif defined(__gfx90a__)
  // On gfx90a, if we do not have the __hip_atomic_fetch_add builtin, we need to
  // force a CAS loop here.
  double old_val;
#if __has_builtin(__hip_atomic_load)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#else   // !__has_builtin(__hip_atomic_load)
  old_val = __longlong_as_double(
      __atomic_load_n(reinterpret_cast<unsigned long long*>(addr), __ATOMIC_RELAXED));
#endif  // __has_builtin(__hip_atomic_load)
  double expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
      __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
#else   // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false, __ATOMIC_RELAXED,
                                __ATOMIC_RELAXED);
#endif  // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__double_as_longlong(temp) != __double_as_longlong(old_val));
  return old_val;
#else   // !defined(__gfx90a__)
#if __has_builtin(__hip_atomic_fetch_add)
  __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
    return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#else   // !__has_builtin(__hip_atomic_fetch_add)
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif  // __has_builtin(__hip_atomic_fetch_add)
#endif
}

/**
 * @brief Safe double precision rmw atomic max.
 *
 * Performs a relaxed read-modify-write double precision atomic max with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if greater.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double safeAtomicMax(double* addr, double val) {
#if __has_builtin(__builtin_amdgcn_is_private)
  if (__builtin_amdgcn_is_private((const __attribute__((address_space(0))) void*)addr)) {
    double old = *addr;
    *addr = __builtin_fmax(old, val);
    return old;
  } else {
#endif
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
    __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
      double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      bool done = false;
      while (!done && value < val) {
        done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      return value;
    }
#else
  unsigned long long* uaddr = (unsigned long long*)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
#endif
#if __has_builtin(__builtin_amdgcn_is_private)
  }
#endif
}

/**
 * @brief Safe double precision rmw atomic min.
 *
 * Performs a relaxed read-modify-write double precision atomic min with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if lesser.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double safeAtomicMin(double* addr, double val) {
#if __has_builtin(__builtin_amdgcn_is_private)
  if (__builtin_amdgcn_is_private((const __attribute__((address_space(0))) void*)addr)) {
    double old = *addr;
    *addr = __builtin_fmin(old, val);
    return old;
  } else {
#endif
#if __has_builtin(__hip_atomic_load) && __has_builtin(__hip_atomic_compare_exchange_strong)
    __HIP_ATOMICS_IGNORE_DENORMAL_MODE {
      double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      bool done = false;
      while (!done && value > val) {
        done = __hip_atomic_compare_exchange_strong(addr, &value, val, __ATOMIC_RELAXED,
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      return value;
    }
#else
  unsigned long long* uaddr = (unsigned long long*)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
#endif
#if __has_builtin(__builtin_amdgcn_is_private)
  }
#endif
}

#pragma pop_macro("__HIP_ATOMICS_IGNORE_DENORMAL_MODE")

#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "amd_device_functions.h"
#endif

#if !defined(__HIP_ATOMIC_BACKWARD_COMPAT)
#define __HIP_ATOMIC_BACKWARD_COMPAT 1
#endif

#if defined(__has_extension) && __has_extension(clang_atomic_attributes) && __HIP_ATOMIC_BACKWARD_COMPAT
#define __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY [[clang::atomic(fine_grained_memory, remote_memory)]]
#else
#define __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY
#endif

template <bool B, typename T, typename F> struct Cond_t;

template <typename T, typename F> struct Cond_t<true, T, F> {
  using type = T;
};
template <typename T, typename F> struct Cond_t<false, T, F> {
  using type = F;
};

#if !__HIP_DEVICE_COMPILE__
// TODO: Remove this after compiler pre-defines the following Macros.
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_unsafe_atomics.h"
#endif

// Atomic expanders
template <int mem_order = __ATOMIC_SEQ_CST, int mem_scope = __HIP_MEMORY_SCOPE_SYSTEM, typename T,
          typename Op, typename F>
inline __attribute__((always_inline, device)) T hip_cas_expander(T* p, T x, Op op, F f) noexcept {
  using FP = __attribute__((address_space(0))) const void*;

  __device__ extern bool is_shared_workaround(FP) asm("llvm.amdgcn.is.shared");

  if (is_shared_workaround((FP)p)) return f();

  using U =
      typename Cond_t<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;

  auto q = reinterpret_cast<U*>(p);

  U tmp0{__hip_atomic_load(q, mem_order, mem_scope)};
  U tmp1;
  do {
    tmp1 = tmp0;

    op(reinterpret_cast<T&>(tmp1), x);
  } while (!__hip_atomic_compare_exchange_strong(q, &tmp0, tmp1, mem_order, mem_order, mem_scope));

  return reinterpret_cast<const T&>(tmp0);
}

template <int mem_order = __ATOMIC_SEQ_CST, int mem_scope = __HIP_MEMORY_SCOPE_SYSTEM, typename T,
          typename Cmp, typename F>
inline __attribute__((always_inline, device)) T hip_cas_extrema_expander(T* p, T x, Cmp cmp,
                                                                         F f) noexcept {
  using FP = __attribute__((address_space(0))) const void*;

  __device__ extern bool is_shared_workaround(FP) asm("llvm.amdgcn.is.shared");

  if (is_shared_workaround((FP)p)) return f();

  using U =
      typename Cond_t<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;

  auto q = reinterpret_cast<U*>(p);

  U tmp{__hip_atomic_load(q, mem_order, mem_scope)};
  while (cmp(x, reinterpret_cast<const T&>(tmp)) &&
         !__hip_atomic_compare_exchange_strong(q, &tmp, x, mem_order, mem_order, mem_scope));

  return reinterpret_cast<const T&>(tmp);
}

__device__ inline unsigned short int atomicCAS(unsigned short int* address,
                                               unsigned short int compare, unsigned short int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned short int atomicCAS_system(unsigned short int* address,
                                                      unsigned short int compare,
                                                      unsigned short int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline int atomicCAS(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline int atomicCAS_system(int* address, int compare, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline unsigned int atomicCAS(unsigned int* address, unsigned int compare,
                                         unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned int atomicCAS_system(unsigned int* address, unsigned int compare,
                                                unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline unsigned long atomicCAS(unsigned long* address, unsigned long compare,
                                          unsigned long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned long atomicCAS_system(unsigned long* address, unsigned long compare,
                                                 unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline unsigned long long atomicCAS(unsigned long long* address,
                                               unsigned long long compare, unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline unsigned long long atomicCAS_system(unsigned long long* address,
                                                      unsigned long long compare,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline float atomicCAS(float* address, float compare, float val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline float atomicCAS_system(float* address, float compare, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline double atomicCAS(double* address, double compare, double val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__ inline double atomicCAS_system(double* address, double compare, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
  return compare;
}

__device__ inline int atomicAdd(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicAdd_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicAdd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicAdd_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicAdd_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicAdd(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicAdd_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicAdd(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicAdd_system(float* address, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

#if !defined(__HIPCC_RTC__)
HIP_DEPRECATED("use atomicAdd instead")
#endif  // !defined(__HIPCC_RTC__)
__device__ inline void atomicAddNoRet(float* address, float val) { unsafeAtomicAdd(address, val); }

__device__ inline double atomicAdd(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicAdd_system(double* address, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicSub(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicSub_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicSub(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicSub_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicSub(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicSub_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicSub(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicSub_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicSub(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicSub_system(float* address, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline double atomicSub(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicSub_system(double* address, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicExch(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicExch_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicExch(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicExch_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicExch(unsigned long* address, unsigned long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicExch_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicExch(unsigned long long* address,
                                                unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicExch_system(unsigned long long* address,
                                                       unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicExch(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline float atomicExch_system(float* address, float val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline double atomicExch(double* address, double val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline double atomicExch_system(double* address, double val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicMin(int* address, int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicMin_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicMin(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicMin_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicMin(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicMin_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicMin(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicMin_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline long long atomicMin(long long* address, long long val) {
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline long long atomicMin_system(long long* address, long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicMin(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicMin_system(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline double atomicMin(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicMin_system(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_min(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline int atomicMax(int* address, int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicMax_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicMax(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicMax_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicMax(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicMax_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicMax(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicMax_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
__device__ inline long long atomicMax(long long* address, long long val) {
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline long long atomicMax_system(long long* address, long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline float atomicMax(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline float atomicMax_system(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline double atomicMax(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
#endif
}

__device__ inline double atomicMax_system(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_max(addr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
#endif
}

__device__ inline unsigned int atomicInc(unsigned int* address, unsigned int val) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_atomic_inc32))
    return __builtin_amdgcn_atomic_inc32(address, val, __ATOMIC_RELAXED, "agent");
  return 0;
}

__device__ inline unsigned int atomicDec(unsigned int* address, unsigned int val) {
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_atomic_dec32))
    return __builtin_amdgcn_atomic_dec32(address, val, __ATOMIC_RELAXED, "agent");
  return 0;
}

__device__ inline int atomicAnd(int* address, int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicAnd_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicAnd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicAnd_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicAnd(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicAnd_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicAnd(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicAnd_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicOr(int* address, int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicOr_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicOr(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicOr_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicOr(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicOr_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicOr(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicOr_system(unsigned long long* address,
                                                     unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline int atomicXor(int* address, int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline int atomicXor_system(int* address, int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned int atomicXor(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned int atomicXor_system(unsigned int* address, unsigned int val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long atomicXor(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long atomicXor_system(unsigned long* address, unsigned long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__device__ inline unsigned long long atomicXor(unsigned long long* address,
                                               unsigned long long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ inline unsigned long long atomicXor_system(unsigned long long* address,
                                                      unsigned long long val) {
  __HIP_ATOMIC_BACKWARD_COMPAT_MEMORY {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#include "amd_hip_vector_types.h"  // For Native_vec_
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__device__ __attribute__((const)) int __ockl_sdot2(HIP_vector_base<short, 2>::Native_vec_,
                                                   HIP_vector_base<short, 2>::Native_vec_, int,
                                                   bool);

__device__ __attribute__((const)) unsigned int __ockl_udot2(
    HIP_vector_base<unsigned short, 2>::Native_vec_,
    HIP_vector_base<unsigned short, 2>::Native_vec_, unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot4(HIP_vector_base<char, 4>::Native_vec_,
                                                   HIP_vector_base<char, 4>::Native_vec_, int,
                                                   bool);

__device__ __attribute__((const)) unsigned int __ockl_udot4(
    HIP_vector_base<unsigned char, 4>::Native_vec_, HIP_vector_base<unsigned char, 4>::Native_vec_,
    unsigned int, bool);

__device__ __attribute__((const)) int __ockl_sdot8(int, int, int, bool);

__device__ __attribute__((const)) unsigned int __ockl_udot8(unsigned int, unsigned int,
                                                            unsigned int, bool);
#endif

// BEGIN FLOAT
__device__ __attribute__((const)) float __ocml_acos_f32(float);
__device__ __attribute__((pure)) float __ocml_acosh_f32(float);
__device__ __attribute__((const)) float __ocml_asin_f32(float);
__device__ __attribute__((pure)) float __ocml_asinh_f32(float);
__device__ __attribute__((const)) float __ocml_atan2_f32(float, float);
__device__ __attribute__((const)) float __ocml_atan_f32(float);
__device__ __attribute__((pure)) float __ocml_atanh_f32(float);
__device__ __attribute__((pure)) float __ocml_cbrt_f32(float);
__device__ __attribute__((const)) float __ocml_ceil_f32(float);
__device__ __attribute__((const)) __device__ float __ocml_copysign_f32(float, float);
__device__ float __ocml_cos_f32(float);
__device__ float __ocml_native_cos_f32(float);
__device__ __attribute__((pure)) __device__ float __ocml_cosh_f32(float);
__device__ float __ocml_cospi_f32(float);
__device__ float __ocml_i0_f32(float);
__device__ float __ocml_i1_f32(float);
__device__ __attribute__((pure)) float __ocml_erfc_f32(float);
__device__ __attribute__((pure)) float __ocml_erfcinv_f32(float);
__device__ __attribute__((pure)) float __ocml_erfcx_f32(float);
__device__ __attribute__((pure)) float __ocml_erf_f32(float);
__device__ __attribute__((pure)) float __ocml_erfinv_f32(float);
__device__ __attribute__((pure)) float __ocml_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp10_f32(float);
__device__ __attribute__((pure)) float __ocml_exp2_f32(float);
__device__ __attribute__((pure)) float __ocml_exp_f32(float);
__device__ __attribute__((pure)) float __ocml_native_exp_f32(float);
__device__ __attribute__((pure)) float __ocml_expm1_f32(float);
__device__ __attribute__((const)) float __ocml_fabs_f32(float);
__device__ __attribute__((const)) float __ocml_fdim_f32(float, float);
__device__ __attribute__((const)) float __ocml_floor_f32(float);
__device__ __attribute__((const)) float __ocml_fma_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fmax_f32(float, float);
__device__ __attribute__((const)) float __ocml_fmin_f32(float, float);
__device__ __attribute__((const)) __device__ float __ocml_fmod_f32(float, float);
__device__ float __ocml_frexp_f32(float, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) float __ocml_hypot_f32(float, float);
__device__ __attribute__((const)) int __ocml_ilogb_f32(float);
__device__ __attribute__((const)) int __ocml_isfinite_f32(float);
__device__ __attribute__((const)) int __ocml_isinf_f32(float);
__device__ __attribute__((const)) int __ocml_isnan_f32(float);
__device__ float __ocml_j0_f32(float);
__device__ float __ocml_j1_f32(float);
__device__ __attribute__((const)) float __ocml_ldexp_f32(float, int);
__device__ float __ocml_lgamma_f32(float);
__device__ __attribute__((pure)) float __ocml_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log10_f32(float);
__device__ __attribute__((pure)) float __ocml_log1p_f32(float);
__device__ __attribute__((pure)) float __ocml_log2_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log2_f32(float);
__device__ __attribute__((const)) float __ocml_logb_f32(float);
__device__ __attribute__((pure)) float __ocml_log_f32(float);
__device__ __attribute__((pure)) float __ocml_native_log_f32(float);
__device__ float __ocml_modf_f32(float, __attribute__((address_space(5))) float*);
__device__ __attribute__((const)) float __ocml_nearbyint_f32(float);
__device__ __attribute__((const)) float __ocml_nextafter_f32(float, float);
__device__ __attribute__((const)) float __ocml_len3_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_len4_f32(float, float, float, float);
__device__ __attribute__((pure)) float __ocml_ncdf_f32(float);
__device__ __attribute__((pure)) float __ocml_ncdfinv_f32(float);
__device__ __attribute__((pure)) float __ocml_pow_f32(float, float);
__device__ __attribute__((pure)) float __ocml_pown_f32(float, int);
__device__ __attribute__((pure)) float __ocml_rcbrt_f32(float);
__device__ __attribute__((const)) float __ocml_remainder_f32(float, float);
__device__ float __ocml_remquo_f32(float, float, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) float __ocml_rhypot_f32(float, float);
__device__ __attribute__((const)) float __ocml_rint_f32(float);
__device__ __attribute__((const)) float __ocml_rlen3_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_rlen4_f32(float, float, float, float);
__device__ __attribute__((const)) float __ocml_round_f32(float);
__device__ __attribute__((pure)) float __ocml_rsqrt_f32(float);
__device__ __attribute__((const)) float __ocml_scalb_f32(float, float);
__device__ __attribute__((const)) float __ocml_scalbn_f32(float, int);
__device__ __attribute__((const)) int __ocml_signbit_f32(float);
__device__ float __ocml_sincos_f32(float, __attribute__((address_space(5))) float*);
__device__ float __ocml_sincospi_f32(float, __attribute__((address_space(5))) float*);
__device__ float __ocml_sin_f32(float);
__device__ float __ocml_native_sin_f32(float);
__device__ __attribute__((pure)) float __ocml_sinh_f32(float);
__device__ float __ocml_sinpi_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_f32(float);
__device__ __attribute__((const)) float __ocml_native_sqrt_f32(float);
__device__ float __ocml_tan_f32(float);
__device__ __attribute__((pure)) float __ocml_tanh_f32(float);
__device__ float __ocml_tgamma_f32(float);
__device__ __attribute__((const)) float __ocml_trunc_f32(float);
__device__ float __ocml_y0_f32(float);
__device__ float __ocml_y1_f32(float);

// BEGIN INTRINSICS
__device__ __attribute__((const)) float __ocml_add_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_add_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_sub_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_mul_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rte_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtn_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtp_f32(float, float);
__device__ __attribute__((const)) float __ocml_div_rtz_f32(float, float);
__device__ __attribute__((const)) float __ocml_sqrt_rte_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtn_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtp_f32(float);
__device__ __attribute__((const)) float __ocml_sqrt_rtz_f32(float);
__device__ __attribute__((const)) float __ocml_fma_rte_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtn_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtp_f32(float, float, float);
__device__ __attribute__((const)) float __ocml_fma_rtz_f32(float, float, float);
// END INTRINSICS
// END FLOAT

// BEGIN DOUBLE
__device__ __attribute__((const)) double __ocml_acos_f64(double);
__device__ __attribute__((pure)) double __ocml_acosh_f64(double);
__device__ __attribute__((const)) double __ocml_asin_f64(double);
__device__ __attribute__((pure)) double __ocml_asinh_f64(double);
__device__ __attribute__((const)) double __ocml_atan2_f64(double, double);
__device__ __attribute__((const)) double __ocml_atan_f64(double);
__device__ __attribute__((pure)) double __ocml_atanh_f64(double);
__device__ __attribute__((pure)) double __ocml_cbrt_f64(double);
__device__ __attribute__((const)) double __ocml_ceil_f64(double);
__device__ __attribute__((const)) double __ocml_copysign_f64(double, double);
__device__ double __ocml_cos_f64(double);
__device__ __attribute__((pure)) double __ocml_cosh_f64(double);
__device__ double __ocml_cospi_f64(double);
__device__ double __ocml_i0_f64(double);
__device__ double __ocml_i1_f64(double);
__device__ __attribute__((pure)) double __ocml_erfc_f64(double);
__device__ __attribute__((pure)) double __ocml_erfcinv_f64(double);
__device__ __attribute__((pure)) double __ocml_erfcx_f64(double);
__device__ __attribute__((pure)) double __ocml_erf_f64(double);
__device__ __attribute__((pure)) double __ocml_erfinv_f64(double);
__device__ __attribute__((pure)) double __ocml_exp10_f64(double);
__device__ __attribute__((pure)) double __ocml_exp2_f64(double);
__device__ __attribute__((pure)) double __ocml_exp_f64(double);
__device__ __attribute__((pure)) double __ocml_expm1_f64(double);
__device__ __attribute__((const)) double __ocml_fabs_f64(double);
__device__ __attribute__((const)) double __ocml_fdim_f64(double, double);
__device__ __attribute__((const)) double __ocml_floor_f64(double);
__device__ __attribute__((const)) double __ocml_fma_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fmax_f64(double, double);
__device__ __attribute__((const)) double __ocml_fmin_f64(double, double);
__device__ __attribute__((const)) double __ocml_fmod_f64(double, double);
__device__ double __ocml_frexp_f64(double, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) double __ocml_hypot_f64(double, double);
__device__ __attribute__((const)) int __ocml_ilogb_f64(double);
__device__ __attribute__((const)) int __ocml_isfinite_f64(double);
__device__ __attribute__((const)) int __ocml_isinf_f64(double);
__device__ __attribute__((const)) int __ocml_isnan_f64(double);
__device__ double __ocml_j0_f64(double);
__device__ double __ocml_j1_f64(double);
__device__ __attribute__((const)) double __ocml_ldexp_f64(double, int);
__device__ double __ocml_lgamma_f64(double);
__device__ __attribute__((pure)) double __ocml_log10_f64(double);
__device__ __attribute__((pure)) double __ocml_log1p_f64(double);
__device__ __attribute__((pure)) double __ocml_log2_f64(double);
__device__ __attribute__((const)) double __ocml_logb_f64(double);
__device__ __attribute__((pure)) double __ocml_log_f64(double);
__device__ double __ocml_modf_f64(double, __attribute__((address_space(5))) double*);
__device__ __attribute__((const)) double __ocml_nearbyint_f64(double);
__device__ __attribute__((const)) double __ocml_nextafter_f64(double, double);
__device__ __attribute__((const)) double __ocml_len3_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_len4_f64(double, double, double, double);
__device__ __attribute__((pure)) double __ocml_ncdf_f64(double);
__device__ __attribute__((pure)) double __ocml_ncdfinv_f64(double);
__device__ __attribute__((pure)) double __ocml_pow_f64(double, double);
__device__ __attribute__((pure)) double __ocml_pown_f64(double, int);
__device__ __attribute__((pure)) double __ocml_rcbrt_f64(double);
__device__ __attribute__((const)) double __ocml_remainder_f64(double, double);
__device__ double __ocml_remquo_f64(double, double, __attribute__((address_space(5))) int*);
__device__ __attribute__((const)) double __ocml_rhypot_f64(double, double);
__device__ __attribute__((const)) double __ocml_rint_f64(double);
__device__ __attribute__((const)) double __ocml_rlen3_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_rlen4_f64(double, double, double, double);
__device__ __attribute__((const)) double __ocml_round_f64(double);
__device__ __attribute__((pure)) double __ocml_rsqrt_f64(double);
__device__ __attribute__((const)) double __ocml_scalb_f64(double, double);
__device__ __attribute__((const)) double __ocml_scalbn_f64(double, int);
__device__ __attribute__((const)) int __ocml_signbit_f64(double);
__device__ double __ocml_sincos_f64(double, __attribute__((address_space(5))) double*);
__device__ double __ocml_sincospi_f64(double, __attribute__((address_space(5))) double*);
__device__ double __ocml_sin_f64(double);
__device__ __attribute__((pure)) double __ocml_sinh_f64(double);
__device__ double __ocml_sinpi_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_f64(double);
__device__ double __ocml_tan_f64(double);
__device__ __attribute__((pure)) double __ocml_tanh_f64(double);
__device__ double __ocml_tgamma_f64(double);
__device__ __attribute__((const)) double __ocml_trunc_f64(double);
__device__ double __ocml_y0_f64(double);
__device__ double __ocml_y1_f64(double);

// BEGIN INTRINSICS
__device__ __attribute__((const)) double __ocml_add_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_add_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_sub_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_mul_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rte_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtn_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtp_f64(double, double);
__device__ __attribute__((const)) double __ocml_div_rtz_f64(double, double);
__device__ __attribute__((const)) double __ocml_sqrt_rte_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtn_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtp_f64(double);
__device__ __attribute__((const)) double __ocml_sqrt_rtz_f64(double);
__device__ __attribute__((const)) double __ocml_fma_rte_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fma_rtn_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fma_rtp_f64(double, double, double);
__device__ __attribute__((const)) double __ocml_fma_rtz_f64(double, double, double);
// END INTRINSICS
// END DOUBLE

#if defined(__cplusplus)
}  // extern "C"
#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

// /*
// Half Math Functions
// */
#if !defined(__HIPCC_RTC__)
#include "host_defines.h"
#endif
extern "C" {
__device__ __attribute__((const)) _Float16 __ocml_ceil_f16(_Float16);
__device__ _Float16 __ocml_cos_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_exp_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_exp10_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_exp2_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_floor_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_fma_f16(_Float16, _Float16, _Float16);
__device__ __attribute__((const)) _Float16 __ocml_fabs_f16(_Float16);
__device__ __attribute__((const)) int __ocml_isinf_f16(_Float16);
__device__ __attribute__((const)) int __ocml_isnan_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_log_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_log10_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_log2_f16(_Float16);
__device__ __attribute__((pure)) _Float16 __ocml_pown_f16(_Float16, int);
__device__ __attribute__((const)) _Float16 __ocml_rint_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_rsqrt_f16(_Float16);
__device__ _Float16 __ocml_sin_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_sqrt_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_trunc_f16(_Float16);
__device__ __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
__device__ __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);

typedef _Float16 __2f16 __attribute__((ext_vector_type(2)));
typedef short __2i16 __attribute__((ext_vector_type(2)));

__device__ __attribute__((const)) float __ockl_fdot2(__2f16 a, __2f16 b, float c, bool s);
__device__ __attribute__((const)) __2f16 __ocml_ceil_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_fabs_2f16(__2f16);
__device__ __2f16 __ocml_cos_2f16(__2f16);
__device__ __attribute__((pure)) __2f16 __ocml_exp_2f16(__2f16);
__device__ __attribute__((pure)) __2f16 __ocml_exp10_2f16(__2f16);
__device__ __attribute__((pure)) __2f16 __ocml_exp2_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_floor_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_fma_2f16(__2f16, __2f16, __2f16);
__device__ __attribute__((const)) __2i16 __ocml_isinf_2f16(__2f16);
__device__ __attribute__((const)) __2i16 __ocml_isnan_2f16(__2f16);
__device__ __attribute__((pure)) __2f16 __ocml_log_2f16(__2f16);
__device__ __attribute__((pure)) __2f16 __ocml_log10_2f16(__2f16);
__device__ __attribute__((pure)) __2f16 __ocml_log2_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_rint_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_rsqrt_2f16(__2f16);
__device__ __2f16 __ocml_sin_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_sqrt_2f16(__2f16);
__device__ __attribute__((const)) __2f16 __ocml_trunc_2f16(__2f16);

__device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
}
// TODO: remove these after they get into clang header __clang_hip_libdevice_declares.h'
extern "C" {
__device__ __attribute__((const)) _Float16 __ocml_fmax_f16(_Float16, _Float16);
__device__ __attribute__((const)) _Float16 __ocml_fmin_f16(_Float16, _Float16);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtn_f16_f32(float);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtp_f16_f32(float);
__device__ __attribute__((const)) _Float16 __ocml_cvtrtz_f16_f32(float);
}
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H

#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#define __HOST_DEVICE__ __host__ __device__
#include "amd_hip_common.h"
#include "host_defines.h"
#include "amd_hip_vector_types.h"
#include <assert.h>
#if defined(__cplusplus)
#include <algorithm>
#include <type_traits>
#include <utility>
#endif
#endif  // !defined(__HIPCC_RTC__)

#define HIPRT_INF_FP16 __ushort_as_half((unsigned short)0x7C00U)
#define HIPRT_MAX_NORMAL_FP16 __ushort_as_half((unsigned short)0x7BFFU)
#define HIPRT_MIN_DENORM_FP16 __ushort_as_half((unsigned short)0x0001U)
#define HIPRT_NAN_FP16 __ushort_as_half((unsigned short)0x7FFFU)
#define HIPRT_NEG_ZERO_FP16 __ushort_as_half((unsigned short)0x8000U)
#define HIPRT_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)
#define HIPRT_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)

#if defined(__clang__) && defined(__HIP__)
typedef _Float16 _Float16_2 __attribute__((ext_vector_type(2)));

struct __half_raw {
  union {
    static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

    _Float16 data;
    unsigned short x;
  };
};

struct __half2_raw {
  union {
    static_assert(sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

    struct {
      __half_raw x;
      __half_raw y;
    };
    _Float16_2 data;
  };
};

#if defined(__cplusplus)
#if !defined(__HIPCC_RTC__)
#include "hip_fp16_math_fwd.h"
#include "host_defines.h"
#include "amd_device_functions.h"
#include "amd_warp_functions.h"
#endif
namespace __hip_internal {
template <> struct is_floating_point<_Float16> : __hip_internal::true_type {};
}  // namespace __hip_internal

template <bool cond, typename T = void> using Enable_if_t =
    typename __hip_internal::enable_if<cond, T>::type;

// BEGIN STRUCT __HALF
struct __half {
 protected:
  union {
    static_assert(sizeof(_Float16) == sizeof(unsigned short), "");

    _Float16 data;
    unsigned short __x;
  };

 public:
  // CREATORS
  __HOST_DEVICE__
  __half() = default;
  __HOST_DEVICE__ constexpr __half(const __half_raw& x) : __x{x.x} {}
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  __HOST_DEVICE__
  __half(decltype(data) x) : data{x} {}
  template <typename T, Enable_if_t<__hip_internal::is_floating_point<T>{}>* = nullptr>
  __HOST_DEVICE__ __half(T x) : data{static_cast<_Float16>(x)} {}
#endif
  __HOST_DEVICE__
  __half(const __half&) = default;
  __HOST_DEVICE__
  __half(__half&&) = default;
  __HOST_DEVICE__
  ~__half() = default;

// CREATORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<__hip_internal::is_integral<T>{}>* = nullptr>
  __HOST_DEVICE__ __half(T x) : data{static_cast<_Float16>(x)} {}
#endif

  // MANIPULATORS
  __HOST_DEVICE__
  __half& operator=(const __half&) = default;
  __HOST_DEVICE__
  __half& operator=(__half&&) = default;
  __HOST_DEVICE__
  __half& operator=(const __half_raw& x) {
    data = x.data;
    return *this;
  }
  __HOST_DEVICE__
  volatile __half& operator=(const __half_raw& x) volatile {
    data = x.data;
    return *this;
  }
  volatile __half& operator=(const volatile __half_raw& x) volatile {
    data = x.data;
    return *this;
  }
  __half& operator=(__half_raw&& x) {
    data = x.data;
    return *this;
  }
  volatile __half& operator=(__half_raw&& x) volatile {
    data = x.data;
    return *this;
  }
  volatile __half& operator=(volatile __half_raw&& x) volatile {
    data = x.data;
    return *this;
  }
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<__hip_internal::is_floating_point<T>{}>* = nullptr>
  __HOST_DEVICE__ __half& operator=(T x) {
    data = static_cast<_Float16>(x);
    return *this;
  }
#endif

// MANIPULATORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<__hip_internal::is_integral<T>{}>* = nullptr>
  __device__ __half& operator=(T x) {
    data = static_cast<_Float16>(x);
    return *this;
  }
#endif

#if !defined(__HIP_NO_HALF_OPERATORS__)
  __HOST_DEVICE__
  __half& operator+=(const __half& x) {
    data += x.data;
    return *this;
  }
  __HOST_DEVICE__
  __half& operator-=(const __half& x) {
    data -= x.data;
    return *this;
  }
  __HOST_DEVICE__
  __half& operator*=(const __half& x) {
    data *= x.data;
    return *this;
  }
  __HOST_DEVICE__
  __half& operator/=(const __half& x) {
    data /= x.data;
    return *this;
  }
  __HOST_DEVICE__
  __half& operator++() {
    ++data;
    return *this;
  }
  __HOST_DEVICE__
  __half operator++(int) {
    __half tmp{*this};
    ++*this;
    return tmp;
  }
  __HOST_DEVICE__
  __half& operator--() {
    --data;
    return *this;
  }
  __HOST_DEVICE__
  __half operator--(int) {
    __half tmp{*this};
    --*this;
    return tmp;
  }
#endif

// ACCESSORS
#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<__hip_internal::is_floating_point<T>{}>* = nullptr>
  __HOST_DEVICE__ operator T() const {
    return data;
  }
#endif
  __HOST_DEVICE__
  operator __half_raw() const { return __half_raw{data}; }
  __HOST_DEVICE__
  operator __half_raw() const volatile { return __half_raw{data}; }

#if !defined(__HIP_NO_HALF_CONVERSIONS__)
  template <typename T, Enable_if_t<__hip_internal::is_integral<T>{}>* = nullptr>
  __HOST_DEVICE__ operator T() const {
    return data;
  }
#endif

#if !defined(__HIP_NO_HALF_OPERATORS__)
  __HOST_DEVICE__
  __half operator+() const { return *this; }
  __HOST_DEVICE__
  __half operator-() const {
    __half tmp{*this};
    tmp.data = -tmp.data;
    return tmp;
  }
#endif

// FRIENDS
#if !defined(__HIP_NO_HALF_OPERATORS__)
  friend inline __HOST_DEVICE__ __half operator+(const __half& x, const __half& y) {
    return __half{x} += y;
  }
  friend inline __HOST_DEVICE__ __half operator-(const __half& x, const __half& y) {
    return __half{x} -= y;
  }
  friend inline __HOST_DEVICE__ __half operator*(const __half& x, const __half& y) {
    return __half{x} *= y;
  }
  friend inline __HOST_DEVICE__ __half operator/(const __half& x, const __half& y) {
    return __half{x} /= y;
  }
  friend inline __HOST_DEVICE__ bool operator==(const __half& x, const __half& y) {
    return x.data == y.data;
  }
  friend inline __HOST_DEVICE__ bool operator!=(const __half& x, const __half& y) {
    return !(x == y);
  }
  friend inline __HOST_DEVICE__ bool operator<(const __half& x, const __half& y) {
    return x.data < y.data;
  }
  friend inline __HOST_DEVICE__ bool operator>(const __half& x, const __half& y) {
    return y.data < x.data;
  }
  friend inline __HOST_DEVICE__ bool operator<=(const __half& x, const __half& y) {
    return !(y < x);
  }
  friend inline __HOST_DEVICE__ bool operator>=(const __half& x, const __half& y) {
    return !(x < y);
  }
#endif  // !defined(__HIP_NO_HALF_OPERATORS__)
};
// END STRUCT __HALF

// BEGIN STRUCT __HALF2
struct __half2 {
 public:
  union {
    static_assert(sizeof(_Float16_2) == sizeof(unsigned short[2]), "");

    struct {
      __half x;
      __half y;
    };
    _Float16_2 data;
  };

  // CREATORS
  __HOST_DEVICE__
  __half2() = default;
  __HOST_DEVICE__
  __half2(const __half2_raw& xx) : data{xx.data} {}
  __HOST_DEVICE__
  __half2(decltype(data) xx) : data{xx} {}
  __HOST_DEVICE__ constexpr __half2(const __half& xx, const __half& yy) : x(xx), y(yy) {}
  __HOST_DEVICE__
  __half2(const __half2&) = default;
  __HOST_DEVICE__
  __half2(__half2&&) = default;
  __HOST_DEVICE__
  ~__half2() = default;

  // MANIPULATORS
  __HOST_DEVICE__
  __half2& operator=(const __half2&) = default;
  __HOST_DEVICE__
  __half2& operator=(__half2&&) = default;
  __HOST_DEVICE__
  __half2& operator=(const __half2_raw& xx) {
    data = xx.data;
    return *this;
  }

// MANIPULATORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_OPERATORS__)
  __HOST_DEVICE__
  __half2& operator+=(const __half2& xx) {
    data += xx.data;
    return *this;
  }
  __HOST_DEVICE__
  __half2& operator-=(const __half2& xx) {
    data -= xx.data;
    return *this;
  }
  __HOST_DEVICE__
  __half2& operator*=(const __half2& xx) {
    data *= xx.data;
    return *this;
  }
  __HOST_DEVICE__
  __half2& operator/=(const __half2& xx) {
    data /= xx.data;
    return *this;
  }
  __HOST_DEVICE__
  __half2& operator++() { return *this += _Float16_2{1, 1}; }
  __HOST_DEVICE__
  __half2 operator++(int) {
    __half2 tmp{*this};
    ++*this;
    return tmp;
  }
  __HOST_DEVICE__
  __half2& operator--() { return *this -= _Float16_2{1, 1}; }
  __HOST_DEVICE__
  __half2 operator--(int) {
    __half2 tmp{*this};
    --*this;
    return tmp;
  }
#endif

  // ACCESSORS
  __HOST_DEVICE__
  operator decltype(data)() const { return data; }
  __HOST_DEVICE__
  operator __half2_raw() const {
    __half2_raw r;
    r.data = data;
    return r;
  }

// ACCESSORS - DEVICE ONLY
#if !defined(__HIP_NO_HALF_OPERATORS__)
  __HOST_DEVICE__
  __half2 operator+() const { return *this; }
  __HOST_DEVICE__
  __half2 operator-() const {
    __half2 tmp{*this};
    tmp.data = -tmp.data;
    return tmp;
  }
#endif

// FRIENDS
#if !defined(__HIP_NO_HALF_OPERATORS__)
  friend inline __HOST_DEVICE__ __half2 operator+(const __half2& xx, const __half2& yy) {
    return __half2{xx} += yy;
  }
  friend inline __HOST_DEVICE__ __half2 operator-(const __half2& xx, const __half2& yy) {
    return __half2{xx} -= yy;
  }
  friend inline __HOST_DEVICE__ __half2 operator*(const __half2& xx, const __half2& yy) {
    return __half2{xx} *= yy;
  }
  friend inline __HOST_DEVICE__ __half2 operator/(const __half2& xx, const __half2& yy) {
    return __half2{xx} /= yy;
  }
  friend inline __HOST_DEVICE__ bool operator==(const __half2& xx, const __half2& yy) {
    auto r = xx.data == yy.data;
    return r.x != 0 && r.y != 0;
  }
  friend inline __HOST_DEVICE__ bool operator!=(const __half2& xx, const __half2& yy) {
    return !(xx == yy);
  }
  friend inline __HOST_DEVICE__ bool operator<(const __half2& xx, const __half2& yy) {
    auto r = xx.data < yy.data;
    return r.x != 0 && r.y != 0;
  }
  friend inline __HOST_DEVICE__ bool operator>(const __half2& xx, const __half2& yy) {
    return yy < xx;
  }
  friend inline __HOST_DEVICE__ bool operator<=(const __half2& xx, const __half2& yy) {
    return !(yy < xx);
  }
  friend inline __HOST_DEVICE__ bool operator>=(const __half2& xx, const __half2& yy) {
    return !(xx < yy);
  }
#endif  // !defined(__HIP_NO_HALF_OPERATORS__)
};
// END STRUCT __HALF2

inline __HOST_DEVICE__ __half2 make_half2(__half x, __half y) { return __half2{x, y}; }

inline __HOST_DEVICE__ __half __low2half(__half2 x) {
  return __half{__half_raw{static_cast<__half2_raw>(x).data.x}};
}

inline __HOST_DEVICE__ __half __high2half(__half2 x) {
  return __half{__half_raw{static_cast<__half2_raw>(x).data.y}};
}

inline __HOST_DEVICE__ __half2 __half2half2(__half x) { return __half2{x, x}; }

inline __HOST_DEVICE__ __half2 __halves2half2(__half x, __half y) { return __half2{x, y}; }

inline __HOST_DEVICE__ __half2 __low2half2(__half2 x) {
  return __half2{
      _Float16_2{static_cast<__half2_raw>(x).data.x, static_cast<__half2_raw>(x).data.x}};
}

inline __HOST_DEVICE__ __half2 __high2half2(__half2 x) {
  return __half2{
      _Float16_2{static_cast<__half2_raw>(x).data.y, static_cast<__half2_raw>(x).data.y}};
}

inline __HOST_DEVICE__ __half2 __lows2half2(__half2 x, __half2 y) {
  return __half2{
      _Float16_2{static_cast<__half2_raw>(x).data.x, static_cast<__half2_raw>(y).data.x}};
}

inline __HOST_DEVICE__ __half2 __highs2half2(__half2 x, __half2 y) {
  return __half2{
      _Float16_2{static_cast<__half2_raw>(x).data.y, static_cast<__half2_raw>(y).data.y}};
}

inline __HOST_DEVICE__ __half2 __lowhigh2highlow(__half2 x) {
  return __half2{
      _Float16_2{static_cast<__half2_raw>(x).data.y, static_cast<__half2_raw>(x).data.x}};
}

// Bitcasts
inline __HOST_DEVICE__ short __half_as_short(__half x) { return static_cast<__half_raw>(x).x; }

inline __HOST_DEVICE__ unsigned short __half_as_ushort(__half x) {
  return static_cast<__half_raw>(x).x;
}

inline __HOST_DEVICE__ __half __short_as_half(short x) {
  __half_raw r;
  r.x = x;
  return r;
}

inline __HOST_DEVICE__ __half __ushort_as_half(unsigned short x) {
  __half_raw r;
  r.x = x;
  return r;
}

// float -> half | half2
inline __HOST_DEVICE__ __half __float2half(float x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __HOST_DEVICE__ __half __float2half_rn(float x) {
  return __half_raw{static_cast<_Float16>(x)};
}
#if !defined(__HIPCC_RTC__)
// TODO: rounding behaviour is not correct for host functions.
inline __host__ __half __float2half_rz(float x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __host__ __half __float2half_rd(float x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __host__ __half __float2half_ru(float x) { return __half_raw{static_cast<_Float16>(x)}; }
#endif
inline __device__ __half __float2half_rz(float x) { return __half_raw{__ocml_cvtrtz_f16_f32(x)}; }
inline __device__ __half __float2half_rd(float x) { return __half_raw{__ocml_cvtrtn_f16_f32(x)}; }
inline __device__ __half __float2half_ru(float x) { return __half_raw{__ocml_cvtrtp_f16_f32(x)}; }
inline __HOST_DEVICE__ __half2 __float2half2_rn(float x) {
  return __half2{_Float16_2{static_cast<_Float16>(x), static_cast<_Float16>(x)}};
}
inline __HOST_DEVICE__ __half2 __floats2half2_rn(float x, float y) {
  return __half2{_Float16_2{static_cast<_Float16>(x), static_cast<_Float16>(y)}};
}
inline __HOST_DEVICE__ __half2 __float22half2_rn(float2 x) { return __floats2half2_rn(x.x, x.y); }

// half | half2 -> float
inline __HOST_DEVICE__ float __half2float(__half x) { return static_cast<__half_raw>(x).data; }
inline __HOST_DEVICE__ float __low2float(__half2 x) { return static_cast<__half2_raw>(x).data.x; }
inline __HOST_DEVICE__ float __high2float(__half2 x) { return static_cast<__half2_raw>(x).data.y; }
inline __HOST_DEVICE__ float2 __half22float2(__half2 x) {
  return make_float2(static_cast<__half2_raw>(x).data.x, static_cast<__half2_raw>(x).data.y);
}

// half -> int
inline __device__ int __half2int_rn(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ int __half2int_rz(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ int __half2int_rd(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ int __half2int_ru(__half x) { return static_cast<__half_raw>(x).data; }

// int -> half
inline __HOST_DEVICE__ __half __int2half_rn(int x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __int2half_rz(int x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __int2half_rd(int x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __int2half_ru(int x) { return __half_raw{static_cast<_Float16>(x)}; }

// half -> short
inline __device__ short __half2short_rn(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ short __half2short_rz(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ short __half2short_rd(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ short __half2short_ru(__half x) { return static_cast<__half_raw>(x).data; }

// short -> half
inline __device__ __half __short2half_rn(short x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __short2half_rz(short x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __short2half_rd(short x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __short2half_ru(short x) { return __half_raw{static_cast<_Float16>(x)}; }

// half -> long long
inline __device__ long long __half2ll_rn(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ long long __half2ll_rz(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ long long __half2ll_rd(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ long long __half2ll_ru(__half x) { return static_cast<__half_raw>(x).data; }

// long long -> half
inline __device__ __half __ll2half_rn(long long x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __ll2half_rz(long long x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __ll2half_rd(long long x) { return __half_raw{static_cast<_Float16>(x)}; }
inline __device__ __half __ll2half_ru(long long x) { return __half_raw{static_cast<_Float16>(x)}; }

// half -> unsigned int
inline __device__ unsigned int __half2uint_rn(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ unsigned int __half2uint_rz(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ unsigned int __half2uint_rd(__half x) { return static_cast<__half_raw>(x).data; }
inline __device__ unsigned int __half2uint_ru(__half x) { return static_cast<__half_raw>(x).data; }

// unsigned int -> half
inline __device__ __half __uint2half_rn(unsigned int x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __uint2half_rz(unsigned int x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __uint2half_rd(unsigned int x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __uint2half_ru(unsigned int x) {
  return __half_raw{static_cast<_Float16>(x)};
}

// half -> unsigned short
inline __device__ unsigned short __half2ushort_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned short __half2ushort_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned short __half2ushort_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned short __half2ushort_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// unsigned short -> half
inline __device__ __half __ushort2half_rn(unsigned short x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __ushort2half_rz(unsigned short x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __ushort2half_rd(unsigned short x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __ushort2half_ru(unsigned short x) {
  return __half_raw{static_cast<_Float16>(x)};
}

// half -> unsigned long long
inline __device__ unsigned long long __half2ull_rn(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned long long __half2ull_rz(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned long long __half2ull_rd(__half x) {
  return static_cast<__half_raw>(x).data;
}
inline __device__ unsigned long long __half2ull_ru(__half x) {
  return static_cast<__half_raw>(x).data;
}

// unsigned long long -> half
inline __device__ __half __ull2half_rn(unsigned long long x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __ull2half_rz(unsigned long long x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __ull2half_rd(unsigned long long x) {
  return __half_raw{static_cast<_Float16>(x)};
}
inline __device__ __half __ull2half_ru(unsigned long long x) {
  return __half_raw{static_cast<_Float16>(x)};
}

// Load primitives
inline __device__ __half __ldg(const __half* ptr) { return *ptr; }
inline __device__ __half __ldcg(const __half* ptr) { return *ptr; }
inline __device__ __half __ldca(const __half* ptr) { return *ptr; }
inline __device__ __half __ldcs(const __half* ptr) { return *ptr; }

inline __HOST_DEVICE__ __half2 __ldg(const __half2* ptr) { return *ptr; }
inline __HOST_DEVICE__ __half2 __ldcg(const __half2* ptr) { return *ptr; }
inline __HOST_DEVICE__ __half2 __ldca(const __half2* ptr) { return *ptr; }
inline __HOST_DEVICE__ __half2 __ldcs(const __half2* ptr) { return *ptr; }

// Relations
inline __HOST_DEVICE__ bool __heq(__half x, __half y) {
  return static_cast<__half_raw>(x).data == static_cast<__half_raw>(y).data;
}
inline __HOST_DEVICE__ bool __hne(__half x, __half y) {
  return (static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data) ||
         (static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
}
inline __HOST_DEVICE__ bool __hle(__half x, __half y) {
  return static_cast<__half_raw>(x).data <= static_cast<__half_raw>(y).data;
}
inline __HOST_DEVICE__ bool __hge(__half x, __half y) {
  return static_cast<__half_raw>(x).data >= static_cast<__half_raw>(y).data;
}
inline __HOST_DEVICE__ bool __hlt(__half x, __half y) {
  return static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data;
}
inline __HOST_DEVICE__ bool __hgt(__half x, __half y) {
  return static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data;
}
inline __HOST_DEVICE__ bool __hequ(__half x, __half y) {
  return !(static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data) &&
         !(static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
}
inline __HOST_DEVICE__ bool __hneu(__half x, __half y) {
  return !(static_cast<__half_raw>(x).data == static_cast<__half_raw>(y).data);
}
inline __HOST_DEVICE__ bool __hleu(__half x, __half y) {
  return !(static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data);
}
inline __HOST_DEVICE__ bool __hgeu(__half x, __half y) {
  return !(static_cast<__half_raw>(x).data < static_cast<__half_raw>(y).data);
}
inline __HOST_DEVICE__ bool __hltu(__half x, __half y) {
  return !(static_cast<__half_raw>(x).data >= static_cast<__half_raw>(y).data);
}
inline __HOST_DEVICE__ bool __hgtu(__half x, __half y) {
  return !(static_cast<__half_raw>(x).data <= static_cast<__half_raw>(y).data);
}

inline __HOST_DEVICE__ __half2 __heq2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data == static_cast<__half2_raw>(y).data;
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hne2(__half2 x, __half2 y) {
  auto r = (static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data) |
           (static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hle2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data <= static_cast<__half2_raw>(y).data;
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hge2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data >= static_cast<__half2_raw>(y).data;
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hlt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data;
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hgt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data;
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hequ2(__half2 x, __half2 y) {
  auto r = !(static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data) &&
           !(static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hneu2(__half2 x, __half2 y) {
  auto r = !(static_cast<__half2_raw>(x).data == static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hleu2(__half2 x, __half2 y) {
  auto r = !(static_cast<__half2_raw>(x).data > static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hgeu2(__half2 x, __half2 y) {
  auto r = !(static_cast<__half2_raw>(x).data < static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hltu2(__half2 x, __half2 y) {
  auto r = !(static_cast<__half2_raw>(x).data >= static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}
inline __HOST_DEVICE__ __half2 __hgtu2(__half2 x, __half2 y) {
  auto r = !(static_cast<__half2_raw>(x).data <= static_cast<__half2_raw>(y).data);
  return __builtin_convertvector(-r, _Float16_2);
}

inline __HOST_DEVICE__ bool __hbeq2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__heq2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbne2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hne2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hble2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hle2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbge2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hge2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hblt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hlt2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbgt2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hgt2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbequ2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hequ2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbneu2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hneu2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbleu2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hleu2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbgeu2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hgeu2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbltu2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hltu2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hbgtu2(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hgtu2(x, y));
  return r.data.x != 0 && r.data.y != 0;
}
inline __HOST_DEVICE__ bool __hisnan(__half x) {
  __half_raw hr = x;
  return (hr.x & 0x7FFFU) > 0x7C00u;
}
inline __HOST_DEVICE__ __half __hmax(const __half x, const __half y) {
  if (__hisnan(x) && !__hisnan(y)) return y;
  if (!__hisnan(x) && __hisnan(y)) return x;
  if (__hisnan(x) && __hisnan(y)) return HIPRT_NAN_FP16;
  if (static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data)
    return __half_raw{static_cast<__half_raw>(x).data};
  return __half_raw{static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __hmax_nan(const __half x, const __half y) {
  if (__hisnan(x)) return x;
  if (__hisnan(y)) return y;
  return __hmax(x, y);
}
inline __HOST_DEVICE__ __half __hmin(const __half x, const __half y) {
  if (__hisnan(x) && !__hisnan(y)) return y;
  if (!__hisnan(x) && __hisnan(y)) return x;
  if (__hisnan(x) && __hisnan(y)) return HIPRT_NAN_FP16;
  if (static_cast<__half_raw>(x).data > static_cast<__half_raw>(y).data)
    return __half_raw{static_cast<__half_raw>(y).data};
  return __half_raw{static_cast<__half_raw>(x).data};
}
inline __HOST_DEVICE__ __half __hmin_nan(const __half x, const __half y) {
  if (__hisnan(x)) return x;
  if (__hisnan(y)) return y;
  return __hmin(x, y);
}

// Arithmetic
inline __device__ __half __clamp_01(__half x) {
  auto r = static_cast<__half_raw>(x);

  if (__hlt(x, __half_raw{0})) return __half_raw{0};
  if (__hlt(__half_raw{1}, x)) return __half_raw{1};
  return r;
}

inline __HOST_DEVICE__ __half __hadd(__half x, __half y) {
  return __half_raw{static_cast<__half_raw>(x).data + static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __hadd_rn(__half x, __half y) {
#pragma clang fp contract(off)
  return __half_raw{static_cast<__half_raw>(x).data + static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __habs(__half x) {
  static_assert(sizeof(_Float16) == sizeof(unsigned short));
  union {
    _Float16 fp16;
    unsigned short us;
  } u{static_cast<__half_raw>(x).data};
  u.us &= 0x7FFFu;
  return __half_raw{u.fp16};
}
inline __HOST_DEVICE__ __half __hsub(__half x, __half y) {
  return __half_raw{static_cast<__half_raw>(x).data - static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __hsub_rn(__half x, __half y) {
#pragma clang fp contract(off)
  return __half_raw{static_cast<__half_raw>(x).data - static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __hmul(__half x, __half y) {
  return __half_raw{static_cast<__half_raw>(x).data * static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __hmul_rn(__half x, __half y) {
#pragma clang fp contract(off)
  return __half_raw{static_cast<__half_raw>(x).data * static_cast<__half_raw>(y).data};
}
inline __HOST_DEVICE__ __half __hadd_sat(__half x, __half y) { return __clamp_01(__hadd(x, y)); }
inline __HOST_DEVICE__ __half __hsub_sat(__half x, __half y) { return __clamp_01(__hsub(x, y)); }
inline __HOST_DEVICE__ __half __hmul_sat(__half x, __half y) { return __clamp_01(__hmul(x, y)); }
inline __device__ __half __hfma(__half x, __half y, __half z) {
  return __half_raw{__builtin_elementwise_fma(static_cast<__half_raw>(x).data,
                                              static_cast<__half_raw>(y).data,
                                              static_cast<__half_raw>(z).data)};
}
inline __device__ __half __hfma_sat(__half x, __half y, __half z) {
  return __clamp_01(__hfma(x, y, z));
}
inline __HOST_DEVICE__ __half __hdiv(__half x, __half y) {
  return __half_raw{static_cast<__half_raw>(x).data / static_cast<__half_raw>(y).data};
}

inline __HOST_DEVICE__ __half2 __hadd2(__half2 x, __half2 y) {
  return __half2{static_cast<__half2_raw>(x).data + static_cast<__half2_raw>(y).data};
}
inline __HOST_DEVICE__ __half2 __hadd2_rn(__half2 x, __half2 y) {
#pragma clang fp contract(off)
  return __half2{static_cast<__half2_raw>(x).data + static_cast<__half2_raw>(y).data};
}

inline __HOST_DEVICE__ __half2 __habs2(__half2 x) { return __half2{__habs(x.x), __habs(x.y)}; }
inline __HOST_DEVICE__ __half2 __hsub2(__half2 x, __half2 y) {
  return __half2{static_cast<__half2_raw>(x).data - static_cast<__half2_raw>(y).data};
}
inline __HOST_DEVICE__ __half2 __hsub2_rn(__half2 x, __half2 y) {
#pragma clang fp contract(off)
  return __half2{static_cast<__half2_raw>(x).data - static_cast<__half2_raw>(y).data};
}
inline __HOST_DEVICE__ __half2 __hmul2(__half2 x, __half2 y) {
  return __half2{static_cast<__half2_raw>(x).data * static_cast<__half2_raw>(y).data};
}
inline __HOST_DEVICE__ __half2 __hmul2_rn(__half2 x, __half2 y) {
#pragma clang fp contract(off)
  return __half2{static_cast<__half2_raw>(x).data * static_cast<__half2_raw>(y).data};
}
inline __HOST_DEVICE__ __half2 __hadd2_sat(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hadd2(x, y));
  return __half2{__clamp_01(__half_raw{r.data.x}), __clamp_01(__half_raw{r.data.y})};
}
inline __HOST_DEVICE__ __half2 __hsub2_sat(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hsub2(x, y));
  return __half2{__clamp_01(__half_raw{r.data.x}), __clamp_01(__half_raw{r.data.y})};
}
inline __HOST_DEVICE__ __half2 __hmul2_sat(__half2 x, __half2 y) {
  auto r = static_cast<__half2_raw>(__hmul2(x, y));
  return __half2{__clamp_01(__half_raw{r.data.x}), __clamp_01(__half_raw{r.data.y})};
}
inline __device__ __half2 __hfma2(__half2 x, __half2 y, __half2 z) {
  return __half2{__builtin_elementwise_fma(static_cast<__half2_raw>(x).data,
                                           static_cast<__half2_raw>(y).data,
                                           static_cast<__half2_raw>(z).data)};
}
inline __device__ __half2 __hfma2_sat(__half2 x, __half2 y, __half2 z) {
  auto r = static_cast<__half2_raw>(__hfma2(x, y, z));
  return __half2{__clamp_01(__half_raw{r.data.x}), __clamp_01(__half_raw{r.data.y})};
}
inline __HOST_DEVICE__ __half2 __h2div(__half2 x, __half2 y) {
  return __half2{static_cast<__half2_raw>(x).data / static_cast<__half2_raw>(y).data};
}

// Device specific functions
#if defined(__clang__) && defined(__HIP__)
inline __device__ __half atomicAdd(__half* const address, const __half value) {
  return __half_raw{__scoped_atomic_fetch_add((_Float16*)address,
                                              static_cast<__half_raw>(value).data,
                                              __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE)};
}

inline __device__ __half2 atomicAdd(__half2* const address, const __half2 value) {
  static_assert(sizeof(_Float16_2) == sizeof(unsigned int));
  union {
    _Float16_2 vec;
    unsigned int u32;
  } expected, desired;

  unsigned int* atomic_ptr = (unsigned int*)address;
  expected.u32 = __scoped_atomic_load_n(atomic_ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);

  do {
    desired.vec = expected.vec + static_cast<_Float16_2>(value);
  } while (!__scoped_atomic_compare_exchange_n(atomic_ptr, &expected.u32, desired.u32, 0,
                                               __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                               __MEMORY_SCOPE_DEVICE));
  return static_cast<__half2>(expected.vec);
}

inline __device__ __half2 unsafeAtomicAdd(__half2* address, __half2 value) {
#if __has_builtin(__builtin_amdgcn_flat_atomic_fadd_v2f16)
  // The api expects an ext_vector_type of half
  typedef _Float16 __attribute__((ext_vector_type(2))) vec_fp162;
  static_assert(sizeof(vec_fp162) == sizeof(__half2_raw));
  union {
    __half2_raw h2r;
    vec_fp162 fp16;
  } u{static_cast<__half2_raw>(value)};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_v2f16))
    u.fp16 = __builtin_amdgcn_flat_atomic_fadd_v2f16((vec_fp162*)address, u.fp16);
  return static_cast<__half2>(u.h2r);
#else
  static_assert(sizeof(__half2_raw) == sizeof(unsigned int));
  union u_hold {
    __half2_raw h2r;
    unsigned int u32;
  };
  u_hold old_val, new_val;
  old_val.u32 =
      __hip_atomic_load((unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  do {
    new_val.h2r = __hadd2(old_val.h2r, value);
  } while (!__hip_atomic_compare_exchange_strong((unsigned int*)address, &old_val.u32, new_val.u32,
                                                 __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT));
  return old_val.h2r;
#endif
}

inline __device__ __half unsafeAtomicAdd(__half* address, __half value) {
  static_assert(sizeof(unsigned short int) == sizeof(__half_raw));
  unsigned short int* address_as_short = reinterpret_cast<unsigned short int*>(address);
  // Align to 4 bytes
  unsigned int* aligned_addr = __builtin_bit_cast(
      unsigned int*, __builtin_bit_cast(unsigned long long int, address_as_short) &
                         (unsigned long long int)(~0x3));

  bool is_lower = __builtin_bit_cast(unsigned long long int, aligned_addr) ==
                  __builtin_bit_cast(unsigned long long int, address);
  __half2 fval;
  if (is_lower)
    fval = __halves2half2(value, __float2half(0.0f));
  else
    fval = __halves2half2(__float2half(0.0f), value);

  __half2* in = (__half2*)(aligned_addr);
  __half2 out = unsafeAtomicAdd(in, fval);
  if (is_lower) return __low2half(out);
  return __high2half(out);
}

namespace __hip_internal {
template <>
struct NumericLimits<__half> {
    static constexpr __half maximum() {
      __half_raw raw { .x = 0x7C00U };
      return __half(raw);
    }
    static constexpr __half minimum() {
      __half_raw raw { .x = 0xFC00U };
      return __half(raw);
    }
};
}  // namespace __hip_internal
#endif  // defined(__clang__) && defined(__HIP__)

// Math functions
#if defined(__clang__) && defined(__HIP__)
inline __device__ float amd_mixed_dot(__half2 a, __half2 b, float c, bool saturate) {
  return __ockl_fdot2(static_cast<__half2_raw>(a).data, static_cast<__half2_raw>(b).data, c,
                      saturate);
}
#endif
inline __device__ __half htrunc(__half x) {
  return __half_raw{__builtin_elementwise_trunc(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hceil(__half x) {
  return __half_raw{__builtin_elementwise_ceil(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hfloor(__half x) {
  return __half_raw{__builtin_elementwise_floor(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hrint(__half x) {
  return __half_raw{__builtin_elementwise_rint(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hsin(__half x) {
  return __half_raw{__ocml_sin_f16(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hcos(__half x) {
  return __half_raw{__ocml_cos_f16(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hexp(__half x) {
  return __half_raw{__builtin_elementwise_exp(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hexp2(__half x) {
  return __half_raw{__builtin_elementwise_exp2(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hexp10(__half x) {
  return __half_raw{__builtin_elementwise_exp10(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hlog2(__half x) {
  return __half_raw{__builtin_elementwise_log2(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hlog(__half x) {
  return __half_raw{__builtin_elementwise_log(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hlog10(__half x) {
  return __half_raw{__builtin_elementwise_log10(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hrcp(__half x) {
  return __half_raw{static_cast<_Float16>(1.0f) / static_cast<__half_raw>(x).data};
}
inline __device__ __half hrsqrt(__half x) {
  return __half_raw{__ocml_rsqrt_f16(static_cast<__half_raw>(x).data)};
}
inline __device__ __half hsqrt(__half x) {
  return __half_raw{__builtin_elementwise_sqrt(static_cast<__half_raw>(x).data)};
}
inline __HOST_DEVICE__ bool __hisinf(__half x) {
  __half_raw hr = x;
  // +/-Inf
  return hr.x == 0x7C00U || hr.x == 0xFC00U;
}
inline __HOST_DEVICE__ __half __hneg(__half x) {
  return __half_raw{-static_cast<__half_raw>(x).data};
}

inline __device__ __half2 h2trunc(__half2 x) {
  return __half2{__builtin_elementwise_trunc(static_cast<__half2_raw>(x).data)};
}
inline __device__ __half2 h2ceil(__half2 x) {
  return __half2{__builtin_elementwise_ceil(static_cast<__half2_raw>(x).data)};
}
inline __device__ __half2 h2floor(__half2 x) {
  return __half2{__builtin_elementwise_floor(static_cast<__half2_raw>(x).data)};
}
inline __device__ __half2 h2rint(__half2 x) {
  return __half2{__builtin_elementwise_rint(static_cast<__half2_raw>(x).data)};
}
inline __device__ __half2 h2sin(__half2 x) { return __half2{__ocml_sin_2f16(x)}; }
inline __device__ __half2 h2cos(__half2 x) { return __half2{__ocml_cos_2f16(x)}; }
inline __device__ __half2 h2exp(__half2 x) {
  return __half2{__builtin_elementwise_exp(static_cast<__half2_raw>(x).data)};
}
inline __device__ __half2 h2exp2(__half2 x) {
  return __half2{__builtin_elementwise_exp2(static_cast<__half2_raw>(x).data)};
}
inline __device__ __half2 h2exp10(__half2 x) {
  return __half2{__builtin_elementwise_exp10(static_cast<__half2_raw>(x).data)};
}

inline __device__ __half2 h2log2(__half2 x) {
  return __half2{__builtin_elementwise_log2(static_cast<__half2_raw>(x).data)};
}

inline __device__ __half2 h2log(__half2 x) {
  return __half2{__builtin_elementwise_log(static_cast<__half2_raw>(x).data)};
}

inline __device__ __half2 h2log10(__half2 x) {
  return __half2{__builtin_elementwise_log10(static_cast<__half2_raw>(x).data)};
}

inline __device__ __half2 h2rcp(__half2 x) {
  return _Float16_2{_Float16_2{static_cast<_Float16>(1.0f), static_cast<_Float16>(1.0f)} / x.data};
}
inline __device__ __half2 h2rsqrt(__half2 x) { return __ocml_rsqrt_2f16(x); }
inline __device__ __half2 h2sqrt(__half2 x) {
  return __half2{__builtin_elementwise_sqrt(static_cast<__half2_raw>(x).data)};
}

inline __device__ __half2 __hisinf2(__half2 x) {
  return __half2{
      _Float16_2{static_cast<_Float16>(__builtin_isinf(static_cast<__half2_raw>(x).data.x)),
                 static_cast<_Float16>(__builtin_isinf(static_cast<__half2_raw>(x).data.y))}};
}

inline __HOST_DEVICE__ __half2 __hisnan2(__half2 x) {
  return __half2{_Float16_2{static_cast<_Float16>(__hisnan(x.x) ? 1.0f : 0.0f),
                            static_cast<_Float16>(__hisnan(x.y) ? 1.0f : 0.0f)}};
}
inline __HOST_DEVICE__ __half2 __hneg2(__half2 x) {
  return __half2{-static_cast<__half2_raw>(x).data};
}

#if !defined(HIP_NO_HALF)
using half = __half;
using half2 = __half2;
#endif
__device__ inline __half __shfl(__half var, int src_lane, int width = warpSize) {
  union {
    int i;
    __half h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.h;
}
__device__ inline __half2 __shfl(__half2 var, int src_lane, int width = warpSize) {
  union {
    int i;
    __half2 h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.h;
}
__device__ inline __half __shfl_up(__half var, unsigned int lane_delta, int width = warpSize) {
  union {
    int i;
    __half h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.h;
}
__device__ inline __half2 __shfl_up(__half2 var, unsigned int lane_delta, int width = warpSize) {
  union {
    int i;
    __half2 h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.h;
}
__device__ inline __half __shfl_down(__half var, unsigned int lane_delta, int width = warpSize) {
  union {
    int i;
    __half h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.h;
}
__device__ inline __half2 __shfl_down(__half2 var, unsigned int lane_delta, int width = warpSize) {
  union {
    int i;
    __half2 h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.h;
}
__device__ inline __half __shfl_xor(__half var, int lane_mask, int width = warpSize) {
  union {
    int i;
    __half h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.h;
}
__device__ inline __half2 __shfl_xor(__half2 var, int lane_mask, int width = warpSize) {
  union {
    int i;
    __half2 h;
  } tmp;
  tmp.h = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.h;
}

#if defined(HIP_ENABLE_EXTRA_WARP_SYNC_TYPES) && !defined(__HIP_NO_HALF_OPERATORS__)
extern "C" __device__ __attribute__((const)) __half __ockl_wfred_add_f16(__half);
extern "C" __device__ __attribute__((const)) __half __ockl_wfred_min_f16(__half);
extern "C" __device__ __attribute__((const)) __half __ockl_wfred_max_f16(__half);

template <typename MaskT> __device__ inline __half __reduce_add_sync(MaskT mask, __half val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_f16(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline __half __reduce_min_sync(MaskT mask, __half val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_f16(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline __half __reduce_max_sync(MaskT mask, __half val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_f16(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}
#endif

#if !defined(__HIP_NO_HALF_OPERATORS__)
namespace cooperative_groups {
namespace impl {
HIP_IMPL_GENERATE_SCAN_FUNC(add, f16, __half);
HIP_IMPL_GENERATE_SCAN_FUNC(min, f16, __half);
HIP_IMPL_GENERATE_SCAN_FUNC(max, f16, __half);
}
}
#endif

#endif  // defined(__cplusplus)
#elif defined(__GNUC__) || defined(_MSC_VER)
#if !defined(__HIPCC_RTC__)
#include "hip_fp16_gcc.h"
#endif
#endif  // !defined(__clang__) && defined(__GNUC__)

#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP16_H
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "hip_fp16_math_fwd.h"
#include "amd_hip_vector_types.h"
#include "math_fwd.h"

#include <hip/amd_detail/host_defines.h>

#include <algorithm>
// assert.h is only for the host version of assert.
// The device version of assert is implemented in hip/amd_detail/hip_runtime.h.
// Users should include hip_runtime.h for the device version of assert.
#if !__HIP_DEVICE_COMPILE__
#include <assert.h>
#endif
#include <limits.h>
#include <limits>
#include <stdint.h>
#endif  // !defined(__HIPCC_RTC__)

#pragma push_macro("__DEVICE__")
#pragma push_macro("__RETURN_TYPE")

#define __DEVICE__ static __device__
#define __RETURN_TYPE bool

// DOT FUNCTIONS
#if defined(__clang__) && defined(__HIP__)
__DEVICE__
inline int amd_mixed_dot(short2 a, short2 b, int c, bool saturate) {
  return __ockl_sdot2(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline uint amd_mixed_dot(ushort2 a, ushort2 b, uint c, bool saturate) {
  return __ockl_udot2(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline int amd_mixed_dot(char4 a, char4 b, int c, bool saturate) {
  return __ockl_sdot4(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate) {
  return __ockl_udot4(get_native_vector(a), get_native_vector(b), c, saturate);
}
__DEVICE__
inline int amd_mixed_dot(int a, int b, int c, bool saturate) {
  return __ockl_sdot8(a, b, c, saturate);
}
__DEVICE__
inline uint amd_mixed_dot(uint a, uint b, uint c, bool saturate) {
  return __ockl_udot8(a, b, c, saturate);
}
#endif

#pragma pop_macro("__DEVICE__")
#pragma pop_macro("__RETURN_TYPE")
// For backward compatibility.
// There are HIP applications e.g. TensorFlow, expecting __HIP_ARCH_* macros
// defined after including math_functions.h.
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h>
#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file
 * \brief hip_bf16.h provides struct for __hip_bfloat16 types
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16 bfloat16 Precision Intrinsics
 * This section describes hip_bfloat16 precision intrinsic functions.
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_ARITH Bfloat16 Arithmetic Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_COMP Bfloat16 Comparision Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_COMP Bfloat162 Comparision Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_ARITH Bfloat162 Arithmetic Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_CONV Bfloat16 Conversion Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_CONV Bfloat162 Conversion Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_MOVE Bfloat16 Data Movement Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_MOVE Bfloat162 Data Movement Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_MATH Bfloat16 Math Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_MATH Bfloat162 Math Functions
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_RAW Bfloat16 Raw Struct
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_RAW Bfloat162 Raw Struct
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * To use these functions, include the header file \p hip_bf16.h in your program.
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BF16_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_BF16_H_

#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_common.h>
#include "amd_hip_vector_types.h"  // float2 etc
#if defined(__clang__) && defined(__HIP__)
#include "math_fwd.h"              // ocml device functions
#include <hip/amd_detail/amd_warp_functions.h>       // define warpSize
#include <hip/amd_detail/amd_warp_sync_functions.h>  // Sync functions
#endif
#endif  // !defined(__HIPCC_RTC__)

#define __BF16_DEVICE__ __device__
#if defined(__HIPCC_RTC__)
#define __BF16_HOST_DEVICE__ __BF16_DEVICE__
#else
#include <climits>
#include <cmath>
#define __BF16_HOST_DEVICE__ __host__ __BF16_DEVICE__
#endif
#define __BF16_DEVICE_STATIC__ __BF16_DEVICE__ static inline
#define __BF16_HOST_DEVICE_STATIC__ __BF16_HOST_DEVICE__ static inline

#pragma push_macro("MAYBE_UNDEF")
#if defined(__has_attribute) && __has_attribute(maybe_undef)
#define MAYBE_UNDEF __attribute__((maybe_undef))
#else
#define MAYBE_UNDEF
#endif

#define HIPRT_ONE_BF16 __ushort_as_bfloat16((unsigned short)0x3F80U)
#define HIPRT_ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x0000U)
#define HIPRT_INF_BF16 __ushort_as_bfloat16((unsigned short)0x7F80U)
#define HIPRT_MAX_NORMAL_BF16 __ushort_as_bfloat16((unsigned short)0x7F7FU)
#define HIPRT_MIN_DENORM_BF16 __ushort_as_bfloat16((unsigned short)0x0001U)
#define HIPRT_NAN_BF16 __ushort_as_bfloat16((unsigned short)0x7FFFU)
#define HIPRT_NEG_ZERO_BF16 __ushort_as_bfloat16((unsigned short)0x8000U)

#if defined(__clang__)
// Since we are using unsigned short to represent data in bfloat16, it can be of different sizes on
// different machines. These naive checks should prevent some undefined behavior on systems which
// have different sizes for basic types.
#if !defined(__HIPCC_RTC__)
static_assert(CHAR_BIT == 8, "byte size should be of 8 bits");
#endif
static_assert(sizeof(unsigned short) == 2, "size of unsigned short should be 2 bytes");
static_assert(sizeof(__bf16) == sizeof(unsigned short));

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_RAW
 * \brief represents raw bfloat16 type
 */
typedef struct __attribute__((aligned(2))) {
  unsigned short x;
} __hip_bfloat16_raw;

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_RAW
 * \brief represents raw bfloat16x2 vector type
 */
typedef struct __attribute__((aligned(4))) {
  unsigned short x;
  unsigned short y;
} __hip_bfloat162_raw;

/**
 * \defgroup HIP_INTRINSIC_BFLOAT16_STRUCT
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * \brief Struct to represent a 16 bit brain floating point number.
 * @{
 */
struct __attribute__((aligned(2))) __hip_bfloat16 {
 protected:
  union {
    /*! \brief raw representation of bfloat16 */
    unsigned short __x;
    /*! \brief bf16 represenation */
    __bf16 __x_bf16;
  };

 public:
  // TODO: SWDEV-452411
  // Need to add constructor of __hip_bfloat16 from
  // unsigned long long
  // long long
  // long
  // unsigned long
  // Casting directly to double might lead to double rounding.

  /*! \brief create __hip_bfloat16 from an unsigned int */
  __BF16_HOST_DEVICE__ __hip_bfloat16(unsigned int val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a int */
  __BF16_HOST_DEVICE__ __hip_bfloat16(int val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from an unsigned short */
  __BF16_HOST_DEVICE__ __hip_bfloat16(unsigned short val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a short */
  __BF16_HOST_DEVICE__ __hip_bfloat16(short val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a double */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const double val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a float */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const float val) : __x_bf16(static_cast<__bf16>(val)) {}

  /*! \brief create __hip_bfloat16 from a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ constexpr __hip_bfloat16(const __hip_bfloat16_raw& val) : __x(val.x) {}

  /*! \brief create __hip_bfloat16 from __bf16 */
  __BF16_HOST_DEVICE__ __hip_bfloat16(const __bf16 val) : __x_bf16(val) {}

  /*! \brief default constructor */
  __BF16_HOST_DEVICE__ __hip_bfloat16() = default;

  /*! \brief return a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ operator __hip_bfloat16_raw() const { return __hip_bfloat16_raw{__x}; }

  /*! \brief return a __hip_bfloat16_raw cv qualifier */
  __BF16_HOST_DEVICE__ operator __hip_bfloat16_raw() const volatile {
    return __hip_bfloat16_raw{__x};
  }

  /*! \brief return false if bfloat value is +0.0 or -0.0, returns true otherwise */
  __BF16_HOST_DEVICE__ constexpr operator bool() const { return __x_bf16 != 0.0f; }

  /*! \brief return a casted char from underlying float val */
  __BF16_HOST_DEVICE__ operator char() const { return static_cast<char>(__x_bf16); }

  /*! \brief return a float */
  __BF16_HOST_DEVICE__ operator float() const { return static_cast<float>(__x_bf16); }

  /*! \brief return a casted int casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator int() const { return static_cast<int>(__x_bf16); }

  /*! \brief return a casted long casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator long() const { return static_cast<long>(__x_bf16); }

  /*! \brief return a casted long long casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator long long() const { return static_cast<long long>(__x_bf16); }

  /*! \brief return a casted short casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator short() const { return static_cast<short>(__x_bf16); }

  /*! \brief return a casted signed char from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator signed char() const { return static_cast<signed char>(__x_bf16); }

  /*! \brief return a casted unsigned char casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned char() const {
    return static_cast<unsigned char>(__x_bf16);
  }

  /*! \brief return a casted unsigned int casted from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned int() const { return static_cast<unsigned int>(__x_bf16); }

  /*! \brief return a casted unsigned from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned long() const {
    return static_cast<unsigned long>(__x_bf16);
  }

  /*! \brief return a casted unsigned long long from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned long long() const {
    return static_cast<unsigned long long>(__x_bf16);
  }

  /*! \brief return a casted unsigned short from float of underlying bfloat16 value */
  __BF16_HOST_DEVICE__ operator unsigned short() const {
    return static_cast<unsigned short>(__x_bf16);
  }

  __BF16_HOST_DEVICE__ operator __bf16() const { return __x_bf16; }

  // TODO: SWDEV-452411 add operator which converts unsigned long long and long long to bfloat

  /*! \brief assign value from an unsigned int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(unsigned int val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from a int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(int val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from an unsigned short */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(unsigned short val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from a short int */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(short val) {
    __x_bf16 = static_cast<__bf16>(val);
    return *this;
  }

  /*! \brief assign value from a double */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const double f) {
    __x_bf16 = static_cast<__bf16>(f);
    return *this;
  }

  /*! \brief assign value from a float */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const float f) {
    __x_bf16 = static_cast<__bf16>(f);
    return *this;
  }

  /*! \brief assign value from a __hip_bfloat16_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat16& operator=(const __hip_bfloat16_raw& hr) {
    __x = hr.x;
    return *this;
  }

  /*! \brief assign value from a __hip_bfloat16_raw volatile */
  __BF16_HOST_DEVICE__ volatile __hip_bfloat16& operator=(const __hip_bfloat16_raw& hr) volatile {
    __x = hr.x;
    return *this;
  }

  /*! \brief assign value from a __hip_bfloat16_raw cv qualifier */
  __BF16_HOST_DEVICE__ volatile __hip_bfloat16& operator=(
      const volatile __hip_bfloat16_raw& hr) volatile {
    __x = hr.x;
    return *this;
  }
};
/**@}*/

#if defined(__clang__)
typedef __bf16 __bf16_2 __attribute__((ext_vector_type(2)));
#else
typedef __bf16 __bf16_2 __attribute__((vector_size(sizeof(__bf16) * 2)));
#endif

/**
 * \defgroup HIP_INTRINSIC_BFLOAT162_STRUCT
 * \ingroup HIP_INTRINSIC_BFLOAT16
 * \brief Struct to represent a two 16 bit brain floating point number.
 * @{
 */
struct __attribute__((aligned(4))) __hip_bfloat162 {
  static_assert(sizeof(__hip_bfloat16[2]) == sizeof(__bf16_2));

 public:
#if defined(__clang__)
  union {
    struct {
      __hip_bfloat16 x; /*! \brief raw representation of bfloat16 */
      __hip_bfloat16 y; /*! \brief raw representation of bfloat16 */
    };
    __bf16_2 __xy_bf162;
  };
#else
  /* GCC does not support anonymous structs with members that have non-trivial constructors (Clang
  allows this as an extension). Expose x and y directly instead. */
  __hip_bfloat16 x;
  __hip_bfloat16 y;
#endif

 public:
  /*! \brief create __hip_bfloat162 from __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat162_raw& h2r)
      : x(__hip_bfloat16(__hip_bfloat16_raw{h2r.x})),
        y(__hip_bfloat16(__hip_bfloat16_raw{h2r.y})) {}

  /*! \brief copy constructor of __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __hip_bfloat162& val) : x(val.x), y(val.y) {}

  /*! \brief create __hip_bfloat162 from two __hip_bfloat16 */
  __BF16_HOST_DEVICE__ constexpr __hip_bfloat162(const __hip_bfloat16& a, const __hip_bfloat16& b)
      : x(a), y(b) {}

  /*! \brief create __hip_bfloat162 from vector of __bf16_2 */
#if defined(__clang__)
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __bf16_2 in) : __xy_bf162(in) {}
#else
  __BF16_HOST_DEVICE__ __hip_bfloat162(const __bf16_2 in) : x{in[0]}, y{in[1]} {}
#endif

  /*! \brief default constructor of __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162() = default;

  /*! \brief return a __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    __hip_bfloat16_raw l = x;
    __hip_bfloat16_raw r = y;
    return __hip_bfloat162_raw{l.x, r.x};
  }

  /*! \brief return a float2 */
  __BF16_HOST_DEVICE__ operator float2() const {
    float2 ret(x, y);
    return ret;
  }

  /*! \brief return a vector of bf16 */
  __BF16_HOST_DEVICE__ operator __bf16_2() const {
#if defined(__clang__)
    return __xy_bf162;
#else
    return __bf16_2{x, y};
#endif
  }

  /*! \brief return a vector of bf16 */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __bf16_2 in) {
#if defined(__clang__)
    __xy_bf162 = in;
#else
    x = __hip_bfloat16{in[0]};
    y = __hip_bfloat16{in[1]};
#endif
    return *this;
  }

  /*! \brief assign value from __hip_bfloat162_raw */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __hip_bfloat162_raw& h2r) {
    x = __hip_bfloat16(__hip_bfloat16_raw{h2r.x});
    y = __hip_bfloat16(__hip_bfloat16_raw{h2r.y});
    return *this;
  }

  /*! \brief assign value from __hip_bfloat162 */
  __BF16_HOST_DEVICE__ __hip_bfloat162& operator=(const __hip_bfloat162& src) {
    x = src.x;
    y = src.y;
    return *this;
  }
};
/**@}*/

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is nan
 */
__BF16_HOST_DEVICE_STATIC__ bool __hisnan(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  return !(~hr.x & 0x7f80) && +(hr.x & 0x7f);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Converts bfloat16 to float
 */
__BF16_HOST_DEVICE_STATIC__ float __bfloat162float(__hip_bfloat16 a) {
  float ret = a;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Converts float to bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __float2bfloat16(float f) {
  __hip_bfloat16 ret{f};
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts and moves bfloat162 to float2
 */
__BF16_HOST_DEVICE_STATIC__ float2 __bfloat1622float2(const __hip_bfloat162 a) {
  float2 ret = a;
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Moves bfloat16 value to bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __bfloat162bfloat162(const __hip_bfloat16 a) {
  return __hip_bfloat162(a, a);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as a signed short integer
 */
__BF16_HOST_DEVICE_STATIC__ short int __bfloat16_as_short(const __hip_bfloat16 h) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(short int));
  union {
    __hip_bfloat16 bf16;
    short int si;
  } u{h};
  return u.si;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Reinterprets bits in a __hip_bfloat16 as an unsigned signed short integer
 */
__BF16_HOST_DEVICE_STATIC__ unsigned short int __bfloat16_as_ushort(const __hip_bfloat16 h) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(unsigned short int));
  union {
    __hip_bfloat16 bf16;
    unsigned short int usi;
  } u{h};
  return u.usi;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Convert double to __hip_bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __double2bfloat16(const double a) {
  __hip_bfloat16 ret{a};
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_CONV
 * \brief Convert int to __hip_bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __int2bfloat16_rn(const int a) {
  __hip_bfloat16 ret{a};
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Convert float2 to __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __float22bfloat162_rn(const float2 a) {
  return __hip_bfloat162{__float2bfloat16(a.x), __float2bfloat16(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Combine two __hip_bfloat16 to __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __halves2bfloat162(const __hip_bfloat16 a,
                                                               const __hip_bfloat16 b) {
  return __hip_bfloat162(a, b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __high2bfloat16(const __hip_bfloat162 a) { return a.y; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns high 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __high2bfloat162(const __hip_bfloat162 a) {
  return __hip_bfloat162(a.y, a.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts high 16 bits of __hip_bfloat162 to float and returns the result
 */
__BF16_HOST_DEVICE_STATIC__ float __high2float(const __hip_bfloat162 a) {
  return __bfloat162float(a.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts high 16 bits from each and combines them
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __highs2bfloat162(const __hip_bfloat162 a,
                                                              const __hip_bfloat162 b) {
  return __hip_bfloat162(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __low2bfloat16(const __hip_bfloat162 a) { return a.x; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Returns low 16 bits of __hip_bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __low2bfloat162(const __hip_bfloat162 a) {
  return __hip_bfloat162(a.x, a.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Converts low 16 bits of __hip_bfloat162 to float and returns the result
 */
__BF16_HOST_DEVICE_STATIC__ float __low2float(const __hip_bfloat162 a) {
  return __bfloat162float(a.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Swaps both halves
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lowhigh2highlow(const __hip_bfloat162 a) {
  return __hip_bfloat162(a.y, a.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Extracts low 16 bits from each and combines them
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __lows2bfloat162(const __hip_bfloat162 a,
                                                             const __hip_bfloat162 b) {
  return __hip_bfloat162(a.x, b.x);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets short int into a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __short_as_bfloat16(const short int a) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(short int));
  union {
    short int si;
    __hip_bfloat16 bf16;
  } u{a};
  return u.bf16;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_CONV
 * \brief Reinterprets unsigned short int into a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __ushort_as_bfloat16(const unsigned short int a) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(unsigned short int));
  union {
    unsigned short int usi;
    __hip_bfloat16 bf16;
  } u{a};
  return u.bf16;
}

#if defined(__clang__) && defined(__HIP__)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl(MAYBE_UNDEF __hip_bfloat16 var, int src_lane, int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl(tmp.i, src_lane, width);
  return tmp.f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl up warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl_up(MAYBE_UNDEF __hip_bfloat16 var, unsigned int lane_delta,
                         int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_up(tmp.i, lane_delta, width);
  return tmp.f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl down warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl_down(MAYBE_UNDEF __hip_bfloat16 var, unsigned int lane_delta,
                           int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_down(tmp.i, lane_delta, width);
  return tmp.f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_SHFL
 * \brief shfl xor warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__
__hip_bfloat16 __shfl_xor(MAYBE_UNDEF __hip_bfloat16 var, int lane_mask, int width = warpSize) {
  union {
    int i;
    __hip_bfloat16 f;
  } tmp;
  tmp.f = var;
  tmp.i = __shfl_xor(tmp.i, lane_mask, width);
  return tmp.f;
}

#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl down warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_down_sync(const unsigned long long mask,
                                                       const __hip_bfloat16 in,
                                                       const unsigned int delta,
                                                       const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_down_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl down warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_down_sync(const unsigned long long mask,
                                                        const __hip_bfloat162 in,
                                                        const unsigned int delta,
                                                        const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_down_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl sync warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_sync(const unsigned long long mask,
                                                  const __hip_bfloat16 in, const int delta,
                                                  const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl sync warp intrinsic for bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_sync(const unsigned long long mask,
                                                   const __hip_bfloat162 in, const int delta,
                                                   const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl up sync warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_up_sync(const unsigned long long mask,
                                                     const __hip_bfloat16 in,
                                                     const unsigned int delta,
                                                     const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_up_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl up sync warp intrinsic for bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_up_sync(const unsigned long long mask,
                                                      const __hip_bfloat162 in,
                                                      const unsigned int delta,
                                                      const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_up_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MOVE
 * \brief shfl xor sync warp intrinsic for bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __shfl_xor_sync(const unsigned long long mask,
                                                      const __hip_bfloat16 in, const int delta,
                                                      const int width = warpSize) {
  return __ushort_as_bfloat16(__shfl_xor_sync(mask, __bfloat16_as_ushort(in), delta, width));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MOVE
 * \brief shfl xor sync warp intrinsic for bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __shfl_xor_sync(const unsigned long long mask,
                                                       const __hip_bfloat162 in, const int delta,
                                                       const int width = warpSize) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(unsigned int));
  union {
    __hip_bfloat162 bf162;
    unsigned int ui;
  } u{in};
  u.ui = __shfl_xor_sync<unsigned long long, unsigned int>(mask, u.ui, delta, width);
  return u.bf162;
}
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
#endif  // defined(__clang__) && defined(__HIP__)

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Adds two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hadd(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a + (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Adds two bfloat16 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hadd_rn(const __hip_bfloat16 a,
                                                     const __hip_bfloat16 b) {
#pragma clang fp contract(off)
  return (__bf16)a + (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Subtracts two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hsub(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a - (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Subtracts two bfloat16 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hsub_rn(const __hip_bfloat16 a,
                                                     const __hip_bfloat16 b) {
#pragma clang fp contract(off)
  return (__bf16)a - (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Divides two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hdiv(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a / (__bf16)b;
}

#if defined(__clang__) && defined(__HIP__)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Performs FMA of given bfloat16 values
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 __hfma(const __hip_bfloat16 a, const __hip_bfloat16 b,
                                             const __hip_bfloat16 c) {
  return __hip_bfloat16(__builtin_elementwise_fma(__bf16(a), __bf16(b), __bf16(c)));
  ;
}
#endif

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Multiplies two bfloat16 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmul(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a * (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Multiplies two bfloat16 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmul_rn(const __hip_bfloat16 a,
                                                     const __hip_bfloat16 b) {
#pragma clang fp contract(off)
  return (__bf16)a * (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Negate a bfloat16 value
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hneg(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  hr.x ^= 0x8000;
  return __hip_bfloat16(hr);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Returns absolute of a bfloat16
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __habs(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  hr.x &= 0x7FFF;
  return __hip_bfloat16(hr);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Divides bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __h2div(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) / __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Returns absolute of a bfloat162
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __habs2(const __hip_bfloat162 a) {
  return __hip_bfloat162{__habs(a.x), __habs(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Adds two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hadd2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) + __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Adds two bfloat162 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hadd2_rn(const __hip_bfloat162 a,
                                                       const __hip_bfloat162 b) {
#pragma clang fp contract(off)
  return __hip_bfloat162{__bf16_2(a) + __bf16_2(b)};
}


#if defined(__clang__) && defined(__HIP__)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Performs FMA of given bfloat162 values
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 __hfma2(const __hip_bfloat162 a, const __hip_bfloat162 b,
                                               const __hip_bfloat162 c) {
  return __hip_bfloat162{__builtin_elementwise_fma(__bf16_2(a), __bf16_2(b), __bf16_2(c))};
}
#endif

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Multiplies two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmul2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) * __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Multiplies two bfloat162 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmul2_rn(const __hip_bfloat162 a,
                                                       const __hip_bfloat162 b) {
#pragma clang fp contract(off)
  return __hip_bfloat162{__bf16_2(a) * __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Converts a bfloat162 into negative
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hneg2(const __hip_bfloat162 a) {
  return __hip_bfloat162{__hneg(a.x), __hneg(a.y)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Subtracts two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hsub2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162{__bf16_2(a) - __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Subtracts two bfloat162 values, will not fuse into fma
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hsub2_rn(const __hip_bfloat162 a,
                                                       const __hip_bfloat162 b) {
#pragma clang fp contract(off)
  return __hip_bfloat162{__bf16_2(a) - __bf16_2(b)};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to multiply two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator*(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hmul(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to multiply-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator*=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hmul(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to unary+ on a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator+(const __hip_bfloat16& l) { return l; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to add two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator+(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hadd(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to negate a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator-(const __hip_bfloat16& l) { return __hneg(l); }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to subtract two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator-(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hsub(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to post increment a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator++(__hip_bfloat16& l, const int) {
  auto ret = l;
  l = __hadd(l, HIPRT_ONE_BF16);
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to pre increment a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator++(__hip_bfloat16& l) {
  l = __hadd(l, HIPRT_ONE_BF16);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to post decrement a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator--(__hip_bfloat16& l, const int) {
  auto ret = l;
  l = __hsub(l, HIPRT_ONE_BF16);
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to pre decrement a __hip_bfloat16 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator--(__hip_bfloat16& l) {
  l = __hsub(l, HIPRT_ONE_BF16);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to add-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator+=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hadd(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to subtract-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator-=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hsub(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to divide two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 operator/(const __hip_bfloat16& l,
                                                     const __hip_bfloat16& r) {
  return __hdiv(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_ARITH
 * \brief Operator to divide-assign two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16& operator/=(__hip_bfloat16& l, const __hip_bfloat16& r) {
  l = __hdiv(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to multiply two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator*(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __hmul2(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to multiply-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator*=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __hmul2(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to unary+ on a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator+(const __hip_bfloat162& l) { return l; }

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to add two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator+(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __hadd2(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to negate a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator-(const __hip_bfloat162& l) {
  return __hneg2(l);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to subtract two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator-(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __hsub2(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to post increment a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator++(__hip_bfloat162& l, const int) {
  auto ret = l;
  l = __hadd2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to pre increment a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator++(__hip_bfloat162& l) {
  l = __hadd2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to post decrement a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator--(__hip_bfloat162& l, const int) {
  auto ret = l;
  l = __hsub2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return ret;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to pre decrement a __hip_bfloat162 number
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator--(__hip_bfloat162& l) {
  l = __hsub2(l, {HIPRT_ONE_BF16, HIPRT_ONE_BF16});
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to add-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator+=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __hadd2(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to subtract-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator-=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __hsub2(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to divide two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 operator/(const __hip_bfloat162& l,
                                                      const __hip_bfloat162& r) {
  return __h2div(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_ARITH
 * \brief Operator to divide-assign two __hip_bfloat162 numbers
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162& operator/=(__hip_bfloat162& l,
                                                        const __hip_bfloat162& r) {
  l = __h2div(l, r);
  return l;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values
 */
__BF16_HOST_DEVICE_STATIC__ bool __heq(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a == (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hequ(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a < (__bf16)b) && !((__bf16)a > (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a > (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgtu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a <= (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - greater than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hge(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a >= (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered greater than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hgeu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a < (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - not equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hne(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return ((__bf16)a < (__bf16)b) || ((__bf16)a > (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered not equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hneu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a == (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return max
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmax(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  auto a_nan = __hisnan(a), b_nan = __hisnan(b);
  if (a_nan || b_nan) {
    if (a_nan && b_nan) return HIPRT_NAN_BF16;  // return canonical NaN
    return a_nan ? b : a;
  }
  return (__bf16)a > (__bf16)b ? a : b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - return min
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __hmin(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  auto a_nan = __hisnan(a), b_nan = __hisnan(b);
  if (a_nan || b_nan) {
    if (a_nan && b_nan) return HIPRT_NAN_BF16;  // return canonical NaN
    return a_nan ? b : a;
  }
  return (__bf16)a < (__bf16)b ? a : b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than operator
 */
__BF16_HOST_DEVICE_STATIC__ bool __hlt(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a < (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than
 */
__BF16_HOST_DEVICE_STATIC__ bool __hltu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a >= (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - less than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hle(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return (__bf16)a <= (__bf16)b;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Compare two bfloat162 values - unordered less than equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hleu(const __hip_bfloat16 a, const __hip_bfloat16 b) {
  return !((__bf16)a > (__bf16)b);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Checks if number is inf
 */
__BF16_HOST_DEVICE_STATIC__ int __hisinf(const __hip_bfloat16 a) {
  __hip_bfloat16_raw hr = a;
  return !(~hr.x & 0x7f80) && !(hr.x & 0x7f);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbeq2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __heq(a.x, b.x) && __heq(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks if two numbers are equal - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbequ2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hequ(a.x, b.x) && __hequ(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbge2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hge(a.x, b.x) && __hge(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgeu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgeu(a.x, b.x) && __hgeu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgt(a.x, b.x) && __hgt(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbgtu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hgtu(a.x, b.x) && __hgtu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hble2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hle(a.x, b.x) && __hle(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbleu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hleu(a.x, b.x) && __hleu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hblt2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hlt(a.x, b.x) && __hlt(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b - unordered
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbltu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hltu(a.x, b.x) && __hltu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbne2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hne(a.x, b.x) && __hne(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b
 */
__BF16_HOST_DEVICE_STATIC__ bool __hbneu2(const __hip_bfloat162 a, const __hip_bfloat162 b) {
  return __hneu(a.x, b.x) || __hneu(a.y, b.y);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a != b, returns 1.0 if equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __heq2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__heq(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__heq(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a >= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hge2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hge(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hge(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a > b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hgt2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hgt(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hgt(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ONE_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a is NaN, returns 1.0 if NaN, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hisnan2(const __hip_bfloat162 a) {
  return __hip_bfloat162{{__hisnan(a.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hisnan(a.y) ? HIPRT_ONE_BF16 : HIPRT_ONE_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a <= b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hle2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hle(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hle(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Check for a < b, returns 1.0 if greater than equal, otherwise 0.0
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hlt2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hlt(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hlt(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns max of two elements
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmax2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162(__hmax(a.x, b.x), __hmax(a.y, b.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Returns min of two elements
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hmin2(const __hip_bfloat162 a,
                                                    const __hip_bfloat162 b) {
  return __hip_bfloat162(__hmin(a.x, b.x), __hmin(a.y, b.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Checks for not equal to
 */
__BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __hne2(const __hip_bfloat162 a,
                                                   const __hip_bfloat162 b) {
  return __hip_bfloat162{{__hne(a.x, b.x) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16},
                         {__hne(a.y, b.y) ? HIPRT_ONE_BF16 : HIPRT_ZERO_BF16}};
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform an equal compare on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator==(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __heq(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a not equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator!=(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hneu(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a less than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hlt(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a less than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<=(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hle(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a greater than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hgt(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a greater than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>=(const __hip_bfloat16& l, const __hip_bfloat16& r) {
  return __hge(l, r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform an equal compare on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator==(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 ret = __heq2(l, r);
  return ret.x != 0.0f && ret.y != 0.0f;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a not equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator!=(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  return !(l == r);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a less than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x < fr.x && fl.x < fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a less than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator<=(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x <= fr.x && fl.x <= fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_COMP
 * \brief Operator to perform a greater than on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x > fr.x && fl.x > fr.y;
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_COMP
 * \brief Operator to perform a greater than equal on two __hip_bfloat16 numbers
 */
__BF16_HOST_DEVICE_STATIC__ bool operator>=(const __hip_bfloat162& l, const __hip_bfloat162& r) {
  float2 fl = l, fr = r;
  return fl.x >= fr.x && fl.x >= fr.y;
}

#if defined(__clang__) && defined(__HIP__)
/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate ceil of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hceil(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_ceil(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate cosine of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hcos(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_cos_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hexp(const __hip_bfloat16 h) {
  // FIXME: Manual promotion to float unnecessary
  return __float2bfloat16(__builtin_elementwise_exp(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential 10 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hexp10(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_exp10(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate exponential 2 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hexp2(const __hip_bfloat16 h) {
  // FIXME: Manual promotion to float unnecessary
  return __float2bfloat16(__builtin_elementwise_exp2(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate floor of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hfloor(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_floor(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate natural log of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hlog(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_log(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate log 10 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hlog10(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_log10(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate log 2 of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hlog2(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_log2(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate reciprocal
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hrcp(const __hip_bfloat16 h) {
  return __float2bfloat16(1.0f / (__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Round to nearest int
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hrint(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_rint(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Reciprocal square root
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hrsqrt(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_rsqrt_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate sin of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hsin(const __hip_bfloat16 h) {
  return __float2bfloat16(__ocml_sin_f32(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate sqrt of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 hsqrt(const __hip_bfloat16 h) {
  // FIXME: Just directly use elementwise sqrt on the bfloat value
  // and don't promote
  return __float2bfloat16(__builtin_elementwise_sqrt(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Calculate truncate of bfloat16
 */
__BF16_DEVICE_STATIC__ __hip_bfloat16 htrunc(const __hip_bfloat16 h) {
  return __float2bfloat16(__builtin_elementwise_trunc(__bfloat162float(h)));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate ceil of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2ceil(const __hip_bfloat162 h) {
  return __hip_bfloat162(hceil(h.x), hceil(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate cosine of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2cos(const __hip_bfloat162 h) {
  return __hip_bfloat162(hcos(h.x), hcos(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp(const __hip_bfloat162 h) {
  return __hip_bfloat162(hexp(h.x), hexp(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 10 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp10(const __hip_bfloat162 h) {
  return __hip_bfloat162(hexp10(h.x), hexp10(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate exponential 2 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2exp2(const __hip_bfloat162 h) {
  return __hip_bfloat162(hexp2(h.x), hexp2(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate floor of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2floor(const __hip_bfloat162 h) {
  return __hip_bfloat162(hfloor(h.x), hfloor(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate natural log of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log(const __hip_bfloat162 h) {
  return __hip_bfloat162(hlog(h.x), hlog(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 10 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log10(const __hip_bfloat162 h) {
  return __hip_bfloat162(hlog10(h.x), hlog10(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate log 2 of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2log2(const __hip_bfloat162 h) {
  return __hip_bfloat162(hlog2(h.x), hlog2(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rcp(const __hip_bfloat162 h) {
  return __hip_bfloat162(hrcp(h.x), hrcp(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector round to nearest int
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rint(const __hip_bfloat162 h) {
  return __hip_bfloat162(hrint(h.x), hrint(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate vector reciprocal square root
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2rsqrt(const __hip_bfloat162 h) {
  return __hip_bfloat162(hrsqrt(h.x), hrsqrt(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sin of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2sin(const __hip_bfloat162 h) {
  return __hip_bfloat162(hsin(h.x), hsin(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate sqrt of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2sqrt(const __hip_bfloat162 h) {
  return __hip_bfloat162(hsqrt(h.x), hsqrt(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Calculate truncate of bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 h2trunc(const __hip_bfloat162 h) {
  return __hip_bfloat162(htrunc(h.x), htrunc(h.y));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT16_MATH
 * \brief Atomic add bfloat16
 */
inline __device__ __hip_bfloat16 atomicAdd(__hip_bfloat16* address, __hip_bfloat16 value) {
  return static_cast<__hip_bfloat16>(__scoped_atomic_fetch_add(
      (__bf16*)address, static_cast<__bf16>(value), __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE));
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Atomic add bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 atomicAdd(__hip_bfloat162* address, __hip_bfloat162 value) {
  typedef __bf16 __bf16_2 __attribute__((ext_vector_type(2)));
  static_assert(sizeof(__bf16_2) == sizeof(unsigned int));

  union {
    __bf16_2 vec;
    unsigned int u32;
  } expected, desired;

  unsigned int* atomic_ptr = (unsigned int*)address;
  expected.u32 = __scoped_atomic_load_n(atomic_ptr, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);

  do {
    desired.vec = expected.vec + static_cast<__bf16_2>(value);
  } while (!__scoped_atomic_compare_exchange_n(atomic_ptr, &expected.u32, desired.u32, 0,
                                               __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE,
                                               __MEMORY_SCOPE_DEVICE));
  return static_cast<__hip_bfloat162>(expected.vec);
}

/**
 * \ingroup HIP_INTRINSIC_BFLOAT162_MATH
 * \brief Unsafe Atomic add bfloat162
 */
__BF16_DEVICE_STATIC__ __hip_bfloat162 unsafeAtomicAdd(__hip_bfloat162* address,
                                                       __hip_bfloat162 value) {
#if __has_builtin(__builtin_amdgcn_flat_atomic_fadd_v2bf16)
  typedef short __attribute__((ext_vector_type(2))) vec_short2;
  static_assert(sizeof(vec_short2) == sizeof(__hip_bfloat162_raw));
  union {
    __hip_bfloat162_raw bf162_raw;
    vec_short2 vs2;
  } u{static_cast<__hip_bfloat162_raw>(value)};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_flat_atomic_fadd_v2bf16))
    u.vs2 = __builtin_amdgcn_flat_atomic_fadd_v2bf16((vec_short2*)address, u.vs2);
  return static_cast<__hip_bfloat162>(u.bf162_raw);
#else
  static_assert(sizeof(unsigned int) == sizeof(__hip_bfloat162_raw));
  union u_hold {
    __hip_bfloat162_raw h2r;
    unsigned int u32;
  };
  u_hold old_val, new_val;
  old_val.u32 =
      __hip_atomic_load((unsigned int*)address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  do {
    new_val.h2r = __hadd2(old_val.h2r, value);
  } while (!__hip_atomic_compare_exchange_strong((unsigned int*)address, &old_val.u32, new_val.u32,
                                                 __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT));
  return old_val.h2r;
#endif
}
__BF16_DEVICE_STATIC__ __hip_bfloat16 unsafeAtomicAdd(__hip_bfloat16* address,
                                                      __hip_bfloat16 value) {
  static_assert(sizeof(unsigned short int) == sizeof(__hip_bfloat16_raw));
  unsigned short int* address_as_short = reinterpret_cast<unsigned short int*>(address);
  // Align to 4 bytes
  unsigned int* aligned_addr = __builtin_bit_cast(
      unsigned int*, __builtin_bit_cast(unsigned long long int, address_as_short) &
                         (unsigned long long int)(~0x3));

  bool is_lower = __builtin_bit_cast(unsigned long long int, aligned_addr) ==
                  __builtin_bit_cast(unsigned long long int, address);

  __hip_bfloat162 fval;
  if (is_lower)
    fval = __halves2bfloat162(value, __float2bfloat16(0.0f));
  else
    fval = __halves2bfloat162(__float2bfloat16(0.0f), value);

  __hip_bfloat162* in = (__hip_bfloat162*)(aligned_addr);
  __hip_bfloat162 out = unsafeAtomicAdd(in, fval);
  if (is_lower) return __low2bfloat16(out);
  return __high2bfloat16(out);
}
#endif  // defined(__clang__) && defined(__HIP__)
#elif defined(__GNUC__) || defined(_MSC_VER)
#if !defined(__HIPCC_RTC__)
#include "amd_hip_bf16_gcc.h"
#endif
#endif
#pragma pop_macro("MAYBE_UNDEF")
#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(__HIPCC_RTC__)
#define __OCP_FP_HOST__
#define __OCP_FP_DEVICE__ __device__
#define __OCP_FP_HOST_DEVICE__ __OCP_FP_HOST__ __OCP_FP_DEVICE__
#define __OCP_FP_DEVICE_STATIC__ __OCP_FP_DEVICE__ static __inline__ __attribute__((always_inline))
#define __OCP_FP_HOST_DEVICE_STATIC__ __OCP_FP_HOST_DEVICE__ static
#else
#include "host_defines.h"

#define __OCP_FP_HOST__ __host__
#define __OCP_FP_DEVICE__ __device__
#define __OCP_FP_HOST_DEVICE__ __OCP_FP_HOST__ __OCP_FP_DEVICE__
#define __OCP_FP_DEVICE_STATIC__ __OCP_FP_DEVICE__ static __inline__ __attribute__((always_inline))
#define __OCP_FP_HOST_DEVICE_STATIC__ __OCP_FP_HOST_DEVICE__ static
#endif

static_assert(sizeof(unsigned int) == 4, "");
static_assert(sizeof(float) == 4, "");
static_assert(sizeof(unsigned short) == 2, "");

#if (defined(__clang__) && (__clang_major__ > 17) && defined(__HIP__)) ||                          \
    (defined(__GNUC__) && (__GNUC__ > 13))
static_assert(sizeof(__bf16) == 2, "");
static_assert(sizeof(_Float16) == 2, "");
#endif

// Although we do have some abstractions of half and bfloat16, since this will be a standalone
// header which will act as a base abstraction, and will be maintained in the future, it makes sense
// to keep these vector types separate from existing implementations. We can add conversion
// functions in a different header using these functions.
typedef __hip_uint8_t __amd_fp8_storage_t;
typedef __hip_uint16_t __amd_fp8x2_storage_t;
typedef __hip_uint8_t __amd_fp4x2_storage_t;
typedef __hip_uint32_t __amd_fp4x8_storage_t;
typedef __bf16 __amd_bf16_storage_t;
typedef _Float16 __amd_fp16_storage_t;
typedef __hip_int8_t __amd_scale_t;

#if defined(__clang__) && (__clang_major__ > 17) && defined(__HIP__)
typedef unsigned int __attribute__((ext_vector_type(2))) __amd_uintx2_storage_t;
typedef __hip_uint8_t __attribute__((ext_vector_type(8))) __amd_fp8x8_storage_t;
typedef __bf16 __attribute__((ext_vector_type(2))) __amd_bf16x2_storage_t;
typedef __bf16 __attribute__((ext_vector_type(8))) __amd_bf16x8_storage_t;
typedef __bf16 __attribute__((ext_vector_type(16))) __amd_bf16x16_storage_t;
typedef __bf16 __attribute__((ext_vector_type(32))) __amd_bf16x32_storage_t;
typedef float __attribute__((ext_vector_type(2))) __amd_floatx2_storage_t;
typedef float __attribute__((ext_vector_type(8))) __amd_floatx8_storage_t;
typedef float __attribute__((ext_vector_type(16))) __amd_floatx16_storage_t;
typedef float __attribute__((ext_vector_type(32))) __amd_floatx32_storage_t;
typedef _Float16 __attribute__((ext_vector_type(2))) __amd_fp16x2_storage_t;
typedef _Float16 __attribute__((ext_vector_type(8))) __amd_fp16x8_storage_t;
typedef _Float16 __attribute__((ext_vector_type(16))) __amd_fp16x16_storage_t;
typedef _Float16 __attribute__((ext_vector_type(32))) __amd_fp16x32_storage_t;
typedef __hip_uint32_t __attribute__((ext_vector_type(3))) __amd_fp6x16_storage_t;
typedef __hip_uint32_t __attribute__((ext_vector_type(6))) __amd_fp6x32_storage_t;
typedef short __attribute__((ext_vector_type(2))) __amd_shortx2_storage_t;
#elif defined(__GNUC__) && (__GNUC__ > 13)
/* GCC expects vector size in bytes */
typedef unsigned int __attribute__((vector_size(8))) __amd_uintx2_storage_t;
typedef uint8_t __attribute__((vector_size(8))) __amd_fp8x8_storage_t;
typedef __bf16 __attribute__((vector_size(4))) __amd_bf16x2_storage_t;
typedef __bf16 __attribute__((vector_size(16))) __amd_bf16x8_storage_t;
typedef __bf16 __attribute__((vector_size(32))) __amd_bf16x16_storage_t;
typedef __bf16 __attribute__((vector_size(64))) __amd_bf16x32_storage_t;
typedef float __attribute__((vector_size(8))) __amd_floatx2_storage_t;
typedef float __attribute__((vector_size(32))) __amd_floatx8_storage_t;
typedef float __attribute__((vector_size(64))) __amd_floatx16_storage_t;
typedef float __attribute__((vector_size(128))) __amd_floatx32_storage_t;
typedef _Float16 __attribute__((vector_size(4))) __amd_fp16x2_storage_t;
typedef _Float16 __attribute__((vector_size(16))) __amd_fp16x8_storage_t;
typedef _Float16 __attribute__((vector_size(32))) __amd_fp16x16_storage_t;
typedef _Float16 __attribute__((vector_size(64))) __amd_fp16x32_storage_t;
typedef uint32_t __attribute__((vector_size(12))) __amd_fp6x16_storage_t;
typedef uint32_t __attribute__((vector_size(24))) __amd_fp6x32_storage_t;
typedef short __attribute__((vector_size(4))) __amd_shortx2_storage_t;
#endif

#if (defined(__clang__) && (__clang_major__ > 17) && defined(__HIP__)) ||                          \
    (defined(__GNUC__) && (__GNUC__ > 13))
static_assert(sizeof(__amd_uintx2_storage_t) == sizeof(__amd_fp8x8_storage_t), "");
#endif
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once
#if defined(__gfx950__)
#define HIP_ENABLE_GFX950_OCP_BUILTINS 1
#else
#define HIP_ENABLE_GFX950_OCP_BUILTINS 0
#endif
#if defined(__gfx1250__)
#define HIP_ENABLE_GFX1250_OCP_BUILTINS 1
#else
#define HIP_ENABLE_GFX1250_OCP_BUILTINS 0
#endif
#if !defined(__gfx950__) && !defined(__gfx1250__)
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 1
#else
#define HIP_ENABLE_HOST_OCP_CONVERSIONS 0
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_ocp_types.h"
#include "amd_hip_fp16.h"
#include "amd_hip_bf16.h"
#endif

enum hipRoundMode {
  hipRoundNearest = 0,
  hipRoundZero = 1,
  hipRoundPosInf = 2,
  hipRoundMinInf = 3,
};

#if defined(__clang__) && defined(__HIP__)

namespace internal {
__host__ __device__ static inline __amd_fp16_storage_t half_to_f16(const __half val) {
  __half_raw tmp = val;
  return tmp.data;
}

__host__ __device__ static inline __amd_fp16x2_storage_t half2_to_f16x2(const __half2 val) {
  __half2_raw tmp = val;
  return tmp.data;
}

__host__ __device__ static inline __amd_bf16_storage_t hipbf16_to_bf16(const __hip_bfloat16 val) {
  static_assert(sizeof(__hip_bfloat16) == sizeof(__amd_bf16_storage_t));
  union {
    __hip_bfloat16 hip_bf16;
    __amd_bf16_storage_t bf16;
  } u{val};
  return u.bf16;
}

__host__ __device__ static inline __amd_bf16x2_storage_t hipbf162_to_bf16x2(const __hip_bfloat162 val) {
  static_assert(sizeof(__hip_bfloat162) == sizeof(__amd_bf16x2_storage_t));
  union {
    __hip_bfloat162 hip_bf16;
    __amd_bf16x2_storage_t bf16;
  } u{val};
  return u.bf16;
}

}  // namespace internal

#endif  // defined(__clang__) && defined(__HIP__)
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file
 * \brief amd_hip_fp8.h header, for AMD fp8 data types
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP8_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP8_H_

#if (defined(__gfx942__) || defined(__gfx1200__) || defined(__gfx1201__) ||                        \
     defined(__gfx950__) || defined (__gfx1250__)) &&                                              \
    __HIP_DEVICE_COMPILE__
#define HIP_FP8_CVT_FAST_PATH 1
#else
#define HIP_FP8_CVT_FAST_PATH 0
#endif

#if defined(__gfx942__) && __HIP_DEVICE_COMPILE__
#define HIP_FP8_TYPE_OCP 0
#define HIP_FP8_TYPE_FNUZ 1
#elif (defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx950__) ||                      \
       defined(__gfx1250__)) &&                                                                    \
       __HIP_DEVICE_COMPILE__
#define HIP_FP8_TYPE_OCP 1
#define HIP_FP8_TYPE_FNUZ 0
#else
#define HIP_FP8_TYPE_FNUZ 1
#define HIP_FP8_TYPE_OCP 1
#endif

#if defined(__HIPCC_RTC__)
#if HIP_FP8_TYPE_FNUZ
#define ENABLE_FNUZ_HIPRTC 1
#else
#define ENABLE_FNUZ_HIPRTC 0
#endif
#if HIP_FP8_TYPE_OCP
#define ENABLE_OCP_HIPRTC 1
#else
#define ENABLE_OCP_HIPRTC 0
#endif
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_bf16.h"
#include "amd_hip_mx_common.h"

#include <hip/amd_detail/amd_hip_common.h>
#include <climits>

#include "host_defines.h"
#include "amd_hip_vector_types.h"  // float2 etc
#include "amd_hip_fp16.h"          // __half_raw
#include "math_fwd.h"              // ocml device functions
#include "hip_assert.h"            // hip assertions
#define __HIP_SCHAR_MAX SCHAR_MAX
#define __HIP_SCHAR_MIN SCHAR_MIN
#define __HIP_UCHAR_MAX UCHAR_MAX
#define __HIP_SHRT_MIN SHRT_MIN
#define __HIP_SHRT_MAX SHRT_MAX
#define __HIP_CHAR_MIN CHAR_MIN
#define __HIP_CHAR_MAX CHAR_MAX
#else
// fp8 header uses all this, since we do not include standard header, we include this
#define __HIP_SCHAR_MAX __SCHAR_MAX__
#define __HIP_SCHAR_MIN (-__SCHAR_MAX__ - 1)
#define __HIP_UCHAR_MAX (__SCHAR_MAX__ * 2 + 1)
#define __HIP_SHRT_MIN (-__SHRT_MAX__ - 1)
#define __HIP_SHRT_MAX __SHRT_MAX__
#ifdef __CHAR_UNSIGNED__ /* -funsigned-char */
#define __HIP_CHAR_MIN 0
#define __HIP_CHAR_MAX __HIP_UCHAR_MAX
#else
#define __HIP_CHAR_MIN __HIP_SCHAR_MIN
#define __HIP_CHAR_MAX __SCHAR_MAX__
#endif
#endif  // !defined(__HIPCC_RTC__)

#if defined(__HIPCC_RTC__)
#define __FP8_HOST_DEVICE__ __device__
#define __FP8_HOST_DEVICE_STATIC__ __FP8_HOST_DEVICE__ static
#else
#define __FP8_HOST_DEVICE__ __host__ __device__
#define __FP8_HOST_DEVICE_STATIC__ __FP8_HOST_DEVICE__ static inline
#endif  // __HIPCC_RTC__

#define __FP8_HOST__ __host__
#define __FP8_HOST_STATIC__ __FP8_HOST__ static inline


#if !defined(__HIPCC_RTC__)
static_assert(CHAR_BIT == 8, "byte size should be of 8 bits");
#endif
static_assert(sizeof(unsigned char) == 1, "");
static_assert(sizeof(unsigned short int) == 2, "");
static_assert(sizeof(unsigned int) == 4, "");

/**
 * \brief Describes FP8 interpretation
 */
enum __hip_fp8_interpretation_t {
  __HIP_E4M3 = 0,      /**< OCP E4M3 */
  __HIP_E5M2 = 1,      /**< OCP E5M2 */
  __HIP_E4M3_FNUZ = 2, /**< Standard FP8*/
  __HIP_E5M2_FNUZ = 3, /**< BF8 */
};

/**
 * \brief Describes saturation behavior
 */
enum __hip_saturation_t {
  __HIP_NOSAT = 0,     /**< No saturation */
  __HIP_SATFINITE = 1, /**< Saturate to finite */
};

/** \typedef __hip_fp8_storage_t
 *
 * \brief type to store single fp8 number
 */
typedef unsigned char __hip_fp8_storage_t;


/** \typedef __hip_fp8x2_storage_t
 *
 * \brief type to store two fp8 numbers
 */
typedef unsigned short int __hip_fp8x2_storage_t;


/** \typedef __hip_fp8x4_storage_t
 *
 * \brief type to store four fp8 numbers
 */
typedef unsigned int __hip_fp8x4_storage_t;


namespace internal {

// Assertions to check for supported conversion types
#define __assert_ocp_support(interp)                                                               \
  {                                                                                                \
    if (interp != __HIP_E4M3 && interp != __HIP_E5M2) {                                            \
      __hip_assert(false && "type is unsupported by current target device");                       \
    }                                                                                              \
  }
#define __assert_fnuz_support(interp)                                                              \
  {                                                                                                \
    if (interp != __HIP_E4M3_FNUZ && interp != __HIP_E5M2_FNUZ) {                                  \
      __hip_assert(false && "type is unsupported by current target device");                       \
    }                                                                                              \
  }

__FP8_HOST_DEVICE_STATIC__ void __is_interpret_supported(__hip_fp8_interpretation_t interp) {
#if __HIP_DEVICE_COMPILE__
#if HIP_FP8_TYPE_OCP
  __assert_ocp_support(interp);
#endif
#if HIP_FP8_TYPE_FNUZ
  __assert_fnuz_support(interp);
#endif
#endif
}

// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L39
// This has been modified to add double types conversion as well
template <typename T, bool is_fnuz>
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t cast_to_f8(T _x, int wm, int we, bool clip = false,
                                                          bool stoch = false,
                                                          unsigned int rng = 0) {
#if defined(__clang__) && defined(__HIP__)
  constexpr bool is_half = __hip_internal::is_same<T, _Float16>::value;
  constexpr bool is_float = __hip_internal::is_same<T, float>::value;
  constexpr bool is_double = __hip_internal::is_same<T, double>::value;
#else   // compiling for host
  constexpr bool is_half = std::is_same<T, _Float16>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  constexpr bool is_double = std::is_same<T, double>::value;
#endif  // defined(__clang__) && defined(__HIP__)
  static_assert(is_half || is_float || is_double, "Only half, float and double can be cast to f8");

  const int mfmt = (sizeof(T) == 8) ? 52 : ((sizeof(T) == 4) ? 23 : 10);
  unsigned long long x;

  if (sizeof(T) == 8)
    x = reinterpret_cast<unsigned long long&>(_x);
  else if (sizeof(T) == 4)
    x = reinterpret_cast<unsigned int&>(_x);
  else
    x = reinterpret_cast<unsigned short int&>(_x);


  unsigned long long head, mantissa;
  int exponent, bias;
  unsigned int sign;
  unsigned long long fInf, mask;

  if (sizeof(T) == 8) {
    head = x & 0xFFF0000000000000ull;
    mantissa = x & 0xFFFFFFFFFFFFFull;
    exponent = (head >> 52) & 0x7FF;
    sign = head >> 63;
    bias = 1023;
    fInf = 0x7FF0000000000000ull;
    mask = 0x7FFFFFFFFFFFFFFFull;
  } else if (sizeof(T) == 4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign = head >> 31;
    bias = 127;
    fInf = 0x7F800000;
    mask = 0x7FFFFFFF;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head >> 10) & 0x1F;
    sign = head >> 15;
    bias = 15;
    fInf = 0x7C00;
    mask = 0x7FFF;
  }
  unsigned int signed_inf = 0;
  unsigned int nan = 0;
  if (is_fnuz) {
    signed_inf = clip ? ((sign << 7) + 0x7f) : 0x80;
    nan = 0x80;
  } else {
    if (we == 4) {  // e4m3
      signed_inf = (sign << 7) + (clip ? 0x7e : 0x7f);
    } else {  // e5m2
      signed_inf = (sign << 7) + (clip ? 0x7b : 0x7c);
    }
    nan = (sign << 7) + 0x7f;
  }
  // Max values
  unsigned long long ifmax = 0;
  if (sizeof(T) == 8) {
    if (we == 5) {  // 57344
      ifmax = 0x40EC000000000000ull;
    } else {
      if (is_fnuz) {  // 240
        ifmax = 0x406E000000000000ull;
      } else {  // 448
        ifmax = 0x407C000000000000ull;
      }
    }
  } else if (sizeof(T) == 4) {
    if (we == 5) {
      ifmax = 0x47600000;
    } else {
      if (is_fnuz) {
        ifmax = 0x43700000;
      } else {
        ifmax = 0x43E00000;
      }
    }
  } else {
    if (we == 5) {
      ifmax = 0x7B00;
    } else {
      if (is_fnuz) {
        ifmax = 0x5B80;
      } else {
        ifmax = 0x5F00;
      }
    }
  }
  // Deal with inf and NaNs
  if ((x & fInf) == fInf) {
    if (is_fnuz || we == 4) return nan;  // funz and OCP E4M3 has no INF
    if (mantissa != 0) return nan;       // NaN
    return sign == 0 ? 0x7C : 0xFC;      // E5M2 Inf
  }

  if ((x & mask) > ifmax) {
    return signed_inf;
  }

  if (x == 0) {
    return 0;
  }

  // First need to check if it is normal or denorm as there is a difference of implict 1
  // Then need to adjust the exponent to align with the F8 exponent, in the meanwhile, shift
  // The mantissa. Then for stochastic rounding, add rng to mantissa and truncate. And for
  // RNE, no need to add rng. Then probably need to check whether there is carry and adjust
  // exponent and mantissa again

  // For IEEE bias mode, the bias is 2^(k-1) -1 where k is the width of exponent bits
  const int f8_bias = (1 << (we - 1)) - 1 + (is_fnuz ? 1 : 0);
  const int f8_denormal_act_exponent = 1 - f8_bias;  // actual exponent of f8 denormal
  // act_exponent is the actual exponent of fp32/fp16 (after subtracting bias)
  // f8_exponent is the converted f8 exponent with bias encoding
  // exponent_diff is the diff between fp32/fp16 exponent and f8 exponent,
  // the difference needs to be adjusted and mantissa shifted
  int act_exponent, f8_exponent, exponent_diff;

  if (exponent == 0) {  // fp32/fp16 is in denormal.
    /* fp32 denormal is below 2^-127 so it is usually not a concern here, we mostly concern fp16
here. In this case, f8 is usually in denormal. But there could be exceptions. fp16 denormal has
exponent bias 15 while bf8 with NANOO has exponent bias 16. It means that there are some numbers in
fp16 denormal but they are bf8 (NANOO) normals - smallest bf8 (NANOO) normal is 2^-15. fp16 numbers
where exponent==0 (actual exponent -14) and highest bit of mantissa is 1 are bf8 (NANOO) normal. In
this case, the fp16 mantissa should be shift left by 1  */
    act_exponent = exponent - bias + 1;
    exponent_diff = f8_denormal_act_exponent -
                    act_exponent;  // actual exponent is exponent-bias+1 as it is denormal
  } else {                         // fp32/fp16 is normal with implicit 1
    act_exponent = exponent - bias;
    if (act_exponent <= f8_denormal_act_exponent) {
      /* This is the case where fp32/fp16 is normal but it is in f8 denormal range.
For example fp8 nanoo mode, denormal exponent is -7, but if the fp32/fp16
actual exponent is -7, it is actually larger due to the implict 1,
Therefore it needs to be adjust to -6 and mantissa shift right by 1.
So for fp32/fp16, exponent -8 is the cut point to convert to fp8 nanoo */
      exponent_diff = f8_denormal_act_exponent - act_exponent;
    } else {              // both fp32/fp16 and f8 are in normal range
      exponent_diff = 0;  // exponent_diff=0 does not mean there is no difference for this case,
                          // act_exponent could be larger. Just that it does not need shift mantissa
    }
    mantissa += (1ull << mfmt);  // Add the implicit 1 into mantissa
  }

  bool midpoint = (mantissa & ((1ull << (mfmt - wm + exponent_diff)) - 1)) ==
                  (1ull << (mfmt - wm + exponent_diff - 1));
  /* This part is a bit tricky. The judgment of whether it is a tie needs to be done before we shift
right as shift right could rip off some residual part and make something not midpoint look like
midpoint. For example, the fp16 number 0x1002 (0 00100 0000000010), it is larger than midpoint, but
after shift right by 4 bits, it would look like midpoint.
*/

  if (exponent_diff > 0)
    mantissa >>= exponent_diff;
  else if (exponent_diff == -1)
    mantissa <<= -exponent_diff;
  bool implicit_one = mantissa & (1ull << mfmt);
  // if there is no implict 1, it  means the f8 is denormal and need to adjust to denorm exponent
  f8_exponent =
      (act_exponent + exponent_diff) /*actual f8 exponent*/ + f8_bias - (implicit_one ? 0 : 1);

  // Now we have the exponent and mantissa adjusted
  unsigned long long drop_mask = (1ull << (mfmt - wm)) - 1;
  bool odd =
      mantissa & (1ull << (mfmt - wm));  // if the least significant bit that is not truncated is 1
  mantissa +=
      (stoch ? rng : (midpoint ? (odd ? mantissa : mantissa - 1ull) : mantissa)) & drop_mask;

  // Now we deal with overflow
  if (f8_exponent == 0) {
    if ((1ull << mfmt) & mantissa) {
      f8_exponent = 1;  // denormal overflow to become normal, promote exponent
    }
  } else {
    if ((1ull << (mfmt + 1)) & mantissa) {
      mantissa >>= 1;
      f8_exponent++;
    }
  }

  mantissa >>= (mfmt - wm);

  // above range: quantize to maximum possible float of the same sign
  const int max_exp = (1 << we) - 1;
  if (f8_exponent > max_exp) {
    if (clip) {
      mantissa = (1 << wm) - 1;
      f8_exponent = max_exp;
    } else {
      return signed_inf;
    }
  }

  if (f8_exponent == 0 && mantissa == 0) return is_fnuz ? 0 : (sign << 7);
  mantissa &= (1 << wm) - 1;
  return (sign << 7) | (f8_exponent << wm) | mantissa;
}
// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_hip_f8_impl.h#L220
// This has been modified to handle double types as well
template <typename T, bool is_fnuz> __FP8_HOST_DEVICE_STATIC__ T cast_from_f8(__hip_fp8_storage_t x,
                                                                              int wm, int we,
                                                                              bool clip = false) {
#if defined(__clang__) && defined(__HIP__)
  constexpr bool is_half = __hip_internal::is_same<T, _Float16>::value;
  constexpr bool is_float = __hip_internal::is_same<T, float>::value;
  constexpr bool is_double = __hip_internal::is_same<T, double>::value;
#else
  constexpr bool is_half = std::is_same<T, _Float16>::value;
  constexpr bool is_float = std::is_same<T, float>::value;
  constexpr bool is_double = std::is_same<T, double>::value;
#endif  // defined(__clang__) && defined(__HIP__)
  static_assert(is_half || is_float || is_double, "only half, float and double are supported");

  constexpr int weo = is_half ? 5 : (is_float ? 8 : 11);
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 52);

  T fInf, fNegInf, fNaN, fNeg0, fmax, fmin;
  if (is_half) {
    const unsigned short int ihInf = 0x7C00;
    const unsigned short int ihNegInf = 0xFC00;
    const unsigned short int ihNaN = 0x7C01;
    const unsigned short int ihNeg0 = 0x8000;
    /* Max number in e5m2 57344*/
    const unsigned short int ifmax = 0x7B00;
    const unsigned short int ifmin = 0xFB00;
    fInf = reinterpret_cast<const _Float16&>(ihInf);
    fNegInf = reinterpret_cast<const _Float16&>(ihNegInf);
    fNaN = reinterpret_cast<const _Float16&>(ihNaN);
    fNeg0 = reinterpret_cast<const _Float16&>(ihNeg0);
    fmax = reinterpret_cast<const _Float16&>(ifmax);
    fmin = reinterpret_cast<const _Float16&>(ifmin);
  } else if (is_float) {
    const unsigned int ifInf = 0x7F800000;
    const unsigned int ifNegInf = 0xFF800000;
    const unsigned int ifNaN = 0x7F800001;
    const unsigned int ifNeg0 = 0x80000000;
    /* Max number in e5m2 57344*/
    const unsigned int ifmax = 0x47600000;
    const unsigned int ifmin = 0xC7600000;
    fInf = reinterpret_cast<const float&>(ifInf);
    fNegInf = reinterpret_cast<const float&>(ifNegInf);
    fNaN = reinterpret_cast<const float&>(ifNaN);
    fNeg0 = reinterpret_cast<const float&>(ifNeg0);
    fmax = reinterpret_cast<const float&>(ifmax);
    fmin = reinterpret_cast<const float&>(ifmin);
  } else if (is_double) {
    const unsigned long long ifInf = 0x7FF0000000000000ull;
    const unsigned long long ifNegInf = 0xFFF0000000000000ull;
    const unsigned long long ifNaN = 0x7FF0000000000001ull;
    const unsigned long long ifNeg0 = 0x8000000000000000ull;
    /* Max number in e5m2 57344*/
    const unsigned long long ifmax = 0x40EC000000000000ull;
    const unsigned long long ifmin = 0xC0EC000000000000ull;
    fInf = reinterpret_cast<const double&>(ifInf);
    fNegInf = reinterpret_cast<const double&>(ifNegInf);
    fNaN = reinterpret_cast<const double&>(ifNaN);
    fNeg0 = reinterpret_cast<const double&>(ifNeg0);
    fmax = reinterpret_cast<const double&>(ifmax);
    fmin = reinterpret_cast<const double&>(ifmin);
  }

  if (x == 0) {
    return 0;
  }

  unsigned long long sign = x >> 7;
  unsigned long long mantissa = x & ((1 << wm) - 1);
  int exponent = (x & 0x7F) >> wm;
  if (is_fnuz) {
    if (x == 0x80) {
      return fNaN;
    }
  } else {
    if (x == 0x80) {
      return fNeg0;
    }
    if (we == 4) {  // e4m3
      if ((x & 0x7F) == 0x7F) {
        return fNaN;
      }
    } else if ((x & 0x7C) == 0x7C) {  // e5m2 NaN/Inf
      if ((x & 0x3) == 0) {           // Inf
        if (clip) {
          return sign ? fmin : fmax;
        }
        return sign ? fNegInf : fInf;
      }
      return fNaN;
    }
  }

#if defined(__clang__) && defined(__HIP__)
  typename __hip_internal::conditional<
      sizeof(T) == 2, unsigned short int,
      typename __hip_internal::conditional<sizeof(T) == 4, unsigned int,
                                           unsigned long long>::type>::type retval;
#else
  typename std::conditional<sizeof(T) == 2, unsigned short int,
                            typename std::conditional<sizeof(T) == 4, unsigned int,
                                                      unsigned long long>::type>::type retval;
#endif

  if (we == 5 && is_half && !is_fnuz) {
    retval = x << 8;
    return reinterpret_cast<const T&>(retval);
  }

  const int exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1)) + 1 - (is_fnuz ? 1 : 0);

  // subnormal input
  if (exponent == 0) {
#if __HIP_DEVICE_COMPILE__
    // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + __clz(mantissa) - (32 - wm);
#else
    int sh = 1 + __builtin_clz(mantissa) - (32 - wm);
#endif
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1ull << wm) - 1);
  }
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if (exponent <= 0) {
    mantissa |= 1ull << wmo;
    mantissa >>= 1 - exponent;
    exponent = 0;
  }

  if (sizeof(T) == 2)
    retval = (sign << 15) | (exponent << 10) | mantissa;
  else if (sizeof(T) == 4)
    retval = (sign << 31) | (exponent << 23) | mantissa;
  else
    retval = (sign << 63) | (static_cast<unsigned long long>(exponent) << 52) | mantissa;
  return reinterpret_cast<const T&>(retval);
}

#if HIP_FP8_CVT_FAST_PATH
// The conversion function is from rocblas
// https://github.com/ROCm/rocBLAS/blob/9b7f692abe3c54b88d1e77e045a7db7f1f188b69/library/include/internal/rocblas_float8.h#L79
template <bool stochastic_rounding = false>
static __device__ __hip_fp8_storage_t cast_to_f8_from_f32(float v, bool saturate,
                                                          __hip_fp8_interpretation_t interpret,
                                                          unsigned int rng = 0) {
  __hip_fp8_storage_t i8data;
  union {
    float fval;
    unsigned int i32val;
    unsigned char i8val[4];  // NOTE: not endian independent
  } val;

  unsigned int ival = 0;
  val.fval = v;

  if (saturate) {
    if (interpret == __HIP_E4M3_FNUZ) {
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          val.fval = __builtin_amdgcn_fmed3f(val.fval, 240.0, -240.0);
      }
    } else if (interpret == __HIP_E4M3) {             // OCP type
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          val.fval = __builtin_amdgcn_fmed3f(val.fval, 448.0, -448.0);
      }
    } else {
      if ((val.i32val & 0x7F800000) != 0x7F800000) {  /// propagate NAN/INF, no clipping
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          val.fval = __builtin_amdgcn_fmed3f(val.fval, 57344.0, -57344.0);
      }
    }
  }

  if (stochastic_rounding) {
    ival = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
               ? (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_sr_fp8_f32)
                      ? __builtin_amdgcn_cvt_sr_fp8_f32(val.fval, rng, ival, 0)
                      : 0u)
               : (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_sr_bf8_f32)
                      ? __builtin_amdgcn_cvt_sr_bf8_f32(val.fval, rng, ival, 0)
                      : 0u);  // 0 pos
    val.i32val = ival;
    i8data = val.i8val[0];  // little endian
  } else {                  // RNE CVT
    ival = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
               ? (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f32)
                      ? __builtin_amdgcn_cvt_pk_fp8_f32(val.fval, val.fval, ival, false)
                      : 0u)
               : (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f32)
                      ? __builtin_amdgcn_cvt_pk_bf8_f32(val.fval, val.fval, ival, false)
                      : 0u);  // false -> WORD0
    val.i32val = ival;
    i8data = val.i8val[0];
  }
  return i8data;
}

static __device__ __hip_fp8x2_storage_t
cast_to_f8x2_from_f32x2(float2 v, bool saturate, __hip_fp8_interpretation_t interpret) {
  union {
    static_assert(sizeof(float2) == sizeof(unsigned int[2]), "size mismatch");
    static_assert(sizeof(float2) == sizeof(unsigned short[4]), "size mismatch");
    float2 fval;
    unsigned int i32val[2];
    unsigned short i16val[4];
  } f2val;

  f2val.fval = v;

  if (saturate) {  /// propagate NAN/INF, no clipping
    if (interpret == __HIP_E4M3_FNUZ) {
      if ((f2val.i32val[0] & 0x7F800000) != 0x7F800000) {
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          f2val.fval.x = __builtin_amdgcn_fmed3f(f2val.fval.x, 240.0, -240.0);
      }
      if ((f2val.i32val[1] & 0x7F800000) != 0x7F800000) {
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          f2val.fval.y = __builtin_amdgcn_fmed3f(f2val.fval.y, 240.0, -240.0);
      }
    } else if (interpret == __HIP_E4M3) {
      if ((f2val.i32val[0] & 0x7F800000) != 0x7F800000) {
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          f2val.fval.x = __builtin_amdgcn_fmed3f(f2val.fval.x, 448.0, -448.0);
      }
      if ((f2val.i32val[1] & 0x7F800000) != 0x7F800000) {
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          f2val.fval.y = __builtin_amdgcn_fmed3f(f2val.fval.y, 448.0, -448.0);
      }
    } else {
      if ((f2val.i32val[0] & 0x7F800000) != 0x7F800000) {
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          f2val.fval.x = __builtin_amdgcn_fmed3f(f2val.fval.x, 57344.0, -57344.0);
      }
      if ((f2val.i32val[1] & 0x7F800000) != 0x7F800000) {
        if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_fmed3f))
          f2val.fval.y = __builtin_amdgcn_fmed3f(f2val.fval.y, 57344.0, -57344.0);
      }
    }
  }

  f2val.i32val[0] = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
                        ? (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_fp8_f32)
                               ? __builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.y, 0, false)
                               : 0u)
                        : (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_bf8_f32)
                               ? __builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.y, 0, false)
                               : 0u);

  return static_cast<__hip_fp8x2_storage_t>(f2val.i16val[0]);
}

static __device__ float cast_to_f32_from_f8(__hip_fp8_storage_t v,
                                            __hip_fp8_interpretation_t interpret) {
  union {
    unsigned int i32val;
    unsigned char i8val[4];
  } val;
  val.i8val[0] = v;

  float fval = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
                   ? (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_f32_fp8)
                          ? __builtin_amdgcn_cvt_f32_fp8(val.i32val, 0)
                          : 0.0f)
                   : (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_f32_bf8)
                          ? __builtin_amdgcn_cvt_f32_bf8(val.i32val, 0)
                          : 0.0f);
  return fval;
}

static __device__ float2 cast_to_f32x2_from_f8x2(__hip_fp8x2_storage_t v,
                                                 __hip_fp8_interpretation_t interpret) {
  union {
    unsigned int i32val;
    unsigned short i16val[2];
  } val;
  val.i16val[0] = v;

  auto f2 = (interpret == __HIP_E4M3_FNUZ) || (interpret == __HIP_E4M3)
                ? (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_fp8)
                       ? __builtin_amdgcn_cvt_pk_f32_fp8(val.i32val, false)
                       : __amd_floatx2_storage_t{})
                : (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_pk_f32_bf8)
                       ? __builtin_amdgcn_cvt_pk_f32_bf8(val.i32val, false)
                       : __amd_floatx2_storage_t{});
  return float2{f2[0], f2[1]};
}
#endif  // HIP_FP8_CVT_FAST_PATH

/* For fp8 fnuz types, finite and NaN values are supported. Zero is unsigned.
Inf are not supported. This gives us one additional number to represent.
NaN are represented by 1-0000-000 or 1-00000-00 */
__FP8_HOST_DEVICE_STATIC__ bool hip_fp8_fnuz_is_nan(__hip_fp8_storage_t a) {
  return static_cast<unsigned char>(a) == 0x80;
}

__FP8_HOST_DEVICE_STATIC__ bool hip_fp8_ocp_is_nan(__hip_fp8_storage_t a,
                                                   const __hip_fp8_interpretation_t type) {
  return (type == __HIP_E4M3)   ? ((a & 0x7f) == 0x7f)
         : (type == __HIP_E5M2) ? ((a & 0x7f) > 0x7c)
                                : false;
}

__FP8_HOST_DEVICE_STATIC__ bool hip_fp8_ocp_is_inf(__hip_fp8_storage_t a,
                                                   const __hip_fp8_interpretation_t type) {
  return (type == __HIP_E5M2) ? (a & 0x7f) == 0x7c : false;
}

}  // namespace internal

/**
 * \brief convert float to @p __hip_fp8_storage_t
 *
 * \param f float number
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(
    const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
  return internal::cast_to_f8_from_f32<false>(f, sat == __HIP_SATFINITE, interp);
#else
#if HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(
    const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(
    const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  if (interp == __HIP_E4M3_FNUZ || interp == __HIP_E5M2_FNUZ) {
    int we = interp == __HIP_E4M3_FNUZ ? 4 : 5;
    int wm = interp == __HIP_E4M3_FNUZ ? 3 : 2;
    return internal::cast_to_f8<float, true>(f, wm, we, sat == __HIP_SATFINITE);
  } else {
    int we = interp == __HIP_E4M3 ? 4 : 5;
    int wm = interp == __HIP_E4M3 ? 3 : 2;
    return internal::cast_to_f8<float, false>(f, wm, we, sat == __HIP_SATFINITE);
  }
#endif  // HIP_FP8_CVT_FAST_PATH
}


/**
 * \brief convert float2 to @p __hip_fp8x2_storage_t
 *
 * \param f2 float2 number
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(
    const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
  return internal::cast_to_f8x2_from_f32x2(f2, sat == __HIP_SATFINITE, interp);
#else
#if HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(
    const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(
    const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return static_cast<__hip_fp8x2_storage_t>(
      static_cast<unsigned short int>(__hip_cvt_float_to_fp8(f2.y, sat, interp)) << 8 |
      static_cast<unsigned short int>(__hip_cvt_float_to_fp8(f2.x, sat, interp)));
#endif  // HIP_FP8_CVT_FAST_PATH
}

/**
 * \brief convert double to @p __hip_fp8_storage_t
 *
 * \param d double val
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(
    const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(
    const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(
    const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  if (interp == __HIP_E4M3_FNUZ || interp == __HIP_E5M2_FNUZ) {
    int we = interp == __HIP_E4M3_FNUZ ? 4 : 5;
    int wm = interp == __HIP_E4M3_FNUZ ? 3 : 2;
    return internal::cast_to_f8<double, true>(d, wm, we, sat == __HIP_SATFINITE);
  } else {
    int we = interp == __HIP_E4M3 ? 4 : 5;
    int wm = interp == __HIP_E4M3 ? 3 : 2;
    return internal::cast_to_f8<double, false>(d, wm, we, sat == __HIP_SATFINITE);
  }
}

/**
 * \brief convert double2 to @p __hip_fp8x2_storage_t
 *
 * \param d2 double2 val
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(
    const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(
    const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(
    const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return static_cast<__hip_fp8x2_storage_t>(
      static_cast<unsigned short int>(__hip_cvt_double_to_fp8(d2.y, sat, interp)) << 8 |
      static_cast<unsigned short int>(__hip_cvt_double_to_fp8(d2.x, sat, interp)));
}

/**
 * \brief convert __hip_bfloat16_raw to @p __hip_fp8_storage_t
 *
 * \param hr __hip_bfloat16_raw val
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat,
                             const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat,
                             const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat,
                             const __hip_fp8_interpretation_t interp) {
#endif
  float fval = __hip_bfloat16(hr);
  return __hip_cvt_float_to_fp8(fval, sat, interp);
}

/**
 * \brief convert double2 to @p __hip_fp8x2_storage_t
 *
 * \param hr __hip_bfloat162_raw value
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t
__hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat,
                                const __hip_fp8_interpretation_t interp) {
#endif
  float2 f2 = __hip_bfloat162(hr);
  return __hip_cvt_float2_to_fp8x2(f2, sat, interp);
}

/**
 * \brief convert @p __hip_fp8_storage_t to __half_raw
 *
 * \param x __hip_fp8_storage_t val
 * \param interp interpretation of fp8
 * \return __half_raw
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __half_raw
__hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __half_raw
__hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __half_raw __hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x,
                                                        const __hip_fp8_interpretation_t interp) {
#endif
  if (interp == __HIP_E4M3_FNUZ || interp == __HIP_E5M2_FNUZ) {
    unsigned int we = interp == __HIP_E4M3_FNUZ ? 4 : 5;
    unsigned int wm = interp == __HIP_E4M3_FNUZ ? 3 : 2;
    return __half_raw{internal::cast_from_f8<_Float16, true>(x, wm, we)};
  } else {
    unsigned int we = interp == __HIP_E4M3 ? 4 : 5;
    unsigned int wm = interp == __HIP_E4M3 ? 3 : 2;
    return __half_raw{internal::cast_from_f8<_Float16, false>(x, wm, we)};
  }
}

/**
 * \brief convert @p __hip_fp8x2_storage_t to __half2_raw
 *
 * \param x __hip_fp8x2_storage_t val
 * \param interp interpretation of fp8
 * \return __half2_raw
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(
    const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(
    const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(
    const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t interp) {
#endif
  __half2 ret(static_cast<__half>(
                  __hip_cvt_fp8_to_halfraw(static_cast<__hip_fp8_storage_t>(x & 0xFF), interp)),
              static_cast<__half>(
                  __hip_cvt_fp8_to_halfraw(static_cast<__hip_fp8_storage_t>(x >> 8), interp)));
  return static_cast<__half2_raw>(ret);
}

/**
 * \brief convert __half_raw to @p __hip_fp8_storage_t
 *
 * \param x __half_raw value
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(
    const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(
    const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(
    const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return __hip_cvt_float_to_fp8(__half2float(__half(x)), sat, interp);
}

/**
 * \brief convert __half2_raw to @p __hip_fp8x2_storage_t
 *
 * \param x __half2_raw value
 * \param sat saturation of fp8
 * \param interp interpretation of fp8
 * \return __hip_fp8x2_storage_t
 */
#if HIP_FP8_CVT_FAST_PATH
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
  internal::__is_interpret_supported(interp);
#elif HIP_FP8_TYPE_OCP && HIP_FP8_TYPE_FNUZ
__FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#else
__FP8_HOST_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(
    const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t interp) {
#endif
  return __hip_cvt_float2_to_fp8x2(__half22float2(__half2(x)), sat, interp);
}

namespace hip_detail {

constexpr __hip_fp8_storage_t e8m0_NaN = 0xFFU;
constexpr __hip_internal::uint16_t bf16_NaN = 0x7FFFU;
constexpr __hip_internal::uint32_t default_float_NaN = 0x7FFF0000U;
constexpr __hip_internal::uint64_t default_double_NaN = 0x7FFF000000000000U;

constexpr __hip_internal::uint16_t bf16_sig_mask = 0x007FU;
constexpr __hip_internal::uint32_t float_sig_mask = 0x007FFFFFU;
constexpr __hip_internal::uint64_t double_sig_mask = 0x000FFFFFFFFFFFFFU;

constexpr __hip_internal::uint16_t bf16_max_exp = 0x7F80U;
constexpr __hip_internal::uint32_t float_max_exp = 0x7F800000U;
constexpr __hip_internal::uint64_t double_max_exp = 0x7FF0000000000000U;

constexpr __hip_internal::uint16_t bf16_sign_mask = 0x8000U;
constexpr __hip_internal::uint32_t float_sign_mask = 0x80000000U;
constexpr __hip_internal::uint64_t double_sign_mask = 0x8000000000000000U;

constexpr __hip_internal::uint16_t bf16_half_sig_bit = 0x0040U;
constexpr __hip_internal::uint32_t float_half_sig_bit = 0x00400000U;
constexpr __hip_internal::uint64_t double_half_sig_bit = 0x0008000000000000U;

__FP8_HOST_DEVICE_STATIC__ float __internal_cvt_e8m0_to_float(const __hip_fp8_storage_t x) {
  union {
    float as_float;
    __hip_internal::uint32_t as_int;
  } ret;

  switch (x) {
    case 0x00U: {
      ret.as_int = float_half_sig_bit;
      break;
    }
    case hip_detail::e8m0_NaN:
      ret.as_int = default_float_NaN;
      break;
    default:
      ret.as_int = static_cast<__hip_internal::uint32_t>(x) << 23;
  };
  return ret.as_float;
}

__FP8_HOST_DEVICE_STATIC__ double __internal_cvt_e8m0_to_double(const __hip_fp8_storage_t x) {
  union {
    double as_double;
    __hip_internal::uint64_t as_int;
  } ret;

  switch (x) {
    case hip_detail::e8m0_NaN:
      ret.as_int = default_double_NaN;
      break;
    default:
      // shift due to bias difference between double and single precision exponents
      ret.as_int = (static_cast<__hip_internal::uint64_t>(x + 0x0380U) << 52);
  }
  return ret.as_double;
}

// Converts e8m0 to arbitrary integer type T
// Uses conversion to float for bounds check
// Double round not a concern as e8m0 is just the f32 mantissa
template <typename T_int>
__FP8_HOST_DEVICE_STATIC__ T_int internal_cvt_e8m0_to_int_type(__hip_fp8_storage_t x) {
  float f = __internal_cvt_e8m0_to_float(x);
  if (x == hip_detail::e8m0_NaN) {
    return 0;
  }
  if (f >= __hip_internal::numeric_limits<T_int>::max()) {
    return __hip_internal::numeric_limits<T_int>::max();
  }
  // e8m0 encodes only non-negative scales (2^-127..2^127) and NaN is handled
  // above, so f is always >= 0. The lowest() clamp is therefore unreachable
  // even when T_int is signed.
  return static_cast<T_int>(f);
}

}  // namespace hip_detail

__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_e8m0(
    const double val, const __hip_saturation_t saturate, const enum hipRoundMode rounding) {
  union {
    double as_double;
    __hip_internal::uint64_t as_int;
  } u{val};

  // Shifts out mantissa bits from double dtype
  unsigned short double_exp =
      static_cast<unsigned short>((~hip_detail::double_sign_mask & u.as_int) >> 52);
  __hip_fp8_storage_t e8m0;
  if (double_exp == 0x0U) {
    e8m0 = 0x0U;
  } else {
    // Use signed arithmetic to avoid unsigned underflow for small normal exponents
    // 0x0380 is a bias difference between double and single precision (1023 - 127)
    const int float_exp = static_cast<int>(double_exp) - 0x0380;
    if (float_exp <= 0) {
      // Underflow to subnormal/zero in float domain
      e8m0 = 0x0U;
    } else if (float_exp > 0x00FF) {
      // NaN/Inf or overflow beyond 8-bit exponent range
      e8m0 = hip_detail::e8m0_NaN;
    } else {
      // Valid finite range [1..254], 0xFF yields NaN and is later handled by saturation
      e8m0 = static_cast<__hip_fp8_storage_t>(float_exp);
    }
  }

  // If there is a mantissa and the exp wont overflow round up
  if ((rounding == hipRoundPosInf) && (u.as_int & hip_detail::double_sig_mask) &&
      (!((u.as_int & ~hip_detail::double_sign_mask) < hip_detail::double_half_sig_bit)) &&
      (e8m0 < hip_detail::e8m0_NaN)) {
    ++e8m0;
  }

  // If e8m0 is NaN and exponent is a large non-inf value round down to a value
  if ((saturate == __HIP_SATFINITE) && (e8m0 == hip_detail::e8m0_NaN) &&
      ((u.as_int & ~hip_detail::double_sign_mask) <= hip_detail::double_max_exp)) {
    --e8m0;
  }
  return e8m0;
}

__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_e8m0(
    const float val, const __hip_saturation_t saturate, const enum hipRoundMode rounding) {
  union {
    float as_float;
    __hip_internal::uint32_t as_int;
  } u{val};

  // Shifts out mantissa bits from float dtype
  __hip_fp8_storage_t e8m0 = static_cast<unsigned char>(u.as_int >> 23);

  // If there is a mantissa and the exp wont overflow round up
  if ((rounding == hipRoundPosInf) && (u.as_int & hip_detail::float_sig_mask) &&
      (!((u.as_int & ~hip_detail::float_sign_mask) < hip_detail::float_half_sig_bit)) &&
      (e8m0 < hip_detail::e8m0_NaN)) {
    ++e8m0;
  }

  // If e8m0 is NaN and exponent is a large non-inf value round down to a value
  if ((saturate == __HIP_SATFINITE) && (e8m0 == hip_detail::e8m0_NaN) &&
      ((u.as_int & ~hip_detail::float_sign_mask) <= hip_detail::float_max_exp)) {
    --e8m0;
  }
  return e8m0;
}

__FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t
__hip_cvt_bfloat16raw_to_e8m0(const __hip_bfloat16_raw hr, const __hip_saturation_t saturate,
                              const enum hipRoundMode rounding) {
  // Shifts out mantissa bits from bf16 dtype
  __hip_fp8_storage_t e8m0 = static_cast<unsigned char>(hr.x >> 7);

  // If there is a mantissa and the exp wont overflow round up
  if ((rounding == hipRoundPosInf) && (hr.x & hip_detail::bf16_sig_mask) &&
      (!((hr.x & ~hip_detail::bf16_sign_mask) < hip_detail::bf16_half_sig_bit)) &&
      (e8m0 < hip_detail::e8m0_NaN)) {
    ++e8m0;
  }

  // If e8m0 is NaN and exponent is a large non-inf value round down to a value
  if ((saturate == __HIP_SATFINITE) && (e8m0 == hip_detail::e8m0_NaN) &&
      ((hr.x & ~hip_detail::bf16_sign_mask) <= hip_detail::bf16_max_exp)) {
    --e8m0;
  }
  return e8m0;
}

__FP8_HOST_DEVICE_STATIC__ __hip_bfloat16_raw
__hip_cvt_e8m0_to_bf16raw(const __hip_fp8_storage_t x) {
  switch (x) {
    case 0x00U:
      return __hip_bfloat16_raw{hip_detail::bf16_half_sig_bit};
    case hip_detail::e8m0_NaN:
      return __hip_bfloat16_raw{hip_detail::bf16_NaN};
    default:
      return __hip_bfloat16_raw{static_cast<unsigned short>(x << 7)};
  }
}

struct __hip_fp8_e8m0 {
  __hip_fp8_storage_t __x;  //! raw storage of fp8 number
  constexpr static __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  constexpr static hipRoundMode __default_interpret = hipRoundPosInf;

  __hip_fp8_e8m0() = default;
  __FP8_HOST_DEVICE__ inline explicit __hip_fp8_e8m0(const __half f) {
    __x = __hip_cvt_float_to_e8m0(__half2float(f), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__ inline explicit __hip_fp8_e8m0(const __hip_bfloat16 f) {
    __x = __hip_cvt_bfloat16raw_to_e8m0(f, __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__ inline explicit __hip_fp8_e8m0(const double f) {
    __x = __hip_cvt_double_to_e8m0(f, __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__ inline explicit __hip_fp8_e8m0(const float f) {
    __x = __hip_cvt_float_to_e8m0(f, __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__ inline explicit __hip_fp8_e8m0(const int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__ inline explicit __hip_fp8_e8m0(const long int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__
  inline explicit __hip_fp8_e8m0(const long long int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__
  inline explicit __hip_fp8_e8m0(const short int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__
  inline explicit __hip_fp8_e8m0(const unsigned int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__
  inline explicit __hip_fp8_e8m0(const unsigned long int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__
  inline explicit __hip_fp8_e8m0(const unsigned long long int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }
  __FP8_HOST_DEVICE__
  inline explicit __hip_fp8_e8m0(const unsigned short int val) {
    __x =
        __hip_cvt_float_to_e8m0(static_cast<float>(val), __default_saturation, __default_interpret);
  }

  __FP8_HOST_DEVICE__ inline explicit operator __half() const {
    return __float2half_rn(static_cast<float>(*this));
  }
  __FP8_HOST_DEVICE__ inline explicit operator __hip_bfloat16() const {
    return static_cast<__hip_bfloat16>(__hip_cvt_e8m0_to_bf16raw(__x));
  }

  /* fp8_e8m0 can't represent a zero value, so always return true.*/
  __FP8_HOST_DEVICE__ inline explicit operator bool() const { return true; }
  __FP8_HOST_DEVICE__ inline explicit operator char() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<char>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator double() const {
    return hip_detail::__internal_cvt_e8m0_to_double(__x);
  }

  __FP8_HOST_DEVICE__ inline explicit operator float() const {
    return hip_detail::__internal_cvt_e8m0_to_float(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator long int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<long int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator long long int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<long long int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator short int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<short int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator signed char() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<signed char>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator unsigned char() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<unsigned char>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator unsigned int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<unsigned int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator unsigned long int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<unsigned long int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator unsigned long long int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<unsigned long long int>(__x);
  }
  __FP8_HOST_DEVICE__ inline explicit operator unsigned short int() const {
    return hip_detail::internal_cvt_e8m0_to_int_type<unsigned short int>(__x);
  }
};

/**
 * \brief struct representing single fp8 number with e4m3 interpretation
 *
 */

#if !defined(ENABLE_FNUZ_HIPRTC) || ENABLE_FNUZ_HIPRTC
struct __hip_fp8_e4m3_fnuz {
  __hip_fp8_storage_t __x;  //! raw storage of fp8 number
  constexpr static __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  constexpr static __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3_FNUZ;
  constexpr static unsigned int __we = 4;
  constexpr static unsigned int __wm = 3;

  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e4m3 from long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from short int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const short int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned short */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const double f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const float f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from __half */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e4m3 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8_e4m3_fnuz() = default;
#endif

  /*! convert fp8 e4m3 to __half */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e4m3 to __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0);
  }

  /*! convert fp8 e4m3 to char, clamp number to __HIP_CHAR_MIN/__HIP_CHAR_MAX if its out of range */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    auto fval = internal::cast_from_f8<float, true>(__x, __wm, __we);
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e4m3 to double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, true>(__x, __wm, __we);
  }

  /*! convert fp8 e4m3 to float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, true>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e4m3 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e4m3 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e4m3 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e4m3 to short int, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e4m3 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e4m3 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e4m3 to long long int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e4m3 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_fnuz_is_nan(__x)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two fp8 numbers with e4m3 interpretation
 *
 */
struct __hip_fp8x2_e4m3_fnuz {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3_FNUZ;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x2 e4m3 type from double2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! Default construct of fp8x2 e4m3 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e4m3_fnuz() = default;
#endif

  /*! convert fp8x2 e4m3 to __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e4m3 to float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x & 0xFF),
                                                      __wm, __we),
                  internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x >> 8),
                                                      __wm, __we));
#endif
}
}
;

/**
 * \brief struct representing four fp8 numbers with e4m3 interpretation
 *
 */
struct __hip_fp8x4_e4m3_fnuz {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3_FNUZ;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x4 e4m3 type from double4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const double4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const float4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e4m3 type from two __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /*! Default construct fp8x4 e4m3 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e4m3_fnuz() = default;
#endif

  /*! convert fp8x4 e4m3 to float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we),
                         internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8), __wm, __we));
    float2 low = float2(internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we),
                        internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm, __we));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;

/**
 * \brief struct representing one fp8 number with e5m2 interpretation
 *
 */
struct __hip_fp8_e5m2_fnuz {
  __hip_fp8_storage_t __x;  //! raw storage of one fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2_FNUZ;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;


  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e5m2 type from long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from short int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned long */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned int */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned short */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const double f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const float f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e5m2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8_e5m2_fnuz() = default;
#endif

  /*! convert fp8 e5m2 to float */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, true>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e5m2 to __half */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e5m2 to __hip_bfloat16 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0);
  }

  /*! convert fp8 e5m2 to char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e5m2 to double */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, true>(__x, __wm, __we);
  }

  /*! convert fp8 e5m2 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e5m2 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e5m2 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e5m2 to short, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e5m2 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_fnuz_is_nan(__x)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_fnuz_is_nan(__x)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x2_e5m2_fnuz {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2_FNUZ;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x2 e5m2 type from double2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! default construct fp8x2 e5m2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e5m2_fnuz() = default;
#endif

  /*! convert fp8x2 e5m2 to __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e5m2 to float2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x & 0xFF),
                                                      __wm, __we),
                  internal::cast_from_f8<float, true>(static_cast<__hip_fp8_storage_t>(__x >> 8),
                                                      __wm, __we));
#endif
}
}
;

/**
 * \brief struct representing four fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x4_e5m2_fnuz {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2_FNUZ;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x4 e5m2 type from double4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const double4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const float4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e5m2 type from two __half2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /* default construct fp8x4 e5m2 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2_fnuz() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e5m2_fnuz() = default;
#endif

  /*! convert fp8x4 e5m2 to float4 */
#if HIP_FP8_TYPE_FNUZ
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we),
                         internal::cast_from_f8<float, true>(
                             static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8), __wm, __we));
    float2 low = float2(internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we),
                        internal::cast_from_f8<float, true>(
                            static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm, __we));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;

#endif  // ENABLE_FNUZ_HIPRTC

/**
 * \brief struct representing ocp fp8 numbers with e4m3 interpretation
 *
 * */

#if !defined(ENABLE_OCP_HIPRTC) || ENABLE_OCP_HIPRTC

struct __hip_fp8_e4m3 {
  __hip_fp8_storage_t __x;  //! raw storage of fp8 number
  constexpr static __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  constexpr static __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3;
  constexpr static unsigned int __we = 4;
  constexpr static unsigned int __wm = 3;

  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e4m3 from long */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from short int */
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const short int val)
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {}

  /*! create fp8 e4m3 from unsigned long */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from unsigned short */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const double f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const float f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e4m3 from __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e4m3 from __half */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e4m3(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e4m3 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e4m3() = default;
#else
  __FP8_HOST__ __hip_fp8_e4m3() = default;
#endif

  /*! convert fp8 e4m3 to __half */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e4m3 to __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0 || static_cast<unsigned short>(__x) == 0x80);
  }

  /*! convert fp8 e4m3 to char, clamp number to __HIP_CHAR_MIN/__HIP_CHAR_MAX if its out of range */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    auto fval = internal::cast_from_f8<float, false>(__x, __wm, __we);
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e4m3 to double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, false>(__x, __wm, __we);
  }

  /*! convert fp8 e4m3 to float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, false>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e4m3 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e4m3 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e4m3 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e4m3 to short int, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e4m3 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e4m3 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e4m3 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e4m3 to long long int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e4m3 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two ocp fp8 numbers with e4m3 interpretation
 *
 * */
struct __hip_fp8x2_e4m3 {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x2 e4m3 type from double2 */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e4m3 type from __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e4m3(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! Default construct of fp8x2 e4m3 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e4m3() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e4m3() = default;
#endif

  /*! convert fp8x2 e4m3 to __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e4m3 to float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x & 0xFF),
                                                       __wm, __we),
                  internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x >> 8),
                                                       __wm, __we));
#endif
}
}
;

/**
 * \brief struct representing four ocp fp8 numbers with e4m3 interpretation
 *
 * */
 struct __hip_fp8x4_e4m3 {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E4M3;
  static constexpr unsigned int __we = 4;
  static constexpr unsigned int __wm = 3;

  /*! create fp8x4 e4m3 type from double4 */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const double4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const float4 val)
#endif
      : __x{reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))} {
  }

  /*! create fp8x4 e4m3 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e4m3 type from two __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e4m3(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /*! Default construct fp8x4 e4m3 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e4m3() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e4m3() = default;
#endif

  /*! convert fp8x4 e4m3 to float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(internal::cast_from_f8<float, false>(
                             static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we),
                         internal::cast_from_f8<float, false>(
                             static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8), __wm, __we));
    float2 low = float2(internal::cast_from_f8<float, false>(
                            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we),
                        internal::cast_from_f8<float, false>(
                            static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm, __we));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;

/**
 * \brief struct representing  ocp fp8 numbers with e5m2 interpretation
 *
 * */
struct __hip_fp8_e5m2 {
  __hip_fp8_storage_t __x;  //! raw storage of one fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;


  // TODO: SWDEV-452411
  // Add cast from unsigned long long, long long to fp8

  /*! create fp8 e5m2 type from long */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from short int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned long */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const unsigned long int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const unsigned long int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned int */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const unsigned int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const unsigned int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from unsigned short */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const unsigned short int val)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const unsigned short int val)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(val), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const double f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const double f)
#endif
      : __x(__hip_cvt_double_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const float f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const float f)
#endif
      : __x(__hip_cvt_float_to_fp8(f, __default_saturation, __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const __hip_bfloat16 f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const __hip_bfloat16 f)
#endif
      : __x(__hip_cvt_float_to_fp8(static_cast<float>(f), __default_saturation,
                                   __default_interpret)) {
  }

  /*! create fp8 e5m2 type from __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2(const __half f)
#else
  __FP8_HOST__ __hip_fp8_e5m2(const __half f)
#endif
      : __x(__hip_cvt_halfraw_to_fp8(static_cast<__half_raw>(f), __default_saturation,
                                     __default_interpret)) {
  }

  /*! default construct fp8 e5m2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8_e5m2() = default;
#else
  __FP8_HOST__ __hip_fp8_e5m2() = default;
#endif

  /*! convert fp8 e5m2 to float */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float() const {
#else
  __FP8_HOST__ operator float() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
    return internal::cast_to_f32_from_f8(__x, __default_interpret);
#else
    return internal::cast_from_f8<float, false>(__x, __wm, __we);
#endif
  }

  /*! convert fp8 e5m2 to __half */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half() const {
#else
  __FP8_HOST__ operator __half() const {
#endif
    return __half(__hip_cvt_fp8_to_halfraw(__x, __default_interpret));
  }

  /*! convert fp8 e5m2 to __hip_bfloat16 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __hip_bfloat16() const {
#else
  __FP8_HOST__ operator __hip_bfloat16() const {
#endif
    float f = *this;
    return __hip_bfloat16(f);
  }

  /*! convert fp8 e4m3 to bool, return false if value is 0, true otherwise */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator bool() const {
#else
  __FP8_HOST__ operator bool() const {
#endif
    // it can be 0x00 (+0.0) since 0x80 will be nan
    return !(static_cast<unsigned short>(__x) == 0 || static_cast<unsigned short>(__x) == 0x80);
  }

  /*! convert fp8 e5m2 to char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator char() const {
#else
  __FP8_HOST__ operator char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_CHAR_MIN) {
      return __HIP_CHAR_MIN;
    } else if (llval >= __HIP_CHAR_MAX) {
      return __HIP_CHAR_MAX;
    }
    return static_cast<char>(fval);
  }

  /*! convert fp8 e5m2 to double */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator double() const {
#else
  __FP8_HOST__ operator double() const {
#endif
    return internal::cast_from_f8<double, false>(__x, __wm, __we,
                                                 __default_saturation == __HIP_SATFINITE);
  }

  /*! convert fp8 e5m2 to int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator int() const {
#else
  __FP8_HOST__ operator int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<int>(fval);
  }

  /*! convert fp8 e5m2 to long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long int() const {
#else
  __FP8_HOST__ operator long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long>(fval);
  }

  /*! convert fp8 e5m2 to long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator long long int() const {
#else
  __FP8_HOST__ operator long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    return static_cast<long long>(fval);
  }

  /*! convert fp8 e5m2 to short, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator short int() const {
#else
  __FP8_HOST__ operator short int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SHRT_MIN) {
      return __HIP_SHRT_MIN;
    } else if (llval >= __HIP_SHRT_MAX) {
      return __HIP_SHRT_MAX;
    }
    return static_cast<short>(fval);
  }

  /*! convert fp8 e5m2 to signed char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator signed char() const {
#else
  __FP8_HOST__ operator signed char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= __HIP_SCHAR_MIN) {
      return __HIP_SCHAR_MIN;
    } else if (llval >= __HIP_SCHAR_MAX) {
      return __HIP_SCHAR_MAX;
    }
    return static_cast<signed char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned char, clamp out of bound values, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned char() const {
#else
  __FP8_HOST__ operator unsigned char() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    } else if (llval >= __HIP_UCHAR_MAX) {
      return __HIP_UCHAR_MAX;
    }
    return static_cast<unsigned char>(fval);
  }

  /*! convert fp8 e5m2 to unsigned int, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned int() const {
#else
  __FP8_HOST__ operator unsigned int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned int>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long int() const {
#else
  __FP8_HOST__ operator unsigned long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned long long, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned long long int() const {
#else
  __FP8_HOST__ operator unsigned long long int() const {
#endif
    if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
      return 0;
    }

    float fval = *this;
    auto llval = static_cast<long long>(fval);
    if (llval <= 0) {
      return 0;
    }
    return static_cast<unsigned long long>(fval);
  }

  /*! convert fp8 e5m2 to unsigned short, return 0 if value is NaN */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator unsigned short int() const {
#else
  __FP8_HOST__ operator unsigned short int() const {
#endif
      if (internal::hip_fp8_ocp_is_nan(__x, __default_interpret)) {
        return 0;
}

float fval = *this;
auto llval = static_cast<long long>(fval);
if (llval <= 0) {
  return 0;
}
return static_cast<unsigned short>(fval);
}
}
;

/**
 * \brief struct representing two ocp fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x2_e5m2 {
  __hip_fp8x2_storage_t __x;  //! raw storage of two fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x2 e5m2 type from double2 */

#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const double2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const double2 val)
#endif
      : __x(__hip_cvt_double2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const float2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const float2 val)
#endif
      : __x(__hip_cvt_float2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const __hip_bfloat162 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const __hip_bfloat162 val)
#endif
      : __x(__hip_cvt_bfloat16raw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! create fp8x2 e5m2 type from __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2(const __half2 val)
#else
  __FP8_HOST__ __hip_fp8x2_e5m2(const __half2 val)
#endif
      : __x(__hip_cvt_halfraw2_to_fp8x2(val, __default_saturation, __default_interpret)) {
  }

  /*! default construct fp8x2 e5m2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x2_e5m2() = default;
#else
  __FP8_HOST__ __hip_fp8x2_e5m2() = default;
#endif

  /*! convert fp8x2 e5m2 to __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator __half2() const {
#else
  __FP8_HOST__ operator __half2() const {
#endif
    return __half2(__hip_cvt_fp8x2_to_halfraw2(__x, __default_interpret));
  }

  /*! convert fp8x2 e5m2 to float2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float2() const {
#else
  __FP8_HOST__ operator float2() const {
#endif
#if HIP_FP8_CVT_FAST_PATH
      return internal::cast_to_f32x2_from_f8x2(__x, __default_interpret);
#else
    return float2(
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x & 0xFF), __wm,
                                             __we, __default_saturation == __HIP_SATFINITE),
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(__x >> 8), __wm, __we,
                                             __default_saturation == __HIP_SATFINITE));
#endif
}
}
;

/**
 * \brief struct representing four ocp fp8 numbers with e5m2 interpretation
 *
 */
struct __hip_fp8x4_e5m2 {
  __hip_fp8x4_storage_t __x;  //! raw storage of four fp8 numbers
  static constexpr __hip_saturation_t __default_saturation = __HIP_SATFINITE;
  static constexpr __hip_fp8_interpretation_t __default_interpret = __HIP_E5M2;
  static constexpr unsigned int __we = 5;
  static constexpr unsigned int __wm = 2;

  /*! create fp8x4 e5m2 type from double4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const double4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const double4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_double_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const float4 val)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const float4 val)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.x, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.y, __default_saturation, __default_interpret))
                                          << 8 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.z, __default_saturation, __default_interpret))
                                          << 16 |
                                      reinterpret_cast<unsigned char>(__hip_cvt_float_to_fp8(
                                          val.w, __default_saturation, __default_interpret))
                                          << 24))) {
  }

  /*! create fp8x4 e5m2 type from two __hip_bfloat162 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const __hip_bfloat162 low, const __hip_bfloat162 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const __hip_bfloat162 low, const __hip_bfloat162 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(static_cast<unsigned int>(
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(high, __default_saturation, __default_interpret)) |
            reinterpret_cast<unsigned short>(
                __hip_cvt_bfloat16raw2_to_fp8x2(low, __default_saturation, __default_interpret))
                << 16))) {
  }

  /*! create fp8x4 e5m2 type from two __half2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2(const __half2 low, const __half2 high)
#else
  __FP8_HOST__ __hip_fp8x4_e5m2(const __half2 low, const __half2 high)
#endif
      : __x(reinterpret_cast<__hip_fp8x4_storage_t>(
            static_cast<unsigned int>(reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          high, __default_saturation, __default_interpret)) |
                                      reinterpret_cast<unsigned short>(__hip_cvt_halfraw2_to_fp8x2(
                                          low, __default_saturation, __default_interpret))
                                          << 16))) {
  }

  /* default construct fp8x4 e5m2 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ __hip_fp8x4_e5m2() = default;
#else
  __FP8_HOST__ __hip_fp8x4_e5m2() = default;
#endif

  /*! convert fp8x4 e5m2 to float4 */
#if HIP_FP8_TYPE_OCP
  __FP8_HOST_DEVICE__ operator float4() const {
#else
  __FP8_HOST__ operator float4() const {
#endif
      auto x = __x;                                                // bypass const
  auto fp8x2_low = *reinterpret_cast<__hip_fp8x2_storage_t*>(&x);  // Little E
  auto fp8x2_high = *(reinterpret_cast<__hip_fp8x2_storage_t*>(&x) + 1);
#if HIP_FP8_CVT_FAST_PATH
  float2 high = internal::cast_to_f32x2_from_f8x2(fp8x2_high, __default_interpret);
  float2 low = internal::cast_to_f32x2_from_f8x2(fp8x2_low, __default_interpret);
#else
    float2 high = float2(
        internal::cast_from_f8<float, false>(
            static_cast<__hip_fp8_storage_t>((fp8x2_high << 8) >> 8), __wm, __we,
            __default_saturation == __HIP_SATFINITE),
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(fp8x2_high >> 8),
                                             __wm, __we, __default_saturation == __HIP_SATFINITE));
    float2 low = float2(
        internal::cast_from_f8<float, false>(
            static_cast<__hip_fp8_storage_t>((fp8x2_low << 8) >> 8), __wm, __we,
            __default_saturation == __HIP_SATFINITE),
        internal::cast_from_f8<float, false>(static_cast<__hip_fp8_storage_t>(fp8x2_low >> 8), __wm,
                                             __we, __default_saturation == __HIP_SATFINITE));
#endif
  return float4(low.x, low.y, high.x, high.y);
}
}
;
#endif  // ENABLE_OCP_HIPRTC
#endif  // _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP8_H_
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if !defined(__HIPCC_RTC__)
#include "amd_hip_ocp_types.h"
#include <cstdint>
#include <limits>
#include <cstdlib>
#include <cmath>
#endif

namespace fcbx {
constexpr __hip_int8_t OCP_SCALE_EXP_NAN = -128;

enum class Encoding : size_t {
  E2M1 = 0,
  E2M3,
  E3M2,
  E4M3,
  E4M3Mx,
  E4M3Nanoo,
  E5M2,
  E5M2Mx,
  E5M2Nanoo,

  E5M10,  // FP16
  E8M7,   // BF16

  IEEE754,

  // Keep this one last
  NumEncodings,
};
enum fp16 : __hip_uint16_t {};
enum bf16 : __hip_uint16_t {};

// Map a source storage type to its Encoding as a true compile-time constant.
// Using a traits struct (rather than an IIFE assigned to `const auto`) guarantees
// the value is usable as a non-type template argument across all supported
// toolchains, including the comgr-bundled Clang used by HIPRTC on Windows where
// implicit-constexpr lambda rules differ from the Linux host toolchain.
template <typename T> struct EncodingOf;
template <> struct EncodingOf<float> {
  static constexpr Encoding value = Encoding::IEEE754;
};
template <> struct EncodingOf<__amd_fp16_storage_t> {
  static constexpr Encoding value = Encoding::E5M10;
};
template <> struct EncodingOf<__amd_bf16_storage_t> {
  static constexpr Encoding value = Encoding::E8M7;
};

struct Float {
  __hip_int32_t ExpBias;
  __hip_uint32_t ExpBits;
  __hip_uint32_t ExpMask;
  __hip_uint32_t ManBits;
  __hip_uint32_t ManMask;
  __hip_int32_t MaxExp;
  __hip_int32_t MinExp;
  bool MxScale;
  bool HasNaN;
  bool HasInf;
};

static const float ieee754_nan = __hip_internal::NumericLimits<float>::quiet_NaN();
static const float ieee754_inf = __hip_internal::NumericLimits<float>::infinity();

__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t U32(float f) {
  static_assert(sizeof(__hip_uint32_t) == sizeof(float), "");
  union {
    float f32;
    __hip_uint32_t ui32;
  } u{f};
  return u.ui32;
}

__OCP_FP_HOST_DEVICE_STATIC__ float F32(__hip_uint32_t u32) {
  static_assert(sizeof(__hip_uint32_t) == sizeof(float), "");
  union {
    __hip_uint32_t ui32;
    float f32;
  } u{u32};
  return u.f32;
}

constexpr __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t bitmask(__hip_uint32_t bits) {
  if (bits < 1) return 0;
  return ((__hip_uint32_t)1 << bits) - 1;
}

struct EncodingTable {
  Float __elements[(size_t)Encoding::NumEncodings];
#if __cplusplus >= 201402L
  constexpr Float& operator[](size_t i) { return __elements[i]; }
#else
  Float& operator[](size_t i) { return __elements[i]; }
#endif
  constexpr const Float& operator[](size_t i) const { return __elements[i]; }
};

constexpr EncodingTable init() {
  EncodingTable a{};

  a[(size_t)Encoding::E2M1] = {
      .ExpBias = 1,
      .ExpBits = 2,
      .ExpMask = bitmask(2),
      .ManBits = 1,
      .ManMask = bitmask(1),
      .MaxExp = 2,
      .MinExp = 0,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E2M3] = {
      .ExpBias = 1,
      .ExpBits = 2,
      .ExpMask = bitmask(2),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 2,
      .MinExp = 0,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E3M2] = {
      .ExpBias = 3,
      .ExpBits = 3,
      .ExpMask = bitmask(3),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 4,
      .MinExp = -2,
      .MxScale = true,
      .HasNaN = false,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3] = {
      .ExpBias = 7,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 8,
      .MinExp = -6,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3Mx] = {
      .ExpBias = 7,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 8,
      .MinExp = -6,
      .MxScale = true,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E4M3Nanoo] = {
      .ExpBias = 8,
      .ExpBits = 4,
      .ExpMask = bitmask(4),
      .ManBits = 3,
      .ManMask = bitmask(3),
      .MaxExp = 7,
      .MinExp = -7,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = false,
  };

  a[(size_t)Encoding::E5M2] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M2Mx] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = true,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M2Nanoo] = {
      .ExpBias = 16,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 2,
      .ManMask = bitmask(2),
      .MaxExp = 15,
      .MinExp = -15,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E5M10] = {
      .ExpBias = 15,
      .ExpBits = 5,
      .ExpMask = bitmask(5),
      .ManBits = 10,
      .ManMask = bitmask(10),
      .MaxExp = 15,
      .MinExp = -14,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::E8M7] = {
      .ExpBias = 127,
      .ExpBits = 8,
      .ExpMask = bitmask(8),
      .ManBits = 7,
      .ManMask = bitmask(7),
      .MaxExp = 127,
      .MinExp = -126,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  a[(size_t)Encoding::IEEE754] = {
      .ExpBias = 127,
      .ExpBits = 8,
      .ExpMask = bitmask(8),
      .ManBits = 23,
      .ManMask = bitmask(23),
      .MaxExp = 127,
      .MinExp = -126,
      .MxScale = false,
      .HasNaN = true,
      .HasInf = true,
  };

  return a;
}

static constexpr auto encodings = init();

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t exponentbits(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  return (val >> enc.ManBits) & enc.ExpMask;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t mantissa(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  return val & enc.ManMask;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool issubnorm(__hip_uint32_t val) {
  switch (E) {
    default:
      return exponentbits<E, sat>(val) == 0 && mantissa<E, sat>(val) != 0;
  }

  __builtin_trap();
  // Unreachable
  return false;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_int32_t exponent(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  auto unbiased_exp = exponentbits<E, sat>(val);
  unbiased_exp = issubnorm<E, sat>(val) ? 1 : unbiased_exp;
  return (__hip_int32_t)unbiased_exp - enc.ExpBias;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t signbit(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  return (val >> (enc.ExpBits + enc.ManBits)) & 1;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t nan(__hip_uint32_t sign) {
  const auto& enc = encodings[(size_t)E];

  switch (E) {
    case Encoding::E2M1:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0b0111;
    case Encoding::E2M3:
    case Encoding::E3M2:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0b011111;
    case Encoding::E4M3:
    case Encoding::E4M3Mx:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0x7f;
    case Encoding::E5M2:
    case Encoding::E5M2Mx:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0x7e;
    case Encoding::E4M3Nanoo:
    case Encoding::E5M2Nanoo:
      return 0b10000000;
    case Encoding::E5M10:
    case Encoding::E8M7:
      return (sign << (enc.ExpBits + enc.ManBits)) | (enc.ExpMask << enc.ManBits) | enc.ManMask;
    case Encoding::IEEE754:
      return U32(sign ? __hip_internal::copysign(ieee754_nan, -1.0F) : ieee754_nan);
    default:
      __builtin_trap();
      return 0;
  }
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t zero(__hip_uint32_t sign) {
  const auto& enc = encodings[(size_t)E];

  switch (E) {
    case Encoding::E2M1:
    case Encoding::E2M3:
    case Encoding::E3M2:
    case Encoding::E4M3:
    case Encoding::E4M3Mx:
    case Encoding::E5M2:
    case Encoding::E5M2Mx:
    case Encoding::E5M10:
    case Encoding::E8M7:
      return (sign << (enc.ExpBits + enc.ManBits)) | 0;
    case Encoding::E4M3Nanoo:
    case Encoding::E5M2Nanoo:
      return 0;
    case Encoding::IEEE754:
      return U32(sign ? __hip_internal::copysign(0.0F, -1.0F) : 0.0F);
    default:
      __builtin_trap();
      return 0;
  }
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t inf(__hip_uint32_t sign) {
  const auto& enc = encodings[(size_t)E];

  switch (E) {
    case Encoding::E2M1:
    case Encoding::E2M3:
    case Encoding::E3M2:
      return nan<E, sat>(sign);
    case Encoding::E4M3:
    case Encoding::E4M3Mx:
    case Encoding::E4M3Nanoo:
    case Encoding::E5M2Nanoo:
      if constexpr (sat) {
        sign <<= enc.ExpBits + enc.ManBits;
        return sign | 0b01111111;
      }

      return nan<E, sat>(sign);
    case Encoding::E5M2:
    case Encoding::E5M2Mx:
      sign <<= enc.ExpBits + enc.ManBits;
      if constexpr (sat) {
        return sign | 0b01111011;
      }

      return sign | 0b01111100;
    case Encoding::E5M10:
    case Encoding::E8M7:
      sign <<= enc.ExpBits + enc.ManBits;
      return sign | (enc.ExpMask << enc.ManBits) | 0;
    case Encoding::IEEE754:
      return U32(sign ? __hip_internal::copysign(ieee754_inf, -1.0F) : ieee754_inf);
    default:
      __builtin_trap();
      return 0;
  }
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool isnan(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  if (!enc.HasNaN) return false;

  if constexpr (E == Encoding::E4M3Mx || E == Encoding::E4M3 || E == Encoding::E4M3Nanoo ||
                E == Encoding::E5M2Nanoo)
    return nan<E, sat>(signbit<E, sat>(val)) == val;

  return exponentbits<E, sat>(val) == enc.ExpMask && mantissa<E, sat>(val) != 0;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool isinf(__hip_uint32_t val) {
  const auto& enc = encodings[(size_t)E];
  if (!enc.HasInf) return false;

  if constexpr (E == Encoding::E5M10 || E == Encoding::E8M7) {
    return exponentbits<E, sat>(val) == enc.ExpMask && mantissa<E, sat>(val) == 0;
  }

  return inf<E, sat>(signbit<E, sat>(val)) == val;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool iszero(__hip_uint32_t val) {
  return zero<E, sat>(signbit<E, sat>(val)) == val;
}

template <Encoding E, bool sat> __OCP_FP_HOST_DEVICE_STATIC__ bool inrange(__hip_uint32_t val) {
  return !(isnan<E, sat>(val) || isinf<E, sat>(val));
}

template <typename T> __OCP_FP_HOST_DEVICE_STATIC__ T makenan(Encoding E, __hip_uint32_t sign) {
  switch (E) {
    case Encoding::E5M10:
      return (T)nan<Encoding::E5M10, false>(sign);
    case Encoding::E8M7:
      return (T)nan<Encoding::E8M7, false>(sign);
    case Encoding::IEEE754:
      return (T)F32(nan<Encoding::IEEE754, false>(sign));
    default:
      __builtin_trap();
      // Unreachable
      return T();
  }
}

template <typename T> __OCP_FP_HOST_DEVICE_STATIC__ T makeinf(Encoding E, __hip_uint32_t sign) {
  switch (E) {
    case Encoding::E5M10:
      return (T)inf<Encoding::E5M10, false>(sign);
    case Encoding::E8M7:
      return (T)inf<Encoding::E8M7, false>(sign);
    case Encoding::IEEE754:
      return (T)F32(inf<Encoding::IEEE754, false>(sign));
    default:
      __builtin_trap();
      // Unreachable
      return T();
  }
}

template <typename T> __OCP_FP_HOST_DEVICE_STATIC__ T makezero(Encoding E, __hip_uint32_t sign) {
  switch (E) {
    case Encoding::E5M10:
      return (T)zero<Encoding::E5M10, false>(sign);
    case Encoding::E8M7:
      return (T)zero<Encoding::E8M7, false>(sign);
    case Encoding::IEEE754:
      return (T)F32(zero<Encoding::IEEE754, false>(sign));
    default:
      __builtin_trap();
      // Unreachable
      return T();
  }
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ T to_float(__hip_uint32_t u32, __hip_int8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 && E != Encoding::E8M7, "");

  const auto& enc = encodings[(size_t)E];
  static_assert(__hip_internal::is_same<T, float>() ||
                    __hip_internal::is_same<T, __amd_fp16_storage_t>() ||
                    __hip_internal::is_same<T, __amd_bf16_storage_t>(),
                "to_float: unsupported destination type T");
  constexpr Encoding dstE = EncodingOf<T>::value;
  const auto& dstEnc = encodings[(size_t)dstE];

  if (isnan<E, sat>(u32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return makenan<T>(dstE, signbit<E, sat>(u32));

  if (isinf<E, sat>(u32)) return makeinf<T>(dstE, signbit<E, sat>(u32));

  if (iszero<E, sat>(u32)) return makezero<T>(dstE, signbit<E, sat>(u32));

  auto dstMan = mantissa<E, sat>(u32) << (dstEnc.ManBits - enc.ManBits);
  auto dstExp = (__hip_uint32_t)(exponent<E, sat>(u32) + dstEnc.ExpBias);
  dstExp &= dstEnc.ExpMask;

  if (issubnorm<E, sat>(u32)) {
    auto leadbit = (__hip_uint32_t)1 << dstEnc.ManBits;
    while ((dstMan & leadbit) == 0) {
      dstMan <<= 1;
      dstExp -= 1;
    }

    dstMan &= dstEnc.ManMask;
  }

  auto sign = signbit<E, sat>(u32) << (dstEnc.ExpBits + dstEnc.ManBits);

  if (enc.MxScale) {
    __hip_int32_t exp = dstExp - dstEnc.ExpBias;
    __hip_int32_t tmp = exp + (__hip_int32_t)scale_exp;
    size_t diff = abs(tmp - dstEnc.MinExp);


    if (tmp < dstEnc.MinExp) {
      if (diff > dstEnc.ManBits + 1) return makezero<T>(dstE, signbit<E, sat>(u32));

      dstExp = 0;  // Subnormal
      dstMan |= (__hip_uint32_t)1 << dstEnc.ManBits;

      auto roundBitShift = diff - 1;
      auto roundBit = (dstMan & ((__hip_uint32_t)1 << roundBitShift)) != 0;
      auto stickyMask = ((__hip_uint32_t)1 << roundBitShift) - 1;
      auto stickyBits = dstMan & stickyMask;
      auto odd = (dstMan & ((__hip_uint32_t)1 << diff)) != 0;

      dstMan >>= diff;

      if ((roundBit && stickyBits != 0) || (roundBit && odd)) {
        ++dstMan;
        if ((dstMan & ((__hip_uint32_t)1 << dstEnc.ManBits)) != 0) ++dstExp;
      }

      dstMan &= dstEnc.ManMask;
    } else {
      dstExp = (__hip_uint32_t)(exp + scale_exp + dstEnc.ExpBias);

      // Overflow: return infinity (gfx950 HW behavior)
      if (dstExp >= dstEnc.ExpMask) return makeinf<T>(dstE, signbit<E, sat>(u32));

      dstExp &= dstEnc.ExpMask;
    }
  }

  auto dst = sign | (dstExp << dstEnc.ManBits) | dstMan;

  union {
    float f32;
    __amd_fp16_storage_t fp16[2];
    __amd_bf16_storage_t bf16[2];
    __hip_uint32_t u32;
  } u;
  u.u32 = dst;
  if constexpr (__hip_internal::is_same<T, float>())
    return u.f32;
  else if constexpr (__hip_internal::is_same<T, __amd_fp16_storage_t>())
    return u.fp16[0];
  else if constexpr (__hip_internal::is_same<T, __amd_bf16_storage_t>())
    return u.bf16[0];
  else
    __builtin_trap();
}

template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t from_float_sr(T f, __hip_uint32_t seed, __hip_int8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 && E != Encoding::E8M7, "");
  static_assert(sizeof(__amd_fp16_storage_t[2]) == sizeof(float), "");
  static_assert(sizeof(__amd_bf16_storage_t[2]) == sizeof(float), "");
  union {
    float f32;
    __amd_fp16_storage_t fp16[2];
    __amd_bf16_storage_t bf16[2];
    __hip_uint32_t u32;
  } u{0};

  if constexpr (__hip_internal::is_same<T, float>())
    u.f32 = f;
  else if constexpr (__hip_internal::is_same<T, __amd_fp16_storage_t>())
    u.fp16[0] = f;
  else if constexpr (__hip_internal::is_same<T, __amd_bf16_storage_t>())
    u.bf16[0] = f;
  else
    __builtin_trap();

  const auto& enc = encodings[(size_t)E];
  static_assert(__hip_internal::is_same<T, float>() ||
                    __hip_internal::is_same<T, __amd_fp16_storage_t>() ||
                    __hip_internal::is_same<T, __amd_bf16_storage_t>(),
                "from_float_sr: unsupported source type T");
  constexpr Encoding srcE = EncodingOf<T>::value;
  const auto& srcEnc = encodings[(size_t)srcE];

  auto srcU32 = u.u32;  // (srcE == Encoding::IEEE754) ? U32(f) : (__hip_uint32_t)f;
  auto signBit = signbit<srcE, false>(srcU32);
  auto sign = signBit << (enc.ExpBits + enc.ManBits);

  if (isnan<srcE, sat>(srcU32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return nan<E, sat>(signBit);

  if (isinf<srcE, sat>(srcU32)) return inf<E, sat>(signBit);

  if (iszero<srcE, sat>(srcU32)) return zero<E, sat>(signBit);

  auto srcMan = mantissa<srcE, false>(srcU32);
  auto srcExp = exponent<srcE, false>(srcU32);
  if (enc.MxScale) {
    if (issubnorm<srcE, false>(srcU32)) {
      auto leadbit = (__hip_uint32_t)1 << srcEnc.ManBits;
      while ((srcMan & leadbit) == 0) {
        srcMan <<= 1;
        srcExp -= 1;
      }

      srcMan &= srcEnc.ManMask;
    }

    srcExp -= scale_exp;
  }

  auto exp = srcExp;
  auto man = srcMan;
  bool subnorm = false;

  if (exp > enc.MaxExp) {
    return inf<E, sat>(signBit);
  } else if (exp >= enc.MinExp) {
    man = srcMan;
  } else if (exp < enc.MinExp) {
    subnorm = true;
    exp = 0;

    auto diff = (__hip_uint32_t)(enc.MinExp - srcExp);
    if (diff >= 32) {
      man = 0;
      srcMan = 0;
    } else {
      srcMan |= (__hip_uint32_t)1 << srcEnc.ManBits;
      srcMan >>= diff;
    }

    man = srcMan;
  }

  // Align random value to be one past the kept mant bit
  size_t sr_shift = (32 - srcEnc.ManBits) + enc.ManBits;

  // For stochastic-rounding we add the aligned random value to the
  // mantissa and then truncate (RTZ).
  man += seed >> sr_shift;

  // Increment exponent when mantissa overflows due to rounding
  if (man >= (__hip_uint32_t)1 << srcEnc.ManBits) ++exp;
  man >>= (srcEnc.ManBits - enc.ManBits);
  man &= enc.ManMask;

  if (exp > enc.MaxExp) return inf<E, sat>(signBit);

  auto biasedExp = (__hip_uint32_t)exp;
  if (!subnorm) biasedExp = (__hip_uint32_t)(exp + enc.ExpBias);
  biasedExp &= enc.ExpMask;

  auto val = sign | biasedExp << enc.ManBits | man;
  if (inrange<E, sat>(val))
    return val;
  else if (man == 0 && exp == 0)
    return zero<E, sat>(signBit);
  else
    return inf<E, sat>(signBit);
}


template <typename T, Encoding E, bool sat>
__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint32_t from_float(T f, __hip_int8_t scale_exp) {
  // We do not support bf16/fp16 <-> float
  static_assert(E != Encoding::IEEE754 && E != Encoding::E5M10 && E != Encoding::E8M7, "");
  static_assert(sizeof(__amd_fp16_storage_t[2]) == sizeof(float), "");
  static_assert(sizeof(__amd_bf16_storage_t[2]) == sizeof(float), "");
  union {
    float f32;
    __amd_fp16_storage_t fp16[2];
    __amd_bf16_storage_t bf16[2];
    __hip_uint32_t u32;
  } u{0};

  if constexpr (__hip_internal::is_same<T, float>())
    u.f32 = f;
  else if constexpr (__hip_internal::is_same<T, __amd_fp16_storage_t>())
    u.fp16[0] = f;
  else if constexpr (__hip_internal::is_same<T, __amd_bf16_storage_t>())
    u.bf16[0] = f;
  else
    __builtin_trap();

  const auto& enc = encodings[(size_t)E];
  static_assert(__hip_internal::is_same<T, float>() ||
                    __hip_internal::is_same<T, __amd_fp16_storage_t>() ||
                    __hip_internal::is_same<T, __amd_bf16_storage_t>(),
                "from_float: unsupported source type T");
  constexpr Encoding srcE = EncodingOf<T>::value;
  const auto& srcEnc = encodings[(size_t)srcE];

  auto srcU32 = u.u32;  // (srcE == Encoding::IEEE754) ? U32(f) : (__hip_uint32_t)f;
  auto signBit = signbit<srcE, false>(srcU32);
  auto sign = signBit << (enc.ExpBits + enc.ManBits);

  if (isnan<srcE, sat>(srcU32) || (enc.MxScale && scale_exp == OCP_SCALE_EXP_NAN))
    return nan<E, sat>(signBit);

  if (isinf<srcE, sat>(srcU32)) return inf<E, sat>(signBit);

  if (iszero<srcE, sat>(srcU32)) return zero<E, sat>(signBit);

  auto srcMan = mantissa<srcE, false>(srcU32);
  auto srcExp = exponent<srcE, false>(srcU32);
  if (enc.MxScale) {
    if (issubnorm<srcE, false>(srcU32)) {
      auto leadbit = (__hip_uint32_t)1 << srcEnc.ManBits;
      while ((srcMan & leadbit) == 0) {
        srcMan <<= 1;
        srcExp -= 1;
      }

      srcMan &= srcEnc.ManMask;
    }

    srcExp -= scale_exp;
  }

  auto exp = srcExp;
  auto man = srcMan;
  __hip_uint32_t stickyBits = 0;
  bool subnorm = false;

  if (exp > enc.MaxExp) {
    return inf<E, sat>(signBit);
  } else if (exp >= enc.MinExp) {
    man >>= srcEnc.ManBits - enc.ManBits;
  } else if (exp < enc.MinExp) {
    subnorm = true;
    exp = 0;

    auto diff = (__hip_uint32_t)(enc.MinExp - srcExp);
    if (diff >= 32) {
      man = 0;
      srcMan = 0;
    } else {
      srcMan |= (__hip_uint32_t)1 << srcEnc.ManBits;
      stickyBits = srcMan & (((__hip_uint32_t)1 << diff) - (__hip_uint32_t)1);
      srcMan >>= diff;

      man = srcMan;
      man >>= srcEnc.ManBits - enc.ManBits;
      man &= enc.ManMask;
    }
  }

  auto roundBitShift = srcEnc.ManBits - (enc.ManBits + 1);
  auto roundBit = ((srcMan >> roundBitShift) & 1) != 0;
  stickyBits |= srcMan & (((__hip_uint32_t)1 << roundBitShift) - 1);
  auto odd = (man & 1) != 0;

  if ((roundBit && stickyBits != 0) || (roundBit && odd)) {
    ++man;
    if ((man & ((__hip_uint32_t)1 << enc.ManBits)) != 0) ++exp;
    man &= enc.ManMask;
  }

  if (exp > enc.MaxExp) return inf<E, sat>(signBit);

  auto biasedExp = (__hip_uint32_t)exp;
  if (!subnorm) biasedExp = (__hip_uint32_t)(exp + enc.ExpBias);
  biasedExp &= enc.ExpMask;

  auto val = sign | biasedExp << enc.ManBits | man;
  if (inrange<E, sat>(val))
    return val;
  else if (man == 0 && exp == 0)
    return zero<E, sat>(signBit);
  else
    return inf<E, sat>(signBit);
}

// Manual little-endian, LSB-first 6-bit packing.
__OCP_FP_HOST_DEVICE_STATIC__ void pack_fp6(__hip_uint8_t* buf, int idx, __hip_uint8_t val) {
  const unsigned bit = static_cast<unsigned>(idx) * 6u;
  const unsigned byte = bit >> 3;
  const unsigned off = bit & 7u;
  val &= 0x3f;
  buf[byte] = static_cast<__hip_uint8_t>((buf[byte] & ~(0x3f << off)) | (val << off));
  if (off > 2u)
    buf[byte + 1] = static_cast<__hip_uint8_t>((buf[byte + 1] & ~(0x3f >> (8u - off))) |
                                               (val >> (8u - off)));
}

__OCP_FP_HOST_DEVICE_STATIC__ __hip_uint8_t unpack_fp6(const __hip_uint8_t* buf, int idx) {
  const unsigned bit = static_cast<unsigned>(idx) * 6u;
  const unsigned byte = bit >> 3;
  const unsigned off = bit & 7u;
  unsigned v = static_cast<unsigned>(buf[byte]) >> off;
  if (off > 2u) v |= static_cast<unsigned>(buf[byte + 1]) << (8u - off);
  return static_cast<__hip_uint8_t>(v & 0x3f);
}

// ------------
template <typename InType, typename OutType, typename float_base_t, Encoding in_encode,
          Encoding out_encode, bool sr = false>
__OCP_FP_HOST_DEVICE_STATIC__ OutType fp6_cvt_packedx16(InType in, __hip_int8_t scale = 0,
                                                        __hip_uint32_t seed = 0) {
  // This is tightly coupled with the definitions of the amd_ocp_types
  constexpr bool in_float = __hip_internal::is_same<InType, __amd_floatx16_storage_t>::value ||
      __hip_internal::is_same<InType, __amd_fp16x16_storage_t>::value ||
      __hip_internal::is_same<InType, __amd_bf16x16_storage_t>::value;
  using other_type = __hip_internal::conditional<in_float, OutType, InType>::type;

  union {
    other_type o;
    __hip_uint8_t bytes[sizeof(other_type)];
  } u;

  if constexpr (in_float) {
    for (int i = 0; i < static_cast<int>(sizeof(other_type)); ++i) {
       u.bytes[i] = 0;
    }
    for (int i = 0; i < 16; ++i) {
      __hip_uint8_t v;
      if constexpr (sr) {
        v = static_cast<__hip_uint8_t>(
            from_float_sr<float_base_t, out_encode, true>(in[i], seed, scale));
      } else {
        v = static_cast<__hip_uint8_t>(from_float<float_base_t, out_encode, true>(in[i], scale));
      }
      pack_fp6(u.bytes, i, v);
    }
    return u.o;
  } else {
    OutType ret;
    u.o = in;
    for (int i = 0; i < 16; ++i) {
      ret[i] = to_float<float_base_t, in_encode, true>(unpack_fp6(u.bytes, i), scale);
    }
    return ret;
  }
}
// ------------

template <typename InType, typename OutType, typename float_base_t, Encoding in_encode,
          Encoding out_encode, bool sr = false>
__OCP_FP_HOST_DEVICE_STATIC__ OutType fp6_cvt_packedx32(InType in, __hip_int8_t scale = 0,
                                                        __hip_uint32_t seed = 0) {
  // This is tightly coupled with the definitions of the amd_ocp_types
  constexpr bool in_float = __hip_internal::is_same<InType, __amd_floatx32_storage_t>::value ||
                            __hip_internal::is_same<InType, __amd_fp16x32_storage_t>::value ||
                            __hip_internal::is_same<InType, __amd_bf16x32_storage_t>::value;
  using other_type = __hip_internal::conditional<in_float, OutType, InType>::type;

  union {
    other_type o;
    __hip_uint8_t bytes[sizeof(other_type)];
  } u;

  if constexpr (in_float) {
    for (int i = 0; i < static_cast<int>(sizeof(other_type)); ++i) {
      u.bytes[i] = 0;
    }
    for (int i = 0; i < 32; ++i) {
      __hip_uint8_t v;
      if constexpr (sr) {
        v = static_cast<__hip_uint8_t>(
            from_float_sr<float_base_t, out_encode, true>(in[i], seed, scale));
      } else {
        v = static_cast<__hip_uint8_t>(from_float<float_base_t, out_encode, true>(in[i], scale));
      }
      pack_fp6(u.bytes, i, v);
    }
    return u.o;
  } else {
    OutType ret;
    u.o = in;
    for (int i = 0; i < 32; ++i) {
      ret[i] = to_float<float_base_t, in_encode, true>(unpack_fp6(u.bytes, i), scale);
    }
    return ret;
  }
}
}  // namespace fcbx
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP6_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP6_H_

#if defined(__HIPCC_RTC__)
#define __FP6_HOST_DEVICE__ __device__
#define __FP6_HOST_DEVICE_STATIC__ __FP6_HOST_DEVICE__ static
#else
#define __FP6_HOST_DEVICE__ __host__ __device__
#define __FP6_HOST_DEVICE_STATIC__ __FP6_HOST_DEVICE__ static inline
#include "amd_hip_mx_common.h"
#include "amd_hip_fp8.h"

#include "amd_hip_ocp_host.hpp"
#endif  // __HIPCC_RTC__

typedef __hip_fp8_storage_t __hip_fp6_storage_t;
typedef __hip_fp8x2_storage_t __hip_fp6x2_storage_t;
typedef __hip_fp8x4_storage_t __hip_fp6x4_storage_t;

static_assert(sizeof(__hip_fp6_storage_t[4]) == sizeof(__hip_uint32_t), "");
static_assert(sizeof(__hip_fp6x2_storage_t[2]) == sizeof(__hip_uint32_t), "");
static_assert(sizeof(__hip_fp6x4_storage_t[2]) == sizeof(__hip_uint64_t), "");

enum __hip_fp6_interpretation_t {
  __HIP_E3M2 = 0, /**< FP6 E3M2 Type*/
  __HIP_E2M3 = 1, /**< FP6 E2M3 Type */
};

// Note: Ignore rounding input on AMD GPUs for now. At the moment AMD GPUs do not support rounding
// modes, all the inputs are rounded to nearest or use an input to do stochastic rounding.
// We hide the rounding variable to not trigger the unused variable compiler warning.
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t __hip_cvt_bfloat16raw_to_fp6(
    const __hip_bfloat16_raw x, const __hip_fp6_interpretation_t fp6_interpretation,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_bf16x32_storage_t in{};
  __amd_fp6x32_storage_t out{};
  in[0] = internal::hipbf16_to_bf16(x);
  if (fp6_interpretation == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(in, 1.0f /* scale */);
  } else if (fp6_interpretation == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(in, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_bf16x16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = internal::hipbf16_to_bf16(x);
  if (fp6_interpretation == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16(in, 1.0f /* scale */);
  } else if (fp6_interpretation == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16(in, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation == __HIP_E2M3)
    u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M3, true>(
        internal::hipbf16_to_bf16(x), 0);
  else if (fp6_interpretation == __HIP_E3M2)
    u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E3M2, true>(
        internal::hipbf16_to_bf16(x), 0);
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t __hip_cvt_bfloat16raw2_to_fp6x2(
    const __hip_bfloat162_raw x, const __hip_fp6_interpretation_t fp6_interpretation,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_bf16x32_storage_t in{};
  in[0] = internal::hipbf16_to_bf16(x.x);
  in[1] = internal::hipbf16_to_bf16(x.y);
  __amd_fp6x32_storage_t out{};
  if (fp6_interpretation == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16(in, 1.0f /* scale */);
  } else if (fp6_interpretation == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16(in, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_bf16x16_storage_t in{};
  in[0] = internal::hipbf16_to_bf16(x.x);
  in[1] = internal::hipbf16_to_bf16(x.y);
  __amd_fp6x16_storage_t out{};
  if (fp6_interpretation == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16(in, 1.0f /* scale */);
  } else if (fp6_interpretation == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16(in, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6x2[0];
#else
  if (fp6_interpretation == __HIP_E2M3) {
    auto bf16x2 = internal::hipbf162_to_bf16x2(x);
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M3, true>(bf16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M3, true>(bf16x2[0], 0);
  } else if (fp6_interpretation == __HIP_E3M2) {
    auto bf16x2 = internal::hipbf162_to_bf16x2(x);
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E3M2, true>(bf16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E3M2, true>(bf16x2[0], 0);
  }
  return u.fp6x2[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t
__hip_cvt_double_to_fp6(const double x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                        const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_floatx16_storage_t in1{};
  __amd_floatx16_storage_t in2{};
  __amd_fp6x32_storage_t out{};
  in1[0] = float(x);
  in2[0] = 0.0f;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = float(x);
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32(in, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32(in, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x), 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x), 0);
  }
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t
__hip_cvt_double2_to_fp6x2(const double2 x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                           const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_floatx16_storage_t in1{};
  __amd_floatx16_storage_t in2{};
  __amd_fp6x32_storage_t out{};
  in1[0] = float(x.x);
  in2[0] = float(x.y);
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  }
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = float(x.x);
  in[1] = float(x.y);
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32(in, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32(in, 1.0f /* scale */);
  }
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x.y), 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(float(x.x), 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x.y), 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(float(x.x), 0);
  }
  return u.fp6x2[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t
__hip_cvt_float_to_fp6(const float x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                       const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_floatx16_storage_t in1{};
  __amd_floatx16_storage_t in2{};
  __amd_fp6x32_storage_t out{};
  in1[0] = x;
  in2[0] = 0.0f;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = float(x);
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32(in, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32(in, 1.0f /* scale */);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3)
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M3, true>(x, 0);
  else if (fp6_interpretation_t == __HIP_E3M2)
    u.ui32 = fcbx::from_float<float, fcbx::Encoding::E3M2, true>(x, 0);
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t
__hip_cvt_float2_to_fp6x2(const float2 x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                          const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_floatx16_storage_t in1{};
  __amd_floatx16_storage_t in2{};
  __amd_fp6x32_storage_t out{};
  in1[0] = x.x;
  in2[0] = x.y;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(in1, in2, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(in1, in2, 1.0f /* scale */);
  }
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = float(x.x);
  in[1] = float(x.y);
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32(in, 1.0f /* scale */);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_f32))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32(in, 1.0f /* scale */);
  }
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(x.y, 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M3, true>(x.x, 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(x.y, 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E3M2, true>(x.x, 0);
  }
  return u.fp6x2[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp6_to_halfraw(
    const __hip_fp6_storage_t x, const __hip_fp6_interpretation_t fp6_interpretation_t) {
  __half_raw ret{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x32_storage_t out{};
  __amd_fp6x32_storage_t in{};
  in[0] = (__hip_uint32_t)x;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f16_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(in, 1.0f);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f16_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(in, 1.0f);
  }
  ret.data = out[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_fp16x16_storage_t out{};
  __amd_fp6x16_storage_t in{};
  in[0] = (__hip_uint32_t)x;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f16_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_f16_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f16_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_f16_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
  }
  ret.data = out[0];
#else
  using namespace fcbx;
  if (fp6_interpretation_t == __HIP_E2M3) {
    ret.data = __amd_fp16_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M3, true>(x, 0)};
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    ret.data = __amd_fp16_storage_t{to_float<__amd_fp16_storage_t, Encoding::E3M2, true>(x, 0)};
  }
#endif
  return ret;
}
__FP6_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp6x2_to_halfraw2(
    const __hip_fp6x2_storage_t x, const __hip_fp6_interpretation_t fp6_interpretation_t) {
  __half2_raw ret{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x32_storage_t out{};
  __amd_fp6x32_storage_t in{};
  in[0] = x & 0x3Fu;            // first 6 bits
  in[0] |= (x & 0x3F00u) >> 2;  // next 6 bits
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f16_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f16_fp6(in, 1.0f);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f16_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f16_bf6(in, 1.0f);
  }
  ret.data = {out[0], out[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_fp16x16_storage_t out{};
  __amd_fp6x16_storage_t in{};
  in[0] = x & 0x3Fu;            // first 6 bits
  in[0] |= (x & 0x3F00u) >> 2;  // next 6 bits
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f16_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_f16_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f16_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_f16_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
  }
  ret.data = {out[0], out[1]};
#else
  using namespace fcbx;
  if (fp6_interpretation_t == __HIP_E2M3) {
    ret.data =
        __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M3, true>(x & 0xFFu, 0),
                               to_float<__amd_fp16_storage_t, Encoding::E2M3, true>(x >> 8, 0)};
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    ret.data =
        __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E3M2, true>(x & 0xFFu, 0),
                               to_float<__amd_fp16_storage_t, Encoding::E3M2, true>(x >> 8, 0)};
  }
#endif
  return ret;
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6_storage_t
__hip_cvt_halfraw_to_fp6(const __half_raw x, const __hip_fp6_interpretation_t fp6_interpretation_t,
                         const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6_storage_t fp6[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x32_storage_t in{};
  __amd_fp6x32_storage_t out{};
  in[0] = x.data;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_fp6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(in, 1.0f);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(in, 1.0f);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_fp16x16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = x.data;
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_f16(in, 1.0f);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_f16(in, 1.0f);
  }
  u.ui32 = out[0];
  return u.fp6[0];
#else
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 = fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M3, true>(
        internal::half_to_f16(x), 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 = fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E3M2, true>(
        internal::half_to_f16(x), 0);
  }
  return u.fp6[0];
#endif
}
__FP6_HOST_DEVICE_STATIC__ __hip_fp6x2_storage_t __hip_cvt_halfraw2_to_fp6x2(
    const __half2_raw x, const __hip_fp6_interpretation_t fp6_interpretation_t,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp6x2_storage_t fp6x2[2];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  __amd_fp16x32_storage_t in{};
  __amd_fp6x32_storage_t out{};
  in[0] = x.data[0];
  in[1] = x.data[1];
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_fp6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16(in, 1.0f);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16(in, 1.0f);
  }
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_fp16x16_storage_t in{};
  __amd_fp6x16_storage_t out{};
  in[0] = x.data[0];
  in[1] = x.data[1];
  if (fp6_interpretation_t == __HIP_E2M3) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_fp6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_fp6_f16(in, 1.0f);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk16_bf6_f16))
      out = __builtin_amdgcn_cvt_scalef32_pk16_bf6_f16(in, 1.0f);
  }
  u.ui32 = out[0] & 0x3Fu;
  u.ui32 |= ((out[0] & 0xFC0u) << 2);
  return u.fp6x2[0];
#else
  auto fp16x2 = internal::half2_to_f16x2(x);
  if (fp6_interpretation_t == __HIP_E2M3) {
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M3, true>(fp16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M3, true>(fp16x2[0], 0);
  } else if (fp6_interpretation_t == __HIP_E3M2) {
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E3M2, true>(fp16x2[1], 0);
    u.ui32 <<= 8;
    u.ui32 |= fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E3M2, true>(fp16x2[0], 0);
  }
  return u.fp6x2[0];
#endif
}

//======================== structs ====================
struct __hip_fp6_e2m3 {
  __hip_fp6_storage_t __x;

  __FP6_HOST_DEVICE__ __hip_fp6_e2m3() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const __half f)
      : __x(__hip_cvt_halfraw_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const __hip_bfloat16 f)
      : __x(__hip_cvt_bfloat16raw_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const double f)
      : __x(__hip_cvt_double_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const float f)
      : __x(__hip_cvt_float_to_fp6(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e2m3(const unsigned short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E2M3, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)

#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half_raw() const {
    return __hip_cvt_fp6_to_halfraw(__x, __HIP_E2M3);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat16_raw() const {
    static_assert(sizeof(__hip_bfloat16_raw) == sizeof(__amd_bf16_storage_t), "");
    union {
      __hip_bfloat16_raw bf16_raw;
      __amd_bf16_storage_t bf16;
    } u{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_bf16x32_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(in, 1.0f /* scale */);
    u.bf16 = out[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_bf16x16_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_bf16_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_bf16_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    u.bf16 = out[0];
#else
    using namespace fcbx;
    u.bf16 = to_float<__amd_bf16_storage_t, Encoding::E2M3, true>(__x, 0);
#endif
    return u.bf16_raw;
  }
  __FP6_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_floatx32_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(in, 1.0f /* scale */);
    auto ret = out[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_floatx16_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f32_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_f32_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    auto ret = out[0];
#else
    using namespace fcbx;
    float ret{to_float<float, Encoding::E2M3, true>(__x, 0)};
#endif
    return ret;
  }
  __FP6_HOST_DEVICE__ operator double() const { return double(float(*this)); }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6_e3m2 {
  __hip_fp6_storage_t __x;

  __FP6_HOST_DEVICE__ __hip_fp6_e3m2() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const __half f)
      : __x(__hip_cvt_halfraw_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const __hip_bfloat16 f)
      : __x(__hip_cvt_bfloat16raw_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const double f)
      : __x(__hip_cvt_double_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const float f)
      : __x(__hip_cvt_float_to_fp6(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned long long int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6_e3m2(const unsigned short int val)
      : __x(__hip_cvt_float_to_fp6(float(val), __HIP_E3M2, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half_raw() const {
    return __hip_cvt_fp6_to_halfraw(__x, __HIP_E3M2);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat16_raw() const {
    static_assert(sizeof(__hip_bfloat16_raw) == sizeof(__amd_bf16_storage_t), "");
    union {
      __hip_bfloat16_raw bf16_raw;
      __amd_bf16_storage_t bf16;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_bf16x32_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(in, 1.0f /* scale */);
    u.bf16 = out[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_bf16x16_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_bf16_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_bf16_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    u.bf16 = out[0];
#else
    using namespace fcbx;
    u.bf16 = to_float<__amd_bf16_storage_t, Encoding::E3M2, true>(__x, 0);
#endif
    return u.bf16_raw;
  }
  __FP6_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_floatx32_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f32_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(in, 1.0f /* scale */);
    auto ret = out[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_floatx16_storage_t out{};
    in[0] = (__hip_uint32_t)__x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f32_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_f32_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    auto ret = out[0];
#else
    using namespace fcbx;
    float ret{to_float<float, Encoding::E3M2, true>(__x, 0)};
#endif
    return ret;
  }
  __FP6_HOST_DEVICE__ operator double() const { return double(float(*this)); }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x2_e2m3 {
  __hip_fp6x2_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x2_e2m3() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const __half2 f)
      : __x(__hip_cvt_halfraw2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const __hip_bfloat162 f)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const double2 f)
      : __x(__hip_cvt_double2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e2m3(const float2 f)
      : __x(__hip_cvt_float2_to_fp6x2(f, __HIP_E2M3, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half2_raw() const {
    return __hip_cvt_fp6x2_to_halfraw2(__x, __HIP_E2M3);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    static_assert(sizeof(__hip_bfloat162_raw) == sizeof(__amd_bf16x2_storage_t), "");
    union {
      __hip_bfloat162_raw bf162_raw;
      __amd_bf16x2_storage_t bf16x2;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_bf16x32_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_fp6(in, 1.0f /* scale */);
    u.bf16x2 = {out[0], out[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_bf16x16_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_bf16_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_bf16_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    u.bf16x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M3, true>(__x & 0xFFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M3, true>(__x >> 8, 0)};
#endif
    return u.bf162_raw;
  }
  __FP6_HOST_DEVICE__ operator float2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_floatx32_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(in, 1.0f /* scale */);
    __amd_floatx2_storage_t fp32x2 = {out[0], out[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_floatx16_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f32_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_f32_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    __amd_floatx2_storage_t fp32x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M3, true>(__x & 0xFFu, 0),
                                          to_float<float, Encoding::E2M3, true>(__x >> 8, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }
  __FP6_HOST_DEVICE__ operator double2() const {
    auto fp32 = float2(*this);
    return double2(fp32.x, fp32.y);
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x2_e3m2 {
  __hip_fp6x2_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x2_e3m2() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const __half2 f)
      : __x(__hip_cvt_halfraw2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const __hip_bfloat162 f)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const double2 f)
      : __x(__hip_cvt_double2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x2_e3m2(const float2 f)
      : __x(__hip_cvt_float2_to_fp6x2(f, __HIP_E3M2, hipRoundNearest)) {}
#endif  //! defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator __half2_raw() const {
    return __hip_cvt_fp6x2_to_halfraw2(__x, __HIP_E3M2);
  }
  __FP6_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    static_assert(sizeof(__hip_bfloat162_raw) == sizeof(__amd_bf16x2_storage_t), "");
    union {
      __hip_bfloat162_raw bf162_raw;
      __amd_bf16x2_storage_t bf16x2;
    } u;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_bf16x32_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_bf16_bf6(in, 1.0f /* scale */);
    u.bf16x2 = {out[0], out[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_bf16x16_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_bf16_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_bf16_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    u.bf16x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E3M2, true>(__x & 0xFFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E3M2, true>(__x >> 8, 0)};
#endif
    return u.bf162_raw;
  }
  __FP6_HOST_DEVICE__ operator float2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_floatx32_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f32_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(in, 1.0f /* scale */);
    __amd_floatx2_storage_t fp32x2 = {out[0], out[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_floatx16_storage_t out{};
    in[0] = __x & 0x3Fu;            // first 6 bits
    in[0] |= (__x & 0x3F00u) >> 2;  // next 6 bits
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f32_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_f32_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    __amd_floatx2_storage_t fp32x2 = {out[0], out[1]};
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E3M2, true>(__x & 0xFFu, 0),
                                          to_float<float, Encoding::E3M2, true>(__x >> 8, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }
  __FP6_HOST_DEVICE__ operator double2() const {
    auto fp32 = float2(*this);
    return double2(fp32.x, fp32.y);
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x4_e2m3 {
  __hip_fp6x4_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x4_e2m3() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const __half2 low, const __half2 high)
      : __x(__hip_cvt_halfraw2_to_fp6x2(high, __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_halfraw2_to_fp6x2(low, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const __hip_bfloat162 low,
                                                       const __hip_bfloat162 high)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(high, __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_bfloat16raw2_to_fp6x2(low, __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const double4 f)
      : __x(__hip_cvt_double2_to_fp6x2(double2(f.z, f.w), __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_double2_to_fp6x2(double2(f.x, f.y), __HIP_E2M3, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e2m3(const float4 f)
      : __x(__hip_cvt_float2_to_fp6x2(float2(f.z, f.w), __HIP_E2M3, hipRoundNearest) << 16 |
            __hip_cvt_float2_to_fp6x2(float2(f.x, f.y), __HIP_E2M3, hipRoundNearest)) {}
#endif  // !defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator float4() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_floatx32_storage_t out{};
    in[0] = __x & 0x3Fu;                 // first 6 bits
    in[0] |= ((__x >> 8) & 0x3Fu) << 6;  // second 6 bits
    in[0] |= ((__x >> 16) & 0x3Fu) << 12;
    in[0] |= ((__x >> 24) & 0x3Fu) << 18;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(in, 1.0f /* scale */);
    __amd_floatx2_storage_t fp32x2_1 = {out[0], out[1]};
    __amd_floatx2_storage_t fp32x2_2 = {out[2], out[3]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_floatx16_storage_t out{};
    in[0] = __x & 0x3Fu;                 // first 6 bits
    in[0] |= ((__x >> 8) & 0x3Fu) << 6;  // second 6 bits
    in[0] |= ((__x >> 16) & 0x3Fu) << 12;
    in[0] |= ((__x >> 24) & 0x3Fu) << 18;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f32_fp6))
      out = __builtin_amdgcn_cvt_scale_pk16_f32_fp6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    __amd_floatx2_storage_t fp32x2_1 = {out[0], out[1]};
    __amd_floatx2_storage_t fp32x2_2 = {out[2], out[3]};
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M3, true>(__x & 0xFFu, 0),
                                to_float<float, Encoding::E2M3, true>((__x >> 8) & 0xFFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M3, true>((__x >> 16) & 0xFFu, 0),
                                to_float<float, Encoding::E2M3, true>(__x >> 24, 0)};
#endif
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
  }
  __FP6_HOST_DEVICE__ operator double4() const {
    auto fp32 = float4(*this);
    return double4{fp32.x, fp32.y, fp32.z, fp32.w};
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

struct __hip_fp6x4_e3m2 {
  __hip_fp6x4_storage_t __x;
  __FP6_HOST_DEVICE__ inline __hip_fp6x4_e3m2() = default;
#if !defined(__HIP_NO_FP6_CONVERSIONS__)
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const __half2 low, const __half2 high)
      : __x(__hip_cvt_halfraw2_to_fp6x2(high, __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_halfraw2_to_fp6x2(low, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const __hip_bfloat162 low,
                                                       const __hip_bfloat162 high)
      : __x(__hip_cvt_bfloat16raw2_to_fp6x2(high, __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_bfloat16raw2_to_fp6x2(low, __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const double4 f)
      : __x(__hip_cvt_double2_to_fp6x2(double2(f.z, f.w), __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_double2_to_fp6x2(double2(f.x, f.y), __HIP_E3M2, hipRoundNearest)) {}
  __FP6_HOST_DEVICE__ inline explicit __hip_fp6x4_e3m2(const float4 f)
      : __x(__hip_cvt_float2_to_fp6x2(float2(f.z, f.w), __HIP_E3M2, hipRoundNearest) << 16 |
            __hip_cvt_float2_to_fp6x2(float2(f.x, f.y), __HIP_E3M2, hipRoundNearest)) {}
#endif  //! defined(__HIP_NO_FP6_CONVERSIONS__)
#if !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
  __FP6_HOST_DEVICE__ operator float4() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    __amd_fp6x32_storage_t in{};
    __amd_floatx32_storage_t out{};
    in[0] = __x & 0x3Fu;                 // first 6 bits
    in[0] |= ((__x >> 8) & 0x3Fu) << 6;  // second 6 bits
    in[0] |= ((__x >> 16) & 0x3Fu) << 12;
    in[0] |= ((__x >> 24) & 0x3Fu) << 18;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk32_f32_bf6))
      out = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(in, 1.0f /* scale */);
    __amd_floatx2_storage_t fp32x2_1 = {out[0], out[1]};
    __amd_floatx2_storage_t fp32x2_2 = {out[2], out[3]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    __amd_fp6x16_storage_t in{};
    __amd_floatx16_storage_t out{};
    in[0] = __x & 0x3Fu;                 // first 6 bits
    in[0] |= ((__x >> 8) & 0x3Fu) << 6;  // second 6 bits
    in[0] |= ((__x >> 16) & 0x3Fu) << 12;
    in[0] |= ((__x >> 24) & 0x3Fu) << 18;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk16_f32_bf6))
      out = __builtin_amdgcn_cvt_scale_pk16_f32_bf6(in, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    __amd_floatx2_storage_t fp32x2_1 = {out[0], out[1]};
    __amd_floatx2_storage_t fp32x2_2 = {out[2], out[3]};
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E3M2, true>(__x & 0xFFu, 0),
                                to_float<float, Encoding::E3M2, true>((__x >> 8) & 0xFFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E3M2, true>((__x >> 16) & 0xFFu, 0),
                                to_float<float, Encoding::E3M2, true>(__x >> 24, 0)};
#endif
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
  }
  __FP6_HOST_DEVICE__ operator double4() const {
    auto fp32 = float4(*this);
    return double4{fp32.x, fp32.y, fp32.z, fp32.w};
  }
#endif  // !defined(__HIP_NO_FP6_CONVERSION_OPERATORS__)
};

#endif  // _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP6_H_
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP4_H_
#define _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP4_H_

#if defined(__HIPCC_RTC__)
#define __FP4_HOST_DEVICE__ __device__
#define __FP4_HOST_DEVICE_STATIC__ __FP4_HOST_DEVICE__ static
#else
#define __FP4_HOST_DEVICE__ __host__ __device__
#define __FP4_HOST_DEVICE_STATIC__ __FP4_HOST_DEVICE__ static inline
#include "amd_hip_mx_common.h"
#include "amd_hip_fp8.h"

#include "amd_hip_ocp_host.hpp"
#endif  // __HIPCC_RTC__

typedef __hip_fp8_storage_t __hip_fp4_storage_t;
typedef __hip_fp8_storage_t __hip_fp4x2_storage_t;
typedef __hip_fp8x2_storage_t __hip_fp4x4_storage_t;

static_assert(sizeof(__hip_fp4_storage_t[4]) == sizeof(__hip_uint32_t), "");
static_assert(sizeof(__hip_fp4x2_storage_t[4]) == sizeof(__hip_uint32_t), "");
static_assert(sizeof(__hip_fp4x4_storage_t[2]) == sizeof(__hip_uint32_t), "");

enum __hip_fp4_interpretation_t {
  __HIP_E2M1 = 0,
};

// Note: Ignore rounding input on AMD GPUs for now. At the moment AMD GPUs do not support rounding
// modes, all the inputs are rounded to nearest or use an input to do stochastic rounding.
// We hide the rounding variable to not trigger the unused variable compiler warning.
__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_bfloat16raw_to_fp4(
    const __hip_bfloat16_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(
        u.ui32, internal::hipbf162_to_bf16x2(__hip_bfloat162{x, 0}), 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  union {
    __hip_bfloat16_raw bhalf_raw;
    __bf16 bf16;
  } bhalf{x};
  __amd_bf16x8_storage_t bf16x8{bhalf.bf16, bhalf.bf16, bhalf.bf16, bhalf.bf16,
                                bhalf.bf16, bhalf.bf16, bhalf.bf16, bhalf.bf16};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16(bf16x8, 1.0f);
  return u.fp4[0] & 0xFu;
#else
  u.ui32 = fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(
      internal::hipbf16_to_bf16(x), 0 /* scale */);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_bfloat16raw2_to_fp4x2(
    const __hip_bfloat162_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(u.ui32, internal::hipbf162_to_bf16x2(x),
                                                       1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  auto bf16x2 = internal::hipbf162_to_bf16x2(x);
  __amd_bf16x8_storage_t bf16x8{bf16x2[0], bf16x2[1], bf16x2[0], bf16x2[1],
                                bf16x2[0], bf16x2[1], bf16x2[0], bf16x2[1]};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16(bf16x8, 1.0f);
  return u.fp4x2[0];
#else
  auto bf16x2 = internal::hipbf162_to_bf16x2(x);
  u.ui32 |=
      fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(bf16x2[1], 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |=
      fcbx::from_float<__amd_bf16_storage_t, fcbx::Encoding::E2M1, true>(bf16x2[0], 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t
__hip_cvt_double_to_fp4(const double x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
                        const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, float(x), 0.0f, 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{float(x), float(x), float(x), float(x),
                                 float(x), float(x), float(x), float(x)};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4[0] & 0xFu;
#else
  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x), 0 /* scale */);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_double2_to_fp4x2(
    const double2 x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, float(x.x), float(x.y),
                                                      1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{float(x.x), float(x.y), float(x.x), float(x.y),
                                 float(x.x), float(x.y), float(x.x), float(x.y)};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4x2[0];
#else
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x.y), 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(float(x.x), 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t
__hip_cvt_float_to_fp4(const float x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
                       const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x, 0.0f, 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{x, x, x, x, x, x, x, x};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4[0] & 0xFu;
#else
  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x, 0 /*scale*/);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t
__hip_cvt_float2_to_fp4x2(const float2 x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
                          const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x.x, x.y, 1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  __amd_floatx8_storage_t fp32x8{x.x, x.y, x.x, x.y, x.x, x.y, x.x, x.y};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f32))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32(fp32x8, 1.0f);
  return u.fp4x2[0];
#else
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x.y, 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |= fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x.x, 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp4_to_halfraw(
    const __hip_fp4_storage_t x, const __hip_fp4_interpretation_t /* fp4_interpretation */) {
  __half2_raw ret;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  ret.data =
      __amd_fp16x2_storage_t{__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp4)
                                 ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(x, 0, 0)
                                 : __amd_fp16x2_storage_t{}};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  static_assert(sizeof(__amd_fp16x2_storage_t[4]) == sizeof(__amd_fp16x8_storage_t));
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    __amd_fp16x8_storage_t fp16x8;
    __amd_fp16x2_storage_t fp16x2[4];
  } uhalf{};
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.fp4x2[0] = (x & 0xFu) | ((x & 0xFu) << 4);
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f16_fp4))
    uhalf.fp16x8 =
        __builtin_amdgcn_cvt_scale_pk8_f16_fp4(u.ui32, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
  ret.data = uhalf.fp16x2[0];
#else
  using namespace fcbx;
  ret.data =
      __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x & 0xFu, 0),
                             to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x >> 4, 0)};
#endif
  return ret.x;
}

__FP4_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp4x2_to_halfraw2(
    const __hip_fp4x2_storage_t x, const __hip_fp4_interpretation_t /* fp4_interpretation */) {
  __half2_raw ret;
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  ret.data =
      __amd_fp16x2_storage_t{__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f16_fp4)
                                 ? __builtin_amdgcn_cvt_scalef32_pk_f16_fp4(x, 0, 0)
                                 : __amd_fp16x2_storage_t{}};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  static_assert(sizeof(__amd_fp16x2_storage_t[4]) == sizeof(__amd_fp16x8_storage_t));
  static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
  union {
    __amd_fp16x8_storage_t fp16x8;
    __amd_fp16x2_storage_t fp16x2[4];
  } uhalf{};
  union {
    unsigned int ui32;
    __amd_fp4x2_storage_t fp4x2[4];
  } u{0};
  u.fp4x2[0] = x;
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f16_fp4))
    uhalf.fp16x8 = __builtin_amdgcn_cvt_scale_pk8_f16_fp4(u.ui32, 0x7F7Fu, 0);
  ret.data = uhalf.fp16x2[0];
#else
  using namespace fcbx;
  ret.data =
      __amd_fp16x2_storage_t{to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x & 0xFu, 0),
                             to_float<__amd_fp16_storage_t, Encoding::E2M1, true>(x >> 4, 0)};
#endif
  return ret;
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_halfraw_to_fp4(
    const __half_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4_storage_t fp4[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(
        u.ui32, internal::half2_to_f16x2(__half2{x, 0}), 1.0f /* scale */, 0);
  return u.fp4[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  auto val = internal::half2_to_f16x2(__half2{x, x});
  __amd_fp16x8_storage_t fp16x8{val[0], val[1], val[0], val[1], val[0], val[1], val[0], val[1]};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16(fp16x8, 1.0f);
  return u.fp4[0] & 0xFu;
#else
  u.ui32 = fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M1, true>(
      internal::half_to_f16(x), 0 /* scale */);
  return u.fp4[0];
#endif
}

__FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_halfraw2_to_fp4x2(
    const __half2_raw x, const __hip_fp4_interpretation_t /* fp4_interpretation */,
    const enum hipRoundMode /* rounding */) {
  union {
    __hip_uint32_t ui32;
    __hip_fp4x2_storage_t fp4x2[4];
  } u{0};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f16(u.ui32, internal::half2_to_f16x2(x),
                                                      1.0f /* scale */, 0);
  return u.fp4x2[0];
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
  auto val = internal::half2_to_f16x2(x);
  __amd_fp16x8_storage_t fp16x8{val[0], val[1], val[0], val[1], val[0], val[1], val[0], val[1]};
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk8_fp4_f16))
    u.ui32 = __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16(fp16x8, 1.0f);
  return u.fp4x2[0];
#else
  auto fp16x2 = internal::half2_to_f16x2(x);
  u.ui32 |=
      fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M1, true>(fp16x2[1], 0 /*scale*/);
  u.ui32 <<= 4;
  u.ui32 |=
      fcbx::from_float<__amd_fp16_storage_t, fcbx::Encoding::E2M1, true>(fp16x2[0], 0 /*scale*/);
  return u.fp4x2[0];
#endif
}

struct __hip_fp4_e2m1 {
  __hip_fp4_storage_t __x;

 public:
  __FP4_HOST_DEVICE__ __hip_fp4_e2m1() = default;

#if !defined(__HIP_NO_FP4_CONVERSIONS__)
  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const __half f)
      : __x(__hip_cvt_halfraw_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const __hip_bfloat16 f)
      : __x(__hip_cvt_bfloat16raw_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__
  explicit __hip_fp4_e2m1(const double f)
      : __x(__hip_cvt_double_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const float f)
      : __x(__hip_cvt_float_to_fp4(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const long long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const short int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned long long int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4_e2m1(const unsigned short int val)
      : __x(__hip_cvt_float_to_fp4(float(val), __HIP_E2M1, hipRoundNearest)) {}
#endif  // #if !defined(__HIP_NO_FP4_CONVERSIONS__)

#if !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
  __FP4_HOST_DEVICE__ operator __half_raw() const {
    return __hip_cvt_fp4_to_halfraw(__x, __HIP_E2M1);
  }

  __FP4_HOST_DEVICE__ operator __hip_bfloat16_raw() const {
    static_assert(sizeof(__hip_bfloat16_raw[2]) == sizeof(__amd_bf16x2_storage_t), "");
    union {
      __hip_bfloat16_raw bf16_raw[2];
      __amd_bf16x2_storage_t bf16x2;
    } u{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4))
      u.bf16x2 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(__x, 1.0f /* scale */, 0);
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    static_assert(sizeof(__amd_bf16x2_storage_t[4]) == sizeof(__amd_bf16x8_storage_t));
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
    union {
      __amd_bf16x8_storage_t bf16x8;
      __amd_bf16x2_storage_t bf16x2[4];
    } bhalf{};
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u_in{0};
    u_in.fp4x2[0] = (__x & 0xFu) | ((__x & 0xFu) << 4);
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_bf16_fp4))
      bhalf.bf16x8 =
          __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(u_in.ui32, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
    u.bf16x2[0] = bhalf.bf16x8[0];
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x & 0xFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return u.bf16_raw[0];
  }

  __FP4_HOST_DEVICE__ operator float() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto ret = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                   ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, 1.0f /* scale */, 0)
                   : __amd_floatx2_storage_t{};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    // Broadcast the single fp4 value (low nibble) into both nibbles of byte 0.
    u.fp4x2[0] = (__x & 0xFu) | ((__x & 0xFu) << 4);
    __amd_floatx8_storage_t ret{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      ret = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu /* scale e8m0 = 1.0 */, 0);
#else
    using namespace fcbx;
    __amd_floatx2_storage_t ret{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return ret[0];
  }

  __FP4_HOST_DEVICE__ operator double() const { return double(float(*this)); }
#endif  // !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
};

/* FP4x2 E2M1 */
struct __hip_fp4x2_e2m1 {
  __hip_fp4x2_storage_t __x;

  __FP4_HOST_DEVICE__ __hip_fp4x2_e2m1() = default;

#if !defined(__HIP_NO_FP4_CONVERSIONS__)
  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const __half2 f)
      : __x(__hip_cvt_halfraw2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const __hip_bfloat162 f)
      : __x(__hip_cvt_bfloat16raw2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const double2 f)
      : __x(__hip_cvt_double2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ explicit __hip_fp4x2_e2m1(const float2 f)
      : __x(__hip_cvt_float2_to_fp4x2(f, __HIP_E2M1, hipRoundNearest)) {}

#endif  // #if !defined(__HIP_NO_FP4_CONVERSIONS__)

#if !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
  __FP4_HOST_DEVICE__ operator __half2_raw() const {
    return __hip_cvt_fp4x2_to_halfraw2(__x, __HIP_E2M1);
  }

  __FP4_HOST_DEVICE__ operator __hip_bfloat162_raw() const {
    static_assert(sizeof(__hip_bfloat162_raw) == sizeof(__amd_bf16x2_storage_t), "");
    union {
      __hip_bfloat162_raw bf162_raw;
      __amd_bf16x2_storage_t bf16x2;
    } u{};
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_bf16_fp4))
      u.bf16x2 = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(__x, 1.0f /* scale */, 0);
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    static_assert(sizeof(__amd_bf16x2_storage_t[4]) == sizeof(__amd_bf16x8_storage_t));
    static_assert(sizeof(__amd_fp4x2_storage_t[4]) == sizeof(unsigned int));
    union {
      __amd_bf16x8_storage_t bf16x8;
      __amd_bf16x2_storage_t bf16x2[4];
    } bhalf{};
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u_in{0};
    u_in.fp4x2[0] = __x;
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_bf16_fp4))
      bhalf.bf16x8 = __builtin_amdgcn_cvt_scale_pk8_bf16_fp4(u_in.ui32, 0x7F7Fu, 0);
    u.bf16x2[0] = bhalf.bf16x8[0];
    u.bf16x2[1] = bhalf.bf16x8[1];
#else
    using namespace fcbx;
    u.bf16x2 =
        __amd_bf16x2_storage_t{to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x & 0xFu, 0),
                               to_float<__amd_bf16_storage_t, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return u.bf162_raw;
  }

  __FP4_HOST_DEVICE__ operator float2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto fp32x2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                      ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, 1.0f /* scale */, 0)
                      : __amd_floatx2_storage_t{};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.fp4x2[0] = __x;

    // Even though it's x8, we call it x2 so that we have a common return line
    __amd_floatx8_storage_t fp32x2{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      fp32x2 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                          to_float<float, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return float2(fp32x2[0], fp32x2[1]);
  }

  __FP4_HOST_DEVICE__ operator double2() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto fp32x2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                      ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x, 1.0f /* scale */, 0)
                      : __amd_floatx2_storage_t{};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __amd_fp4x2_storage_t fp4x2[4];
    } u{0};
    u.fp4x2[0] = __x;
    // Although it's x8 but we still call it x2 since we will use it in common code.
    __amd_floatx8_storage_t fp32x2{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      fp32x2 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
#else
    using namespace fcbx;
    auto fp32x2 = __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                          to_float<float, Encoding::E2M1, true>(__x >> 4, 0)};
#endif
    return double2(fp32x2[0], fp32x2[1]);
  }
#endif  // !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
};

/* FP4x4 E2M1 */
struct __hip_fp4x4_e2m1 {
  __hip_fp4x4_storage_t __x;

  __FP4_HOST_DEVICE__ inline __hip_fp4x4_e2m1() = default;

#if !defined(__HIP_NO_FP4_CONVERSIONS__)
  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const __half2 low, const __half2 high)
      : __x(__hip_cvt_halfraw2_to_fp4x2(high, __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_halfraw2_to_fp4x2(low, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const __hip_bfloat162 low,
                                                       const __hip_bfloat162 high)
      : __x(__hip_cvt_bfloat16raw2_to_fp4x2(high, __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_bfloat16raw2_to_fp4x2(low, __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const double4 f)
      : __x(__hip_cvt_double2_to_fp4x2(double2(f.z, f.w), __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_double2_to_fp4x2(double2(f.x, f.y), __HIP_E2M1, hipRoundNearest)) {}

  __FP4_HOST_DEVICE__ inline explicit __hip_fp4x4_e2m1(const float4 f)
      : __x(__hip_cvt_float2_to_fp4x2(float2(f.z, f.w), __HIP_E2M1, hipRoundNearest) << 8 |
            __hip_cvt_float2_to_fp4x2(float2(f.x, f.y), __HIP_E2M1, hipRoundNearest)) {}
#endif  // #if !defined(__HIP_NO_FP4_CONVERSIONS__)

#if !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
  __FP4_HOST_DEVICE__ operator float4() const {
#if HIP_ENABLE_GFX950_OCP_BUILTINS
    auto fp32x2_1 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                        ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x & 0xFFu, 1.0f /* scale */, 0)
                        : __amd_floatx2_storage_t{};
    auto fp32x2_2 = __builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)
                        ? __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(__x >> 8, 1.0f /* scale */, 0)
                        : __amd_floatx2_storage_t{};
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
#elif HIP_ENABLE_GFX1250_OCP_BUILTINS
    union {
      unsigned int ui32;
      __hip_fp4x4_storage_t fp4x4[2];
    } u{0};
    u.fp4x4[0] = __x;
    __amd_floatx8_storage_t fp32x8{};
    if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_cvt_scale_pk8_f32_fp4))
      fp32x8 = __builtin_amdgcn_cvt_scale_pk8_f32_fp4(u.ui32, 0x7F7Fu, 0);
    return float4{fp32x8[0], fp32x8[1], fp32x8[2], fp32x8[3]};
#else
    using namespace fcbx;
    auto fp32x2_1 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>(__x & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>((__x >> 4) & 0xFu, 0)};
    auto fp32x2_2 =
        __amd_floatx2_storage_t{to_float<float, Encoding::E2M1, true>((__x >> 8) & 0xFu, 0),
                                to_float<float, Encoding::E2M1, true>(__x >> 12, 0)};
    return float4{fp32x2_1[0], fp32x2_1[1], fp32x2_2[0], fp32x2_2[1]};
#endif
  }

  __FP4_HOST_DEVICE__ operator double4() const {
    auto fp32 = float4(*this);
    return double4{fp32.x, fp32.y, fp32.z, fp32.w};
  }
#endif  // !defined(__HIP_NO_FP4_CONVERSION_OPERATORS__)
};

#endif  // _HIP_INCLUDE_HIP_AMD_DETAIL_HIP_FP4_H_
#pragma clang diagnostic pop
