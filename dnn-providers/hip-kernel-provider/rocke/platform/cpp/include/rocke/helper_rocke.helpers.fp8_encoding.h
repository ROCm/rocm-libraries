/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.fp8_encoding.h -- C99 port of
 * rocke.helpers.fp8_encoding: the fp8 / bf8 element encodings and the one each
 * target's conversions produce and its matrix cores read.
 *
 *   Python (helpers/fp8_encoding.py)       C99 (this file)
 *   Fp8ElementFormat                   ->  rocke_mx_element_format_t
 *   FP8_E4M3_OCP / BF8_E5M2_OCP        ->  ROCKE_FP8_E4M3_OCP / ROCKE_BF8_E5M2_OCP
 *   FP8_E4M3_FNUZ / BF8_E5M2_FNUZ      ->  ROCKE_FP8_E4M3_FNUZ / ROCKE_BF8_E5M2_FNUZ
 *   fp8_element_format(dtype, arch)    ->  rocke_fp8_element_format()
 *
 * The C type keeps its original rocke_mx_element_format_t name: renaming a
 * public C type would break every C caller's source. The MX scale helpers
 * (helper_rocke.helpers.mx_scale.h) include this header.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_FP8_ENCODING_H
#define ROCKE_HELPER_ROCKE_HELPERS_FP8_ENCODING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Python Fp8ElementFormat: an fp8 / bf8 element encoding; name names it in
 * messages and largest_value is its largest finite magnitude. */
typedef struct rocke_mx_element_format
{
    const char* name;
    double largest_value;
} rocke_mx_element_format_t;

/* The four encodings (Python FP8_E4M3_OCP, BF8_E5M2_OCP, FP8_E4M3_FNUZ,
 * BF8_E5M2_FNUZ). */
extern const rocke_mx_element_format_t ROCKE_FP8_E4M3_OCP;
extern const rocke_mx_element_format_t ROCKE_BF8_E5M2_OCP;
extern const rocke_mx_element_format_t ROCKE_FP8_E4M3_FNUZ;
extern const rocke_mx_element_format_t ROCKE_BF8_E5M2_FNUZ;

/* Python fp8_element_format(dtype, arch): the encoding `arch` uses for `dtype`
 * ("fp8e4m3" or "bf8e5m2"): the FNUZ encodings on gfx942, the OCP ones on
 * gfx950 and gfx1250.
 *
 * Returns one of the four formats above (borrowed, valid for the program's
 * lifetime), or NULL with the Python-matching message in err (capacity
 * err_cap, may be NULL) for any other dtype or arch. */
const rocke_mx_element_format_t*
    rocke_fp8_element_format(const char* dtype, const char* arch, char* err, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_FP8_ENCODING_H */
