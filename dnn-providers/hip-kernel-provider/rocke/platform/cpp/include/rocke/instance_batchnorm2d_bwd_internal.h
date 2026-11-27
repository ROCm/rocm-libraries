/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_batchnorm2d_bwd_internal.h -- builder pieces shared by the
 * NCHW backward (batchnorm2d_bwd.cpp) and NHWC (batchnorm2d_nhwc.cpp)
 * BatchNorm2D mirrors. Not part of the public C surface.
 */
#ifndef ROCKE_INSTANCE_BATCHNORM2D_BWD_INTERNAL_H
#define ROCKE_INSTANCE_BATCHNORM2D_BWD_INTERNAL_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Pointer-parameter access facts for rocke_bn_ptr_param. */
#define ROCKE_BN_PARAM_PLAIN 0 /* noalias only                    */
#define ROCKE_BN_PARAM_READONLY 1 /* noalias, readonly=True          */
#define ROCKE_BN_PARAM_WRITEONLY 2 /* noalias, writeonly=True         */
#define ROCKE_BN_PARAM_NOT_READONLY 3 /* noalias, readonly=False (explicit) */

/* b.param(name, PtrType(elem, "global"), noalias=True, <access>, align=<align>);
 * align 0 leaves the alignment unset. */
rocke_value_t* rocke_bn_ptr_param(
    rocke_ir_builder_t* b, const char* name, const rocke_type_t* elem, int access, int align);

/* Plain fp32 add for rocke_tree_reduce (Python ``b.fadd``). */
rocke_value_t*
    rocke_bn_fadd_combine(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c, void* user);

/* Python _wave_size_error: true when wave_size is a positive power of two.
 * Every BatchNorm2D validator refuses a spec for which this is false. */
static inline int rocke_bn_wave_size_ok(int w)
{
    return w > 0 && (w & (w - 1)) == 0;
}

/* Python _bwd_coefs(b, s1, s2, scale, mean, invstd, m_f): out = (a, k2, k3)
 * with dx = a * dy - k2 * (x - mean) - k3. */
void rocke_bn_bwd_coefs(rocke_ir_builder_t* b,
                        rocke_value_t* s1,
                        rocke_value_t* s2,
                        rocke_value_t* scale,
                        rocke_value_t* mean,
                        rocke_value_t* invstd,
                        rocke_value_t* m_f,
                        rocke_value_t* out[3]);

/* Per-CTA ranges of the dynamic split kernels (Python _split_dyn_geometry);
 * shared by the forward split (batchnorm2d.cpp) and backward split
 * (batchnorm2d_bwd.cpp) builders. */
typedef struct rocke_bn_split_geom
{
    rocke_value_t *sh_id, *sn_id, *hw_lo, *seg_len, *n_lo, *n_hi, *n_cnt;
} rocke_bn_split_geom_t;

void rocke_bn_split_dyn_geometry(rocke_ir_builder_t* b,
                                 rocke_value_t* p_n,
                                 rocke_value_t* p_nch,
                                 rocke_value_t* p_hwch,
                                 rocke_value_t* HW,
                                 rocke_bn_split_geom_t* g);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_BATCHNORM2D_BWD_INTERNAL_H */
