/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_dgrad_interior.h
 *
 * C++ port of library/kernels/conv/conv_dgrad_interior.py: interior-only dX
 * for halo-form dgrad (KernelIdentity.dx_interior / dx_interior_axes, the
 * x-halo W-column store mask view) and the halo shell zero-fill kernel
 * (DgradHaloZeroSpec + build_dgrad_halo_zero).
 *
 * dx_interior and dx_interior_axes are launch-time fields: the dgrad binary is
 * unchanged, the launch binds a view problem, a moved dX origin and the full
 * tensor's dX strides (or, for pp_boxes kernels, interior-only records, which
 * the host-side record builders in Python produce). These helpers are the
 * host-side algebra of those launches and the launch rules the Python
 * identity_supports_problem applies, with the same reason strings.
 *
 * Zero-fill kernel signature: dx (i32*, writeonly), dx_bytes, Di, Do, Hi, Ho,
 * ru, hd, hh, e, L1, L2, L3, n1, n2, u1, u2, u3 (i32).
 */
#ifndef ROCKE_INSTANCE_CONV_DGRAD_INTERIOR_H
#define ROCKE_INSTANCE_CONV_DGRAD_INTERIOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/instance_conv_implicit_gemm_dgrad.h" /* rocke_dgrad_conv_spec_t */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* dx_interior modes (Python DXI_SKIP / DXI_ZERO). */
#define ROCKE_DXI_SKIP 1
#define ROCKE_DXI_ZERO 2
/* dx_interior_axes bits (0 = all; Python DXA_W / DXA_H / DXA_D). */
#define ROCKE_DXA_W 1
#define ROCKE_DXA_H 2
#define ROCKE_DXA_D 4
#define ROCKE_DXA_ALL 7

/* halo_widths(p, axes): (hd, hh, hw), (F - 1) / 2 per axis, 0 on an axis
 * `axes` leaves out (hd = 0 in 2-D). */
void rocke_dgrad_halo_widths(const rocke_conv_problem_t* p, int axes, int out[3]);

/* dx_interior_problem_ok(p, mask): is p a halo-form problem on the axes of
 * mask (pad 0, odd filter), stride 1 / dilation 1 on all, groups 1? */
bool rocke_dgrad_dx_interior_problem_ok(const rocke_conv_problem_t* p,
                                        int mask,
                                        char* reason,
                                        size_t reason_cap);

/* interior_view_problem(p, keep_w, axes). */
rocke_conv_problem_t
    rocke_dgrad_interior_view_problem(const rocke_conv_problem_t* p, bool keep_w, int axes);

/* The dX binding of a view launch: the element offset of the view's origin
 * in the full dX and the dX stride kernargs (p_dX_stride_n / _di / _hi / _wi;
 * stride_di is 0 for a 2-D problem, which has no such kernarg). */
typedef struct rocke_dgrad_dx_view
{
    int64_t offset;
    int64_t stride_n;
    int64_t stride_di;
    int64_t stride_hi;
    int64_t stride_wi;
} rocke_dgrad_dx_view_t;

/* interior_view_dx(p, keep_w, axes). */
void rocke_dgrad_interior_view_dx(const rocke_conv_problem_t* p,
                                  bool keep_w,
                                  int axes,
                                  rocke_dgrad_dx_view_t* out);

/* wmask_view(p): xh_dx_wmask on a pW = (X-1)/2 problem -> the W-extended
 * pW = 0 view, its (negative) dX origin offset and the dX strides. */
void rocke_dgrad_wmask_view(const rocke_conv_problem_t* p,
                            rocke_conv_problem_t* view,
                            rocke_dgrad_dx_view_t* out);

/* The dx_interior launch rules of identity_supports_problem for a dgrad
 * binary of spec `s` (before the view / record problem is checked against the
 * binary itself): dx_interior mode, tap-outer loop, split-K only as
 * pp_box_det, no stream-K, the axes mask, xh_dx_wmask needs W, and the
 * problem rule. dx_interior == 0 with s->xh_dx_wmask checks the W-masked
 * pW = (X-1)/2 launch instead. Reason strings are Python's. */
bool rocke_dgrad_dx_interior_launch_ok(const rocke_dgrad_conv_spec_t* s,
                                       int dx_interior,
                                       int dx_interior_axes,
                                       const rocke_conv_problem_t* p,
                                       char* reason,
                                       size_t reason_cap);

/* ---- the shell zero-fill (dx_interior 2) ---- */

/* Mirror of Python DgradHaloZeroSpec. */
typedef struct rocke_dgrad_halo_zero_spec
{
    int block; /* default 256 */
    const char* name; /* default "conv_dgrad_halo_zero" */
} rocke_dgrad_halo_zero_spec_t;

rocke_dgrad_halo_zero_spec_t rocke_dgrad_halo_zero_spec_default(void);

rocke_status_t rocke_dgrad_halo_zero_kernel_name(const rocke_dgrad_halo_zero_spec_t* spec,
                                                 char* out,
                                                 size_t out_cap);

rocke_kernel_def_t* rocke_build_dgrad_halo_zero(rocke_ir_builder_t* b,
                                                const rocke_dgrad_halo_zero_spec_t* spec);

rocke_kernel_def_t* rocke_build_dgrad_halo_zero_new(rocke_ir_builder_t* b,
                                                    const rocke_dgrad_halo_zero_spec_t* spec);

/* halo_zero_units(p, elem_bytes, axes): the shell's three sections in 16-byte
 * units, in the kernarg order Di Do Hi Ho ru hd hh e L1 L2 L3 u1 u2 u3.
 * ROCKE_ERR_VALUE (Python ValueError) when C * elem_bytes % 16 != 0. */
#define ROCKE_DGRAD_HALO_ZERO_UNITS 14
rocke_status_t rocke_dgrad_halo_zero_units(const rocke_conv_problem_t* p,
                                           int elem_bytes,
                                           int axes,
                                           int64_t out[ROCKE_DGRAD_HALO_ZERO_UNITS]);

/* halo_zero_launch(...): the scalar kernargs after dx / dx_bytes in signature
 * order (Di Do Hi Ho ru hd hh e L1 L2 L3 n1 n2 u1 u2 u3), the grid and the
 * block. */
#define ROCKE_DGRAD_HALO_ZERO_SCALARS 16
rocke_status_t rocke_dgrad_halo_zero_launch(const rocke_conv_problem_t* p,
                                            int elem_bytes,
                                            int axes,
                                            const rocke_dgrad_halo_zero_spec_t* spec,
                                            int64_t values[ROCKE_DGRAD_HALO_ZERO_SCALARS],
                                            int grid[3],
                                            int block[3]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_CONV_DGRAD_INTERIOR_H */
