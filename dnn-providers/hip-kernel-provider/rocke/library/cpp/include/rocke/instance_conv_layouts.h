// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_conv_layouts.h -- conv-owned LDS layout helpers, C++ twin of the
 * library's kernels/common/_conv_layouts.py.
 *
 *   ConvKOuterFragmentReader.wave32_lane_offsets <-> rocke_conv_tr_lane_offsets32
 *
 * The reader's fragment() itself is rocke_conv_tr_frag
 * (instances/common/conv_implicit_gemm_conv_compute_phase.cpp). Only the conv
 * instances use these, so they live next to them, not in the generic helpers
 * (rocke/helper_rocke.helpers.layouts.h). The byte-identity gate holds the two
 * engines together: the emitted ops are part of every caller's IR.
 */
#ifndef ROCKE_INSTANCE_CONV_LAYOUTS_H
#define ROCKE_INSTANCE_CONV_LAYOUTS_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The wave32 ds_load_tr16_b128 lane map of a K-outer conv LDS tile:
 *   *col_out  = mn_base + ((lane % 16) / 8) * 8
 *   *row0_out = k_base  + (lane / 16) * n + lane % 8
 * mn_base / k_base may be NULL: the offsets then come back relative to the
 * tile origin (no add emitted), for callers that address a flat LDS buffer.
 * Emits, in this exact order: const 16, const 8, lane % 16, / 8, * 8,
 * [mn_base +], lane / 16, const n, *, lane % 8, +, [k_base +].
 * rocke_conv_tr_frag and the direct wgrad builders call it. The wave32 dgrad
 * WMMA builders still compute the same map inline: conv_direct_dgrad_wmma.cpp,
 * and instance_conv_dgrad_stem_wmma_internal.hpp for the patch and small-C
 * stem builders. */
void rocke_conv_tr_lane_offsets32(rocke_ir_builder_t* b,
                                  rocke_value_t* lane,
                                  int n,
                                  rocke_value_t* mn_base,
                                  rocke_value_t* k_base,
                                  rocke_value_t** col_out,
                                  rocke_value_t** row0_out);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_INSTANCE_CONV_LAYOUTS_H */
