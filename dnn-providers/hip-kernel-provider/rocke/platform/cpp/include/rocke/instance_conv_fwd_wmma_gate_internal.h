/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_wmma_gate_internal.h -- PRIVATE capability gate
 * shared by the C++ ports of the wave32 WMMA forward convolutions
 * (conv_direct_fwd_wmma.cpp, conv_fwd_small_wmma.cpp). Emits no IR and
 * declares no public API.
 *
 *   Python (conv_fwd_wmma_gate.py)       C++ (this header)
 *   ----------------------------------   ----------------------------------
 *   WMMA_M / WMMA_N / WMMA_K             ROCKE_FWD_WMMA_M / _N / _K
 *   wmma_ab_dtype(dtype)                 rocke_wmma_ab_dtype
 *   wmma_fwd_gate(p, arch, name, wave)   rocke_wmma_fwd_gate
 *
 * The checks run in the Python order and write the same rejection messages.
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_WMMA_GATE_INTERNAL_H
#define ROCKE_INSTANCE_CONV_FWD_WMMA_GATE_INTERNAL_H

#include <stddef.h>

#include "rocke/helper_rocke.core.arch.h" /* rocke_archtarget_t */
#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */

/* The fp32-accumulating WMMA atom every forward family issues: M x N x K. */
#define ROCKE_FWD_WMMA_M 16
#define ROCKE_FWD_WMMA_N 16
#define ROCKE_FWD_WMMA_K 32

/* wmma_ab_dtype: the WMMA operand dtype of a conv dtype (NULL reads as fp16). */
const char* rocke_wmma_ab_dtype(const char* dtype);

/* wmma_fwd_gate: whether kernel `name` can run `p` on `arch` at the spec's
 * `wave_size`. On success stores the resolved target in `*target_out` and
 * returns true, so callers need not resolve it again; otherwise writes the
 * reason and returns false. */
bool rocke_wmma_fwd_gate(const rocke_direct_conv_problem_t* p,
                         const char* arch,
                         const char* name,
                         int wave_size,
                         const rocke_archtarget_t** target_out,
                         char* reason,
                         size_t reason_cap);

#endif /* ROCKE_INSTANCE_CONV_FWD_WMMA_GATE_INTERNAL_H */
