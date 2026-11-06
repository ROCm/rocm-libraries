/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_wmma_tdm_internal.h -- PRIVATE TDM staging helpers
 * shared by the C++ ports of the wave32 WMMA forward families that stage
 * tiles with rank-3 TDM transfers into an LDS ring (conv_direct_fwd_wmma.cpp,
 * conv_fwd_small_wmma.cpp, conv_direct_fwd_gstream_wmma.cpp,
 * conv_direct_fwd_tapouter_wmma.cpp), as their Python ports all import these
 * from conv_direct_fwd_wmma.py. Declares no public API.
 *
 *   Python (conv_direct_fwd_wmma.py)               C++ (this header)
 *   ---------------------------------------------   ------------------------------
 *   _tdm_descriptor_3d(b, global_ptr, lds_addr,     rocke_fwd_tdm_descriptor_3d
 *       dims, strides, tile0, tile1, tile2)
 *   _tdm_ring_wait(b, step, stages, n_steps)        rocke_fwd_tdm_ring_wait
 *   _lds_frag16(b, lds, base, off, const_off,       rocke_fwd_lds_frag16
 *       c0=, c8=, io_type=)
 *
 * The ops are emitted in the Python order (byte identity).
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_WMMA_TDM_INTERNAL_H
#define ROCKE_INSTANCE_CONV_FWD_WMMA_TDM_INTERNAL_H

#include <stdint.h>

#include "rocke/ir.h"

/* The five D# groups of a rank-3 TDM tile transfer of 2-byte elements:
 * dimension 0 is contiguous, (s1, s2) are the element strides of dimensions
 * 1 and 2, (d0, d1, d2) the extents left from global_ptr (the hardware
 * zero-fills past them); the tile lands packed [tile2][tile1][tile0] at the
 * i32 LDS byte address lds_addr. tile1_v, when not NULL, is a runtime tile1
 * (Python's ``tile1: Value``) used in place of tile1. Writes out[0..4]. */
void rocke_fwd_tdm_descriptor_3d(rocke_ir_builder_t* b,
                                 rocke_value_t* global_ptr,
                                 rocke_value_t* lds_addr,
                                 rocke_value_t* d0,
                                 rocke_value_t* d1,
                                 rocke_value_t* d2,
                                 rocke_value_t* s1,
                                 rocke_value_t* s2,
                                 int tile0,
                                 int tile1,
                                 rocke_value_t* tile2,
                                 rocke_value_t* out[5],
                                 rocke_value_t* tile1_v = NULL);

/* Wait until this wave's TDM transfer of ring step `step` has landed. With
 * `stages` ring slots the step's transfer is the oldest in flight; the
 * stages - 2 younger ones exist only while steps below `n_steps` remain to be
 * issued, so the wait leaves those in flight or none. */
void rocke_fwd_tdm_ring_wait(rocke_ir_builder_t* b,
                             rocke_value_t* step,
                             int stages,
                             rocke_value_t* n_steps);

/* One 16-element WMMA operand fragment from the [1, n] LDS buffer `lds`: the 8
 * elements at `base` (base + (off + const_off) when `off` is not NULL) and the
 * 8 after them. c0 / c8 are the caller's i32 constants 0 and 8. */
rocke_value_t* rocke_fwd_lds_frag16(rocke_ir_builder_t* b,
                                    rocke_value_t* lds,
                                    rocke_value_t* base,
                                    rocke_value_t* off,
                                    int64_t const_off,
                                    rocke_value_t* c0,
                                    rocke_value_t* c8,
                                    const rocke_type_t* io_type);

#endif /* ROCKE_INSTANCE_CONV_FWD_WMMA_TDM_INTERNAL_H */
