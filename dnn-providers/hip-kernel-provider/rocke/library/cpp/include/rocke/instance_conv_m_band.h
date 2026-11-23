/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C port of library/kernels/conv/conv_m_band.py: the output-plane band
 * order of a conv's M tiles (m_band), shared by the fwd / dgrad / wgrad
 * implicit-GEMM builders. Byte-identical to the Python emitter.
 */
#ifndef ROCKE_INSTANCE_CONV_M_BAND_H
#define ROCKE_INSTANCE_CONV_M_BAND_H

#include "rocke/ir.h"

/* emit_m_band_order(b, mt, num_m_tiles=, plane=, tile_m=, m_band=): the M
 * tile `mt` (logical, i32) mapped to its band-order tile. `tile_m_val` NULL
 * takes the build-time `tile_m`; otherwise it is the runtime tile width (e.g. a
 * split-K slice's). A bijection of [0, num_m_tiles). */
rocke_value_t* rocke_conv_emit_m_band_order(rocke_ir_builder_t* b,
                                            rocke_value_t* mt,
                                            rocke_value_t* num_m_tiles,
                                            rocke_value_t* plane,
                                            int tile_m,
                                            rocke_value_t* tile_m_val,
                                            int m_band);

#endif /* ROCKE_INSTANCE_CONV_M_BAND_H */
