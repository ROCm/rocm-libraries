/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1250_mxfp4_tdm_gemm.h -- C99 port of the gfx1250 MX FP4
 * TDM/LDS GEMM instance builder.
 *
 * Python (instances/gfx1250/mxfp4_tdm_gemm.py)    C99 (this header)
 * ----------------------------------------------  --------------------------------
 * MxFp4TdmGemmSpec                                rocke_mxfp4_tdm_gemm_spec_t
 * build_mxfp4_tdm_gemm(spec, arch)                rocke_build_mxfp4_tdm_gemm(...)
 */
#ifndef ROCKE_INSTANCE_GFX1250_MXFP4_TDM_GEMM_H
#define ROCKE_INSTANCE_GFX1250_MXFP4_TDM_GEMM_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_mxfp4_tdm_gemm_spec
{
    const char* name;
    int M;
    int N;
    int K;
    int tile_m;
    int tile_n;
    int tile_k;
    int warp_m;
    int warp_n;
    int warp_k;
    int warp_tile_m;
    int warp_tile_n;
    int warp_tile_k;
    int block_k;
    bool tdm_lds;
    bool tdm_prefetch;
    int tdm_prefetch_depth;
    /* Step 6 flags (default off). Non-default values are rejected until the
     * matching Python substep is mirrored. */
    bool local_read_prefetch;
    bool prefetch_gl2;
    bool scale_double_buffer;
    bool wave_separated_tdm;
    bool scale_tdm;
    bool lds_pad;
    bool lds_segment_interleave;
    bool lds_segment_b_rotate;
    bool tdm_plus_lds_buf;
    bool init_c_wmma;
    bool cluster_dim_set;
    int cluster_dim_x;
    int cluster_dim_y;
    int cluster_dim_z;
} rocke_mxfp4_tdm_gemm_spec_t;

rocke_mxfp4_tdm_gemm_spec_t rocke_mxfp4_tdm_gemm_spec_default(void);
rocke_kernel_def_t* rocke_build_mxfp4_tdm_gemm(rocke_ir_builder_t* b,
                                               const rocke_mxfp4_tdm_gemm_spec_t* spec,
                                               const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_MXFP4_TDM_GEMM_H */
