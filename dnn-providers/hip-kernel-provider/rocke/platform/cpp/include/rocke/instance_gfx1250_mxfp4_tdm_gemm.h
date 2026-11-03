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
     * matching Python step is mirrored. */
    bool vector_epilogue;
    bool local_read_prefetch;
    bool wave_parity_tdm;
    bool scale_tdm;
    bool lds_pad;
    bool lds_segment_interleave;
    bool init_c_wmma;
    bool cluster_dim_set;
    int cluster_dim_x;
    int cluster_dim_y;
    int cluster_dim_z;
    bool tdm_multicast;
    bool prefetch_gl2;
    bool tile_refill_mid;
    bool tile_read_ahead;
    bool cluster_signal_wmmas_set;
    int cluster_signal_wmmas;
    bool gl2_prefetch_early;
    bool cluster_signal_wave0;
    bool tdm_desc_hoist;
    bool wmma_n_outer;
    bool coarse_ds_wait;
    bool wmma_arb_stall_off;
    bool kernarg_preload;
    bool cluster_aid_remap;
    bool epilogue_nan_clamp;
    bool epilogue_pk8_cvt;
    bool epilogue_inst_prefetch_set;
    int epilogue_inst_prefetch;
    bool loop_unroll2;
    bool expert_scheduling;
    bool source_order;
} rocke_mxfp4_tdm_gemm_spec_t;

rocke_mxfp4_tdm_gemm_spec_t rocke_mxfp4_tdm_gemm_spec_default(void);
rocke_kernel_def_t* rocke_build_mxfp4_tdm_gemm(rocke_ir_builder_t* b,
                                               const rocke_mxfp4_tdm_gemm_spec_t* spec,
                                               const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1250_MXFP4_TDM_GEMM_H */
