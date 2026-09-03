# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  if(FMHA_FWD_GFX1250_D192_TDM_ESM2 AND fwd_blob_name MATCHES
     "^fmha_fwd_d192_bf16_.*_qr_tdm_d192_v128_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  if(FMHA_FWD_GFX1250_D192_BATCH_NMASK_WWM_FAST AND fwd_blob_name MATCHES
     "^fmha_fwd_d192_bf16_batch_.*_qr_tdm_d192_v128_.*_nmask_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -wwm-regalloc=fast)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
