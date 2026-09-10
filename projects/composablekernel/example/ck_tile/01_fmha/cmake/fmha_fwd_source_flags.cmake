# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  set(family_esm2 ${FMHA_FWD_GFX1250_D192_TDM_ESM2})
  if(DEFINED FMHA_FWD_GFX1250_TDM_V128_ESM2)
    set(family_esm2 ${FMHA_FWD_GFX1250_TDM_V128_ESM2})
  endif()
  if(family_esm2 AND
     (fwd_blob_name MATCHES "^fmha_fwd_d192_bf16_.*_qr_tdm_d192_v128_.*_gfx125\\.cpp$" OR
      fwd_blob_name MATCHES "^fmha_fwd_d128_bf16_.*_qr_tdm_v128_.*_gfx125\\.cpp$"))
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
