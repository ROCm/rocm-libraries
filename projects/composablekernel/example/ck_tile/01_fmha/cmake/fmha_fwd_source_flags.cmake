# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  # Keep the compiler scheduling mode used by the validated scheduled instances.
  if(fwd_blob_name MATCHES "^fmha_fwd_d(64|128|192)_(fp16|bf16)_.*_qr_tdm_sched_.*_gfx125\\.cpp$"
     OR fwd_blob_name MATCHES "^fmha_fwd_d[0-9]+_[a-z0-9]+_.*_qr_tdm_fp8_sched_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  # Break large PHIs so the blockscale loop keeps its loop-carried K group in dwords.
  if(fwd_blob_name MATCHES "^fmha_fwd_d[0-9]+_[a-z0-9]+_.*_qr_tdm_fp8_sched_.*_blockscale_ntrload_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-codegenprepare-force-break-large-phis)
  endif()
  # D128 MX instances keep the scale copy's uniform branches free of StructurizeCFG flow flags.
  if(fwd_blob_name MATCHES "^fmha_fwd_d128_mxfp8[a-z0-9]*_.*_qr_tdm_fp8_sched_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -structurizecfg-skip-uniform-regions)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
