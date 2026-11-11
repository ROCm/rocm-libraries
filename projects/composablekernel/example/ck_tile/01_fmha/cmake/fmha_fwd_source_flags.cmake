# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  # Keep the compiler scheduling mode used by the validated qr_tdm_sched and
  # qr_tdm_pingpong instances. The qr_tdm_pingpong instances that store the LSE
  # keep the default: built in this mode, which leaves the ALU dependency checks
  # to the compiler, they produce intermittently wrong O and LSE values.
  if(fwd_blob_name MATCHES "^fmha_fwd_d(64|128|192)_(fp16|bf16)_.*_qr_tdm_(sched|pingpong_.*_nlse)(_.*)?_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  # Same scheduling mode for the paged-KV counterpart (qr_pagedkv_tdm_sched),
  # which currently only ships the hdim=128 geometry.
  if(fwd_blob_name MATCHES "^fmha_fwd_pagedkv_d128_(fp16|bf16)_.*_qr_pagedkv_tdm_sched_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
