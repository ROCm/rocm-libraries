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
  # CR's operand prefetch and barrier placement were tuned with these
  # scheduling costs. Scope them to this pipeline's translation units.
  if(fwd_blob_name MATCHES "^fmha_fwd_d(32|64|96|128|160|192)_(fp16|bf16)_.*_qr_tdm_cr_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode
      -mllvm -amdgpu-sched-strategy=max-ilp
      -mllvm -amdgpu-barrier-signal-wait-latency=32)
  endif()
  # D64 and these D32 geometries favor packed register allocation in
  # same-device measurements. Keep this setting tied to their tile shapes.
  if(fwd_blob_name MATCHES "^fmha_fwd_d64_(fp16|bf16)_.*_qr_tdm_cr_.*_gfx125\\.cpp$" OR
     fwd_blob_name MATCHES "^fmha_fwd_d32_(fp16|bf16)_.*_b(64x128|128x(64|128))x.*_qr_tdm_cr_.*_gfx125\\.cpp$")
    list(APPEND compile_options -Xclang -target-feature -Xclang -real-true16)
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
