# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(ck_tile_fmha_fwd_get_source_compile_options fwd_blob_name out_var)
  set(compile_options)
  # Keep the compiler scheduling mode used by the validated scheduled instances.
  if(fwd_blob_name MATCHES "^fmha_fwd_d(64|128|192)_(fp16|bf16)_.*_qr_tdm_sched_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
  endif()
  # The fp8 family (fp8, fp8bf16, fp8fp32, mxfp8, mxfp8bf16) qr_tdm instances use the same mode.
  if(fwd_blob_name MATCHES "^fmha_fwd_d[0-9]+_(mx)?fp8(bf16|fp32)?_.*_qr_tdm_.*_gfx125\\.cpp$")
    list(APPEND compile_options -mllvm -amdgpu-expert-scheduling-mode)
    # gfx1250 enables packed fp32 through packed-fp32-single-sgpr-ops since clang 24, which the
    # kernel's no-packed-fp32-ops attribute does not cover. Disabled for the whole TU so every
    # function keeps the same features and still inlines.
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 24)
      list(APPEND compile_options -Xclang -target-feature -Xclang -packed-fp32-single-sgpr-ops)
    endif()
  endif()
  set(${out_var} "${compile_options}" PARENT_SCOPE)
endfunction()
