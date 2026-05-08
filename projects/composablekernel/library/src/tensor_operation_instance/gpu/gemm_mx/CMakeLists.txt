# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ONLY MX_KERNELS
set(GEMM_MX_INSTANCES)

list(APPEND GEMM_MX_INSTANCES
        device_gemm_mx_xdl_f6_f6_f16/device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instance.cpp
        device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instance.cpp
        device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn_default_instance.cpp
        device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_mfma_mn_default_instance.cpp
        device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn_default_instance.cpp
        device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_km_nk_mn_default_instance.cpp
        device_gemm_mx_xdl_bf8_f8_f16/device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_default_instance.cpp
        device_gemm_mx_xdl_f4_f4_f16/device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn_default_instance.cpp
        device_gemm_mx_xdl_f4_f4_f16/device_gemm_mx_xdl_f4_f4_f16_mk_mfma_mn_default_instance.cpp
    )

# Temporary workaround: gfx13 currently fails template instantiation
# for several MX low-precision instance families in this library path.
if(SUPPORTED_GPU_TARGETS MATCHES "gfx13")
    list(REMOVE_ITEM GEMM_MX_INSTANCES
         device_gemm_mx_xdl_f6_f6_f16/device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instance.cpp
         device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instance.cpp)
endif()


set_source_files_properties(device_gemm_mx_xdl_f6_f6_f16/device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_mfma_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_km_nk_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_bf8_f8_f16/device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_f4_f4_f16/device_gemm_mx_xdl_f4_f4_f16_mk_nk_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")
set_source_files_properties(device_gemm_mx_xdl_f4_f4_f16/device_gemm_mx_xdl_f4_f4_f16_mk_mfma_mn_default_instance.cpp PROPERTIES COMPILE_OPTIONS ";-mllvm;-greedy-reverse-local-assignment=1")


add_instance_library(device_gemm_mx_instance ${GEMM_MX_INSTANCES})
