# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Reuse small example executables where they provide useful end-to-end coverage
# that is not otherwise represented by a CK-Tile GTest. Each alias is bounded
# to one validated configuration, while repeated aliases share one binary.

if(TARGET tile_example_batched_contraction)
    add_ck_tile_category_test(
        test_ck_tile_batched_contraction_basic_quick
        tile_example_batched_contraction
        -g_dims=2 -m_dims=128 -n_dims=128 -k_dims=64 -num_d=0
        -v=1 -warmup=0 -repeat=1 -log=0
    )
    add_ck_tile_category_test(
        test_ck_tile_batched_contraction_multidim_quick
        tile_example_batched_contraction
        -g_dims=2,2 -m_dims=8,16 -n_dims=16,8 -k_dims=64,2 -num_d=2
        -v=1 -warmup=0 -repeat=1 -log=0
    )
    rocm_install(TARGETS tile_example_batched_contraction COMPONENT tests)
endif()

if(TARGET tile_example_streamk_gemm_basic)
    add_ck_tile_category_test(
        test_ck_tile_streamk_fp16_quick
        tile_example_streamk_gemm_basic
        -m=512 -n=512 -k=1024 -prec=fp16 -reduction_strategy=atomic
        -persistent_dp=0 -v=1 -warmup=0 -repeat=1 -flush_cache=false
    )
    add_ck_tile_category_test(
        test_ck_tile_streamk_bf16_persistent_quick
        tile_example_streamk_gemm_basic
        -m=768 -n=512 -k=1024 -prec=bf16 -reduction_strategy=linear
        -persistent_dp=1 -v=1 -warmup=0 -repeat=1 -flush_cache=false
    )
    add_ck_tile_category_test(
        test_ck_tile_streamk_fp8_quick
        tile_example_streamk_gemm_basic
        -m=512 -n=768 -k=1024 -prec=fp8 -reduction_strategy=linear
        -persistent_dp=0 -v=1 -warmup=0 -repeat=1 -flush_cache=false
    )
    rocm_install(TARGETS tile_example_streamk_gemm_basic COMPONENT tests)
endif()

set(_grouped_conv_quick_args
    -g=2 -n=2 -k=32 -c=32 -h=16 -w=16 -y=3 -x=3
    -stride_h=1 -stride_w=1 -lpad_h=1 -lpad_w=1 -rpad_h=1 -rpad_w=1
    -prec=fp16 -v=1 -warmup=0 -repeat=1 -init=1
)
if(TARGET tile_example_grouped_conv_fwd)
    add_ck_tile_category_test(
        test_ck_tile_grouped_conv_fwd_quick
        tile_example_grouped_conv_fwd
        ${_grouped_conv_quick_args}
    )
    rocm_install(TARGETS tile_example_grouped_conv_fwd COMPONENT tests)
endif()
if(TARGET tile_example_grouped_conv_bwd_data)
    add_ck_tile_category_test(
        test_ck_tile_grouped_conv_bwd_data_quick
        tile_example_grouped_conv_bwd_data
        ${_grouped_conv_quick_args}
    )
    rocm_install(TARGETS tile_example_grouped_conv_bwd_data COMPONENT tests)
endif()
if(TARGET tile_example_grouped_conv_bwd_weight)
    add_ck_tile_category_test(
        test_ck_tile_grouped_conv_bwd_weight_quick
        tile_example_grouped_conv_bwd_weight
        ${_grouped_conv_quick_args} -split_k=2
    )
    rocm_install(TARGETS tile_example_grouped_conv_bwd_weight COMPONENT tests)
endif()

if(TARGET tile_example_flatmm_basic)
    add_ck_tile_category_test(
        test_ck_tile_flatmm_fp16_quick
        tile_example_flatmm_basic
        -m=128 -n=256 -k=128 -prec=fp16 -wave_tile=16 -split_k=1
        -persistent=0 -scale=0 -v=1 -warmup=0 -repeat=1 -init=1
    )
    add_ck_tile_category_test(
        test_ck_tile_flatmm_fp8_persistent_quick
        tile_example_flatmm_basic
        -m=256 -n=256 -k=256 -prec=fp8 -wave_tile=16 -split_k=1
        -persistent=1 -scale=0 -v=1 -warmup=0 -repeat=1 -init=1
    )
    rocm_install(TARGETS tile_example_flatmm_basic COMPONENT tests)
endif()

if(TARGET tile_example_jenga_sparse_attn)
    add_ck_tile_category_test(
        test_ck_tile_jenga_sparse_attn_quick
        tile_example_jenga_sparse_attn
        -v=1 -b=1 -h=2 -h_k=1 -s=256 -s_k=384 -d=128 -d_v=128
        -block_size=128 -sparsity=0.5 -prec=fp16 -iperm=0 -operm=0
        -seed=42 -warmup=0 -repeat=1 -kname=0
    )
    rocm_install(TARGETS tile_example_jenga_sparse_attn COMPONENT tests)
endif()

if(TARGET tile_example_vsa_sparse_attn)
    add_ck_tile_category_test(
        test_ck_tile_vsa_sparse_attn_quick
        tile_example_vsa_sparse_attn
        -v=1 -b=1 -h=2 -h_k=1 -s=384 -s_k=256 -d=128 -d_v=128
        -block_size=128 -sparsity=0.75 -prec=bf16 -iperm=1 -operm=1
        -seed=43 -warmup=0 -repeat=1 -kname=0
    )
    rocm_install(TARGETS tile_example_vsa_sparse_attn COMPONENT tests)
endif()

# The full SageAttention example emits 193 files. The quick target keeps just
# four FP8/BF16 per-warp kernels: batch/group crossed with masked/unmasked.
set(_sage_quick_targets ${SUPPORTED_GPU_TARGETS})
list(FILTER _sage_quick_targets INCLUDE REGEX "gfx942|gfx950")
if(_sage_quick_targets)
    set(_sage_source_dir "${PROJECT_SOURCE_DIR}/example/ck_tile/49_sageattention")
    set(_sage_codegen_dir "${CMAKE_CURRENT_BINARY_DIR}/sageattn_quick_codegen")
    set(_sage_blob_list "${_sage_codegen_dir}/blob_list.txt")
    list(JOIN _sage_quick_targets , _sage_targets_arg)
    set(_sage_codegen_args
        "${_sage_source_dir}/generate.py"
        --targets "${_sage_targets_arg}"
        --api fwd
        --optdim 128
        --filter
        "sageattn_fwd_d128_fp8bf16_*_*_vr_psskddv_*_perwarp"
    )
    file(GLOB_RECURSE _sage_codegen_scripts CONFIGURE_DEPENDS
        "${_sage_source_dir}/generate.py"
        "${_sage_source_dir}/codegen/*.py"
    )
    file(MAKE_DIRECTORY "${_sage_codegen_dir}")
    file(REMOVE "${_sage_blob_list}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${_sage_codegen_args}
                --list_blobs "${_sage_blob_list}"
        RESULT_VARIABLE _sage_codegen_result
    )
    if(_sage_codegen_result AND NOT _sage_codegen_result EQUAL 0)
        message(FATAL_ERROR "Failed to list quick SageAttention kernels")
    endif()
    file(STRINGS "${_sage_blob_list}" _sage_quick_blobs)

    add_custom_command(
        OUTPUT ${_sage_quick_blobs}
        COMMAND ${Python3_EXECUTABLE} ${_sage_codegen_args}
                --output_dir "${_sage_codegen_dir}"
        DEPENDS ${_sage_codegen_scripts}
        COMMENT "Generate quick SageAttention kernels"
        VERBATIM
    )
    add_library(tile_sageattn_fwd_quick_instances OBJECT EXCLUDE_FROM_ALL)
    target_sources(tile_sageattn_fwd_quick_instances PRIVATE ${_sage_quick_blobs})
    set_source_files_properties(${_sage_quick_blobs} PROPERTIES LANGUAGE HIP)
    set_property(
        TARGET tile_sageattn_fwd_quick_instances
        PROPERTY HIP_ARCHITECTURES ${_sage_quick_targets}
    )
    target_include_directories(
        tile_sageattn_fwd_quick_instances PRIVATE "${_sage_source_dir}"
    )
    target_compile_options(
        tile_sageattn_fwd_quick_instances PRIVATE
        -Wno-undefined-func-template -Wno-float-equal
        -fgpu-flush-denormals-to-zero
    )
    if(CK_USE_OCP_FP8)
        target_compile_options(
            tile_sageattn_fwd_quick_instances PRIVATE -DCK_TILE_USE_OCP_FP8
        )
    endif()

    add_executable(
        tile_example_sageattn_fwd_quick EXCLUDE_FROM_ALL
        "${_sage_source_dir}/example_sageattn_fwd.cpp"
    )
    target_include_directories(
        tile_example_sageattn_fwd_quick PRIVATE "${_sage_source_dir}"
    )
    target_link_libraries(
        tile_example_sageattn_fwd_quick PRIVATE tile_sageattn_fwd_quick_instances
    )
    target_compile_options(
        tile_example_sageattn_fwd_quick PRIVATE
        -Wno-undefined-func-template -Wno-float-equal
        -fgpu-flush-denormals-to-zero
    )
    if(CK_USE_OCP_FP8)
        target_compile_options(
            tile_example_sageattn_fwd_quick PRIVATE -DCK_TILE_USE_OCP_FP8
        )
    endif()
    set_property(
        TARGET tile_example_sageattn_fwd_quick
        PROPERTY HIP_ARCHITECTURES ${_sage_quick_targets}
    )

    add_ck_tile_category_test(
        test_ck_tile_sageattn_batch_quick
        tile_example_sageattn_fwd_quick
        -v=1 -prec=fp8bf16 -qscale=3 -mode=0 -b=2 -h=2 -h_k=1
        -d=128 -d_v=128 -s=55 -s_k=256 -mask=1
        -iperm=0 -operm=0 -vlayout=r -warmup=0 -repeat=1 -kname=0
    )
    add_ck_tile_category_test(
        test_ck_tile_sageattn_group_quick
        tile_example_sageattn_fwd_quick
        -v=1 -prec=fp8bf16 -qscale=3 -mode=1 -b=3 -h=2 -h_k=1
        -d=128 -d_v=128 -s=50,60,40 -s_k=128,256,192 -mask=0
        -iperm=0 -operm=0 -vlayout=r -warmup=0 -repeat=1 -kname=0
    )
    rocm_install(TARGETS tile_example_sageattn_fwd_quick COMPONENT tests)
endif()

# MoE smoothquant previously pulled twenty instance files. Two small instance
# translation units cover FP16/BF16 input and FP8/INT8 output for h <= 128.
set(_moe_sq_quick_targets ${SUPPORTED_GPU_TARGETS})
list(FILTER _moe_sq_quick_targets INCLUDE REGEX "gfx942|gfx950")
if(_moe_sq_quick_targets)
    set(_moe_sq_source_dir "${PROJECT_SOURCE_DIR}/example/ck_tile/14_moe_smoothquant")
    add_executable(
        tile_example_moe_smoothquant_quick EXCLUDE_FROM_ALL
        "${_moe_sq_source_dir}/moe_smoothquant.cpp"
        "${_moe_sq_source_dir}/instances/moe_smoothquant_fwd_api.cpp"
        "${_moe_sq_source_dir}/instances/moe_smoothquant_fp16_n64_n128_instance.cpp"
        "${_moe_sq_source_dir}/instances/moe_smoothquant_bf16_n64_n128_instance.cpp"
    )
    set_source_files_properties(
        "${_moe_sq_source_dir}/moe_smoothquant.cpp"
        "${_moe_sq_source_dir}/instances/moe_smoothquant_fwd_api.cpp"
        "${_moe_sq_source_dir}/instances/moe_smoothquant_fp16_n64_n128_instance.cpp"
        "${_moe_sq_source_dir}/instances/moe_smoothquant_bf16_n64_n128_instance.cpp"
        PROPERTIES LANGUAGE HIP
    )
    target_include_directories(
        tile_example_moe_smoothquant_quick PRIVATE "${_moe_sq_source_dir}"
    )
    target_compile_options(
        tile_example_moe_smoothquant_quick PRIVATE
        -Wno-undefined-func-template -Wno-float-equal
    )
    if(CK_USE_OCP_FP8)
        target_compile_options(
            tile_example_moe_smoothquant_quick PRIVATE -DCK_TILE_USE_OCP_FP8
        )
    endif()
    set_property(
        TARGET tile_example_moe_smoothquant_quick
        PROPERTY HIP_ARCHITECTURES ${_moe_sq_quick_targets}
    )

    add_ck_tile_category_test(
        test_ck_tile_moe_smoothquant_fp16_fp8_quick
        tile_example_moe_smoothquant_quick
        -t=32 -h=128 -e=8 -k=2 -v=1 -kname=0
        -prec_i=fp16 -prec_o=fp8 -warmup=0 -repeat=1
    )
    add_ck_tile_category_test(
        test_ck_tile_moe_smoothquant_bf16_int8_quick
        tile_example_moe_smoothquant_quick
        -t=17 -h=64 -e=4 -k=2 -stride=80 -v=1 -kname=0
        -prec_i=bf16 -prec_o=int8 -warmup=0 -repeat=1
    )
    rocm_install(TARGETS tile_example_moe_smoothquant_quick COMPONENT tests)
endif()
