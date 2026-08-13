# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/caching_library_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/secure_env_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_stats_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_gtest.cpp
        # New coverage improvement tests
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_core_gemm_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_epilogue_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scaling_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_layout_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_extended_ops_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_advanced_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_error_handling_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_performance_gtest.cpp
        # New comprehensive multi-GPU tests
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_complete_implementations_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_fp8_int8_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_grouped_gemm_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_streams_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_communication_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_algorithm_selection_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_combined_features_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_large_scale_gtest.cpp
        # Parameter variation tests (systematic testing of all hipBLASLt parameters)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_matrix_dimensions_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_data_types_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_transpose_ops_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scalar_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_epilogue_ops_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_batching_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scaling_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_matrix_layout_variations_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_algorithm_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_auxiliary_outputs_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_stream_sync_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_device_params_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_edge_cases_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_leading_dim_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_problem_sizes_gtest.cpp
        # Real multi-GPU computation tests (distributed GEMM with CPU verification)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_compute_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_params_gtest.cpp
        # Comprehensive real multi-GPU tests covering all parameters
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_data_types_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_transpose_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_scalars_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_epilogue_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_combinations_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_k_partition_gtest.cpp
        # Complete coverage: dimensions, batching, leading dims/strides
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_dimensions_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_batching_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_leading_stride_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_matrix_coverage_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_real_fp8_int8_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_gradient_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_transformer_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_scaling_efficiency_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_fault_tolerance_gtest.cpp
        # Gap-filling tests (topology, heterogeneous, dynamic/adaptive patterns)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_topology_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_heterogeneous_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_dynamic_adaptive_gtest.cpp
        # Critical P0 gap tests (IMPLEMENTED - addresses critical production needs)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_memory_management_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_synchronization_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_ml_workloads_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_checkpoint_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_collectives_gtest.cpp
        # NEW: Critical ML workload gap tests for 100% coverage (2024)
        # Phase 1-2: BF16, Advanced Attention, KV Cache (P0 - Critical)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_bfloat16_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_advanced_attention_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_kv_cache_gtest.cpp
        # Phase 3-4: Sequence Parallelism, RCCL Integration (P0-P1)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_sequence_parallel_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_rccl_collectives_gtest.cpp
        # Phase 5: Modern Inference/Training Features (2024-2026) - Comprehensive Coverage
        # Week 1-2: Weight Quantization (GPTQ, AWQ, INT4/INT8, W4A16)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_weight_quantization_gtest.cpp
        # Week 3: Mixture of Experts (MoE) patterns (DeepSeek-V3, Mixtral, Gemini)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_moe_patterns_gtest.cpp
        # Week 5: Speculative Decoding (vLLM, TensorRT-LLM, 3× speedup)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_speculative_decoding_gtest.cpp
        # Week 6: ZeRO Optimizer Partitioning (DeepSpeed ZeRO-1/2/3, FSDP)
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_zero_optimizer_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk5_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ulp_gtest.cpp
)

# NOTE: secure_env_gtest.cpp (ROCM-26729) intentionally includes the header-only
# rocblaslt_secure_env.hpp by relative path rather than via a target include
# directory. Adding the internal rocblaslt/src/include dir to hipblaslt-test
# shadows the clients' own "utility.hpp" with the library's internal one and
# breaks the build of unrelated test sources, so it must not be added here.
