# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hipblaslt-test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/caching_library_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/contraction_problem_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hip_solution_adapter_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/secure_env_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/arch_revision_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/adaptive_sm_count_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/efficiency_monitor_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/math_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/scale_buffer_size_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/heuristic_workspace_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/grouped_matmul_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/data_type_names_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_stats_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_collective_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/candidate_search_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/candidate_preparation_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matrix_transform_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/hipblaslt_gtest_ext_op.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk5_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk_multistream_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/streamk_null_stream_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/synchronizer_size_check_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/row_uniformity_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/algo_order_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/contraction_problem_update_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tuning_cache_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tuning_store_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tuning_tune_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tuning_scratch_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ulp_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/arguments_a2a_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/unit_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/algo_errors_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fast_check_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/w4a16_layout_gtest.cpp
)

if(NOT WIN32)
    target_sources(hipblaslt-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/multi_gpu_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cotenant_gtest.cpp
    )
endif()

# The online tuner's bookkeeping, against a private copy of the tuner built
# from OnlineTuner.cpp and the onlineTunerConfig() the test file defines. It is
# compiled in here, not built as its own binary, so hipBLASLt keeps one ctest
# entry and test_categories.yaml selects the cases with "*OnlineTuner.*", for
# the reason given on hipblaslt-test-tuning-store in clients/CMakeLists.txt.
# The private copy does not meet the library's: a shared libhipblaslt has
# hidden visibility and exports neither.
#
# Where hipblaslt-test links the library's own objects instead, the two copies
# would be duplicate definitions, and one process can hold only one tuner.
# Coverage builds and static builds do that, and leave these cases out; the
# coverage job's filter does not select them either.
if(HIPBLASLT_BUILD_SHARED_LIBS AND NOT HIPBLASLT_ENABLE_COVERAGE)
    target_sources(hipblaslt-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/online_tuner_test.cpp
        ${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/OnlineTuner.cpp
    )
endif()

if(HIPBLASLT_ENABLE_GEMM_A2A_FUSION AND NOT WIN32)
    target_sources(hipblaslt-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_a2a_gtest.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/fused_a2a_multiprocess_gtest.cpp
    )
endif()

if(HIPBLASLT_ENABLE_LOCAL_KERNELS)
    target_sources(hipblaslt-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/local_kernel_library_gtest.cpp
    )
endif()

# NOTE: secure_env_gtest.cpp (ROCM-26729) intentionally includes the header-only
# rocblaslt_secure_env.hpp by relative path rather than via a target include
# directory. Adding the internal rocblaslt/src/include dir to hipblaslt-test
# shadows the clients' own "utility.hpp" with the library's internal one and
# breaks the build of unrelated test sources, so it must not be added here.
