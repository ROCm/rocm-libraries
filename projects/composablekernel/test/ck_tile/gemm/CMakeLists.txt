# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Currently ck_tile_gemm is only built on gfx94/gfx95
set(EXAMPLE_GEMM_COMPILE_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
set(EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS)
if(CK_USE_OCP_FP8)
    list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DCK_TILE_USE_OCP_FP8)
endif()
list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS
    -mllvm
    -enable-noalias-to-md-conversion=0
)
set(EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})

if(GPU_TARGETS MATCHES "gfx94|gfx95|gfx90a|gfx11|gfx12")
    # Collect all test targets for umbrella label
    set(CK_TILE_GEMM_TEST_TARGETS)

    if(GPU_TARGETS MATCHES "gfx94|gfx95")
        add_gtest_executable(test_ck_tile_gemm_pipeline_mem test_gemm_pipeline_mem.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3 test_gemm_pipeline_compv3.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv4 test_gemm_pipeline_compv4.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_persistent test_gemm_pipeline_persistent.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv6 test_gemm_pipeline_compv6.cpp)

        set(_ck_tile_gemm_quick_target test_ck_tile_gemm_pipeline_quick)
        set_source_files_properties(
            test_gemm_pipeline_quick.cpp PROPERTIES LANGUAGE HIP
        )
        add_executable(
            ${_ck_tile_gemm_quick_target} test_gemm_pipeline_quick.cpp
        )
        set_property(
            TARGET ${_ck_tile_gemm_quick_target}
            PROPERTY HIP_ARCHITECTURES ${SUPPORTED_GPU_TARGETS}
        )
        target_compile_options(
            ${_ck_tile_gemm_quick_target}
            PRIVATE
                -Wno-global-constructors
                -Wno-undef
                ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS}
        )
        target_link_libraries(
            ${_ck_tile_gemm_quick_target} PRIVATE gtest_main getopt::getopt
        )
        add_dependencies(tests ${_ck_tile_gemm_quick_target})
        add_dependencies(check ${_ck_tile_gemm_quick_target})
        rocm_install(TARGETS ${_ck_tile_gemm_quick_target} COMPONENT tests)

        target_compile_options(test_ck_tile_gemm_pipeline_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv3 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv4 PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_persistent PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv6 PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_mem
            test_ck_tile_gemm_pipeline_compv3
            test_ck_tile_gemm_pipeline_compv4
            test_ck_tile_gemm_pipeline_persistent
            test_ck_tile_gemm_pipeline_compv6
        )
    endif()

    if(GPU_TARGETS MATCHES "gfx95")
        add_gtest_executable(test_ck_tile_gemm_pipeline_comp_async test_gemm_pipeline_comp_async.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_comp_async PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        add_gtest_executable(test_ck_tile_gemm_pipeline_tf32_mem test_gemm_pipeline_tf32_mem.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_tf32_mem PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_comp_async
            test_ck_tile_gemm_pipeline_tf32_mem
        )

        add_gtest_executable(test_ck_tile_gemm_pipeline_comp_async_eight_waves test_gemm_pipeline_comp_async_eight_waves.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_comp_async_eight_waves PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_comp_async_eight_waves
        )

        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_async_rrr test_gemm_pipeline_compv3_async_rrr.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_async_rrr PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_compv3_async_rrr
        )

        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_async_rcr test_gemm_pipeline_compv3_async_rcr.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_async_rcr PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_compv3_async_rcr
        )

        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_async_ccr test_gemm_pipeline_compv3_async_ccr.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_async_ccr PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_compv3_async_ccr
        )

        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_async_crr test_gemm_pipeline_compv3_async_crr.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_async_crr PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_compv3_async_crr
        )

        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_async_persistent test_gemm_pipeline_compv3_async_persistent.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_async_persistent PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_compv3_async_persistent
        )
    endif()

    if(GPU_TARGETS MATCHES "gfx11|gfx12")
        # On Radeon devices, build the WMMA version instead
        # Define architecture macros for compile-time detection
        if(GPU_TARGETS MATCHES "gfx12")
            list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DARCH_GFX12)
            list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DARCH_GFX12)
        elseif(GPU_TARGETS MATCHES "gfx11")
            list(APPEND EXAMPLE_GEMM_COMPILE_OPTIONS -DARCH_GFX11)
            list(APPEND EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS -DARCH_GFX11)
        endif()

        add_gtest_executable(test_ck_tile_gemm_pipeline_mem_wmma test_gemm_pipeline_mem_wmma.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv3_wmma test_gemm_pipeline_compv3_wmma.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_compv4_wmma test_gemm_pipeline_compv4_wmma.cpp)
        add_gtest_executable(test_ck_tile_gemm_pipeline_persistent_wmma test_gemm_pipeline_persistent_wmma.cpp)
        target_compile_options(test_ck_tile_gemm_pipeline_mem_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv3_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_compv4_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_V4_OPTIONS})
        target_compile_options(test_ck_tile_gemm_pipeline_persistent_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})

        list(APPEND CK_TILE_GEMM_TEST_TARGETS
            test_ck_tile_gemm_pipeline_mem_wmma
            test_ck_tile_gemm_pipeline_compv3_wmma
            test_ck_tile_gemm_pipeline_compv4_wmma
            test_ck_tile_gemm_pipeline_persistent_wmma
        )
        
        if(GPU_TARGETS MATCHES "gfx125")
            add_gtest_executable(test_ck_tile_gemm_pipeline_tdm_wmma test_gemm_pipeline_comp_tdm_wmma.cpp)
            target_compile_options(test_ck_tile_gemm_pipeline_tdm_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
            add_gtest_executable(test_ck_tile_gemm_pipeline_comp_async_wmma test_gemm_pipeline_comp_async_wmma.cpp)
            target_compile_options(test_ck_tile_gemm_pipeline_comp_async_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_COMPUTE_ASYNC_OPTIONS})
            target_compile_options(test_ck_tile_gemm_pipeline_comp_async_wmma PRIVATE ${EXAMPLE_GEMM_COMPILE_OPTIONS})
        endif()
    endif()

    # Label all ck_tile gemm tests with CK_TILE_GEMM_TESTS for selective execution
    foreach(test_target ${CK_TILE_GEMM_TEST_TARGETS})
        set_tests_properties(${test_target} PROPERTIES LABELS "CK_TILE_GEMM_TESTS")
    endforeach()

    if(TARGET test_ck_tile_gemm_pipeline_quick)
        # Exercise every curated pipeline/layout configuration on one tile,
        # then sample padded-K and regular shapes across pipeline families.
        set(_ck_tile_gemm_quick_cases
            "TestCkTileGemmPipelineQuick/*.SingleTile"
            "TestCkTileGemmPipelineQuick/0.PaddK"
            "TestCkTileGemmPipelineQuick/4.PaddK"
            "TestCkTileGemmPipelineQuick/8.PaddK"
            "TestCkTileGemmPipelineQuick/10.PaddK"
            "TestCkTileGemmPipelineQuick/12.PaddK"
            "TestCkTileGemmPipelineQuick/1.Regular"
            "TestCkTileGemmPipelineQuick/5.Regular"
            "TestCkTileGemmPipelineQuick/9.Regular"
            "TestCkTileGemmPipelineQuick/11.Regular"
            "TestCkTileGemmPipelineQuick/13.Regular"
        )
        list(JOIN _ck_tile_gemm_quick_cases ":" _ck_tile_gemm_quick_filter)
        add_ck_tile_category_test(
            test_ck_tile_gemm_quick
            test_ck_tile_gemm_pipeline_quick
            "--gtest_filter=${_ck_tile_gemm_quick_filter}"
        )
        set_property(
            TEST test_ck_tile_gemm_quick APPEND PROPERTY LABELS CK_TILE_GEMM_TESTS
        )
    endif()

    # Umbrella target to build and run all ck_tile gemm tests
    # Usage: ninja ck_tile_gemm_tests
    add_custom_target(ck_tile_gemm_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_GEMM_TESTS"
        DEPENDS ${CK_TILE_GEMM_TEST_TARGETS}
        USES_TERMINAL
        COMMENT "Running all ck_tile gemm tests..."
    )
else()
    message(DEBUG "Skipping ck_tile_gemm tests for current target test_ck_tile_gemm_pipeline") 
endif()
