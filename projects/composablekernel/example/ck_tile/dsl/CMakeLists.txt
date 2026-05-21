# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Root CMakeLists for example/ck_tile/dsl/. Holds:
#   1. The shared launcher binary (built once, used by every dsl/<N>_*).
#   2. add_subdirectory() for each example directory.
#
# This subtree is intentionally separated from `example/ck_tile/52_ck_dsl`
# (the original PoC); the layout in `dsl/` is what the feature/perf
# parity work uses going forward.

cmake_minimum_required(VERSION 3.16)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(CK_DSL_PY_PKG "${CMAKE_SOURCE_DIR}/python")

# --- shared launcher -------------------------------------------------------

add_executable(ck_dsl_launcher
    ${CMAKE_CURRENT_SOURCE_DIR}/common/launcher.cpp
)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/common/launcher.cpp
    PROPERTIES LANGUAGE HIP)
target_compile_features(ck_dsl_launcher PRIVATE cxx_std_20)
target_include_directories(ck_dsl_launcher PRIVATE
    ${HIP_INCLUDE_DIRS}
    /opt/rocm/include
)
set_target_properties(ck_dsl_launcher PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# --- per-example helper: build manifest+hsaco, register a benchmark target -

function(ck_dsl_add_example dir_name)
    set(example_dir ${CMAKE_CURRENT_SOURCE_DIR}/${dir_name})
    set(out_dir ${CMAKE_BINARY_DIR}/ck_dsl_examples/${dir_name})
    set(gen_py ${example_dir}/gen.py)
    set(manifest ${out_dir}/manifest.json)

    if(NOT EXISTS ${gen_py})
        message(WARNING "ck_dsl_add_example(${dir_name}): missing gen.py at ${gen_py}")
        return()
    endif()

    add_custom_command(
        OUTPUT ${manifest}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out_dir}
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CK_DSL_PY_PKG}
                ${Python3_EXECUTABLE} ${gen_py} --output-dir ${out_dir}
        DEPENDS ${gen_py}
        COMMENT "ck_dsl gen ${dir_name}"
        VERBATIM
    )

    add_custom_target(ck_dsl_example_${dir_name} ALL
        DEPENDS ${manifest} ck_dsl_launcher
    )

    # Smoke-run target (manually invoked): builds + launches with verify.
    add_custom_target(run_ck_dsl_example_${dir_name}
        COMMAND $<TARGET_FILE:ck_dsl_launcher>
                ${out_dir}/*.hsaco
                ${manifest}
                --verify
        DEPENDS ck_dsl_example_${dir_name}
        VERBATIM
        USES_TERMINAL
    )
endfunction()

# --- register every example ------------------------------------------------

ck_dsl_add_example(06_gemm_universal_cshuffle)
ck_dsl_add_example(07_gemm_universal_sweep)
ck_dsl_add_example(08_bake_off_implicit_gemm)
ck_dsl_add_example(09_bake_off_direct_conv_16c)
ck_dsl_add_example(10_bake_off_direct_conv_4c)
ck_dsl_add_example(02_layernorm2d)
ck_dsl_add_example(05_reduce)
ck_dsl_add_example(10_rmsnorm2d)
ck_dsl_add_example(16_batched_gemm)
ck_dsl_add_example(21_elementwise)
ck_dsl_add_example(37_transpose)
