# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# ============================================================================
# ASM Kernel Code Generation
# ============================================================================
# Find Python interpreter for code generation
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Define paths for code generation
set(ASM_CODEGEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/codegen.py")
set(ASM_CODEGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(ASM_FMHA_FWD_CONFIG_HEADER "${ASM_CODEGEN_OUTPUT_DIR}/asm_fmha_v3_fwd_configs.hpp")
set(ASM_FMHA_BWD_CONFIG_HEADER "${ASM_CODEGEN_OUTPUT_DIR}/asm_fmha_v3_bwd_configs.hpp")

# Define supported GPU architectures (used for both glob patterns and AITER_GPU_ARCHS)
set(ASM_SDPA_GPU_ARCHS gfx942 gfx950)

# CSV files that the codegen script depends on (iterate over all architectures)
set(ASM_KERNEL_CSV_FILES "")
foreach(ARCH ${ASM_SDPA_GPU_ARCHS})
    file(GLOB_RECURSE ARCH_CSV_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/${ARCH}/**/*.csv"
    )
    list(APPEND ASM_KERNEL_CSV_FILES ${ARCH_CSV_FILES})
endforeach()

# Create the output directory
file(MAKE_DIRECTORY ${ASM_CODEGEN_OUTPUT_DIR})

# Custom command to generate ASM kernel config headers (forward path)
add_custom_command(
    OUTPUT ${ASM_FMHA_FWD_CONFIG_HEADER}
    COMMAND ${CMAKE_COMMAND} -E env "AITER_GPU_ARCHS=${ASM_SDPA_GPU_ARCHS}" --
            ${Python3_EXECUTABLE} ${ASM_CODEGEN_SCRIPT}
            -m fmha_v3_fwd
            -o ${ASM_CODEGEN_OUTPUT_DIR}
    DEPENDS ${ASM_CODEGEN_SCRIPT} ${ASM_KERNEL_CSV_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels
    COMMENT "Generating ASM fwd kernel configuration header for ${ASM_SDPA_GPU_ARCHS}"
    VERBATIM
)

# Custom command to generate ASM kernel config headers (backward path)
add_custom_command(
    OUTPUT ${ASM_FMHA_BWD_CONFIG_HEADER}
    COMMAND ${CMAKE_COMMAND} -E env "AITER_GPU_ARCHS=${ASM_SDPA_GPU_ARCHS}" --
            ${Python3_EXECUTABLE} ${ASM_CODEGEN_SCRIPT}
            -m fmha_v3_bwd
            -o ${ASM_CODEGEN_OUTPUT_DIR}
    DEPENDS ${ASM_CODEGEN_SCRIPT} ${ASM_KERNEL_CSV_FILES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels
    COMMENT "Generating ASM bwd kernel configuration header for ${ASM_SDPA_GPU_ARCHS}"
    VERBATIM
)

# Create a custom target for the generated headers
add_custom_target(asm_kernel_codegen
    DEPENDS ${ASM_FMHA_FWD_CONFIG_HEADER} ${ASM_FMHA_BWD_CONFIG_HEADER}
    COMMENT "Generating headers for asm sdpa kernels..."
)

# Ensure codegen runs before compiling the implementation
add_dependencies(hip_kernel_provider_impl asm_kernel_codegen)

# Add generated headers to include directories
target_include_directories(hip_kernel_provider_impl PRIVATE
    ${ASM_CODEGEN_OUTPUT_DIR}
)

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/SdpaBwdPlan.cpp
)

target_include_directories(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_ASM_SDPA=1)
target_compile_definitions(hip_kernel_provider_impl PRIVATE
    AITER_ASM_DIR="${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels"
)

# Copy ASM kernel binaries to build directory for local testing
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/
     DESTINATION ${CMAKE_BINARY_DIR}/hip_kernel_provider/asm_kernels
     FILES_MATCHING PATTERN "*.co"
)

# ============================================================================
# Kpack Packing (Phase 1 — build-time .co → .kpack)
# ============================================================================

# --- rocm_kpack Python source (for pack.py at build time) ---
set(KPACK_PYTHON_DIR "" CACHE PATH
    "Path to rocm_kpack Python source (e.g. <rocm-systems>/shared/kpack/python)")

if(KPACK_PYTHON_DIR AND NOT "${KPACK_PYTHON_DIR}" STREQUAL "")
    get_filename_component(_KPACK_PYTHON_DIR "${KPACK_PYTHON_DIR}" ABSOLUTE)
else()
    # Sparse-clone rocm-systems to get the rocm_kpack Python source
    set(_KPACK_SRC "${CMAKE_CURRENT_BINARY_DIR}/rocm_systems_kpack")
    set(_KPACK_PYTHON_DIR "${_KPACK_SRC}/shared/kpack/python")
    if(NOT EXISTS "${_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py")
        find_package(Git REQUIRED)
        execute_process(COMMAND ${GIT_EXECUTABLE} clone --depth 1 --filter=blob:none
            --sparse --branch develop https://github.com/ROCm/rocm-systems.git
            "${_KPACK_SRC}" RESULT_VARIABLE _rc)
        if(_rc EQUAL 0)
            execute_process(COMMAND ${GIT_EXECUTABLE} -C "${_KPACK_SRC}"
                sparse-checkout set shared/kpack/python)
        endif()
    endif()
endif()

if(NOT EXISTS "${_KPACK_PYTHON_DIR}/rocm_kpack/kpack.py")
    message(FATAL_ERROR "rocm_kpack Python source not found. "
        "Set -DKPACK_PYTHON_DIR=<rocm-systems>/shared/kpack/python")
endif()

# --- Ensure rocm_kpack runtime dependencies are available ---
set(_KPACK_PIP_DEPS "zstandard>=0.20.0" "msgpack")
foreach(_dep ${_KPACK_PIP_DEPS})
    string(REGEX REPLACE "[>=<].*" "" _mod "${_dep}")
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import ${_mod}"
        RESULT_VARIABLE _dep_check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(NOT _dep_check EQUAL 0)
        message(STATUS "Installing Python '${_dep}' package (required by rocm_kpack)...")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip install "${_dep}"
            RESULT_VARIABLE _pip_result
        )
        if(NOT _pip_result EQUAL 0)
            message(FATAL_ERROR "Failed to install '${_dep}' Python package. "
                "Install manually via: pip install ${_dep}")
        endif()
    endif()
endforeach()

# --- Per-arch kpack packing custom commands ---
set(ASM_KPACK_PACK_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/pack.py")
set(ASM_KPACK_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/.kpack")

# Collect all .co files as dependencies for the packing step
set(ASM_KERNEL_CO_FILES "")
foreach(ARCH ${ASM_SDPA_GPU_ARCHS})
    file(GLOB_RECURSE ARCH_CO_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels/${ARCH}/**/*.co")
    list(APPEND ASM_KERNEL_CO_FILES ${ARCH_CO_FILES})
endforeach()

set(ASM_KPACK_OUTPUTS "")
foreach(ARCH ${ASM_SDPA_GPU_ARCHS})
    set(_KPACK_FILE "${ASM_KPACK_OUTPUT_DIR}/hip_kernel_provider_sdpa_${ARCH}.kpack")
    list(APPEND ASM_KPACK_OUTPUTS "${_KPACK_FILE}")

    add_custom_command(
        OUTPUT "${_KPACK_FILE}"
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${_KPACK_PYTHON_DIR}"
            "PYTHONDONTWRITEBYTECODE=1"
            --
            ${Python3_EXECUTABLE} ${ASM_KPACK_PACK_SCRIPT}
            --kernel-dir ${CMAKE_CURRENT_SOURCE_DIR}/asm/asm_kernels
            --arch ${ARCH}
            --out-dir ${ASM_KPACK_OUTPUT_DIR}
        DEPENDS ${ASM_KPACK_PACK_SCRIPT} ${ASM_KERNEL_CO_FILES}
        COMMENT "Packing ASM SDPA kernels for ${ARCH} into .kpack"
        VERBATIM
    )
endforeach()

add_custom_target(asm_kernel_kpack ALL
    DEPENDS ${ASM_KPACK_OUTPUTS}
    COMMENT "Pack ASM SDPA .co kernels into .kpack archives"
)
add_dependencies(asm_kernel_kpack asm_kernel_codegen)

# --- Copy .kpack files to build directory for local testing ---
foreach(ARCH ${ASM_SDPA_GPU_ARCHS})
    set(_SRC "${ASM_KPACK_OUTPUT_DIR}/hip_kernel_provider_sdpa_${ARCH}.kpack")
    set(_DST "${CMAKE_BINARY_DIR}/hip_kernel_provider/asm_kernels/.kpack/")
    add_custom_command(TARGET asm_kernel_kpack POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_DST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_SRC}" "${_DST}"
        COMMENT "Copying ${ARCH} .kpack to build dir"
    )
endforeach()

# --- Install .kpack archives (Phase 2 runtime will load from these) ---
install(FILES ${ASM_KPACK_OUTPUTS}
    DESTINATION ${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/asm_kernels/.kpack
)
