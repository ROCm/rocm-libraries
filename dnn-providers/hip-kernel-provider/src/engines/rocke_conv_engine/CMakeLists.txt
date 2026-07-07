# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# rocKE conv engine: hipDNN engine that JIT-compiles rocKE implicit-GEMM conv
# kernels and selects tile configs using the LightGBM ML heuristic.
#
# Required:
#   rocke_core    -- static library built by rocKE/CMakeLists.txt
#   LightGBM      -- shared library (lgbm) for the ML heuristic
#   hipRTC        -- for JIT compilation of rocKE LLVM IR
#
# The installed models directory is embedded at build time via -DROCKE_MODELS_DIR.
# At runtime the ROCKE_MODELS_DIR environment variable can override this.

target_sources(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/RockeConvEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/ConvFwdPlanBuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/plans/ConvFwdPlan.cpp
)

target_include_directories(hip_kernel_provider_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../rocke/platform/cpp/include
)

target_compile_definitions(hip_kernel_provider_impl PUBLIC HIPDNN_ENGINE_ROCKE_CONV=1)

# Embed the installed models directory.  The path is set by the superbuild or
# can be overridden via -DROCKE_MODELS_INSTALL_DIR on the cmake command line.
if(NOT DEFINED ROCKE_MODELS_INSTALL_DIR)
    set(ROCKE_MODELS_INSTALL_DIR
        "${CMAKE_INSTALL_PREFIX}/${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/hip_kernel_provider/rocke_models"
        CACHE PATH "Installed path for rocKE conv heuristic models")
endif()

target_compile_definitions(hip_kernel_provider_impl PRIVATE
    ROCKE_MODELS_DIR="${ROCKE_MODELS_INSTALL_DIR}"
)

# Link rocKE core (provides rocke_conv_problem_t, rocke_implicit_gemm_conv_*,
# rocke_conv_implicit_gemm_lower_to_llvm).
# hip_kernel_provider_impl is an OBJECT library; its PRIVATE link deps don't
# propagate to hip_kernel_provider (the shared lib) via $<TARGET_OBJECTS:...>.
# We set a target property here and let src/CMakeLists.txt forward it to the
# shared lib after hip_kernel_provider is defined.
target_link_libraries(hip_kernel_provider_impl PRIVATE rocke_core)

# hipRTC: used by compileIrToModule to JIT-compile rocKE LLVM IR to HSACO.
# hiprtc::hiprtc is already found by the top-level CMakeLists.txt.
target_link_libraries(hip_kernel_provider_impl PRIVATE hiprtc::hiprtc)

# LightGBM: loaded by ConvMLHeuristic.  Link as an imported shared library if
# found; otherwise fall back to a runtime dlopen (ConvMLHeuristic already uses
# that path on missing LGBM).
find_library(LGBM_LIB NAMES lgbm lib_lightgbm PATHS /opt/rocm/lib ${ROCM_PATH}/lib)
if(LGBM_LIB)
    target_link_libraries(hip_kernel_provider_impl PRIVATE ${LGBM_LIB})
    message(STATUS "rocke_conv_engine: found LightGBM at ${LGBM_LIB}")
else()
    message(STATUS "rocke_conv_engine: LightGBM not found at link time; will rely on dlopen at runtime")
endif()
