# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Dispatcher instance OBJECT libraries for CK Tile grouped convolution.
#
# Defines ck_add_dispatcher_conv_instances() and creates targets:
#   ck_dispatcher_grouped_conv_fwd
#   ck_dispatcher_grouped_conv_bwd_weight
#   ck_dispatcher_grouped_conv_bwd_data
#
# Each target is an OBJECT library compiled from codegen-generated kernel sources.
# Include via add_subdirectory() from a parent that has already called find_package(Python3).

# ---- Configuration (inherits from parent scope) ----

set(DISPATCHER_DIR "${PROJECT_SOURCE_DIR}/dispatcher")
set(DISPATCHER_CODEGEN "${DISPATCHER_DIR}/codegen/unified_grouped_conv_codegen.py")
set(DISPATCHER_CONFIG_SET "tests" CACHE STRING "Dispatcher config set: tests or profiler")

# Extract first GPU target for codegen
string(REPLACE ";" " " _GPU_TARGETS_SPACE "${GPU_TARGETS}")
string(REPLACE " " ";" _GPU_TARGETS_LIST "${_GPU_TARGETS_SPACE}")
list(GET _GPU_TARGETS_LIST 0 _GPU_TARGET)

# ---- Variant-to-path mapping tables ----

set(_DISP_CONFIG_SUBDIR_fwd "forward")
set(_DISP_CONFIG_SUBDIR_bwd_weight "backward_weight")
set(_DISP_CONFIG_SUBDIR_bwd_data "backward_data")

set(_DISP_GEN_SCRIPT "generate_profiler_kernels.py")

set(_DISP_HEADER_PREFIX_fwd "grouped_conv_fwd_")
set(_DISP_HEADER_PREFIX_bwd_weight "grouped_conv_bwd_weight_")
set(_DISP_HEADER_PREFIX_bwd_data "grouped_conv_bwd_data_")


# =============================================================================
# ck_add_dispatcher_conv_instances(<variant>)
#
# Creates an OBJECT library target: ck_dispatcher_grouped_conv_<variant>
# and a convenience target: dispatcher_<variant>_lib
#
# <variant> must be one of: fwd, bwd_weight, bwd_data
# =============================================================================
function(ck_add_dispatcher_conv_instances VARIANT)
  # Look up variant-specific paths
  set(CONFIG_SUBDIR "${_DISP_CONFIG_SUBDIR_${VARIANT}}")
  set(HEADER_PREFIX "${_DISP_HEADER_PREFIX_${VARIANT}}")

  if(NOT CONFIG_SUBDIR)
    message(FATAL_ERROR "Unknown dispatcher variant: ${VARIANT}")
  endif()

  set(CONFIG_DIR "${DISPATCHER_DIR}/codegen/configs/grouped_conv/${CONFIG_SUBDIR}")
  set(SCRIPT_PATH "${DISPATCHER_DIR}/scripts/${_DISP_GEN_SCRIPT}")
  set(KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/dispatcher_${VARIANT}_kernels")
  set(TARGET_NAME "ck_dispatcher_grouped_conv_${VARIANT}")

  # --- Step 1: Run Python codegen at configure time ---
  # Always regenerate: DISPATCHER_CONFIG_SET is a CACHE variable, so changing it
  # triggers a full reconfigure. Wipe the kernel dir first to remove stale files.
  file(REMOVE_RECURSE ${KERNEL_DIR})
  file(MAKE_DIRECTORY ${KERNEL_DIR})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SCRIPT_PATH}
            --variant ${VARIANT}
            --config-dir ${CONFIG_DIR}
            --codegen ${DISPATCHER_CODEGEN}
            --output-dir ${KERNEL_DIR}
            --arch ${_GPU_TARGET}
            --config-set ${DISPATCHER_CONFIG_SET}
    RESULT_VARIABLE ret
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    WORKING_DIRECTORY ${DISPATCHER_DIR}/codegen
  )
  if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Dispatcher ${VARIANT} kernel generation failed.\nReturn: ${ret}\nOutput: ${output}\nError: ${error}")
  endif()
  message(STATUS "Dispatcher ${VARIANT} kernels generated (config set: ${DISPATCHER_CONFIG_SET}): ${output}")

  # --- Step 2: Collect kernel implementation .cpp files ---
  # Only match full implementation headers (exclude *_decl.hpp declaration-only headers).
  # The codegen writes kernel_name.cpp (includes kernel_name.hpp) alongside each
  # kernel_name_decl.hpp.  The .cpp files are what we compile here.
  file(GLOB _KERNEL_IMPL_HEADERS "${KERNEL_DIR}/${HEADER_PREFIX}*.hpp")
  set(_KERNEL_SOURCES "")
  foreach(HEADER ${_KERNEL_IMPL_HEADERS})
    get_filename_component(STEM ${HEADER} NAME_WE)
    # Skip declaration-only headers — they have no matching .cpp to compile
    if(STEM MATCHES "_decl$")
      continue()
    endif()
    set(WRAPPER_CPP "${KERNEL_DIR}/${STEM}.cpp")
    if(NOT EXISTS ${WRAPPER_CPP})
      file(WRITE ${WRAPPER_CPP} "// Auto-generated wrapper\n#include \"${STEM}.hpp\"\n")
    endif()
    list(APPEND _KERNEL_SOURCES ${WRAPPER_CPP})
  endforeach()

  if(NOT _KERNEL_SOURCES)
    message(WARNING "Dispatcher ${VARIANT}: no kernel sources generated, skipping target creation")
    return()
  endif()

  # --- Step 3: Collect chunked registration .cpp files (dispatch layer) ---
  file(GLOB _REG_CPPS "${KERNEL_DIR}/register_*.cpp")

  # =========================================================================
  # Split into two OBJECT libraries to minimise rebuild scope:
  #
  #   <TARGET_NAME>          (ck_dispatcher_grouped_conv_<variant>)
  #     — kernel implementation .cpp files
  #     — includes full kernel headers (CK Tile kernel impl headers)
  #     — rebuilt when kernel implementation headers change
  #
  #   <TARGET_NAME>_dispatch (ck_dispatcher_grouped_conv_<variant>_dispatch)
  #     — registration/dispatch chunk .cpp files
  #     — includes only *_decl.hpp (forward-declarations, no CK Tile includes)
  #     — NOT rebuilt when kernel implementation headers change
  #
  # The profiler links both via target_link_libraries.
  # =========================================================================

  # --- Step 4a: Kernel implementation OBJECT library ---
  add_library(${TARGET_NAME} OBJECT ${_KERNEL_SOURCES})
  target_include_directories(${TARGET_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dispatcher/include
    ${KERNEL_DIR}
  )
  target_compile_options(${TARGET_NAME} PRIVATE
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=0
    -DCK_TILE_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK=1
    -Wno-undefined-func-template
    -Wno-float-equal
    -Wno-header-hygiene
    -Wno-unused-parameter
    -Wno-missing-variable-declarations
  )
  if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
    target_compile_options(${TARGET_NAME} PRIVATE --offload-compress)
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # --- Step 4b: Dispatch/registration OBJECT library ---
  set(DISPATCH_TARGET_NAME "${TARGET_NAME}_dispatch")
  if(_REG_CPPS)
    add_library(${DISPATCH_TARGET_NAME} OBJECT ${_REG_CPPS})
    target_include_directories(${DISPATCH_TARGET_NAME} PRIVATE
      ${PROJECT_SOURCE_DIR}/include
      ${PROJECT_SOURCE_DIR}/dispatcher/include
      ${KERNEL_DIR}
    )
    # Registration files only need lightweight _decl.hpp headers; compile options
    # are therefore lighter — no need for the GPU kernel-compilation flags.
    target_compile_options(${DISPATCH_TARGET_NAME} PRIVATE
      -Wno-ctad-maybe-unsupported
      -Wno-old-style-cast
      -Wno-header-hygiene
      -Wno-unused-parameter
    )
    set_target_properties(${DISPATCH_TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  else()
    message(WARNING "Dispatcher ${VARIANT}: no registration .cpp files found")
  endif()

  # Progress monitoring
  list(LENGTH _KERNEL_SOURCES _KERNEL_COUNT)
  list(LENGTH _REG_CPPS _REG_COUNT)
  message(STATUS "Dispatcher ${VARIANT}: ${_KERNEL_COUNT} kernel TUs + ${_REG_COUNT} dispatch TUs (separate OBJECT libraries)")

  # Convenience ninja target: ninja dispatcher_<variant>_lib
  set(_ALL_DEPS ${TARGET_NAME})
  if(TARGET ${DISPATCH_TARGET_NAME})
    list(APPEND _ALL_DEPS ${DISPATCH_TARGET_NAME})
  endif()
  add_custom_target(dispatcher_${VARIANT}_lib
    DEPENDS ${_ALL_DEPS}
    COMMENT "All dispatcher ${VARIANT} instances compiled (${_KERNEL_COUNT} kernels + ${_REG_COUNT} dispatch TUs)."
  )
endfunction()


# =============================================================================
# Create all three dispatcher instance libraries
# =============================================================================
ck_add_dispatcher_conv_instances(fwd)
ck_add_dispatcher_conv_instances(bwd_weight)
ck_add_dispatcher_conv_instances(bwd_data)
