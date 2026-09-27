# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Dispatcher instance OBJECT libraries for CK Tile grouped convolution.
#
# Per-arch OBJECT libraries:
#   ck_dispatcher_grouped_conv_<variant>_<arch>
#     HIP_ARCHITECTURES=<arch>, sources in dispatcher_<variant>_<arch>_kernels
#
# INTERFACE aliases (consumer names):
#   ck_dispatcher_grouped_conv_fwd
#   ck_dispatcher_grouped_conv_bwd_weight
#   ck_dispatcher_grouped_conv_bwd_data
#
# Each INTERFACE links every per-arch OBJECT for that variant plus a dispatcher
# TU that defines the unmangled public register_all_grouped_conv_<variant>_kernels.
#
# Include via add_subdirectory() from a parent that has already called find_package(Python3).

# ---- Configuration (inherits from parent scope) ----

set(DISPATCHER_DIR "${PROJECT_SOURCE_DIR}/dispatcher")
set(DISPATCHER_RULE_SET "tests" CACHE STRING "Dispatcher rule set: profiler, tests, full, full-tests, tiny, or default")

include(${DISPATCHER_DIR}/grouped_conv_arch_plan.cmake)
string(REPLACE ";" " " _GPU_TARGETS_SPACE "${GPU_TARGETS}")
string(REPLACE " " ";" _GPU_TARGETS_LIST "${_GPU_TARGETS_SPACE}")
list(FILTER _GPU_TARGETS_LIST EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _GPU_TARGETS_LIST)
ck_dispatcher_rdna_target("${_GPU_TARGETS_LIST}" _RDNA_TARGET)

# ---- Variant-to-path mapping tables ----

set(_DISP_GEN_SCRIPT "generate_profiler_kernels.py")

set(_DISP_HEADER_PREFIX_fwd "grouped_conv_fwd_")
set(_DISP_HEADER_PREFIX_bwd_weight "grouped_conv_bwd_weight_")
set(_DISP_HEADER_PREFIX_bwd_data "grouped_conv_bwd_data_")


# =============================================================================
# ck_add_dispatcher_conv_instances(<variant> <arch> <rule_set>)
#
# Creates an OBJECT library: ck_dispatcher_grouped_conv_<variant>_<arch>
#
# <variant> must be one of: fwd, bwd_weight, bwd_data
# =============================================================================
function(ck_add_dispatcher_conv_instances VARIANT ARCH RULE_SET)
  # Look up variant-specific paths
  set(HEADER_PREFIX "${_DISP_HEADER_PREFIX_${VARIANT}}")

  if(NOT HEADER_PREFIX)
    message(FATAL_ERROR "Unknown dispatcher variant: ${VARIANT}")
  endif()

  set(SCRIPT_PATH "${DISPATCHER_DIR}/scripts/${_DISP_GEN_SCRIPT}")
  set(KERNEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/dispatcher_${VARIANT}_${ARCH}_kernels")
  set(TARGET_NAME "ck_dispatcher_grouped_conv_${VARIANT}_${ARCH}")

  # --- Step 1: Run Python codegen at configure time ---
  # Always regenerate at configure time. The driver stages output and atomically
  # replaces KERNEL_DIR only after the complete catalog succeeds.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} ${SCRIPT_PATH}
            --variant ${VARIANT}
            --output-dir ${KERNEL_DIR}
            --arch ${ARCH}
            --rule-set ${RULE_SET}
    RESULT_VARIABLE ret
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    WORKING_DIRECTORY ${DISPATCHER_DIR}/codegen
  )
  if(NOT ret EQUAL 0)
    message(FATAL_ERROR "Dispatcher ${VARIANT} kernel generation failed.\nReturn: ${ret}\nOutput: ${output}\nError: ${error}")
  endif()
  message(STATUS "Dispatcher ${VARIANT} kernels generated (rule set: ${RULE_SET}): ${output}")

  # --- Step 2: Create .cpp wrappers for each generated kernel header ---
  file(GLOB _KERNEL_HEADERS "${KERNEL_DIR}/${HEADER_PREFIX}*.hpp")
  set(_KERNEL_SOURCES "")
  foreach(HEADER ${_KERNEL_HEADERS})
    get_filename_component(STEM ${HEADER} NAME_WE)
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

  # --- Step 3: Add chunked registration .cpp files ---
  file(GLOB _REG_CPPS "${KERNEL_DIR}/register_*.cpp")
  foreach(REG_CPP ${_REG_CPPS})
    list(APPEND _KERNEL_SOURCES "${REG_CPP}")
    set_source_files_properties("${REG_CPP}" PROPERTIES
      COMPILE_OPTIONS "-Wno-ctad-maybe-unsupported;-Wno-old-style-cast"
    )
  endforeach()

  # --- Step 4: Create OBJECT library ---
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
    # New clang -Weverything suggestion firing on builder-pattern setters that
    # return *this in the shared dispatcher headers; not a real defect.
    -Wno-lifetime-safety-intra-tu-suggestions
  )
  if(NOT WIN32 AND ${hip_VERSION_FLAT} GREATER 600241132)
    target_compile_options(${TARGET_NAME} PRIVATE --offload-compress)
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${TARGET_NAME} PROPERTIES HIP_ARCHITECTURES ${ARCH})

  # Progress monitoring
  list(LENGTH _KERNEL_SOURCES _TOTAL_COUNT)
  math(EXPR _KERNEL_COUNT "${_TOTAL_COUNT} - 1")
  message(STATUS "Dispatcher ${VARIANT}: ${_TOTAL_COUNT} compilation units (${_KERNEL_COUNT} kernels + registration)")
endfunction()


# =============================================================================
# ck_add_dispatcher_conv_variant_interface(<variant> <arch>...)
#
# Writes a dispatcher TU that restores the unmangled public
# register_all_grouped_conv_<variant>_kernels symbols, then creates the
# INTERFACE alias ck_dispatcher_grouped_conv_<variant>.
# =============================================================================
function(ck_add_dispatcher_conv_variant_interface VARIANT)
  if(NOT ARGN)
    return()
  endif()
  set(_arches ${ARGN})
  set(_fn "register_all_grouped_conv_${VARIANT}_kernels")
  set(_dispatch_cpp "${CMAKE_CURRENT_BINARY_DIR}/register_all_grouped_conv_${VARIANT}_dispatch.cpp")
  set(_dispatch_target "ck_dispatcher_grouped_conv_${VARIANT}_dispatch")
  set(_iface "ck_dispatcher_grouped_conv_${VARIANT}")

  set(_content "")
  string(APPEND _content "#include \"ck_tile/dispatcher/grouped_conv_registry.hpp\"\n")
  string(APPEND _content "#include <stdexcept>\n")
  string(APPEND _content "#include <string>\n")
  string(APPEND _content "\n")
  string(APPEND _content "namespace ck_tile {\n")
  string(APPEND _content "namespace dispatcher {\n")
  string(APPEND _content "\n")
  foreach(_arch IN LISTS _arches)
    string(APPEND _content "void ${_fn}_${_arch}(GroupedConvRegistry&, const std::string&);\n")
  endforeach()
  string(APPEND _content "\n")
  string(APPEND _content "void ${_fn}(GroupedConvRegistry& registry, const std::string& arch)\n")
  string(APPEND _content "{\n")
  foreach(_arch IN LISTS _arches)
    string(APPEND _content "    if(arch == \"${_arch}\")\n")
    string(APPEND _content "    {\n")
    string(APPEND _content "        ${_fn}_${_arch}(registry, arch);\n")
    string(APPEND _content "        return;\n")
    string(APPEND _content "    }\n")
  endforeach()
  string(APPEND _content "    throw std::runtime_error(\"no grouped conv ${VARIANT} catalog for arch \" + arch);\n")
  string(APPEND _content "}\n")
  string(APPEND _content "\n")
  string(APPEND _content "void ${_fn}(const std::string& arch)\n")
  string(APPEND _content "{\n")
  string(APPEND _content "    ${_fn}(GroupedConvRegistry::instance(), arch);\n")
  string(APPEND _content "}\n")
  string(APPEND _content "\n")
  string(APPEND _content "} // namespace dispatcher\n")
  string(APPEND _content "} // namespace ck_tile\n")
  file(WRITE "${_dispatch_cpp}" "${_content}")

  add_library(${_dispatch_target} OBJECT ${_dispatch_cpp})
  target_include_directories(${_dispatch_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dispatcher/include)
  # base_registry.hpp uses std::scoped_lock CTAD; BUILD_DEV enables -Werror.
  target_compile_options(${_dispatch_target} PRIVATE
    -Wno-ctad-maybe-unsupported
    -Wno-old-style-cast)
  set_source_files_properties(
    ${_dispatch_cpp}
    PROPERTIES LANGUAGE CXX)
  set_target_properties(${_dispatch_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  set(_link_libs)
  foreach(_arch IN LISTS _arches)
    list(APPEND _link_libs ck_dispatcher_grouped_conv_${VARIANT}_${_arch})
  endforeach()
  list(APPEND _link_libs ${_dispatch_target})

  add_library(${_iface} INTERFACE)
  # OBJECT files are not propagated through INTERFACE via target_link_libraries.
  foreach(_lib IN LISTS _link_libs)
    target_sources(${_iface} INTERFACE $<TARGET_OBJECTS:${_lib}>)
  endforeach()
  target_link_libraries(${_iface} INTERFACE ${_link_libs})

  list(LENGTH _arches _arch_count)
  add_custom_target(dispatcher_${VARIANT}_lib
    DEPENDS ${_link_libs}
    COMMENT "All dispatcher ${VARIANT} instance libraries compiled (${_arch_count} arch(es))."
  )
endfunction()


# =============================================================================
# Create the dispatcher instance libraries selected for each architecture
# =============================================================================
set(_DISP_ARCHES_fwd "")
set(_DISP_ARCHES_bwd_weight "")
set(_DISP_ARCHES_bwd_data "")

foreach(_ARCH IN LISTS _GPU_TARGETS_LIST)
  ck_dispatcher_conv_variants_for_arch("${_ARCH}" _ARCH_VARIANTS)
  ck_dispatcher_conv_rule_set_for_arch("${_ARCH}" "${DISPATCHER_RULE_SET}" _ARCH_RULE_SET)
  foreach(_VARIANT IN LISTS _ARCH_VARIANTS)
    ck_add_dispatcher_conv_instances(${_VARIANT} ${_ARCH} ${_ARCH_RULE_SET})
    if(TARGET ck_dispatcher_grouped_conv_${_VARIANT}_${_ARCH})
      list(APPEND _DISP_ARCHES_${_VARIANT} ${_ARCH})
    endif()
  endforeach()
endforeach()

foreach(_VARIANT fwd bwd_weight bwd_data)
  ck_add_dispatcher_conv_variant_interface(${_VARIANT} ${_DISP_ARCHES_${_VARIANT}})
endforeach()

# RDNA production GPU gate. Keep it here so gfx9 tile gtests do not grow an
# RDNA catalog, and so the test is created whenever the instance library is.
if(_RDNA_TARGET AND BUILD_TESTING AND TARGET ck_dispatcher_grouped_conv_fwd AND
   NOT TARGET test_grouped_conv_gpu_correctness)
  add_executable(test_grouped_conv_gpu_correctness
    ${DISPATCHER_DIR}/tests/test_grouped_conv_gpu_correctness.cpp)
  target_include_directories(test_grouped_conv_gpu_correctness PRIVATE
    ${PROJECT_SOURCE_DIR}/dispatcher/include
    ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(test_grouped_conv_gpu_correctness PRIVATE
    -Wno-ctad-maybe-unsupported)
  target_compile_definitions(test_grouped_conv_gpu_correctness PRIVATE
    CK_DISPATCHER_TEST_ARCH="${_RDNA_TARGET}")
  target_link_libraries(test_grouped_conv_gpu_correctness PRIVATE
    ck_dispatcher_grouped_conv_fwd hip::host)
  add_test(NAME test_grouped_conv_gpu_correctness
    COMMAND test_grouped_conv_gpu_correctness)
  set_tests_properties(test_grouped_conv_gpu_correctness PROPERTIES
    LABELS "gpu;${_RDNA_TARGET}"
    SKIP_RETURN_CODE 77
    TIMEOUT 120)
endif()
