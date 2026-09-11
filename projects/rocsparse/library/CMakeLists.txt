# ########################################################################
# Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# MEMSTAT ?
if(BUILD_MEMSTAT)
  add_compile_options(-DROCSPARSE_WITH_MEMSTAT)
endif()

# ROCSPARSE_DEBUG ?
if(BUILD_ROCSPARSE_DEBUGGING)
  add_compile_options(-DROCSPARSE_DEBUGGING)
endif()

if(BUILD_ROCSPARSE_ILP64)
  add_compile_options(-Drocsparse_ILP64)
endif()

if (rocblas_FOUND)
  message("-- Build rocSPARSE library with rocBLAS")
  add_compile_options(-DROCSPARSE_WITH_ROCBLAS)
endif()

# Configure the version header (version numbers only) and the config header
# (build-time feature flags). Variables referenced via #cmakedefine in the
# .h.in must be set (or unset) prior to the configure_file() calls.
set(ROCSPARSE_WITH_HANDLE_CREATE ${ROCSPARSE_ENABLE_HANDLE_CREATE})
set(ROCSPARSE_WITH_ILDLT0 ${BUILD_WITH_ILDLT0})
set(ROCSPARSE_WITH_U16_REMOVED ${BUILD_WITH_U16_REMOVED})
set(ROCSPARSE_WITH_ILU0_BOOST_SIGN ${ROCSPARSE_ENABLE_ILU0_BOOST_SIGN})
set(ROCSPARSE_WITH_SPMAT_SCALE ${ROCSPARSE_ENABLE_SPMAT_SCALE})
set(ROCSPARSE_WITH_SDDMM_BATCHED ${ROCSPARSE_ENABLE_SDDMM_BATCHED})
set(ROCSPARSE_WITH_ELL_TRSV ${BUILD_WITH_ELL_TRSV})
set(ROCSPARSE_WITH_DIAGONAL_SOLVE ${BUILD_WITH_DIAGONAL_SOLVE})

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-version.h"
)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-config.h.in"
               "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-config.h"
)

# Guardrail: every feature flag that gates a public header must be registered in
# the config template above (see cmake/CheckFeatureFlags.cmake). EXCLUDE lists
# flags that appear in a public header but are intentionally build-only and never
# reach consumers; only ROCSPARSE_WITH_MEMSTAT qualifies (it gates dev-only
# memory-tracing routines in rocsparse-auxiliary.h). Other build-only flags
# (ASAN, ROCBLAS, CSC_TRSV, CSC_TRSM) are used solely in src, never in public
# headers, so the guardrail never sees them and they do not need excluding.
include(CheckFeatureFlags)
check_config_header_flags(
  PREFIX   ROCSPARSE_WITH_
  TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-config.h.in"
  HEADERS  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  EXCLUDE  ROCSPARSE_WITH_MEMSTAT
)

# Public rocSPARSE headers
set(rocsparse_headers_public
  include/rocsparse-auxiliary.h
  include/rocsparse-debugging.h
  include/rocsparse-functions.h
  include/rocsparse-types.h
  include/rocsparse-complex-types.h
  include/rocsparse.h
  include/internal
  ${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-version.h
  ${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-config.h
  include/internal
)

source_group("Header Files\\Public" FILES ${rocsparse_headers_public})

# Include sources
include(src/CMakeLists.txt)

# Create rocSPARSE library
add_library(rocsparse ${rocsparse_source} ${rocsparse_headers_public})
add_library(roc::rocsparse ALIAS rocsparse)

# Add Windows version resource to library
if(WIN32 AND BUILD_SHARED_LIBS)
    target_sources(rocsparse PRIVATE "${PROJECT_BINARY_DIR}/version.rc")
endif()

# Target compile options
target_compile_options(rocsparse PRIVATE -Wno-unused-command-line-argument -Wall)

# Optionally remove the deprecated rocsparse_indextype_u16 index type from the
# public API. The ROCSPARSE_WITH_U16_REMOVED flag itself is baked into the
# installed rocsparse-version.h (see configure_file above) so that it propagates
# to downstream consumers and makes their use of the enumerator a compile error.
# ROCSPARSE_KEEP_INTERNAL_U16 is defined only for the library's own translation
# units, keeping the enumerator available so that internal switch statements
# remain valid (avoiding -Wswitch warnings) while still returning "not
# implemented" for the deprecated value.
if(BUILD_WITH_U16_REMOVED)
    target_compile_definitions(rocsparse PRIVATE ROCSPARSE_KEEP_INTERNAL_U16)
    message(STATUS "Building rocSPARSE with rocsparse_indextype_u16 removed.")
endif()

# Target include directories
target_include_directories(rocsparse
                           PRIVATE ${ROCPRIM_INCLUDE_DIRS}
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                           PUBLIC  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocsparse>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Target link libraries
target_link_libraries(rocsparse PRIVATE roc::rocprim hip::device)
set(static_depends PACKAGE rocprim)

if (BUILD_WITH_ROCBLAS AND rocblas_FOUND)
  target_link_libraries(rocsparse PRIVATE roc::rocblas)
  if(NOT BUILD_SHARED_LIBS)
    list(APPEND static_depends PACKAGE rocblas)
  endif()
endif()

if(BUILD_CODE_COVERAGE)
  target_compile_options(rocsparse PRIVATE -fprofile-instr-generate -fcoverage-mapping)
  target_link_libraries(rocsparse PRIVATE -fprofile-instr-generate)
endif()

if (NOT WIN32)
  if (BUILD_SHARED_LIBS AND BUILD_WITH_ROCTX)

    set(ROCTX_PATH "" CACHE STRING "Path to the roctx library (directory containing libroctx64.so)")

    find_path(ROCTRACER_INCLUDE_DIR
      NAMES roctracer/roctx.h
      HINTS
      ${ROCTX_PATH}
      ${ROCTX_PATH}/include
      /opt/rocm/include
      ${ROCTX_PATH}/../include
      DOC "Path to the roctracer include directory containing roctx.h")

    find_library(ROCTX_LIBRARY
      NAMES roctx64
      HINTS
      ${ROCTX_PATH} # User-provided path
      ${ROCTX_PATH}/lib
      ${ROCTX_PATH}/lib64
      PATHS
      /opt/rocm/lib # Default ROCm path
    )

    if(ROCTRACER_INCLUDE_DIR AND ROCTX_LIBRARY)
      message(STATUS "Found roctracer include directory: ${ROCTRACER_INCLUDE_DIR}")
      message(STATUS "Found roctx library: ${ROCTX_LIBRARY}")

      # Add the include directory
      target_include_directories(rocsparse PRIVATE ${ROCTRACER_INCLUDE_DIR})

      # Link the rocTx lib
      target_link_libraries(rocsparse PRIVATE ${ROCTX_LIBRARY})
      target_compile_definitions( rocsparse PRIVATE ROCSPARSE_BUILT_WITH_ROCTX )
      message(STATUS "ROCTX tracing support enabled for target rocsparse.")
    else()
      message(WARNING "ROCTX tracing will be disabled for target rocsparse.")

      if(NOT ROCTRACER_INCLUDE_DIR)
        message(WARNING "Header 'roctracer/roctx.h' not found.")
      endif()

      if(NOT ROCTX_LIBRARY)
        message(WARNING "Library 'roctx64' not found. ROCTX_PATH: '${ROCTX_PATH}' Default PATH: '/opt/rocm/lib'.")
      endif()
    endif()
  endif()
endif()

# CSC triangular solve support (rocsparse_spsv / rocsparse_sptrsv)
if(BUILD_WITH_CSC_TRSV)
  target_compile_definitions(rocsparse PRIVATE ROCSPARSE_WITH_CSC_TRSV)
endif()

# CSC triangular solve support (rocsparse_spsm / rocsparse_sptrsm)
if(BUILD_WITH_CSC_TRSM)
  target_compile_definitions(rocsparse PRIVATE ROCSPARSE_WITH_CSC_TRSM)
endif()

# Target properties

# Offload compression ?
if(BUILD_WITH_OFFLOAD_COMPRESS)
  set_target_properties(rocsparse PROPERTIES COMPILE_FLAGS "--offload-compress")
endif()

rocm_set_soversion(rocsparse ${rocsparse_SOVERSION})
set_target_properties(rocsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(rocsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_properties(rocsparse PROPERTIES DEBUG_POSTFIX "-d")

if (WIN32 AND BUILD_CLIENTS)
  add_custom_command( TARGET rocsparse POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:rocsparse> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:rocsparse> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET rocsparse POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/rocsparse.pdb ${PROJECT_BINARY_DIR}/clients/staging/rocsparse.pdb )
  endif()
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(rocsparse EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-export.h)

# Force installation of .f module file
rocm_install(FILES "${CMAKE_SOURCE_DIR}/library/src/rocsparse.f90"
              "${CMAKE_SOURCE_DIR}/library/src/rocsparse_enums.f90"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocsparse"
)

# Get all header files from the source include directory
file(GLOB_RECURSE ROCSPARSE_SOURCE_HEADERS "${PROJECT_SOURCE_DIR}/library/include/*")

# Create custom command to copy headers with proper dependencies
add_custom_command(
  OUTPUT ${PROJECT_BINARY_DIR}/include/rocsparse/.header_stamp
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/library/include ${PROJECT_BINARY_DIR}/include/rocsparse
  COMMAND ${CMAKE_COMMAND} -E touch ${PROJECT_BINARY_DIR}/include/rocsparse/.header_stamp
  DEPENDS ${ROCSPARSE_SOURCE_HEADERS}
  COMMENT "Copying rocSPARSE headers to build directory"
)

# Create custom target for header copying
add_custom_target(rocsparse_headers DEPENDS ${PROJECT_BINARY_DIR}/include/rocsparse/.header_stamp)

# Make the library depend on the headers
add_dependencies(rocsparse rocsparse_headers)

# Install targets
rocm_install_targets(TARGETS rocsparse
                     INCLUDE
                       ${CMAKE_BINARY_DIR}/include
)

# Export targets
rocm_export_targets(TARGETS roc::rocsparse
                    DEPENDS PACKAGE hip
                    STATIC_DEPENDS ${static_depends}
                    NAMESPACE roc::
)
