# ########################################################################
# Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

if(BUILD_ROCSPARSE_ILP64)
  add_compile_options(-Drocsparse_ILP64)
endif()

if (rocblas_FOUND)
  message("-- Build rocSPARSE library with rocBLAS")
  add_compile_options(-DROCSPARSE_WITH_ROCBLAS)
endif()

# Configure a header file to pass the rocSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/rocsparse-version.h.in"
               "${PROJECT_BINARY_DIR}/include/rocsparse/rocsparse-version.h"
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
  include/internal
)

source_group("Header Files\\Public" FILES ${rocsparse_headers_public})

# Include sources
include(src/CMakeLists.txt)

# Create rocSPARSE library
add_library(rocsparse ${rocsparse_source} ${rocsparse_headers_public})
add_library(roc::rocsparse ALIAS rocsparse)

# Target compile options
target_compile_options(rocsparse PRIVATE -Wno-unused-command-line-argument -Wall)

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
