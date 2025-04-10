# ########################################################################
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

include(../cmake/Utilities.cmake)

# Configure a header file to pass the hipBLASLt version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h"
)

# Copy Public Headers to Build Dir
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt.h" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt.h" COPYONLY)
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt-ext.hpp" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt-ext.hpp" COPYONLY)
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt-types.h" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt-types.h" COPYONLY)
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_xfloat32.h" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt_xfloat32.h" COPYONLY)
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_float8.h" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt_float8.h" COPYONLY)
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt_float8_bc.h" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt_float8_bc.h" COPYONLY)
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipblaslt-ext-op.h" "${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt-ext-op.h" COPYONLY)

# Public hipBLASLt headers
set(hipblaslt_headers_public
  include/hipblaslt.h
  include/hipblaslt-ext.hpp
  include/hipblaslt_xfloat32.h
  include/hipblaslt_float8.h
  include/hipblaslt_float8_bc.h
  include/hipblaslt-ext-op.h
  ${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt-version.h
)

if(NOT BUILD_CUDA)
    set(hipblaslt_headers_public ${hipblaslt_headers_public} include/hipblaslt-types.h)
endif()
source_group("Header Files\\Public" FILES ${hipblaslt_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

# Create hipBLASLt library
add_library(hipblaslt ${hipblaslt_source} ${RocRoller_SRC} ${hipblaslt_headers_public})
add_library(roc::hipblaslt ALIAS hipblaslt)

# Target compile definitions
if(NOT BUILD_CUDA)

  if( BUILD_SHARED_LIBS )
    target_link_libraries( hipblaslt PRIVATE TensileHost )
  else()
    target_compile_definitions( hipblaslt PRIVATE HIPBLASLT_STATIC_LIB )

    # bypassing cmake dependencies chain for static link as it won't allow target from different directory

    # including tensile headers into hipblaslt tensileHost client so get compile properties
    get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
    target_include_directories( hipblaslt PRIVATE ${TensileHost_INCLUDES} )
    get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
    target_compile_definitions( hipblaslt PRIVATE ${TensileHost_DEFINES} )

    get_target_property( TensileHost_LIBDIR TensileHost BINARY_DIR )
    get_target_property( TensileHost_SOURCES TensileHost SOURCES )
    get_target_property( TensileHost_SOURCE_DIR TensileHost SOURCE_DIR )

    message (STATUS "TensileHost_INCLUDES == ${TensileHost_INCLUDES}")
    message (STATUS "TensileHost_DEFINES == ${TensileHost_DEFINES}")
    message (STATUS "TensileHost_LIBDIR == ${TensileHost_LIBDIR}")
    message (STATUS "TensileHost_SOURCE_DIR == ${TensileHost_SOURCE_DIR}")
    message (STATUS "TensileHost_SOURCES == ${TensileHost_SOURCES}")
    cmake_path(SET seperator NORMALIZE "/")
    list(TRANSFORM TensileHost_SOURCES PREPEND "${TensileHost_SOURCE_DIR}${seperator}")
    # add dependent sources from TensileHost for static build
    target_sources(hipblaslt PRIVATE ${TensileHost_SOURCES})
  endif()

  target_compile_definitions(hipblaslt PRIVATE ${TENSILE_DEFINES} )

  target_compile_options(hipblaslt PRIVATE -Wno-unused-command-line-argument )
  target_compile_definitions(hipblaslt PRIVATE ROCM_USE_FLOAT16 __HIP_PLATFORM_AMD__)
else()
    target_compile_definitions(hipblaslt PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

if( LEGACY_HIPBLAS_DIRECT )
  target_compile_definitions(hipblaslt PUBLIC LEGACY_HIPBLAS_DIRECT )
endif()

# Target compile features
target_compile_features(hipblaslt PRIVATE cxx_nullptr)

# Target include directories
target_include_directories(hipblaslt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${Tensile_INC}>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                                   $<INSTALL_INTERFACE:include>
)

if( LEGACY_HIPBLAS_DIRECT )
  target_include_directories(hipblaslt
                             PUBLIC  $<BUILD_INTERFACE:${HIPBLAS_INCLUDE_DIRS}>
  )
else()
  target_include_directories(hipblaslt
                             PUBLIC  $<BUILD_INTERFACE:${HIPBLAS-COMMON_INCLUDE_DIRS}>
  )
endif()

if(BUILD_CUDA)
target_include_directories(hipblaslt
                           PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
else()
target_include_directories(hipblaslt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/include>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/amd_detail/rocblaslt/src/include>
)
endif()

# Target link libraries
if(NOT BUILD_CUDA)
# Target link libraries
  target_link_libraries(hipblaslt PRIVATE hip::device ${DL_LIB})
endif()

if(HIPBLASLT_ENABLE_MARKER)
  target_link_libraries(hipblaslt PRIVATE -lroctx64)
endif()

if( NOT BUILD_CUDA AND NOT LEGACY_HIPBLAS_DIRECT )
  target_link_libraries(hipblaslt INTERFACE roc::hipblas-common)
endif()

# RocRoller integration
include(FetchContent)
if(USE_ROCROLLER)
  _save_var(BUILD_TESTING)
  _save_var(BUILD_CLIENTS)
  set(BUILD_CLIENTS OFF)
  set(BUILD_TESTING OFF)

  FetchContent_Declare(
    rocRoller
    GIT_REPOSITORY https://github.com/ROCm/rocRoller.git
    GIT_TAG 34307b30956cac16e459db6cfdd3ba1f07c7e9b8
  )
  FetchContent_MakeAvailable(rocRoller)
  target_link_libraries(hipblaslt PRIVATE rocroller_interface)
  add_definitions(-DUSE_ROCROLLER)

  _restore_var(BUILD_TESTING)
  _restore_var(BUILD_CLIENTS)
endif()

# Target properties
rocm_set_soversion(hipblaslt ${hipblaslt_SOVERSION})
set_target_properties(hipblaslt PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(hipblaslt PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipblaslt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipblaslt PROPERTIES DEBUG_POSTFIX "-d")

# TODO ??
# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipblaslt PROPERTIES PREFIX "lib")
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipblaslt EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipblaslt/hipblaslt-export.h)

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_file(
    hipblaslt-version.h hipblaslt-export.h hipblaslt-ext.hpp
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include hipblaslt/include
  )
endif( )

# Install targets
rocm_install_targets(TARGETS hipblaslt
                     INCLUDE
                       ${CMAKE_BINARY_DIR}/include
)

if ( NOT BUILD_CUDA )
    if (WIN32)
      set( HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}hipblaslt/bin" CACHE PATH "path to tensile library" )
    else()
      set( HIPBLASLT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipblaslt" CACHE PATH "path to tensile library" )
    endif()
# For ASAN package, Tensile library files are not required
    if( NOT ENABLE_ASAN_PACKAGING )
    if( BUILD_SHARED_LIBS )
      set( TENSILE_DATA_COMPONENT_NAME ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME} )
    else()
      set( TENSILE_DATA_COMPONENT_NAME devel )
    endif()

      rocm_install(
        DIRECTORY ${CMAKE_BINARY_DIR}/Tensile/library
        DESTINATION ${HIPBLASLT_TENSILE_LIBRARY_DIR}
        COMPONENT ${TENSILE_DATA_COMPONENT_NAME})
    endif()
endif()

# Export targets
if(NOT BUILD_CUDA)
  if( LEGACY_HIPBLAS_DIRECT )
    rocm_export_targets(TARGETS roc::hipblaslt
                        DEPENDS PACKAGE hip
                        NAMESPACE roc::)
  else()
    rocm_export_targets(TARGETS roc::hipblaslt
                        DEPENDS PACKAGE hip
                        DEPENDS PACKAGE hipblas-common
                        NAMESPACE roc::)
  endif()
else()
  rocm_export_targets(TARGETS roc::hipblaslt
                      PREFIX hipblaslt
                      NAMESPACE roc::)
endif()


if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/hipblaslt"
        DESTINATION "." )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()
