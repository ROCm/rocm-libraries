# ########################################################################
# Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

# Configure a header file to pass the hipSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparselt-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-version.h"
)

# Copy Public Headers to Build Dir
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparselt.h" "${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt.h" COPYONLY)

# Public hipSPARSE headers
set(hipsparselt_headers_public
  include/hipsparselt.h
  ${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-version.h
)

source_group("Header Files\\Public" FILES ${hipsparselt_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

# Create hipSPARSELt library
add_library(hipsparselt ${hipsparselt_source} ${hipsparselt_headers_public})
add_library(roc::hipsparselt ALIAS hipsparselt)

# Target compile definitions
if(NOT BUILD_CUDA)
  target_compile_options(hipsparselt PRIVATE -Wno-unused-command-line-argument -Wall)
  target_compile_definitions(hipsparselt PRIVATE ROCM_USE_FLOAT16 __HIP_PLATFORM_AMD__)

  if( BUILD_WITH_TENSILE )
    if( BUILD_SHARED_LIBS )
      target_link_libraries( hipsparselt PRIVATE TensileHost )
    else()
      target_compile_definitions( hipsparselt PRIVATE HIPSPARSELT_STATIC_LIB )

      # bypassing cmake dependencies chain for static link as it won't allow target from different directory

      # including tensile headers into rocblas tensileHost client so get compile properties
      get_target_property(TensileHost_INCLUDES TensileHost INCLUDE_DIRECTORIES)
      target_include_directories( hipsparselt PRIVATE ${TensileHost_INCLUDES} )
      get_target_property(TensileHost_DEFINES TensileHost COMPILE_DEFINITIONS)
      target_compile_definitions( hipsparselt PRIVATE ${TensileHost_DEFINES} )

      get_target_property( TensileHost_LIBDIR TensileHost BINARY_DIR )

      message (STATUS "TensileHost_INCLUDES == ${TensileHost_INCLUDES}")
      message (STATUS "TensileHost_DEFINES == ${TensileHost_DEFINES}")
      message (STATUS "TensileHost_LIBDIR == ${TensileHost_LIBDIR}")
      # recreate LLVM static dependencies
      if (${Tensile_LIBRARY_FORMAT} STREQUAL "yaml")
        find_package(LLVM 6.0 QUIET CONFIG)
        if(NOT LLVM_FOUND)
            find_package(LLVM 7.0 QUIET CONFIG)
            if(NOT LLVM_FOUND)
                find_package(LLVM 9.0 QUIET CONFIG)
                if(NOT LLVM_FOUND)
                    find_package(LLVM REQUIRED CONFIG)
                endif()
            endif()
        endif()
        find_library(LLVMObjectYAML_LIBRARY
          NAMES LLVMObjectYAML
          PATHS ${LLVM_LIBRARY_DIR})
        message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
        target_link_libraries(hipsparselt PRIVATE LLVMObjectYAML )  # match tensile
      endif()

      # to get TensileHost built first, not to link target
      # as dependency chain can not be created
      add_dependencies(hipsparselt TensileHost)
    endif()
    target_compile_definitions(hipsparselt PRIVATE ${TENSILE_DEFINES} )
  endif()

else()
  target_compile_definitions(hipsparselt PRIVATE __HIP_PLATFORM_NVIDIA__)
endif()

# Target compile features
target_compile_features(hipsparselt PRIVATE cxx_nullptr)

# Target include directories
target_include_directories(hipsparselt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                                   $<BUILD_INTERFACE:${Tensile_INC}>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                                   $<INSTALL_INTERFACE:include>
)

if(BUILD_CUDA)
target_include_directories(hipsparselt
                           PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
)
else()
target_include_directories(hipsparselt
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/rocsparselt/include>
                                   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/hcc_detail/rocsparselt/src/include>
)
endif()

# Target link libraries
if(NOT BUILD_CUDA)
# Target link libraries
  target_link_libraries(hipsparselt PRIVATE hip::device ${DL_LIB})
else()
  find_library(CUDA_CUSPARSELT_LIBRARY NAMES cusparseLt PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64 REQUIRED)
  target_link_libraries(hipsparselt PRIVATE ${CUDA_CUSPARSELT_LIBRARY} ${CUDA_CUSPARSE_LIBRARY})
endif()

if(HIPSPARSELT_ENABLE_MARKER)
  target_link_libraries(hipsparselt PRIVATE -lroctx64)
endif()

# Target properties
rocm_set_soversion(hipsparselt ${hipsparselt_SOVERSION})
set_target_properties(hipsparselt PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(hipsparselt PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipsparselt PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipsparselt PROPERTIES DEBUG_POSTFIX "-d")

# TODO ??
# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipsparselt PROPERTIES PREFIX "lib")
endif()

# Package that helps me set visibility for function names exported from shared library
include( GenerateExportHeader )
generate_export_header( hipsparselt EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsparselt/hipsparselt-export.h)

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_file(
    hipsparselt-version.h hipsparselt-export.h
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include hipsparselt/include
  )
endif( )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( hipsparselt PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS hipsparselt
  INCLUDE
    ${CMAKE_BINARY_DIR}/include
)

if ( NOT BUILD_CUDA )
  if( BUILD_WITH_TENSILE )
    if (WIN32)
      set( HIPSPARSELT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/bin/hipsparselt" CACHE PATH "path to tensile library" )
    else()
      set( HIPSPARSELT_TENSILE_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipsparselt" CACHE PATH "path to tensile library" )
    endif()
    # For ASAN Enabled Build package only library & license
    if( NOT ENABLE_ASAN_PACKAGING )
      rocm_install(
        DIRECTORY ${CMAKE_BINARY_DIR}/Tensile/library
        DESTINATION ${HIPSPARSELT_TENSILE_LIBRARY_DIR}
        COMPONENT ${CMAKE_INSTALL_DEFAULT_COMPONENT_NAME}) # Use this cmake variable to be compatible with rocm-cmake 0.6 and 0.7
    endif()
  else()
    if (WIN32)
      set( HIPSPARSELT_SPMM_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/bin/hipsparselt" CACHE PATH "path to spmm kernels" )
    else()
      set( HIPSPARSELT_SPMM_LIBRARY_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}${CMAKE_INSTALL_LIBDIR}/hipsparselt" CACHE PATH "path to spmm kernels" )
    endif()
    # For ASAN Enabled Build package only library & license
    if( NOT ENABLE_ASAN_PACKAGING )
      rocm_install(
        DIRECTORY ${CMAKE_BINARY_DIR}/SPMM_KERNELS/library
        DESTINATION ${HIPSPARSELT_SPMM_LIBRARY_DIR}
        COMPONENT ${CMAKE_INSTALL_DEFAULT_COMPONENT})
    endif()
  endif()
endif()

if ( NOT BUILD_CUDA )
    rocm_export_targets(
        TARGETS roc::hipsparselt
	DEPENDS PACKAGE hip
	NAMESPACE roc::
    )
else( )
    rocm_export_targets(
        TARGETS roc::hipsparselt
	DEPENDS PACKAGE HIP
	NAMESPACE roc::
    )
endif( )

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/hipsparselt"
        DESTINATION "." )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()
