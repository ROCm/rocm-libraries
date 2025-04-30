# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ########################################################################

# The following helper functions wrap common cmake functions.  They are
# used to cope with a few wierdnesses of hipcc/nvcc.
# ########################################################################
# HELPER FUNCTIONS
# ########################################################################


# ########################################################################
# A helper function to prefix a source list of files with a common path into a new list (non-destructive)
# ########################################################################
function( prepend_path prefix source_list_of_files return_list_of_files )
  foreach( file ${${source_list_of_files}} )
    if(IS_ABSOLUTE ${file} )
      list( APPEND new_list ${file} )
    else( )
      list( APPEND new_list ${prefix}/${file} )
    endif( )
  endforeach( )
  set( ${return_list_of_files} ${new_list} PARENT_SCOPE )
endfunction( )

# ########################################################################
# standardized library settings
# ########################################################################
function( rocblas_library_settings lib_target_ )

  message(STATUS "** rocblas_library_settings: ${lib_target_}")

  target_include_directories( ${lib_target_} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include/internal>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocblas/internal>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocblas>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<BUILD_INTERFACE:${Tensile_INC}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  # Do not allow Variable Length Arrays (use unique_ptr instead)
  target_compile_options( ${lib_target_} PRIVATE -Werror=vla )

  target_compile_definitions( ${lib_target_} PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_BETA_FEATURES_API )

  # both libraries will use rocblas_EXPORTS
  target_compile_definitions( ${lib_target_} PRIVATE rocblas_EXPORTS )

  set_target_properties( ${lib_target_} PROPERTIES CXX_EXTENSIONS NO )
  set_target_properties( ${lib_target_} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

  if(BUILD_OFFLOAD_COMPRESS AND CXX_COMPILER_SUPPORTS_OFFLOAD_COMPRESS)
    set_target_properties( ${lib_target_} PROPERTIES COMPILE_FLAGS "--offload-compress" )
  endif()

  target_link_libraries( ${lib_target_} INTERFACE hip::host )
  if (WIN32)
    target_link_libraries( ${lib_target_} PRIVATE hip::device )
  else()
    target_link_libraries( ${lib_target_} PRIVATE hip::device -lstdc++fs --rtlib=compiler-rt --unwindlib=libgcc)
    if (BUILD_SHARED_LIBS)
      target_link_libraries(${lib_target_} PRIVATE -lroctx64)
    endif()
  endif()
    target_link_libraries( ${lib_target_} PRIVATE Threads::Threads )
  #  -fno-gpu-rdc compiler option was used with hcc, so revisit feature at some point

  set_target_properties( ${lib_target_} PROPERTIES CXX_VISIBILITY_PRESET "hidden" C_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )

  if( NOT BUILD_SHARED_LIBS )
    # Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
    set_target_properties( ${lib_target_} PROPERTIES PREFIX "lib" )
  endif()

  rocm_set_soversion( ${lib_target_} ${rocblas_SOVERSION} )

endfunction()

# ########################################################################
# Main
# ########################################################################


set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# This is incremented when the ABI to the library changes
set( rocblas_SOVERSION 4.5 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-Bsymbolic")

# include( build-bitness )

# Print out compiler flags for viewing/debug
if( BUILD_VERBOSE )
  message( STATUS "rocfft_VERSION: ${rocfft_VERSION}" )
  message( STATUS "\t==>CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )
  message( STATUS "\t==>BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}" )
  message( STATUS "\t==>ROCM_PATH link: " ${ROCM_PATH} )
  message( STATUS "\t==>CMAKE_INSTALL_PREFIX link: " ${CMAKE_INSTALL_PREFIX} )
  message( STATUS "\t==>CMAKE_MODULE_PATH link: " ${CMAKE_MODULE_PATH} )
  message( STATUS "\t==>CMAKE_PREFIX_PATH link: " ${CMAKE_PREFIX_PATH} )
  message( STATUS "\t==>CPACK_PACKAGING_INSTALL_PREFIX link: " ${CPACK_PACKAGING_INSTALL_PREFIX} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_CXX_COMPILER: " ${CMAKE_CXX_FLAGS} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER debug: " ${CMAKE_CXX_FLAGS_DEBUG} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER release: " ${CMAKE_CXX_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_CXX_COMPILER relwithdebinfo: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS: " ${CMAKE_EXE_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_EXE_LINKER_FLAGS_RELEASE: " ${CMAKE_EXE_LINKER_FLAGS_RELEASE} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS} )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: " ${CMAKE_SHARED_LINKER_FLAGS_RELEASE} )
  message( STATUS "==============" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_C_FLAGS: ${CMAKE_SHARED_LIBRARY_C_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LIBRARY_CXX_FLAGS: ${CMAKE_SHARED_LIBRARY_CXX_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_DEBUG: ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}" )
  message( STATUS "\t==>CMAKE_SHARED_LINKER_FLAGS_RELEASE: ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}" )
endif( )

find_package(Git REQUIRED)

# Get the git hash of the rocBLAS branch
execute_process(
          COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
          WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
          OUTPUT_VARIABLE GIT_HASH_ROCBLAS
          OUTPUT_STRIP_TRAILING_WHITESPACE)

#set the rocBLAS commit hash
set(rocblas_VERSION_COMMIT_ID "${GIT_HASH_ROCBLAS}")

#set the Tensile commit hash
set(tensile_VERSION_COMMIT_ID "${tensile_tag}")

# log build commits
message( STATUS "*** Building rocBLAS commit: ${rocblas_VERSION_COMMIT_ID}" )
if (BUILD_WITH_TENSILE)
  if (NOT Tensile_TEST_LOCAL_PATH)
    message( STATUS "*** Building Tensile commit: ${tensile_tag}" )
  else()
    message( STATUS "*** Building Tensile path: ${Tensile_TEST_LOCAL_PATH}" )
  endif()
endif()

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/internal/rocblas-version.h.in"
	"${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h" )


set( rocblas_headers_public
  include/rocblas.h
  include/internal/rocblas-types.h
  include/internal/rocblas_bfloat16.h
  include/internal/rocblas_float8.h
  include/internal/rocblas-auxiliary.h
  include/internal/rocblas-functions.h
  include/internal/rocblas-macros.h
  include/internal/rocblas-beta.h
  ${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h
)

source_group( "Header Files\\Public" FILES ${rocblas_headers_public} )

# Build into subdirectories

# src contains original rocblas with Tensile and core functionality
add_subdirectory( src )


############################################################
# Installation

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/library/include ${PROJECT_BINARY_DIR}/include/rocblas)

# Force installation of .f90 module file
rocm_install(FILES "include/rocblas_module.f90"
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/rocblas/"
)

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_dir(
    ${CMAKE_SOURCE_DIR}/library/include
    PATTERNS "*.h"
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR}
  )
  rocm_wrap_header_file(
    internal/rocblas-version.h internal/rocblas-export.h internal/rocblas-exported-proto.hpp internal/rocblas_device_malloc.hpp
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR} rocblas/${CMAKE_INSTALL_INCLUDEDIR}
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/rocblas/internal/rocblas-version.h
  )

  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/rocblas"
        DESTINATION "." )

    #Create SymLink for Fortran Object Module for backward compatibility
    rocm_install(
      CODE "
        set(PREFIX \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX})
        set(INPUT_FILE \${PREFIX}/include/rocblas/rocblas_module.f90)
        set(SYMLINK_LOCATIONS \${PREFIX}/rocblas/include \${PREFIX}/include)
        foreach(LOCATION IN LISTS SYMLINK_LOCATIONS)
          file(MAKE_DIRECTORY \${LOCATION})
          execute_process(COMMAND ln -sfr \${INPUT_FILE} \${LOCATION})
          message(STATUS \"Created symlink in \${LOCATION} to \${INPUT_FILE}.\")
        endforeach()
        "
    )
endif()

############################################################
# Packaging

#         PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ

if( BUILD_SHARED_LIBS )
  rocm_export_targets(
    TARGETS roc::rocblas
    DEPENDS PACKAGE hip
    NAMESPACE roc::
    )
else()
  # static
  if (NOT BUILD_WITH_TENSILE OR (Tensile_LIBRARY_FORMAT MATCHES "msgpack"))
    # header only tensile usage of msgpack
    rocm_export_targets(
      TARGETS roc::rocblas
      DEPENDS PACKAGE hip
      NAMESPACE roc::
      )
  else()
    # yaml based Tensile uses LLVM
    rocm_export_targets(
      TARGETS roc::rocblas
      DEPENDS PACKAGE hip
      STATIC_DEPENDS PACKAGE LLVM
      NAMESPACE roc::
      )
  endif()
endif()

if(BUILD_WITH_HIPBLASLT)
  rocm_package_add_dependencies(SHARED_DEPENDS "hipblaslt >= ${HIPBLASLT_VERSION}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hipblaslt >= ${HIPBLASLT_VERSION}")
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "hipblaslt >= ${HIPBLASLT_VERSION}")
endif()
