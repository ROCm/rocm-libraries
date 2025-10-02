# ########################################################################
# Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required( VERSION 3.24.4 )

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

# ROCM_BUILD_ID is added to the package name by rocm-cmake ROCMSetupVersion
unset(ENV{ROCM_BUILD_ID})

project( rocblas LANGUAGES CXX )

# Append our library helper cmake path and the cmake path for hip (for convenience)
list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )
include(build-options) # library and client
include(toolchain-options)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if(NOT python)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(python "${Python3_EXECUTABLE}")
endif()

# ########################################################################
# Main
# ########################################################################

set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )

# This finds the rocm-cmake project, and installs it if not found
include(get-rocm-cmake)

# rocm-cmake helpers
include( ROCMSetupVersion )
include( ROCMCreatePackage )
include( ROCMInstallTargets )
include( ROCMPackageConfigHelpers )
include( ROCMInstallSymlinks )
include( ROCMCheckTargetIds )
include( ROCMHeaderWrapper )
include( ROCMClients )


include (os-detection)
get_os_id(OS_ID)
message (STATUS "OS detected is ${OS_ID}")

# Versioning via rocm-cmake
set ( VERSION_STRING "5.1.0" )
rocm_setup_version( VERSION ${VERSION_STRING} )

# Users may override HIP path by specifying their own in CMAKE_MODULE_PATH
# NOTE:  workaround until llvm & hip cmake modules fixes symlink logic in their config files; remove when fixed
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/llvm ${ROCM_PATH} ${ROCM_PATH}/hip /opt/rocm/llvm /opt/rocm /opt/rocm/hip )

# setting target list based on ROCm version
if (NOT BUILD_ADDRESS_SANITIZER)
  set( TARGET_LIST_ROCM_5.6 "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102")
  set( TARGET_LIST_ROCM_5.7 "gfx803;gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102")
  set( TARGET_LIST_ROCM_6.0 "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack+;gfx90a:xnack-;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102")
  set( TARGET_LIST_ROCM_6.3 "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201")
  set( TARGET_LIST_ROCM_7.0 "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx950;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201")
  set( TARGET_LIST_ROCM_7.1 "gfx900;gfx906:xnack-;gfx908:xnack-;gfx90a;gfx942;gfx950;gfx1010;gfx1012;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201")
else()
  set( TARGET_LIST_ROCM_5.6 "gfx908:xnack+;gfx90a:xnack+")
  set( TARGET_LIST_ROCM_5.7 "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+")
  set( TARGET_LIST_ROCM_6.0 "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+")
  set( TARGET_LIST_ROCM_6.3 "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+")
  set( TARGET_LIST_ROCM_7.0 "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+")
  set( TARGET_LIST_ROCM_7.1 "gfx908:xnack+;gfx90a:xnack+;gfx942:xnack+;gfx950:xnack+")
endif()

if(ROCM_PLATFORM_VERSION)
  if(${ROCM_PLATFORM_VERSION} VERSION_LESS 5.7.0)
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_5.6}")
  elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 6.0.0)
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_5.7}")
  elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 6.3.0)
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_6.0}")
  elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 7.0.0)
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_6.3}")
  elseif(${ROCM_PLATFORM_VERSION} VERSION_LESS 7.1.0)
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_7.0}")
  else()
    set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_7.1}")
  endif()
  message(STATUS "ROCM Platform Version: ${ROCM_PLATFORM_VERSION} default supported gfx list")
else()
  message(STATUS "ROCM Platform Version: ${ROCM_PLATFORM_VERSION} is not set, using latest supported gfx list")
  set(SUPPORTED_TARGETS "${TARGET_LIST_ROCM_7.1}")
endif()

if( AMDGPU_TARGETS AND NOT GPU_TARGETS )
  message( DEPRECATION "AMDGPU_TARGETS use is deprecated. Use GPU_TARGETS." )
  set( GFX_LIST ${AMDGPU_TARGETS} )
elseif( GPU_TARGETS )
  set( GFX_LIST ${GPU_TARGETS} )
else ()
  set( GFX_LIST "all" ) # default if not set
endif()

# gpu arch configuration. GFX_LIST is ignored if already set by -DGPU_TARGETS
set( GPU_TARGETS ${GFX_LIST} CACHE STRING "List of GPUs (gfx targets) to support")
set_property(CACHE GPU_TARGETS PROPERTY STRINGS
  all
  ${SUPPORTED_TARGETS})

# Detect if target ID syntax if supported for default all GPU_TARGETS list
# Sets the GPU_TARGETS with backward compatibility
if (GPU_TARGETS STREQUAL "all")
  rocm_check_target_ids(target_list TARGETS "${SUPPORTED_TARGETS}")
else()
  rocm_check_target_ids(target_list TARGETS "${GPU_TARGETS}")
endif()

# if 'target_list' is empty, then compile does not support the gpu arch passed as an argument.
if(target_list STREQUAL "")
  message(FATAL_ERROR "Unsupported target ${GPU_TARGETS} by compiler, List of supported targets: ${SUPPORTED_TARGETS}")
endif()

# set GPU_TARGETS and AMDGPU_TARGETS before any find_package( hip ...), via tensile or in this file
# to set --offload-arch flags via hip-config.cmake hip::device dependency
set( GPU_TARGETS "${target_list}" CACHE STRING "List of GPUs (gfx targets) to support" FORCE )
set( AMDGPU_TARGETS "${GPU_TARGETS}" CACHE STRING "List of GPUs (gfx targets) to support" FORCE ) # for old hip cmake files backward compatibility

option( BUILD_WITH_HIPBLASLT "Build with HipBLASLt" ON )
set( hipblaslt_path "/opt/rocm" CACHE PATH "Use local HipBLASLt directory" )
set( HIPBLASLT_VERSION 1.0.0 CACHE STRING "The version of HipBLASLt to be used" )

if( NOT BUILD_WITH_TENSILE )
  set( BUILD_WITH_HIPBLASLT OFF )
endif()

if(BUILD_WITH_HIPBLASLT)
  # don't wait until after Tensile to find hipblaslt if enabled; needed for clients-only build too
  find_package( hipblaslt ${HIPBLASLT_VERSION} REQUIRED CONFIG PATHS ${hipblaslt_path} ${ROCM_PATH})
  message(STATUS "FOUND hipBLASLt, INCLUDE_DIR: ${HIPBLASLT_INCLUDE_DIR}")
endif()

function(list_replace list old new)
  foreach( i ${${list}} )
    if( i STREQUAL ${old} )
      list( APPEND new_list ${new} )
    else( )
      list( APPEND new_list ${i} )
    endif( )
    set( ${list} ${new_list} PARENT_SCOPE )
  endforeach( )
endfunction( )

if(NOT SKIP_LIBRARY)
  if( BUILD_WITH_TENSILE )
    # we will have expanded "all" for tensile to ensure consistency as we have local rules
    set( Tensile_ARCHITECTURE "${GPU_TARGETS}" CACHE STRING "Tensile to use which architecture?" FORCE)

    list_replace( Tensile_ARCHITECTURE "gfx90a" "gfx90a:xnack+\;gfx90a:xnack-")

    set( TENSILE_VERSION 4.44.0 CACHE STRING "The version of Tensile to be used")

    if(BUILD_WITH_PIP)
      if (WIN32)
        set( Tensile_ROOT "${CMAKE_BINARY_DIR}/virtualenv/Lib/site-packages/Tensile" )
      endif()

      set( Tensile_TEST_LOCAL_PATH "" CACHE PATH "Use local Tensile directory instead of fetching a GitHub branch" )

      include(virtualenv)

      set(ROCBLAS_TENSILE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../shared/tensile" CACHE PATH "Path to monorepo Tensile directory")

      if (Tensile_TEST_LOCAL_PATH)
        virtualenv_install(${Tensile_TEST_LOCAL_PATH})
        message (STATUS "using local Tensile from ${Tensile_TEST_LOCAL_PATH}, copied to ${Tensile_ROOT}")
      elseif (IS_DIRECTORY "${ROCBLAS_TENSILE_DIR}")
        virtualenv_install(${ROCBLAS_TENSILE_DIR})
        message (STATUS "using local path to shared Tensile from ${ROCBLAS_TENSILE_DIR}, copied to ${Tensile_ROOT}")
      else()
        message(WARNING "deprecated fetch from legacy Tensile repository")
        # Use the virtual-env setup and download package from legacy repo, requires tensile tag file creation or tensile_tag cache variable
        set( tensile_fork "ROCm" CACHE STRING "Tensile fork to use (deprecated)" )
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tensile_tag.txt")
          file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/tensile_tag.txt" read_tensile_tag)
          set( tensile_tag ${read_tensile_tag} CACHE STRING "Tensile tag to download" )
        else()
          set( tensile_tag "develop" CACHE STRING "Default Tensile tag to download (deprecated)" )
          message(WARNING "tensile_tag.txt not found, using default Tensile tag: ${tensile_tag}")
        endif()
        virtualenv_install("git+https://github.com/${tensile_fork}/Tensile.git@${tensile_tag}")
        message (STATUS "using GIT Tensile fork=${tensile_fork} from branch=${tensile_tag}")
      endif()
      message(STATUS "Adding ${VIRTUALENV_HOME_DIR} to CMAKE_PREFIX_PATH")
      list(APPEND CMAKE_PREFIX_PATH ${VIRTUALENV_HOME_DIR})
    endif()
    find_package(Tensile ${TENSILE_VERSION} EXACT REQUIRED HIP LLVM)
  endif()
endif()

message(STATUS "Using GPU_TARGETS: ${GPU_TARGETS}")

# Find HIP dependencies
if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
  find_package( hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm )
endif( )


# setup rocblas defines used for both the library and clients
if( BUILD_WITH_TENSILE )
    list(APPEND TENSILE_DEFINES BUILD_WITH_TENSILE)
    if( Tensile_SEPARATE_ARCHITECTURES )
        list(APPEND TENSILE_DEFINES ROCBLAS_TENSILE_SEPARATE_ARCH=1)
    else()
        list(APPEND TENSILE_DEFINES ROCBLAS_TENSILE_SEPARATE_ARCH=0)
    endif()
    if( Tensile_LAZY_LIBRARY_LOADING )
        list(APPEND TENSILE_DEFINES ROCBLAS_TENSILE_LAZY_LOAD=1)
  else()
        list(APPEND TENSILE_DEFINES ROCBLAS_TENSILE_LAZY_LOAD=0)
  endif()
endif()

if( BUILD_WITH_HIPBLASLT )
  list( APPEND HIPBLASLT_DEFINES BUILD_WITH_HIPBLASLT )
  rocm_package_add_dependencies(DEPENDS "hipblaslt >= ${HIPBLASLT_VERSION}")
endif()

if( BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS )
  if(NOT CLIENTS_OS)
    rocm_set_os_id(CLIENTS_OS)
    string(TOLOWER "${CLIENTS_OS}" CLIENTS_OS)
    rocm_read_os_release(CLIENTS_OS_VERSION VERSION_ID)
  endif()
  message(STATUS "OS: ${CLIENTS_OS} ${CLIENTS_OS_VERSION}")
  set(OPENMP_RPM "libgomp")
  set(OPENMP_DEB "libomp-dev")
  set(GFORTRAN_RPM "libgfortran4")
  set(GFORTRAN_DEB "libgfortran4")
  if(CLIENTS_OS STREQUAL "centos" OR CLIENTS_OS STREQUAL "rhel" OR CLIENTS_OS STREQUAL "almalinux")
    if(CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "8")
      set(GFORTRAN_RPM "libgfortran")
    endif()
  elseif(CLIENTS_OS STREQUAL "ubuntu" AND CLIENTS_OS_VERSION VERSION_GREATER_EQUAL "20.04")
    set(GFORTRAN_DEB "libgfortran5")
  elseif(CLIENTS_OS STREQUAL "sles")
    set(OPENMP_RPM "libgomp1")
  elseif(CLIENTS_OS STREQUAL "mariner" OR CLIENTS_OS STREQUAL "azurelinux")
    set(GFORTRAN_RPM "gfortran")
  endif()

  set( BUILD_CLIENTS ON )
  rocm_package_setup_component(clients)
  rocm_package_setup_client_component( clients-common )
  if(BUILD_CLIENTS_TESTS)
    rocm_package_setup_client_component(
      tests
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()
  if(BUILD_CLIENTS_BENCHMARKS)
    rocm_package_setup_client_component(
      benchmarks
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()
  if(BUILD_CLIENTS_SAMPLES)
    rocm_package_setup_client_component(
      samples
      DEPENDS
        COMPONENT clients-common
        DEB "${OPENMP_DEB}"
        RPM "${OPENMP_RPM}"
    )
  endif()
  if(BUILD_FORTRAN_CLIENTS)
    rocm_package_add_rpm_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT tests DEPENDS "${GFORTRAN_DEB}")
    rocm_package_add_rpm_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_RPM}")
    rocm_package_add_deb_dependencies(COMPONENT benchmarks DEPENDS "${GFORTRAN_DEB}")
  endif()
endif()

if( NOT SKIP_LIBRARY )
    add_subdirectory( library )
endif()

# Build clients of the library
if( BUILD_CLIENTS )
  add_subdirectory( clients )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
    set( CPACK_SOURCE_GENERATOR "ZIP" )
    set( CPACK_GENERATOR "ZIP" )
endif( )

# Package specific CPACK vars
# As of ROCm 3.8, HIP_RUNTIME has been changed from "ROCclr" to "rocclr"
# As of ROCm 3.8, HIP_RUNTIME has been changed from "cuda" to "cudart"
if( HIP_RUNTIME MATCHES ".*cuda.*" )
  rocm_package_add_dependencies(DEPENDS "hip-nvcc >= 3.5.0")
else()
  if(BUILD_ADDRESS_SANITIZER)
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd-asan" )
  else()
    set(DEPENDS_HIP_RUNTIME "hip-runtime-amd" )
  endif()
  set(HIP_RUNTIME_MINIMUM 4.5.0)
  rocm_package_add_dependencies(SHARED_DEPENDS "${DEPENDS_HIP_RUNTIME} >= ${HIP_RUNTIME_MINIMUM}")
  ## add dependency on hip runtime for static libraries
  rocm_package_add_deb_dependencies(STATIC_DEPENDS "hip-static-dev >= ${HIP_RUNTIME_MINIMUM}")
  rocm_package_add_rpm_dependencies(STATIC_DEPENDS "hip-static-devel >= ${HIP_RUNTIME_MINIMUM}")
endif()

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md" )
set( CPACK_RPM_PACKAGE_LICENSE "MIT and BSD")

if (WIN32)
  if( CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
    SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  endif()
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  if( NOT CPACK_PACKAGING_INSTALL_PREFIX )
    set( CPACK_PACKAGING_INSTALL_PREFIX ${ROCM_PATH} )
  endif()
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" )

# work around code object stripping failure if using /usr/bin/strip
set( CPACK_RPM_SPEC_MORE_DEFINE "%define __strip ${rocm_bin}/../llvm/bin/llvm-strip")

# Give rocblas compiled for CUDA backend a different name
if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    set( package_name rocblas )
else( )
    set( package_name rocblas-alt )
endif( )

set( ROCBLAS_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file" )

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "AMD library for BLAS in ROCm. Implemented in HIP and optimized for AMD GPUs"
    MAINTAINER "rocBLAS Maintainer <rocblas-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCBLAS_CONFIG_DIR}
)

#
# ADDITIONAL TARGETS FOR CODE COVERAGE
if(BUILD_CODE_COVERAGE)
  #
  # > make coverage_cleanup (clean coverage related files.)
  #
  # > make coverage
  # will run:
  #  > make coverage_analysis (analyze tests)
  #  > make coverage (generate html documentation)
  #

  #
  # Run coverage analysis (rtest.xml sets specify coverage runs)
  #
  add_custom_target(coverage_analysis
    COMMAND ${CMAKE_COMMAND} -E make_directory ./coverage-report/profraw
    COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE="./coverage-report/profraw/rocblas-coverage_%m.profraw" ${python} ./clients/staging/rocblas_rtest.py -t code_coverage
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

  add_dependencies(coverage_analysis rocblas)

  find_program(
    LLVM_PROFDATA
    llvm-profdata
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  find_program(
    LLVM_COV
    llvm-cov
    REQUIRED
    HINTS ${ROCM_PATH}/llvm/bin
    PATHS /opt/rocm/llvm/bin
  )

  add_custom_target(
    coverage
    DEPENDS coverage_analysis
    COMMAND ${LLVM_PROFDATA} merge -sparse ./coverage-report/profraw/rocblas-coverage_*.profraw -o ./coverage-report/rocblas.profdata
    COMMAND ${LLVM_COV} report -object ./library/src/librocblas.so -instr-profile=./coverage-report/rocblas.profdata
    COMMAND ${LLVM_COV} show -object ./library/src/librocblas.so -instr-profile=./coverage-report/rocblas.profdata -format=html -output-dir=coverage-report
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  #
  # Coverage cleanup
  #
  add_custom_target(coverage_cleanup
    COMMAND find ${CMAKE_BINARY_DIR} -name *.gcda -delete
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# CMake to build documentation
## Sphinx
set(SPHINX_DIR "docs/sphinx" CACHE PATH "Path to sphinx root directory")
set(SPHINX_BUILD_DIR "_build" CACHE PATH "Path to sphinx build directory")
set(SPHINX_BUILD_TYPE "html" CACHE STRING "Sphinx build type")
set(SPHINX_EXTRA_ARGS "" CACHE STRING "Extra arguments for Sphinx build")
set(
    SPHINX_ARGS
    -T
    -b ${SPHINX_BUILD_TYPE}
    -d ${SPHINX_BUILD_DIR}/doctrees
    -D language=en
    ${SPHINX_EXTRA_ARGS}
)
add_custom_target(
    sphinx_docs
    COMMAND pip3 install -r requirements.txt
    COMMAND python3 -m sphinx ${SPHINX_ARGS} . ${SPHINX_BUILD_DIR}/${SPHINX_BUILD_TYPE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${SPHINX_DIR}
)

## Doxygen
set(DOXYGEN_DIR "docs/doxygen" CACHE PATH "Path to doxygen root directory")
### Any other doxygen arguments can go here
find_package(Doxygen)
if (DOXYGEN_FOUND)
    add_custom_target(
        doxygen_docs
        Doxygen::doxygen Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/${DOXYGEN_DIR}
    )
endif()

## All docs phony target
add_custom_target(docs DEPENDS sphinx_docs)
if (DOXYGEN_FOUND)
    add_dependencies(docs doxygen_docs)
endif()
