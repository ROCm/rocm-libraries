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

cmake_minimum_required( VERSION 3.16.8 )

# Apply OpenMP settings and RPATH configuration to a target
# Handles both BUILD and INSTALL RPATH for OpenMP and BLAS libraries
function( apply_omp_settings lib_target_ )
  # Collect RPATH entries for both BUILD and INSTALL
  set(build_rpath_list "")
  set(install_rpath_list "")

  # Add OpenMP library paths
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::OpenMP_CXX)
    list(APPEND build_rpath_list "${HIP_CLANG_ROOT}/lib")
    list(APPEND install_rpath_list "$ORIGIN/../llvm/lib")
    target_compile_options( ${lib_target_} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}> )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
    list(APPEND build_rpath_list "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}")
    list(APPEND install_rpath_list "$ORIGIN/../llvm/${openmp_LIB_DIR}")
    target_compile_options( ${lib_target_} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}> )
  endif()

  # Add AOCL library path if it's a shared library (.so)
  if (BLAS_LIBRARY MATCHES "\\.so$")
    get_filename_component(blas_lib_dir "${BLAS_LIBRARY}" DIRECTORY)
    list(APPEND build_rpath_list "${blas_lib_dir}")
    # For install RPATH, use relative path if BLAS is in build deps
    if (BLAS_LIBRARY MATCHES "${BUILD_DIR}/deps/")
      list(APPEND install_rpath_list "$ORIGIN/../lib")
    else()
      list(APPEND install_rpath_list "${blas_lib_dir}")
    endif()
  endif()

  # Apply collected RPATH entries if any
  if (build_rpath_list)
    list(REMOVE_DUPLICATES build_rpath_list)
    string(REPLACE ";" ":" build_rpath_str "${build_rpath_list}")
    set_target_properties( ${lib_target_} PROPERTIES BUILD_RPATH "${build_rpath_str}" )
  endif()

  if (install_rpath_list)
    list(REMOVE_DUPLICATES install_rpath_list)
    string(REPLACE ";" ":" install_rpath_str "${install_rpath_list}")
    set_target_properties( ${lib_target_} PROPERTIES INSTALL_RPATH "${install_rpath_str}" )
  endif()
endfunction()

# Search for AOCL 4.x BLIS library for the specified compiler (gcc or aocc)
# Sets BLAS_LIBRARY and BLIS_INCLUDE_DIR in parent scope if found
function(find_libblis compiler)
  file(GLOB DIRS "/opt/AMD/aocl/aocl-linux-${compiler}-*")
  list(SORT DIRS)
  list(REVERSE DIRS)

  foreach(dir IN LISTS DIRS)
    get_filename_component(dir_name "${dir}" NAME)

    if(dir_name MATCHES "aocl-linux-${compiler}-([0-9]+)\\.([0-9]+)\\.([0-9]+)")
      set(library_path "${dir}/${compiler}/lib_ILP64/libblis-mt.a")
      if(EXISTS "${library_path}")
        set(BLAS_LIBRARY "${library_path}" PARENT_SCOPE)
        set(BLIS_INCLUDE_DIR "${dir}/${compiler}/include_ILP64" PARENT_SCOPE)
        string(TOUPPER ${compiler} compiler_upper)
        message(STATUS "✓ Found AOCL 4.x BLIS library (${compiler_upper}, ILP64)")
        message(STATUS "  Location: ${library_path}")
        return()
      endif()
    endif()
  endforeach()
endfunction()

# Helper: Search $HOME/aocl for AOCL 5.x libaocl libraries
# Sets BLAS_LIBRARY and BLIS_INCLUDE_DIR in parent scope if found
function(find_libaocl_in_home)
  if(NOT DEFINED ENV{HOME})
    return()
  endif()

  file(GLOB AOCL_DIRS "$ENV{HOME}/aocl/*/*") # e.g. $HOME/aocl/5.1.0/gcc
  foreach(dir IN LISTS AOCL_DIRS)
    foreach(libdir IN ITEMS lib lib64)
      foreach(ext IN ITEMS so a)
        set(candidate "${dir}/${libdir}/libaocl.${ext}")
        if(EXISTS "${candidate}")
          set(BLAS_LIBRARY "${candidate}" PARENT_SCOPE)
          if(EXISTS "${dir}/include_ILP64")
            set(BLIS_INCLUDE_DIR "${dir}/include_ILP64" PARENT_SCOPE)
          elseif(EXISTS "${dir}/include")
            set(BLIS_INCLUDE_DIR "${dir}/include" PARENT_SCOPE)
          endif()
          message(STATUS "✓ Found AOCL 5.x unified library (in $HOME/aocl/)")
          message(STATUS "  Location: ${candidate}")
          return()
        endif()
      endforeach()
    endforeach()
  endforeach()
endfunction()

# Helper: Search /opt/AMD/aocl for AOCL 5.x or 4.x libaocl libraries
# Sets BLAS_LIBRARY and BLIS_INCLUDE_DIR in parent scope if found
function(find_libaocl_in_opt_amd)
  # First, try AOCL 5.x version-based directories (e.g., /opt/AMD/aocl/5.1.0/)
  file(GLOB OPT_AOCL5_DIRS "/opt/AMD/aocl/[0-9]*")
  list(SORT OPT_AOCL5_DIRS)
  list(REVERSE OPT_AOCL5_DIRS) # Prefer newer versions
  foreach(dir IN LISTS OPT_AOCL5_DIRS)
    foreach(libdir IN ITEMS lib lib64)
      foreach(ext IN ITEMS so a)
        set(candidate "${dir}/${libdir}/libaocl.${ext}")
        if(EXISTS "${candidate}")
          set(BLAS_LIBRARY "${candidate}" PARENT_SCOPE)
          if(EXISTS "${dir}/include_ILP64")
            set(BLIS_INCLUDE_DIR "${dir}/include_ILP64" PARENT_SCOPE)
          elseif(EXISTS "${dir}/include")
            set(BLIS_INCLUDE_DIR "${dir}/include" PARENT_SCOPE)
          endif()
          message(STATUS "✓ Found AOCL 5.x unified library (in /opt/AMD/aocl/)")
          message(STATUS "  Location: ${candidate}")
          return()
        endif()
      endforeach()
    endforeach()
  endforeach()

  # Second, try legacy AOCL 4.x aocl-linux-* directories (rare for libaocl)
  file(GLOB OPT_AOCL_DIRS "/opt/AMD/aocl/aocl-linux-*")
  list(SORT OPT_AOCL_DIRS)
  list(REVERSE OPT_AOCL_DIRS) # Prefer newer versions
  foreach(dir IN LISTS OPT_AOCL_DIRS)
    foreach(libdir IN ITEMS lib lib64)
      foreach(ext IN ITEMS so a)
        set(candidate "${dir}/${libdir}/libaocl.${ext}")
        if(EXISTS "${candidate}")
          set(BLAS_LIBRARY "${candidate}" PARENT_SCOPE)
          if(EXISTS "${dir}/include_ILP64")
            set(BLIS_INCLUDE_DIR "${dir}/include_ILP64" PARENT_SCOPE)
          elseif(EXISTS "${dir}/include")
            set(BLIS_INCLUDE_DIR "${dir}/include" PARENT_SCOPE)
          endif()
          message(STATUS "✓ Found AOCL 4.x unified library (in /opt/AMD/aocl/)")
          message(STATUS "  Location: ${candidate}")
          return()
        endif()
      endforeach()
    endforeach()
  endforeach()
endfunction()

# Unified AOCL (libaocl) detection ---
# Looks for libaocl.{so,a} and sets BLAS_LIBRARY and BLIS_INCLUDE_DIR in parent scope.
function(find_libaocl)
  # 1) Your build dependency layout (e.g., deps/aocl/install_package)
  foreach(ext IN ITEMS so a)
    set(candidate "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.${ext}")
    if(EXISTS "${candidate}")
      set(BLAS_LIBRARY "${candidate}" PARENT_SCOPE)
      # Heuristic: AOCL unified "install_package" usually has a flat include folder
      set(BLIS_INCLUDE_DIR "${BUILD_DIR}/deps/aocl/install_package/include" PARENT_SCOPE)
      message(STATUS "✓ Found AOCL 5.x unified library")
      message(STATUS "  Location: ${candidate}")
      return()
    endif()
  endforeach()

  # 2) Respect user override: AOCL_ROOT
  if(DEFINED ENV{AOCL_ROOT})
    foreach(libdir IN ITEMS lib lib64)
      foreach(ext IN ITEMS so a)
        set(candidate "$ENV{AOCL_ROOT}/${libdir}/libaocl.${ext}")
        if(EXISTS "${candidate}")
          set(BLAS_LIBRARY "${candidate}" PARENT_SCOPE)
          # Prefer ILP64 includes if present, else generic include
          if(EXISTS "$ENV{AOCL_ROOT}/include_ILP64")
            set(BLIS_INCLUDE_DIR "$ENV{AOCL_ROOT}/include_ILP64" PARENT_SCOPE)
          elseif(EXISTS "$ENV{AOCL_ROOT}/include")
            set(BLIS_INCLUDE_DIR "$ENV{AOCL_ROOT}/include" PARENT_SCOPE)
          endif()
          message(STATUS "✓ Found AOCL 5.x unified library (via AOCL_ROOT environment variable)")
          message(STATUS "  Location: ${candidate}")
          return()
        endif()
      endforeach()
    endforeach()
  endif()

  # 3) Default AOCL 5.x install location: $HOME/aocl/<version>/{aocc|gcc}/lib{,64}
  find_libaocl_in_home()
  if(BLAS_LIBRARY)
    set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()

  # 4) System AOCL in /opt/AMD/aocl (searches 5.x then 4.x)
  find_libaocl_in_opt_amd()
  if(BLAS_LIBRARY)
    set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()
endfunction()

# Search for bundled BLIS libraries built by install.sh
# Sets BLAS_LIBRARY, BLIS_INCLUDE_DIR, and WARN_NOT_ILP64_PREFERRED in parent scope
function(find_bundled_blis)
  if (EXISTS "${BUILD_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a") # 4.0 and 4.1.0
    set(BLAS_LIBRARY "${BUILD_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BUILD_DIR}/deps/amd-blis/include/ILP64" PARENT_SCOPE)
    set(WARN_NOT_ILP64_PREFERRED true PARENT_SCOPE)
    message(STATUS "✓ Found bundled AMD-BLIS (ILP64)")
    message(STATUS "  Location: ${BUILD_DIR}/deps/amd-blis")
  elseif (EXISTS "${BUILD_DIR}/deps/blis/lib/libblis.a")
    set(BLAS_LIBRARY "${BUILD_DIR}/deps/blis/lib/libblis.a" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BUILD_DIR}/deps/blis/include/blis" PARENT_SCOPE)
    set(WARN_NOT_ILP64_PREFERRED true PARENT_SCOPE)
    message(STATUS "✓ Found bundled BLIS (downloaded by install.sh)")
    message(STATUS "  Location: ${BUILD_DIR}/deps/blis")
  elseif (EXISTS "/usr/local/lib/libblis.a")
    set(BLAS_LIBRARY "/usr/local/lib/libblis.a" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "/usr/local/include/blis" PARENT_SCOPE)
    message(STATUS "✓ Found system-installed BLIS")
    message(STATUS "  Location: /usr/local/lib/libblis.a")
  endif()
endfunction()

# Search for system cblas via pkg-config
# Sets BLAS_LIBRARY and BLIS_INCLUDE_DIR in parent scope, or errors if not found
function(find_system_cblas)
  find_package(PkgConfig)
  if (NOT PKG_CONFIG_FOUND)
    message(FATAL_ERROR "Could not find libaocl/libblis and pkg-config is not available")
  endif()

  pkg_search_module(PKGBLAS cblas)
  if (PKGBLAS_FOUND)
    set(BLAS_LIBRARY "${PKGBLAS_LIBRARIES}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${PKGBLAS_INCLUDE_DIRS}" PARENT_SCOPE)
    message(STATUS "✓ Found system CBLAS via pkg-config (likely OpenBLAS)")
    message(STATUS "  Libraries: ${PKGBLAS_LIBRARIES}")
    message(STATUS "  Include: ${PKGBLAS_INCLUDE_DIRS}")
  else()
    message(FATAL_ERROR "Could not find libaocl/libblis and pkg-config cannot find any other implementation of cblas")
  endif()
endfunction()

# Main BLAS library search orchestrator
# Tries in order: AOCL unified -> AOCL-BLIS (gcc) -> AOCL-BLIS (aocc) -> bundled BLIS -> pkg-config cblas
# Sets BLAS_LIBRARY, BLIS_INCLUDE_DIR, and WARN_NOT_ILP64_PREFERRED in parent scope
function(setup_blas_library)
  message(STATUS "=====================================")
  message(STATUS "Searching for BLAS library (LINK_BLIS=ON)...")
  set(BLAS_LIBRARY "" PARENT_SCOPE)
  set(BLIS_INCLUDE_DIR "" PARENT_SCOPE)
  set(WARN_NOT_ILP64_PREFERRED false PARENT_SCOPE)

  # 1. Try AOCL unified library (libaocl)
  find_libaocl()
  if (BLAS_LIBRARY)
    set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()

  # 2. Try AOCL-BLIS with GCC (preferred, ILP64)
  find_libblis(gcc)
  if (BLAS_LIBRARY)
    set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
    return()
  endif()

  # 3. Try AOCL-BLIS with AOCC
  find_libblis(aocc)
  if (BLAS_LIBRARY)
    set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
    set(WARN_NOT_ILP64_PREFERRED true PARENT_SCOPE)
    return()
  endif()

  # 4. Try bundled builds or system BLIS
  find_bundled_blis()
  if (BLAS_LIBRARY)
    set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
    set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
    set(WARN_NOT_ILP64_PREFERRED "${WARN_NOT_ILP64_PREFERRED}" PARENT_SCOPE)
    return()
  endif()

  # 5. Last resort: try pkg-config cblas
  find_system_cblas()
  set(BLAS_LIBRARY "${BLAS_LIBRARY}" PARENT_SCOPE)
  set(BLIS_INCLUDE_DIR "${BLIS_INCLUDE_DIR}" PARENT_SCOPE)
endfunction()

# This has to be initialized before the project() command appears
# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT DEFINED CMAKE_CONFIGURATION_TYPES AND NOT DEFINED CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." )
endif()

if ( NOT DEFINED CMAKE_Fortran_COMPILER AND NOT DEFINED ENV{FC} )
  set( CMAKE_Fortran_COMPILER "gfortran" )
endif()

# This project may compile dependencies for clients
if ( BUILD_FORTRAN_CLIENTS )
  set( fortran_language "Fortran" )
endif()
project( rocblas-clients LANGUAGES CXX C ${fortran_language} )

# Note: BUILD_DIR is a custom variable passed by rmake.py (via -DBUILD_DIR=<build_path>)
# to locate bundled dependencies. Use rmake.py or install.sh to build, not cmake directly.

if ( BUILD_FORTRAN_CLIENTS )
  set(rocblas_f90_source_clients
      include/rocblas_fortran.f90
  )

  # Set Fortran module output directory
  set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/rocblas/internal)

  # Create rocBLAS Fortran module
  add_library(rocblas_fortran OBJECT ../library/include/rocblas_module.f90)

  if( BUILD_WITH_TENSILE )
    list( APPEND rocblas_f90_source_clients include/rocblas_fortran_tensile.f90 )
  endif()

  add_library(rocblas_fortran_client STATIC ${rocblas_f90_source_clients} $<TARGET_OBJECTS:rocblas_fortran>)
else()
  set( rocblas_fortran_client "")
endif()

if( SKIP_LIBRARY )
  include_directories(${ROCBLAS_LIBRARY_DIR}/include/rocblas)
  include_directories(${ROCBLAS_LIBRARY_DIR}/include/rocblas/internal)
else()
  include_directories(${CMAKE_BINARY_DIR}/include/rocblas)
  include_directories(${CMAKE_BINARY_DIR}/include/rocblas/internal)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Look for openmp config in ROCm install to populate openmp_LIB_DIR and openmp_LIB_INSTALL_DIR
  find_package(OpenMP CONFIG PATHS "${HIP_CLANG_ROOT}/lib/cmake")
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
  set( COMMON_LINK_LIBS "OpenMP::omp")
  message(STATUS "Found openmp-config.cmake at ${OpenMP_DIR}")
  list( APPEND COMMON_CXX_OPTIONS "-fopenmp") # this is linux only target
else()
  # if it fails to find OpenMP compile and link flags in strange configurations it can just use non-parallel reference computation
  # if there is no omp.h to find the client compilation will fail and this should be obvious, used to be REQUIRED
  find_package(OpenMP)
  if (TARGET OpenMP::OpenMP_CXX)
    set( COMMON_LINK_LIBS "OpenMP::OpenMP_CXX")
  endif()
endif()

if (TARGET Threads::Threads)
  list( APPEND COMMON_LINK_LIBS "Threads::Threads")
endif()

message(STATUS "CLIENT COMMON CXX_OPTIONS: ${COMMON_CXX_OPTIONS}")
message(STATUS "CLIENT COMMON LINK: ${COMMON_LINK_LIBS}")

list( APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

include( client-build-options )

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

if( NOT TARGET rocblas )
  find_package( rocblas REQUIRED CONFIG PATHS ${ROCM_PATH} /opt/rocm ${ROCM_PATH}/rocblas /opt/rocm/rocblas ${ROCBLAS_LIBRARY_DIR})
endif( )

# Hip headers required of all clients; clients use hip to allocate device memory
list( APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} /opt/rocm )
if ( NOT hip_FOUND )
  find_package( hip REQUIRED CONFIG PATHS ${ROCM_PATH} )
endif( )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if ( NOT WIN32 )
    if (LINK_BLIS)
      setup_blas_library()
    else()
      find_package( BLAS REQUIRED )
      set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
      message(STATUS "LINK_BLIS=OFF: Using CMake FindBLAS module")
    endif()
  else() # WIN32
    file(TO_CMAKE_PATH "C:/Program\ Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT.lib" AOCL_BLAS_LIBRARY)
    if (LINK_BLIS AND EXISTS ${AOCL_BLAS_LIBRARY})
      set( BLAS_LIBRARY "-l\"C:/Program\ Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT\"" )
      set( BLIS_INCLUDE_DIR "C:/Program\ Files/AMD/AOCL-Windows/amd-blis/include/ILP64" )
      set( BLIS_DEFINES BLIS_ENABLE_NO_UNDERSCORE_API BLIS_ENABLE_CBLAS )
    else()
      set( BLAS_INCLUDE_DIR ${OPENBLAS_DIR}/include CACHE PATH "OpenBLAS library include path" )
      find_library( BLAS_LIBRARY libopenblas
                    PATHS ${OPENBLAS_DIR}/lib
                    NO_DEFAULT_PATH
                  )
      if (NOT BLAS_LIBRARY)
        find_package( OpenBLAS CONFIG REQUIRED )
        set( BLAS_LIBRARY OpenBLAS::OpenBLAS )
        set( BLAS_INCLUDE_DIR "" )
      endif()
      set( WARN_NOT_ILP64_PREFERRED true )
    endif()
  endif()

  # Print BLAS library configuration summary
  message(STATUS "=====================================")
  message(STATUS "BLAS Library Configuration Summary:")
  message(STATUS "  BLAS_LIBRARY: ${BLAS_LIBRARY}")
  if (DEFINED BLIS_INCLUDE_DIR AND BLIS_INCLUDE_DIR)
    message(STATUS "  BLIS_INCLUDE_DIR: ${BLIS_INCLUDE_DIR}")
  endif()
  if (DEFINED BLAS_INCLUDE_DIR AND BLAS_INCLUDE_DIR)
    message(STATUS "  BLAS_INCLUDE_DIR: ${BLAS_INCLUDE_DIR}")
  endif()
  if (WARN_NOT_ILP64_PREFERRED)
    message(STATUS "  ILP64 Support: No (LP64/32-bit integers)")
    message(WARNING "Using non-ILP64 BLAS library. Large matrix operations may overflow.")
  else()
    message(STATUS "  ILP64 Support: Yes (64-bit integers)")
  endif()
  message(STATUS "=====================================")

  if ( DEFINED BLIS_INCLUDE_DIR )
    set( BLIS_CPP ../common/blis_interface.cpp )
  endif()

  message(STATUS "Linking reference BLAS LIB: ${BLAS_LIBRARY}")

  if ( WARN_NOT_ILP64_PREFERRED )
    message( WARNING "Using ${BLAS_LIBRARY} as reference library, 64-bit tests may fail. Test suite should be run with --gtest_filter=-*stress*")
  endif()

  # Find the package ROCmSMI
  if(NOT WIN32)
    find_package(rocm_smi CONFIG REQUIRED)
    list( APPEND COMMON_LINK_LIBS rocm_smi64 )
  endif()

  find_package( GTest REQUIRED )

  add_subdirectory( common )

  if( BUILD_CLIENTS_BENCHMARKS )
    add_subdirectory( benchmarks )
  endif( )

  if( BUILD_CLIENTS_TESTS )
    add_subdirectory( gtest )
  endif( )

endif()

if( BUILD_CLIENTS_SAMPLES )
  add_subdirectory( samples )
endif( )

set( ROCBLAS_COMMON "${PROJECT_BINARY_DIR}/staging/rocblas_common.yaml")
add_custom_command( OUTPUT "${ROCBLAS_COMMON}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_common.yaml "${ROCBLAS_COMMON}"
                    DEPENDS include/rocblas_common.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Copying rocblas_common.yaml to staging" )

set( ROCBLAS_TEMPLATE "${PROJECT_BINARY_DIR}/staging/rocblas_template.yaml")
add_custom_command( OUTPUT "${ROCBLAS_TEMPLATE}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_template.yaml "${ROCBLAS_TEMPLATE}"
                    DEPENDS include/rocblas_template.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Copying rocblas_template.yaml to staging" )

set( ROCBLAS_GENERAL_YAML "${PROJECT_BINARY_DIR}/staging/rocblas_general.yaml")
add_custom_command( OUTPUT "${ROCBLAS_GENERAL_YAML}"
                    COMMAND ${CMAKE_COMMAND} -E copy include/rocblas_general.yaml "${ROCBLAS_GENERAL_YAML}"
                    DEPENDS include/rocblas_general.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Copying rocblas_general.yaml to staging" )

set( ROCBLAS_GENTEST "${PROJECT_BINARY_DIR}/staging/rocblas_gentest.py")
add_custom_command( OUTPUT "${ROCBLAS_GENTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy common/rocblas_gentest.py "${ROCBLAS_GENTEST}"
                    DEPENDS common/rocblas_gentest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Copying rocblas_gentest.py to staging" )

set( ROCBLAS_CLIENTS_README "${PROJECT_BINARY_DIR}/staging/rocblas_clients_readme.txt")
add_custom_command( OUTPUT "${ROCBLAS_CLIENTS_README}"
                    COMMAND ${CMAKE_COMMAND} -E copy rocblas_clients_readme.txt "${ROCBLAS_CLIENTS_README}"
                    DEPENDS rocblas_clients_readme.txt
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    COMMENT "Copying rocblas_clients_readme.txt to staging" )

add_custom_target( rocblas-common DEPENDS "${ROCBLAS_COMMON}" "${ROCBLAS_TEMPLATE}" "${ROCBLAS_GENERAL_YAML}" "${ROCBLAS_GENTEST}" "${ROCBLAS_CLIENTS_README}"
                   COMMENT "Building rocblas-common files" )

rocm_install(
  FILES ${ROCBLAS_COMMON} ${ROCBLAS_TEMPLATE} ${ROCBLAS_GENERAL_YAML}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clients-common
)

rocm_install(
  PROGRAMS ${ROCBLAS_GENTEST}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clients-common
)

# this readme also serves to prevent an empty package rocblas-clients which dpkg may auto-remove entire rocblas-clients and non empty children
rocm_install(
  FILES ${ROCBLAS_CLIENTS_README}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT clients
)

# Install AOCL library if it's a shared library built from source
if (BUILD_CLIENTS AND BLAS_LIBRARY MATCHES "\\.so$" AND BLAS_LIBRARY MATCHES "${BUILD_DIR}/deps/aocl")
  # Get the directory containing the AOCL library
  get_filename_component(aocl_lib_dir "${BLAS_LIBRARY}" DIRECTORY)

  # Find all shared libraries in the AOCL lib directory
  file(GLOB AOCL_SHARED_LIBS "${aocl_lib_dir}/*.so*")

  if(AOCL_SHARED_LIBS)
    message(STATUS "Installing AOCL shared libraries to package:")
    foreach(lib ${AOCL_SHARED_LIBS})
      message(STATUS "  ${lib}")
    endforeach()

    rocm_install(
      FILES ${AOCL_SHARED_LIBS}
      DESTINATION "${CMAKE_INSTALL_LIBDIR}"
      COMPONENT clients-common
    )
  endif()
endif()
