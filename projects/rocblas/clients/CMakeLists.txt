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

function( apply_omp_settings lib_target_ )
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::OpenMP_CXX)
    set_target_properties( ${lib_target_} PROPERTIES
      BUILD_RPATH "${HIP_CLANG_ROOT}/lib"
    )
    set_target_properties( ${lib_target_} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../llvm/lib"
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET OpenMP::omp)
    set_target_properties( ${lib_target_} PROPERTIES
      BUILD_RPATH "${HIP_CLANG_ROOT}/${openmp_LIB_DIR}"
    )
    set_target_properties( ${lib_target_} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../llvm/${openmp_LIB_DIR}"
    )
    target_compile_options( ${lib_target_} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}> )
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
        message(STATUS "Found BLIS at: ${library_path}")
        return()
      endif()
    endif()
  endforeach()
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

# Client tests/benchmarks with LINK_BLIS=OFF: default to ILP64 OpenBLAS via find_package(BLAS) + BLA_SIZEOF_INTEGER=8
# (TheRock / bundled OpenBLAS64). When OFF, Linux uses generic find_package(BLAS) (LP64); Windows uses legacy OPENBLAS_DIR + LP64.
if( (BUILD_CLIENTS_TESTS OR BUILD_CLIENTS_BENCHMARKS) AND NOT LINK_BLIS )
  set( _rocblas_clients_ref_ilp64_default ON )
else()
  set( _rocblas_clients_ref_ilp64_default OFF )
endif()
option( ROCBLAS_CLIENTS_REFERENCE_ILP64
  "Use ILP64 OpenBLAS for client tests/benchmarks (find_package(BLAS) with BLA_SIZEOF_INTEGER=8). When OFF, Linux uses LP64 BLAS; Windows may use legacy OPENBLAS_DIR / LP64."
  ${_rocblas_clients_ref_ilp64_default} )

if( BUILD_CLIENTS_BENCHMARKS OR BUILD_CLIENTS_TESTS)
  if ( NOT WIN32 )
    unset( ROCBLAS_REFERENCE_BLAS_DEFINES )
    if (LINK_BLIS)
      set(BLAS_LIBRARY)
      set(BLIS_INCLUDE_DIR)
      
      # Try AOCL 5.x unified library (static only)
      if(NOT BLAS_LIBRARY)
        # Check build dependencies
        if(EXISTS "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.a")
          set(BLAS_LIBRARY "${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.a")
          set(BLIS_INCLUDE_DIR "${BUILD_DIR}/deps/aocl/install_package/include")
          message(STATUS "Found AOCL 5.x unified library at: ${BLAS_LIBRARY}")
        endif()
      endif()
      
      # Check system AOCL installations in /opt/AMD/aocl (static only)
      if(NOT BLAS_LIBRARY)
        file(GLOB AOCL5_DIRS "/opt/AMD/aocl/[0-9]*")
        list(SORT AOCL5_DIRS)
        list(REVERSE AOCL5_DIRS)
        foreach(dir IN LISTS AOCL5_DIRS)
          foreach(libdir IN ITEMS lib lib64)
            if(EXISTS "${dir}/${libdir}/libaocl.a")
              set(BLAS_LIBRARY "${dir}/${libdir}/libaocl.a")
              if(EXISTS "${dir}/include_ILP64")
                set(BLIS_INCLUDE_DIR "${dir}/include_ILP64")
              elseif(EXISTS "${dir}/include")
                set(BLIS_INCLUDE_DIR "${dir}/include")
              endif()
              message(STATUS "Found AOCL 5.x system library at: ${BLAS_LIBRARY}")
              break()
            endif()
          endforeach()
          if(BLAS_LIBRARY)
            break()
          endif()
        endforeach()
      endif()
      
      # Try AOCL 4.x BLIS libraries
      if(NOT BLAS_LIBRARY)
        find_libblis(gcc)
      endif()
      if(NOT BLAS_LIBRARY)
        find_libblis(aocc)
        set(WARN_NOT_ILP64_PREFERRED true) # found via aocc
      endif()
      if(NOT BLAS_LIBRARY)
        set(WARN_NOT_ILP64_PREFERRED false) # reset warning, not found via installed aocc
        if(EXISTS      "${BUILD_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a") # 4.0 and 4.1.0
          set( BLAS_LIBRARY ${BUILD_DIR}/deps/amd-blis/lib/ILP64/libblis-mt.a )
          set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/amd-blis/include/ILP64 )
          set( WARN_NOT_ILP64_PREFERRED true )
        elseif(EXISTS      "${BUILD_DIR}/deps/blis/lib/libblis.a")
          set( BLAS_LIBRARY ${BUILD_DIR}/deps/blis/lib/libblis.a )
          set( BLIS_INCLUDE_DIR ${BUILD_DIR}/deps/blis/include/blis )
          set( WARN_NOT_ILP64_PREFERRED true )
        elseif(EXISTS      "/usr/local/lib/libblis.a")
          set( BLAS_LIBRARY /usr/local/lib/libblis.a )
          set( BLIS_INCLUDE_DIR /usr/local/include/blis )
        else()
          # Try to find BLAS library directly (most reliable across distros)
          find_library(SYSTEM_BLAS_LIB 
            NAMES openblas cblas blas
            PATHS /usr/lib /usr/lib64 /usr/local/lib /usr/lib/x86_64-linux-gnu
          )
          
          if(SYSTEM_BLAS_LIB)
            set(BLAS_LIBRARY ${SYSTEM_BLAS_LIB})
            message(STATUS "Found system BLAS library at: ${SYSTEM_BLAS_LIB}")
            
            # Try to get include dirs from pkg-config if available
            find_package(PkgConfig)
            if(PKG_CONFIG_FOUND)
              pkg_search_module(PKGBLAS cblas)
              if(PKGBLAS_FOUND AND PKGBLAS_INCLUDE_DIRS)
                set(BLAS_INCLUDE_DIR ${PKGBLAS_INCLUDE_DIRS})
                message(STATUS "Found BLAS include directory via pkg-config: ${PKGBLAS_INCLUDE_DIRS}")
              endif()
            endif()
          else()
            message(FATAL_ERROR "Could not find any BLAS library (openblas, cblas, or blas)")
          endif()
        endif()
      endif()
    elseif( ROCBLAS_CLIENTS_REFERENCE_ILP64 )
      # ILP64 OpenBLAS via CMake FindBLAS (TheRock finders: OpenBLAS64 when BLA_SIZEOF_INTEGER=8).
      set( WARN_NOT_ILP64_PREFERRED false )
      set( BLA_SIZEOF_INTEGER 8 )
      set( BLA_VENDOR OpenBLAS )
      find_package( BLAS REQUIRED )
      if( TARGET BLAS::BLAS )
        set( BLAS_LIBRARY BLAS::BLAS )
      else()
        set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
      endif()
      set( BLAS_INCLUDE_DIR "" )
      if( TARGET OpenBLAS64::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS64::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET OpenBLAS::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET BLAS::BLAS )
        get_target_property( BLAS_INCLUDE_DIR BLAS::BLAS INTERFACE_INCLUDE_DIRECTORIES )
      endif()
      if( BLAS_INCLUDE_DIR MATCHES "-NOTFOUND$" )
        set( BLAS_INCLUDE_DIR "" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED BLAS_INCLUDE_DIRS )
        set( BLAS_INCLUDE_DIR "${BLAS_INCLUDE_DIRS}" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS64_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS64_DIR}/include" )
      elseif( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS_DIR}/include" )
      endif()
      set( ROCBLAS_REFERENCE_BLAS_DEFINES "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT" )
    else()
      find_package( BLAS REQUIRED )
      set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
    endif()
  else() # WIN32
    unset( ROCBLAS_REFERENCE_BLAS_DEFINES )
    file(TO_CMAKE_PATH "C:/Program\ Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT.lib" AOCL_BLAS_LIBRARY)
    if (LINK_BLIS AND EXISTS ${AOCL_BLAS_LIBRARY})
      set( BLAS_LIBRARY "-l\"C:/Program\ Files/AMD/AOCL-Windows/amd-blis/lib/ILP64/AOCL-LibBlis-Win-MT\"" )
      set( BLIS_INCLUDE_DIR "C:/Program\ Files/AMD/AOCL-Windows/amd-blis/include/ILP64" )
      set( BLIS_DEFINES BLIS_ENABLE_NO_UNDERSCORE_API BLIS_ENABLE_CBLAS )
    elseif( NOT LINK_BLIS AND ROCBLAS_CLIENTS_REFERENCE_ILP64 )
      # ILP64 OpenBLAS via CMake FindBLAS (CMAKE_MODULE_PATH / CMAKE_PREFIX_PATH unchanged so TheRock finders apply).
      set( WARN_NOT_ILP64_PREFERRED false )
      set( BLA_SIZEOF_INTEGER 8 )
      set( BLA_VENDOR OpenBLAS )
      find_package( BLAS REQUIRED )
      if( TARGET BLAS::BLAS )
        set( BLAS_LIBRARY BLAS::BLAS )
      else()
        set( BLAS_LIBRARY "${BLAS_LIBRARIES}" )
      endif()
      set( BLAS_INCLUDE_DIR "" )
      if( TARGET OpenBLAS64::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS64::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET OpenBLAS::OpenBLAS )
        get_target_property( BLAS_INCLUDE_DIR OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES )
      elseif( TARGET BLAS::BLAS )
        get_target_property( BLAS_INCLUDE_DIR BLAS::BLAS INTERFACE_INCLUDE_DIRECTORIES )
      endif()
      if( BLAS_INCLUDE_DIR MATCHES "-NOTFOUND$" )
        set( BLAS_INCLUDE_DIR "" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED BLAS_INCLUDE_DIRS )
        set( BLAS_INCLUDE_DIR "${BLAS_INCLUDE_DIRS}" )
      endif()
      if( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS64_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS64_DIR}/include" )
      elseif( NOT BLAS_INCLUDE_DIR AND DEFINED OpenBLAS_DIR )
        set( BLAS_INCLUDE_DIR "${OpenBLAS_DIR}/include" )
      endif()
      set( ROCBLAS_REFERENCE_BLAS_DEFINES "LAPACK_ILP64" "USE64BITINT" "OPENBLAS_USE64BITINT" )
    else()
      # Legacy: LP64 OpenBLAS via OPENBLAS_DIR / vcpkg (may warn about 64-bit stress tests).
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

  if ( DEFINED BLIS_INCLUDE_DIR )
    set( BLIS_CPP ../common/blis_interface.cpp )
  endif()

  # One token for ROCBLAS_REFERENCE_LIB (runtime log stringification). BLAS_LIBRARY may be an
  # imported target or a CMake list when falling back to BLAS_LIBRARIES; the latter breaks
  # target_compile_definitions if passed through verbatim.
  if( TARGET "${BLAS_LIBRARY}" )
    set( ROCBLAS_REFERENCE_LIB_LABEL "${BLAS_LIBRARY}" )
  elseif( BLAS_LIBRARIES )
    list( GET BLAS_LIBRARIES 0 ROCBLAS_REFERENCE_LIB_LABEL )
  else()
    set( ROCBLAS_REFERENCE_LIB_LABEL "${BLAS_LIBRARY}" )
  endif()

  message(STATUS "Linking reference BLAS LIB: ${BLAS_LIBRARY}")

  if ( WARN_NOT_ILP64_PREFERRED )
    message( WARNING "Using ${BLAS_LIBRARY} as reference library, 64-bit tests may fail. Test suite should be run with --gtest_filter=-*stress*")
  endif()

  # Find the package AMD SMI
  if(NOT WIN32)
    find_package(amd_smi CONFIG REQUIRED)
    list( APPEND COMMON_LINK_LIBS amd_smi )
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

# On Windows custom builds, install vcpkg DLLs required by clients
if(CREATE_TEST_APP_LOCAL_DEPLOY AND WIN32)
  # Find vcpkg installation directory via VCPKG_ROOT environment variable
  set(VCPKG_INSTALLED_DIR "")  # Initialize to avoid undefined variable
  if(DEFINED ENV{VCPKG_ROOT} AND EXISTS "$ENV{VCPKG_ROOT}/installed/x64-windows/bin")
    set(VCPKG_INSTALLED_DIR "$ENV{VCPKG_ROOT}/installed/x64-windows/bin")
  endif()

  if(VCPKG_INSTALLED_DIR)
    # Install GTest DLLs (required by rocblas-test)
    if(EXISTS "${VCPKG_INSTALLED_DIR}/gtest.dll")
      install(FILES "${VCPKG_INSTALLED_DIR}/gtest.dll"
              DESTINATION "${CMAKE_INSTALL_BINDIR}"
              COMPONENT clients)
    endif()
    if(EXISTS "${VCPKG_INSTALLED_DIR}/gtest_main.dll")
      install(FILES "${VCPKG_INSTALLED_DIR}/gtest_main.dll"
              DESTINATION "${CMAKE_INSTALL_BINDIR}"
              COMPONENT clients)
    endif()

    # Install OpenBLAS DLL only when using OpenBLAS as reference BLAS (not AOCL)
    if(NOT LINK_BLIS AND EXISTS "${VCPKG_INSTALLED_DIR}/openblas.dll")
      install(FILES "${VCPKG_INSTALLED_DIR}/openblas.dll"
              DESTINATION "${CMAKE_INSTALL_BINDIR}"
              COMPONENT clients)
    endif()
  endif()
endif()