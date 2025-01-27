# #############################################################################
# Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# #############################################################################

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

# Link to an internal object library.  Setting LINK_LIBRARIES
# directly prevents the object libraries from being exported by
# rocm_export_targets in a static build, since they're not supposed
# to be visible by users of rocFFT.
function( rocfft_link_internal_lib target lib )
  set_property( TARGET ${target} APPEND PROPERTY
    LINK_LIBRARIES ${lib}
  )
endfunction()

add_executable( rocfft_rtc_helper rocfft_rtc_helper.cpp )

# each backend requires different libraries for host and device code
if( USE_CUDA )
  set( ROCFFT_HOST_LINK_LIBS -lcuda )
  set( ROCFFT_RTC_LINK_LIBS -lnvrtc -lnvrtc-builtins -lnvptxcompiler_static )
else()
  set( ROCFFT_HOST_LINK_LIBS hip::host )
  set( ROCFFT_DEVICE_LINK_LIBS hip::device )
  if( WIN32 )
    set( ROCFFT_RTC_LINK_LIBS "${HIP_PATH}/lib/hiprtc.lib" )
  else()
    set( ROCFFT_RTC_LINK_LIBS -L${ROCM_PATH}/lib -lhiprtc -ldl )
  endif()
endif()

if( ROCFFT_MPI_ENABLE )
  set( ROCFFT_HOST_LINK_LIBS "${ROCFFT_HOST_LINK_LIBS}" "MPI::MPI_CXX" )
  if ( ROCFFT_CRAY_MPI_ENABLE)
    set( ROCFFT_HOST_LINK_LIBS "${ROCFFT_HOST_LINK_LIBS}" "mpi_gtl_hsa" )
    get_filename_component(MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY)
    set( ROCFFT_HOST_LINK_DIRS ${MPI_LIBDIR}/../../../../gtl/lib)
  endif()
endif()

set( package_targets rocfft )
target_include_directories( rocfft_rtc_helper
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
)
# rocfft_rtc_helper will be installed in /opt/rocm-ver/lib/rocfft/VERSION_STRING
# To find rocm libraries need the relative RPATH
set(APPEND_ROCMLIB_RPATH "\$ORIGIN/../../../lib")
target_link_libraries( rocfft_rtc_helper PRIVATE rocfft-rtc-compile )
set_target_properties( rocfft_rtc_helper PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
                                         BUILD_WITH_INSTALL_RPATH TRUE
                                         INSTALL_RPATH "${APPEND_ROCMLIB_RPATH}" )
target_link_directories( rocfft_rtc_helper PRIVATE ${ROCFFT_HOST_LINK_DIRS} )

# Package that helps me set visibility for function names exported
# from shared library
include( GenerateExportHeader )

find_package (Python3 3.6 COMPONENTS Interpreter REQUIRED)

if( NOT DEFINED PYTHON3_EXE )
  set(PYTHON3_EXE python3)
endif()

add_subdirectory( device )

#
# embed the generator itself into c++ files
#

# script that produces an include file
set( kgen_embed_command ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed-cpp.py )

# location of the generated source file
set( kgen_embed_cpp ${CMAKE_BINARY_DIR}/library/src/device/kernel-generator-embed.cpp )

# files that need to be embedded into the library, to be able to generate code
set( kgen_embed_files
     ${CMAKE_SOURCE_DIR}/shared/rocfft_complex.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/common.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/memory_gfx.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/callback.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/butterfly_constant.h
     ${CMAKE_SOURCE_DIR}/library/src/device/kernels/real2complex_device.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/large_twiddles.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_2.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_3.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_4.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_5.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_6.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_7.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_8.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_9.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_10.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_11.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_13.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_16.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/rtc_radix_functions/radix_17.h

     # extra files for generating standalone test harnesses for kernels
     ${CMAKE_SOURCE_DIR}/shared/device_properties.h
     ${CMAKE_SOURCE_DIR}/shared/rocfft_hip.h
     ${CMAKE_SOURCE_DIR}/shared/gpubuf.h
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_kernel.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_kernel.cpp
     ${CMAKE_SOURCE_DIR}/library/src/rtc_test_harness_helper.cpp
)

# files that contribute to the logic of how code gets generated -
# embedded files obviously already contribute.  these are checksummed
# to serve as a "version" for the code generator.
set( kgen_logic_files

     # Complex number datatype
     ${CMAKE_SOURCE_DIR}/shared/rocfft_complex.h

     # python code that does the embedding
     ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator-embed-cpp.py

     # python code that decides kernel parameters
     ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py
     ${CMAKE_SOURCE_DIR}/library/src/device/generator.py

     # stockham generator code
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/generator.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/generator.cpp
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/fftgenerator.cpp
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/fftgenerator.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen.cpp
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_2d.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_base.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_cc.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_cr.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_rc.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/stockham_gen_rr.h
     ${CMAKE_SOURCE_DIR}/library/src/device/generator/bluestein_generator.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_compile.cpp
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_stockham_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_stockham_gen.cpp

     # transpose generator code
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_transpose_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_transpose_gen.cpp

     # realcomplex generator code
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_realcomplex_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_realcomplex_gen.cpp

     # bluestein generator code
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_bluestein_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_bluestein_gen.cpp

     # twiddle generator code
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_twiddle_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_twiddle_gen.cpp

     # chirp generator code
     ${CMAKE_SOURCE_DIR}/library/src/include/rtc_chirp_gen.h
     ${CMAKE_SOURCE_DIR}/library/src/rtc_chirp_gen.cpp
)

add_custom_command(
  OUTPUT ${kgen_embed_cpp}
  COMMAND ${PYTHON3_EXE} ${kgen_embed_command}
  --embed ${kgen_embed_files} --logic ${kgen_logic_files} --output ${kgen_embed_cpp}
  DEPENDS ${kgen_embed_command} ${kgen_embed_files} ${kgen_logic_files}
)

# location of the generated solutions map cpp
set( gen_solutions ${CMAKE_BINARY_DIR}/library/src/solutions.cpp )
set( solship_py ${CMAKE_SOURCE_DIR}/library/src/device/solution-shipping.py )

# default folder of solution maps that will be built in library,
# user can specify their own arch and folder
set( sol_gpu_arch ${GPU_TARGETS} )
if( NOT ROCFFT_SOLUTION_MAP_DIR )
  set( ROCFFT_SOLUTION_MAP_DIR ${CMAKE_SOURCE_DIR}/library/solution_map CACHE STRING "RocFFT solution map directory" )
endif()

file(GLOB solution_map_files "${ROCFFT_SOLUTION_MAP_DIR}/*.dat")

add_custom_command(
  OUTPUT ${gen_solutions}
  COMMAND ${PYTHON3_EXE} ${solship_py}
  --gpu-arch="${sol_gpu_arch}"
  --data-folder=${ROCFFT_SOLUTION_MAP_DIR}
  DEPENDS ${solution_map_files}
  COMMENT "Put solution map from external text file into library"
)

# The following is a list of implementation files defining the library
set( rocfft_source
  auxiliary.cpp
  plan.cpp
  transform.cpp
  repo.cpp
  powX.cpp
  chirp.cpp
  twiddles.cpp
  kargs.cpp
  tree_node.cpp
  tree_node_1D.cpp
  tree_node_2D.cpp
  tree_node_3D.cpp
  tree_node_bluestein.cpp
  tree_node_real.cpp
  fuse_shim.cpp
  assignment_policy.cpp
  node_factory.cpp
  enum_printer.cpp
  rtc_exports.cpp
  tuning_kernel_tuner.cpp
  tuning_plan_tuner.cpp
  )

set(static_depends)

include( ../../cmake/sqlite.cmake )

include( ../../cmake/std-filesystem.cmake )

# RTC stuff is used by both core library and helpers, so create
# separate libraries
#
# common things like embedded generator strings, schemes, logging
add_library( rocfft-rtc-common OBJECT
  ${kgen_embed_cpp}
  compute_scheme.cpp
  rocfft_ostream.cpp
)
# compilation of rtc kernels (in-process)
add_library( rocfft-rtc-compile OBJECT
  rtc_compile.cpp
)
# compilation of rtc kernels (sub-process)
add_library( rocfft-rtc-subprocess OBJECT
  rtc_subprocess.cpp
)
target_compile_definitions( rocfft-rtc-subprocess PRIVATE
  -DROCFFT_VERSION=${VERSION_STRING}
)
# generation of kernel source
add_library( rocfft-rtc-gen OBJECT
  rtc_bluestein_gen.cpp
  rtc_realcomplex_gen.cpp
  rtc_stockham_gen.cpp
  rtc_transpose_gen.cpp
  rtc_twiddle_gen.cpp
  rtc_chirp_gen.cpp
  rtc_test_harness.cpp
  load_store_ops_gen.cpp
)
# caching of generation/compilation
add_library( rocfft-rtc-cache OBJECT
  rtc_cache.cpp
)
target_link_libraries( rocfft-rtc-cache PUBLIC ${ROCFFT_SQLITE_LIB} )
target_link_std_experimental_filesystem( rocfft-rtc-cache )

# generating kernels from TreeNodes and launching them
add_library( rocfft-rtc-launch OBJECT
  rtc_kernel.cpp
  rtc_bluestein_kernel.cpp
  rtc_realcomplex_kernel.cpp
  rtc_stockham_kernel.cpp
  rtc_transpose_kernel.cpp
  rtc_twiddle_kernel.cpp
  rtc_chirp_kernel.cpp
  rtc_partial_pass_sbcc_64_64_64.cpp
  rtc_partial_pass_sbrr_64_64_64.cpp
  load_store_ops_kernel.cpp
  tree_node_callback.cpp
)
target_link_libraries( rocfft-rtc-launch PRIVATE rocfft-rtc-cache )

# compilation of solution map object and solutions
add_library( rocfft-solution-map OBJECT
  solution_map.cpp
  solutions.cpp
)

# compilation of tuning helper object
add_library( rocfft-tuning-helper OBJECT
  tuning_helper.cpp
)

foreach( target
  rocfft-rtc-common
  rocfft-rtc-compile
  rocfft-rtc-subprocess
  rocfft-rtc-gen
  rocfft-rtc-cache
  rocfft-rtc-launch
  rocfft-solution-map
  rocfft-tuning-helper
  )
  target_include_directories( ${target}
    PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  )
  set_target_properties( ${target} PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_options( ${target} PRIVATE ${WARNING_FLAGS} )
  if( ${CMAKE_BUILD_TYPE} STREQUAL Debug )
    target_compile_options( ${target} PRIVATE -Og )
  endif()
  target_link_libraries( ${target} PUBLIC ${ROCFFT_HOST_LINK_LIBS} ${ROCFFT_RTC_LINK_LIBS} )
  target_link_directories( ${target} PRIVATE ${ROCFFT_HOST_LINK_DIRS} )

endforeach()

add_executable( rocfft_aot_helper
  enum_printer.cpp
  rocfft_aot_helper.cpp
  rocfft_stub.cpp
)

add_executable( rocfft_kernel_config_search
  rocfft_kernel_config_search.cpp
  rocfft_stub.cpp
)

if( ROCFFT_BUILD_OFFLINE_TUNER )
  add_executable( rocfft_offline_tuner
    ../../shared/array_validator.cpp
    enum_printer.cpp
    rocfft_offline_tuner.cpp
    rocfft_stub.cpp
  )
  target_compile_options( rocfft_offline_tuner PRIVATE -DROCFFT_BUILD_OFFLINE_TUNER )

  add_executable( rocfft_solmap_convert
    enum_printer.cpp
    rocfft_solmap_convert.cpp
    rocfft_stub.cpp
  )
  target_compile_options( rocfft_solmap_convert PRIVATE -DROCFFT_BUILD_OFFLINE_TUNER )
endif()

prepend_path( ".." rocfft_headers_public relative_rocfft_headers_public )

add_library( rocfft
  ${rocfft_source}
  ${relative_rocfft_headers_public}
  )

if( ROCFFT_MPI_ENABLE )
  target_compile_definitions(rocfft PRIVATE ROCFFT_MPI_ENABLE)
  include_directories(SYSTEM ${MPI_INCLUDE_PATH})
endif()

add_library( roc::rocfft ALIAS rocfft )

if( ROCFFT_BUILD_OFFLINE_TUNER )
  target_compile_options( rocfft PRIVATE -DROCFFT_BUILD_OFFLINE_TUNER )
endif()

if( NOT BUILD_SHARED_LIBS )
  target_link_libraries( rocfft INTERFACE ${ROCFFT_HOST_LINK_LIBS} )
  target_compile_options( rocfft PRIVATE -DROCFFT_STATIC_LIB )
  target_link_directories( rocfft PRIVATE ${ROCFFT_HOST_LINK_DIRS} )
endif()

target_link_libraries( rocfft PRIVATE ${ROCFFT_DEVICE_LINK_LIBS} )

foreach( target rocfft rocfft_offline_tuner rocfft_solmap_convert rocfft_aot_helper rocfft_kernel_config_search )

  if(( NOT ROCFFT_BUILD_OFFLINE_TUNER ) AND ((${target} STREQUAL "rocfft_offline_tuner") OR (${target} STREQUAL "rocfft_solmap_convert")))
    continue()
  endif()

  # RTC uses dladdr to find the RTC helper program
  if( NOT WIN32 )
    target_link_libraries( ${target} PUBLIC -ldl pthread )
  endif()

  target_include_directories( ${target}
    PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/src/device>
    ${sqlite_local_SOURCE_DIR}
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/rocfft>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
  set_target_properties( ${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
  rocfft_link_internal_lib( ${target} rocfft-rtc-cache )
  rocfft_link_internal_lib( ${target} ${ROCFFT_SQLITE_LIB} )
  rocfft_link_internal_lib( ${target} rocfft-rtc-gen )
  rocfft_link_internal_lib( ${target} rocfft-rtc-compile )
  rocfft_link_internal_lib( ${target} rocfft-rtc-subprocess )
  rocfft_link_internal_lib( ${target} rocfft-rtc-common )

  target_link_std_experimental_filesystem(${target})

  
  if ( ROCFFT_CRAY_MPI_ENABLE )
      get_filename_component( MPI_LIBDIR ${MPI_LIBRARY} DIRECTORY )
      target_link_directories( ${target}
	PRIVATE
	${MPI_LIBDIR}/../../../../gtl/lib )
    endif()

  target_compile_options( ${target} PRIVATE ${WARNING_FLAGS} )
  if( ${CMAKE_BUILD_TYPE} STREQUAL Debug )
    target_compile_options(${target} PRIVATE -Og)
  endif()
endforeach()

rocfft_link_internal_lib( rocfft generator )
rocfft_link_internal_lib( rocfft rocfft-function-pool )
rocfft_link_internal_lib( rocfft rocfft-rtc-launch )
rocfft_link_internal_lib( rocfft rocfft-solution-map )
rocfft_link_internal_lib( rocfft rocfft-tuning-helper )
target_link_std_experimental_filesystem( rocfft )

target_link_libraries( rocfft_kernel_config_search PRIVATE
  ${ROCFFT_HOST_LINK_LIBS}
  generator
  rocfft-rtc-launch
  rocfft-function-pool
  )
target_link_directories( rocfft_kernel_config_search PRIVATE ${ROCFFT_HOST_LINK_DIRS} )
target_link_std_experimental_filesystem( rocfft_kernel_config_search )

target_link_libraries( rocfft_aot_helper PRIVATE
  generator
  rocfft-function-pool
  rocfft-solution-map
  )
target_link_std_experimental_filesystem( rocfft_aot_helper )

 # build executable rocfft-offline-tuner
if( ROCFFT_BUILD_OFFLINE_TUNER )
  target_link_libraries( rocfft_offline_tuner PRIVATE
    ${ROCFFT_DEVICE_LINK_LIBS}
    rocfft
    generator
    rocfft-function-pool
    rocfft-rtc-launch
    rocfft-solution-map
    rocfft-tuning-helper
    )
  target_link_libraries( rocfft_solmap_convert PRIVATE
    rocfft
    generator
    rocfft-function-pool
    rocfft-rtc-launch
    rocfft-solution-map
    rocfft-tuning-helper
    )
endif()

# compile kernels into the cache file we ship

# While useful in most situations, building the kernel cache takes a long time
# enable a configure-time option to skip kernel cache building
option( ROCFFT_KERNEL_CACHE_ENABLE "Enable building rocFFT kernel cache" ON)

# cache file should go next to the shared object - on Windows this
# would be the DLL, not the import library.
if( WIN32 )
  set( ROCFFT_KERNEL_CACHE_PATH ${CMAKE_BINARY_DIR}/staging/rocfft_kernel_cache.db )
else()
  set( ROCFFT_KERNEL_CACHE_PATH ${CMAKE_BINARY_DIR}/library/src/rocfft_kernel_cache.db )
endif()

# ROCFFT_BUILD_KERNEL_CACHE_PATH may be specified as a temporary file
# to collect compiled kernels before writing them out as part of the
# build.  any kernels that already exist in this file will be reused
# between builds.
#
# If ROCFFT_BUILD_KERNEL_CACHE_PATH is unspecified, rocfft_aot_helper
# uses a temporary file.

# Only build kernels ahead-of-time for a more limited set of
# architectures.  Less common architectures are filtered out from the
# list and kernels for them are built at runtime instead.
if ( ROCFFT_KERNEL_CACHE_ENABLE )

  set( GPU_TARGETS_AOT ${GPU_TARGETS} )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx803 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx900 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx906 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx940 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx941 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx1101 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx1102 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx1151 )
  list( REMOVE_ITEM GPU_TARGETS_AOT gfx1200 )
  # The binary will be having relative RUNPATH with respect to install directory
  # Set LD_LIBRARY_PATH for executing the binary from build directory.
  add_custom_command(
    OUTPUT rocfft_kernel_cache.db
    COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}" ./rocfft_aot_helper \"${ROCFFT_BUILD_KERNEL_CACHE_PATH}\" ${ROCFFT_KERNEL_CACHE_PATH} $<TARGET_FILE:rocfft_rtc_helper> ${GPU_TARGETS_AOT}
    DEPENDS rocfft_aot_helper rocfft_rtc_helper
    COMMENT "Compile kernels into shipped cache file"
  )
  add_custom_target( rocfft_kernel_cache_target ALL
    DEPENDS rocfft_kernel_cache.db
    VERBATIM
  )
endif()

rocm_set_soversion( rocfft ${rocfft_SOVERSION} )
set_target_properties( rocfft PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

set_target_properties( rocfft PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON )

generate_export_header( rocfft EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/rocfft/rocfft-export.h )

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_wrap_header_file(
    rocfft-version.h rocfft-export.h
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS ${CMAKE_INSTALL_INCLUDEDIR} rocfft/${CMAKE_INSTALL_INCLUDEDIR}
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/rocfft/rocfft-version.h
  )
endif( )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( NOT BUILD_SHARED_LIBS )
  set_target_properties( rocfft PROPERTIES PREFIX "lib" )
endif( )

############################################################
# Installation

rocm_install_targets(
  TARGETS ${package_targets}
  INCLUDE
  ${CMAKE_BINARY_DIR}/include
  )

# kernel cache is architecture-dependent data for the library, placed
# in a rocFFT subdirectory next to the library.  Linux puts shared
# objects in lib, Windows puts DLLs in bin
if(WIN32)
  set(ROCFFT_KERNEL_CACHE_INSTALL_DIR ${CMAKE_INSTALL_BINDIR}/rocfft)
else()
  set(ROCFFT_KERNEL_CACHE_INSTALL_DIR ${ROCM_INSTALL_LIBDIR}/rocfft)
endif()

if( NOT ENABLE_ASAN_PACKAGING AND ROCFFT_KERNEL_CACHE_ENABLE)
  rocm_install(FILES ${ROCFFT_KERNEL_CACHE_PATH}
    DESTINATION "${ROCFFT_KERNEL_CACHE_INSTALL_DIR}"
    COMPONENT runtime
  )
endif()

# rtc helper is an internal library executable on Linux, placed in a
# rocFFT subdirectory of the library directory.  On Windows it goes
# into bin next to the library, to simplify finding DLLs.
if(WIN32)
  set(ROCFFT_RTC_HELPER_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
else()
  set(ROCFFT_RTC_HELPER_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/rocfft/${VERSION_STRING} )
endif()

if( NOT ENABLE_ASAN_PACKAGING )
  rocm_install(PROGRAMS $<TARGET_FILE:rocfft_rtc_helper>
    DESTINATION "${ROCFFT_RTC_HELPER_INSTALL_DIR}"
    COMPONENT runtime
  )
endif()

rocm_export_targets(
  TARGETS roc::rocfft
  DEPENDS PACKAGE hip
  STATIC_DEPENDS
    ${static_depends}
  NAMESPACE roc::
  )

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/rocfft"
        DESTINATION "." )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()
