# #############################################################################
# Copyright (C) 2016 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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


# A helper function to prefix a source list of files with a common
# path into a new list (non-destructive)
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

option(ROCFFT_DEVICE_FORCE_RELEASE "Force the rocfft-device library to Release build type" OFF)
if(ROCFFT_DEVICE_FORCE_RELEASE)
  if(WIN32)
    message(
      FATAL_ERROR
      "ROCFFT_DEVICE_FORCE_RELEASE cannot be used on Windows, as the debug and normal runtimes "
      "are ABI-incompatible.  The core rocFFT lib and device libs must both use the same runtime."
    )
  endif()
  set (CMAKE_BUILD_TYPE Release)
endif()

# This builds the generator executable
add_subdirectory( generator )

if(USE_CUDA)
  add_compile_options(--cuda-path=${CUDA_PREFIX} --cuda-gpu-arch=${CUDA_ARCH} -xcuda)
endif()

# Generated kernels

# Make it possible to let install.sh control this ?
set( kgen ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py )
set( kgendeps ${CMAKE_SOURCE_DIR}/library/src/device/kernel-generator.py
              ${CMAKE_SOURCE_DIR}/library/src/device/kernels/configs/config_sbrr.py
              ${CMAKE_SOURCE_DIR}/library/src/device/kernels/configs/config_sbcc.py
              ${CMAKE_SOURCE_DIR}/library/src/device/kernels/configs/config_sbcr.py
              ${CMAKE_SOURCE_DIR}/library/src/device/kernels/configs/config_sbrc.py
              ${CMAKE_SOURCE_DIR}/library/src/device/kernels/configs/config_2d_single.py
              ${CMAKE_SOURCE_DIR}/library/src/device/kernels/configs/config_pp_3d.py
              ${CMAKE_SOURCE_DIR}/library/src/device/generator.py )

# create list of all N files that will initialize function pool
if( ROCFFT_FUNCTION_POOL_N LESS 1 )
  message( FATAL_ERROR "ROCFFT_FUNCTION_POOL_N must be positive (value passed is ${ROCFFT_FUNCTION_POOL_N})")
endif()

# RANGE includes end number, so subtract 1
math(EXPR LOOP_END "${ROCFFT_FUNCTION_POOL_N} - 1")
set(FUNCTION_POOLS)
foreach(i RANGE ${LOOP_END})
  list(APPEND FUNCTION_POOLS "function_pool_init_${i}.cpp")
endforeach()

add_custom_command(OUTPUT function_pool.cpp
  OUTPUT ${FUNCTION_POOLS}
  COMMAND ${Python3_EXECUTABLE} ${kgen}
  --runtime-compile-default=${ROCFFT_RUNTIME_COMPILE_DEFAULT}
  --num-files=${ROCFFT_FUNCTION_POOL_N}
  generate $<TARGET_FILE:stockham_gen>
  DEPENDS stockham_gen ${kgendeps}
  COMMENT "Generator producing device kernels for rocfft-device"
)

# add virtual build target for generated kernels
add_custom_target(gen_headers_target
  DEPENDS function_pool.cpp
  VERBATIM
)

prepend_path( "../.."
  rocfft_headers_public relative_rocfft_device_headers_public )

option(ROCFFT_CALLBACKS_ENABLED "Enable user-defined callbacks for load/stores from global memory" ON)

# function pool is a generated file, but put it in its own library so it's easier to link to.
add_library( rocfft-function-pool OBJECT
  ${FUNCTION_POOLS}
  function_pool.cpp
)
# Don't add function pool to coverage as it's just generated code
# that sets up kernel config data structures, and isn't interesting
# to cover.
#rocfft_add_coverage_flags( rocfft-function-pool )

target_include_directories( rocfft-function-pool
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src/device>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  )
set_target_properties( rocfft-function-pool PROPERTIES
  CXX_VISIBILITY_PRESET "hidden"
  VISIBILITY_INLINES_HIDDEN ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  )
add_dependencies(rocfft-function-pool gen_headers_target)
