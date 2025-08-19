# Modifications Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

macro(pythonize_bool var)
  if (${var})
    set(${var} True)
  else()
    set(${var} False)
  endif()
endmacro()

if (LIBCUDACXX_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  get_target_property(CUDA_INCLUDE_DIR CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)

  set(LIBCUDACXX_HIGHEST_COMPUTE_ARCH 90)

  foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
    set(_compute_message "${_compute_message} sm_${COMPUTE_ARCH}")
    set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}")
  endforeach ()

  message(STATUS "Enabled CUDA architectures:${_compute_message}")

  option(LIBCUDACXX_TEST_WITH_NVRTC
    "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
    OFF)
elseif(LIBCUDACXX_ENABLE_HIP)

  # If LIBCUDACXX_COMPUTE_ARCHS is not set, use the HIP architectures
  if (NOT LIBCUDACXX_COMPUTE_ARCHS)
    set(LIBCUDACXX_COMPUTE_ARCHS ${CMAKE_HIP_ARCHITECTURES})
  endif()

  # Separate the architectures with a space
  foreach (COMPUTE_ARCH ${LIBCUDACXX_COMPUTE_ARCHS})
    set(_compute_message "${_compute_message} ${COMPUTE_ARCH}")
    set(LIBCUDACXX_COMPUTE_ARCHS_STRING "${LIBCUDACXX_COMPUTE_ARCHS_STRING} ${COMPUTE_ARCH}")
  endforeach ()

  message(STATUS "Enabled HIP architectures:${_compute_message}")

  option(LIBHIPCXX_TEST_WITH_HIPRTC
  "Test libcu++ with runtime compilation instead of offline compilation. Only runs device side tests."
  OFF)
endif()

if (LIBCUDACXX_TEST_WITH_NVRTC)
  # TODO: Use project properties to get path to binary.
  # Should also set up dependency on the project when NVRTC is enabled
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_BINARY_DIR}/libcudacxx/test/utils/nvidia/nvrtc/nvrtcc")
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "True")
  set(LIBCUDACXX_FORCE_INCLUDE "-include '${libhipcxx_SOURCE_DIR}/test/support/nvrtc_limit_macros.h'")
  set(LIBCUDACXX_TEST_COMPILER_FLAGS "-I'${CUDA_INCLUDE_DIR}'")
  # Use the NVRTCC utility to run the built test outputs
  set(LIBCUDACXX_EXECUTOR "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())")
elseif (LIBHIPCXX_TEST_WITH_HIPRTC)
  # TODO: Use project properties to get path to binary.
  # Should also set up dependency on the project when NVRTC is enabled
  set(CMAKE_CUDA_COMPILER_ID "HIPCC")
  set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_BINARY_DIR}/libcudacxx/test/utils/amd/hiprtc/hiprtcc")
  set(LIBCUDACXX_CUDA_COMPILER_ARG1 "")
  set(LIBCUDACXX_HIP_TEST_WITH_HIPRTC "True")
  set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
  set(LIBCUDACXX_FORCE_INCLUDE "-include '${libhipcxx_SOURCE_DIR}/test/support/nvrtc_limit_macros.h'")
  set(LIBCUDACXX_TEST_COMPILER_FLAGS "-I'${HIP_INCLUDE_DIR}'")
  # Use the NVRTCC utility to run the built test outputs
  set(LIBCUDACXX_EXECUTOR "PrefixExecutor(['${LIBCUDACXX_CUDA_COMPILER}'], LocalExecutor())")
else() # NOT LIBCUDACXX_TEST_WITH_NVRTC
  if (LIBCUDACXX_ENABLE_CUDA)
    set(LIBCUDACXX_FORCE_INCLUDE "-include ${libhipcxx_SOURCE_DIR}/test/force_include_cuda.h")
    set(LIBCUDACXX_CUDA_COMPILER "${CMAKE_CUDA_COMPILER}")
    set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
    set(LIBCUDACXX_TEST_COMPILER_FLAGS "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
  elseif(LIBCUDACXX_ENABLE_HIP)
    set(LIBCUDACXX_FORCE_INCLUDE "-include ${libhipcxx_SOURCE_DIR}/test/force_include_hip.h")
    #temporary workaround until we've found a way to make cmake find hipcc
    set(LIBCUDACXX_CUDA_COMPILER "${HIP_HIPCC_EXECUTABLE}")
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    set(CMAKE_CUDA_COMPILER_ID "HIPCC")
    set(LIBCUDACXX_CUDA_TEST_WITH_NVRTC "False")
    set(LIBCUDACXX_HIP_TEST_WITH_HIPRTC "False")
    set(LIBCUDACXX_TEST_COMPILER_FLAGS "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE")
  endif()
endif()

if (NOT MSVC AND NOT ${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  set(LIBCUDACXX_WARNING_LEVEL "--compiler-options=-Wall --compiler-options=-Wextra")
endif()

# sccache cannot handle the -Fd option generationg pdb files
if (MSVC)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
endif()

# Intel OneAPI compiler has fast math enabled by default which breaks almost all floating point tests
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "IntelLLVM" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS " --compiler-options=-fno-fast-math")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " -Xclang -fcuda-allow-variadic-functions"
    " -Xclang -Wno-unused-parameter"
    " -Wno-unknown-cuda-version"
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " -I${libhipcxx_SOURCE_DIR}/include"
    " ${LIBCUDACXX_WARNING_LEVEL}")

  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " -L${CUDAToolkit_LIBRARY_DIR}"
    " -lcuda"
    " -lcudart")
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " ${LIBCUDACXX_FORCE_INCLUDE}"
    " ${LIBCUDACXX_WARNING_LEVEL}")
elseif (${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVHPC")
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
    " -stdpar")
  string(APPEND LIBCUDACXX_TEST_LINKER_FLAGS
    " -stdpar")
endif()

if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "HIPCC")
  message(STATUS "libhipcxx_SOURCE_DIR: ${libhipcxx_SOURCE_DIR}" )
  string(APPEND LIBCUDACXX_TEST_COMPILER_FLAGS
#    "${LIBCUDACXX_TEST_COMPILER_FLAGS} \
    " ${LIBCUDACXX_FORCE_INCLUDE}" 
    " -I${CMAKE_SOURCE_DIR}/include"
    " -I${libhipcxx_SOURCE_DIR}/include"
#    " CACHE INTERNAL "Flags for libcxx testing."
  )
endif()

include(AddLLVM)

set(LIBCUDACXX_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(LIBCUDACXX_TARGET_INFO "libcudacxx.test.target_info.LocalTI" CACHE STRING
"TargetInfo to use when setting up test environment.")
set(LIBCUDACXX_EXECUTOR "None" CACHE STRING
"Executor to use when running tests.")

set(LIBCUDACXX_TEST_TIMEOUT "200" CACHE STRING "Enable test timeouts (Default = 200, Off = 0)")

set(AUTO_GEN_COMMENT "## Autogenerated by libcudacxx configuration.\n# Do not edit!")

set(lit_site_cfg_path "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg")
configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in"
  "${lit_site_cfg_path}")

add_lit_testsuite(check-hipcxx
  "Running libhip++ tests"
  "${CMAKE_CURRENT_BINARY_DIR}")

find_program(libcudacxx_LIT lit REQUIRED)

set(libcudacxx_LIT_FLAGS "" CACHE STRING "Semi-colon separated list of flags passed to the invocation of lit.")
message(STATUS "libcudacxx_LIT_FLAGS: ${libcudacxx_LIT_FLAGS}")

# Restricted to avoid oversubscribing the GPU:
set(libcudacxx_LIT_PARALLEL_LEVEL 8 CACHE STRING
"Parallelism used to run libcudacxx's lit test suite."
)

if (LIBCUDACXX_ENABLE_CUDA)
  if (NOT LIBCUDACXX_TEST_WITH_NVRTC)
    # Build but don't run the tests. Used by CI to pre-seed sccache for the test machines.
    # Only executed if explicitly requested.
    add_custom_target(libcudacxx.test.lit.precompile
      DEPENDS libcudacxx.test.public_headers libcudacxx.test.internal_headers libcudacxx.test.public_headers_host_only
      COMMAND "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}" -vv --no-progress-bar ${libcudacxx_LIT_FLAGS} -j "${libcudacxx_LIT_PARALLEL_LEVEL}" "-Dexecutor=\"NoopExecutor()\"" "${libhipcxx_SOURCE_DIR}/test/libcudacxx"
    )

  endif()
elseif(LIBCUDACXX_ENABLE_HIP)
  if (NOT LIBHIPCXX_TEST_WITH_HIPRTC)
    # Build but don't run the tests. Used by CI to pre-seed sccache for the test machines.
    # Only executed if explicitly requested.
    add_custom_target(libcudacxx.test.lit.precompile
      DEPENDS libcudacxx.test.public_headers libcudacxx.test.internal_headers libcudacxx.test.public_headers_host_only
      COMMAND "${CMAKE_COMMAND}" -E env "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
      "${libcudacxx_LIT}" -vv --no-progress-bar ${libcudacxx_LIT_FLAGS} -j "${libcudacxx_LIT_PARALLEL_LEVEL}" "-Dexecutor=\"NoopExecutor()\"" "${libhipcxx_SOURCE_DIR}/test"
    )

  endif()
endif()

add_test(NAME libcudacxx.test.lit COMMAND
  "${CMAKE_COMMAND}" -E env
    "LIBCUDACXX_SITE_CONFIG=${lit_site_cfg_path}"
  "${libcudacxx_LIT}" -vv --no-progress-bar ${libcudacxx_LIT_FLAGS}
    -j "${libcudacxx_LIT_PARALLEL_LEVEL}"
      "${libhipcxx_SOURCE_DIR}/test"
)

set_tests_properties(libcudacxx.test.lit PROPERTIES
  TIMEOUT 4800
  RUN_SERIAL TRUE
)

# Set appropriate warning levels for MSVC/sane
if("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
  # CUDA 11.5 and down do not support '-use-local-env'
  if(MSVC)
    set(headertest_warning_levels_device -Xcompiler=/W4 -Xcompiler=/WX)
    if ("${CMAKE_CUDA_COMPILER_VERSION}" GREATER_EQUAL "11.6.0")
      list(APPEND headertest_warning_levels_device --use-local-env)
    endif()
  else()
    set(headertest_warning_levels_device -Wall -Werror all-warnings)
  endif()
# Set warnings for Clang as device compiler
elseif("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
  set(headertest_warning_levels_device -Wall -Werror -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions)
# If the CMAKE_CUDA_COMPILER is unknown, try to use gcc style warnings
else()
  set(headertest_warning_levels_device -Wall -Werror)
endif()

# Set raw host/device warnings
if(MSVC)
  set(headertest_warning_levels_host /W4 /WX)
else()
  set(headertest_warning_levels_host -Wall -Werror)
endif()

# Libcudacxx auto-generated internal header tests
add_subdirectory(internal_headers)

# Libcudacxx auto-generated public header tests
add_subdirectory(public_headers)

# Libcudacxx auto-generated public header as std tests
add_subdirectory(public_headers_host_only)

if(LIBCUDACXX_ENABLE_CUDA)
  # Enable building the nvrtcc project if NVRTC is enabled
  if (LIBCUDACXX_TEST_WITH_NVRTC)
    add_subdirectory(utils/nvidia/nvrtc)
  endif()
elseif(LIBCUDACXX_ENABLE_HIP)
  if (LIBHIPCXX_TEST_WITH_HIPRTC)
    add_subdirectory(
      ${libhipcxx_SOURCE_DIR}/utils/amd/hiprtc
      ${CMAKE_BINARY_DIR}/libcudacxx/test/utils/amd/hiprtc
      )
  endif()
endif()

find_program(filecheck "FileCheck")

if (filecheck)
  message("-- ${filecheck} found... building atomic codegen tests")
  add_subdirectory(atomic_codegen)
endif()
