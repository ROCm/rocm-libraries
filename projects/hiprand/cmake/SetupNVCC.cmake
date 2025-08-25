# MIT License
#
# Copyright (c) 2018-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Find HIP package and verify that correct C++ compiler was selected for available
# platform. On ROCm software host and device code is compiled by the same compiler.
# On CUDA host can be compiled by any C++ compiler while device code is compiled
# by nvcc compiler (CMake's CUDA package handles this).

# A function for automatic detection of the lowest CC of the installed NV GPUs
function(hip_cuda_detect_lowest_cc out_variable)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_nvgpus_cc.cu)

    file(WRITE ${__cufile} ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  int major = 1000;\n"
        "  int minor = 1000;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
        "      if (prop.major < major || (prop.major == major && prop.minor < minor)){\n"
        "        major = prop.major; minor = prop.minor;\n"
        "      }\n"
        "  }\n"
        "  std::printf(\"%d%d\", major, minor);\n"
        "  return 0;\n"
        "}\n")

    execute_process(
        COMMAND ${HIP_HIPCC_EXECUTABLE} "-Wno-deprecated-gpu-targets" "--run" "${__cufile}"
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
        RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
    )

    if(__nvcc_res EQUAL 0)
        set(HIP_CUDA_lowest_cc ${__nvcc_out} CACHE INTERNAL "The lowest CC of installed NV GPUs" FORCE)
    else()
        # fallback method which uses nvidia-smi to get min CC; requires CUDA Toolkit 11.6 or newer
        execute_process(
            COMMAND nvidia-smi "--query-gpu=compute_cap" "--format=csv,noheader"
            RESULT_VARIABLE __nvsmi_res OUTPUT_VARIABLE __nvsmi_out
        )

        if(__nvsmi_res EQUAL 0)
            message(STATUS "nvidia-smi reported CCs: ${__nvsmi_out}")
            string(REPLACE "." "" __nvsmi_out_nodot ${__nvsmi_out})
            string(REPLACE "\n" " " __nvsmi_out_formatted ${__nvsmi_out_nodot})
            separate_arguments(__cc_list NATIVE_COMMAND ${__nvsmi_out_formatted})
            list(SORT __cc_list ORDER ASCENDING)
            list(GET __cc_list 0 __min_cc)
            set(HIP_CUDA_lowest_cc ${__min_cc} CACHE INTERNAL "The lowest CC of installed NV GPUs" FORCE)
        else()
            message(STATUS "nvidia-smi command returned error code ${__nvsmi_res}.  Defaulting to cc_35")
        endif()
    endif()

    if(NOT HIP_CUDA_lowest_cc)
        set(HIP_CUDA_lowest_cc "35")
        set(${out_variable} ${HIP_CUDA_lowest_cc} PARENT_SCOPE)
    else()
        set(${out_variable} ${HIP_CUDA_lowest_cc} PARENT_SCOPE)
    endif()
endfunction()

################################################################################################
###  Non macro/function section
################################################################################################

# Set the default value for CMAKE_CUDA_COMPILER if it's empty
if(CMAKE_CUDA_COMPILER STREQUAL "")
    set(CMAKE_CUDA_COMPILER "nvcc")
endif()

# Get CUDA
enable_language("CUDA")

if( CMAKE_VERSION VERSION_LESS 3.17 )
    find_package(CUDA REQUIRED)
else()
    find_package(CUDAToolkit)
    set(CUDA_curand_LIBRARY CUDA::curand)
endif()


# Suppressing warnings
set(HIP_NVCC_FLAGS " ${HIP_NVCC_FLAGS} -Wno-deprecated-gpu-targets -Xcompiler -Wno-return-type -Wno-deprecated-declarations ")

# Use NVGPU_TARGETS to set CUDA architectures (compute capabilities)
# For example: -DNVGPU_TARGETS="50;61;62"
set(DEFAULT_NVGPU_TARGETS "")
# If NVGPU_TARGETS is empty get default value for it
if("x${NVGPU_TARGETS}" STREQUAL "x")
    hip_cuda_detect_lowest_cc(lowest_cc)
    set(DEFAULT_NVGPU_TARGETS "${lowest_cc}")
endif()
set(NVGPU_TARGETS "${DEFAULT_NVGPU_TARGETS}"
    CACHE STRING "List of NVIDIA GPU targets (compute capabilities), for example \"35;50\""
)
# Generate compiler flags based on targeted CUDA architectures
foreach(CUDA_ARCH ${NVGPU_TARGETS})
    list(APPEND HIP_NVCC_FLAGS "--generate-code arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} ")
    list(APPEND HIP_NVCC_FLAGS "--generate-code arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH} ")
endforeach()

execute_process(
    COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
    OUTPUT_VARIABLE HIP_CPP_CONFIG_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

# Update list parameter
string(REPLACE ";" " " HIP_NVCC_FLAGS ${HIP_NVCC_FLAGS})

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${HIP_CPP_CONFIG_FLAGS} ${HIP_NVCC_FLAGS}"
    CACHE STRING "Cuda compile flags" FORCE)

# Ignore warnings about #pragma unroll
# and about deprecated CUDA function(s) used in hip/nvidia_detail/hip_runtime_api.h
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CPP_CONFIG_FLAGS_STRIP} -Wno-unknown-pragmas -Wno-deprecated-declarations"
    CACHE STRING "compile flags" FORCE)
