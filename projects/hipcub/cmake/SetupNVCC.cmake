# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# platform. On ROCm platform host and device code is compiled by the same compiler:
# hipcc or clang. On CUDA host can be compiled by any C++ compiler while device 
# code is compiled by nvcc compiler (CMake's CUDA package handles this).

# A function for automatic detection of the CC of the installed NV GPUs
function(hip_cuda_detect_cc out_variable)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_nvgpus_cc.cu)

    file(WRITE ${__cufile} ""
        "#include <iostream>\n"
        "#include <set>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  std::set<int> list_cc;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
        "      list_cc.insert(prop.major*10+prop.minor);\n"
        "  }\n"
        "  for (std::set<int>::iterator itr = list_cc.begin(); itr != list_cc.end(); itr++)\n"
        "  {\n"
        "    if(itr != list_cc.begin()) std::cout << ';';\n"
        "    std::cout << *itr;\n"
        "  }\n"
        "  return 0;\n"
        "}\n")

    execute_process(
        COMMAND ${HIP_HIPCC_EXECUTABLE} "-Wno-deprecated-gpu-targets" "--run" "${__cufile}"
        WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
        RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
    )

    if(__nvcc_res EQUAL 0)
        set(HIP_CUDA_detected_cc ${__nvcc_out} CACHE INTERNAL "The detected CC of installed NV GPUs" FORCE)
    endif()

    if(NOT HIP_CUDA_detected_cc)
        set(HIP_CUDA_detected_cc "53")
        set(${out_variable} ${HIP_CUDA_detected_cc} PARENT_SCOPE)
    else()
        set(${out_variable} ${HIP_CUDA_detected_cc} PARENT_SCOPE)
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
set(CMAKE_CUDA_STANDARD 17)

# Suppressing warnings
set(HIP_NVCC_FLAGS " ${HIP_NVCC_FLAGS} -Wno-deprecated-gpu-targets -Xcompiler -Wno-return-type -Wno-deprecated-declarations ")

# Use NVGPU_TARGETS to set CUDA architectures (compute capabilities)
# For example: -DNVGPU_TARGETS="50;61;62"
set(DEFAULT_NVGPU_TARGETS "")
# If NVGPU_TARGETS is empty get default value for it
if("x${NVGPU_TARGETS}" STREQUAL "x")
    hip_cuda_detect_cc(detected_cc)
    set(DEFAULT_NVGPU_TARGETS "${detected_cc}")
endif()
set(NVGPU_TARGETS "${DEFAULT_NVGPU_TARGETS}"
    CACHE STRING "List of NVIDIA GPU targets (compute capabilities), for example \"35;50\""
)
set(CMAKE_CUDA_ARCHITECTURES ${NVGPU_TARGETS})

if (NOT _HIPCUB_HIP_NVCC_FLAGS_SET)
    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
        OUTPUT_VARIABLE HIP_CPP_CONFIG_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    # Generate compiler flags based on targeted CUDA architectures if CMake doesn't. (Controlled by policy CP0104, on by default after 3.18)
    if(CMAKE_VERSION VERSION_LESS "3.18")
        foreach(CUDA_ARCH ${NVGPU_TARGETS})
            list(APPEND HIP_NVCC_FLAGS "--generate-code" "arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
            list(APPEND HIP_NVCC_FLAGS "--generate-code" "arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH}")
        endforeach()
    endif()

    # Update list parameter
    list(JOIN HIP_NVCC_FLAGS " " HIP_NVCC_FLAGS)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${HIP_CPP_CONFIG_FLAGS} ${HIP_NVCC_FLAGS}"
        CACHE STRING "Cuda compile flags" FORCE)
    set(_HIPCUB_HIP_NVCC_FLAGS_SET ON CACHE INTERNAL "")
endif()

# Ignore warnings about #pragma unroll
# and about deprecated CUDA function(s) used in hip/nvcc_detail/hip_runtime_api.h
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CPP_CONFIG_FLAGS_STRIP} -Wno-unknown-pragmas -Wno-deprecated-declarations" CACHE STRING "compile flags" FORCE)
