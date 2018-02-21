################################################################################################
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
    endif()

    if(NOT HIP_CUDA_lowest_cc)
        set(HIP_CUDA_lowest_cc "20")
        set(${out_variable} ${HIP_CUDA_lowest_cc} PARENT_SCOPE)
    else()
        set(${out_variable} ${HIP_CUDA_lowest_cc} PARENT_SCOPE)
    endif()
endfunction()

################################################################################################
###  Non macro/function section
################################################################################################

find_package(CUDA REQUIRED)

# Finds lowest supported CUDA CC
#
# Use NVGPU_TARGETS to set CUDA arch compilation flags
# For example: -DNVGPU_TARGETS="--gpu-architecture=compute_50 --gpu-code=compute_50,sm_50,sm_52"
set(HIP_NVCC_FLAGS " ${HIP_NVCC_FLAGS} -Wno-deprecated-gpu-targets") # Suppressing warnings
if("x${NVGPU_TARGETS}" STREQUAL "x")
    hip_cuda_detect_lowest_cc(lowest_cc)
    set(HIP_NVCC_FLAGS "${HIP_NVCC_FLAGS} --gpu-architecture=sm_${lowest_cc}")
else()
    set(HIP_NVCC_FLAGS "${HIP_NVCC_FLAGS} ${NVGPU_TARGETS}")
endif()

# Add HIP flags/options/includes to CUDA_NVCC_FLAGS
execute_process(
    COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
    OUTPUT_VARIABLE HIP_CPP_CONFIG_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
string(REPLACE " " ";" HIP_CPP_CONFIG_FLAGS ${HIP_CPP_CONFIG_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "-std=c++11 ${HIP_CPP_CONFIG_FLAGS} ${HIP_NVCC_FLAGS}")

# Ignore warnings about #pragma unroll
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")