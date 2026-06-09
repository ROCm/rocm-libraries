#[[
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

project(modules)

function(remove_audio_augmentations SOURCE_LIST BACKEND)
    if( "${BACKEND}" STREQUAL "HIP")
        list(REMOVE_ITEM ${SOURCE_LIST}
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/non_silent_region_detection.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/to_decibels.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/pre_emphasis_filter.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/down_mixing.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/spectrogram.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/resample.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tensor/hip/kernel/mel_filter_bank.cpp"
        )
    endif()
    list(REMOVE_ITEM ${SOURCE_LIST}
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/rppt_tensor_audio_augmentations.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/non_silent_region_detection.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/to_decibels.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/pre_emphasis_filter.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/down_mixing.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/spectrogram.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/resample.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensor/cpu/kernel/mel_filter_bank.cpp"
    )
    set(${SOURCE_LIST} ${${SOURCE_LIST}} PARENT_SCOPE)
endfunction()

function(add_fma_property)
    set_source_files_properties(tensor/rppt_tensor_audio_augmentations.cpp
                                tensor/rppt_tensor_statistical_operations.cpp
                                tensor/cpu/kernel/non_silent_region_detection.cpp
                                tensor/cpu/kernel/to_decibels.cpp
                                tensor/cpu/kernel/pre_emphasis_filter.cpp
                                tensor/cpu/kernel/down_mixing.cpp
                                tensor/cpu/kernel/spectrogram.cpp
                                tensor/cpu/kernel/slice.cpp
                                tensor/cpu/kernel/resample.cpp
                                tensor/cpu/kernel/mel_filter_bank.cpp
                                tensor/cpu/kernel/tensor_sum.cpp
                                tensor/cpu/kernel/tensor_min.cpp
                                tensor/cpu/kernel/tensor_max.cpp
                                tensor/cpu/kernel/tensor_mean.cpp
                                tensor/cpu/kernel/tensor_stddev.cpp
                                tensor/cpu/kernel/normalize.cpp
                                tensor/cpu/kernel/threshold.cpp
                                PROPERTIES COMPILE_FLAGS -mno-fma)
endfunction()

set(CPPFILES "handle_api.cpp" "handle_host.cpp")
# Addition of Tensor Outer API CPP files
file(GLOB TENSOR_CPPFILES "tensor/*.cpp" )
list(APPEND Rpp_Source ${CPPFILES})
list(APPEND Rpp_Source ${TENSOR_CPPFILES})

if("${TIME_INFO}" STREQUAL "1")
    add_definitions(-DTIME_INFO)
endif()

# Backend specific settings

if( "${BACKEND}" STREQUAL "HIP")
    # Add HIP kernels
    set(MOD_HIP_CPP "handle_hip.cpp")
    file(GLOB MOD_HIP_CPP_TENSOR_KERNELS "tensor/hip/kernel/*.cpp" )
    file(GLOB MOD_HOST_CPP_TENSOR_KERNELS "tensor/cpu/kernel/*.cpp" )
    list(APPEND Rpp_Source ${CPPFILES} ${MOD_HIP_CPP} ${MOD_HIP_ERRORS_CPP} ${MOD_HIP_CPP_TENSOR_KERNELS} ${MOD_HOST_CPP_TENSOR_KERNELS})
    # Remove audio augmentations from the list of files to be compiled
    if(NOT RPP_AUDIO_SUPPORT)
        remove_audio_augmentations(Rpp_Source "${BACKEND}")
    endif()
    message("-- ${Green}HIP kernels added${ColourReset}")

    # Set compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HIPCC_FLAGS}")
    add_fma_property()

    # Add HIP specific includes
    set(ROCM_INC ${ROCM_PATH}/include/)
    set(INCLUDE_LIST ${ROCM_INC} ${HIP_LOCAL_INCLUDE_DIRS} ${INCLUDE_LIST})
elseif( "${BACKEND}" STREQUAL "CPU")
    # Add CPU specific includes
    file(GLOB MOD_HOST_CPP_TENSOR_KERNELS "tensor/cpu/kernel/*.cpp" )
    list(APPEND Rpp_Source ${CPPFILES} ${MOD_HOST_CPP_TENSOR_KERNELS})
    # Remove audio augmentations from the list of files to be compiled
    if(NOT RPP_AUDIO_SUPPORT)
        remove_audio_augmentations(Rpp_Source "${BACKEND}")
    endif()
    message("-- ${Green}HOST kernels added${ColourReset}")
    set(INCLUDE_LIST ${RPP_SOURCE_DIR}/src/include/common/)
    add_fma_property()
endif()
message("-- ${White}AMD RPP ${PROJECT_NAME} -- Include Directories:${INCLUDE_LIST}${ColourReset}")
add_compile_options("-Wno-unused-result")

# Add library
add_library(${PROJECT_NAME} OBJECT ${Rpp_Source})
target_link_libraries(${PROJECT_NAME} PUBLIC HALF::half)

# Target settings
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories( ${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/api
        ${ROCM_INC}
    PRIVATE
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/api
        ${RPP_SOURCE_DIR}/src/include/common
        ${RPP_SOURCE_DIR}/src/include/common/cpu
        ${RPP_SOURCE_DIR}/src/include/tensor
        ${INCLUDE_LIST}
)

if( "${BACKEND}" STREQUAL "HIP")
    target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${RPP_SOURCE_DIR}/src/include/common/hip
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE hip::device)
endif()
