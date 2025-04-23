# Copyright (c) 2017 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)

# Set AMD Clang as default compiler
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
endif()
set(VERSION "1.3.1")

project(amd_nn VERSION ${VERSION} LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake)

find_package(miopen     PATHS ${ROCM_PATH} REQUIRED)

if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL")
    find_package(miopengemm PATHS ${ROCM_PATH} REQUIRED)
    find_package(OpenCL    REQUIRED)
    list(APPEND PACKAGE_DEPENDS PACKAGE OpenCL)
elseif(GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP")
    set(OpenCL_FOUND FALSE)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
    find_package(HIP REQUIRED)
    find_package(rocblas PATHS ${ROCM_PATH} REQUIRED)
    list(APPEND PACKAGE_DEPENDS PACKAGE HIP)
endif()

include_directories(../../amd_openvx/openvx/include
                    ${OpenCL_INCLUDE_DIRS}
                    src
                    include
                    ../../utilities/inference_generator
                    )

list(APPEND SOURCES
    src/kernels.cpp
    src/activation_layer.cpp
    src/convolution_layer.cpp
    src/deconvolution_layer.cpp
    src/fully_connected_layer.cpp
    src/normalization_layer.cpp
    src/local_response_normalization_layer.cpp
    src/pooling_layer.cpp
    src/roi_pooling_layer.cpp
    src/softmax_layer.cpp
    src/batch_normalization_layer.cpp
    src/concat_layer.cpp
    src/slice_layer.cpp
    src/image_tensor_converter.cpp
    src/tensor_image_converter.cpp
    src/tensor_add.cpp
    src/tensor_subtract.cpp
    src/tensor_multiply.cpp
    src/tensor_matrix_multiply.cpp
    src/tensor_min.cpp
    src/tensor_max.cpp
    src/tensor_exp.cpp
    src/tensor_log.cpp
    src/argmax_layer.cpp
    src/scale_layer.cpp
    src/upsample_nearest_layer.cpp
    src/tensor_table_lookup.cpp
    src/reshape_layer.cpp
    src/permute_layer.cpp
    src/prior_box_layer.cpp
    src/crop_layer.cpp
    src/crop_and_resize_layer.cpp
    src/detection_output.cpp
    src/cast_layer.cpp
    src/nms_layer.cpp
    src/gather_layer.cpp
    src/topk_layer.cpp
    src/reduce_min.cpp
    src/tile_layer.cpp
    src/tensor_compare.cpp
    src/profiler.cpp
    )


if(GPU_SUPPORT AND "${BACKEND}" STREQUAL "OPENCL"  AND OpenCL_FOUND)
    message("-- ${Green}amd_nn -- Building with OpenCL${ColourReset}")
    set(ENABLE_OPENCL 1)
    set(ENABLE_HIP 0)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP})
    include_directories(${OpenCL_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS}/Headers)
    add_library(vx_nn SHARED ${SOURCES})
    target_link_libraries(vx_nn openvx MIOpen miopengemm ${OpenCL_LIBRARIES})
elseif (GPU_SUPPORT AND "${BACKEND}" STREQUAL "HIP" AND HIP_FOUND)
    message("-- ${Green}amd_nn -- Building with HIP${ColourReset}")
    set(ENABLE_OPENCL 0)
    set(ENABLE_HIP 1)
    add_definitions(-DENABLE_OPENCL=${ENABLE_OPENCL} -DENABLE_HIP=${ENABLE_HIP} -D__HIP_PLATFORM_AMD__)
    include_directories(${ROCM_PATH}/include)
    link_directories(${HIP_PATH}/lib)
    add_subdirectory(nn_hip)
    add_library(vx_nn SHARED ${SOURCES} $<TARGET_OBJECTS:nn_hip>)
    set_target_properties(openvx PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(openvx PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(vx_nn openvx MIOpen roc::rocblas hip::host)
else()
    message("-- ${Red}WARNING: OpenCL/HIP Not Found -- amd_nn module excluded${ColourReset}")
endif()
set_target_properties(vx_nn PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# install MIVisionX libs -- {ROCM_PATH}/lib
install(TARGETS vx_nn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime NAMELINK_SKIP)
install(TARGETS vx_nn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT dev NAMELINK_ONLY)
install(TARGETS vx_nn LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT asan)
# install MIVisionX include files -- {ROCM_PATH}/include/mivisionx/
install(FILES include/vx_amd_nn.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mivisionx COMPONENT dev)
# install MIVisionX scripts -- {ROCM_PATH}/libexec/mivisionx/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/model_compiler DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/mivisionx COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/toolkit DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/mivisionx COMPONENT dev)
# install MIVisionX apps & samples -- {ROCM_PATH}/share/mivisionx/
install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/cloud_inference DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/apps COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/dg_test DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/apps COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/mivisionx_inference_analyzer DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/apps COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/apps/mivisionx_openvx_classifier DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/apps COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/mv_objdetect DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/samples/model_compiler_samples DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/samples COMPONENT dev)
# install MIVisionX tests -- {ROCM_PATH}/share/mivisionx/test
install(DIRECTORY ${CMAKE_SOURCE_DIR}/tests/neural_network_tests DESTINATION ${CMAKE_INSTALL_DATADIR}/mivisionx/test COMPONENT test)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
endif()

message("-- ${White}amd_nn -- CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}${ColourReset}")
