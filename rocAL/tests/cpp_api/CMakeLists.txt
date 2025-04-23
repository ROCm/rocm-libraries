################################################################################
#
# MIT License
#
# Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#
################################################################################
cmake_minimum_required(VERSION 3.10)
# ROCM Path
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
  message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
else()
  set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()
# Set AMD Clang as default compiler
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/bin/amdclang++")
    set(CMAKE_CXX_COMPILER clang++)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

project(rocal-test)

# make test with CTest
enable_testing()
include(CTest)

# find rocAL
find_library(ROCAL_LIBRARY NAMES rocal HINTS ${ROCM_PATH}/lib)
find_path(ROCAL_INCLUDE_DIR NAMES rocal_api.h PATHS ${ROCM_PATH}/include/rocal)

if(ROCAL_LIBRARY AND ROCAL_INCLUDE_DIR)
    set(ROCAL_FOUND TRUE)
    message("-- ${White}${PROJECT_NAME}: Using rocAL -- \n\tLibraries:${ROCAL_LIBRARY} \n\tIncludes:${ROCAL_INCLUDE_DIR}${ColourReset}")
endif()

if(NOT ROCAL_FOUND)
    message("-- ${Yellow}${PROJECT_NAME} requires rocAL. Install rocAL before running CTests")
else()
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${ROCM_PATH}/include/rpp/rpp_backend.h)
        file(READ ${ROCM_PATH}/include/rpp/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    endif()
    if(NOT DEFINED BACKEND AND RPP_BACKEND_HIP_FOUND)
        set(BACKEND "HIP")
    elseif(NOT DEFINED BACKEND)
        set(BACKEND "CPU")
    endif()
endif(NOT ROCAL_FOUND)

# 1 - basic_test_cpu
add_test(
  NAME
    basic_test_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/basic_test"
                              "${CMAKE_CURRENT_BINARY_DIR}/basic_test"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "basic_test"
            ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet-val.txt 1 0 224 224
)
# 2 - basic_test_gpu
add_test(NAME basic_test_gpu 
              COMMAND basic_test 
              ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet-val.txt 1 1 224 224
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)
# 3 - basic_test_gray
add_test(NAME basic_test_gray 
              COMMAND basic_test 
              ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet-val.txt 1 1 224 224 0
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)
# 4 - basic_test_rgb
add_test(NAME basic_test_rgb 
              COMMAND basic_test 
              ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet-val.txt 1 1 224 224 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)

# TBD - dataloader unit test options non-functional with database - NEEDS TO BE ADDED ONCE RESOLVED
#add_test(
#  NAME
#  dataloader
#  COMMAND
#    "${CMAKE_CTEST_COMMAND}"
#            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader"
#                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader"
#            --build-generator "${CMAKE_GENERATOR}"
#            --test-command "dataloader"
#            ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet
#)

# 5 - dataloader_multithread_cpu
add_test(
  NAME
  dataloader_multithread_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader_multithread"
                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader_multithread"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "dataloader_multithread"
            ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet 0
)
# 6 - dataloader_multithread_gpu
add_test(
  NAME
  dataloader_multithread_gpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader_multithread"
                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader_multithread_gpu"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "dataloader_multithread"
            ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet 1
)

# TBD - dataloader_tf unit test non-functional with tf dataset
#add_test(
#  NAME
#  dataloader_tf
#  COMMAND
#    "${CMAKE_CTEST_COMMAND}"
#            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/dataloader_tf"
#                              "${CMAKE_CURRENT_BINARY_DIR}/dataloader_tf"
#            --build-generator "${CMAKE_GENERATOR}"
#            --test-command "dataloader_tf"
#            ${CMAKE_CURRENT_BINARY_DIR}/data/images/AMD-tinyDataSet
#)

# performance_tests
# TBD - peformance test needs to run with default options
# 7 - performance_tests_cpu
add_test(
  NAME
    performance_tests_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/performance_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/performance_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "performance_tests"
            ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet 224 224 1 16 0
)
# 8 - performance_tests_gpu
add_test(NAME performance_tests_gpu 
              COMMAND performance_tests 
              ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet 224 224 1 16 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/performance_tests)

# 9 - performance_tests_with_depth_cpu
add_test(
  NAME
  performance_tests_with_depth_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/performance_tests_with_depth"
                              "${CMAKE_CURRENT_BINARY_DIR}/performance_tests_with_depth"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "performance_tests_with_depth"
            ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet 224 224 1 1 1 0
)
# 10 - performance_tests_with_depth_gpu
add_test(NAME performance_tests_with_depth_gpu 
              COMMAND performance_tests_with_depth 
              ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet 224 224 1 1 1 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/performance_tests_with_depth)

# 11 - unit_tests_cpu
add_test(
  NAME
    unit_tests_cpu
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/unit_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "unit_tests"
            0 ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet test 224 224 1 0 1
)
# 12 - unit_tests_gpu
add_test(NAME unit_tests_gpu 
              COMMAND unit_tests 
              0 ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet test 224 224 1 1 1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/unit_tests)
# 13 - unit_tests_gray
add_test(NAME unit_tests_gray 
              COMMAND unit_tests 
              0 ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet test 224 224 1 1 0
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/unit_tests)

# 14 - video_tests -- video reader with ffmpeg decoder
add_test(
  NAME
  video_tests_ffmpeg
  COMMAND
    "${CMAKE_CTEST_COMMAND}"
            --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/video_tests"
                              "${CMAKE_CURRENT_BINARY_DIR}/video_tests"
            --build-generator "${CMAKE_GENERATOR}"
            --test-command "video_tests"
            ${ROCM_PATH}/share/rocal/test/data/videos/AMD_driving_virtual_20.mp4
)

# 15 - video_tests -- video reader with rocDecode
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
find_package(ROCDECODE QUIET)
if(ROCDECODE_FOUND)
  if(${ROCDECODE_VERSION_MAJOR} VERSION_GREATER "0" OR
    (${ROCDECODE_VERSION_MAJOR} VERSION_EQUAL "0" AND ${ROCDECODE_VERSION_MINOR} VERSION_GREATER_EQUAL "12" AND ${ROCDECODE_VERSION_PATCH} VERSION_GREATER_EQUAL "0"))
    add_test(
      NAME video_tests_rocdecode
      COMMAND video_tests
                ${ROCM_PATH}/share/rocal/test/data/videos/AMD_driving_virtual_20.mp4 1 1 2
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/video_tests)
  endif()
endif()

# 16 - basic_test_rocjpeg -- basic test with rocJpeg
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
find_package(ROCJPEG QUIET)
if(ROCJPEG_FOUND)
  if(${ROCJPEG_VERSION_MAJOR} VERSION_GREATER "0" OR
     ${ROCJPEG_VERSION_MAJOR} VERSION_EQUAL "0" AND ${ROCJPEG_VERSION_MINOR} VERSION_GREATER_EQUAL "11" AND ${ROCJPEG_VERSION_PATCH} VERSION_GREATER_EQUAL "0")
    add_test(NAME basic_test_rocjpeg 
                  COMMAND basic_test 
                  ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet ${ROCM_PATH}/share/rocal/test/data/images/AMD-tinyDataSet-val.txt 1 1 224 224 1 1 2
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/basic_test)
  endif()
endif()
