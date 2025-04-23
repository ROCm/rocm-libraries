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

cmake_minimum_required(VERSION 3.10)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- rpp-test: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
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

project(rpp-test)
# make test with CTest
enable_testing()
include(CTest)

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(BoldBlue    "${Esc}[1;34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
endif()

#[[
List of high level dependency checks - for RPP QA tests in utilities/test_suite/CMakeLists.txt are below.
(The utilities/test_suite/HOST and utilities/test_suite/HIP folders are also designed to be built separately, so list of checks for those are present in the files "utilities/test_suite/HOST/CMakeLists.txt" and "utilities/test_suite/HIP/CMakeLists.txt")
- RPP (For ALL tests - to test any RPP functionalities)
- Python3 (For ALL tests - to run any RPP tests)
- Pandas (For ALL tests - to run and display rocprof performance test results)
- OpenMP (For ALL tests - to measure wall-times, and parallelize input tensor-image PKD3->PLN3 conversions across threads)
- ROCm HIP (For ALL HIP tests - HIP to be installed and found to run all RPP GPU HIP backend API)
- RPP_AUDIO_AUGMENTATIONS_SUPPORT (For AUDIO tests - to decide if RPP Audio tests script even needs to be built and executed)
- libsnd (For AUDIO tests - if RPP_AUDIO_AUGMENTATIONS_SUPPORT is enabled, to decide if RPP Audio tests script can be built and executed)
- NIFTI (For VOXEL tests - to decide if RPP Voxel QA tests script can be built and executed)
- OpenCV (For IMAGE and VOXEL tests - to decode image inputs and write images for unit test outputs)
- TurboJPEG (For IMAGE tests - to decode image inputs)
]]
# add find modules
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)

# find RPP
find_library(RPP_LIBRARY NAMES rpp HINTS ${ROCM_PATH}/lib)
find_path(RPP_INCLUDE_DIR NAMES rpp.h PATHS ${ROCM_PATH}/include/rpp)

if(RPP_LIBRARY AND RPP_INCLUDE_DIR)
    set(RPP_FOUND TRUE)
    message("-- ${White}${PROJECT_NAME}: Using RPP -- \n\tLibraries:${RPP_LIBRARY} \n\tIncludes:${RPP_INCLUDE_DIR}${ColourReset}")
endif()

if(NOT RPP_FOUND)
    message("-- ${Red}${PROJECT_NAME} requires RPP. Install RPP before running CTests")
else()
    # RPP installation - Backend check
    set(RPP_BACKEND_HIP_FOUND 0)
    if(EXISTS ${RPP_INCLUDE_DIR}/rpp_backend.h)
        file(READ ${RPP_INCLUDE_DIR}/rpp_backend.h RPP_BACKEND_FILE)
        string(REGEX MATCH "RPP_BACKEND_HIP ([0-9]*)" _ ${RPP_BACKEND_FILE})
        set(RPP_BACKEND_HIP_FOUND ${CMAKE_MATCH_1})
    endif()
    if(NOT DEFINED BACKEND AND RPP_BACKEND_HIP_FOUND)
        set(BACKEND "HIP")
    elseif(NOT DEFINED BACKEND)
        set(BACKEND "CPU")
    endif()
    message("-- ${White}${PROJECT_NAME}: Using RPP Backend: ${BACKEND}${ColourReset}")

    # RPP installation - Audio support check
    set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND 0)
    if(EXISTS ${RPP_INCLUDE_DIR}/rpp_audio_augmentations_support.h)
        file(READ ${RPP_INCLUDE_DIR}/rpp_audio_augmentations_support.h RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE)
        string(REGEX MATCH "RPP_AUDIO_AUGMENTATIONS_SUPPORT ([0-9]*)" _ ${RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE})
        set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND ${CMAKE_MATCH_1})
    endif()

    # find required libraries
    set(Python3_FIND_VIRTUALENV FIRST)
    find_package(Python3 QUIET)
    find_package(NIFTI QUIET)
    find_library(libsnd_LIBS
        NAMES sndfile libsndfile
        PATHS ${CMAKE_SYSTEM_PREFIX_PATH} ${LIBSND_ROOT_DIR} "/usr/local"
        PATH_SUFFIXES lib lib64)
    find_package(OpenMP QUIET)
    find_package(OpenCV QUIET)
    find_package(TurboJpeg QUIET)

    # find required python3-pip imports
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip show pandas
        RESULT_VARIABLE PANDAS_FOUND
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(Python3_FOUND)
        if (PANDAS_FOUND EQUAL 0)
            if(OpenMP_CXX_FOUND)
                message("-- ${White}${PROJECT_NAME}: Adding RPP HOST tests${ColourReset}")
                # HOST test set 1 - rpp_qa_tests_tensor_image_host_all - HOST Tensor-Image QA PASS/FAIL tests
                if(TurboJpeg_FOUND)
                    if(OpenCV_FOUND)
                        add_test(
                            NAME rpp_qa_tests_tensor_image_host_all
                            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runImageTests.py --qa_mode 1 --batch_size 3
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        )
                    else()
                        message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 1 - rpp_qa_tests_tensor_image_host_all' since OpenCV is not found!${ColourReset}")
                    endif(OpenCV_FOUND)
                else()
                    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 1 - rpp_qa_tests_tensor_image_host_all' since TurboJpeg is not found!${ColourReset}")
                endif(TurboJpeg_FOUND)

                # HOST test set 2 - rpp_qa_tests_tensor_voxel_host_all - HOST Tensor-Voxel QA PASS/FAIL tests
                if(NIFTI_FOUND)
                    if(OpenCV_FOUND)
                        add_test(
                            NAME rpp_qa_tests_tensor_voxel_host_all
                            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runVoxelTests.py --qa_mode 1 --batch_size 3
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        )
                    else()
                        message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 2 - rpp_qa_tests_tensor_voxel_host_all' since OpenCV is not found!${ColourReset}")
                    endif(OpenCV_FOUND)
                else()
                    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 2 - rpp_qa_tests_tensor_voxel_host_all' since libniftiio is not found!${ColourReset}")
                endif(NIFTI_FOUND)

                # HOST test set 3 - rpp_qa_tests_tensor_audio_host_all - HOST Tensor-Audio QA PASS/FAIL tests
                if(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)
                    if(libsnd_LIBS)
                        add_test(
                            NAME rpp_qa_tests_tensor_audio_host_all
                            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runAudioTests.py --qa_mode 1 --batch_size 3
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        )
                    else()
                        message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 3 - rpp_qa_tests_tensor_audio_host_all' since libsnd is not found!${ColourReset}")
                    endif(libsnd_LIBS)
                else()
                    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 3 - rpp_qa_tests_tensor_audio_host_all' since RPP_AUDIO_AUGMENTATIONS_SUPPORT is not found!${ColourReset}")
                endif(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)

                # HOST test set 4 - rpp_qa_tests_tensor_misc_host_all - HOST Tensor-Misc QA PASS/FAIL tests
                add_test(
                    NAME rpp_qa_tests_tensor_misc_host_all
                    COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HOST/runMiscTests.py --qa_mode 1 --batch_size 3
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                )

                if( "${BACKEND}" STREQUAL "HIP")
                    if(NOT DEFINED HIP_PATH)
                        if(NOT DEFINED ENV{HIP_PATH})
                            set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
                        else()
                            set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
                        endif()
                    endif()
                    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip)
                    find_package(HIP QUIET)
                    if(HIP_FOUND)
                        message("-- ${White}${PROJECT_NAME}: Adding RPP HIP tests${ColourReset}")
                        # HIP test set 1 - rpp_qa_tests_tensor_image_hip_all - HIP Tensor-Image QA PASS/FAIL tests
                        if(TurboJpeg_FOUND)
                            if(OpenCV_FOUND)
                                add_test(
                                    NAME rpp_qa_tests_tensor_image_hip_all
                                    COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HIP/runImageTests.py --qa_mode 1 --batch_size 3
                                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                                )
                            else()
                                message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 1 - rpp_qa_tests_tensor_image_hip_all' since OpenCV is not found!${ColourReset}")
                            endif(OpenCV_FOUND)
                        else()
                            message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 1 - rpp_qa_tests_tensor_image_hip_all' since TurboJpeg is not found!${ColourReset}")
                        endif(TurboJpeg_FOUND)

                        # HIP test set 2 - rpp_qa_tests_tensor_voxel_hip_all - HIP Tensor-Voxel QA PASS/FAIL tests
                        if(NIFTI_FOUND)
                            if(OpenCV_FOUND)
                                add_test(
                                    NAME rpp_qa_tests_tensor_voxel_hip_all
                                    COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HIP/runVoxelTests.py --qa_mode 1 --batch_size 3
                                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                                )
                            else()
                                message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 2 - rpp_qa_tests_tensor_voxel_hip_all' since OpenCV is not found!${ColourReset}")
                            endif(OpenCV_FOUND)
                        else()
                            message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 2 - rpp_qa_tests_tensor_voxel_hip_all' since libniftiio is not found!${ColourReset}")
                        endif(NIFTI_FOUND)

                        # HIP test set 3 - rpp_qa_tests_tensor_audio_hip_all - HIP Tensor-Audio QA PASS/FAIL tests
                        if(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)
                            if(libsnd_LIBS)
                                add_test(
                                    NAME rpp_qa_tests_tensor_audio_hip_all
                                    COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HIP/runAudioTests.py --qa_mode 1 --batch_size 3
                                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                                )
                            else()
                                message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 3 - rpp_qa_tests_tensor_audio_hip_all' since libsnd is not found!${ColourReset}")
                            endif(libsnd_LIBS)
                        else()
                            message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 3 - rpp_qa_tests_tensor_audio_hip_all' since RPP_AUDIO_AUGMENTATIONS_SUPPORT is not found!${ColourReset}")
                        endif(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)

                        # HIP test set 4 - rpp_qa_tests_tensor_misc_hip_all - HIP Tensor-Misc QA PASS/FAIL tests
                        add_test(
                            NAME rpp_qa_tests_tensor_misc_hip_all
                            COMMAND ${Python3_EXECUTABLE} ${ROCM_PATH}/share/rpp/test/HIP/runMiscTests.py --qa_mode 1 --batch_size 3
                            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                        )

                    endif(HIP_FOUND)
                elseif( "${BACKEND}" STREQUAL "OCL")
                    # TBD: Add OCL Tests
                    message("-- ${Yellow}${PROJECT_NAME} Warning: OpenCL tests not enabled${ColourReset}")
                endif()
            else()
                message("-- ${Yellow}Warning: Skipping all 4 RPP-ctest test sets since OpenMP is not found!${ColourReset}")
            endif(OpenMP_CXX_FOUND)
        else()
            message("-- ${Yellow}Warning: Skipping all 4 RPP-ctest test sets since Pandas is not found!${ColourReset}")
        endif(PANDAS_FOUND EQUAL 0)
    else()
        message("-- ${Yellow}Warning: Skipping all 4 RPP-ctest test sets since Python3 is not found!${ColourReset}")
    endif(Python3_FOUND)
endif(NOT RPP_FOUND)
