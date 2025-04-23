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
    message("-- test_suite-HOST: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
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

project(test_suite/HOST)

list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

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
utilities/test_suite/HOST:
Independent dependency-checks for Unit + Performance + QA tests in utilities/test_suite/HOST/CMakeLists.txt are:
- RPP (For ALL tests - to test any RPP functionalities)
- Python3 (For ALL tests - to run any RPP tests)
- Pandas (For ALL tests - to run and display rocprof performance test results)
- OpenMP (For ALL tests - to measure wall-times, and parallelize input tensor-image PKD3->PLN3 conversions across threads)
- RPP_AUDIO_AUGMENTATIONS_SUPPORT (For AUDIO tests - to decide if RPP Audio tests script even needs to be built and executed)
- libsnd (For AUDIO tests - if RPP_AUDIO_AUGMENTATIONS_SUPPORT is enabled, to decide if RPP Audio tests script can be built and executed)
- NIFTI (For VOXEL tests - to decide if RPP Voxel QA tests script can be built and executed)
- OpenCV (For IMAGE and VOXEL tests - to decode image inputs and write images for unit test outputs)
- TurboJPEG (For IMAGE tests - to decode image inputs)
- Openpyxl (For IMAGE tests - to run and display QA performance test results)
]]

# find RPP
find_library(RPP_LIBRARY NAMES rpp HINTS ${ROCM_PATH}/lib)
find_path(RPP_INCLUDE_DIR NAMES rpp.h PATHS ${ROCM_PATH}/include/rpp)

if(RPP_LIBRARY AND RPP_INCLUDE_DIR)
    set(RPP_FOUND TRUE)
    message("-- ${White}${PROJECT_NAME}: Using RPP -- \n\tLibraries:${RPP_LIBRARY} \n\tIncludes:${RPP_INCLUDE_DIR}${ColourReset}")
endif()

if(NOT RPP_FOUND)
    message("-- ${Red}${PROJECT_NAME} requires RPP. Install RPP before running RPP Test Suite")
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
        message("-- ${Green}${PROJECT_NAME} is supported to build and run with RPP HIP backend installation${ColourReset}")
    elseif(NOT DEFINED BACKEND)
        set(BACKEND "CPU")
        message("-- ${Green}${PROJECT_NAME} is supported to build and run with RPP HOST backend installation${ColourReset}")
    endif()

    # RPP HOST test suite works for RPP built with HIP or HOST backend
    if("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "CPU")

        # RPP installation - Audio support check
        set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND 0)
        if(EXISTS ${RPP_INCLUDE_DIR}/rpp_audio_augmentations_support.h)
            file(READ ${RPP_INCLUDE_DIR}/rpp_audio_augmentations_support.h RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE)
            string(REGEX MATCH "RPP_AUDIO_AUGMENTATIONS_SUPPORT ([0-9]*)" _ ${RPP_AUDIO_AUGMENTATIONS_SUPPORT_FILE})
            set(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND ${CMAKE_MATCH_1})
        endif()

        # add find modules
        list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)

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
        find_package(StdFilesystem QUIET)

        # find required python3-pip imports
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip show pandas
            RESULT_VARIABLE PANDAS_FOUND
            OUTPUT_QUIET
            ERROR_QUIET
        )
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip show openpyxl
            RESULT_VARIABLE OPENPYXL_FOUND
            OUTPUT_QUIET
            ERROR_QUIET
        )

        if(Python3_FOUND)
            message("-- ${Green}Python3 found at ${Python3_EXECUTABLE}${ColourReset}")
            # Check pandas for all RPP rocprof performance test results
            if (PANDAS_FOUND EQUAL 0)
                message("-- ${Green}Pandas for Python3 found${ColourReset}")

                # include and link directories
                include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
                link_directories(${ROCM_PATH}/lib /usr/local/lib)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")

                if(OpenMP_CXX_FOUND)
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
                    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenMP_CXX_LIBRARIES})
                    message("-- ${White}${PROJECT_NAME}: Using OpenMP -- \n\tOpenMP_CXX_FLAGS:${OpenMP_CXX_FLAGS}\n\tOpenMP_CXX_LIBRARIES:${OpenMP_CXX_LIBRARIES}${ColourReset}")
                    message("-- ${Green}${PROJECT_NAME} All Tensor HOST tests set to build with OpenMP${ColourReset}")

                    # HOST test set 1 - RPP Tensor-Image HOST test suite
                    if(TurboJpeg_FOUND)
                        include_directories(${TurboJpeg_INCLUDE_DIRS})
                        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES_DIR})
                        message("-- ${Green}${PROJECT_NAME} Tensor-Image HOST tests set to build with TurboJpeg${ColourReset}")
                        if(OpenCV_FOUND)
                            include_directories(${OpenCV_INCLUDE_DIRS})
                            message("-- ${Green}${PROJECT_NAME} Tensor-Image HOST tests set to build with OpenCV${ColourReset}")
                            add_executable(Tensor_image_host Tensor_image_host.cpp)
                            target_link_libraries(Tensor_image_host -lrpp -lturbojpeg ${OpenCV_LIBS} ${LINK_LIBRARY_LIST})
                            if (OPENPYXL_FOUND EQUAL 0)
                                message("-- ${Green}Openpyxl for Python3 found ${ColourReset}")
                                add_executable(BatchPD_host_pkd3 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_host_pkd3.cpp) # OLD - To be deprecated and removed
                                add_executable(BatchPD_host_pln1 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_host_pln1.cpp) # OLD - To be deprecated and removed
                                add_executable(BatchPD_host_pln3 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_host_pln3.cpp) # OLD - To be deprecated and removed
                                target_link_libraries(BatchPD_host_pkd3 ${OpenCV_LIBS} -lturbojpeg -lrpp ${LINK_LIBRARY_LIST}) # OLD - To be deprecated and removed
                                target_link_libraries(BatchPD_host_pln1 ${OpenCV_LIBS} -lturbojpeg -lrpp ${LINK_LIBRARY_LIST}) # OLD - To be deprecated and removed
                                target_link_libraries(BatchPD_host_pln3 ${OpenCV_LIBS} -lturbojpeg -lrpp ${LINK_LIBRARY_LIST}) # OLD - To be deprecated and removed
                            else()
                                message("-- ${Yellow}Warning: Openpyxl for Python3 must be installed to run any RPP test_suite QA performance tests successfully${ColourReset}")
                            endif(OPENPYXL_FOUND EQUAL 0)
                        else()
                            message("-- ${Yellow}Warning: OpenCV must be installed to install ${PROJECT_NAME} Tensor-Image tests successfully!${ColourReset}")
                        endif(OpenCV_FOUND)
                    else()
                        message("-- ${Yellow}Warning: TurboJpeg must be installed to install ${PROJECT_NAME} Tensor-Image tests successfully!${ColourReset}")
                    endif(TurboJpeg_FOUND)

                    # HOST test set 2 - RPP Tensor-Voxel HOST test suite
                    if(NIFTI_FOUND)
                        message("-- ${Green}${PROJECT_NAME} Tensor-Voxel HOST tests set to build with libniftiio${ColourReset}")
                        if(OpenCV_FOUND)
                            include_directories(${OpenCV_INCLUDE_DIRS})
                            message("-- ${Green}${PROJECT_NAME} Tensor-Voxel HOST tests set to build with OpenCV${ColourReset}")
                            add_executable(Tensor_voxel_host Tensor_voxel_host.cpp)
                            target_link_libraries(Tensor_voxel_host -lrpp ${NIFTI_PACKAGE_PREFIX}NIFTI::${NIFTI_PACKAGE_PREFIX}niftiio ${OpenCV_LIBS} ${LINK_LIBRARY_LIST})
                        else()
                            message("-- ${Yellow}Warning: OpenCV must be installed to install ${PROJECT_NAME} Tensor-Voxel tests successfully!${ColourReset}")
                        endif(OpenCV_FOUND)
                    else()
                        message("-- ${Yellow}Warning: libniftiio must be installed to install ${PROJECT_NAME} Tensor-Voxel tests successfully!${ColourReset}")
                    endif(NIFTI_FOUND)

                    # HOST test set 3 - RPP Tensor-Audio HOST test suite
                    if(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUDIO_SUPPORT=1")
                        message("-- ${Green}${PROJECT_NAME} Tensor-Audio HOST tests set to build with RPP_AUDIO_SUPPORT set to ON${ColourReset}")
                        if(libsnd_LIBS)
                            include_directories(${SndFile_INCLUDE_DIRS})
                            link_directories(${SndFile_LIBRARIES_DIR} /usr/local/lib/)
                            message("-- ${Green}${PROJECT_NAME} Tensor-Audio HOST tests set to build with libsnd${ColourReset}")
                            add_executable(Tensor_audio_host Tensor_audio_host.cpp)
                            target_link_libraries(Tensor_audio_host -lrpp ${libsnd_LIBS} -lsndfile ${LINK_LIBRARY_LIST})
                        else()
                            message("-- ${Yellow}Warning: libsnd must be installed to install ${PROJECT_NAME} Tensor-Audio tests successfully!${ColourReset}")
                        endif(libsnd_LIBS)
                    else()
                        message("-- ${Yellow}Warning: RPP_AUDIO_SUPPORT not found in RPP installation. It must be installed to build ${PROJECT_NAME} Tensor-Audio tests successfully!${ColourReset}")
                    endif(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)

                    # HOST test set 4 - RPP Tensor-Misc HOST test suite
                    add_executable(Tensor_misc_host Tensor_misc_host.cpp)
                    target_link_libraries(Tensor_misc_host -lrpp ${LINK_LIBRARY_LIST})

                else()
                    message("-- ${Red}${PROJECT_NAME} Error: OpenMP must be installed to run any RPP tests successfully!${ColourReset}")
                endif(OpenMP_CXX_FOUND)
            else()
                message("-- ${Red}${PROJECT_NAME} Error: Pandas for Python3 must be installed to run any RPP test_suite performance tests with profiling successfully${ColourReset}")
            endif(PANDAS_FOUND EQUAL 0)
        else()
            message("-- ${Red}${PROJECT_NAME} Error: Python3 must be installed to run any RPP test_suite tests successfully${ColourReset}")
        endif(Python3_FOUND)
    endif("${BACKEND}" STREQUAL "HIP" OR "${BACKEND}" STREQUAL "CPU")
endif(NOT RPP_FOUND)
