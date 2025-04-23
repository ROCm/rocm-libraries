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
    message("-- test_suite-HIP: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
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

project(test_suite/HIP)

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
utilities/test_suite/HIP:
Independent dependency-checks for Unit + Performance + QA tests in utilities/test_suite/HIP/CMakeLists.txt are:
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
        message("-- ${Red}${PROJECT_NAME} is not supported to build and run with RPP HOST backend installation${ColourReset}")
    endif()
    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${RPP_LIBRARY})
    include_directories(${RPP_INCLUDE_DIR})

    # RPP HIP test suite works for RPP built with HIP backend
    if("${BACKEND}" STREQUAL "HIP")

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
        find_package(HIP QUIET)
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
        # execute_process(                                          # added only on HOST
        #     COMMAND ${Python3_EXECUTABLE} -m pip show openpyxl    # added only on HOST
        #     RESULT_VARIABLE OPENPYXL_FOUND                        # added only on HOST
        #     OUTPUT_QUIET                                          # added only on HOST
        #     ERROR_QUIET                                           # added only on HOST
        # )

        if(HIP_FOUND)
            message("-- ${Green}HIP found at HIP_INCLUDE_DIRS - ${HIP_INCLUDE_DIRS} and HIP_LIBRARIES - ${HIP_LIBRARIES}${ColourReset}")
            include_directories(${HIP_INCLUDE_DIRS} ${HIP_INCLUDE_DIRS}/hip)
            set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${HIP_LIBRARIES})

            if(Python3_FOUND)
                message("-- ${Green}Python3 found${ColourReset}")
                # Check pandas for all RPP rocprof performance test results
                if (PANDAS_FOUND EQUAL 0)
                    message("-- ${Green}Python3 found at ${Python3_EXECUTABLE}${ColourReset}")
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPU_SUPPORT=1 -DRPP_BACKEND_HIP=1 -std=gnu++17")

                    # include and link directories
                    include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
                    link_directories(${ROCM_PATH}/lib /usr/local/lib)
                    link_directories(${ROCM_PATH}/lib/llvm/lib)

                    if(OpenMP_CXX_FOUND)
                        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
                        set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenMP_CXX_LIBRARIES})
                        message("-- ${White}${PROJECT_NAME}: Using OpenMP -- \n\tOpenMP_CXX_FLAGS:${OpenMP_CXX_FLAGS}\n\tOpenMP_CXX_LIBRARIES:${OpenMP_CXX_LIBRARIES}${ColourReset}")
                        message("-- ${Green}${PROJECT_NAME} All Tensor HIP tests set to build with OpenMP${ColourReset}")

                        # HIP test set 1 - RPP Tensor-Image HIP test suite
                        if(TurboJpeg_FOUND)
                            include_directories(${TurboJpeg_INCLUDE_DIRS})
                            set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${TurboJpeg_LIBRARIES})
                            message("-- ${Green}${PROJECT_NAME} Tensor-Image HIP tests set to build with TurboJpeg${ColourReset}")
                            if(OpenCV_FOUND)
                                include_directories(${OpenCV_INCLUDE_DIRS})
                                set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} ${OpenCV_LIBRARIES})
                                message("-- ${Green}${PROJECT_NAME} Tensor-Image HIP tests set to build with OpenCV${ColourReset}")
                                add_executable(Tensor_image_hip Tensor_image_hip.cpp)
                                target_link_libraries(Tensor_image_hip ${LINK_LIBRARY_LIST})
                                message("-- ${White}Tensor_image_hip -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
                                # if (OPENPYXL_FOUND EQUAL 0)                                                                                                                                   # added only on HOST
                                #     message("-- ${Green}Openpyxl for Python3 found ${ColourReset}")                                                                                           # added only on HOST
                                #     add_executable(BatchPD_hip_pkd3 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_hip_pkd3.cpp) # OLD - To be deprecated and removed      # added only on HOST
                                #     add_executable(BatchPD_hip_pln1 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_hip_pln1.cpp) # OLD - To be deprecated and removed      # added only on HOST
                                #     add_executable(BatchPD_hip_pln3 ${ROCM_PATH}/share/rpp/test/rpp-performancetests/HOST_NEW/BatchPD_hip_pln3.cpp) # OLD - To be deprecated and removed      # added only on HOST
                                #     target_link_libraries(BatchPD_hip_pkd3 ${OpenCV_LIBS} -lturbojpeg -lrpp ${LINK_LIBRARY_LIST}) # OLD - To be deprecated and removed                # added only on HOST
                                #     target_link_libraries(BatchPD_hip_pln1 ${OpenCV_LIBS} -lturbojpeg -lrpp ${LINK_LIBRARY_LIST}) # OLD - To be deprecated and removed                # added only on HOST
                                #     target_link_libraries(BatchPD_hip_pln3 ${OpenCV_LIBS} -lturbojpeg -lrpp ${LINK_LIBRARY_LIST}) # OLD - To be deprecated and removed                # added only on HOST
                                # else()                                                                                                                                                        # added only on HOST
                                #     message("-- ${Yellow}Warning: Openpyxl for Python3 must be installed to run any RPP test_suite QA performance tests successfully${ColourReset}")          # added only on HOST
                                # endif(OPENPYXL_FOUND EQUAL 0)                                                                                                                                 # added only on HOST
                            else()
                                message("-- ${Yellow}Warning: OpenCV must be installed to install ${PROJECT_NAME} Tensor-Image tests successfully!${ColourReset}")
                            endif(OpenCV_FOUND)
                        else()
                            message("-- ${Yellow}Warning: TurboJpeg must be installed to install ${PROJECT_NAME} Tensor-Image tests successfully!${ColourReset}")
                        endif(TurboJpeg_FOUND)

                        # HIP test set 2 - RPP Tensor-Voxel HIP test suite
                        if(NIFTI_FOUND)
                            message("-- ${Green}${PROJECT_NAME} Tensor-Voxel HIP tests set to build with libniftiio${ColourReset}")
                            if(OpenCV_FOUND)
                                message("-- ${Green}${PROJECT_NAME} Tensor-Voxel HIP tests set to build with OpenCV${ColourReset}")
                                add_executable(Tensor_voxel_hip Tensor_voxel_hip.cpp)
                                target_link_libraries(Tensor_voxel_hip ${LINK_LIBRARY_LIST} ${NIFTI_PACKAGE_PREFIX}NIFTI::${NIFTI_PACKAGE_PREFIX}niftiio)
                                message("-- ${White}Tensor_voxel_hip -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
                            else()
                                message("-- ${Yellow}Warning: OpenCV must be installed to install ${PROJECT_NAME} Tensor-Voxel tests successfully!${ColourReset}")
                            endif(OpenCV_FOUND)
                        else()
                            message("-- ${Yellow}Warning: libniftiio must be installed to install ${PROJECT_NAME} Tensor-Voxel tests successfully!${ColourReset}")
                        endif(NIFTI_FOUND)

                        # HIP test set 3 - RPP Tensor-Audio HIP test suite
                        if(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)
                            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUDIO_SUPPORT=1")
                            message("-- ${Green}${PROJECT_NAME} Tensor-Audio HIP tests set to build with RPP_AUDIO_SUPPORT set to ON${ColourReset}")
                            if(libsnd_LIBS)
                                include_directories(${SndFile_INCLUDE_DIRS})
                                link_directories(${SndFile_LIBRARIES_DIR} /usr/local/lib/)
                                message("-- ${Green}${PROJECT_NAME} Tensor-Audio HIP tests set to build with libsnd${ColourReset}")
                                add_executable(Tensor_audio_hip Tensor_audio_hip.cpp)
                                target_link_libraries(Tensor_audio_hip ${LINK_LIBRARY_LIST} ${libsnd_LIBS} -lsndfile)
                                message("-- ${White}Tensor_audio_hip -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
                            else()
                                message("-- ${Yellow}Warning: libsnd must be installed to install ${PROJECT_NAME} Tensor-Audio tests successfully!${ColourReset}")
                            endif(libsnd_LIBS)
                        else()
                            message("-- ${Yellow}Warning: RPP_AUDIO_SUPPORT not found in RPP installation. It must be installed to build ${PROJECT_NAME} Tensor-Audio tests successfully!${ColourReset}")
                        endif(RPP_AUDIO_AUGMENTATIONS_SUPPORT_FOUND)

                        # HIP test set 4 - RPP Tensor-Misc HIP test suite
                        add_executable(Tensor_misc_hip Tensor_misc_hip.cpp)
                        target_link_libraries(Tensor_misc_hip ${LINK_LIBRARY_LIST})
                        message("-- ${White}Tensor_misc_hip -- Link Libraries: ${LINK_LIBRARY_LIST}${ColourReset}")
                    else()
                        message("-- ${Red}${PROJECT_NAME} Error: OpenMP must be installed to run any RPP tests successfully!${ColourReset}")
                    endif(OpenMP_CXX_FOUND)
                else()
                    message("-- ${Red}${PROJECT_NAME} Error: Pandas for Python3 must be installed to run any RPP test_suite performance tests with profiling successfully${ColourReset}")
                endif(PANDAS_FOUND EQUAL 0)
            else()
                message("-- ${Red}${PROJECT_NAME} Error: Python3 must be installed to run any RPP test_suite tests successfully${ColourReset}")
            endif(Python3_FOUND)
        else()
            message("-- ${Red}${PROJECT_NAME} Error: ROCm HIP must be installed to run any RPP test_suite HIP tests successfully${ColourReset}")
        endif(HIP_FOUND)
    else()
        message("-- ${Red}${PROJECT_NAME} is supported to build and run only with RPP HIP backend installation${ColourReset}")
    endif("${BACKEND}" STREQUAL "HIP")
endif(NOT RPP_FOUND)
