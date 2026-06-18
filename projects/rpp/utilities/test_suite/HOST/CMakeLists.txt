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
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS ON)
if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang)
    set(CMAKE_CXX_COMPILER ${ROCM_PATH}/lib/llvm/bin/amdclang++)
elseif(NOT DEFINED CMAKE_CXX_COMPILER AND NOT EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
endif()

project(test_suite/HOST)

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)

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
- Pandas (Optional - to print consolidated rocprof CSV tables from Python; not required for QA/unit drivers or Tensor_misc HOST build)
- OpenMP (For ALL tests - to measure wall-times, and parallelize input tensor-image PKD3->PLN3 conversions across threads)
- RPP_AUDIO_AUGMENTATIONS_SUPPORT (For AUDIO tests - to decide if RPP Audio tests script even needs to be built and executed)
- libsnd (For AUDIO tests - if RPP_AUDIO_AUGMENTATIONS_SUPPORT is enabled, to decide if RPP Audio tests script can be built and executed)
- NIFTI (For VOXEL tests - to decide if RPP Voxel QA tests script can be built and executed)
- OpenCV (Optional — enables decoder_type 1, unit image dumps, voxel slice JPGs; Tensor_image/voxel build without it when absent)
- RGB inputs (decoder_type 0): Tensor-Image tests expect pre-decoded ``.rgb`` files with an embedded 24-byte header; Generate or refresh them with the scripts under ``utilities/test_suite/scripts/`` (see ``scripts/README.md``)
]]

# find RPP
find_package(rpp 3.0.0 QUIET)
if(rpp_FOUND)
    message("-- ${White}${PROJECT_NAME}: rpp found with find_package(rpp QUIET)${ColourReset}")
    message("-- \t${White}rpp_INCLUDE_DIR -- ${rpp_INCLUDE_DIR}${ColourReset}")
    message("-- \t${White}rpp_LIBRARY -- ${rpp_LIBRARY}${ColourReset}")
    message("-- \t${White}rpp_BACKEND_TYPE -- ${rpp_BACKEND_TYPE}${ColourReset}")
    message("-- \t${White}rpp_AUDIO_AUGMENTATIONS_SUPPORT -- ${rpp_AUDIO_AUGMENTATIONS_SUPPORT}${ColourReset}")
else()
    message("-- ${Yellow}${PROJECT_NAME} requires rpp. Install rpp before running CTests")
endif(rpp_FOUND)

if(NOT rpp_FOUND)
    message("-- ${Red}${PROJECT_NAME} requires RPP. Install RPP before running RPP Test Suite")
else()
    # RPP installation - Backend check
    if(rpp_BACKEND_TYPE STREQUAL "HIP")
        message("-- ${Green}${PROJECT_NAME} is supported to build and run with RPP HIP backend installation${ColourReset}")
    else()
        message("-- ${Red}${PROJECT_NAME} is not supported to build and run with RPP HIP backend installation${ColourReset}")
    endif()

    set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} rpp::rpp)
    include_directories(${rpp_INCLUDE_DIR})

    # RPP HOST test suite works for RPP built with HIP or HOST backend
    if(rpp_BACKEND_TYPE STREQUAL "HIP" OR rpp_BACKEND_TYPE STREQUAL "CPU")
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
        find_package(OpenMP COMPONENTS CXX QUIET)
        find_package(OpenCV QUIET)
        find_package(StdFilesystem QUIET)

        # find required python3 imports (only if Python3 is found)
        if(Python3_FOUND)
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pandas"
                RESULT_VARIABLE PANDAS_RESULT
                OUTPUT_QUIET
                ERROR_QUIET
            )
            # execute_process returns 0 on success (pandas found), non-zero on failure (pandas not found)
            if(PANDAS_RESULT EQUAL 0)
                set(PANDAS_FOUND 1)
            else()
                set(PANDAS_FOUND 0)
            endif()

            # Python3 status message (only needed for running Python scripts, not for building executables)
            message("-- ${Green}Python3 found at ${Python3_EXECUTABLE}${ColourReset}")
            if(NOT PANDAS_FOUND)
                message("-- ${Yellow}Warning: Pandas not found (optional — needed only to print consolidated rocprof CSV tables from Python).${ColourReset}")
            endif()
        else()
            set(PANDAS_FOUND 0)  # Set to not-found value when Python3 is missing
            message("-- ${Yellow}${PROJECT_NAME} Warning: Python3 not found — executables will be built but Python test scripts cannot run${ColourReset}")
        endif()

        # include and link directories
        include_directories(${ROCM_PATH}/include ${ROCM_PATH}/include/rpp /usr/local/include)
        link_directories(${ROCM_PATH}/lib /usr/local/lib)

        if(NOT APPLE)
            set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} stdc++fs)
        endif()

        if(OpenMP_CXX_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
            set(LINK_LIBRARY_LIST ${LINK_LIBRARY_LIST} OpenMP::OpenMP_CXX)
            message("-- ${Green}${PROJECT_NAME} All Tensor HOST tests set to build with OpenMP${ColourReset}")

            # HOST test set 1 - RPP Tensor-Image HOST test suite (decoder_type 0: packed .raw + .info)
            add_executable(Tensor_image_host Tensor_image_host.cpp)
            target_link_libraries(Tensor_image_host ${LINK_LIBRARY_LIST})
            if(OpenCV_FOUND)
                target_include_directories(Tensor_image_host PRIVATE ${OpenCV_INCLUDE_DIRS})
                target_link_libraries(Tensor_image_host ${OpenCV_LIBS})
                target_compile_definitions(Tensor_image_host PRIVATE RPP_TEST_SUITE_HAVE_OPENCV=1)
                message("-- ${Green}${PROJECT_NAME} Tensor-Image HOST tests set to build with OpenCV (decoder_type 1 and image dumps enabled)${ColourReset}")
            else()
                target_compile_definitions(Tensor_image_host PRIVATE RPP_TEST_SUITE_HAVE_OPENCV=0)
                message("-- ${Yellow}Warning: OpenCV not found — Tensor_image_host builds without OpenCV; decoder_type 1 / unit image dumps disabled.${ColourReset}")
            endif()
            message("-- ${White}Tensor_image_host -- Link Libraries: see target_link_libraries${ColourReset}")

            # HOST test set 2 - RPP Tensor-Voxel HOST test suite
            if(NIFTI_FOUND)
                message("-- ${Green}${PROJECT_NAME} Tensor-Voxel HOST tests set to build with libniftiio${ColourReset}")
                add_executable(Tensor_voxel_host Tensor_voxel_host.cpp)
                target_link_libraries(Tensor_voxel_host ${NIFTI_PACKAGE_PREFIX}NIFTI::${NIFTI_PACKAGE_PREFIX}niftiio ${LINK_LIBRARY_LIST})
                if(OpenCV_FOUND)
                    target_include_directories(Tensor_voxel_host PRIVATE ${OpenCV_INCLUDE_DIRS})
                    target_link_libraries(Tensor_voxel_host ${OpenCV_LIBS})
                    target_compile_definitions(Tensor_voxel_host PRIVATE RPP_TEST_SUITE_HAVE_OPENCV=1)
                    message("-- ${Green}${PROJECT_NAME} Tensor-Voxel HOST tests set to build with OpenCV (slice JPG dumps enabled)${ColourReset}")
                else()
                    target_compile_definitions(Tensor_voxel_host PRIVATE RPP_TEST_SUITE_HAVE_OPENCV=0)
                    message("-- ${Yellow}Warning: OpenCV not found — Tensor_voxel_host builds without OpenCV; F32 golden QA still runs; non-QA slice dumps disabled.${ColourReset}")
                endif()
            else()
                message("-- ${Yellow}Warning: libniftiio must be installed to install ${PROJECT_NAME} Tensor-Voxel tests successfully!${ColourReset}")
            endif(NIFTI_FOUND)

            # HOST test set 3 - RPP Tensor-Audio HOST test suite
            if(rpp_AUDIO_AUGMENTATIONS_SUPPORT)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DAUDIO_SUPPORT=1")
                message("-- ${Green}${PROJECT_NAME} Tensor-Audio HOST tests set to build with RPP_AUDIO_SUPPORT set to ON${ColourReset}")
                if(libsnd_LIBS)
                    include_directories(${SndFile_INCLUDE_DIRS})
                    link_directories(${SndFile_LIBRARIES_DIR} /usr/local/lib/)
                    message("-- ${Green}${PROJECT_NAME} Tensor-Audio HOST tests set to build with libsnd${ColourReset}")
                    add_executable(Tensor_audio_host Tensor_audio_host.cpp)
                    target_link_libraries(Tensor_audio_host ${libsnd_LIBS} -lsndfile ${LINK_LIBRARY_LIST})
                else()
                    message("-- ${Yellow}Warning: libsnd must be installed to install ${PROJECT_NAME} Tensor-Audio tests successfully!${ColourReset}")
                endif(libsnd_LIBS)
            else()
                message("-- ${Yellow}Warning: RPP_AUDIO_SUPPORT not found in RPP installation. It must be installed to build ${PROJECT_NAME} Tensor-Audio tests successfully!${ColourReset}")
            endif(rpp_AUDIO_AUGMENTATIONS_SUPPORT)

            # HOST test set 4 - RPP Tensor-Misc HOST test suite
            add_executable(Tensor_misc_host Tensor_misc_host.cpp)
            target_link_libraries(Tensor_misc_host ${LINK_LIBRARY_LIST})

        else()
            message("-- ${Red}${PROJECT_NAME} Error: OpenMP must be installed to build any RPP tests successfully!${ColourReset}")
        endif(OpenMP_CXX_FOUND)
    endif(rpp_BACKEND_TYPE STREQUAL "HIP" OR rpp_BACKEND_TYPE STREQUAL "CPU")
endif(NOT rpp_FOUND)
