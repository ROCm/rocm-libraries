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

if(NOT DEFINED ENHANCED_MESSAGE AND NOT WIN32)
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

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- rpp-test: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
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

project(rpp-test)

# Enable testing
enable_testing()
include(CTest)

# Per-test CTest TIMEOUT (seconds) for selected long-running suites; all other tests use DartConfiguration.tcl TimeOut.
# Increased timeout for: rpp_qa_tests_tensor_image_host_qa_mode_0, rpp_qa_tests_tensor_misc_host_test_type_1, rpp_qa_tests_tensor_image_hip_qa_mode_0
# Image qa_mode_0 CTests are omitted when USING_THE_ROCK is ON; misc test_type_1 timeouts apply whenever those tests are registered.
set(RPP_CTEST_LONG_TEST_TIMEOUT_SEC 4000 CACHE STRING "CTest TIMEOUT override in seconds for long RPP tests (see set_tests_properties below).")

# ROCm "TheRock" / sysdeps layout. When ON, CTest registers only golden QA (--qa_mode 1) for image/voxel to avoid
# OpenCV imwrite-heavy unit paths (qa_mode 0). Misc performance CTests are unchanged — Tensor_misc_* has no extra deps.
if(NOT DEFINED USING_THE_ROCK)
    set(USING_THE_ROCK OFF)
    if(DEFINED ROCM_PATH AND EXISTS "${ROCM_PATH}/lib/rocm_sysdeps/lib")
        set(USING_THE_ROCK ON)
    endif()
endif()
if(USING_THE_ROCK)
    message("-- ${White}${PROJECT_NAME}: USING_THE_ROCK — CTest image/voxel QA limited to --qa_mode 1.${ColourReset}")
endif()

#[[
List of high level dependency checks - for RPP QA tests in utilities/test_suite/CMakeLists.txt are below.
(The utilities/test_suite/HOST and utilities/test_suite/HIP folders are also designed to be built separately, so list of checks for those are present in the files "utilities/test_suite/HOST/CMakeLists.txt" and "utilities/test_suite/HIP/CMakeLists.txt")
- RPP (For ALL tests - to test any RPP functionalities)
- Python3 (For ALL tests - to run any RPP tests)
- Pandas (Optional - for pretty-printing consolidated rocprof CSV results; QA/unit drivers do not require it)
- OpenMP (For ALL tests - to measure wall-times, and parallelize input tensor-image PKD3->PLN3 conversions across threads)
- ROCm HIP (For ALL HIP tests - HIP to be installed and found to run all RPP GPU HIP backend API)
- rpp_AUDIO_AUGMENTATIONS_SUPPORT from find_package(rpp) (For AUDIO tests - to decide if RPP Audio tests script even needs to be built and executed)
- libsnd (For AUDIO tests - if rpp_AUDIO_AUGMENTATIONS_SUPPORT is enabled, to decide if RPP Audio tests script can be built and executed)
- NIFTI (For VOXEL tests - to decide if RPP Voxel QA tests script can be built and executed)
- OpenCV (Optional for IMAGE/VOXEL Tensor_* binaries — enables decoder_type 1, unit image dumps, voxel slice JPGs; golden --qa_mode 1 image QA uses packed .raw + .info when OpenCV is absent)
- USING_THE_ROCK (optional): when the ROCm sysdeps/TheRock layout is detected, only --qa_mode 1 CTests are registered for image/voxel. Misc CTests are unchanged.
]]
# Add find modules
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)

set(RPP_TEST_SCRIPT_DIR "${ROCM_PATH}/share/rpp/test")

# We need to handle the in-tree build of RPP separately since find_package(rpp) will not work.
if(NOT TARGET rpp)
    find_package(rpp 3.0.0 REQUIRED)
    message("-- ${White}${PROJECT_NAME}: rpp found with find_package(rpp REQUIRED)${ColourReset}")
else()
    message("-- ${White}${PROJECT_NAME}: rpp found with target rpp (in-tree build)${ColourReset}")
    message("-- ${Yellow}${PROJECT_NAME}: Please install RPP before running CTests${ColourReset}")
    set(rpp_AUDIO_AUGMENTATIONS_SUPPORT ${RPP_AUDIO_SUPPORT})
    set(rpp_BACKEND_TYPE ${BACKEND})
endif()

# Display RPP information
message("-- \t${White}rpp_VERSION -- ${rpp_VERSION}${ColourReset}")
message("-- \t${White}rpp_VERSION_MAJOR -- ${rpp_VERSION_MAJOR}${ColourReset}")
message("-- \t${White}rpp_VERSION_MINOR -- ${rpp_VERSION_MINOR}${ColourReset}")
message("-- \t${White}rpp_VERSION_PATCH -- ${rpp_VERSION_PATCH}${ColourReset}")
message("-- \t${White}rpp_BACKEND_TYPE -- ${rpp_BACKEND_TYPE}${ColourReset}")
message("-- \t${White}rpp_AUDIO_AUGMENTATIONS_SUPPORT -- ${rpp_AUDIO_AUGMENTATIONS_SUPPORT}${ColourReset}")

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

# find required python3-pip imports (only if Python3 is found)
if(Python3_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip show pandas
        RESULT_VARIABLE PANDAS_FOUND
        OUTPUT_QUIET
        ERROR_QUIET
    )
else()
    set(PANDAS_FOUND 1)  # Set to not-found value when Python3 is missing
endif()

# Add HOST/HIP subdirectories to build executables (OpenMP required, Python3 optional)
add_subdirectory(HOST)
if(rpp_BACKEND_TYPE STREQUAL "HIP")
    add_subdirectory(HIP)
endif()

# Minimal sanity CTests: Always run these basic tests to verify HOST and HIP backends
# These tests use the executables built by the subdirectories above
if(OpenMP_CXX_FOUND AND TARGET Tensor_image_host)
    # Create output directory for HOST test
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/HOST/output")

    # Minimal sanity test for HOST: brightness f32 test
    add_test(
        NAME rpp_sanity_test_brightness_host_f32
        COMMAND $<TARGET_FILE:Tensor_image_host>
            "${CMAKE_CURRENT_SOURCE_DIR}/TEST_IMAGES/three_images_mixed_src1"
            "${CMAKE_CURRENT_SOURCE_DIR}/TEST_IMAGES/three_images_mixed_src1"
            "${CMAKE_CURRENT_BINARY_DIR}/HOST/output"
            2 0 0 0 1 0 0 1 0 0 1 0 0 0 0
            "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/HOST"
    )
endif()

if(OpenMP_CXX_FOUND AND rpp_BACKEND_TYPE STREQUAL "HIP" AND TARGET Tensor_image_hip)
    # Create output directory for HIP test
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/HIP/output")

    # Minimal sanity test for HIP: brightness f32 test
    add_test(
        NAME rpp_sanity_test_brightness_hip_f32
        COMMAND $<TARGET_FILE:Tensor_image_hip>
            "${CMAKE_CURRENT_SOURCE_DIR}/TEST_IMAGES/three_images_mixed_src1"
            "${CMAKE_CURRENT_SOURCE_DIR}/TEST_IMAGES/three_images_mixed_src1"
            "${CMAKE_CURRENT_BINARY_DIR}/HIP/output"
            2 0 0 0 1 0 0 1 0 0 1 0 0 0 0
            "${CMAKE_CURRENT_SOURCE_DIR}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/HIP"
    )
endif()

# Top-level guards: Python3 and OpenMP are required to register Python-based CTests. Pandas is optional (see warning below).
if(NOT Python3_FOUND)
    message("-- ${Yellow}Warning: Python3 not found — skipping Python-script based CTests; only sanity tests are registered.${ColourReset}")
    return()
endif()
if(NOT PANDAS_FOUND EQUAL 0)
    message("-- ${Yellow}Warning: Pandas not found — rocprof consolidated reports cannot be printed from Python; Tensor misc QA (and other suites) can still run without it.${ColourReset}")
endif()
if(NOT OpenMP_CXX_FOUND)
    message("-- ${Yellow}Warning: Skipping all RPP-ctest test sets since OpenMP is not found!${ColourReset}")
    return()
endif()

# Helper to register an RPP QA test. `script_rel_path` is relative to RPP_TEST_SCRIPT_DIR.
# Extra arguments after `script_rel_path` are passed verbatim to the python script.
function(rpp_add_qa_test test_name script_rel_path)
    add_test(
        NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE} ${RPP_TEST_SCRIPT_DIR}/${script_rel_path} ${ARGN}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

message("-- ${White}${PROJECT_NAME}: Adding RPP HOST tests${ColourReset}")

# HOST test set 1 - Tensor-Image QA tests (packed .raw + .info for decoder_type 0; OpenCV optional for CTest — see qa_mode 0 below)
rpp_add_qa_test(rpp_qa_tests_tensor_image_host_all     HOST/runImageTests.py --qa_mode 1 --batch_size 3)
# qa_mode 0 exercises unit paths that dump outputs via OpenCV imwrite when QA flag is off in the binary.
if(OpenCV_FOUND AND NOT USING_THE_ROCK)
    rpp_add_qa_test(rpp_qa_tests_tensor_image_host_qa_mode_0 HOST/runImageTests.py --qa_mode 0 --batch_size 3)
    set_tests_properties(rpp_qa_tests_tensor_image_host_qa_mode_0 PROPERTIES TIMEOUT ${RPP_CTEST_LONG_TEST_TIMEOUT_SEC})
endif()

# HOST test set 2 - Tensor-Voxel QA tests (NIfTI required; OpenCV optional — qa_mode 0 non-QA visualization uses OpenCV JPGs)
if(NIFTI_FOUND)
    rpp_add_qa_test(rpp_qa_tests_tensor_voxel_host_all       HOST/runVoxelTests.py --qa_mode 1 --batch_size 3)
    if(OpenCV_FOUND AND NOT USING_THE_ROCK)
        rpp_add_qa_test(rpp_qa_tests_tensor_voxel_host_qa_mode_0 HOST/runVoxelTests.py --qa_mode 0 --batch_size 3)
    endif()
else()
    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 2 - rpp_qa_tests_tensor_voxel_host_all' since libniftiio is not found!${ColourReset}")
endif()

# HOST test set 3 - Tensor-Audio QA tests
if(rpp_AUDIO_AUGMENTATIONS_SUPPORT AND libsnd_LIBS)
    rpp_add_qa_test(rpp_qa_tests_tensor_audio_host_all HOST/runAudioTests.py --qa_mode 1 --batch_size 3)
else()
    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HOST test set 3 - rpp_qa_tests_tensor_audio_host_all' since RPP_AUDIO_AUGMENTATIONS_SUPPORT and/or libsnd is not found!${ColourReset}")
endif()

# HOST test set 4 - Tensor-Misc QA tests
rpp_add_qa_test(rpp_qa_tests_tensor_misc_host_all         HOST/runMiscTests.py --qa_mode 1 --batch_size 3)
# runMiscTests.py uses a different bit-depth matrix for test_type 1 (performance) than for test_type 0 (unit); only misc needs both.
rpp_add_qa_test(rpp_qa_tests_tensor_misc_host_test_type_1 HOST/runMiscTests.py --test_type 1 --batch_size 3)
set_tests_properties(rpp_qa_tests_tensor_misc_host_test_type_1 PROPERTIES TIMEOUT ${RPP_CTEST_LONG_TEST_TIMEOUT_SEC})
# Misc suite: Tensor_misc_* links only RPP + OpenMP (HIP adds HIP runtime). No OpenCV / NIFTI / libsnd. Pandas optional at runtime.
set_tests_properties(
    rpp_qa_tests_tensor_misc_host_all
    rpp_qa_tests_tensor_misc_host_test_type_1
    PROPERTIES LABELS "rpp_rocm_stack_only")

# HIP backend is opt-in; bail out before the HIP test sets if not selected or HIP isn't installed.
if(NOT rpp_BACKEND_TYPE STREQUAL "HIP")
    return()
endif()

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH ${ROCM_PATH} CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
find_package(HIP QUIET)
if(NOT HIP_FOUND)
    message("-- ${Yellow}Warning: Skipping all RPP HIP tests since HIP is not found!${ColourReset}")
    return()
endif()

message("-- ${White}${PROJECT_NAME}: Adding RPP HIP tests${ColourReset}")

# HIP test set 1 - Tensor-Image QA tests (packed .raw + .info for decoder_type 0)
rpp_add_qa_test(rpp_qa_tests_tensor_image_hip_all       HIP/runImageTests.py --qa_mode 1 --batch_size 3)
if(OpenCV_FOUND AND NOT USING_THE_ROCK)
    rpp_add_qa_test(rpp_qa_tests_tensor_image_hip_qa_mode_0 HIP/runImageTests.py --qa_mode 0 --batch_size 3)
    set_tests_properties(rpp_qa_tests_tensor_image_hip_qa_mode_0 PROPERTIES TIMEOUT ${RPP_CTEST_LONG_TEST_TIMEOUT_SEC})
endif()

# HIP test set 2 - Tensor-Voxel QA tests
if(NIFTI_FOUND)
    rpp_add_qa_test(rpp_qa_tests_tensor_voxel_hip_all       HIP/runVoxelTests.py --qa_mode 1 --batch_size 3)
    if(OpenCV_FOUND AND NOT USING_THE_ROCK)
        rpp_add_qa_test(rpp_qa_tests_tensor_voxel_hip_qa_mode_0 HIP/runVoxelTests.py --qa_mode 0 --batch_size 3)
    endif()
else()
    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 2 - rpp_qa_tests_tensor_voxel_hip_all' since libniftiio is not found!${ColourReset}")
endif()

# HIP test set 3 - Tensor-Audio QA tests
if(rpp_AUDIO_AUGMENTATIONS_SUPPORT AND libsnd_LIBS)
    rpp_add_qa_test(rpp_qa_tests_tensor_audio_hip_all HIP/runAudioTests.py --qa_mode 1 --batch_size 3)
else()
    message("-- ${Yellow}Warning: Skipping RPP-ctest 'HIP test set 3 - rpp_qa_tests_tensor_audio_hip_all' since RPP_AUDIO_AUGMENTATIONS_SUPPORT and/or libsnd is not found!${ColourReset}")
endif()

# HIP test set 4 - Tensor-Misc QA tests
rpp_add_qa_test(rpp_qa_tests_tensor_misc_hip_all         HIP/runMiscTests.py --qa_mode 1 --batch_size 3)
# runMiscTests.py uses a different bit-depth matrix for test_type 1 (performance) than for test_type 0 (unit); only misc needs both.
rpp_add_qa_test(rpp_qa_tests_tensor_misc_hip_test_type_1 HIP/runMiscTests.py --test_type 1 --batch_size 3)
set_tests_properties(rpp_qa_tests_tensor_misc_hip_test_type_1 PROPERTIES TIMEOUT ${RPP_CTEST_LONG_TEST_TIMEOUT_SEC})
set_tests_properties(
    rpp_qa_tests_tensor_misc_hip_all
    rpp_qa_tests_tensor_misc_hip_test_type_1
    PROPERTIES LABELS "rpp_rocm_stack_only")
