################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

cmake_minimum_required(VERSION 3.13)

project(TensileHostLibraryTest)

set( CMAKE_CXX_STANDARD 17 )
set( CMAKE_CXX_EXTENSIONS OFF )

list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} /opt/rocm)

option(TENSILE_USE_HIP       "Use the Hip runtime." ON)
option(TENSILE_USE_LLVM      "Use LLVM for parsing config files." ON)
option(TENSILE_USE_MSGPACK   "Use msgpack for parsing config files." ON)
option(TENSILE_USE_OPENMP    "Enable OpenMP multithreading." ON)
option(TENSILE_STATIC_ONLY   "Disable exposing Tensile symbols in a shared library." ON)
option(TENSILE_DISABLE_CTEST "Disable CTest integration." ON)
option(TENSILE_ENABLE_COVERAGE "Enable code coverage for TensileHost library." OFF)
option(BUILD_CODE_COVERAGE "Enable code coverage for TensileHost library." OFF)

if((TENSILE_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE) AND (NOT CMAKE_BUILD_TYPE STREQUAL "Debug"))
    message(FATAL_ERROR "Code coverage requires CMAKE_BUILD_TYPE=Debug.")
endif()
if(TENSILE_USE_HIP)
    set(TENSILE_COMPONENTS HIP)
    if(TENSILE_USE_LLVM)
        set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} Client)
    endif()
endif()
if(TENSILE_USE_LLVM)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} LLVM)
endif()
if(TENSILE_USE_OPENMP)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} OpenMP)
endif()
if(TENSILE_STATIC_ONLY)
    set(TENSILE_COMPONENTS ${TENSILE_COMPONENTS} STATIC_ONLY)
endif()

if(NOT Tensile_FOUND)
    find_package(Tensile 4.44.0 EXACT REQUIRED ${TENSILE_COMPONENTS} PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../Tensile")
endif()

if(NOT TENSILE_DISABLE_CTEST)
    enable_testing()

    if(CMAKE_VERSION VERSION_GREATER 3.9)
        include(GoogleTest)
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." ON)
    else()
        option(GTEST_INTEGRATION "Enable Googletest integration with CMake." OFF)
    endif()
endif()

set(INSTALL_GTEST OFF CACHE BOOL "Install GTest")
add_subdirectory(googletest)
if(NOT gtest_SOURCE_DIR)
    message(FATAL_ERROR "googletest not found.  Run git submodule update --init")
endif()

find_package(Boost COMPONENTS filesystem)

#if(TENSILE_USE_HIP AND NOT HIP_DIR)
if(TENSILE_USE_HIP)
    find_package(HIP REQUIRED CONFIG PATHS $ENV{ROCM_PATH} /opt/rocm)
endif()

if(TENSILE_USE_OPENMP AND NOT TARGET custom_openmp_cxx)

    # Workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/21787
    # ensures we link to HIP's libomp and get an rpath to it.
    add_library(custom_openmp_cxx INTERFACE)

    if(TENSILE_USE_HIP)
        target_compile_options(custom_openmp_cxx INTERFACE "-fopenmp")
        target_link_options(custom_openmp_cxx INTERFACE "-fopenmp")
    else ()
        find_package(OpenMP REQUIRED)
        target_link_libraries(custom_openmp_cxx INTERFACE OpenMP::OpenMP_CXX)
    endif ()
endif()


add_subdirectory(configs)

set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS ${FILE})
        message(WARNING "Overwriting ${FILE} from ${FILE}.gz")
    endif()
    execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach(FILE)

include_directories("." ${GTEST_SOURCE_DIR}/include)
add_subdirectory(testlib)

set(test_sources ${test_sources}
    test.cpp
    CachingLibrary_test.cpp
    ContractionProblem_test.cpp
    ContractionSelectionLibrary_test.cpp
    ContractionFitness_test.cpp
    MultipleSolutionsPerSize_test.cpp
    DataTypes_test.cpp
    EmbeddedData_test.cpp
    KernelArguments_test.cpp
    Predicates_test.cpp
    ProjectedPerformance_test.cpp
    DecisionTree_test.cpp
    TensorDescriptor_test.cpp
    TestData_test.cpp
    Utils_test.cpp
)

if(TENSILE_USE_LLVM)
    find_package(LLVM REQUIRED CONFIG)

    set(test_sources ${test_sources}
        ContractionLibraryLoading_test.cpp
        ContractionFitness_test.cpp
        MultipleSolutionsPerSize_test.cpp
        llvm/ArithmeticUnitPredicate_test.cpp
        llvm/CUEfficiencyPredicate_test.cpp
        llvm/DeterministicModePredicate_test.cpp
        llvm/KernelLanguagePredicate_test.cpp
        llvm/LLVMYAMLContraction_test.cpp
        llvm/LibraryPerformance_test.cpp
        )
endif()

if(Client IN_LIST TENSILE_COMPONENTS)
    set(test_sources ${test_sources}
        client/DataInitialization_test.cpp)
endif()

if(TENSILE_USE_HIP)
    add_subdirectory(hip)
endif()

add_executable(TensileTests ${test_sources})

if(NOT TENSILE_DISABLE_CTEST)
    if(GTEST_INTEGRATION)
        gtest_discover_tests(TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)
    else()
        add_test(NAME TensileHostLibraryTests COMMAND TensileTests WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endif()

target_link_libraries(TensileTests PUBLIC gtest TensileHost TensileTestLib ${Boost_LIBRARIES})

if(TENSILE_USE_LLVM)
    find_library(LLVMObjectYAML_LIBRARY
        NAMES LLVMObjectYAML
        PATHS ${LLVM_LIBRARY_DIR})
    message("LLVMObjectYAML_LIBRARY: ${LLVMObjectYAML_LIBRARY}")
    #Use LLVM shared libs if LLVMObjectYAML static lib cannot be found
    if(LLVMObjectYAML_LIBRARY)
        target_link_libraries(TensileTests PUBLIC LLVMObjectYAML )
    else()
        target_link_libraries(TensileTests PUBLIC LLVM )
    endif()
    target_include_directories(TensileTests PUBLIC ${LLVM_INCLUDE_DIRS})
endif()

if(TENSILE_USE_HIP)

    target_link_libraries(TensileTests PRIVATE "-Xlinker --whole-archive")
    target_link_libraries(TensileTests PUBLIC ${HIP_TEST_LIBRARIES})
    target_link_libraries(TensileTests PRIVATE "-Xlinker --no-whole-archive")
    foreach(arch IN LISTS GPU_TARGETS)
        target_link_libraries(TensileTests PRIVATE "--offload-arch=${arch}")
    endforeach(arch)
endif()

if(TENSILE_USE_OPENMP)
    target_link_libraries(TensileTests PRIVATE custom_openmp_cxx)
endif()

if(TENSILE_ENABLE_COVERAGE OR BUILD_CODE_COVERAGE)
    message(STATUS "Code coverage for TensileHost is ENABLED.")

    target_compile_options(TensileHost PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(TensileHost PUBLIC -fprofile-instr-generate)
    message(STATUS "Applied coverage flags to TensileHost.")

    set(coverage_dir "${CMAKE_CURRENT_BINARY_DIR}/coverage-report")

    add_custom_target(
        code_cov_tests
        DEPENDS TensileTests
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${coverage_dir}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${coverage_dir}/profraw"
        COMMAND ${CMAKE_COMMAND} -E env "LLVM_PROFILE_FILE=${coverage_dir}/profraw/tensile-host-coverage_%m.profraw"
                                        GTEST_LISTENER=NO_PASS_LINE_IN_LOG
                                        $<TARGET_FILE:TensileTests>
                                        "--gtest_filter=-*Extended*:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/504:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/505:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/506:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/507:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/508:HipSolutionAdapter/RunGEMMKernelTestHip.TestAllSolutions/509"
      )

    find_program(
        LLVM_PROFDATA
        llvm-profdata
        PATH_SUFFIXES llvm/bin
        REQUIRED
    )
    find_program(
        LLVM_COV
        llvm-cov
        REQUIRED
        PATH_SUFFIXES llvm/bin
    )

    add_custom_target(
        coverage
        DEPENDS code_cov_tests
        COMMAND ${LLVM_PROFDATA} merge -sparse "${coverage_dir}/profraw/tensile-host-coverage_*.profraw" -o "${coverage_dir}/tensile-host.profdata"
        COMMAND ${LLVM_COV} report -object $<TARGET_FILE:TensileHost> -instr-profile="${coverage_dir}/tensile-host.profdata"
        COMMAND ${LLVM_COV} show -object $<TARGET_FILE:TensileHost> -instr-profile="${coverage_dir}/tensile-host.profdata" -format=html -output-dir="${coverage_dir}"

        COMMENT "Generating TensileHost coverage report... HTML report in ${coverage_dir}/index.html"
    )

endif()
