# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25.0)
project(hostlibrarytests LANGUAGES CXX)

set(HOST_LIB_TESTS_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../tensilelite/HostLibraryTests")

find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(tensilelite-tests)
add_subdirectory(configs)

# from old cmakelists.txt(did not know how to modernize it)
set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

foreach(FILE ${SOLUTION_LIBRARY_FILES})
    if(EXISTS "${FILE}.gz")
        execute_process(COMMAND gzip --keep --decompress --force "${FILE}.gz")
    endif()
    file(COPY ${FILE} DESTINATION "${TEST_DATA_DIR}")
endforeach()

add_subdirectory(testlib)

target_sources(tensilelite-tests
    PRIVATE
        "${HOST_LIB_TESTS_SRC_DIR}/test.cpp"
        "${HOST_LIB_TESTS_SRC_DIR}/ContractionSelectionLibrary_test.cpp"
        "${HOST_LIB_TESTS_SRC_DIR}/DataTypes_test.cpp"
        "${HOST_LIB_TESTS_SRC_DIR}/Predicates_test.cpp"
)

if(HIPBLASLT_ENABLE_LLVM)
    target_sources(tensilelite-tests
        PRIVATE
            "${HOST_LIB_TESTS_SRC_DIR}/llvm/LibraryPerformance_test.cpp"
    )
    target_link_libraries(tensilelite-tests PRIVATE LLVM)
endif()

target_link_libraries(tensilelite-tests
    PUBLIC
        tensilelite::tensilelite-host
        Boost::filesystem
        roc::hipblaslt
)

gtest_discover_tests(tensilelite-tests WORKING_DIRECTORY ${CMAKE_BINARY_DIR} TIMEOUT 60)

target_link_libraries(tensilelite-tests PUBLIC GTest::gtest)

if(HIPBLASLT_ENABLE_OPENMP)
    target_link_libraries(tensilelite-tests PRIVATE OpenMP::OpenMP_CXX)
endif()
