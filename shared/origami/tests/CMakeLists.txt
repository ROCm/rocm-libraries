# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

find_package(GTest REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem)

add_executable(origami-tests)

target_sources(origami-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/origami_gtest.cpp"
)

target_include_directories(origami-tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/origami_gtest.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/origami_gtest.yaml"
    COPYONLY
)

target_link_libraries(origami-tests
    PRIVATE
        roc::origami
        Boost::filesystem
        GTest::gtest
        GTest::gtest_main
)

gtest_discover_tests(origami-tests 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} 
    TIMEOUT 60
)
