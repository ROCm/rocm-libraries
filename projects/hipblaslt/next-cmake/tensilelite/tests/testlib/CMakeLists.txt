# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(TESTLIB_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tensilelite/HostLibraryTests/testlib")

target_include_directories(tensilelite-tests
    PRIVATE
        "${TESTLIB_SRC_DIR}/include"
)
target_sources(tensilelite-tests
    PRIVATE
        "${TESTLIB_SRC_DIR}/source/TestData.cpp"
)

target_compile_definitions(tensilelite-tests
    PRIVATE
        BOOST_ERROR_CODE_HEADER_ONLY
)

target_link_libraries(tensilelite-tests
    PUBLIC
        tensilelite::tensilelite-host
        Boost::filesystem
)
