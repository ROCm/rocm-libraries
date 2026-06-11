# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# C++ test suite for the standalone mosaic library (Catch2 v3).

find_package(Catch2 3 QUIET CONFIG)
if(NOT Catch2_FOUND)
    if(MOSAIC_ENABLE_FETCH)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.5.2
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2 (set MOSAIC_ENABLE_FETCH=ON to auto-fetch)")
    endif()
endif()

add_executable(mosaic-tests)

target_sources(
    mosaic-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test_model.cpp"
)

target_link_libraries(mosaic-tests PRIVATE roc::mosaic Catch2::Catch2WithMain)

# Inject the in-tree default weights path so the test can locate the .bin
# without an explicit MOSAIC_WEIGHTS env override.
target_compile_definitions(mosaic-tests PRIVATE
    MOSAIC_TEST_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/../backends/hipblaslt/data/mosaic_weights.bin"
)

include(CTest)
include(Catch)
catch_discover_tests(mosaic-tests
    DISCOVERY_MODE PRE_TEST
)

# Install the C++ test target when installation is configured.
if(MOSAIC_ENABLE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS mosaic-tests
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT tests
    )
endif()
