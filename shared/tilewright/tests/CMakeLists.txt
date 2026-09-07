# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# C++ test suite for the standalone tilewright library (Catch2 v3).

find_package(Catch2 3 QUIET CONFIG)
if(NOT Catch2_FOUND)
    if(TILEWRIGHT_ENABLE_FETCH)
        include(FetchContent)
        FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.5.2
        )
        FetchContent_MakeAvailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2 (set TILEWRIGHT_ENABLE_FETCH=ON to auto-fetch)")
    endif()
endif()

add_executable(tilewright-tests)

target_sources(
    tilewright-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test_model.cpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/test_real_kernels.cpp"
)

target_link_libraries(tilewright-tests PRIVATE roc::tilewright Catch2::Catch2WithMain)

# Inject the in-tree default weights path so the test can locate the .bin
# without an explicit TILEWRIGHT_WEIGHTS env override, and the bundled real-kernel
# fixture (actual Tensile UserArgs.yaml params) for the ranking test.
target_compile_definitions(tilewright-tests PRIVATE
    TILEWRIGHT_TEST_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/../backends/hipblaslt/weights/gfx950/gfx950/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.tilewright.bin"
    TILEWRIGHT_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
)

include(CTest)
include(Catch)
catch_discover_tests(tilewright-tests
    DISCOVERY_MODE PRE_TEST
)

# Install the C++ test target when installation is configured.
if(TILEWRIGHT_ENABLE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS tilewright-tests
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        COMPONENT tests
    )
endif()
