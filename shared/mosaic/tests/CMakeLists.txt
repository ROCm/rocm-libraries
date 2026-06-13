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
                         "${CMAKE_CURRENT_SOURCE_DIR}/test_real_kernels.cpp"
)

target_link_libraries(mosaic-tests PRIVATE roc::mosaic Catch2::Catch2WithMain)

# Inject the in-tree default weights path so the test can locate the .bin
# without an explicit MOSAIC_WEIGHTS env override, and the bundled real-kernel
# fixture (actual Tensile UserArgs.yaml params) for the ranking test.
#
# This must name a weights file that actually ships in backends/hipblaslt/data
# (see mosaic_index). The BB_BB (BBS_TN, bf16) model -- which the bf16 8192^3 TN
# test problem routes through -- is deployed as the file below; the per-library
# rename dropped the old "TensileLibrary_BB_BB_...mosaic.bin" name, so pointing at
# it left every weights-dependent assertion silently skipped.
target_compile_definitions(mosaic-tests PRIVATE
    MOSAIC_TEST_WEIGHTS="${CMAKE_CURRENT_SOURCE_DIR}/../backends/hipblaslt/data/gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs.mosaic.bin"
    MOSAIC_TEST_KERNELS="${CMAKE_CURRENT_SOURCE_DIR}/data/sample_kernels.tsv"
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
