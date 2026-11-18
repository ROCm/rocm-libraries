# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# ==============================================================================
# GoogleTest for the clients
# ==============================================================================
# Provides GTest::gtest. An installed GoogleTest is used if found. Otherwise
# GoogleTest is built from FETCHCONTENT_SOURCE_DIR_GOOGLETEST, or downloaded
# if ALLOW_FETCH_DEPS is ON.
# ==============================================================================

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    if(IS_DIRECTORY "${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
        message(STATUS "GoogleTest not found. Building it from ${FETCHCONTENT_SOURCE_DIR_GOOGLETEST}")
    elseif(ALLOW_FETCH_DEPS)
        message(STATUS "GoogleTest not found. Fetching googletest because ALLOW_FETCH_DEPS is ON")
    else()
        message(FATAL_ERROR "GoogleTest not found. Install it, for example the libgtest-dev or gtest-devel package, "
                            "and point CMAKE_PREFIX_PATH at its install prefix if needed, set "
                            "FETCHCONTENT_SOURCE_DIR_GOOGLETEST to a googletest source tree, or configure with "
                            "-DALLOW_FETCH_DEPS=ON to download it.")
    endif()

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG b514bdc898e2951020cbdca1304b75f5950d1f59 # v1.15.2
    )

    # The installed clients must not depend on a libgtest that no package ships
    set(HIPSPARSE_SAVED_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    set(INSTALL_GTEST OFF)
    set(BUILD_GMOCK OFF)
    set(gtest_force_shared_crt ON)

    if(COMMAND rocm_check_toolchain_var)
        # No-op ROCMChecks callback while googletest sets compiler flag variables, which ROCMChecks reports, or
        # rejects with ROCM_ERROR_TOOLCHAIN_VAR
        function(rocm_check_toolchain_var)
        endfunction()
    endif()
    FetchContent_MakeAvailable(googletest)
    if(COMMAND rocm_check_toolchain_var)
        # Re-including ROCMChecks restores the callback; with these OFF it does not register its watches a second
        # time, and its ROCMCHECKS_* environment overrides would turn them back on
        set(HIPSPARSE_SAVED_WARN_TOOLCHAIN_VAR ${ROCM_WARN_TOOLCHAIN_VAR})
        set(HIPSPARSE_SAVED_ERROR_TOOLCHAIN_VAR ${ROCM_ERROR_TOOLCHAIN_VAR})
        set(ROCM_WARN_TOOLCHAIN_VAR OFF)
        set(ROCM_ERROR_TOOLCHAIN_VAR OFF)
        foreach(rocmchecks_option ROCMCHECKS_WARN_TOOLCHAIN_VAR ROCMCHECKS_ERROR_TOOLCHAIN_VAR)
            if(DEFINED ENV{${rocmchecks_option}})
                set(HIPSPARSE_SAVED_${rocmchecks_option} "$ENV{${rocmchecks_option}}")
                unset(ENV{${rocmchecks_option}})
            endif()
        endforeach()
        include(ROCMChecks)
        foreach(rocmchecks_option ROCMCHECKS_WARN_TOOLCHAIN_VAR ROCMCHECKS_ERROR_TOOLCHAIN_VAR)
            if(DEFINED HIPSPARSE_SAVED_${rocmchecks_option})
                set(ENV{${rocmchecks_option}} "${HIPSPARSE_SAVED_${rocmchecks_option}}")
            endif()
        endforeach()
        set(ROCM_WARN_TOOLCHAIN_VAR ${HIPSPARSE_SAVED_WARN_TOOLCHAIN_VAR})
        set(ROCM_ERROR_TOOLCHAIN_VAR ${HIPSPARSE_SAVED_ERROR_TOOLCHAIN_VAR})
    endif()
    set(BUILD_SHARED_LIBS ${HIPSPARSE_SAVED_BUILD_SHARED_LIBS})

    if(NOT TARGET GTest::gtest)
        message(FATAL_ERROR "${googletest_SOURCE_DIR} does not provide the GTest::gtest target. Set "
                            "FETCHCONTENT_SOURCE_DIR_GOOGLETEST to the top level of a googletest source tree.")
    endif()

    set_property(DIRECTORY ${googletest_SOURCE_DIR} PROPERTY EXCLUDE_FROM_ALL ON)
    get_target_property(GTEST_INCLUDE_DIRS gtest INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}")
endif()
