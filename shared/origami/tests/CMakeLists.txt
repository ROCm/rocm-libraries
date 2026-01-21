# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(Catch2 3 QUIET CONFIG)
if(NOT Catch2_FOUND)
    if(ORIGAMI_ENABLE_FETCH)
        include(FetchContent)
        fetchcontent_declare(
            Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG devel
        )
        fetchcontent_makeavailable(Catch2)
    else()
        message(FATAL_ERROR "Failed to find Catch2")
    endif()
endif()

add_executable(origami-tests)

target_sources(
    origami-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/test_gemm.cpp"
                          "${CMAKE_CURRENT_SOURCE_DIR}/test_origami.cpp"
)

target_include_directories(origami-tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(origami-tests PRIVATE roc::origami Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(origami-tests
    DISCOVERY_MODE PRE_TEST
)

# Install CTestTestfile.cmake for installed Origami tests
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install"
    "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
    "# SPDX-License-Identifier: MIT\n"
    "\n"
    "# C++ tests\n"
    "add_test(origami-tests \"../origami-tests\")\n"
    "\n"
)

if(ORIGAMI_ENABLE_PYTHON)
    find_package(Python COMPONENTS Interpreter REQUIRED)
    
    set(ORIGAMI_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set(ORIGAMI_INSTALL_PYTHON_BINDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/origami")
    
    file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install"
        "# Python tests\n"
        "add_test(origami_python_test \"${Python_EXECUTABLE}\" \"origami_test.py\")\n"
        "add_test(origami_python_grid_test \"${Python_EXECUTABLE}\" \"origami_grid_test.py\")\n"
        "\n"
        "set_tests_properties(origami_python_test origami_python_grid_test PROPERTIES\n"
        "    ENVIRONMENT \"LD_LIBRARY_PATH=${ORIGAMI_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH};PYTHONPATH=${ORIGAMI_INSTALL_PYTHON_BINDIR}:$ENV{PYTHONPATH}\"\n"
        ")\n"
    )
    
    rocm_install(
        FILES 
            "${CMAKE_CURRENT_SOURCE_DIR}/../python/origami_test.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/../python/origami_grid_test.py"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/origami"
        COMPONENT tests
    )
endif()

rocm_install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.cmake.install"
    DESTINATION "${CMAKE_INSTALL_BINDIR}/origami"
    COMPONENT tests
    RENAME CTestTestfile.cmake
)
