#[[
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

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

cmake_minimum_required(VERSION 3.14)

project(rpp-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm path (only used to locate the installed rpp package by default)
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "ROCm installation path")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "ROCm installation path")
endif()
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)

# Standalone build finds an installed rpp; in-tree build reuses the existing target.
if(NOT TARGET rpp)
    find_package(rpp 3.0.0 REQUIRED)
    set(RPP_TEST_LINK_LIB rpp::rpp)
else()
    set(RPP_TEST_LINK_LIB rpp)

    # The rpp target exports flat api/ dirs, but the suite includes <rpp/...> like an installed
    # consumer. Stage an installed-style layout so a stale ${ROCM_PATH}/include/rpp is never used.
    set(RPP_TEST_STAGED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
    file(GLOB RPP_TEST_API_HEADERS CONFIGURE_DEPENDS
        "${rpp_SOURCE_DIR}/api/*.h"
        "${rpp_BINARY_DIR}/api/*.h")
    foreach(header IN LISTS RPP_TEST_API_HEADERS)
        get_filename_component(header_name "${header}" NAME)
        configure_file("${header}" "${RPP_TEST_STAGED_INCLUDE_DIR}/rpp/${header_name}" COPYONLY)
    endforeach()
endif()

include(cmake/gtest.cmake)
include(cmake/grouped_tests.cmake)

enable_testing()
include(GoogleTest)
include(CTest)

file(GLOB_RECURSE RPP_TEST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tests/*.cpp)

if(NOT RPP_TEST_SOURCES)
    message(FATAL_ERROR "No test sources found under ${CMAKE_CURRENT_SOURCE_DIR}/src/tests")
endif()

add_executable(rpp_tests ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${RPP_TEST_SOURCES})
target_include_directories(rpp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(RPP_TEST_STAGED_INCLUDE_DIR)
    target_include_directories(rpp_tests BEFORE PRIVATE ${RPP_TEST_STAGED_INCLUDE_DIR})
endif()

# Warnings are errors for the suite's own sources.
target_compile_options(rpp_tests PRIVATE -Wall -Wextra -Werror)

# src/main.cpp provides main() (it installs the suite's own reporter), so gtest_main is not used.
target_link_libraries(rpp_tests PRIVATE ${RPP_TEST_LINK_LIB} GTest::gtest)


# librpp depends on ROCm/LLVM libomp. Require that OpenMP CONFIG package (no
# MODULE/libgomp fallback). Link OpenMP::OpenMP_CXX so rpp_tests gets a direct
# DT_NEEDED on libomp.so; BUILD_RPATH must come from openmp_LIB_INSTALL_DIR so
# POST_BUILD --gtest_list_tests can load it in the build tree.
if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(OpenMP CONFIG REQUIRED COMPONENTS CXX
    HINTS
      "${ROCM_PATH}/lib/llvm"
      "${ROCM_PATH}/llvm"
    PATHS
      "${ROCM_PATH}/lib/llvm/lib/cmake/openmp"
  )
endif()

# Bare -fopenmp selects libomp on Clang but also pulls -lgomp on g++. For
# non-Clang, drop it from OpenMP::OpenMP_CXX. Today the flag still lives on
# OpenMP::omp and propagates through INTERFACE_LINK_LIBRARIES -- strip that too
# while the target exists; safe to omit once omp is folded into OpenMP_CXX.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(_rpp_omp_targets OpenMP::OpenMP_CXX)
  if(TARGET OpenMP::omp)
    list(APPEND _rpp_omp_targets OpenMP::omp)
  endif()
  foreach(_rpp_omp_tgt IN LISTS _rpp_omp_targets)
    get_property(_rpp_omp_link_opts TARGET ${_rpp_omp_tgt} PROPERTY INTERFACE_LINK_OPTIONS)
    if(_rpp_omp_link_opts)
      list(REMOVE_ITEM _rpp_omp_link_opts "-fopenmp")
      set_property(TARGET ${_rpp_omp_tgt} PROPERTY INTERFACE_LINK_OPTIONS "${_rpp_omp_link_opts}")
    endif()
    get_property(_rpp_omp_comp_opts TARGET ${_rpp_omp_tgt} PROPERTY INTERFACE_COMPILE_OPTIONS)
    if(_rpp_omp_comp_opts)
      list(REMOVE_ITEM _rpp_omp_comp_opts "-fopenmp")
      set_property(TARGET ${_rpp_omp_tgt} PROPERTY INTERFACE_COMPILE_OPTIONS "${_rpp_omp_comp_opts}")
    endif()
  endforeach()
endif()

target_link_libraries(rpp_tests PRIVATE OpenMP::OpenMP_CXX)
if(NOT openmp_LIB_INSTALL_DIR)
  message(FATAL_ERROR
    "OpenMP CONFIG did not set openmp_LIB_INSTALL_DIR; cannot set rpp_tests BUILD_RPATH for libomp")
endif()
set_property(TARGET rpp_tests APPEND PROPERTY BUILD_RPATH "${openmp_LIB_INSTALL_DIR}")

# One CTest test per GTest suite (per operator) by default; see cmake/grouped_tests.cmake.
# Per-case registration isolates each case in its own process, at the cost of one start per case.
option(RPP_TEST_CTEST_PER_CASE "Register one CTest test per GTest case (slow: one process each)" OFF)

if(RPP_TEST_CTEST_PER_CASE)
    gtest_discover_tests(rpp_tests)
else()
    rpp_discover_grouped_tests(rpp_tests)
endif()

# Installed next to the ROCm libraries, so librpp and the HIP runtime resolve from ../lib.
include(GNUInstallDirs)
if(APPLE)
    set(RPP_TEST_RPATH_ORIGIN "@loader_path")
else()
    set(RPP_TEST_RPATH_ORIGIN "$ORIGIN")
endif()
set_target_properties(rpp_tests PROPERTIES INSTALL_RPATH "${RPP_TEST_RPATH_ORIGIN}/../${CMAKE_INSTALL_LIBDIR}")
install(TARGETS rpp_tests RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT test)

# Tier labels (quick/standard/...) for TheRock.
if(DEFINED ROCM_LIBRARIES_ROOT AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
    include(${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake)

    set(RPP_TEST_INSTALL_CTEST_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
    file(WRITE "${RPP_TEST_INSTALL_CTEST_FILE}" "")
    apply_test_category_labels(rpp_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${RPP_TEST_INSTALL_CTEST_FILE}")
    # Emulator tiers select HIP cases only, which a CPU build does not generate.
    if("${BACKEND}" STREQUAL "HIP")
        apply_test_category_labels(rpp_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/test_categories_ffm.yaml"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${RPP_TEST_INSTALL_CTEST_FILE}")
    endif()
    install(FILES "${RPP_TEST_INSTALL_CTEST_FILE}"
        DESTINATION ${CMAKE_INSTALL_BINDIR}/rpp
        COMPONENT test
        RENAME CTestTestfile.cmake)
endif()
