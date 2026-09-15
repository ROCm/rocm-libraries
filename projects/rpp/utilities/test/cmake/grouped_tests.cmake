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

# Registers one CTest test per GTest suite (i.e. per RPP operator) instead of one per case.
#
# gtest_discover_tests() registers every case separately, which for this suite is ~25000 CTest
# tests, each paying a fresh process start plus RPP/HIP handle setup. That overhead dominates:
# ~0.33s per case of spawn cost against a whole 8000-case category that runs in 18s in one
# process. Grouping by suite keeps a named CTest test per operator -- so a failure still points at
# an operator without reading any logs -- while cutting the process count by ~250x.
#
# Discovery mirrors gtest_discover_tests' POST_BUILD mode: the built binary is asked for its suite
# list and a CTest include file is generated from it, so the registration cannot drift as
# operators are added or renamed.

set(RPP_GROUPED_TESTS_IMPL "${CMAKE_CURRENT_LIST_DIR}/grouped_tests_impl.cmake"
    CACHE INTERNAL "Script that turns --gtest_list_tests output into add_test() calls")

# rpp_discover_grouped_tests(<target> [TIMEOUT <seconds>] [WORKING_DIRECTORY <dir>])
function(rpp_discover_grouped_tests TARGET)
    cmake_parse_arguments(_arg "" "TIMEOUT;WORKING_DIRECTORY" "" ${ARGN})

    if(NOT _arg_TIMEOUT)
        # Suites run 0.5-3s each here; the margin is for slow or contended machines.
        set(_arg_TIMEOUT 900)
    endif()
    if(NOT _arg_WORKING_DIRECTORY)
        set(_arg_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    set(_testsFile "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_grouped_tests.cmake")
    set(_includeFile "${CMAKE_CURRENT_BINARY_DIR}/${TARGET}_grouped_include.cmake")

    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        BYPRODUCTS "${_testsFile}"
        COMMAND ${CMAKE_COMMAND}
                -D "TEST_EXECUTABLE=$<TARGET_FILE:${TARGET}>"
                -D "TEST_OUTPUT_FILE=${_testsFile}"
                -D "TEST_WORKING_DIRECTORY=${_arg_WORKING_DIRECTORY}"
                -D "TEST_TIMEOUT=${_arg_TIMEOUT}"
                -P "${RPP_GROUPED_TESTS_IMPL}"
        VERBATIM
        COMMENT "Discovering GTest suites in ${TARGET}")

    # Indirection so that configuring before the first build still yields a CTest file. The
    # placeholder test fails loudly rather than reporting an empty, all-green run.
    file(WRITE "${_includeFile}"
        "if(EXISTS \"${_testsFile}\")\n"
        "  include(\"${_testsFile}\")\n"
        "else()\n"
        "  add_test(${TARGET}_NOT_BUILT ${TARGET}_NOT_BUILT)\n"
        "endif()\n")

    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_includeFile}")
endfunction()
