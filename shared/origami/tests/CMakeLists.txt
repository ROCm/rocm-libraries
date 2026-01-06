################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

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
catch_discover_tests(origami-tests)
