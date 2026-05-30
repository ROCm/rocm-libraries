################################################################################
#
# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

set(__default_cxx_compile_options
    -Wall
    -Wextra
    -Wcomment
    -Wendif-labels
    -Wformat
    -Winit-self
    -Wreturn-type
    -Wsequence-point
    -Wswitch
    -Wtrigraphs
    -Wundef
    -Wuninitialized
    -Wunreachable-code
    -Wno-ignored-qualifiers
    -Wno-sign-compare
)

set(__clang_cxx_compile_options
    -Weverything
    -Wno-c++98-compat
    -Wno-c++98-compat-pedantic
    -Wno-double-promotion
    -Wno-exit-time-destructors
    -Wno-extra-semi
    -Wno-extra-semi-stmt
    -Wno-missing-prototypes
    -Wno-padded
    -Wno-unused-command-line-argument
    -Wno-weak-vtables
    -Wno-covered-switch-default
    -Wno-unsafe-buffer-usage
    -Wno-deprecated-declarations
    -Wno-global-constructors
    -Wno-reserved-identifier
    -Wno-deprecated
    -Wno-old-style-cast
    -Wno-c++11-narrowing
    -Wno-switch-enum
    -Wno-suggest-override
    -Wno-nonportable-system-include-path
    -Wno-documentation
    -Wmissing-noreturn)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19")
    list(APPEND __clang_cxx_compile_options
        -Wno-switch-default)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "23")
    list(APPEND __clang_cxx_compile_options
        -Wno-lifetime-safety
        -Wno-lifetime-safety-suggestions
        -Wno-lifetime-safety-intra-tu-suggestions
        -Wno-lifetime-safety-cross-tu-suggestions)
endif()

if(WIN32)
    list(APPEND __clang_cxx_compile_options
        -Wno-ignored-attributes
        -Wno-language-extension-token
        -fms-extensions
        -fms-compatibility)
endif()

add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:${__default_cxx_compile_options};${__clang_cxx_compile_options}>"
)

unset(__default_cxx_compile_options)
unset(__clang_cxx_compile_options)
