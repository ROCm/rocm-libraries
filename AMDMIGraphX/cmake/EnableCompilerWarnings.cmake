#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
# - Enable warning all for gcc/clang or use /W4 for visual studio

include(CheckCXXCompilerFlag)

## Strict warning level
if (MSVC)
    # Use the highest warning level for visual studio.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")
    # set(CMAKE_CXX_WARNING_LEVEL 4)
    # if (CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    #     string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # else ()
    #     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    # endif ()

    # set(CMAKE_C_WARNING_LEVEL 4)
    # if (CMAKE_C_FLAGS MATCHES "/W[0-4]")
    #     string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    # else ()
    #     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    # endif ()

else()
    foreach(COMPILER C CXX)
        set(CMAKE_COMPILER_WARNINGS)
        # use -Wall for gcc and clang
        list(APPEND CMAKE_COMPILER_WARNINGS 
            -Wall
            -Wextra
            -Wcomment
            -Wendif-labels
            -Wformat
            -Winit-self
            -Wreturn-type
            -Wsequence-point
            # Shadow is broken on gcc when using lambdas
            # -Wshadow
            -Wswitch
            -Wtrigraphs
            -Wundef
            -Wuninitialized
            -Wunreachable-code
            -Wunused

            -Wno-sign-compare
        )
        # Flags for gcc 7
        if(CMAKE_${COMPILER}_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "7.0")
                list(APPEND CMAKE_COMPILER_WARNINGS 
                -Wduplicated-branches
                -Wduplicated-cond
                -Wno-noexcept-type
                -Wodr
                -Wshift-negative-value
                -Wshift-overflow=2
            )
            endif()
        endif()
        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Weverything
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-conversion
                -Wno-double-promotion
                -Wno-exit-time-destructors
                -Wno-extra-semi
                -Wno-extra-semi-stmt
                -Wno-float-conversion
                -Wno-gnu-anonymous-struct
                -Wno-gnu-zero-variadic-macro-arguments
                -Wno-missing-prototypes
                -Wno-nested-anon-types
                -Wno-option-ignored
                -Wno-padded
                -Wno-shorten-64-to-32
                -Wno-sign-conversion
                -Wno-unused-command-line-argument
                -Wno-weak-vtables
                -Wno-c99-extensions
                -Wno-unsafe-buffer-usage
                # This is broken for now for moved values
                -Wno-shadow-uncaptured-local
                # -Wno-c++2a-designator
            )
            if(WIN32 AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "19")
                list(APPEND CMAKE_COMPILER_WARNINGS
                    -Wno-missing-include-dirs
                    -Wno-switch-default
                    -Wno-deprecated-pragma
                )
            endif()
        else()
            list(APPEND CMAKE_COMPILER_WARNINGS
                -Wno-missing-field-initializers
                -Wno-maybe-uninitialized
                # -Wno-deprecated-declarations
            )
        endif()
        foreach(COMPILER_WARNING ${CMAKE_COMPILER_WARNINGS})
            string(MAKE_C_IDENTIFIER "HAS_${COMPILER}_FLAG${COMPILER_WARNING}" HAS_COMPILER_WARNING)
            check_cxx_compiler_flag(${COMPILER_WARNING} ${HAS_COMPILER_WARNING})
            if(${HAS_COMPILER_WARNING})
                add_compile_options($<$<COMPILE_LANGUAGE:${COMPILER}>:${COMPILER_WARNING}>)
            endif()
        endforeach()
    endforeach()
endif ()
