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

set(__cxx_compile_options
    -Werror
    -Wall
    -Wextra
    # Additional warnings not included in -Wall/-Wextra
    -Wundef
    -Wunreachable-code
    -Wmissing-noreturn
    -Wshadow
    -Wsuggest-override
    -Wold-style-cast
    -Wcast-align
    -Wcast-qual
    -Wformat-nonliteral
    -Wunknown-attributes
    -Wunknown-warning-option
    -Wdeprecated
    -Wdeprecated-builtins
    -Wzero-as-null-pointer-constant
    # TODO: Working to enable these warnings. Each requires code cleanup first.
    # -Wconversion            # ~1000+ implicit narrowing/sign conversions to fix
    # -Wdouble-promotion      # implicit float-to-double promotions
    # Suppress specific warnings -- working to remove these by fixing the code
    # ~40 instances: narrowing in brace init (batchnorm, ck_impl, addkernels)
    -Wno-c++11-narrowing
    -Wno-sign-compare           # ~1000+ instances: signed/unsigned comparisons throughout codebase
    -Wno-deprecated-declarations # 2 deprecated MIOpen APIs still have callers
    -Wno-deprecated-copy-with-dtor           # SolverBase needs Rule-of-5 refactoring
    -Wno-deprecated-copy-with-user-provided-dtor  # TuningIterationScopedLimiter needs refactoring
)

set(__clang_cxx_compile_options
    -Wno-unused-command-line-argument
    -Wthread-safety
    -Wshift-sign-overflow
    -Winconsistent-missing-destructor-override
    -Wcomma
    -Wmicrosoft-cpp-macro
    -Wnested-anon-types
    -Wlanguage-extension-token
    -Wunused-template
    -Wnrvo
)

if(WIN32)
    list(APPEND __clang_cxx_compile_options
        -fms-extensions
        -fms-compatibility)
    # AMD clang reports `__declspec(dllexport)` as "not supported" on the
    # x86_64-pc-windows-msvc target, even though the attribute is honored
    # (verified via llvm-readobj --coff-exports on MIOpen.dll). This produces
    # ~150k spurious warnings from the CMake-generated MIOPEN_EXPORT and
    # MIOPEN_INTERNALS_EXPORT macros. Suppress until the compiler issue is
    # resolved upstream.
    list(APPEND __clang_cxx_compile_options -Wno-ignored-attributes)
endif()

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${__cxx_compile_options}>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:Clang>>:${__clang_cxx_compile_options}>"
)

unset(__cxx_compile_options)
unset(__clang_cxx_compile_options)
