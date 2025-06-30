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

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../client")

find_package(BLAS REQUIRED)

find_package(CLI11)
if(NOT CLI11_FOUND)
    if(ROCROLLER_ENABLE_FETCH)
        FetchContent_Declare(
            CLI11
            GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
            GIT_TAG main
        )
        FetchContent_MakeAvailable(CLI11)
    else()
        message(FATAL_ERROR "Failed to find CLI11")
    endif()
endif()

add_executable(rocroller_gemm
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/GEMMParameters.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/GraphInspector.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/visualize.cpp"
)
target_link_libraries(rocroller_gemm
    PUBLIC
        roc::rocroller
        rocroller::common-test-utilities
        CLI11::CLI11
        ${BLAS_LIBRARIES}
)
if(ROCROLLER_BUILD_SHARED_LIBS)
    target_link_libraries(rocroller_gemm PUBLIC ${llvm_yaml_libs})
endif()

add_executable(rocroller_codegen_stress "${_CMAKE_CURRENT_SOURCE_DIR}/codegen_stress_test.cpp")
target_link_libraries(rocroller_codegen_stress
    PUBLIC
        roc::rocroller-interface
        rocroller::common-test-utilities
        CLI11::CLI11
        ${BLAS_LIBRARIES}
)
if(ROCROLLER_BUILD_SHARED_LIBS)
    target_link_libraries(rocroller_codegen_stress PUBLIC ${llvm_yaml_libs})
endif()
