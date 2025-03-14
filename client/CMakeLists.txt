
################################################################################
#
# MIT License
#
# Copyright 2024-2025 AMD ROCm(TM) Software
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


message("Building client program")

# Header only, not visible to library users, system package not required
rocroller_add_dependency(CLI11 VERSION 2.1.2)

add_executable(
    rocroller_gemm gemm.cpp GEMMParameters.cpp GraphInspector.cpp visualize.cpp
)

target_compile_options(rocroller_gemm PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(rocroller_gemm PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
    rocroller_gemm PUBLIC rocroller_interface common_test_utilities CLI11::CLI11
)
if(BUILD_SHARED_LIBS)
    target_link_libraries(
        rocroller_gemm PUBLIC ${llvm_yaml_libs}
    )
endif()

set_target_properties(
    rocroller_gemm PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/client"
        OUTPUT_NAME "rocRoller_gemm"
)
add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/bin/client/gemm"
    COMMAND
        sudo ln -s $<TARGET_FILE:rocroller_gemm>
            ${PROJECT_BINARY_DIR}/bin/client/gemm
    MAIN_DEPENDENCY rocroller_gemm
)

add_executable(rocroller_codegen_stress codegen_stress_test.cpp)

target_compile_options(rocroller_codegen_stress PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_link_options(rocroller_codegen_stress PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
    rocroller_codegen_stress PUBLIC rocroller_interface CLI11::CLI11
)
if(BUILD_SHARED_LIBS)
    target_link_libraries(
        rocroller_codegen_stress PUBLIC ${llvm_yaml_libs}
    )
endif()

set_target_properties(
    rocroller_codegen_stress PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin/client"
        OUTPUT_NAME "rocRoller_codegen_stress"
)
add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/bin/client/codegen_stress"
    COMMAND
        sudo ln -s $<TARGET_FILE:rocroller_codegen_stress>
            ${PROJECT_BINARY_DIR}/bin/client/codegen_stress
    MAIN_DEPENDENCY rocroller_codegen_stress
)

add_custom_target(all_clients)
add_dependencies(all_clients rocroller_gemm rocroller_codegen_stress)

rocm_install(
  TARGETS rocroller_gemm rocroller_codegen_stress
  COMPONENT benchmarks
)

# Add gemm client tests.  Use pytest-cmake integration for parallelism.
find_package(Pytest)
pytest_discover_tests(
  GEMMClientTests
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/client
  ENVIRONMENT "ROCROLLER_BUILD_DIR=${PROJECT_BINARY_DIR}"
)
