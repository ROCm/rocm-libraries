# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FetchGoogleTest)

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../clients/gtest")

# Test source files
set(TEST_SOURCES
    "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest_main.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_test.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/prune_gtest.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/compress_gtest.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spmm_gtest.cpp"
    "${_CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp"
)

# Test executable
add_executable(hipsparselt-test
    ${TEST_SOURCES}
)

target_compile_features(hipsparselt-test PRIVATE cxx_std_17)

target_compile_definitions(hipsparselt-test PRIVATE 
    GOOGLE_TEST
    ROCM_USE_FLOAT16
    HIPSPARSELT_INTERNAL_API
    "__HIP_HCC_COMPAT_MODE__=1"
)

# TODO: -mf16c may not be required? What does this do?
target_compile_options(hipsparselt-test PRIVATE -mf16c)

# Link with GTest and BLAS
target_link_libraries(hipsparselt-test PRIVATE 
    hipsparselt-clients-common
    GTest::gtest 
    GTest::gtest_main
    ${BLAS_LIBRARIES_TO_LINK}
)

# Platform-specific linking for tests
if(HIPSPARSELT_ENABLE_CUDA)
    target_link_libraries(hipsparselt-test PRIVATE 
        -llapack 
        -lcblas
    )
else()
    target_link_libraries(hipsparselt-test PRIVATE 
        lapack 
        cblas
    )
endif()

# Additional libraries for tests
if(NOT WIN32)
    target_link_libraries(hipsparselt-test PRIVATE 
        -lm 
        -lstdc++fs
    )

    # Fortran runtime - detect based on compiler ID
    if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
        target_link_libraries(hipsparselt-test PRIVATE -lgfortran lapack)
    elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
        target_link_libraries(hipsparselt-test PRIVATE -lflang_rt.runtime lapack)
    else()
        # Fallback to gfortran for most systems
        target_link_libraries(hipsparselt-test PRIVATE -lgfortran lapack)
    endif()
endif()

set_target_properties(hipsparselt-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

set(hipsparselt_test_install_program "${_CMAKE_CURRENT_SOURCE_DIR}/../common/hipsparselt_gentest.py")
set(hipsparselt_test_install_data_file "${PROJECT_BINARY_DIR}/staging/hipsparselt_gtest.data")
set(hipsparselt_test_install_yaml_files)
list(APPEND hipsparselt_test_install_yaml_files
    "${_CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_common.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_template.yaml"
)

add_custom_command(
    OUTPUT "${hipsparselt_test_install_data_file}"
    COMMAND ${Python3_EXECUTABLE} 
        "${hipsparselt_test_install_program}"
        -I "${_CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest.yaml"
        -o "${hipsparselt_test_install_data_file}"
    DEPENDS 
        ${hipsparselt_test_install_yaml_files}
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest.yaml"
    WORKING_DIRECTORY "${_CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating test data for hipsparselt-test"
)

add_custom_target(hipsparselt-test-files ALL
    DEPENDS
        ${hipsparselt_test_install_yaml_files}
        ${hipsparselt_test_install_data_file}
        ${hipsparselt_test_install_program}
)
set_target_properties(hipsparselt-test-files PROPERTIES
    HIPSPARSELT_INSTALL_YAML_FILES "${hipsparselt_test_install_yaml_files}"
    HIPSPARSELT_INSTALL_DATA_FILE "${hipsparselt_test_install_data_file}"
    HIPSPARSELT_INSTALL_PROGRAM "${hipsparselt_test_install_program}"
)

add_dependencies(hipsparselt-test hipsparselt-test-files)
