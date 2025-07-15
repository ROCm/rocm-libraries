# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

find_package(GTest REQUIRED)

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

# Test-specific configuration
target_compile_definitions(hipsparselt-test PRIVATE 
    GOOGLE_TEST
)

# Link with GTest and BLAS
target_link_libraries(hipsparselt-test PRIVATE 
    hipsparselt-clients-common
    GTest::gtest 
    GTest::gtest_main
    ${BLAS_LIBRARY}
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

# Generate test data
set(TEST_DATA_FILE "${PROJECT_BINARY_DIR}/staging/hipsparselt_gtest.data")
add_custom_command(
    OUTPUT "${TEST_DATA_FILE}"
    COMMAND ${Python3_EXECUTABLE} 
        "${_CMAKE_CURRENT_SOURCE_DIR}/../common/hipsparselt_gentest.py"
        -I "${_CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest.yaml"
        -o "${TEST_DATA_FILE}"
    DEPENDS 
        "${_CMAKE_CURRENT_SOURCE_DIR}/../common/hipsparselt_gentest.py"
        "${_CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_common.yaml"
        "${_CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest.yaml"
    WORKING_DIRECTORY "${_CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Generating test data for hipsparselt-test"
)

add_custom_target(hipsparselt-test-data DEPENDS "${TEST_DATA_FILE}")
add_dependencies(hipsparselt-test hipsparselt-test-data)

# Install test executable and data
install(TARGETS hipsparselt-test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tests
)

install(FILES "${TEST_DATA_FILE}"
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT tests
)