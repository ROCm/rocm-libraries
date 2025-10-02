# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

include(FetchGoogleTest)

set(TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest_main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_test.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/prune_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/compress_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spmm_gtest.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/auxiliary_gtest.cpp"
)

add_executable(hipsparselt-test ${TEST_SOURCES})

target_compile_features(hipsparselt-test PRIVATE cxx_std_17)

target_compile_definitions(
  hipsparselt-test PRIVATE GOOGLE_TEST ROCM_USE_FLOAT16 HIPSPARSELT_INTERNAL_API
                           "__HIP_HCC_COMPAT_MODE__=1"
)

target_compile_options(hipsparselt-test PRIVATE -mf16c)

target_link_libraries(
  hipsparselt-test PRIVATE hipsparselt-clients-common GTest::gtest GTest::gtest_main
)
hipsparselt_link_blas_libraries(hipsparselt-test)

if(NOT WIN32)
  target_link_libraries(hipsparselt-test PRIVATE -lm -lstdc++fs)

  if(HIPSPARSELT_ENABLE_FORTRAN)
    if(CMAKE_Fortran_COMPILER_ID MATCH "Flang")
      target_link_libraries(hipsparselt-bench PRIVATE -lflang_rt.runtime lapack)
    else()
      target_link_libraries(hipsparselt-bench PRIVATE -lgfortran lapack)
    endif()
  endif()
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_common.yaml"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_common.yaml" COPYONLY
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_template.yaml"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_template.yaml" COPYONLY
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/../common/hipsparselt_gentest.py"
  "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_gentest.py" COPYONLY
)

set(hipsparselt_test_install_program
    "${CMAKE_CURRENT_SOURCE_DIR}/../common/hipsparselt_gentest.py"
)
set(hipsparselt_test_install_data_file "${CMAKE_CURRENT_BINARY_DIR}/hipsparselt_gtest.data")
set(hipsparselt_test_install_yaml_files)
list(
  APPEND
  hipsparselt_test_install_yaml_files
  "${CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_common.yaml"
  "${CMAKE_CURRENT_SOURCE_DIR}/../include/hipsparselt_template.yaml"
)

add_custom_command(
  OUTPUT "${hipsparselt_test_install_data_file}"
  COMMAND
    ${Python3_EXECUTABLE} "${hipsparselt_test_install_program}" -I
    "${CMAKE_CURRENT_SOURCE_DIR}/../include" "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest.yaml"
    -o "${hipsparselt_test_install_data_file}"
  DEPENDS ${hipsparselt_test_install_yaml_files}
          "${CMAKE_CURRENT_SOURCE_DIR}/hipsparselt_gtest.yaml"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Generating test data for hipsparselt-test"
)

add_custom_target(
  hipsparselt-test-files ALL
  DEPENDS ${hipsparselt_test_install_yaml_files} ${hipsparselt_test_install_data_file}
          ${hipsparselt_test_install_program}
)
set_target_properties(
  hipsparselt-test-files
  PROPERTIES HIPSPARSELT_INSTALL_YAML_FILES "${hipsparselt_test_install_yaml_files}"
             HIPSPARSELT_INSTALL_DATA_FILE "${hipsparselt_test_install_data_file}"
             HIPSPARSELT_INSTALL_PROGRAM "${hipsparselt_test_install_program}"
)

add_dependencies(hipsparselt-test hipsparselt-test-files)
