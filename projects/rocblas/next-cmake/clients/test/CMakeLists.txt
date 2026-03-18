# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(_CMAKE_CURRENT_SOURCE_DIR ${TEMP_CLIENTS_SOURCE_DIR}/gtest)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(rocblas-test)

set_target_properties(rocblas-test PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                              "${PROJECT_BINARY_DIR}/staging")

target_compile_definitions(rocblas-test PRIVATE GOOGLE_TEST)
if(ROCBLAS_ENABLE_TENSILE)
  target_compile_definitions(rocblas-test PRIVATE BUILD_WITH_TENSILE)
endif()

target_link_libraries(rocblas-test PRIVATE rocblas-clients-common)

if(ROCBLAS_ENABLE_ASAN)
  rocblas_target_configure_sanitizers(rocblas-test PRIVATE)
endif()

add_subdirectory(include)
add_subdirectory(src)

set(yaml_test_files
    "${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/asum_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/atomics_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/aux_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/axpy_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas1_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas1_gtest.yaml.new"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/blas3_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/copy_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/dgmm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/dot_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/geam_ex_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/geam_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm_batched_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemm_strided_batched_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemmt_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/gemv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/general_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/ger_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/geruc_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/get_solutions_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hemm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hemv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/her2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/her2k_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/her_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/herk_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/herkx_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hpmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hpr2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/hpr_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/iamax_iamin_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/known_bugs.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/logging_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/multiheaded_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/nrm2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/ostream_threadsafety_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/rocblas_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/rot_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/sbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/scal_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_atomics_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_matrix_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_pointer_mode_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/set_get_vector_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spr2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/spr_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/swap_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/symm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/symv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syr2_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syr2k_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syr_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syrk_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/syrkx_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tbmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tbsv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tpmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/tpsv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trmm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trmv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trsm_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trsv_gtest.yaml"
    "${_CMAKE_CURRENT_SOURCE_DIR}/trtri_gtest.yaml")

set(rocblas_test_data
    "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_gtest.data")

add_custom_command(
  OUTPUT "${rocblas_test_data}"
  COMMAND
    "${Python3_EXECUTABLE}" ARGS "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_gentest.py"
    -I "${_CMAKE_CURRENT_SOURCE_DIR}/../include" "rocblas_gtest.yaml" -o
    "${rocblas_test_data}"
  DEPENDS "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_gentest.py"
          "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_common.yaml"
          ${yaml_test_files}
  WORKING_DIRECTORY "${_CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND_EXPAND_LISTS VERBATIM)

add_custom_target(
  rocblas-test-data ALL
  DEPENDS "${rocblas_test_data}"
  COMMENT "Generating rocBLAS test data")

set_target_properties(rocblas-test-data PROPERTIES ROCBLAS_INSTALL_FILES "${rocblas_test_data}")

function(rocblas_add_copy_command OUTPUT_FILE SOURCE_FILE COMMENT_MSG)
  add_custom_command(
    OUTPUT "${OUTPUT_FILE}"
    COMMAND "${CMAKE_COMMAND}" ARGS -E copy "${SOURCE_FILE}" "${OUTPUT_FILE}"
    DEPENDS "${SOURCE_FILE}"
    WORKING_DIRECTORY "${_CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "${COMMENT_MSG}"
    VERBATIM)
endfunction()

set(rocblas_smoke "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_smoke.yaml")
rocblas_add_copy_command(
  "${rocblas_smoke}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_smoke.yaml"
  "Copying rocBLAS smoke test configuration")

set(rocblas_extras "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_extras.yaml")
rocblas_add_copy_command(
  "${rocblas_extras}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_extras.yaml"
  "Copying rocBLAS extras configuration")

set(rocblas_rtest_py "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_rtest.py")
rocblas_add_copy_command(
  "${rocblas_rtest_py}" "${_CMAKE_CURRENT_SOURCE_DIR}/../../rtest.py"
  "Copying rocBLAS test scripts")

set(rocblas_rtest_xml "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_rtest.xml")
rocblas_add_copy_command(
  "${rocblas_rtest_xml}" "${_CMAKE_CURRENT_SOURCE_DIR}/../../rtest.xml"
  "Copying rocBLAS test XML")

set(rocblas_common "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_common.yaml")
rocblas_add_copy_command(
  "${rocblas_common}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_common.yaml"
  "Copying rocBLAS common configuration")

set(rocblas_template "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_template.yaml")
rocblas_add_copy_command(
  "${rocblas_template}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_template.yaml"
  "Copying rocBLAS template configuration")

set(rocblas_general_yaml "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_general.yaml")
rocblas_add_copy_command(
  "${rocblas_general_yaml}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../include/rocblas_general.yaml"
  "Copying rocBLAS general configuration")

set(rocblas_gentest_py "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_gentest.py")
rocblas_add_copy_command(
  "${rocblas_gentest_py}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../common/rocblas_gentest.py"
  "Copying rocBLAS gentest script")

set(rocblas_clients_readme "${CMAKE_CURRENT_BINARY_DIR}/../staging/rocblas_clients_readme.txt")
rocblas_add_copy_command(
  "${rocblas_clients_readme}"
  "${_CMAKE_CURRENT_SOURCE_DIR}/../rocblas_clients_readme.txt"
  "Copying rocBLAS clients readme")

set(rocblas_install_files_list)
list(APPEND rocblas_install_files_list
  ${rocblas_smoke}
  ${rocblas_extras}
  ${rocblas_common}
  ${rocblas_template}
  ${rocblas_general_yaml}
  ${rocblas_clients_readme}
  ${rocblas_rtest_xml})

set(rocblas_install_programs_list)
list(APPEND rocblas_install_programs_list
  ${rocblas_rtest_py}
  ${rocblas_gentest_py})

set(rocblas_test_dependencies)
list(APPEND rocblas_test_dependencies
  ${rocblas_install_files_list}
  ${rocblas_install_programs_list})

# I think we should move this up into the clients/CMakeLists.txt file
add_custom_target(
  rocblas-test-files ALL
  DEPENDS ${rocblas_test_dependencies}
  COMMENT "Generating rocBLAS test files")

set_target_properties(
  rocblas-test-files
  PROPERTIES
    ROCBLAS_INSTALL_FILES "${rocblas_install_files_list}"
    ROCBLAS_INSTALL_PROGRAMS "${rocblas_install_programs_list}")

add_dependencies(rocblas-test rocblas-test-data rocblas-test-files)
