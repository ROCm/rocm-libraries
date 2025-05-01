# ########################################################################
# Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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
# ########################################################################

#enable_testing()
find_package( GTest REQUIRED )

if( BUILD_WITH_TENSILE )
  set(rocblas_tensile_test_source
      # Keep multiheaded_gtest.cpp first as we want
      # to allow it to create the first TensileHost !
      # Current GTESTs are run in the linking order as added with global variables
      multiheaded_gtest.cpp
      # use of tensile based functions (gemm)
      atomics_mode_gtest.cpp
      get_solutions_gtest.cpp

  )
endif()

set(rocblas_no_tensile_test_source
    # general
    rocblas_gtest_main.cpp
    rocblas_test.cpp
    general_gtest.cpp
    set_get_pointer_mode_gtest.cpp
    set_get_atomics_mode_gtest.cpp
    logging_mode_gtest.cpp
    ostream_threadsafety_gtest.cpp
    set_get_vector_gtest.cpp
    set_get_matrix_gtest.cpp
    # blas1
    blas1/asum_gtest.cpp
    blas1/axpy_gtest.cpp
    blas1/copy_gtest.cpp
    blas1/dot_gtest.cpp
    blas1/iamaxmin_gtest.cpp
    blas1/nrm2_gtest.cpp
    blas1/rot_gtest.cpp
    blas1/scal_gtest.cpp
    blas1/swap_gtest.cpp
    # blas1_ex
    blas_ex/axpy_ex_gtest.cpp
    blas_ex/dot_ex_gtest.cpp
    blas_ex/nrm2_ex_gtest.cpp
    blas_ex/rot_ex_gtest.cpp
    blas_ex/scal_ex_gtest.cpp
    # blas2
    blas2/trsv_gtest.cpp
    blas2/gbmv_gtest.cpp
    blas2/gemv_gtest.cpp
    blas2/hbmv_gtest.cpp
    blas2/hemv_gtest.cpp
    blas2/her_gtest.cpp
    blas2/her2_gtest.cpp
    blas2/hpmv_gtest.cpp
    blas2/hpr_gtest.cpp
    blas2/hpr2_gtest.cpp
    blas2/trmv_gtest.cpp
    blas2/tpmv_gtest.cpp
    blas2/tbmv_gtest.cpp
    blas2/tbsv_gtest.cpp
    blas2/tpsv_gtest.cpp
    blas2/ger_gtest.cpp
    blas2/geru_gtest.cpp
    blas2/gerc_gtest.cpp
    blas2/spr_gtest.cpp
    blas2/spr2_gtest.cpp
    blas2/syr_gtest.cpp
    blas2/syr2_gtest.cpp
    blas2/sbmv_gtest.cpp
    blas2/spmv_gtest.cpp
    blas2/symv_gtest.cpp
    # blas3 may use tensile or source gemm
    blas3/gemm_gtest.cpp
    blas_ex/gemm_ex_gtest.cpp
    blas_ex/gemm_ex3_gtest.cpp
    blas3/symm_gtest.cpp
    blas3/hemm_gtest.cpp
    blas3/trsm_gtest.cpp
    blas3/trtri_gtest.cpp
    blas3/trmm_gtest.cpp
    blas3/syrk_gtest.cpp
    blas3/syrkx_gtest.cpp
    blas3/syr2k_gtest.cpp
    blas3/herk_gtest.cpp
    blas3/herkx_gtest.cpp
    blas3/her2k_gtest.cpp
    blas3/dgmm_gtest.cpp
    blas3/geam_gtest.cpp
    blas_ex/gemmt_gtest.cpp
    blas_ex/geam_ex_gtest.cpp
  )

# Keep ${rocblas_tensile_test_source} first, so that multiheaded tests are the
# first to initialize Tensile.

add_executable( rocblas-test ${rocblas_tensile_test_source} ${rocblas_no_tensile_test_source} )

if (WIN32)
  target_compile_definitions( rocblas-test PRIVATE _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING ${BLIS_DEFINES} )
endif()


rocblas_client_library_settings( rocblas-test )

target_link_libraries( rocblas-test PRIVATE rocblas_clients_testing_common rocblas_clients_common )
if( BUILD_FORTRAN_CLIENTS )
  target_link_libraries( rocblas-test PRIVATE rocblas_fortran_client )
endif( )
target_link_libraries( rocblas-test PRIVATE roc::rocblas hip::host hip::device )

if (NOT WIN32)
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs" )
endif()
target_link_libraries( rocblas-test PRIVATE ${COMMON_LINK_LIBS} )

target_compile_options(rocblas-test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)

target_compile_definitions( rocblas-test PRIVATE ROCM_USE_FLOAT16 ROCBLAS_INTERNAL_API ROCBLAS_NO_DEPRECATED_WARNINGS )

if (WIN32)
  # for now adding in all .dll as dependency chain is not cmake based on win32
  file( GLOB third_party_dlls
    LIST_DIRECTORIES OFF
    CONFIGURE_DEPENDS
    ${OPENBLAS_DIR}/bin/*.dll
    ${HIP_DIR}/bin/amd*.dll
    ${HIP_DIR}/bin/hiprt*.dll
    ${HIP_DIR}/bin/hipinfo.exe
    ${CMAKE_SOURCE_DIR}/rtest.*
    C:/Windows/System32/libomp140*.dll
  )
  foreach( file_i ${third_party_dlls})
    add_custom_command( TARGET rocblas-test POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/staging/ )
  endforeach( file_i )
endif()

set_target_properties( rocblas-test PROPERTIES
  IMPORT_PREFIX ""
  IMPORT_SUFFIX ".lib"
  LINKER_LANGUAGE CXX
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

file(GLOB yaml_files "${CMAKE_CURRENT_SOURCE_DIR}/*.yaml")
set( ROCBLAS_TEST_DATA "${PROJECT_BINARY_DIR}/staging/rocblas_gtest.data")
add_custom_command( OUTPUT "${ROCBLAS_TEST_DATA}"
                    COMMAND ${python} ../common/rocblas_gentest.py -I ../include rocblas_gtest.yaml -o "${ROCBLAS_TEST_DATA}"
                    DEPENDS ../common/rocblas_gentest.py ../include/rocblas_common.yaml ${yaml_files}
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )
add_custom_target( rocblas-test-data DEPENDS "${ROCBLAS_TEST_DATA}" )

set( ROCBLAS_SMOKE "${PROJECT_BINARY_DIR}/staging/rocblas_smoke.yaml")
add_custom_command( OUTPUT "${ROCBLAS_SMOKE}"
                    COMMAND ${CMAKE_COMMAND} -E copy ../include/rocblas_smoke.yaml "${ROCBLAS_SMOKE}"
                    DEPENDS ../include/rocblas_smoke.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_EXTRAS "${PROJECT_BINARY_DIR}/staging/rocblas_extras.yaml")
add_custom_command( OUTPUT "${ROCBLAS_EXTRAS}"
                    COMMAND ${CMAKE_COMMAND} -E copy ../include/rocblas_extras.yaml "${ROCBLAS_EXTRAS}"
                    DEPENDS ../include/rocblas_extras.yaml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_CLIENTS_RTEST "${PROJECT_BINARY_DIR}/staging/rocblas_rtest.py")
add_custom_command( OUTPUT "${ROCBLAS_CLIENTS_RTEST}"
                    COMMAND ${CMAKE_COMMAND} -E copy ../../rtest.py "${ROCBLAS_CLIENTS_RTEST}"
                    DEPENDS ../../rtest.py
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

set( ROCBLAS_CLIENTS_RTEST_XML "${PROJECT_BINARY_DIR}/staging/rocblas_rtest.xml")
add_custom_command( OUTPUT "${ROCBLAS_CLIENTS_RTEST_XML}"
                    COMMAND ${CMAKE_COMMAND} -E copy ../../rtest.xml "${ROCBLAS_CLIENTS_RTEST_XML}"
                    DEPENDS ../../rtest.xml
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

add_custom_target( rocblas-test-files DEPENDS "${ROCBLAS_SMOKE}" "${ROCBLAS_EXTRAS}" "${ROCBLAS_CLIENTS_RTEST}" "${ROCBLAS_CLIENTS_RTEST_XML}" )

add_dependencies( rocblas-test rocblas-test-data rocblas-test-files rocblas-common )

rocm_install(TARGETS rocblas-test COMPONENT tests)

rocm_install(
  PROGRAMS ${ROCBLAS_CLIENTS_RTEST}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT tests
)

rocm_install(
  FILES ${ROCBLAS_SMOKE} ${ROCBLAS_EXTRAS} ${ROCBLAS_CLIENTS_RTEST_XML}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT tests
)

rocm_install(FILES ${ROCBLAS_TEST_DATA}
  DESTINATION "${CMAKE_INSTALL_BINDIR}"
  COMPONENT tests
)
