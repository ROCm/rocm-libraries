# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(hipsparselt-bench "${CMAKE_CURRENT_SOURCE_DIR}/client.cpp")

target_compile_definitions(hipsparselt-bench PRIVATE HIPSPARSELT_BENCH)

target_link_libraries(hipsparselt-bench PRIVATE hipsparselt-clients-common)
hipsparselt_link_blas_libraries(hipsparselt-bench)

if(NOT WIN32)
  target_link_libraries(hipsparselt-bench PRIVATE -lm -lstdc++fs)

  if(HIPSPARSELT_ENABLE_FORTRAN)
    if(CMAKE_Fortran_COMPILER_ID MATCH "Flang")
      target_link_libraries(hipsparselt-bench PRIVATE -lflang_rt.runtime lapack)
    else()
      target_link_libraries(hipsparselt-bench PRIVATE -lgfortran lapack)
    endif()
  endif()
endif()
