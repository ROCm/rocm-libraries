# Copyright Advanced Micro Devices, Inc., or its affiliates. SPDX-License-Identifier: MIT

# find_package(cblas REQUIRED CONFIG)

set(_CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../clients/benchmarks")

# Benchmark executable
add_executable(hipsparselt-bench "${_CMAKE_CURRENT_SOURCE_DIR}/client.cpp")

# Benchmark-specific configuration
target_compile_definitions(hipsparselt-bench PRIVATE HIPSPARSELT_BENCH)

target_link_libraries(hipsparselt-bench PRIVATE hipsparselt-clients-common)
hipsparselt_link_blas_libraries(hipsparselt-bench)

# # Platform-specific linking if(HIPSPARSELT_ENABLE_CUDA) target_link_libraries(hipsparselt-bench
# PRIVATE -llapack -lcblas ) else() target_link_libraries(hipsparselt-bench PRIVATE lapack cblas )
# endif()

if(NOT WIN32)
  target_link_libraries(hipsparselt-bench PRIVATE -lm -lstdc++fs)

  # Fortran runtime - detect based on compiler ID
  if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    target_link_libraries(hipsparselt-bench PRIVATE -lgfortran -llapack)
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Flang")
    target_link_libraries(hipsparselt-bench PRIVATE -lflang_rt.runtime lapack)
  else()
    # Fallback to gfortran for most systems
    target_link_libraries(hipsparselt-bench PRIVATE -lgfortran -llapack)
  endif()
endif()
