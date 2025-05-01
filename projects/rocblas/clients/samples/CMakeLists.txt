# ########################################################################
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

set( rocblas_samples_common
      ../common/singletons.cpp
      ../common/client_utility.cpp
      ../common/rocblas_random.cpp
      ../common/host_alloc.cpp
    )

# C example
add_executable( rocblas-example-c-dgeam example_c_dgeam.c )
target_compile_definitions( rocblas-example-c-dgeam PRIVATE __HIP_PLATFORM_HCC__ )
target_compile_options( rocblas-example-c-dgeam PRIVATE -std=c11 )

set( sample_list_c rocblas-example-c-dgeam )

# C++ examples
add_executable( rocblas-example-sscal example_sscal.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-scal-template example_scal_template.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-scal-multiple-strided-batch example_scal_multiple_strided_batch.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-solver example_solver_rocblas.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-hip-complex-her2 example_hip_complex_her2.cpp )
add_executable( rocblas-example-gemv-graph-capture example_gemv_graph_capture.cpp )
add_executable( rocblas-example-header-check example_header_check.cpp)

set( sample_list_base rocblas-example-sscal rocblas-example-scal-template rocblas-example-scal-multiple-strided-batch rocblas-example-solver rocblas-example-hip-complex-her2 rocblas-example-gemv-graph-capture rocblas-example-header-check)

if ( BUILD_FORTRAN_CLIENTS )
  # Fortran examples
  add_executable( rocblas-example-fortran-axpy example_fortran_axpy.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( rocblas-example-fortran-scal example_fortran_scal.f90 $<TARGET_OBJECTS:rocblas_fortran>)
  add_executable( rocblas-example-fortran-gemv example_fortran_gemv.f90 $<TARGET_OBJECTS:rocblas_fortran>)

  set( sample_list_fortran rocblas-example-fortran-axpy rocblas-example-fortran-scal rocblas-example-fortran-gemv )
endif()

if( BUILD_WITH_TENSILE )
  add_executable( rocblas-example-user-driven-tuning example_user_driven_tuning.cpp ${rocblas_samples_common} )

  set( sample_list_tensile rocblas-example-user-driven-tuning )
endif()

# L3 examples with or with tensile or source gemm builds
add_executable( rocblas-example-sgemm example_sgemm.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-sgemm-strided-batched example_sgemm_strided_batched.cpp ${rocblas_samples_common} )
add_executable( rocblas-example-sgemm-multiple-strided-batch example_sgemm_multiple_strided_batch.cpp  ${rocblas_samples_common} )

set( sample_list_L3 rocblas-example-sgemm rocblas-example-sgemm-strided-batched rocblas-example-sgemm-multiple-strided-batch )

# device includes
set( sample_list_hip_device ${sample_list_base} ${sample_list_tensile} ${sample_list_L3} )

# combined list
set( sample_list_all ${sample_list_c} ${sample_list_base} ${sample_list_fortran} ${sample_list_tensile} ${sample_list_L3} )

#
# apply settings

foreach( exe ${sample_list_fortran} )
  target_link_libraries( ${exe} PRIVATE rocblas_fortran_client )
endforeach( )

foreach( exe ${sample_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::rocblas Threads::Threads )

  if( NOT ${exe} STREQUAL "rocblas-example-header-check" )
    set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 14
    )
    rocm_install(TARGETS ${exe} COMPONENT samples)
  else()
    set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 11
    )
  endif()

  set_target_properties( ${exe} PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  if (NOT ${exe} STREQUAL "rocblas-example-c-dgeam")
    target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )
  endif()

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_definitions( ${exe} PRIVATE ROCBLAS_INTERNAL_API )
  endif( )

endforeach( )

foreach( exe ${sample_list_hip_device} )
  target_link_libraries( ${exe} PRIVATE hip::device )
endforeach( )
