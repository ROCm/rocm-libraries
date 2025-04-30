# ########################################################################
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

set(hipblaslt_bench_source
  client.cpp
  )

add_executable( hipblaslt-bench ${hipblaslt_bench_source} ${hipblaslt_test_bench_common} )

# Internal header includes
target_include_directories( hipblaslt-bench
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
)

# External header includes included as system files
target_include_directories( hipblaslt-bench
  SYSTEM PRIVATE
    $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${BLAS_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${BLIS_INCLUDE_DIR}> # may be blank if not used
)

target_link_libraries( hipblaslt-bench PRIVATE ${BLAS_LIBRARY} roc::hipblaslt )

if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # GCC or hip-clang needs specific flags to turn on f16c intrinsics
  target_compile_options( hipblaslt-bench PRIVATE -mf16c )
endif( )

target_compile_definitions( hipblaslt-bench PRIVATE HIPBLASLT_BENCH ROCM_USE_FLOAT16 HIPBLASLT_INTERNAL_API ${TENSILE_DEFINES} )
if ( NOT BUILD_FORTRAN_CLIENTS )
  target_compile_definitions( hipblaslt-bench PRIVATE CLIENTS_NO_FORTRAN )
endif()

target_compile_options(hipblaslt-bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMMON_CXX_OPTIONS}>)

if( NOT BUILD_CUDA )
  target_link_libraries( hipblaslt-bench PRIVATE hip::host hip::device )
else()
  target_compile_definitions( hipblaslt-bench PRIVATE __HIP_PLATFORM_NVIDIA__ )
  target_include_directories( hipblaslt-bench
    PRIVATE
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>

  )
  target_link_libraries( hipblaslt-bench PRIVATE ${CUDA_LIBRARIES} )
endif()

if( LEGACY_HIPBLAS_DIRECT )
  target_compile_definitions(hipblaslt-bench PUBLIC LEGACY_HIPBLAS_DIRECT )
endif()

# target_compile_options does not go to linker like CMAKE_CXX_FLAGS does, so manually add
if (NOT WIN32)
  if (BUILD_CUDA)
    target_link_libraries( hipblaslt-bench PRIVATE -llapack -lcblas )
  else()
    target_link_libraries( hipblaslt-bench PRIVATE lapack cblas )
  endif()
  list( APPEND COMMON_LINK_LIBS "-lm -lstdc++fs")

  if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    list( APPEND COMMON_LINK_LIBS "-lgfortran") # for lapack
  else()
    list( APPEND COMMON_LINK_LIBS "-lflang -lflangrti") # for lapack
  endif()
else()
  list( APPEND COMMON_LINK_LIBS "libomp")
endif()
target_link_libraries( hipblaslt-bench PRIVATE ${COMMON_LINK_LIBS} )

set_target_properties( hipblaslt-bench PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
)

add_dependencies( hipblaslt-bench hipblaslt-common )

rocm_install(TARGETS hipblaslt-bench COMPONENT benchmarks)

# Other bench files
add_executable( hipblaslt-bench-groupedgemm-fixed-mk client_groupedgemm_fixed_mk.cpp ../common/hipblaslt_random.cpp ../common/hipblaslt_arguments.cpp)
add_executable( hipblaslt-bench-extop-layernorm client_extop_layernorm.cpp ../common/hipblaslt_random.cpp)
add_executable( hipblaslt-bench-extop-matrixtransform client_extop_matrixtransform.cpp ../common/hipblaslt_random.cpp)
add_executable( hipblaslt-bench-extop-softmax client_extop_softmax.cpp ../common/hipblaslt_random.cpp)
add_executable( hipblaslt-bench-extop-amax client_extop_amax.cpp  ../common/hipblaslt_random.cpp)
set(ext_bench_list_all hipblaslt-bench-groupedgemm-fixed-mk hipblaslt-bench-extop-layernorm hipblaslt-bench-extop-matrixtransform hipblaslt-bench-extop-softmax hipblaslt-bench-extop-amax)
# Currently only build this for Ubuntu because other distro's llvm-dev version is not matching the rocm's version.
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  file(READ "/etc/os-release" OS_RELEASE)
  if(OS_RELEASE MATCHES "Ubuntu")
    add_executable( hipblaslt-sequence client_sequence.cpp ../common/hipblaslt_random.cpp)
    find_package(LLVM 13.0 QUIET CONFIG)
    if(NOT LLVM_FOUND)
        find_package(LLVM 12.0 QUIET CONFIG)
        if(NOT LLVM_FOUND)
            find_package(LLVM REQUIRED CONFIG)
        endif()
    endif()
    find_library(LLVMObjectYAML_LIBRARY
      NAMES LLVMObjectYAML
      PATHS ${LLVM_LIBRARY_DIR})
    if(LLVMObjectYAML_LIBRARY)
      target_link_libraries(hipblaslt-sequence PRIVATE LLVMObjectYAML )
    else()
      target_link_libraries(hipblaslt-sequence PRIVATE LLVM )
    endif()
    target_include_directories(hipblaslt-sequence PRIVATE ${LLVM_INCLUDE_DIRS})
    set(ext_bench_list_all ${ext_bench_list_all} hipblaslt-sequence)
    set( HIPBLASLT_SEQUENCE_YAML "${PROJECT_BINARY_DIR}/staging/sequence.yaml")
    add_custom_command( OUTPUT "${HIPBLASLT_SEQUENCE_YAML}"
                        COMMAND ${CMAKE_COMMAND} -E copy sequence.yaml "${HIPBLASLT_SEQUENCE_YAML}"
                        DEPENDS sequence.yaml
                        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" )

    add_custom_target( hipblaslt-sequence-yaml DEPENDS "${HIPBLASLT_SEQUENCE_YAML}" )
    add_dependencies( hipblaslt-sequence hipblaslt-sequence-yaml )
    rocm_install(
      FILES ${HIPBLASLT_SEQUENCE_YAML}
      DESTINATION "${CMAKE_INSTALL_BINDIR}"
      COMPONENT benchmarks
    )
  endif()
endif()

foreach( exe ${ext_bench_list_all} )
  target_link_libraries( ${exe} PRIVATE roc::hipblaslt )

  set_target_properties( ${exe} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging"
  )

  target_compile_definitions( ${exe} PRIVATE ROCM_USE_FLOAT16 )

  target_include_directories( ${exe}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/common>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/include>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../library/src/include>
      )

  target_include_directories( ${exe}
    SYSTEM PRIVATE
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      )

  if( CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # GCC or hip-clang needs specific flags to turn on f16c intrinsics
    target_compile_options( ${exe} PRIVATE -mf16c -Wno-unused-result )
    target_compile_definitions( ${exe} PRIVATE ROCBLASLT_INTERNAL_API )
  endif( )
endforeach( )

foreach( exe ${ext_bench_list_all} )
  target_link_libraries( ${exe} PRIVATE hip::device )
endforeach( )

foreach( exe ${ext_bench_list_all} )
  rocm_install(TARGETS ${exe} COMPONENT benchmarks)
endforeach( )

# Add tests for special cases
add_test(NAME hipblaslt-bench-groupedgemm-fixed-mk
         COMMAND hipblaslt-bench-groupedgemm-fixed-mk --trans_a N --trans_b N --in_datatype fp16 --out_datatype fp16 -m 2048 -k 2048 -n 0 -n 1 -n 1 -n 0 -n 1 -n 0 -n 1 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -n 0 -V)
