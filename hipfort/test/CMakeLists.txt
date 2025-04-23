#
# File: test/CMakeLists.txt

# These tests are all Makefile based tests.
# Each test is contained in its own subdirectory of this test directory
# The Makefile must create a binary with the same name as the directory name.

cmake_minimum_required(VERSION 3.5)
if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
  project(test-hipfort)
endif()

# make-based test infrastructure

set(CMAKE_BUILD_PARALLEL_LEVEL "4" CACHE STRING "Number of build jobs")

add_custom_target(all-f2003-tests
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2003 build)
add_custom_target(all-f2003-tests-run
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j1 PREFIX=f2003 run
	DEPENDS all-f2003-tests)
add_custom_target(all-f2003-tests-clean
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2003 clean)

add_custom_target(all-f2008-tests
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2008 build)
add_custom_target(all-f2008-tests-run
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j1 PREFIX=f2008 run
	DEPENDS all-f2008-tests)
add_custom_target(all-f2008-tests-clean
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} PREFIX=f2008 clean)

add_custom_target(all-tests
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} build_all)
add_custom_target(all-tests-run
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j1 run_all
	DEPENDS all-tests)
add_custom_target(all-tests-clean
	COMMAND make -C ${CMAKE_CURRENT_SOURCE_DIR} -j${CMAKE_BUILD_PARALLEL_LEVEL} clean_all)


# cmake-based test infrastructure

if(BUILD_TESTING)
  function(hipfort_add_test lib func dir)
    add_executable(hipfort_test_${dir}_${lib}_${func} ${dir}/${lib}/${func}.f03)
    target_link_libraries(hipfort_test_${dir}_${lib}_${func} PRIVATE hipfort::${lib} hipfort::hip)

    # the tests call EXIT, which is a GNU extension
    target_compile_options(hipfort_test_${dir}_${lib}_${func} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-std=gnu>)

    add_test(
      NAME hipfort_test_${dir}_${lib}_${func}
      COMMAND hipfort_test_${dir}_${lib}_${func}
    )
  endfunction()

  if(TARGET hipfort::hipblas)
    hipfort_add_test(hipblas cgemm f2003)
    hipfort_add_test(hipblas dgemm f2003)
    hipfort_add_test(hipblas dger f2003)
    hipfort_add_test(hipblas dscal f2003)
    hipfort_add_test(hipblas saxpy f2003)
    hipfort_add_test(hipblas scopy f2003)
    hipfort_add_test(hipblas sgemv f2003)
    hipfort_add_test(hipblas sger f2003)
    hipfort_add_test(hipblas sswap f2003)

    hipfort_add_test(hipblas cgemm f2008)
    hipfort_add_test(hipblas dgemm f2008)
    hipfort_add_test(hipblas dger f2008)
    hipfort_add_test(hipblas dscal f2008)
    hipfort_add_test(hipblas saxpy f2008)
    hipfort_add_test(hipblas scopy f2008)
    hipfort_add_test(hipblas sgemv f2008)
    hipfort_add_test(hipblas sger f2008)
    hipfort_add_test(hipblas sswap f2008)
  endif()

  if(TARGET hipfort::hipfft)
    hipfort_add_test(hipfft hipfft f2003)
    hipfort_add_test(hipfft hipfft f2008)
  endif()

  if(TARGET hipfort::hipsolver)
    hipfort_add_test(hipsolver hipsolverdgetrf f2008)
  endif()

  if(TARGET hipfort::rocblas)
    hipfort_add_test(rocblas saxpy f2003)
    hipfort_add_test(rocblas saxpy f2008)
  endif()

  if(TARGET hipfort::rocfft)
    hipfort_add_test(rocfft rocfft f2003)
    hipfort_add_test(rocfft rocfft f2008)
  endif()

  if(TARGET hipfort::rocsolver)
    hipfort_add_test(rocsolver rocsolver_dgeqrf f2003)
    hipfort_add_test(rocsolver rocsolver_dgeqrf f2008)
  endif()

  if(TARGET hipfort::rocsparse)
    hipfort_add_test(rocsparse ddoti f2003)
    hipfort_add_test(rocsparse ddoti f2008)
  endif()

  if(TARGET hipfort::roctx)
    include(CheckLanguage)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
      enable_language(HIP)
      hipfort_add_test(roctx main f2003)
      target_sources(hipfort_test_f2003_roctx_main PRIVATE f2003/roctx/hip_implementation.cpp)
      set_source_files_properties(f2003/roctx/hip_implementation.cpp
        PROPERTIES
          LANGUAGE HIP
      )
      set_target_properties(hipfort_test_f2003_roctx_main
        PROPERTIES
          LINKER_LANGUAGE Fortran
          POSITION_INDEPENDENT_CODE ON
      )
    endif()
  endif()
endif()
