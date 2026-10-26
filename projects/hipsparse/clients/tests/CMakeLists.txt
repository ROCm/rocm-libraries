# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Optional: CTest test categorization when building in rocm-libraries with shared/ctest.
# Skipped when shared/ is not present (e.g. standalone hipsparse or sparse checkout in CI).
if(DEFINED ROCM_LIBRARIES_ROOT AND EXISTS "${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  include("${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake")
  set(HIPSPARSE_HAS_CTEST_CATEGORIES ON)
endif()

# Provide the client matrices; ClientMatrices.cmake converts only the .bin files that are missing
set(CMAKE_MATRICES_DIR "${PROJECT_BINARY_DIR}/clients/matrices" CACHE STRING "Matrices directory.")
set(CONVERT_SOURCE "${PROJECT_SOURCE_DIR}/deps/convert.cpp" CACHE STRING "Convert tool mtx2csr.")
include("${PROJECT_SOURCE_DIR}/cmake/ClientMatrices.cmake")

# Copy matrices
if(NOT CMAKE_MATRICES_DIR STREQUAL "${PROJECT_BINARY_DIR}/clients/matrices")
    message("Copy matrix files from ${CMAKE_MATRICES_DIR} to ${PROJECT_BINARY_DIR}/clients/matrices")

    file(COPY ${CMAKE_MATRICES_DIR}/ DESTINATION ${PROJECT_BINARY_DIR}/clients/matrices)
endif()

add_executable(hipsparse-test)

# Each backend CMakeLists sets HIPSPARSE_TEST_SOURCES to the routines that
# backend can compile, plus the YAML inputs for hipsparse_test.data.
if(HIPSPARSE_ENABLE_HIP)
  include(rocm/CMakeLists.txt)
elseif(HIPSPARSE_ENABLE_CUDA)
  if(CUDAToolkit_VERSION MATCHES "^12\\.8")
    include(cuda/12.8/CMakeLists.txt)
    message(STATUS "Building hipsparse-test executable using CUDAToolkit_VERSION ${CUDAToolkit_VERSION}")
  elseif(CUDAToolkit_VERSION MATCHES "^13\\.")
    include(cuda/13/CMakeLists.txt)
    message(STATUS "Building hipsparse-test executable using CUDAToolkit_VERSION ${CUDAToolkit_VERSION}")
  else()
    message(WARNING "Testing on CUDAToolkit_VERSION ${CUDAToolkit_VERSION} not explicitly supported; no test data will be generated.")
  endif()
endif()

set(_hipsparse_test_sources "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gtest_main.cpp")
foreach(_src IN LISTS HIPSPARSE_TEST_SOURCES)
  list(APPEND _hipsparse_test_sources "${CMAKE_CURRENT_SOURCE_DIR}/${_src}")
endforeach()
target_sources(hipsparse-test PRIVATE ${_hipsparse_test_sources})

target_compile_definitions(hipsparse-test PRIVATE GOOGLE_TEST)

target_link_libraries(hipsparse-test PRIVATE
    hipsparse::clients-common
    GTest::gtest
)

# GTest requires pthread on Unix platforms
if(NOT WIN32)
    target_link_libraries(hipsparse-test PRIVATE pthread)
endif()

if(HIPSPARSE_ENABLE_OPENMP)
    target_link_libraries(hipsparse-test PRIVATE OpenMP::OpenMP_CXX)
endif()

if(HIPSPARSE_ENABLE_CUDA)
    target_compile_definitions(hipsparse-test PRIVATE __HIP_PLATFORM_NVIDIA__)
    target_link_libraries(hipsparse-test PRIVATE CUDA::cudart CUDA::cuda_driver)
endif()

set_target_properties(hipsparse-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging"
)

# Build mtx2csr converter
if(NOT CONVERT_SOURCE)
    set(CONVERT_SOURCE "${PROJECT_SOURCE_DIR}/deps/convert.cpp")
endif()

add_executable(mtx2csr)
target_sources(mtx2csr PRIVATE "${CONVERT_SOURCE}")
target_compile_options(mtx2csr PRIVATE -O3)
if(NOT WIN32)
    target_link_options(mtx2csr PRIVATE -Wl,--build-id=sha1)
endif()
set_target_properties(mtx2csr PROPERTIES
    OUTPUT_NAME "mtx2csr.exe"
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients"
)

set(HIPSPARSE_CLIENTMATRICES "${PROJECT_SOURCE_DIR}/cmake/hipsparse_clientmatrices.cmake")

set(HIPSPARSE_CONVERT "${PROJECT_BINARY_DIR}/clients/hipsparse_mtx2csr")
add_custom_command(
    OUTPUT "${HIPSPARSE_CONVERT}"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mtx2csr> "${HIPSPARSE_CONVERT}"
    DEPENDS mtx2csr
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(HIPSPARSE_COMMON "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_common.yaml")
add_custom_command(
    OUTPUT "${HIPSPARSE_COMMON}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_common.yaml" "${HIPSPARSE_COMMON}"
    DEPENDS "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_common.yaml"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(HIPSPARSE_TEMPLATE "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_template.yaml")
add_custom_command(
    OUTPUT "${HIPSPARSE_TEMPLATE}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_template.yaml" "${HIPSPARSE_TEMPLATE}"
    DEPENDS "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_template.yaml"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

set(HIPSPARSE_GENTEST "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_gentest.py")
add_custom_command(
    OUTPUT "${HIPSPARSE_GENTEST}"
    COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py" "${HIPSPARSE_GENTEST}"
    DEPENDS "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_custom_target(hipsparse-common
    DEPENDS "${HIPSPARSE_COMMON}" "${HIPSPARSE_TEMPLATE}" "${HIPSPARSE_CLIENTMATRICES}" "${HIPSPARSE_GENTEST}" "${HIPSPARSE_CONVERT}"
)

# Prepare testing data
set(HIPSPARSE_TEST_DATA "${PROJECT_BINARY_DIR}/clients/staging/hipsparse_test.data")
add_custom_command(OUTPUT "${HIPSPARSE_TEST_DATA}"
                   COMMAND ${Python3_EXECUTABLE} "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py" -m ${PROJECT_BINARY_DIR}/clients/matrices -I "${PROJECT_SOURCE_DIR}/clients/include" "${HIPSPARSE_TEST_YAML_INCLUDE}" -o "${HIPSPARSE_TEST_DATA}"
                   DEPENDS "${PROJECT_SOURCE_DIR}/clients/common/hipsparse_gentest.py" "${HIPSPARSE_TEST_YAML_INCLUDE}" "${PROJECT_SOURCE_DIR}/clients/include/hipsparse_common.yaml" ${HIPSPARSE_TEST_YAMLS}
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_target(hipsparse-test-data
  DEPENDS "${HIPSPARSE_TEST_DATA}" )

add_dependencies(hipsparse-test hipsparse-test-data hipsparse-common)


rocm_install(TARGETS hipsparse-test COMPONENT tests)
rocm_install(FILES ${HIPSPARSE_TEST_DATA} DESTINATION "${CMAKE_INSTALL_DATADIR}/hipsparse/test" COMPONENT tests)



# Only when both test_categories.yaml and shared/ctest exist (full rocm-libraries tree).
set(HIPSPARSE_CTEST_INSTALL_FILE "")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml")
  if(HIPSPARSE_HAS_CTEST_CATEGORIES)
    message(STATUS "hipSPARSE: YAML-based test categorization")
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
    # Install-time CTestTestfile for TheRock: relative paths so ctest can run from installed location
    set(HIPSPARSE_CTEST_INSTALL_FILE "${CMAKE_CURRENT_BINARY_DIR}/install_CTestTestfile.cmake")
    file(WRITE "${HIPSPARSE_CTEST_INSTALL_FILE}"
[=[
# This is a test file generated by hipSPARSE for install time.
# Tests are defined with relative paths to work in the installed location.
]=]
    )
    apply_test_category_labels(
      hipsparse-test
      "${CMAKE_CURRENT_SOURCE_DIR}/test_categories.yaml"
      "${PROJECT_BINARY_DIR}/staging"
      "${HIPSPARSE_CTEST_INSTALL_FILE}"
    )
  else()
    message(STATUS "hipSPARSE: shared/ctest not found (e.g. standalone checkout), skipping CTest categories")
  endif()
else()
  message(STATUS "hipSPARSE: No test_categories.yaml (CTest categories not used)")
endif()

if(HIPSPARSE_CTEST_INSTALL_FILE)
    rocm_install(
      FILES "${HIPSPARSE_CTEST_INSTALL_FILE}"
      DESTINATION "${CMAKE_INSTALL_BINDIR}/hipsparse"
      COMPONENT tests
      RENAME "CTestTestfile.cmake"
  )
endif()

# Windows-specific DLL copying
if(WIN32)
    # For now adding in all .dll as dependency chain is not cmake based on win32
    file(GLOB third_party_dlls
        LIST_DIRECTORIES OFF
        CONFIGURE_DEPENDS
        $ENV{HIP_DIR}/bin/*.dll
        $ENV{HIP_DIR}/bin/hipinfo.exe
        ${ROCSPARSE_PATH}/bin/rocsparse.dll
        ${PROJECT_SOURCE_DIR}/rtest.*
        C:/Windows/System32/libomp140*.dll
    )
    foreach(file_i ${third_party_dlls})
        add_custom_command(TARGET hipsparse-test POST_BUILD 
            COMMAND ${CMAKE_COMMAND} ARGS -E copy ${file_i} ${PROJECT_BINARY_DIR}/clients/staging/)
    endforeach(file_i)
endif()
