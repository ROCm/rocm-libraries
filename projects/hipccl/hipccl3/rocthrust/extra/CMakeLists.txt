# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

# This project includes tests that should be run after
# rocThrust is installed from package or using `make install`
project(rocThrust_package_install_test CXX)

# CMake modules
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../cmake
  ${ROCM_PATH}/lib/cmake/hip /opt/rocm/lib/cmake/hip # FindHIP.cmake
  ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Detect compiler support for target ID
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
    OUTPUT_VARIABLE CXX_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

# Use target ID syntax if supported for AMDGPU_TARGETS
if(TARGET_ID_SUPPORT)
  set(AMDGPU_TARGETS gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack- CACHE STRING "List of specific machine types for library to target")
else()
  set(AMDGPU_TARGETS gfx803;gfx900;gfx906;gfx908 CACHE STRING "List of specific machine types for library to target")
endif()

# Verify that hcc or hipcc compiler is used on ROCM platform
include(VerifyCompiler)
include(DownloadProject)

# Download rocPRIM (only for ROCm platform)
if(NOT DEFINED ROCPRIM_ROOT)
  message(STATUS "Downloading and building rocPRIM.")
  set(ROCPRIM_ROOT "${CMAKE_CURRENT_BINARY_DIR}/rocprim" CACHE PATH "")
  download_project(
    PROJ                rocprim
    GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
    GIT_TAG             master
    INSTALL_DIR         ${ROCPRIM_ROOT}
    CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE
  )
endif()
find_package(rocprim REQUIRED CONFIG PATHS "${ROCPRIM_ROOT}")

# Find rocThrust
find_package(rocthrust REQUIRED CONFIG HINTS ${rocthrust_DIR} PATHS "/opt/rocm/rocthrust")

# Build CXX flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")

# Enable testing (ctest)
enable_testing()

function(add_rocthrust_test TEST_NAME TEST_SOURCES)
  list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
  add_executable(${TEST_TARGET} ${TEST_SOURCES})
  target_link_libraries(${TEST_TARGET}
    PRIVATE
      ${rocthrust_LIBRARIES} # rocthrust
  )
  if (NOT WIN32)
    foreach(amdgpu_target ${AMDGPU_TARGETS})
      target_link_libraries(${TEST_TARGET}
        PRIVATE
          --offload-arch=${amdgpu_target}
      )
  endif()
  endforeach()
  add_test(${TEST_NAME} ${TEST_TARGET})
endfunction()

# rocThrust package test
add_rocthrust_test("test_rocthrust_package" test_rocthrust_package.cpp)
