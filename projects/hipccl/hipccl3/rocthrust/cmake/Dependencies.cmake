# ########################################################################
# Copyright 2019 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
  message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# rocPRIM (https://github.com/ROCmSoftwarePlatform/rocPRIM)
message(STATUS "Downloading and building rocPRIM.")
set(ROCPRIM_ROOT ${CMAKE_CURRENT_BINARY_DIR}/rocPRIM CACHE PATH "")
download_project(
  PROJ                rocPRIM
  GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
  GIT_TAG             master
  INSTALL_DIR         ${ROCPRIM_ROOT}
  CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
  LOG_DOWNLOAD        TRUE
  LOG_CONFIGURE       TRUE
  LOG_BUILD           TRUE
  LOG_INSTALL         TRUE
  BUILD_PROJECT       TRUE
  UPDATE_DISCONNECTED TRUE
)
find_package(rocprim REQUIRED CONFIG PATHS ${ROCPRIM_ROOT})

# Test dependencies
if(BUILD_TEST)
  # Google Test (https://github.com/google/googletest)
  message(STATUS "Downloading and building GTest.")
  set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
  download_project(
    PROJ                googletest
    GIT_REPOSITORY      https://github.com/google/googletest.git
    GIT_TAG             release-1.8.1
    INSTALL_DIR         ${GTEST_ROOT}
    CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE
  )
  find_package(GTest REQUIRED)
endif()

# Find or download/install rocm-cmake project
find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(
    DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
    ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
  )
  list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
  if(rocm_cmake_download_error_code)
    message(FATAL_ERROR "Error: downloading "
      "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
      "error_code: ${rocm_cmake_download_error_code} "
      "log: ${rocm_cmake_download_log} "
    )
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE rocm_cmake_unpack_error_code
  )
  if(rocm_cmake_unpack_error_code)
    message(FATAL_ERROR "Error: unpacking  ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
  endif()
  find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
