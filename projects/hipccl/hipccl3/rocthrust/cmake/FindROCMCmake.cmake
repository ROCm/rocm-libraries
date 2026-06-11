# ########################################################################
# Copyright 2021-2026 Advanced Micro Devices, Inc.
# ########################################################################

if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG QUIET PATHS "${ROCM_ROOT}")
endif()
if(NOT ROCmCMakeBuildTools_FOUND)
  message(STATUS "ROCm CMake not found. Fetching...")
  # We don't really want to consume the build and test targets of ROCm CMake.
  # CMake 3.18 allows omitting them, even though there's a CMakeLists.txt in source root.
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(SOURCE_SUBDIR_ARG SOURCE_SUBDIR "DISABLE ADDING TO BUILD")
  else()
    set(SOURCE_SUBDIR_ARG)
  endif()
  include(cmake/FetchContentIsolated.cmake)
  fetch_content_isolated(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG        rocm-6.4.4
    ${SOURCE_SUBDIR_ARG}
  )
  execute_process(
    WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} ${rocm-cmake_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=.
  )
  execute_process(
    WORKING_DIRECTORY ${rocm-cmake_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${rocm-cmake_SOURCE_DIR} --target install
  )
  find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
else()
  find_package(ROCmCMakeBuildTools 0.7.3 CONFIG REQUIRED PATHS "${ROCM_ROOT}")
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)