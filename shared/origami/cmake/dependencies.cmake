# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Dependencies

# Git
find_package(Git REQUIRED)

# Workaround until hcc & hip cmake modules fixes symlink logic in their config files.
# (Thanks to rocBLAS devs for finding workaround for this problem!)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

# ROCm cmake package
find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET) # First version with Sphinx doc gen improvement
if(NOT ROCM_FOUND)
  set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
  set(rocm_cmake_tag "rocm-6.4.1" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(NOT status_code EQUAL 0)
    message(FATAL_ERROR "error: downloading
    'https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}
    ")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})
  execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag})
  execute_process(COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
                  WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCmCMakeBuildTools 0.6 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds OPTIONAL)
include(ROCMClients)
include(ROCMHeaderWrapper)
