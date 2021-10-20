# ########################################################################
# Copyright 2021 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)

# By default, rocm software stack is expected at /opt/rocm
# set environment variable ROCM_PATH to change location
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm)
endif()

find_package(ROCM 0.6 CONFIG QUIET PATHS ${ROCM_PATH} /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  set(rocm_cmake_url "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip")
  set(rocm_cmake_path "${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}")
  set(rocm_cmake_archive "${rocm_cmake_path}.zip")
  file(DOWNLOAD "${rocm_cmake_url}" "${rocm_cmake_archive}" STATUS status LOG log)

  list(GET status 0 status_code)
  list(GET status 1 status_string)

  if(status_code EQUAL 0)
    message(STATUS "downloading... done")
  else()
    message(FATAL_ERROR "error: downloading\n'${rocm_cmake_url}' failed
    status_code: ${status_code}
    status_string: ${status_string}
    log: ${log}\n")
  endif()

  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzvf "${rocm_cmake_archive}"
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})
  execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
  execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
    WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

  find_package( ROCM 0.6 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()

