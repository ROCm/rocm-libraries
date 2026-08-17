# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install

# By default, rocm software stack is expected at /opt/rocm
# set environment variable ROCM_PATH to change location
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm)
endif()

find_package(ROCmCMakeBuildTools QUIET PATHS "${ROCM_PATH}")
if(NOT ROCmCMakeBuildTools_FOUND)
  find_package(ROCM 0.7.3 CONFIG QUIET PATHS "${ROCM_PATH}") # deprecated fallback
  if(NOT ROCM_FOUND)
    include(FetchContent)
    message(STATUS "ROCmCMakeBuildTools not found. Fetching...")
    set(rocm_cmake_tag "develop" CACHE STRING "rocm-cmake tag to download")
    FetchContent_Declare(
      rocm-cmake
      GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
      GIT_TAG        ${rocm_cmake_tag}
      SOURCE_SUBDIR "DISABLE_ADDING_TO_BUILD" # We don't really want to consume the build and test targets of ROCm CMake.
    )
    FetchContent_MakeAvailable(rocm-cmake)
    list(APPEND CMAKE_MODULE_PATH "${rocm-cmake_SOURCE_DIR}/share/rocmcmakebuildtools/cmake")
    find_package(ROCmCMakeBuildTools REQUIRED)
  endif()
endif()
