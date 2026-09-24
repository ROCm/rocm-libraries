# This finds the rocm-cmake project, and installs it if not found
# rocm-cmake contains common cmake code for rocm projects to help setup and install

# By default, rocm software stack is expected at /opt/rocm
# set environment variable ROCM_PATH to change location
if(NOT ROCM_PATH)
  set(ROCM_PATH /opt/rocm)
endif()

find_package(ROCmCMakeBuildTools QUIET PATHS ${ROCM_PATH})
if(NOT ROCmCMakeBuildTools_FOUND)
  find_package(ROCM 0.7.3 CONFIG QUIET PATHS ${ROCM_PATH}) # deprecated fallback
  if(NOT ROCM_FOUND)
    include(FetchContent)
    message(STATUS "ROCmCMakeBuildTools not found. Fetching...")
    # pinned-dep rocm-cmake: immutable commit (was the mutable "master" branch).
    # Fallback only, used when rocm-cmake isn't already installed at /opt/rocm, so the exact
    # version rarely matters. Pinned to a known-good rocm-cmake commit from the rocm-6.4.0 tag
    # for reproducible builds. Bump when the fallback actually needs a newer rocm-cmake.
    # grep "pinned-dep" to find all pins.
    set(rocm_cmake_tag "ecc716b97c2239cff00422ed7a43cd52a0839a0e" CACHE STRING "rocm-cmake commit to download (rocm-6.4.0)")
    FetchContent_Declare(
      rocm-cmake
      GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
      GIT_TAG ${rocm_cmake_tag}
      SOURCE_SUBDIR "DISABLE ADDING TO BUILD" # We don't want to consume rocm-cmake's own build/test targets.
    )
    FetchContent_MakeAvailable(rocm-cmake)
    find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
  endif()
endif()
