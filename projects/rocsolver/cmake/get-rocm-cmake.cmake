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
    # Corresponds to the therock-7.13 tag. Bump at each ROCm release cut.
    # grep "pinned-dep" to find every pin that needs bumping.
    set(rocm_cmake_tag "10155d7272ea1bf79f6b5a9dbc339657af1aa372" CACHE STRING "rocm-cmake commit to download (therock-7.13)")
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
