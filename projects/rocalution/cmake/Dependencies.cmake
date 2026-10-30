# ########################################################################
# Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# Dependencies

# Find OpenMP package
find_package(OpenMP)
if (NOT OPENMP_FOUND)
  message("-- OpenMP not found. Compiling WITHOUT OpenMP support.")
else()
  option(SUPPORT_OMP "Compile WITH OpenMP support." ON)
  if(NOT TARGET OpenMP::OpenMP_CXX)
    # cmake fix for cmake <= 3.9
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
  endif()
endif()

# MPI
set(MPI_HOME ${ROCALUTION_MPI_DIR})
find_package(MPI)
if (NOT MPI_FOUND)
  message("-- MPI not found. Compiling WITHOUT MPI support.")
  if (SUPPORT_MPI)
    message(FATAL_ERROR "Cannot build with MPI support.")
  endif()
else()
  option(SUPPORT_MPI "Compile WITH MPI support." OFF)
  if(NOT TARGET MPI::MPI_CXX)
    # cmake fix for cmake <= 3.9
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}")
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_DEFINITIONS "${MPI_CXX_COMPILE_DEFINITIONS}")
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES "")
    if(MPI_CXX_LINK_FLAGS)
      set_property(TARGET MPI::MPI_CXX APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${MPI_CXX_LINK_FLAGS}")
    endif()
    if(MPI_CXX_LIBRARIES)
      set_property(TARGET MPI::MPI_CXX APPEND PROPERTY INTERFACE_LINK_LIBRARIES "${MPI_CXX_LIBRARIES}")
    endif()
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}")
  endif()
endif()

# ROCm cmake package
option(ALLOW_FETCH_DEPS "Download rocm-cmake and GoogleTest if they are not found in the build environment" OFF)
# Included before setting FETCHCONTENT_FULLY_DISCONNECTED, because on CMake < 3.30 its option() clears the normal
# variable when FetchContent is first included under the CMP0077 OLD policy of clients/CMakeLists.txt
include(FetchContent)
if(NOT ALLOW_FETCH_DEPS)
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()

find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS ${CMAKE_PREFIX_PATH} ${ROCM_PATH} /opt/rocm)
if(NOT ROCmCMakeBuildTools_FOUND)
  if(NOT ALLOW_FETCH_DEPS)
    message(FATAL_ERROR "ROCmCMakeBuildTools >= 0.11.0 not found. Install the rocm-cmake package and pass its "
                        "install prefix with -DROCM_PATH or CMAKE_PREFIX_PATH, set ROCmCMakeBuildTools_DIR to "
                        "its share/rocmcmakebuildtools/cmake directory, or configure with -DALLOW_FETCH_DEPS=ON "
                        "to download it.")
  endif()
  message(STATUS "ROCmCMakeBuildTools not found. Fetching rocm-cmake because ALLOW_FETCH_DEPS is ON")
  FetchContent_Declare(
    rocm-cmake
    GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
    GIT_TAG b3959201846b9d9250a6f2a386ad43be04d1d051 # rocm-7.2.4
    SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
  )
  FetchContent_MakeAvailable(rocm-cmake)
  find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
include(ROCMHeaderWrapper)
