# MIT License
#
# Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

# NOTE: HIPCC includes the default ROCm install directory as a system include directory (-isystem) for header files.
#       This makes it possible that different rocRAND headers are found than those of the package found with
#       find_package. CMake option CMAKE_NO_SYSTEM_FROM_IMPORTED can be used to change the -isystem to -I and to
#       workaround this problem.
if (NOT BUILD_WITH_LIB STREQUAL "CUDA")
    if (NOT ROCRAND_PATH STREQUAL "")
        # Search manually-specified rocRAND path.
        # This assumes that there is no system-installed rocRAND or that CMAKE_NO_SYSTEM_FROM_IMPORTED is ON.
        find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_PATH} NO_DEFAULT_PATH)
    elseif (DOWNLOAD_ROCRAND)
        # Download and install rocRAND.
        # This assumes that there is no system-installed rocRAND or that CMAKE_NO_SYSTEM_FROM_IMPORTED is ON.
        download_project(
                PROJ rocrand
                GIT_REPOSITORY https://github.com/ROCmSoftwarePlatform/rocRAND.git
                GIT_TAG develop
                GIT_SHALLOW TRUE
                INSTALL_DIR ${ROCRAND_ROOT}
                CMAKE_ARGS -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                LOG_DOWNLOAD TRUE
                LOG_CONFIGURE TRUE
                LOG_BUILD TRUE
                LOG_INSTALL TRUE
                BUILD_PROJECT TRUE
                UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
        )
        # search only the install location of the downloaded rocrand
        find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_ROOT} NO_DEFAULT_PATH)
    else ()
        # If neither alternative is specified, follow default search paths which include the default install location.
        find_package(rocrand REQUIRED CONFIG)
    endif ()
    get_target_property(ROCRAND_LINK_LIBRARIES roc::rocrand INTERFACE_LINK_LIBRARIES)
    string(FIND "${ROCRAND_LINK_LIBRARIES}" "TBB::tbb" ROCRAND_REQUIRES_TBB)
    if (ROCRAND_REQUIRES_TBB GREATER_EQUAL 0)
      message(STATUS "The found version of rocRAND requires TBB (Thread Building Blocks)")
      find_package(TBB REQUIRED)
    endif()
endif ()

# Fortran Wrapper
if(BUILD_FORTRAN_WRAPPER)
    enable_language(Fortran)
endif()

# Test dependencies
if(BUILD_TEST)
  # NOTE: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
  #
  # FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
  #
  # GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
  #        will be duplicately defined.
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    list( APPEND CMAKE_PREFIX_PATH ${GTEST_ROOT} )
    find_package(GTest CONFIG REQUIRED PATHS ${GTEST_ROOT})
  endif()
endif()
