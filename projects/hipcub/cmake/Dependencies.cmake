# MIT License
#
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

# ###########################
# hipCUB dependencies
# ###########################

# NOTE1: the reason we don't scope global state meddling using add_subdirectory
#        is because CMake < 3.24 lacks CMAKE_FIND_PACKAGE_TARGETS_GLOBAL which
#        would promote IMPORTED targets of find_package(CONFIG) to be visible
#        by other parts of the build. So we save and restore global state.
#
# NOTE2: We disable the ROCMChecks.cmake warning noting that we meddle with
#        global state. This is consequence of abusing the CMake CXX language
#        which HIP piggybacks on top of. This kind of HIP support has one chance
#        at observing the global flags, at the find_package(HIP) invocation.
#        The device compiler won't be able to pick up changes after that, hence
#        the warning.
#
# NOTE3: hipCUB and rocPRIM share CMake options for building tests, benchmarks
#        and examples. Until that's not fixed, we have to save/restore them.
set(USER_CXX_FLAGS ${CMAKE_CXX_FLAGS})
if(DEFINED BUILD_SHARED_LIBS)
  set(USER_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
endif()
set(USER_ROCM_WARN_TOOLCHAIN_VAR ${ROCM_WARN_TOOLCHAIN_VAR})

set(ROCM_WARN_TOOLCHAIN_VAR OFF CACHE BOOL "")
# Turn off warnings and errors for all warnings in dependencies
separate_arguments(CXX_FLAGS_LIST NATIVE_COMMAND ${CMAKE_CXX_FLAGS})
list(REMOVE_ITEM CXX_FLAGS_LIST /WX -Werror -Werror=pendantic -pedantic-errors)
if(MSVC)
  list(FILTER CXX_FLAGS_LIST EXCLUDE REGEX "/[Ww]([0-4]?)(all)?") # Remove MSVC warning flags
  list(APPEND CXX_FLAGS_LIST /w)
else()
  list(FILTER CXX_FLAGS_LIST EXCLUDE REGEX "-W(all|extra|everything)") # Remove GCC/LLVM flags
  list(APPEND CXX_FLAGS_LIST -w)
endif()
list(JOIN CXX_FLAGS_LIST " " CMAKE_CXX_FLAGS)
# Don't build client dependencies as shared
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Global flag to cause add_library() to create shared libraries if on." FORCE)

foreach(SHARED_OPTION BUILD_TEST BUILD_BENCHMARK BUILD_EXAMPLE)
  set(USER_${SHARED_OPTION} ${${SHARED_OPTION}})
  set(${SHARED_OPTION} OFF)
endforeach()

include(FetchContent)

# This function checks to see if the download branch given by "branch" exists in the repository.
# It does so using the git ls-remote command.
# If the branch cannot be found, the variable described by "branch" is changed to "develop" in the host scope.
function(find_download_branch git_path branch)
  set(branch_value ${${branch}})
  execute_process(COMMAND ${git_path} "ls-remote" "https://github.com/ROCm/rocm-libraries.git" "refs/heads/${branch_value}" RESULT_VARIABLE ret_code OUTPUT_VARIABLE output)

  if(NOT ${ret_code} STREQUAL "0")
    message(WARNING "Unable to check if release branch exists, defaulting to the develop branch.")
    set(${branch} "develop" PARENT_SCOPE)
  else()
    if(${output})
      string(STRIP ${output} output)
    endif()

    if(NOT (${output} MATCHES "[\t ]+refs/heads/${branch_value}(\n)?$"))
      message(WARNING "Unable to locate requested release branch \"${branch_value}\" in repository. Defaulting to the develop branch.")
      set(${branch} "develop" PARENT_SCOPE)
    else()
      message(STATUS "Found release branch \"${branch_value}\" in repository.")
    endif()
  endif()
endfunction()

function(check_git_version git_path)
  execute_process(COMMAND ${git_path} "--version" OUTPUT_VARIABLE git_version_output)
  string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)" GIT_VERSION_STRING ${git_version_output})
  if(DEFINED CMAKE_MATCH_0)
    set(GIT_VERSION ${CMAKE_MATCH_0} PARENT_SCOPE)
  else()
    set(GIT_VERSION "" PARENT_SCOPE)
  endif()
endfunction()

# This function fetches repository "repo_name" using the method specified by "method".
# The result is stored in the parent scope version of "repo_path".
# It does not build the repo.
function(fetch_dep method repo_name repo_path download_branch)
  set(method_value ${${method}})

  # Since the monorepo is large, we want to avoid downloading the whole thing if possible.
  # We can do this if we have access to git's sparse-checkout functionality, which was added in git 2.25.
  # On some Linux systems (eg. Ubuntu), the git in /usr/bin tends to be newer than the git in /usr/local/bin,
  # and the latter is what gets picked up by find_package(Git), since it's what's in PATH.
  # Check for a git binary in /usr/bin first, then if git < 2.25 is not found, use find_package(Git) to search
  # other locations.
  if (NOT(GIT_PATH))
    message(STATUS "Checking git version")
    set(GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT 2.25)

    find_program(find_result git PATHS /usr/bin NO_DEFAULT_PATH)
    if(NOT (${find_result} STREQUAL "find_result-NOTFOUND"))
      set(GIT_PATH ${find_result} CACHE INTERNAL "Path to the git executable")
      check_git_version(${GIT_PATH})
    endif()

    if(NOT GIT_VERSION OR "${GIT_VERSION}" LESS ${GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT})
      find_package(Git QUIET)
      if(GIT_FOUND)
        set(GIT_PATH ${GIT_EXECUTABLE} CACHE INTERNAL "Path to the git executable")
        check_git_version(${GIT_PATH})
      endif()
    endif()

    if(NOT GIT_VERSION OR "${GIT_VERSION}" LESS ${GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT})
      set(USE_SPARSE_CHECKOUT "OFF" CACHE INTERNAL "Records whether git supports sparse checkout functionality")
    else()
      set(USE_SPARSE_CHECKOUT "ON" CACHE INTERNAL "Records whether git supports sparse checkout functionality")
    endif()

    if(NOT GIT_VERSION)
      # Warn the user that we were unable to find git. This will only actually be a problem if we use one of the
      # fetch methods (download, or monorepo with dependency not present) that requires it. If we end up running
      # into one of those scenarios, a fatal error will be issued at that point.
      message(WARNING "Unable to find git.")
    else()
      message(STATUS "Found git at: ${GIT_PATH}, version: ${GIT_VERSION}")
    endif()
  endif()

  if(${method_value} STREQUAL "PACKAGE")
    message(STATUS "Searching for ${repo_name} package")

    # Add default install location for WIN32 and non-WIN32 as hint
    find_package(${repo_name} ${MIN_ROCPRIM_PACKAGE_VERSION} CONFIG QUIET PATHS "${ROCM_ROOT}/lib/cmake/rocprim")

    if(NOT ${${repo_name}_FOUND})
      message(STATUS "No existing ${repo_name} package meeting the minimum version requirement (${MIN_ROCPRIM_PACKAGE_VERSION}) was found. Falling back to downloading it.")
      # update local and parent variable values
      set(${method} "DOWNLOAD" PARENT_SCOPE)
      set(method_value "DOWNLOAD")
    else()
      message(STATUS "Package found (${${repo_name}_DIR})")
    endif()

  elseif(${method_value} STREQUAL "MONOREPO")
    message(STATUS "Searching for ${repo_name} in the parent monorepo directory")

    # Check if this looks like a monorepo checkout
    find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${repo_name}/" NO_CACHE NO_DEFAULT_PATH)

    # If not, see if the local monorepo is a sparse-checkout.
    # If it is a sparse-checkout, try to add the dependency to the sparse-checkout list.
    # If it's not a sparse-checkout (or adding to the sparse-checkout list fails), fallback to downloading the dependency.
    if(${found_path} STREQUAL "found_path-NOTFOUND")
      set(FALLBACK_TO_DOWNLOAD ON)
      message(WARNING "Unable to locate ${repo_name} in parent monorepo (it's not at \"${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${repo_name}/\").")
      message(STATUS "Checking if local monorepo is a sparse-checkout that we can add ${repo_name} to.")
      if(NOT(GIT_PATH))
        message(FATAL_ERROR "Git could not be found on the system. Since ${repo_name} could not be found in the local monorepo, git is required to download it.")
      endif()

      if(USE_SPARSE_CHECKOUT)
        execute_process(COMMAND ${GIT_PATH} "sparse-checkout" "list" OUTPUT_VARIABLE sparse_list ERROR_VARIABLE git_error RESULT_VARIABLE git_result
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)

        if(NOT(git_result EQUAL 0) OR git_error)
          message(STATUS "The local monorepo does not appear to be a sparse-checkout.")
        else()
          message(STATUS "The local monorepo appears to be a sparse checkout. Attempting to add \"projects/${repo_name}\" to the checkout list.")
          # Check if the dependency is already present in the checkout list.
          # Git lists sparse checkout directories each on a separate line.
          # Take care not to match something in the middle of a path, eg. "other_dir/projects/${repo_name}/sub_dir".
          string(REGEX MATCH "(^|\n)projects/${repo_name}($|\n)" find_result ${sparse_list})
          if(find_result)
            message(STATUS "Found existing entry for \"projects/${repo_name}\" in sparse-checkout list - has the directory structure been modified?")
          else()
            # Add project/${repo_name} to the sparse checkout
            execute_process(COMMAND ${GIT_PATH} "sparse-checkout" "add" "projects/${repo_name}" RESULT_VARIABLE sparse_checkout_result
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)
            # Note that in this case, we are forced to checkout the same branch that the sparse-checkout was created with.
            execute_process(COMMAND ${GIT_PATH} "checkout" RESULT_VARIABLE checkout_result
                            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../)

            if(sparse_checkout_result EQUAL 0 AND checkout_result EQUAL 0)
              message(STATUS "Added new checkout list entry.")
              set(FALLBACK_TO_DOWNLOAD OFF)
            else()
              message(STATUS "Unable to add new checkout list entry.")
            endif()
            # Save the monorepo path in the parent scope
            set(${repo_path} "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${repo_name}" PARENT_SCOPE)
          endif()
        endif()
      else()
        message(STATUS "The version of git installed on the system (${GIT_VERSION}) does not support sparse-checkout.")
      endif()

      if (FALLBACK_TO_DOWNLOAD)
        message(WARNING "Unable to locate/fetch dependency ${repo_name} from monorepo. Falling back to downloading it.")
        # update local and parent variable values
        set(${method} "DOWNLOAD" PARENT_SCOPE)
        set(method_value "DOWNLOAD")
      endif()

    else()
      message(STATUS "Found ${repo_name} at ${found_path}")

      # Save the monorepo path in the parent scope
      set(${repo_path} ${found_path} PARENT_SCOPE)
    endif()
  endif()

  if(${method_value} STREQUAL "DOWNLOAD")
    if(NOT DEFINED GIT_PATH)
      message(FATAL_ERROR "Git could not be found on the system. Git is required for downloading ${repo_name}.")
    endif()

    message(STATUS "Checking if repository contains requested branch ${${download_branch}}")
    find_download_branch(${GIT_PATH} ${download_branch})
    set(download_branch_value ${${download_branch}})

    message(STATUS "Downloading ${repo_name} from https://github.com/ROCm/rocm-libraries.git")
    if(${USE_SPARSE_CHECKOUT})
      # In this case, we have access to git sparse-checkout.
      # Check if the dependency has already been downloaded in the past:
      find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src/" NO_CACHE NO_DEFAULT_PATH)
      if(${found_path} STREQUAL "found_path-NOTFOUND")
        # First, git clone with options "--no-checkout" and "--filter=tree:0" to prevent files from being pulled immediately.
        # Use option "--depth=1" to avoid downloading past commit history.
        execute_process(COMMAND ${GIT_PATH} clone --branch ${download_branch_value} --no-checkout --depth=1 --filter=tree:0 https://github.com/ROCm/rocm-libraries.git ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src)

        # Next, use git sparse-checkout to ensure we only pull the directory containing the desired repo.
        execute_process(COMMAND ${GIT_PATH} sparse-checkout init --cone
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src)

        execute_process(COMMAND ${GIT_PATH} sparse-checkout set projects/${repo_name}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src)

        # Finally, download the files using git checkout.
        execute_process(COMMAND ${GIT_PATH} checkout ${download_branch_value}
                        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src)

        message(STATUS "${repo_name} download complete")
      else()
        message("Found previously downloaded directory, skipping download step.")
      endif()

      # Save the downloaded path in the parent scope
      set(${repo_path} "${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src/projects/${repo_name}" PARENT_SCOPE)
    else()
      # In this case, we do not have access to sparse-checkout, so we need to download the whole monorepo.
      # Check if the monorepo has already been downloaded to satisfy a previous dependency
      find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_BINARY_DIR}/monorepo-src/" NO_CACHE NO_DEFAULT_PATH)
      if(${found_path} STREQUAL "found_path-NOTFOUND")
        # Warn the user that this will take some time.
        message(WARNING "The detected version of git (${GIT_VERSION}) is older than 2.25 and does not provide sparse-checkout functionality. Falling back to checking out the whole rocm-libraries repository (this may take a long time).")
        # Avoid downloading anything related to branches other than the target branch (--single-branch), and avoid any past commit history information (--depth=1)
        execute_process(COMMAND ${GIT_PATH} clone --single-branch --branch=${download_branch_value} --depth=1 https://github.com/ROCm/rocm-libraries.git ${CMAKE_CURRENT_BINARY_DIR}/monorepo-src)
        message(STATUS "rocm-libraries download complete")
      else()
        message("Found previously downloaded directory, skipping download step.")
      endif()

      # Save the downloaded path in the parent scope
      set(${repo_path} "${CMAKE_CURRENT_BINARY_DIR}/monorepo-src/projects/${repo_name}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

# Test dependencies
if(USER_BUILD_TEST)
  # NOTE1: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
  #
  # FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
  #
  # GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
  #
  # NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
  #        will be duplicately defined.
  #
  # NOTE3: The following snippet first tries to find Google Test binary either in MODULE or CONFIG modes.
  #        If neither succeeds it goes on to import Google Test into this build either from a system
  #        source package (apt install googletest on Ubuntu 18.04 only) or GitHub and defines the MODULE
  #        mode targets. Otherwise if MODULE or CONFIG succeeded, then it prints the result to the
  #        console via a non-QUIET find_package call and if CONFIG succeeded, creates ALIAS targets
  #        with the MODULE IMPORTED names.
  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()
  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    option(BUILD_GTEST "Builds the googletest subproject" ON)
    option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
    option(INSTALL_GTEST "Enable installation of googletest." OFF)
    if(EXISTS /usr/src/googletest AND NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
      FetchContent_Declare(
        googletest
        SOURCE_DIR /usr/src/googletest
      )
    else()
      message(STATUS "Google Test not found. Fetching...")
      FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.11.0
      )
    endif()
    FetchContent_MakeAvailable(googletest)
    add_library(GTest::GTest ALIAS gtest)
    add_library(GTest::Main  ALIAS gtest_main)
  else()
    find_package(GTest REQUIRED)
    if(TARGET GTest::gtest_main AND NOT TARGET GTest::Main)
      add_library(GTest::GTest ALIAS GTest::gtest)
      add_library(GTest::Main  ALIAS GTest::gtest_main)
    endif()
  endif()
endif(USER_BUILD_TEST)

if(USER_BUILD_BENCHMARK)
  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
    find_package(benchmark CONFIG QUIET)
  endif()
  if(NOT TARGET benchmark::benchmark)
    message(STATUS "Google Benchmark not found. Fetching...")
    option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
    option(BENCHMARK_ENABLE_INSTALL "Enable installation of benchmark." OFF)
    FetchContent_Declare(
      googlebench
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG        v1.8.0
    )
    FetchContent_MakeAvailable(googlebench)
    if(NOT TARGET benchmark::benchmark)
      add_library(benchmark::benchmark ALIAS benchmark)
    endif()
  else()
    find_package(benchmark CONFIG REQUIRED)
  endif()
endif(USER_BUILD_BENCHMARK)

# CUB (only for CUDA platform)
if(HIP_COMPILER STREQUAL "nvcc")
  set(CCCL_MINIMUM_VERSION 2.8.2)
  if(NOT DOWNLOAD_CUB)
    find_package(CCCL ${CCCL_MINIMUM_VERSION} CONFIG)
  endif()

  if (NOT CCCL_FOUND)
    message(STATUS "CCCL not found, downloading and extracting CCCL ${CCCL_MINIMUM_VERSION}")
    file(DOWNLOAD https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip
                  ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
         STATUS cccl_download_status LOG cccl_download_log)

    list(GET cccl_download_status 0 cccl_download_error_code)
    if(cccl_download_error_code)
      message(FATAL_ERROR "Error: downloading "
              "https://github.com/NVIDIA/cccl/archive/refs/tags/v${CCCL_MINIMUM_VERSION}.zip failed "
              "error_code: ${cccl_download_error_code} "
              "log: ${cccl_download_log}")
    endif()

    if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
      file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip)
    else()
      execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      RESULT_VARIABLE cccl_unpack_error_code)
      if(cccl_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION}.zip failed")
      endif()
    endif()

    find_package(CCCL ${CCCL_MINIMUM_VERSION} CONFIG REQUIRED NO_DEFAULT_PATH
                 PATHS ${CMAKE_CURRENT_BINARY_DIR}/cccl-${CCCL_MINIMUM_VERSION})
  endif()
else()
  # rocPRIM (only for ROCm platform)
  fetch_dep(ROCPRIM_FETCH_METHOD rocprim ROCPRIM_PATH ROCM_DEP_RELEASE_BRANCH)

  if(${ROCPRIM_FETCH_METHOD} STREQUAL "DOWNLOAD" OR ${ROCPRIM_FETCH_METHOD} STREQUAL "MONOREPO")
    # The fetch_dep call above should have downloaded/located the source. We just need to make it available.
    message(STATUS "Configuring rocPRIM")
    FetchContent_Declare(
      prim
      SOURCE_DIR    ${ROCPRIM_PATH}
      INSTALL_DIR   ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
      CMAKE_ARGS    -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
      LOG_CONFIGURE TRUE
      LOG_BUILD     TRUE
      LOG_INSTALL   TRUE
    )
    FetchContent_MakeAvailable(prim)
    if(NOT TARGET roc::rocprim)
      add_library(roc::rocprim ALIAS rocprim)
    endif()
    if(NOT TARGET roc::rocprim_hip)
      add_library(roc::rocprim_hip ALIAS rocprim_hip)
    endif()
  endif()
endif()

foreach(SHARED_OPTION BUILD_TEST BUILD_BENCHMARK BUILD_EXAMPLE)
  set(${SHARED_OPTION} ${USER_${SHARED_OPTION}})
endforeach()

# Restore user global state
set(CMAKE_CXX_FLAGS ${USER_CXX_FLAGS})
if(DEFINED USER_BUILD_SHARED_LIBS)
  set(BUILD_SHARED_LIBS ${USER_BUILD_SHARED_LIBS})
else()
  unset(BUILD_SHARED_LIBS CACHE )
endif()
set(ROCM_WARN_TOOLCHAIN_VAR ${USER_ROCM_WARN_TOOLCHAIN_VAR} CACHE BOOL "")
