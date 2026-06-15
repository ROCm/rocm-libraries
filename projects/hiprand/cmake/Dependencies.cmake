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
    find_package(${repo_name} ${MIN_ROCRAND_PACKAGE_VERSION} CONFIG QUIET PATHS "${ROCM_ROOT}/lib/cmake/${repo_name}")

    if(NOT ${${repo_name}_FOUND})
      message(STATUS "No existing ${repo_name} package meeting the minimum version requirement (${MIN_ROCRAND_PACKAGE_VERSION}) was found. Falling back to downloading it.")
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

include(FetchContent)

# NOTE: HIPCC includes the default ROCm install directory as a system include directory (-isystem) for header files.
#       This makes it possible that different rocRAND headers are found than those of the package found with
#       find_package. CMake option CMAKE_NO_SYSTEM_FROM_IMPORTED can be used to change the -isystem to -I and to
#       workaround this problem.
if (NOT BUILD_WITH_LIB STREQUAL "CUDA")
    if (NOT ROCRAND_PATH STREQUAL "")
        # Search manually-specified rocRAND path.
        # This assumes that there is no system-installed rocRAND or that CMAKE_NO_SYSTEM_FROM_IMPORTED is ON.
        find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_PATH} NO_DEFAULT_PATH)
    else()
        fetch_dep(ROCRAND_FETCH_METHOD rocrand ROCRAND_PATH ROCM_DEP_RELEASE_BRANCH)

        if(${ROCRAND_FETCH_METHOD} STREQUAL "DOWNLOAD" OR ${ROCRAND_FETCH_METHOD} STREQUAL "MONOREPO")
            # The fetch_dep call above should have downloaded/located the source. We just need to make it available.
            message(STATUS "Configuring rocRAND")

            # Install rocRAND.
            # This assumes that there is no system-installed rocRAND or that CMAKE_NO_SYSTEM_FROM_IMPORTED is ON.
            FetchContent_Declare(
                rocrand
                SOURCE_DIR    ${ROCRAND_PATH}
                # ${ROCRAND_ROOT}
                INSTALL_DIR   ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand
                CMAKE_ARGS    -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                LOG_CONFIGURE TRUE
                LOG_BUILD     TRUE
                LOG_INSTALL   TRUE
            )
            FetchContent_MakeAvailable(rocrand)

            if (NOT TARGET roc::rocrand)
                add_library(roc::rocrand ALIAS rocrand)
            endif()
        endif()
    endif()
    get_target_property(ROCRAND_LINK_LIBRARIES roc::rocrand INTERFACE_LINK_LIBRARIES)
    string(FIND "${ROCRAND_LINK_LIBRARIES}" "TBB::tbb" ROCRAND_REQUIRES_TBB)
    if (ROCRAND_REQUIRES_TBB GREATER_EQUAL 0)
      message(STATUS "The found version of rocRAND requires TBB (Thread Building Blocks)")
      find_package(TBB REQUIRED)
    endif()
endif()

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
  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
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
