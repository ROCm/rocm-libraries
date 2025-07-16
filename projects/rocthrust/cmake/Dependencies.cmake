# ########################################################################
# Copyright 2019-2025 Advanced Micro Devices, Inc.
# ########################################################################

# ###########################
# rocThrust dependencies
# ###########################

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)
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

    if(NOT (${output} MATCHES "${branch_value}$"))
      message(WARNING "Unable to locate requested release branch \"${branch_value}\" in repository. Defaulting to the develop branch.")
      set(${branch} "develop" PARENT_SCOPE)
    else()
      message(STATUS "Found release branch \"${branch_value}\" in repository.")
    endif()
  endif()
endfunction()

# This function fetches repository "repo_name" using the method specified by "method".
# The result is stored in the parent scope version of "repo_path".
# It does not build the repo.
function(fetch_dep method repo_name repo_path download_branch)
  set(method_value ${${method}})

  if(${method_value} STREQUAL "PACKAGE")
    message(STATUS "Searching for ${repo_name} package")
    find_package(${repo_name} QUIET)

    if(NOT ${${repo_name}_FOUND})
      message(STATUS "No existing ${repo_name} package was found. Falling back to downloading it.")
      # update local and parent variable values
      set(${method} "DOWNLOAD" PARENT_SCOPE)
      set(method_value "DOWNLOAD")
    else()
      message(STATUS "Package found")
    endif()

  elseif(${method_value} STREQUAL "MONOREPO")
    message(STATUS "Searching for ${repo_name} in the parent monorepo directory")
    # Check if this looks like a monorepo checkout
    find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${repo_name}/" NO_CACHE NO_DEFAULT_PATH)

    if(${found_path} STREQUAL "found_path-NOTFOUND")
      message(STATUS "Unable to locate ${repo_name} in parent monorepo (it's not at \"${CMAKE_CURRENT_SOURCE_DIR}/../../projects/${repo_name}/\"). Falling back to downloading it.")
      # update local and parent variable values
      set(${method} "DOWNLOAD" PARENT_SCOPE)
      set(method_value "DOWNLOAD")
    else()
      message(STATUS "Found ${repo_name} at ${found_path}")
      # Save the monorepo path in the parent scope
      set(${repo_path} ${found_path} PARENT_SCOPE)
    endif()
  endif()

  if(${method_value} STREQUAL "DOWNLOAD")
    message(STATUS "Preparing to download ${repo_name}")
    message(STATUS "Checking git version")

    # Since the monorepo is large, we want to avoid downloading the whole thing if possible.
    # We can do this if we have access to git's sparse-checkout functionality, which was added in git 2.25.
    # On some Linux systems (eg. Ubuntu), the git in /usr/bin tends to be newer than the git in /usr/local/bin,
    # and the latter is what gets picked up by find_package(Git), since it's what's in PATH.
    # Check for a git binary in /usr/bin first, then if git < 2.25 is not found, use find_package(Git) to search
    # other locations.
    set(GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT 2.25)
    find_program(find_result git PATHS /usr/bin NO_DEFAULT_PATH)
    if(NOT (${find_result} STREQUAL "find_result-NOTFOUND"))
      set(GIT_PATH ${find_result})
      execute_process(COMMAND ${GIT_PATH} "--version" OUTPUT_VARIABLE git_version_output)
      string(REGEX MATCH "([0-9]+\.[0-9]+\.[0-9]+)" GIT_VERSION_STRING ${git_version_output})
    endif()

    if(NOT DEFINED GIT_PATH OR GIT_VERSION_STRING LESS ${GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT})
      find_package(Git)
      if(NOT GIT_FOUND)
        message(FATAL_ERROR "Git could not be found on the system. Git is required for downloading ${repo_name}.")
      endif()
      set(GIT_PATH ${GIT_EXECUTABLE})
    endif()

    message(STATUS "Checking if repository contains requested branch ${${download_branch}}")
    find_download_branch(${GIT_PATH} ${download_branch})
    set(download_branch_value ${${download_branch}})

    message(STATUS "Downloading ${repo_name} from https://github.com/ROCm/rocm-libraries.git")
    if(GIT_VERSION_STRING GREATER_EQUAL ${GIT_MIN_VERSION_FOR_SPARSE_CHECKOUT})
      # In this case, we have access to git sparse-checkout.
      # Check if the dependency has already been downloaded in the past:
      find_path(found_path NAMES "." PATHS "${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src/" NO_CACHE NO_DEFAULT_PATH)
      if(${found_path} STREQUAL "found_path-NOTFOUND")
        # First, git clone with options "--no-checkout" and "--filter=tree:0" to prevent files from being pulled immediately.
        # Use option "--depth=1" to avoid downloading past commit history.
        execute_process(COMMAND ${GIT_PATH} clone --branch ${download_branch_value} --no-checkout --depth=1 --filter=tree:0 https://github.com/ROCm/rocm-libraries.git ${CMAKE_CURRENT_BINARY_DIR}/${repo_name}-src)

        # Next, use git sparse-checkout to ensure we only pull the directory containing the desired repo.
        execute_process(COMMAND ${GIT_PATH} sparse-checkout set --cone projects/${repo_name}
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
        message(WARNING "The detected version of git (${GIT_VERSION_STRING}) is older than 2.25 and does not provide sparse-checkout functionality. Falling back to checking out the whole rocm-libraries repository (this may take a long time).")
        # Avoid downloading anything other than the specified branch (--single-branch), and avoid any past commit history information (--depth=1)
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

# Find/Get rocPRIM. This may require a download, depending on the fetch method.
fetch_dep(ROCPRIM_FETCH_METHOD rocprim ROCPRIM_PATH ROCM_DEP_RELEASE_BRANCH)

# If rocPRIM was found in the monorepo tree or was downloaded, we need to build it.
# Set up download_project to build from the existing rocprim directory at ${ROCPRIM_PATH}.
# Note that since we don't set any download-related options, nothing is actually downloaded here - just built.
if(${ROCPRIM_FETCH_METHOD} STREQUAL "MONOREPO" OR ${ROCPRIM_FETCH_METHOD} STREQUAL "DOWNLOAD")
  download_project(
    PROJ                rocprim
    SOURCE_DIR          ${ROCPRIM_PATH}
    INSTALL_DIR         ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim
    CMAKE_ARGS          -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    STATUS_MSG          "Building"
  )
  find_package(rocprim REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/deps/rocprim NO_DEFAULT_PATH)
endif()

# Test dependencies
if(BUILD_TEST OR BUILD_HIPSTDPAR_TEST)
  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
    # Google Test (https://github.com/google/googletest)
    find_package(GTest QUIET)
    find_package(TBB QUIET)
  else()
    message(STATUS "Force installing GTest.")
  endif()

  if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
    message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")

    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      GIT_SHALLOW         TRUE
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    find_package(GTest REQUIRED CONFIG PATHS ${GTEST_ROOT})
  endif()

  if (NOT TARGET TBB::tbb AND NOT TARGET tbb AND BUILD_HIPSTDPAR_TEST_WITH_TBB)
    message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
    set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "" FORCE)

    download_project(
      PROJ  TBB
      GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
      GIT_TAG             1c4c93fc5398c4a1acb3492c02db4699f3048dea # v2021.13.0
      INSTALL_DIR         ${TBB_ROOT}
      CMAKE_ARGS          -DCMAKE_CXX_COMPILER=g++ -DTBB_TEST=OFF -DTBB_BUILD=ON -DTBB_INSTALL=ON -DTBBMALLOC_PROXY_BUILD=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    find_package(TBB REQUIRED CONFIG PATHS ${TBB_ROOT})
  
  endif()

  # SQlite (for run-to-run bitwise-reproducibility tests)
  # Note: SQLite 3.36.0 enabled the backup API by default, which we need
  # for cache serialization.  We also want to use a static SQLite,
  # and distro static libraries aren't typically built
  # position-independent.
  if(DEFINED ENV{SQLITE_3_49_2_SRC_URL})
    set(SQLITE_3_49_2_SRC_URL_INIT $ENV{SQLITE_3_49_2_SRC_URL})
  else()
    set(SQLITE_3_49_2_SRC_URL_INIT https://sqlite.org/2025/sqlite-amalgamation-3490200.zip)
  endif()
    set(SQLITE_3_49_2_SRC_URL ${SQLITE_3_49_2_SRC_URL_INIT} CACHE STRING "Location of SQLite source code")
    set(SQLITE_SRC_3_43_2_SHA3_256 fad307cde789046256b4960734d7fec6b31db7f5dc8525474484885faf82866c CACHE STRING "SHA3-256 hash of SQLite source code")

    # embed SQLite
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      # use extract timestamp for fetched files instead of timestamps in the archive
      cmake_policy(SET CMP0135 NEW)
    endif()

  message(STATUS "Downloading SQLite")
  FetchContent_Declare(sqlite_local
    URL ${SQLITE_3_49_2_SRC_URL}
    URL_HASH SHA3_256=${SQLITE_SRC_3_43_2_SHA3_256}
  )
  FetchContent_MakeAvailable(sqlite_local)

  add_library(sqlite3 OBJECT ${sqlite_local_SOURCE_DIR}/sqlite3.c)
  target_include_directories(sqlite3 PUBLIC ${sqlite_local_SOURCE_DIR})
  set_target_properties( sqlite3 PROPERTIES
      C_VISIBILITY_PRESET "hidden"
      VISIBILITY_INLINES_HIDDEN ON
      POSITION_INDEPENDENT_CODE ON
      LINKER_LANGUAGE CXX
  )

  # We don't need extensions, and omitting them from SQLite removes the
  # need for dlopen/dlclose from within rocThrust.
  # We also don't need the shared cache, and omitting it yields some performance improvements.
  target_compile_options(
      sqlite3
      PRIVATE -DSQLITE_OMIT_LOAD_EXTENSION
      PRIVATE -DSQLITE_OMIT_SHARED_CACHE
  )
endif()

# Benchmark dependencies
if(BUILD_BENCHMARKS)
  set(BENCHMARK_VERSION 1.9.0)
  if(NOT EXTERNAL_DEPS_FORCE_DOWNLOAD)
    # Google Benchmark (https://github.com/google/benchmark.git)
    find_package(benchmark ${BENCHMARK_VERSION} QUIET)
  else()
    message(STATUS "Force installing Google Benchmark.")
  endif()

  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found or force download Google Benchmark on. Downloading and building Google Benchmark.")
    if(CMAKE_CONFIGURATION_TYPES)
      message(FATAL_ERROR "DownloadProject.cmake doesn't support multi-configuration generators.")
    endif()
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/googlebenchmark CACHE PATH "")
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    # hip-clang cannot compile googlebenchmark for some reason
      if(WIN32)
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=cl")
      else()
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
      endif()
    endif()

    download_project(
      PROJ                googlebenchmark
      GIT_REPOSITORY      https://github.com/google/benchmark.git
      GIT_TAG             v${BENCHMARK_VERSION}
      GIT_SHALLOW         TRUE
      INSTALL_DIR         ${GOOGLEBENCHMARK_ROOT}
      CMAKE_ARGS          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS=OFF -DBENCHMARK_ENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_CXX_STANDARD=14 ${COMPILER_OVERRIDE}
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE
    )
    find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT} NO_DEFAULT_PATH)
  endif()

  # Find/Get rocRAND. This may require a download, depending on the fetch method.
  fetch_dep(ROCRAND_FETCH_METHOD rocrand ROCRAND_PATH ROCM_DEP_RELEASE_BRANCH)

  # If we downloaded rocRAND or it are pulling it from the monorepo, we need to build it.
  # The path to the repo will is stored in ${ROCRAND_PATH}.
  if(${ROCRAND_FETCH_METHOD} STREQUAL "MONOREPO" OR ${ROCRAND_FETCH_METHOD} STREQUAL "DOWNLOAD")
    set(ROCRAND_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/rocrand CACHE PATH "")

    set(EXTRA_CMAKE_ARGS "-DGPU_TARGETS=${GPU_TARGETS}")
    # CMAKE_ARGS of download_project (or ExternalProject_Add) can't contain ; so another separator
    # is needed and LIST_SEPARATOR is passed to download_project()
    string(REPLACE ";" "|" EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS}")
    # Pass launcher so sccache can be used to speed up building rocRAND
    if(CMAKE_CXX_COMPILER_LAUNCHER)
      set(EXTRA_CMAKE_ARGS "${EXTRA_CMAKE_ARGS} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
    endif()
    download_project(
      PROJ                  rocrand
      SOURCE_DIR            ${ROCRAND_PATH}
      INSTALL_DIR           ${ROCRAND_ROOT}
      LIST_SEPARATOR        |
      CMAKE_ARGS            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_PREFIX_PATH=/opt/rocm ${EXTRA_CMAKE_ARGS}
      LOG_CONFIGURE         TRUE
      LOG_BUILD             TRUE
      LOG_INSTALL           TRUE
      LOG_OUTPUT_ON_FAILURE TRUE
      BUILD_PROJECT         TRUE
      STATUS_MSG            "Building"
    )
    find_package(rocrand REQUIRED CONFIG PATHS ${ROCRAND_ROOT})
  endif()
endif()
