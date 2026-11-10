# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

include_guard(GLOBAL)

include(FetchContent)

# Use the system SQLite3 package instead of downloading a copy.
option(SQLITE_USE_SYSTEM_PACKAGE "Use SQLite3 from find_package" ON)

# Minimum version for the system search. Consumers may set SQLITE_MIN_VERSION
# before including this module. The value only names the download cache
# variables below, for example SQLITE_3_45_1_SRC_URL.
if(NOT DEFINED SQLITE_MIN_VERSION)
  set(SQLITE_MIN_VERSION "3.45.1")
endif()

if(SQLITE_USE_SYSTEM_PACKAGE)
  find_package(SQLite3 ${SQLITE_MIN_VERSION} REQUIRED)
else()
  # If another component already created SQLite::SQLite3 through
  # find_package(SQLite3), building the embedded copy anyway would silently
  # change what gets linked.
  if(TARGET SQLite::SQLite3)
    message(FATAL_ERROR
      "SQLite::SQLite3 already exists in this build. Set "
      "SQLITE_USE_SYSTEM_PACKAGE=ON to use that package, or remove the "
      "other find_package(SQLite3) call."
    )
  endif()

  message(STATUS
    "Force download local copy of SQLite on. Downloading and building SQLite."
  )

  string(REPLACE "." "_" SQLITE_VER_UNDERSCORE ${SQLITE_MIN_VERSION})

  # The cache and environment variable names contain the version, for
  # example SQLITE_3_45_1_SRC_URL. CI sets the environment variable to
  # point at a mirrored copy. Build the names once here.
  set(_src_url_var SQLITE_${SQLITE_VER_UNDERSCORE}_SRC_URL)
  set(_src_hash_var SQLITE_SRC_${SQLITE_VER_UNDERSCORE}_SHA3_256)

  # SQLite 3.51.3 to address CVE-2025-70873:
  #   https://github.com/advisories/GHSA-p36r-6g67-869c
  set(_src_url_default https://sqlite.org/2026/sqlite-amalgamation-3510300.zip)
  if(DEFINED ENV{${_src_url_var}})
    set(_src_url_default $ENV{${_src_url_var}})
  endif()

  set(${_src_url_var} ${_src_url_default}
    CACHE STRING "Location of SQLite source code"
  )
  set(${_src_hash_var}
    ced02ff9738970f338c9c8e269897b554bcda73f6cf1029d49459e1324dbeaea
    CACHE STRING "SHA3-256 hash of SQLite source code"
  )

  # Embed SQLite
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    # Use extract timestamp for fetched files instead of archive timestamps.
    cmake_policy(SET CMP0135 NEW)
  endif()

  FetchContent_Declare(sqlite_local
    URL ${${_src_url_var}}
    URL_HASH SHA3_256=${${_src_hash_var}}
  )
  FetchContent_MakeAvailable(sqlite_local)
  unset(_src_url_var)
  unset(_src_hash_var)
  unset(_src_url_default)

  # Build through the sibling sqlite3/ directory so the object library has
  # its own directory scope. That scope removes leaked global flags such as
  # '--offload-compress', which the plain C compiler rejects. See
  # sqlite3/CMakeLists.txt.
  add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/sqlite3
    ${CMAKE_CURRENT_BINARY_DIR}/primlibs-sqlite3-build
  )
endif()

# After including this module, the target SQLite::SQLite3 always exists and
# is what consumers should link against.
# * If SQLITE_USE_SYSTEM_PACKAGE=ON  then find_package(SQLite3) provides it.
# * If SQLITE_USE_SYSTEM_PACKAGE=OFF then the downloaded source is built as the
#   'sqlite3' object library and aliased to SQLite::SQLite3.
