# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

include( ExternalProject )

# SQLite is used for rtc_cache. Require a safe baseline (>= 3.50.2).
# Note: the backup API we rely on has been enabled by default since 3.36.0.
option( SQLITE_USE_SYSTEM_PACKAGE "Use SQLite3 from find_package" OFF )

if( SQLITE_USE_SYSTEM_PACKAGE )
  # Require a safe baseline (fixes truncation/memory-corruption issues).
  find_package(SQLite3 3.50.2 REQUIRED)
  list(APPEND static_depends PACKAGE SQLite3)
  set(ROCFFT_SQLITE_LIB SQLite::SQLite3)
else()
  include( FetchContent )

  # embed SQLite amalgamation (version 3.50.2 -> serial 3500200).
  # allow override via environment variable for mirrors/airgapped builds.
  if(DEFINED ENV{SQLITE_3_50_2_SRC_URL})
    set(SQLITE_3_50_2_SRC_URL_INIT $ENV{SQLITE_3_50_2_SRC_URL})
  else()
    set(SQLITE_3_50_2_SRC_URL_INIT https://www.sqlite.org/2025/sqlite-amalgamation-3500200.zip)
  endif()
  set(SQLITE_3_50_2_SRC_URL ${SQLITE_3_50_2_SRC_URL_INIT} CACHE STRING "Location of SQLite source code")
  set(SQLITE_SRC_3_50_2_SHA3_256 75c118e727ee6a9a3d2c0e7c577500b0c16a848d109027f087b915b671f61f8a CACHE STRING "SHA3-256 hash of SQLite source code")

  # use extract timestamp for fetched files instead of timestamps in the archive
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
  endif()

  FetchContent_Declare(sqlite_local
    URL ${SQLITE_3_50_2_SRC_URL}
    URL_HASH SHA3_256=${SQLITE_SRC_3_50_2_SHA3_256}
  )
  FetchContent_MakeAvailable(sqlite_local)

  if(NOT TARGET sqlite3)
    add_library( sqlite3 OBJECT ${sqlite_local_SOURCE_DIR}/sqlite3.c )
    target_include_directories( sqlite3 PUBLIC ${sqlite_local_SOURCE_DIR} )
    set_target_properties( sqlite3 PROPERTIES
      C_VISIBILITY_PRESET "hidden"
      VISIBILITY_INLINES_HIDDEN ON
      POSITION_INDEPENDENT_CODE ON
      )
  endif()

  # we don't need extensions, and omitting them from SQLite removes the
  # need for dlopen/dlclose from within rocFFT
  target_compile_options(
    sqlite3
    PRIVATE -DSQLITE_OMIT_LOAD_EXTENSION
  )
  set(ROCFFT_SQLITE_LIB sqlite3)
endif()

