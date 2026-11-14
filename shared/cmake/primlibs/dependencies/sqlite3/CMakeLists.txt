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

# Builds the vendored SQLite amalgamation as an object library. Sources are
# fetched as sqlite_local by the parent sqlite3.cmake.
#
# The global '--offload-compress' flag from offload_compress.cmake breaks
# this target. cc fails with a hard error that cannot be ignored with -Wno-*.
#
# CMake applies directory flags at generate time, so the flag can only be
# removed in a separate directory scope.

# Remove '--offload-compress' from the directory compile options. This can
# be removed once the build system scopes that flag to HIP targets.
get_directory_property(_primlibs_sqlite3_compile_options COMPILE_OPTIONS)
list(FILTER _primlibs_sqlite3_compile_options EXCLUDE REGEX "offload-compress")
set_property(DIRECTORY PROPERTY COMPILE_OPTIONS ${_primlibs_sqlite3_compile_options})
unset(_primlibs_sqlite3_compile_options)

# Remove '--offload-compress' from the inherited per-language flags
# variables, in case it leaked in through CMAKE_{C,CXX}_FLAGS. Same removal
# condition as above.
foreach(_lang IN ITEMS C CXX)
  if(DEFINED CMAKE_${_lang}_FLAGS)
    string(REPLACE "--offload-compress" "" CMAKE_${_lang}_FLAGS "${CMAKE_${_lang}_FLAGS}")
  endif()
endforeach()

add_library(sqlite3 OBJECT ${sqlite_local_SOURCE_DIR}/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${sqlite_local_SOURCE_DIR})
set_target_properties(sqlite3 PROPERTIES
    C_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE CXX
)

# Extensions are not needed. Omitting them removes the need for
# dlopen/dlclose. Omitting the shared cache improves performance.
target_compile_options(sqlite3 PRIVATE
    -DSQLITE_OMIT_LOAD_EXTENSION
    -DSQLITE_OMIT_SHARED_CACHE
)

# Matches the target name provided by find_package(SQLite3). The module
# checks for a conflicting pre-existing SQLite::SQLite3 before entering
# this directory.
add_library(SQLite::SQLite3 ALIAS sqlite3)
