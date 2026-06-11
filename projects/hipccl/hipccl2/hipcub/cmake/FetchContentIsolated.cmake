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

include(FetchContent)
include(CMakeParseArguments)

# Wraps FetchContent to isolate the parent project's strict compiler flags (e.g., -Werror)
# and correctly parses/applies CMAKE_ARGS.
function(fetch_content_isolated proj_name)
  cmake_parse_arguments(FC_ISO "" "" "CMAKE_ARGS" ${ARGN})

  if(COMMAND rocm_check_toolchain_var)
    # Suppress ROCMChecks WARNING on third-party dependencies
    function(rocm_check_toolchain_var)
    endfunction()
  endif()

  # Automatic Isolation
  # Safely remove `-Werror` and `-Werror=...` flags using regex.
  # We also remove MSVC's `/WX` flag for Windows compatibility.
  string(REGEX REPLACE "(^| )-Werror(=[^ ]*)?" " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/WX" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  # Convert CMAKE_ARGS into local variables within the current scope.
  foreach(arg IN LISTS FC_ISO_CMAKE_ARGS)
      if(arg MATCHES "^-D([^=:]+)(:[a-zA-Z0-9_]+)?=(.*)$")
          set(${CMAKE_MATCH_1} "${CMAKE_MATCH_3}")
      endif()
  endforeach()

  # Declare and make the project available.
  FetchContent_Declare(${proj_name} ${FC_ISO_UNPARSED_ARGUMENTS})
  FetchContent_MakeAvailable(${proj_name})

  # Export critical variables.
  string(TOLOWER "${proj_name}" lc_proj)
  set(${lc_proj}_POPULATED "${${lc_proj}_POPULATED}" PARENT_SCOPE)
  set(${lc_proj}_SOURCE_DIR "${${lc_proj}_SOURCE_DIR}" PARENT_SCOPE)
  set(${lc_proj}_BINARY_DIR "${${lc_proj}_BINARY_DIR}" PARENT_SCOPE)
endfunction()
