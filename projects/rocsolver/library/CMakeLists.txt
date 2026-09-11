# ##########################################################################
# Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# ##########################################################################

# This is incremented when the ABI to the library changes
set(rocsolver_SOVERSION 0.12)

# Create version header from templated .in file using CMake info
configure_file(include/rocsolver/rocsolver-version.h.in "${PROJECT_BINARY_DIR}/include/rocsolver/rocsolver-version.h")

# Configure Windows DLL version resource (VERSIONINFO / PE metadata)
if(WIN32)
  find_package(Git QUIET)
  set(rocsolver_VERSION_COMMIT_ID "unknown")
  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE git_commit_id
      RESULT_VARIABLE git_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    if(git_result EQUAL 0 AND git_commit_id)
      set(rocsolver_VERSION_COMMIT_ID "${git_commit_id}")
    endif()
  endif()
  string(TIMESTAMP ROCSOLVER_COPYRIGHT_YEAR "%Y")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/version.rc.in"
    "${PROJECT_BINARY_DIR}/version.rc" @ONLY)
endif()

add_subdirectory(src)
