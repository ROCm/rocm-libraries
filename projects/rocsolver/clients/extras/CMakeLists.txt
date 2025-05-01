# ##########################################################################
# Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

find_package(GTest REQUIRED)

add_executable(test-header
  test_header.c
)
target_link_libraries(test-header PRIVATE roc::rocsolver)
set_target_properties(test-header PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

if(UNIX AND BUILD_SHARED_LIBS)
  if(TARGET rocsolver)
    add_executable(test-rocsolver-dlopen
      test_dlopen_main.cpp
    )
    target_compile_definitions(test-rocsolver-dlopen PRIVATE
      ROCSOLVER_LIB_NAME="$<TARGET_FILE:rocsolver>"
    )
    target_link_libraries(test-rocsolver-dlopen PRIVATE
      $<IF:$<TARGET_EXISTS:GTest::gtest>,GTest::gtest,GTest::GTest>
      ${CMAKE_DL_LIBS}
    )

    add_test(
      NAME test-rocsolver-dlopen
      COMMAND test-rocsolver-dlopen
    )
  endif()

  if(TARGET rocsolver-bench)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
      add_test(
        NAME test-rocsolver-bench
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/test_rocsolver_bench.py"
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:rocsolver-bench>"
      )
    endif()
  endif()
endif()
