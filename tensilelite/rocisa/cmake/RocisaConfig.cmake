################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

include(CMakeParseArguments)
if(NOT DEFINED Rocisa_ROOT)
    # Compute the installation prefix relative to this file.
    get_filename_component(Rocisa_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
    get_filename_component(Rocisa_PREFIX "${Rocisa_PREFIX}" PATH)
    execute_process(COMMAND "${Rocisa_PREFIX}/bin/RocisaGetPath${CMAKE_EXECUTABLE_SUFFIX}" OUTPUT_VARIABLE Rocisa_ROOT)
endif()

message(STATUS "Building rocIsa" ${Rocisa_ROOT})

set(ROCISA_VIRTUAL_DIR ${Rocisa_ROOT})
set(ROCISA_BUILD_DIR ${PROJECT_BINARY_DIR}/rocisa_build)
file(MAKE_DIRECTORY ${ROCISA_BUILD_DIR})
execute_process(
  COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" -DPython_EXECUTABLE=${VIRTUALENV_BIN_DIR}/${VIRTUALENV_PYTHON_EXENAME} -S ${ROCISA_VIRTUAL_DIR} -B ${ROCISA_BUILD_DIR}
  WORKING_DIRECTORY ${ROCISA_BUILD_DIR}
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  RESULT_VARIABLE result
)
if(NOT ${result} EQUAL 0)
  message(STATUS "Output: ${output}")
  message(STATUS "Error: ${error}")
  message(FATAL_ERROR "Result: ${result}")
else()
  message(STATUS "rocIsa configuring done.")
endif()

include(ProcessorCount)
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env CMAKE_BUILD_PARALLEL_LEVEL=${NPROC} -- ${CMAKE_COMMAND} --build ${ROCISA_BUILD_DIR}
  WORKING_DIRECTORY ${ROCISA_BUILD_DIR}
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error
  RESULT_VARIABLE result
)
if(NOT ${result} EQUAL 0)
  message(STATUS "Output: ${output}")
  message(STATUS "Error: ${error}")
  message(FATAL_ERROR "Result: ${result}")
else()
  message(STATUS "rocIsa building done.")
  set(Rocisa_LIBRARIES "${PROJECT_BINARY_DIR}/rocisa_build/lib")
endif()