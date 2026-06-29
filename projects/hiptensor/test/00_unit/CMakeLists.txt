###############################################################################
 #
 # MIT License
 #
 # Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 # THE SOFTWARE.
 #
 ###############################################################################

 add_hiptensor_unit_test(logger_test ${CMAKE_CURRENT_SOURCE_DIR}/logger_test.cpp)
 add_hiptensor_unit_test(yaml_test ${CMAKE_CURRENT_SOURCE_DIR}/yaml_test.cpp)
 add_hiptensor_unit_test(elementwise_op_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/elementwise_op_test.cpp
     )
 add_hiptensor_unit_test(util_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/util_test.cpp
     )
 add_hiptensor_unit_test(hiptensor_options_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/hiptensor_options_test.cpp
     )
 add_hiptensor_unit_test(plan_lifetime_test
     ${HIPTENSOR_COMMON_TEST_SOURCES}
     ${CMAKE_CURRENT_SOURCE_DIR}/plan_lifetime_test.cpp
     )
 # add_hiptensor_unit_test(interface_test
     # ${HIPTENSOR_COMMON_TEST_SOURCES}
     # ${CMAKE_CURRENT_SOURCE_DIR}/interface_test.cpp
     # )

 # Public-API coverage test.
 #
 # A configure-time codegen step parses the public header for every function
 # declared with HIPTENSOR_EXPORT and emits a translation unit that references
 # the address of each. Linking that against libhiptensor fails if the built
 # library does not export some declared public API function, so a missing
 # export (or a new API the library forgot to provide) is caught here instead
 # of by a downstream consumer. Because the source is regenerated from the
 # header at configure time, newly added APIs are covered automatically.
 find_package(Python3 COMPONENTS Interpreter REQUIRED)

 set(_api_coverage_header "${PROJECT_SOURCE_DIR}/library/include/hiptensor/hiptensor.h")
 set(_api_coverage_generator "${CMAKE_CURRENT_SOURCE_DIR}/generate_api_coverage.py")
 set(_api_coverage_generated "${CMAKE_CURRENT_BINARY_DIR}/api_coverage_test.cpp")

 # Generate at configure time so the source exists before the target is created,
 # and add a build-time custom command so header edits refresh it on rebuild.
 execute_process(
     COMMAND ${Python3_EXECUTABLE} ${_api_coverage_generator}
             --header ${_api_coverage_header}
             --output ${_api_coverage_generated}
     RESULT_VARIABLE _api_coverage_gen_result
 )
 if(NOT _api_coverage_gen_result EQUAL 0)
     message(FATAL_ERROR "Failed to generate hipTensor API coverage test source")
 endif()

 add_custom_command(
     OUTPUT ${_api_coverage_generated}
     COMMAND ${Python3_EXECUTABLE} ${_api_coverage_generator}
             --header ${_api_coverage_header}
             --output ${_api_coverage_generated}
     DEPENDS ${_api_coverage_generator} ${_api_coverage_header}
     COMMENT "Generating hipTensor API coverage test source"
     VERBATIM
 )

 add_hiptensor_unit_test(api_coverage_test ${_api_coverage_generated})
