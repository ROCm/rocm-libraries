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

# CMake helpers for the shared primitives libraries test controller.
# See README.md in this directory for usage information.

include_guard(GLOBAL)

set(TEST_CONTROLLER_ROOT "${CMAKE_CURRENT_LIST_DIR}")
set(TEST_CONTROLLER_INCLUDE_DIR "${TEST_CONTROLLER_ROOT}/include")
# Unit tests for the test controller itself. Projects can build these as one of their
# own test targets (they need GTest and the HIP runtime).
set(TEST_CONTROLLER_UNIT_TEST_SOURCE "${TEST_CONTROLLER_ROOT}/test/test_controller_tests.cpp")

# Configures a test target so that it can include test_utils_controller.hpp.
#
#   test_controller_configure_target(<target> PROJECT_DIR <dir>)
#
# PROJECT_DIR is the name of the directory, relative to the installed test binaries,
# that the control file is installed into (eg. "hipcub" for bin/hipcub/control.txt).
# It should match the INSTALL_DESTINATION passed to test_controller_add_control_file().
function(test_controller_configure_target TARGET)
  cmake_parse_arguments(PARSE_ARGV 1 TC "" "PROJECT_DIR" "")
  if(NOT TC_PROJECT_DIR)
    message(FATAL_ERROR "test_controller_configure_target: PROJECT_DIR is required.")
  endif()

  target_include_directories(${TARGET} PRIVATE "${TEST_CONTROLLER_INCLUDE_DIR}")
  target_compile_definitions(${TARGET} PRIVATE "TEST_CONTROLLER_PROJECT_DIR=\"${TC_PROJECT_DIR}\"")
endfunction()

# Copies a project's control file into the build tree whenever it changes, and installs it.
#
#   test_controller_add_control_file(<dependent target>
#                                    SOURCE <path to control.txt>
#                                    BUILD_DIR <build directory to copy it into>
#                                    INSTALL_DESTINATION <install directory>
#                                    [COMPONENT <install component>])
#
# The control file is located at runtime relative to the running test binary. The test
# controller looks for it in:
#   - <binary dir>/../control.txt (build tree), and
#   - <binary dir>/<PROJECT_DIR>/control.txt (install tree),
# so BUILD_DIR should be the parent of the directory the test binaries are built into, and
# INSTALL_DESTINATION should be <test binary install dir>/<PROJECT_DIR>.
#
# A custom target named <dependent target>_copy_control_file is created to perform the copy,
# and <dependent target> is made to depend on it. Rebuilding that target after editing
# the control file is enough to pick up the changes.
function(test_controller_add_control_file DEPENDENT_TARGET)
  cmake_parse_arguments(PARSE_ARGV 1 TC "" "SOURCE;BUILD_DIR;INSTALL_DESTINATION;COMPONENT" "")
  foreach(arg SOURCE BUILD_DIR INSTALL_DESTINATION)
    if(NOT TC_${arg})
      message(FATAL_ERROR "test_controller_add_control_file: ${arg} is required.")
    endif()
  endforeach()
  if(NOT TC_COMPONENT)
    set(TC_COMPONENT tests)
  endif()

  get_filename_component(control_file_name "${TC_SOURCE}" NAME)
  set(build_control_file "${TC_BUILD_DIR}/${control_file_name}")

  # The target name is prefixed with the dependent target so that multiple projects can
  # use the test controller in the same (super)build.
  add_custom_target(${DEPENDENT_TARGET}_copy_control_file
    ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${TC_SOURCE}" "${build_control_file}"
  )
  add_dependencies(${DEPENDENT_TARGET} ${DEPENDENT_TARGET}_copy_control_file)

  # Install the file as well, so that installed binaries can pick it up.
  # Prefer rocm_install (from rocm-cmake) when the project uses it, so that the file is
  # handled consistently with the project's other packaged files.
  if(COMMAND rocm_install)
    rocm_install(
      FILES "${build_control_file}"
      DESTINATION "${TC_INSTALL_DESTINATION}"
      COMPONENT ${TC_COMPONENT}
    )
  else()
    install(
      FILES "${build_control_file}"
      DESTINATION "${TC_INSTALL_DESTINATION}"
      COMPONENT ${TC_COMPONENT}
    )
  endif()
endfunction()
