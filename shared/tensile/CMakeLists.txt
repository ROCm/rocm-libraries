################################################################################
#
# Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.13)
project(TensileTests)

option(BUILD_TENSILE_HOST_LIBRARY_TESTS "Build HostLibraryTests/TensileTests binary" ON)
option(INSTALL_TENSILE_PYTEST_TESTS "Install Tensile pytest tests for CI" ON)

# Install Tensile Python package and pytest tests
if(INSTALL_TENSILE_PYTEST_TESTS)
  # Install the entire Tensile directory (Python package + tests)
  # This includes:
  #   - Tensile/*.py (Python modules)
  #   - Tensile/Tests/ (pytest tests + YAML integration tests)
  #   - pytest.ini, conftest.py, tox.ini
  install(
    DIRECTORY Tensile/
    DESTINATION Tensile
    COMPONENT tests
    FILES_MATCHING
      PATTERN "*.py"
      PATTERN "*.yaml"
      PATTERN "*.yml"
      PATTERN "*.ini"
      PATTERN "__pycache__" EXCLUDE
      PATTERN "*.pyc" EXCLUDE
  )

  # Install top-level test configuration files
  install(
    FILES pytest.ini tox.ini
    DESTINATION Tensile
    COMPONENT tests
  )

  # Install test_categories.yaml for TheRock CI
  install(
    FILES test_categories.yaml
    DESTINATION Tensile
    COMPONENT tests
  )
endif()

# Build and install HostLibraryTests (TensileTests GTest binary)
if(BUILD_TENSILE_HOST_LIBRARY_TESTS)
  # HostLibraryTests has its own CMakeLists.txt - add it as a subdirectory
  # Set Tensile_ROOT to point to the source Tensile directory
  set(Tensile_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Tensile" CACHE PATH "Tensile root directory")

  add_subdirectory(HostLibraryTests)

  # Install the TensileTests binary
  if(TARGET TensileTests)
    install(
      TARGETS TensileTests
      RUNTIME DESTINATION Tensile
      COMPONENT tests
    )
  endif()
endif()
