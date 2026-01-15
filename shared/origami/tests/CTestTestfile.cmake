################################################################################
#
# MIT License
#
# Copyright 2025 AMD ROCm(TM) Software
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

# CTestTestfile for installed Origami tests
# This file is installed to bin/origami/ for TheRock CI testing

# C++ tests (Catch2)
add_test(origami-tests "../origami-tests")

# Python tests
add_test(origami_python_test python3 "origami_test.py")
add_test(origami_python_grid_test python3 "origami_grid_test.py")

# Set environment for Python tests to find liborigami.so and the Python module
# Paths are relative to bin/origami/ where CTestTestfile.cmake is installed
set_tests_properties(origami_python_test origami_python_grid_test PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_CURRENT_LIST_DIR}/../../lib:$ENV{LD_LIBRARY_PATH};PYTHONPATH=${CMAKE_CURRENT_LIST_DIR}:$ENV{PYTHONPATH}"
)
