# MIT License
#
# Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

include(dependencies/monorepo)

# Test dependencies
if(BUILD_TEST)
  include(dependencies/googletest)
endif()

if(WITH_ROCRAND)
  # While we could find rocRAND through 'fetch_monorepo_dep()', this is not the
  # intented development setup. It is instead recommended to install rocRAND
  # seperately instead with e.g. 'apt install rocrand-dev'.
  # 
  # rocRAND only accelerates generating test/benchmark data. It is not a
  # critical depedency.
  find_package(PACKAGE rocrand REQUIRED)
endif()
