# ########################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
# ########################################################################

# Fails when rocalution-test wrote no coverage profile. Without one, the report
# would only contain what the unit tests reach, and nothing else would notice.
#
#   cmake -DPROFRAW_DIR=<dir> -P CheckCoverageProfile.cmake

# rocalution-test writes rocalution-coverage_integration_*. The unit-test
# profiles are rocalution-coverage_unit_* and rocalution-coverage_lifecycle_*,
# so this does not match them.
file(GLOB profiles "${PROFRAW_DIR}/rocalution-coverage_integration_*.profraw")
if(NOT profiles)
  message(FATAL_ERROR "rocalution-test wrote no coverage profile to ${PROFRAW_DIR}")
endif()
