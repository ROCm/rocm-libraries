# ########################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc.
#
# MIT License
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

set(_CMAKE_CURRENT_SOURCE_DIR "${TENSILELITE_CLIENT_DIR}/include")

target_sources(tensilelite-client
    PRIVATE
        "${_CMAKE_CURRENT_SOURCE_DIR}/TypedId.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MetaResultReporter.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitorListener.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ClientProblemFactory.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ResultReporter.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/DataInitialization.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TensorDataManipulation.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ResultReporter_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ResultComparison.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LogReporter.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/RunListener.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/MetaRunListener.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Rotating.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ProgressListener.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Utility.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/LibraryUpdateReporter.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/SolutionIterator.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/PerformanceReporter.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor_fwd.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ResultFileReporter.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/BenchmarkTimer.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/TimingEvents.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/ReferenceValidator.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/Reference.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor.hpp"
        "${_CMAKE_CURRENT_SOURCE_DIR}/CSVStackFile.hpp"
)

