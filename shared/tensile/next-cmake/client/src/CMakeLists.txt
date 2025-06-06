# ##############################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#
# ##############################################################################

set(_CMAKE_CURRENT_SOURCE_DIR "${TENSILE_SOURCE_TREE}/client/source")

target_sources(tensile-client
               PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/../main.cpp")

target_sources(
  tensile-client-lib
  PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/BenchmarkTimer.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/CSVStackFile.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ClientProblemFactory.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ConvolutionProblem.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/DataInitialization.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitorListener.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/LibraryUpdateReporter.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/MetaRunListener.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/PerformanceReporter.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ProgressListener.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/Reference.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ReferenceValidator.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ResultFileReporter.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/ResultReporter.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/SolutionIterator.cpp"
          "${_CMAKE_CURRENT_SOURCE_DIR}/TimingEvents.cpp")

if(NOT WIN32)
  target_sources(tensile-client-lib
                 PRIVATE "${_CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor.cpp")
endif()
