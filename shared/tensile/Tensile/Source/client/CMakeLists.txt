################################################################################
#
# Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

set(client_sources
    source/BenchmarkTimer.cpp
    source/CSVStackFile.cpp
    source/ClientProblemFactory.cpp
    source/ConvolutionProblem.cpp
    source/DataInitialization.cpp
    source/HardwareMonitorListener.cpp
    source/LibraryUpdateReporter.cpp
    source/MetaRunListener.cpp
    source/PerformanceReporter.cpp
    source/ProgressListener.cpp
    source/Reference.cpp
    source/ReferenceValidator.cpp
    source/ResultFileReporter.cpp
    source/ResultReporter.cpp
    source/SolutionIterator.cpp
    source/TimingEvents.cpp
    )

if(NOT WIN32)
    set(client_sources ${client_sources}
        source/HardwareMonitor.cpp
    )
endif()

add_library(TensileClient STATIC ${client_sources})

set_target_properties(TensileClient
                      PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED ON
                      CXX_EXTENSIONS OFF)

find_package(Boost COMPONENTS program_options REQUIRED)

if (NOT WIN32)
    find_package(ROCmSMI QUIET)
    if(NOT ROCmSMI_FOUND)
        set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${Tensile_DIR}" "${Tensile_DIR}/../Source/cmake" "${CMAKE_HOME_DIRECTORY}/cmake")
        find_package(ROCmSMI REQUIRED)
    endif()
endif()

include_directories(${Boost_INCLUDE_DIR})

target_include_directories(TensileClient PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")# "${rocm_smi_root}/include")

target_link_libraries(TensileClient PRIVATE TensileHost ${Boost_LIBRARIES})

if(NOT WIN32)
    target_link_libraries(TensileClient PRIVATE TensileHost rocm_smi)
endif()

if(TENSILE_USE_OPENMP)
    target_link_libraries(TensileClient PRIVATE custom_openmp_cxx)
endif()


add_executable(tensile_client main.cpp)
set_target_properties(tensile_client
                      PROPERTIES
                      CXX_STANDARD 17
                      CXX_STANDARD_REQUIRED ON
                      CXX_EXTENSIONS OFF)

target_link_libraries(tensile_client PRIVATE TensileHost TensileClient ${Boost_LIBRARIES})
if(TENSILE_USE_OPENMP)
    target_link_libraries(tensile_client PRIVATE custom_openmp_cxx)
endif()
