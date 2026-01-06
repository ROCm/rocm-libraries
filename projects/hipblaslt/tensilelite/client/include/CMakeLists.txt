# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(tensilelite-client
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/TypedId.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MetaResultReporter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitorListener.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ClientProblemFactory.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ResultReporter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/DataInitialization.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TensorDataManipulation.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ResultReporter_fwd.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ResultComparison.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LogReporter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/RunListener.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/MetaRunListener.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Rotating.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ProgressListener.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Utility.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/LibraryUpdateReporter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/SolutionIterator.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/PerformanceReporter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor_fwd.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ResultFileReporter.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkTimer.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/TimingEvents.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/ReferenceValidator.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/Reference.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/CSVStackFile.hpp"
)
