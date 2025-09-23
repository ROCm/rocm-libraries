# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(tensilelite-client
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkTimer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ClientProblemFactory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/DataInitialization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PerformanceReporter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitorListener.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/TimingEvents.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/HardwareMonitor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ReferenceValidator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ResultFileReporter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ResultReporter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/MetaRunListener.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Reference.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProgressListener.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/TypedId.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Rotating.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/CSVStackFile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/SolutionIterator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/LibraryUpdateReporter.cpp"
)
