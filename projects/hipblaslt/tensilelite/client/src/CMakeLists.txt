# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

set(COMMON_CLIENT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/BenchmarkTimer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ClientProblemFactory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProgramOptions.cpp"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../clients/common/src/mxDataGen.cpp"
)

target_sources(tensilelite-client PRIVATE ${COMMON_CLIENT_SOURCES})
target_sources(cpu-gemm-driver PRIVATE ${COMMON_CLIENT_SOURCES})

target_include_directories(tensilelite-client
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../clients/common/include"
)

target_include_directories(cpu-gemm-driver
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../clients/common/include"
)

if (TENSILELITE_CLIENT_ENABLE_ROCPROFSDK)

target_sources(tensilelite-client
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Profiler.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/RocProfiler.cpp"
)

endif()
