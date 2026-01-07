# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(rocsparse-bench
    ${CMAKE_CURRENT_SOURCE_DIR}/client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_arguments_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_bench_app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_bench_cmdlines.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_routine.cpp
)

if(ROCSPARSE_ENABLE_MEMSTAT)
    target_compile_definitions(rocsparse-bench PRIVATE ROCSPARSE_WITH_MEMSTAT)
endif()

if(ROCSPARSE_ENABLE_ILP64)
    target_compile_definitions(rocsparse-bench PRIVATE rocsparse_ILP64)
endif()

if(ROCSPARSE_ENABLE_ASAN)
    rocsparse_target_configure_sanitizers(rocsparse-bench PRIVATE)
endif()

target_compile_options(rocsparse-bench PRIVATE
    -Wno-deprecated
    -Wno-unused-command-line-argument
    -Wall
)

target_compile_features(rocsparse-bench PRIVATE cxx_std_17)

target_include_directories(rocsparse-bench PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
)

target_link_libraries(rocsparse-bench PRIVATE
    roc::rocsparse
    hip::host
    hip::device
)

if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(rocsparse-bench PRIVATE OpenMP::OpenMP_CXX)
endif()

if(rocsparseio_FOUND)
    target_link_libraries(rocsparse-bench PRIVATE roc::rocsparseio)
endif()

if(ROCSPARSE_ENABLE_MARKER)
    target_compile_definitions(rocsparse-bench PRIVATE ROCSPARSE_CLIENTS_WITH_ROCTX)
    target_include_directories(rocsparse-bench PRIVATE ${ROCTRACER_INCLUDE_DIR})
    target_link_libraries(rocsparse-bench PRIVATE ${rocTracer})
endif()

rocm_install(TARGETS rocsparse-bench COMPONENT benchmarks DESTINATION "${CMAKE_INSTALL_BINDIR}")
