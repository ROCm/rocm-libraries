

cmake_minimum_required(VERSION 3.16)

project(rocRollerClient VERSION 1.0)

message("Building client program")

list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
find_package(hip)

add_subdirectory(../extern/CLI11 CLI11)

add_executable(gemm
  GraphInspector.cpp
  gemm.cpp
  GEMMParameters.cpp
  visualize.cpp
)

target_compile_options(gemm PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_include_directories(
  gemm
  PUBLIC
  lib/include
  ${HIP_ROOT_DIR}/include
  ${CMAKE_SOURCE_DIR}/extern/mxDataGenerator/lib/include
)
target_link_options(gemm PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  gemm
  PUBLIC
  rocroller_interface
  common_test_utilities
  msgpackc-cxx
  gtest_main
  gmock
  openblas
  CLI11::CLI11
)

target_link_libraries(gemm PUBLIC hip::host)
target_compile_definitions(gemm PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)

if(ROCROLLER_USE_LLVM)
  target_compile_definitions(gemm PUBLIC ROCROLLER_USE_LLVM)
endif()
target_include_directories(
  gemm
  PUBLIC
  ${HIP_ROOT_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${LLD_INCLUDE_DIRS}
)

if(ROCROLLER_USE_YAML_CPP)
  target_compile_definitions(gemm PUBLIC ROCROLLER_USE_YAML_CPP)
  set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

set_target_properties(gemm
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/client"
)

add_executable(codegen_stress
  codegen_stress_test.cpp
)

target_compile_options(codegen_stress PUBLIC ${EXTRA_COMPILE_OPTIONS})
target_include_directories(
  codegen_stress
  PUBLIC
  lib/include
  ${HIP_ROOT_DIR}/include
)
target_link_options(codegen_stress PUBLIC ${EXTRA_LINK_OPTIONS})
target_link_libraries(
  codegen_stress
  PUBLIC
  rocroller_interface
  common_test_utilities
  msgpackc-cxx
  gtest_main
  gmock
  openblas
  CLI11::CLI11
)

target_link_libraries(codegen_stress PUBLIC hip::host)
target_compile_definitions(codegen_stress PUBLIC ROCROLLER_USE_HIP __HIP_PLATFORM_AMD__)

if(ROCROLLER_USE_LLVM)
  target_compile_definitions(codegen_stress PUBLIC ROCROLLER_USE_LLVM)
endif()
target_include_directories(
  codegen_stress
  PUBLIC
  ${HIP_ROOT_DIR}/include
  ${LLVM_INCLUDE_DIRS}
  ${LLD_INCLUDE_DIRS}
)

if(ROCROLLER_USE_YAML_CPP)
  target_compile_definitions(codegen_stress PUBLIC ROCROLLER_USE_YAML_CPP)
  set_property(TARGET yaml-cpp PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

set_target_properties( codegen_stress
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/client"
)

add_custom_target(all_clients)
add_dependencies(all_clients gemm codegen_stress)
