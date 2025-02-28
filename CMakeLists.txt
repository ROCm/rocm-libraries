# Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

if (DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
else()
    set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()
cmake_minimum_required(VERSION 3.5)

project(TransferBench VERSION 1.59.00 LANGUAGES CXX)

# Default GPU architectures to build
#==================================================================================================
set(DEFAULT_GPUS
      gfx906
      gfx908
      gfx90a
      gfx942
      gfx1030
      gfx1100
      gfx1101
      gfx1102
      gfx1200
      gfx1201)

# Build only for local GPU architecture
if (BUILD_LOCAL_GPU_TARGET_ONLY)
  message(STATUS "Building only for local GPU target")
  if (COMMAND rocm_local_targets)
    rocm_local_targets(DEFAULT_GPUS)
  else()
    message(WARNING "Unable to determine local GPU targets. Falling back to default GPUs.")
  endif()
endif()

# Determine which GPU architectures to build for
set(GPU_TARGETS "${DEFAULT_GPUS}" CACHE STRING "Target default GPUs if GPU_TARGETS is not defined.")

# Check if clang compiler can offload to GPU_TARGETS
if (COMMAND rocm_check_target_ids)
  message(STATUS "Checking for ROCm support for GPU targets: " "${GPU_TARGETS}")
  rocm_check_target_ids(SUPPORTED_GPUS TARGETS ${GPU_TARGETS})
else()
  message(WARNING "Unable to check for supported GPU targets. Falling back to default GPUs.")
  set(SUPPORTED_GPUS ${DEFAULT_GPUS})
endif()

set(COMPILING_TARGETS "${SUPPORTED_GPUS}" CACHE STRING "GPU targets to compile for.")
message(STATUS "Compiling for ${COMPILING_TARGETS}")

foreach(target ${COMPILING_TARGETS})
 list(APPEND static_link_flags --offload-arch=${target})
endforeach()
list(JOIN static_link_flags " " flags_str)
set( CMAKE_CXX_FLAGS "${flags_str} ${CMAKE_CXX_FLAGS}")

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -L${ROCM_PATH}/lib")
include_directories(${ROCM_PATH}/include)
find_library(IBVERBS_LIBRARY ibverbs)
find_path(IBVERBS_INCLUDE_DIR infiniband/verbs.h)
if (DEFINED ENV{DISABLE_NIC_EXEC})
  message(STATUS "Disabling NIC Executor support")
elseif(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
  message(STATUS "Found ibverbs: ${IBVERBS_LIBRARY}. Building with NIC executor support. Can set DISABLE_NIC_EXEC=1 to disable")
  add_definitions(-DNIC_EXEC_ENABLED)
  link_libraries(ibverbs)
else()
  if (NOT IBVERBS_LIBRARY)
    message(WARNING "IBVerbs library not found")
  elseif (NOT IBVERBS_INCLUDE_DIR)
    message(WARNING "infiniband/verbs.h not found")
  endif()
  message(WARNING "Building without NIC executor support. To use the TransferBench RDMA executor, check if your system has NICs, the NIC drivers are installed, and libibverbs-dev is installed")
endif()

link_libraries(numa hsa-runtime64 pthread)
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY .)
add_executable(TransferBench src/client/Client.cpp)
target_include_directories(TransferBench PRIVATE src/header src/client src/client/Presets)

find_package(ROCM 0.8 REQUIRED PATHS ${ROCM_PATH})
include(ROCMInstallTargets)
include(ROCMCreatePackage)
set(ROCMCHECKS_WARN_TOOLCHAIN_VAR OFF)

set(PACKAGE_NAME TB)
set(LIBRARY_NAME TransferBench)

rocm_install(TARGETS TransferBench COMPONENT devel)

rocm_package_add_dependencies(DEPENDS numactl hsa-rocr)

rocm_create_package(
    NAME ${LIBRARY_NAME}
    DESCRIPTION "TransferBench package"
    MAINTAINER "RCCL Team <gilbert.lee@amd.com>"
)
