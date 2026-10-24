# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

set(INST_TARGETS ${SUPPORTED_GPU_TARGETS})
set(GPU_TARGETS ${SUPPORTED_GPU_TARGETS})

message(STATUS "Sparse Attention: SUPPORTED_GPU_TARGETS=${SUPPORTED_GPU_TARGETS}, INST_TARGETS=${INST_TARGETS}")

# Sparse attention kernels use the async LDS buffer-load path (buffer_load_dword ... lds),
# which is a gfx9-only instruction. Restrict instances to gfx9 targets.
list(FILTER INST_TARGETS INCLUDE REGEX "gfx9")
if(NOT INST_TARGETS)
  message(WARNING "Skipping Tile Engine Sparse Attention: No supported gfx9 GPU targets found")
  return()
endif()

message(STATUS "Building Sparse Attention (Jenga, VSA, Sparge & Sparge-Sage) for targets: ${INST_TARGETS}")

file(GLOB_RECURSE CODE_GEN_SCRIPTS CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  ${CMAKE_CURRENT_LIST_DIR}/codegen/*.py
)
set_directory_properties(PROPERTIES CMAKE_CONFIGURE_DEPENDS "${CODE_GEN_SCRIPTS}")

# Jenga
set(SPARSE_ATTN_JENGA_CODE_GEN_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd_jenga
  --receipt 600
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_JENGA_CODE_GEN_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/jenga_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to generate Jenga kernel list")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/jenga_blob_list.txt SPARSE_ATTN_JENGA_GEN_BLOBS)

add_custom_command(
  OUTPUT ${SPARSE_ATTN_JENGA_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_JENGA_CODE_GEN_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile Jenga Sparse Attention kernels"
)

set(SPARSE_ATTN_JENGA_INSTANCES "tile_sparse_attn_jenga_instances")

add_library(${SPARSE_ATTN_JENGA_INSTANCES} OBJECT EXCLUDE_FROM_ALL
  ${SPARSE_ATTN_JENGA_GEN_BLOBS}
)
target_include_directories(${SPARSE_ATTN_JENGA_INSTANCES} PRIVATE 
  ${CMAKE_CURRENT_LIST_DIR}
  ${PROJECT_SOURCE_DIR}/include/ck_tile/ops/sparse_attn
)
set_source_files_properties(${SPARSE_ATTN_JENGA_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_property(TARGET ${SPARSE_ATTN_JENGA_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

target_compile_options(${SPARSE_ATTN_JENGA_INSTANCES} PRIVATE
  -DCK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
  -DCK_TILE_FMHA_FWD_FAST_EXP2
  -Wno-undefined-func-template
  -Wno-float-equal
)

# VSA
set(SPARSE_ATTN_VSA_CODE_GEN_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd_vsa
  --receipt 600
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_VSA_CODE_GEN_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/vsa_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to generate VSA kernel list")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/vsa_blob_list.txt SPARSE_ATTN_VSA_GEN_BLOBS)

add_custom_command(
  OUTPUT ${SPARSE_ATTN_VSA_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_VSA_CODE_GEN_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile VSA Sparse Attention kernels"
)

set(SPARSE_ATTN_VSA_INSTANCES "tile_sparse_attn_vsa_instances")

add_library(${SPARSE_ATTN_VSA_INSTANCES} OBJECT EXCLUDE_FROM_ALL
  ${SPARSE_ATTN_VSA_GEN_BLOBS}
)
target_include_directories(${SPARSE_ATTN_VSA_INSTANCES} PRIVATE 
  ${CMAKE_CURRENT_LIST_DIR}
  ${PROJECT_SOURCE_DIR}/include/ck_tile/ops/sparse_attn
)
set_source_files_properties(${SPARSE_ATTN_VSA_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_property(TARGET ${SPARSE_ATTN_VSA_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

target_compile_options(${SPARSE_ATTN_VSA_INSTANCES} PRIVATE
  -DCK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
  -DCK_TILE_FMHA_FWD_FAST_EXP2
  -Wno-undefined-func-template
  -Wno-float-equal
)

# Sparge
set(SPARSE_ATTN_SPARGE_CODE_GEN_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd_sparge
  --receipt 600
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_SPARGE_CODE_GEN_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/sparge_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to generate Sparge kernel list")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/sparge_blob_list.txt SPARSE_ATTN_SPARGE_GEN_BLOBS)

add_custom_command(
  OUTPUT ${SPARSE_ATTN_SPARGE_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_SPARGE_CODE_GEN_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile SpargeAttention kernels"
)

set(SPARSE_ATTN_SPARGE_INSTANCES "tile_sparse_attn_sparge_instances")

add_library(${SPARSE_ATTN_SPARGE_INSTANCES} OBJECT EXCLUDE_FROM_ALL
  ${SPARSE_ATTN_SPARGE_GEN_BLOBS}
)
target_include_directories(${SPARSE_ATTN_SPARGE_INSTANCES} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${PROJECT_SOURCE_DIR}/include/ck_tile/ops/sparse_attn
)
set_source_files_properties(${SPARSE_ATTN_SPARGE_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_property(TARGET ${SPARSE_ATTN_SPARGE_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

target_compile_options(${SPARSE_ATTN_SPARGE_INSTANCES} PRIVATE
  -DCK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
  -DCK_TILE_FMHA_FWD_FAST_EXP2
  -Wno-undefined-func-template
  -Wno-float-equal
)

# Sparge_sage (quantized sparge)
set(SPARSE_ATTN_SPARGE_SAGE_CODE_GEN_ARGS
  ${CMAKE_CURRENT_LIST_DIR}/generate.py
  --api fwd_sparge_sage
  --receipt 600
)

execute_process(
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_SPARGE_SAGE_CODE_GEN_ARGS}
  --list_blobs ${CMAKE_CURRENT_BINARY_DIR}/sparge_sage_blob_list.txt
  RESULT_VARIABLE ret
)
if(ret AND NOT ret EQUAL 0)
  message(FATAL_ERROR "Failed to generate sparge_sage kernel list")
endif()

file(STRINGS ${CMAKE_CURRENT_BINARY_DIR}/sparge_sage_blob_list.txt SPARSE_ATTN_SPARGE_SAGE_GEN_BLOBS)

add_custom_command(
  OUTPUT ${SPARSE_ATTN_SPARGE_SAGE_GEN_BLOBS}
  COMMAND ${Python3_EXECUTABLE} ${SPARSE_ATTN_SPARGE_SAGE_CODE_GEN_ARGS}
  --output_dir ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CODE_GEN_SCRIPTS}
  COMMENT "Generate CK Tile sparge_sage Sparse Attention kernels"
)

set(SPARSE_ATTN_SPARGE_SAGE_INSTANCES "tile_sparse_attn_sparge_sage_instances")

add_library(${SPARSE_ATTN_SPARGE_SAGE_INSTANCES} OBJECT EXCLUDE_FROM_ALL
  ${SPARSE_ATTN_SPARGE_SAGE_GEN_BLOBS}
)
target_include_directories(${SPARSE_ATTN_SPARGE_SAGE_INSTANCES} PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}
  ${PROJECT_SOURCE_DIR}/include/ck_tile/ops/sparse_attn
)
set_source_files_properties(${SPARSE_ATTN_SPARGE_SAGE_GEN_BLOBS} PROPERTIES LANGUAGE HIP)
set_property(TARGET ${SPARSE_ATTN_SPARGE_SAGE_INSTANCES} PROPERTY HIP_ARCHITECTURES ${INST_TARGETS})

target_compile_options(${SPARSE_ATTN_SPARGE_SAGE_INSTANCES} PRIVATE
  -DCK_TILE_USE_BUFFER_ADDRESSING_BUILTIN
  -DCK_TILE_FMHA_FWD_FAST_EXP2
  -Wno-undefined-func-template
  -Wno-float-equal
)

# Unified example executable
set(EXAMPLE_SPARSE_ATTN_FWD "tile_example_sparse_attn_fwd")
message(DEBUG "adding unified example ${EXAMPLE_SPARSE_ATTN_FWD}")
add_executable(${EXAMPLE_SPARSE_ATTN_FWD} EXCLUDE_FROM_ALL
  example_sparse_attn_fwd.cpp
  sparse_attention.cpp
)
target_link_libraries(${EXAMPLE_SPARSE_ATTN_FWD}
  ${SPARSE_ATTN_SPARGE_SAGE_INSTANCES}
  ${SPARSE_ATTN_SPARGE_INSTANCES}
  ${SPARSE_ATTN_VSA_INSTANCES}
  ${SPARSE_ATTN_JENGA_INSTANCES}
)
target_include_directories(${EXAMPLE_SPARSE_ATTN_FWD} PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_options(${EXAMPLE_SPARSE_ATTN_FWD} PRIVATE
  -Wno-undefined-func-template
  -Wno-float-equal
)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
