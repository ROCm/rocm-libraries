# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# C++ VARIANTS (from *_cpp.cpp files)
add_executable(example_hipsparse_axpby_cpp)
target_sources(example_hipsparse_axpby_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_axpby_cpp.cpp")
target_link_libraries(example_hipsparse_axpby_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_dense_to_sparse_cpp)
target_sources(example_hipsparse_dense_to_sparse_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_dense_to_sparse_cpp.cpp")
target_link_libraries(example_hipsparse_dense_to_sparse_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gather_cpp)
target_sources(example_hipsparse_gather_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gather_cpp.cpp")
target_link_libraries(example_hipsparse_gather_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_rot_cpp)
target_sources(example_hipsparse_rot_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_rot_cpp.cpp")
target_link_libraries(example_hipsparse_rot_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_scatter_cpp)
target_sources(example_hipsparse_scatter_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_scatter_cpp.cpp")
target_link_libraries(example_hipsparse_scatter_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_sddmm_cpp)
target_sources(example_hipsparse_sddmm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sddmm_cpp.cpp")
target_link_libraries(example_hipsparse_sddmm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_sparse_to_dense_cpp)
target_sources(example_hipsparse_sparse_to_dense_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sparse_to_dense_cpp.cpp")
target_link_libraries(example_hipsparse_sparse_to_dense_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spgemm_cpp)
target_sources(example_hipsparse_spgemm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spgemm_cpp.cpp")
target_link_libraries(example_hipsparse_spgemm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spgemm_reuse_cpp)
target_sources(example_hipsparse_spgemm_reuse_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spgemm_reuse_cpp.cpp")
target_link_libraries(example_hipsparse_spgemm_reuse_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spmm_cpp)
target_sources(example_hipsparse_spmm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spmm_cpp.cpp")
target_link_libraries(example_hipsparse_spmm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spmv_cpp)
target_sources(example_hipsparse_spmv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spmv_cpp.cpp")
target_link_libraries(example_hipsparse_spmv_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spsm_cpp)
target_sources(example_hipsparse_spsm_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spsm_cpp.cpp")
target_link_libraries(example_hipsparse_spsm_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spsv_cpp)
target_sources(example_hipsparse_spsv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spsv_cpp.cpp")
target_link_libraries(example_hipsparse_spsv_cpp PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spvv_cpp)
target_sources(example_hipsparse_spvv_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spvv_cpp.cpp")
target_link_libraries(example_hipsparse_spvv_cpp PRIVATE hipsparse::samples-common)

# C VARIANTS (from *_c.c files)
add_executable(example_hipsparse_axpby_c)
target_sources(example_hipsparse_axpby_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_axpby_c.c")
target_link_libraries(example_hipsparse_axpby_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_dense_to_sparse_c)
target_sources(example_hipsparse_dense_to_sparse_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_dense_to_sparse_c.c")
target_link_libraries(example_hipsparse_dense_to_sparse_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_gather_c)
target_sources(example_hipsparse_gather_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_gather_c.c")
target_link_libraries(example_hipsparse_gather_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_rot_c)
target_sources(example_hipsparse_rot_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_rot_c.c")
target_link_libraries(example_hipsparse_rot_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_scatter_c)
target_sources(example_hipsparse_scatter_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_scatter_c.c")
target_link_libraries(example_hipsparse_scatter_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_sddmm_c)
target_sources(example_hipsparse_sddmm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sddmm_c.c")
target_link_libraries(example_hipsparse_sddmm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_sparse_to_dense_c)
target_sources(example_hipsparse_sparse_to_dense_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_sparse_to_dense_c.c")
target_link_libraries(example_hipsparse_sparse_to_dense_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spgemm_c)
target_sources(example_hipsparse_spgemm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spgemm_c.c")
target_link_libraries(example_hipsparse_spgemm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spgemm_reuse_c)
target_sources(example_hipsparse_spgemm_reuse_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spgemm_reuse_c.c")
target_link_libraries(example_hipsparse_spgemm_reuse_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spmm_c)
target_sources(example_hipsparse_spmm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spmm_c.c")
target_link_libraries(example_hipsparse_spmm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spmv_c)
target_sources(example_hipsparse_spmv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spmv_c.c")
target_link_libraries(example_hipsparse_spmv_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spsm_c)
target_sources(example_hipsparse_spsm_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spsm_c.c")
target_link_libraries(example_hipsparse_spsm_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spsv_c)
target_sources(example_hipsparse_spsv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spsv_c.c")
target_link_libraries(example_hipsparse_spsv_c PRIVATE hipsparse::samples-common)

add_executable(example_hipsparse_spvv_c)
target_sources(example_hipsparse_spvv_c PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/example_hipsparse_spvv_c.c")
target_link_libraries(example_hipsparse_spvv_c PRIVATE hipsparse::samples-common)

