# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(hipsparse PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_axpby.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_dense2sparse.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_gather.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_rot.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_scatter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_sddmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_sparse2dense.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spgemm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spgemm_reuse.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spmm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spmv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spsm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spsv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hipsparse_spvv.cpp"
)
