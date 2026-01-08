# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

target_sources(rocsparse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_axpby.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_axpyi.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_dotci.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_doti.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_gather.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_gthr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_gthr_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_gthrz.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_rot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_roti.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_scatter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_sctr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_sctr_impl.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rocsparse_spvv.cpp
)
