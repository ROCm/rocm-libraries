# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# _tl_emit nanobind module — compiled emission assets for TensileLite.
#
# This module builds only when rocisa's nanobind extension also builds
# (HIPBLASLT_BUNDLE_PYTHON_DEPS or ROCISA_STANDALONE), since it links
# against rocisa's C++ types and shares the same nanobind version.

if(NOT TARGET _rocisa)
    message(STATUS "tl_emit: skipping (no _rocisa target)")
    return()
endif()

nanobind_add_module(_tl_emit NOMINSIZE NB_SUPPRESS_WARNINGS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/emit_mfma.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/rocisa_shim.cpp"
)

target_include_directories(_tl_emit PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../rocisa/rocisa/include"
)

target_link_libraries(_tl_emit PRIVATE roc::origami hip::host)

set_target_properties(_tl_emit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    INSTALL_RPATH "$ORIGIN"
)

if(ROCISA_STANDALONE)
    install(TARGETS _tl_emit DESTINATION tl_emit)
endif()
