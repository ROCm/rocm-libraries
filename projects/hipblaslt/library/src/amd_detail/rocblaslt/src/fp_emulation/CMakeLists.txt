# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

# Fixed-Point Emulation (FP32/FP64) via Ozaki Scheme II (INT8 tensor cores + CRT).
#
# This translation unit is built as a standalone OBJECT library so that its
# object files can be linked PRIVATE into both the hipblaslt shared library
# (where the symbols stay hidden, leaving the public ABI unchanged) and the
# hipblaslt-test harness (which needs to call the internal entry points
# directly).  This avoids exporting internal symbols from the release .so.
add_library(hipblaslt-fixed-point-emulation OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/emulation.cpp"
)
set_target_properties(hipblaslt-fixed-point-emulation PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_compile_features(hipblaslt-fixed-point-emulation PRIVATE cxx_std_17)
target_link_libraries(hipblaslt-fixed-point-emulation PRIVATE
    hipblaslt::headers
    hip::device
    roc::${hipblas_target}
)
# Inherit all compile definitions from hipblaslt (includes HIPBLASLT_USE_ROCROLLER,
# ROCM_USE_FLOAT16, __HIP_HCC_COMPAT_MODE__, etc.)
target_compile_definitions(hipblaslt-fixed-point-emulation PRIVATE
    $<TARGET_PROPERTY:hipblaslt,COMPILE_DEFINITIONS>)
target_include_directories(hipblaslt-fixed-point-emulation PRIVATE
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/include"
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/rocblaslt/src/include"
    "${PROJECT_SOURCE_DIR}/library/src/amd_detail/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(hipblaslt PRIVATE hipblaslt-fixed-point-emulation)
