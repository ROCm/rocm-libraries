# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

add_executable(mfma_layout mfma_layout.cpp)
target_link_libraries(mfma_layout PRIVATE mosaic)
target_compile_definitions(mfma_layout PRIVATE
    MFMA_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/matrix_calculator_layouts")

if(MOSAIC_BUILD_HIP_EXAMPLES)
    # v_mfma_f32_16x16x16f16 needs a CDNA target (gfx90a / gfx94x / gfx95x).
    # The HIP package defaults to gfx906 when no GPU is visible.
    set(_mosaic_mfma_archs "${CMAKE_HIP_ARCHITECTURES}")
    if(NOT _mosaic_mfma_archs)
        set(_mosaic_mfma_archs "${GPU_TARGETS}")
    endif()
    if(_mosaic_mfma_archs MATCHES "gfx90a|gfx94|gfx95")
        add_executable(mfma_example mfma_example.cpp)
        set_source_files_properties(mfma_example.cpp PROPERTIES LANGUAGE HIP)
        target_link_libraries(mfma_example PRIVATE mosaic hip::host)
    else()
        message(STATUS
            "mosaic: skipping mfma_example; HIP architectures '${_mosaic_mfma_archs}' "
            "do not include a CDNA target required by v_mfma_f32_16x16x16f16")
    endif()
endif()
