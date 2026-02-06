# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Custom kernels

set(CUSTOM_KERNELS
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co"
)

set(CUSTOM_KERNEL_DST_DIR "${HIPBLASLT_TENSILE_LIBPATH}")
if(NOT CUSTOM_KERNEL_DST_DIR)
    set(CUSTOM_KERNEL_DST_DIR "${hipblaslt_BINARY_DIR}/Tensile/library")
endif()

file(MAKE_DIRECTORY "${CUSTOM_KERNEL_DST_DIR}")

set(CUSTOM_KERNEL_OUTPUTS)
foreach(CUSTOM_KERNEL_FILE ${CUSTOM_KERNELS})
    get_filename_component(CUSTOM_KERNEL_FILENAME "${CUSTOM_KERNEL_FILE}" NAME)
    set(CUSTOM_KERNEL_DST "${CUSTOM_KERNEL_DST_DIR}/${CUSTOM_KERNEL_FILENAME}")
    add_custom_command(
        OUTPUT "${CUSTOM_KERNEL_DST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUSTOM_KERNEL_PATH}" "${CUSTOM_KERNEL_DST}"
        DEPENDS "${CUSTOM_KERNEL_PATH}"
        COMMENT "Copying ${CUSTOM_KERNEL_FILENAME}"
        VERBATIM
    )
    list(APPEND ${CUSTOM_KERNEL_OUTPUTS} "${CUSTOM_KERNEL_DST}")
endforeach()

add_custom_target(custom-kernels ALL
    DEPENDS ${CUSTOM_KERNEL_OUTPUTS}
)
