# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Custom kernels

set(CUSTOM_KERNELS
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x384.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x512.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x640.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x768.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x896.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x1024.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x384.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x512.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x640.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x768.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x896.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x384.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x512.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x384.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_160x384.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_224x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.co"
    "${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.co"
)

if(NOT "${HIPBLASLT_TENSILE_LIBPATH}")
    set(CUSTOM_KERNEL_DST_DIR "${hipblaslt_BINARY_DIR}/Tensile/library")
else()
    set(CUSTOM_KERNEL_DST_DIR "${HIPBLASLT_TENSILE_LIBPATH}/library")
endif()

file(MAKE_DIRECTORY "${CUSTOM_KERNEL_DST_DIR}")

set(CUSTOM_KERNEL_OUTPUTS)
foreach(CUSTOM_KERNEL_FILE ${CUSTOM_KERNELS})
    get_filename_component(CUSTOM_KERNEL_FILENAME "${CUSTOM_KERNEL_FILE}" NAME)
    set(CUSTOM_KERNEL_DST "${CUSTOM_KERNEL_DST_DIR}/${CUSTOM_KERNEL_FILENAME}")
    add_custom_command(
        OUTPUT "${CUSTOM_KERNEL_DST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CUSTOM_KERNEL_FILE}" "${CUSTOM_KERNEL_DST}"
        DEPENDS "${CUSTOM_KERNEL_FILE}"
        COMMENT "Copying ${CUSTOM_KERNEL_FILENAME}"
        VERBATIM
    )
    list(APPEND CUSTOM_KERNEL_OUTPUTS "${CUSTOM_KERNEL_DST}")
endforeach()

add_custom_target(custom-kernels ALL
    DEPENDS ${CUSTOM_KERNEL_OUTPUTS}
)

add_dependencies(hipblaslt-rocroller custom-kernels)
