# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Custom kernels - build from assembly source files into a single combined .co

# Determine the destination directory for custom kernels
if(NOT "${HIPBLASLT_TENSILE_LIBPATH}")
    set(CUSTOM_KERNEL_DST_DIR "${hipblaslt_BINARY_DIR}/Tensile/library")
else()
    set(CUSTOM_KERNEL_DST_DIR "${HIPBLASLT_TENSILE_LIBPATH}/library")
endif()

file(MAKE_DIRECTORY "${CUSTOM_KERNEL_DST_DIR}")

find_program(ROCM_LLD
    NAMES ld.lld
    PATH_SUFFIXES lib/llvm/bin llvm/bin bin
)

# Create output directory for built files
set(BUILT_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/obj")
file(MAKE_DIRECTORY "${BUILT_OBJ_DIR}")

# Assembly source files for custom kernels (explicit list)
set(CUSTOM_KERNEL_ASM_SOURCES
    # Aiter kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x384.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x512.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x640.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x768.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x896.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x1024.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x384.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x512.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x640.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x768.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x896.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_64x1024.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x384.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x512.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_96x640.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x384.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_160x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_160x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_160x384.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_192x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_192x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_224x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_224x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/f4gemm_bf16_per1x32Fp4_noBpreShuffle_256x256.s
    # Wave kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_dynamic_gemm_64x64x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_dynamic_gemm_128x32x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_dynamic_gemm_224x160x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_dynamic_gemm_256x160x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_dynamic_gemm_256x192x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_dynamic_gemm_256x224x256.s
    # Static wave kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_64x64x256_256x128x256.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_64x64x256_256x128x1024.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_64x64x256_256x640x768.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_64x64x256_1792x512x7424.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_64x64x256_896x7040x1280.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_64x64x256_2048x2048x16384.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_128x256x256_1024x1024x8192.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x160x256_3200x1536x10496.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x160x256_10880x1024x28416.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x192x256_5376x1792x4096.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x192x256_6144x2048x4608.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x192x256_14976x51712x7680.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x192x256_15360x9984x13824.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x192x256_31104x16896x7168.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x192x256_43392x13056x1792.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.s
    ${CMAKE_CURRENT_SOURCE_DIR}/wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.s
)

# Build combined .co from all assembly sources
if(ROCM_LLD)
    message(STATUS "Building custom kernels from assembly sources")
    message(STATUS "  ROCm clang: ${ROCM_CLANG}")
    message(STATUS "  ROCm ld.lld: ${ROCM_LLD}")
    message(STATUS "  GPU arch: ${CUSTOM_KERNEL_GPU_ARCH}")
    
    # Assemble each .s file into a .o file
    set(BUILT_OBJ_FILES)
    foreach(ASM_FILE ${CUSTOM_KERNEL_ASM_SOURCES})
        get_filename_component(ASM_NAME "${ASM_FILE}" NAME_WE)
        set(OBJ_OUTPUT "${BUILT_OBJ_DIR}/${ASM_NAME}.o")
        
        add_custom_command(
            OUTPUT "${OBJ_OUTPUT}"
            COMMAND ${CMAKE_CXX_COMPILER}
                -x assembler
                -target amdgcn-amd-amdhsa
                "-mcpu=gfx950"
                -c
                "${ASM_FILE}"
                -o "${OBJ_OUTPUT}"
            DEPENDS "${ASM_FILE}"
            COMMENT "Assembling ${ASM_NAME}.s"
            VERBATIM
        )
        list(APPEND BUILT_OBJ_FILES "${OBJ_OUTPUT}")
    endforeach()
    
    # Link all .o files into a single combined .co file
    set(COMBINED_CO_FILE "${BUILT_OBJ_DIR}/rr_custom_kernels.co")
    set(COMBINED_CO_DST "${CUSTOM_KERNEL_DST_DIR}/rr_custom_kernels.co")
    
    add_custom_command(
        OUTPUT "${COMBINED_CO_FILE}"
        COMMAND ${ROCM_LLD}
            -shared
            -o "${COMBINED_CO_FILE}"
            ${BUILT_OBJ_FILES}
        DEPENDS ${BUILT_OBJ_FILES}
        COMMENT "Linking all kernel objects into rr_custom_kernels.co"
        VERBATIM
    )
    
    add_custom_command(
        OUTPUT "${COMBINED_CO_DST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${COMBINED_CO_FILE}" "${COMBINED_CO_DST}"
        DEPENDS "${COMBINED_CO_FILE}"
        COMMENT "Copying rr_custom_kernels.co to ${CUSTOM_KERNEL_DST_DIR}"
        VERBATIM
    )
    
    add_custom_target(custom-kernels ALL
        DEPENDS "${COMBINED_CO_DST}"
    )
    
    message(STATUS "  Combined output: ${COMBINED_CO_DST}")
else()
    if(NOT ROCM_LLD)
        message(FATAL_ERROR "ROCm ld.lld not found, cannot build custom kernels from assembly")
    endif()
endif()

add_dependencies(hipblaslt-rocroller custom-kernels)
