# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

function(mx_flatmm_instance_generate FILE_LIST)
    set(C_DATA_TYPE FP16)
    set(A_LAYOUT ROW)
    set(B_LAYOUT COL)
    set(C_LAYOUT ROW)

    set(MXFLATMM_ARCH)

    if (GPU_TARGETS MATCHES "gfx95")
        list(APPEND MXFLATMM_ARCH MXFlatmm_GFX950_)
    endif()

    # foreach(PERSISTENT false true)
    # TODO: Persistent kernels are disabled due to compilation failures with some LLVM versions.
    foreach(PERSISTENT false)
        # [v3 P0.1] FP6xFP6_Sync is a sync-load variant of FP6xFP6. The trailing
        # _Sync suffix is preserved through to the generated traits name
        # (MXFlatmm_GFX950_FP6FP6_Sync_Traits) but stripped before the AxB split
        # so A_DATA_TYPE/B_DATA_TYPE still resolve to plain FP6/FP6.
        foreach(DATA_TYPE FP4xFP4 FP8xFP8 FP6xFP6 FP8xFP4 FP4xFP8 FP6xFP6_Sync16 FP6xFP6_Sync FP6xFP6_K512)
            set(VARIANT_SUFFIX "")
            set(DATA_TYPE_BASE ${DATA_TYPE})
            if(${DATA_TYPE} MATCHES "_Sync16$")
                string(REGEX REPLACE "_Sync16$" "" DATA_TYPE_BASE ${DATA_TYPE})
                set(VARIANT_SUFFIX "_Sync16")
            elseif(${DATA_TYPE} MATCHES "_Sync$")
                string(REGEX REPLACE "_Sync$" "" DATA_TYPE_BASE ${DATA_TYPE})
                set(VARIANT_SUFFIX "_Sync")
            elseif(${DATA_TYPE} MATCHES "_K512$")
                string(REGEX REPLACE "_K512$" "" DATA_TYPE_BASE ${DATA_TYPE})
                set(VARIANT_SUFFIX "_K512")
            endif()
            string(REPLACE "x" ";" DATA_TYPE_AB ${DATA_TYPE_BASE})
            list(GET DATA_TYPE_AB 0 A_DATA_TYPE)
            list(GET DATA_TYPE_AB 1 B_DATA_TYPE)
            foreach(ARCH ${MXFLATMM_ARCH})
                set(MXFLATMM_ARCH_TRAITS "${ARCH}${A_DATA_TYPE}${B_DATA_TYPE}${VARIANT_SUFFIX}_Traits")
                foreach(SPLIT_K false true)
                    foreach(HAS_HOT_LOOP false true)
                        foreach(TAIL_NUMBER ODD EVEN)
                            set(KERNEL_FILE mxgemm/instance_${ARCH}${DATA_TYPE}_${PERSISTENT}_${SPLIT_K}_${HAS_HOT_LOOP}_${TAIL_NUMBER}.cpp)
                            string(TOLOWER ${KERNEL_FILE} KERNEL_FILE)
                            configure_file(
                                ${CMAKE_CURRENT_SOURCE_DIR}/mxgemm/mx_flatmm_instance.cpp.in
                                ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE}
                                @ONLY)
                            list(APPEND ${FILE_LIST} ${CMAKE_CURRENT_BINARY_DIR}/${KERNEL_FILE})
                        endforeach()
                    endforeach()
                endforeach()
            endforeach()
        endforeach()
    endforeach()
    set(${FILE_LIST} ${${FILE_LIST}} PARENT_SCOPE)
endfunction()
