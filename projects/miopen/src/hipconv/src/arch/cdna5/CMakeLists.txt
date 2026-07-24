include(GpuArch)

hipconv_get_matching_gpu("${HIPCONV_LOGICAL_TO_GPU}" "gfx1250" CDNA5_GPUS)

if(CDNA5_GPUS)
    hipconv_add_arch_lib("cdna5" "${CDNA5_GPUS}")
endif()
