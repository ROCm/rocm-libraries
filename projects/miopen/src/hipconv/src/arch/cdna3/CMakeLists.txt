include(GpuArch)

hipconv_get_matching_gpu("${HIPCONV_LOGICAL_TO_GPU}" "gfx942" CDNA3_GPUS)

if(CDNA3_GPUS)
    hipconv_add_arch_lib("cdna3" "${CDNA3_GPUS}")
endif()
