include(GpuArch)

hipconv_get_matching_gpu("${HIPCONV_LOGICAL_TO_GPU}" "gfx950" CDNA4_GPUS)

if(CDNA4_GPUS)
    hipconv_add_arch_lib("cdna4" "${CDNA4_GPUS}")
endif()
