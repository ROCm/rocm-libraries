/* C++-side comgr wrapper for Arch A: IR text -> HSACO. Cross-platform (amd_comgr
 * ships on Linux + Windows HIP SDK). Mirrors the Phase-0 ffi spike's 3-stage chain. */
#include "comgr_compile.h"

#include <amd_comgr/amd_comgr.h>
#include <stdlib.h>

#define OK(call)                                                 \
    do {                                                         \
        if ((call) != AMD_COMGR_STATUS_SUCCESS) return __LINE__; \
    } while (0)

int comgr_build_hsaco(const char* ir, size_t ir_len, const char* isa, const char** options,
                      size_t n_options, unsigned char** out, size_t* out_len) {
    *out = NULL;
    *out_len = 0;
    amd_comgr_data_set_t in_set, bc_set, reloc_set, exe_set;
    amd_comgr_data_t src, data;
    amd_comgr_action_info_t info;

    OK(amd_comgr_create_data_set(&in_set));
    OK(amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src));
    OK(amd_comgr_set_data(src, ir_len, ir));
    OK(amd_comgr_set_data_name(src, "kernel.ll"));
    OK(amd_comgr_data_set_add(in_set, src));

    OK(amd_comgr_create_action_info(&info));
    OK(amd_comgr_action_info_set_isa_name(info, isa));
    OK(amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR));
    OK(amd_comgr_action_info_set_option_list(info, options, n_options));

    OK(amd_comgr_create_data_set(&bc_set));
    OK(amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc_set));
    OK(amd_comgr_create_data_set(&reloc_set));
    OK(amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set));
    OK(amd_comgr_create_data_set(&exe_set));
    OK(amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc_set,
                           exe_set));

    size_t count = 0;
    OK(amd_comgr_action_data_count(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, &count));
    if (count == 0) return -1;
    OK(amd_comgr_action_data_get_data(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &data));
    size_t sz = 0;
    OK(amd_comgr_get_data(data, &sz, NULL));
    unsigned char* buf = (unsigned char*)malloc(sz);
    if (!buf) return -2;
    OK(amd_comgr_get_data(data, &sz, (char*)buf));
    *out = buf;
    *out_len = sz;

    amd_comgr_release_data(data);
    amd_comgr_release_data(src);
    amd_comgr_destroy_data_set(in_set);
    amd_comgr_destroy_data_set(bc_set);
    amd_comgr_destroy_data_set(reloc_set);
    amd_comgr_destroy_data_set(exe_set);
    amd_comgr_destroy_action_info(info);
    return 0;
}
