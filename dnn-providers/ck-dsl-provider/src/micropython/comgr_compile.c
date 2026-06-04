/* C++-side comgr wrapper for Arch A: IR text -> HSACO. Cross-platform (amd_comgr
 * ships on Linux + Windows HIP SDK). Mirrors the Phase-0 ffi spike's 3-stage chain.
 *
 * Every comgr handle is released on every exit path (success and failure) -- comgr
 * compile failures are a normal outcome in a JIT path and hipDNN CI runs sanitizer
 * builds, so leaking on the error path is not acceptable. */
#include "comgr_compile.h"

#include <amd_comgr/amd_comgr.h>
#include <stdio.h>
#include <stdlib.h>

// Run a comgr call; on failure record the stage + status and jump to cleanup.
#define STEP(name, call)                          \
    do {                                          \
        stage = (name);                           \
        status = (call);                          \
        if (status != AMD_COMGR_STATUS_SUCCESS) { \
            rc = 1;                               \
            goto cleanup;                         \
        }                                         \
    } while (0)

int comgr_build_hsaco(const char* ir, size_t ir_len, const char* isa, const char** options,
                      size_t n_options, unsigned char** out, size_t* out_len, char* err,
                      size_t err_len) {
    *out = NULL;
    *out_len = 0;

    amd_comgr_data_set_t in_set, bc_set, reloc_set, exe_set;
    amd_comgr_data_t src, data;
    amd_comgr_action_info_t info;
    int have_in = 0, have_bc = 0, have_reloc = 0, have_exe = 0;
    int have_src = 0, have_data = 0, have_info = 0;

    amd_comgr_status_t status = AMD_COMGR_STATUS_SUCCESS;
    const char* stage = "";
    int rc = 0;
    unsigned char* buf = NULL;
    size_t count = 0;
    size_t sz = 0;

    STEP("create_data_set(in)", amd_comgr_create_data_set(&in_set));
    have_in = 1;
    STEP("create_data(src)", amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &src));
    have_src = 1;
    STEP("set_data", amd_comgr_set_data(src, ir_len, ir));
    STEP("set_data_name", amd_comgr_set_data_name(src, "kernel.ll"));
    STEP("data_set_add", amd_comgr_data_set_add(in_set, src));

    STEP("create_action_info", amd_comgr_create_action_info(&info));
    have_info = 1;
    STEP("set_isa_name", amd_comgr_action_info_set_isa_name(info, isa));
    STEP("set_language", amd_comgr_action_info_set_language(info, AMD_COMGR_LANGUAGE_LLVM_IR));
    STEP("set_option_list", amd_comgr_action_info_set_option_list(info, options, n_options));

    STEP("create_data_set(bc)", amd_comgr_create_data_set(&bc_set));
    have_bc = 1;
    STEP("compile_source_to_bc",
         amd_comgr_do_action(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC, info, in_set, bc_set));
    STEP("create_data_set(reloc)", amd_comgr_create_data_set(&reloc_set));
    have_reloc = 1;
    STEP("codegen_bc_to_relocatable",
         amd_comgr_do_action(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE, info, bc_set, reloc_set));
    STEP("create_data_set(exe)", amd_comgr_create_data_set(&exe_set));
    have_exe = 1;
    STEP("link_relocatable_to_executable",
         amd_comgr_do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, info, reloc_set,
                             exe_set));

    STEP("action_data_count",
         amd_comgr_action_data_count(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, &count));
    if (count == 0) {
        stage = "no executable produced";
        rc = 1;
        goto cleanup;
    }
    STEP("action_data_get_data",
         amd_comgr_action_data_get_data(exe_set, AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &data));
    have_data = 1;
    STEP("get_data(size)", amd_comgr_get_data(data, &sz, NULL));
    buf = (unsigned char*)malloc(sz ? sz : 1);
    if (buf == NULL) {
        stage = "malloc";
        rc = 1;
        goto cleanup;
    }
    STEP("get_data(bytes)", amd_comgr_get_data(data, &sz, (char*)buf));
    *out = buf;
    *out_len = sz;
    buf = NULL;  // ownership transferred to the caller

cleanup:
    if (rc != 0 && err != NULL && err_len > 0) {
        const char* statusStr = NULL;
        if (status != AMD_COMGR_STATUS_SUCCESS) {
            amd_comgr_status_string(status, &statusStr);
        }
        if (statusStr != NULL) {
            snprintf(err, err_len, "%s: %s", stage, statusStr);
        } else {
            snprintf(err, err_len, "%s", stage);
        }
    }
    if (buf != NULL) {
        free(buf);
    }
    if (have_data) amd_comgr_release_data(data);
    if (have_src) amd_comgr_release_data(src);
    if (have_in) amd_comgr_destroy_data_set(in_set);
    if (have_bc) amd_comgr_destroy_data_set(bc_set);
    if (have_reloc) amd_comgr_destroy_data_set(reloc_set);
    if (have_exe) amd_comgr_destroy_data_set(exe_set);
    if (have_info) amd_comgr_destroy_action_info(info);
    return rc;
}
