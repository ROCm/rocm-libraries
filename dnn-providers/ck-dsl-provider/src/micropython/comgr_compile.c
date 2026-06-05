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

#ifdef _WIN32
/* On Windows we do NOT import-link amd_comgr.dll. The import table records the
 * bare DLL name, which the loader resolves from C:\Windows\System32 *before* any
 * directory on %PATH%; a driver- or HIP-SDK-installed comgr there can use a
 * different amd_comgr_action_kind_t numbering (a later comgr inserts
 * ADD_DEVICE_LIBRARIES, shifting CODEGEN_BC_TO_RELOCATABLE and
 * LINK_RELOCATABLE_TO_EXECUTABLE), so the chain below silently produces the wrong
 * data kinds and every compile fails at compile_source_to_bc. Classic /delayload
 * would let us defer and redirect that bind, but it needs __delayLoadHelper2 from
 * MSVC's delayimp.lib, which this clang/LLVM-only toolchain does not ship. So we
 * resolve comgr the same way the CPython ctypes backend does (see
 * ck_dsl/runtime/hip_module.py): load the DLL that sits on %PATH% (the ROCm-SDK
 * wheel bin precedes System32 there) by full path -- which both wins the search
 * and pins the version -- and bind each entry point with GetProcAddress. A DLL
 * sitting beside this module (a deployed plugin or staged exe) is preferred. */
#include <string.h>
#include <windows.h>

/* The comgr entry points comgr_build_hsaco calls. */
#define CKDSL_COMGR_FNS(X)                   \
    X(amd_comgr_create_data_set)             \
    X(amd_comgr_create_data)                 \
    X(amd_comgr_set_data)                    \
    X(amd_comgr_set_data_name)               \
    X(amd_comgr_data_set_add)                \
    X(amd_comgr_create_action_info)          \
    X(amd_comgr_action_info_set_isa_name)    \
    X(amd_comgr_action_info_set_language)    \
    X(amd_comgr_action_info_set_option_list) \
    X(amd_comgr_do_action)                   \
    X(amd_comgr_action_data_count)           \
    X(amd_comgr_action_data_get_data)        \
    X(amd_comgr_get_data)                    \
    X(amd_comgr_status_string)               \
    X(amd_comgr_release_data)                \
    X(amd_comgr_destroy_data_set)            \
    X(amd_comgr_destroy_action_info)

/* One function pointer per entry point, typed off the real header declaration. */
#define CKDSL_DECL_PTR(name) static __typeof__(name)* p_##name = NULL;
CKDSL_COMGR_FNS(CKDSL_DECL_PTR)
#undef CKDSL_DECL_PTR

/* LoadLibraryEx "amd_comgr.dll" from `dir` (`dir_len` wchars, trailing separator
 * already appended; not necessarily NUL-terminated). Returns the module or NULL. */
static HMODULE ckdsl_try_dir(const wchar_t* dir, size_t dir_len) {
    static const wchar_t name[] = L"amd_comgr.dll";
    const size_t name_n = sizeof(name) / sizeof(name[0]); /* includes NUL */
    wchar_t path[MAX_PATH];
    if (dir_len + name_n > MAX_PATH) {
        return NULL;
    }
    memcpy(path, dir, dir_len * sizeof(wchar_t));
    memcpy(path + dir_len, name, sizeof(name)); /* copies the NUL too */
    if (GetFileAttributesW(path) == INVALID_FILE_ATTRIBUTES) {
        return NULL;
    }
    return LoadLibraryExW(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
}

static HMODULE ckdsl_load_comgr(void) {
    HMODULE h = NULL;

    /* 1. Beside this module (deployed plugin .dll / staged executable dir). */
    HMODULE self = NULL;
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCWSTR)(const void*)&ckdsl_load_comgr, &self)) {
        wchar_t mod[MAX_PATH];
        DWORD n = GetModuleFileNameW(self, mod, MAX_PATH);
        if (n > 0 && n < MAX_PATH) {
            while (n > 0 && mod[n - 1] != L'\\' && mod[n - 1] != L'/') {
                n--; /* strip the file name, keep the trailing separator */
            }
            if (n > 0) {
                h = ckdsl_try_dir(mod, n);
                if (h != NULL) {
                    return h;
                }
            }
        }
    }

    /* 2. The directories on %PATH%, in order (the wheel bin precedes System32). */
    DWORD len = GetEnvironmentVariableW(L"PATH", NULL, 0);
    if (len > 1) {
        wchar_t* path = (wchar_t*)malloc((size_t)len * sizeof(wchar_t));
        if (path != NULL) {
            GetEnvironmentVariableW(L"PATH", path, len);
            size_t start = 0;
            for (size_t i = 0;; i++) {
                if (path[i] == L';' || path[i] == L'\0') {
                    size_t seg = i - start;
                    if (seg > 0 && seg + 2 <= MAX_PATH) {
                        wchar_t dir[MAX_PATH];
                        memcpy(dir, path + start, seg * sizeof(wchar_t));
                        if (dir[seg - 1] != L'\\' && dir[seg - 1] != L'/') {
                            dir[seg++] = L'\\';
                        }
                        h = ckdsl_try_dir(dir, seg);
                        if (h != NULL) {
                            free(path);
                            return h;
                        }
                    }
                    start = i + 1;
                    if (path[i] == L'\0') {
                        break;
                    }
                }
            }
            free(path);
        }
    }

    /* 3. Last resort: the default search order (may bind a System32 comgr). */
    return LoadLibraryW(L"amd_comgr.dll");
}

static int g_comgr_state = 0; /* 0 = untried, 1 = ready, -1 = failed */
static const char* g_comgr_err = "";

/* Resolve comgr once. Serialized by the embedded interpreter mutex (modcomgr is
 * only entered under it), so no extra locking is needed. */
static int ckdsl_ensure_comgr(void) {
    if (g_comgr_state != 0) {
        return g_comgr_state > 0;
    }
    HMODULE h = ckdsl_load_comgr();
    if (h == NULL) {
        g_comgr_err = "amd_comgr.dll not found";
        g_comgr_state = -1;
        return 0;
    }
    int ok = 1;
#define CKDSL_BIND(name)                                       \
    p_##name = (__typeof__(name)*)GetProcAddress(h, #name);    \
    if (p_##name == NULL) {                                    \
        ok = 0;                                                \
    }
    CKDSL_COMGR_FNS(CKDSL_BIND)
#undef CKDSL_BIND
    if (!ok) {
        g_comgr_err = "amd_comgr.dll missing an expected export";
        g_comgr_state = -1;
        return 0;
    }
    g_comgr_state = 1;
    return 1;
}

/* Route the API names used in comgr_build_hsaco through the resolved pointers. */
#define amd_comgr_create_data_set p_amd_comgr_create_data_set
#define amd_comgr_create_data p_amd_comgr_create_data
#define amd_comgr_set_data p_amd_comgr_set_data
#define amd_comgr_set_data_name p_amd_comgr_set_data_name
#define amd_comgr_data_set_add p_amd_comgr_data_set_add
#define amd_comgr_create_action_info p_amd_comgr_create_action_info
#define amd_comgr_action_info_set_isa_name p_amd_comgr_action_info_set_isa_name
#define amd_comgr_action_info_set_language p_amd_comgr_action_info_set_language
#define amd_comgr_action_info_set_option_list p_amd_comgr_action_info_set_option_list
#define amd_comgr_do_action p_amd_comgr_do_action
#define amd_comgr_action_data_count p_amd_comgr_action_data_count
#define amd_comgr_action_data_get_data p_amd_comgr_action_data_get_data
#define amd_comgr_get_data p_amd_comgr_get_data
#define amd_comgr_status_string p_amd_comgr_status_string
#define amd_comgr_release_data p_amd_comgr_release_data
#define amd_comgr_destroy_data_set p_amd_comgr_destroy_data_set
#define amd_comgr_destroy_action_info p_amd_comgr_destroy_action_info
#endif /* _WIN32 */

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

#ifdef _WIN32
    if (!ckdsl_ensure_comgr()) {
        if (err != NULL && err_len > 0) {
            snprintf(err, err_len, "load amd_comgr.dll: %s", g_comgr_err);
        }
        return 1;
    }
#endif

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
