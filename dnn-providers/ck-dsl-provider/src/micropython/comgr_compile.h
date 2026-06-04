#ifndef CKDSL_COMGR_COMPILE_H
#define CKDSL_COMGR_COMPILE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Compile LLVM IR text -> HSACO via libamd_comgr. Returns 0 on success and sets
 * *out (malloc'd, caller frees) + *out_len. Returns non-zero on failure (with
 * *out == NULL); on failure, when err != NULL, a short "<stage>: <comgr status>"
 * description is written to err (NUL-terminated, truncated to err_len). All comgr
 * handles are released on every path. */
int comgr_build_hsaco(const char* ir, size_t ir_len, const char* isa, const char** options,
                      size_t n_options, unsigned char** out, size_t* out_len, char* err,
                      size_t err_len);
#ifdef __cplusplus
}
#endif
#endif
