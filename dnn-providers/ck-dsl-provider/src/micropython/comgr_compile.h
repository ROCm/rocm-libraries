#ifndef CKDSL_COMGR_COMPILE_H
#define CKDSL_COMGR_COMPILE_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Compile LLVM IR text -> HSACO via libamd_comgr. Returns 0 on success and sets
 * *out (malloc'd, caller frees) + *out_len. Non-zero = failure at that stage. */
int comgr_build_hsaco(const char* ir, size_t ir_len, const char* isa, const char** options,
                      size_t n_options, unsigned char** out, size_t* out_len);
#ifdef __cplusplus
}
#endif
#endif
