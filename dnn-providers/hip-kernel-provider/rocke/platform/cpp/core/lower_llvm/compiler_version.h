// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Private native AUTO adapter. Mirrors runtime.comgr.CompilerInfo, but queries
 * a natively discovered candidate, not a caller-owned compilation handle. */
#ifndef ROCKE_LOWER_LLVM_COMPILER_VERSION_H
#define ROCKE_LOWER_LLVM_COMPILER_VERSION_H

namespace ckc
{
/* Compiler evidence from native AUTO candidate detection. All strings and the returned
 * record are borrowed, immutable after the query, and valid for process
 * lifetime. llvm_major==0 means the loaded compiler could not be queried;
 * NULL paths mean the dynamic loader could not report their origin.
 * requested_comgr is the loader input, not necessarily the resolved path. */
struct CompilerInfo
{
    unsigned llvm_major;
    unsigned llvm_minor;
    unsigned llvm_patch;
    const char* source;
    const char* requested_comgr;
    const char* comgr_path;
    const char* query_library_path;
};

/* Retain the first successfully loaded native COMGR candidate and query it once.
 * Failed loads return NULL and may be retried by a later call.
 * Explicit emission flavors do not call this function automatically. */
const CompilerInfo* candidate_compiler_info();

} // namespace ckc

#endif // ROCKE_LOWER_LLVM_COMPILER_VERSION_H
