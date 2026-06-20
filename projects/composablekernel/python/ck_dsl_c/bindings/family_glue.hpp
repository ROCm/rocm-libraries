// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/family_glue.hpp -- macros that emit the canonical per-family
 * serialize-IR and verify function bodies for the ckc_engine module.
 *
 * Every family follows the same three-step shape: build a spec from the Python
 * dict (keeping any string fields alive in a local store), build a kernel with a
 * sticky-error IRBuilder, then either serialize the kernel to ck.dsl.ir/v1 text
 * or run the IR verifier. The build-failure guard, the IRBuilder lifetime, and
 * the serialize/verify tail are identical across families; only the spec type,
 * the spec-builder call, and the kernel-builder call differ.
 *
 * These macros expand to exactly the hand-written body, so the kernel object the
 * binding produces (and therefore its emitted IR) is unchanged. The kernel-build
 * call is passed as a trailing token sequence so build forms that take extra
 * arguments (an arch, a trailing option pointer, or no arch at all) all share
 * the same body.
 *
 * Each macro assumes the enclosing function signature
 *   (const py::dict& d, const std::string& arch)
 * and that the surrounding namespace provides serialize_kernel / verify_kernel /
 * arch_or_default exactly as defined in ckc_engine.cpp.
 */
#ifndef CKC_BINDINGS_FAMILY_GLUE_HPP
#define CKC_BINDINGS_FAMILY_GLUE_HPP

/* serialize-IR body for a family whose kernel is built with a local IRBuilder
 * `b`. FN is the fully-qualified entry name used in diagnostics, SPECT is the
 * spec struct type, SPECBUILD is the spec-builder symbol called as
 * SPECBUILD(d, store), and the kernel-build expression is the trailing tokens
 * (which may reference &b, &s, arch_or_default(arch), etc.). */
#define CKC_FAMILY_SERIALIZE_BODY(FN, SPECT, SPECBUILD, ...)                            \
    std::vector<std::string> store;                                                     \
    store.reserve(32);                                                                  \
    SPECT s = SPECBUILD(d, store);                                                      \
    ckc_ir_builder_t b;                                                                 \
    ckc_kernel_def_t* k = __VA_ARGS__;                                                  \
    if(!k || !ckc_ir_builder_ok(&b))                                                    \
    {                                                                                   \
        std::string msg = std::string(FN " build failed: ") + ckc_ir_builder_error(&b); \
        ckc_ir_builder_free(&b);                                                        \
        throw std::runtime_error(msg);                                                  \
    }                                                                                   \
    std::string out = serialize_kernel(k, FN);                                          \
    ckc_ir_builder_free(&b);                                                            \
    return out

/* verify body, mirror of the serialize body above. */
#define CKC_FAMILY_VERIFY_BODY(FN, SPECT, SPECBUILD, ...)                               \
    std::vector<std::string> store;                                                     \
    store.reserve(32);                                                                  \
    SPECT s = SPECBUILD(d, store);                                                      \
    ckc_ir_builder_t b;                                                                 \
    ckc_kernel_def_t* k = __VA_ARGS__;                                                  \
    if(!k || !ckc_ir_builder_ok(&b))                                                    \
    {                                                                                   \
        std::string msg = std::string(FN " build failed: ") + ckc_ir_builder_error(&b); \
        ckc_ir_builder_free(&b);                                                        \
        throw std::runtime_error(msg);                                                  \
    }                                                                                   \
    std::vector<std::string> out = verify_kernel(k);                                    \
    ckc_ir_builder_free(&b);                                                            \
    return out

#endif /* CKC_BINDINGS_FAMILY_GLUE_HPP */
