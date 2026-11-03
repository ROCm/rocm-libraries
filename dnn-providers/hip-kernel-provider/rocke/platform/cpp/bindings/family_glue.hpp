// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/family_glue.hpp -- macros that emit the canonical per-family
 * serialize-IR and verify function bodies for the rocke_engine module.
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
 * require_arch (defined below).
 */
#ifndef ROCKE_BINDINGS_FAMILY_GLUE_HPP
#define ROCKE_BINDINGS_FAMILY_GLUE_HPP

#include <deque>
#include <stdexcept>
#include <string>

#include "rocke/ir.h" /* rocke_status_t */

/* The gfx target every binding lowers or builds for. Required: there is no
 * default target on either engine. An empty string used to stand for each
 * family's "usual" arch (gfx950, gfx942, gfx1151 or gfx1201), so a caller that
 * forgot `arch=` built for one target and compiled for another. Throws
 * std::invalid_argument, which pybind11 raises as ValueError -- the same type
 * and text the Python lowerer raises for a missing target. */
inline const char* require_arch(const std::string& arch)
{
    if(arch.empty())
    {
        throw std::invalid_argument("lowering requires an explicit gfx target; "
                                    "pass arch=... (there is no default)");
    }
    return arch.c_str();
}

/* A lowering the arch-domain lane refused (ROCKE_ERR_ARCH_DOMAIN): a verdict
 * about the kernel, not an engine failure. The module registers it as
 * rocke_engine.ArchDomainError, a NotImplementedError carrying the engine's
 * message verbatim -- the same bytes the Python lowerer raises -- so the cpp
 * backend can re-raise it rather than fall back to the Python lowerer. */
struct ArchDomainRefusal : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

inline void throw_if_arch_domain(rocke_status_t st, const char* err)
{
    if(st == ROCKE_ERR_ARCH_DOMAIN)
    {
        throw ArchDomainRefusal(err != nullptr ? err : "");
    }
}

/* serialize-IR body for a family whose kernel is built with a local IRBuilder
 * `b`. FN is the fully-qualified entry name used in diagnostics, SPECT is the
 * spec struct type, SPECBUILD is the spec-builder symbol called as
 * SPECBUILD(d, store), and the kernel-build expression is the trailing tokens
 * (which may reference &b, &s, require_arch(arch), etc.). */
#define ROCKE_FAMILY_SERIALIZE_BODY(FN, SPECT, SPECBUILD, ...)                            \
    std::deque<std::string> store;                                                        \
    SPECT s = SPECBUILD(d, store);                                                        \
    rocke_ir_builder_t b;                                                                 \
    rocke_kernel_def_t* k = __VA_ARGS__;                                                  \
    if(!k || !rocke_ir_builder_ok(&b))                                                    \
    {                                                                                     \
        std::string msg = std::string(FN " build failed: ") + rocke_ir_builder_error(&b); \
        rocke_ir_builder_free(&b);                                                        \
        throw std::runtime_error(msg);                                                    \
    }                                                                                     \
    std::string out;                                                                      \
    try                                                                                   \
    {                                                                                     \
        out = serialize_kernel(k, FN);                                                    \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        rocke_ir_builder_free(&b);                                                        \
        throw;                                                                            \
    }                                                                                     \
    rocke_ir_builder_free(&b);                                                            \
    return out

/* verify body, mirror of the serialize body above. */
#define ROCKE_FAMILY_VERIFY_BODY(FN, SPECT, SPECBUILD, ...)                               \
    std::deque<std::string> store;                                                        \
    SPECT s = SPECBUILD(d, store);                                                        \
    rocke_ir_builder_t b;                                                                 \
    rocke_kernel_def_t* k = __VA_ARGS__;                                                  \
    if(!k || !rocke_ir_builder_ok(&b))                                                    \
    {                                                                                     \
        std::string msg = std::string(FN " build failed: ") + rocke_ir_builder_error(&b); \
        rocke_ir_builder_free(&b);                                                        \
        throw std::runtime_error(msg);                                                    \
    }                                                                                     \
    std::vector<std::string> out;                                                         \
    try                                                                                   \
    {                                                                                     \
        out = verify_kernel(k);                                                           \
    }                                                                                     \
    catch(...)                                                                            \
    {                                                                                     \
        rocke_ir_builder_free(&b);                                                        \
        throw;                                                                            \
    }                                                                                     \
    rocke_ir_builder_free(&b);                                                            \
    return out

#endif /* ROCKE_BINDINGS_FAMILY_GLUE_HPP */
