// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/family_glue.hpp -- macros that emit the canonical per-family
 * serialize-IR and verify function bodies, and the dict / lowering helpers,
 * shared by the rocke_engine and rocke_library_engine modules.
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
 * and that serialize_kernel / verify_kernel / require_arch are visible
 * unqualified. require_arch is global (below); serialize_kernel and
 * verify_kernel live in namespace rocke_bind with the other helpers every
 * binding translation unit shares (dict_int / dict_bool / dict_str /
 * take_lowered), and a file that uses the macros brings them into scope with
 * using-declarations. They are namespaced rather than global so a file that
 * defines its own same-named readers (winograd_glue.hpp's rocke_wino_bind,
 * pulled in with a using-directive) does not see two candidates.
 */
#ifndef ROCKE_BINDINGS_FAMILY_GLUE_HPP
#define ROCKE_BINDINGS_FAMILY_GLUE_HPP

#include <pybind11/pybind11.h>

#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "rocke/ir.h" /* rocke_status_t, ROCKE_ERR_MSG_CAP */
#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

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

/* Helpers shared by every binding translation unit of both modules. Each file
 * names the ones it uses with using-declarations; a file whose readers behave
 * differently (img2col's i2c_dict_int, winograd_glue.hpp's None-rejecting
 * readers) keeps its own. */
namespace rocke_bind
{

/* Pull an int out of the dict (with default). */
inline int dict_int(const pybind11::dict& d, const char* key, int dflt)
{
    if(d.contains(key))
        return d[key].cast<int>();
    return dflt;
}

inline bool dict_bool(const pybind11::dict& d, const char* key, bool dflt)
{
    if(d.contains(key))
        return d[key].cast<bool>();
    return dflt;
}

/* Optional string: returns true + sets `out` if present (non-None). */
inline bool dict_str(const pybind11::dict& d, const char* key, std::string& out)
{
    if(d.contains(key) && !d[key].is_none())
    {
        out = d[key].cast<std::string>();
        return true;
    }
    return false;
}

/* Wrap a `*_lower_to_llvm` result: raise on failure, else return the .ll.
 *
 * `err` is the buffer the lowering filled, taken by reference rather than as a
 * pointer so that no call site can pass nullptr. It used to be able to, and an
 * arch-domain refusal lowered through the err-less rocke_lower_kernel_to_llvm
 * then reached Python as an ArchDomainError with an empty message -- the
 * intrinsic, target and flavor that explain the refusal gone, and nothing for
 * backend.py to rebuild the Python error from. Lower through the _ex entry. */
inline std::string take_lowered(rocke_status_t st,
                                char* llvm_text,
                                const char (&err)[ROCKE_ERR_MSG_CAP],
                                const char* fn)
{
    if(st != ROCKE_OK || !llvm_text)
    {
        if(llvm_text)
            free(llvm_text);
        throw_if_arch_domain(st, err);
        std::string msg = std::string(fn) + " failed (status=" + std::to_string((int)st)
                          + "): " + (err[0] ? err : "unknown error");
        throw std::runtime_error(msg);
    }
    std::string out(llvm_text);
    free(llvm_text);
    return out;
}

/* Serialize a built kernel to ck.dsl.ir/v1 text (raises on failure). */
inline std::string serialize_kernel(rocke_kernel_def_t* kernel, const char* fn)
{
    char* text = nullptr;
    rocke_status_t st = rocke_ir_serialize(kernel, &text);
    if(st != ROCKE_OK || !text)
    {
        if(text)
            free(text);
        throw std::runtime_error(std::string(fn)
                                 + " serialize failed (status=" + std::to_string((int)st) + ")");
    }
    std::string out(text);
    free(text);
    return out;
}

/* Run the verifier on a built kernel; return the diagnostic strings. */
inline std::vector<std::string> verify_kernel(rocke_kernel_def_t* kernel)
{
    rocke_diag_t* diags = nullptr;
    size_t n = 0;
    rocke_verify(kernel, &diags, &n);
    std::vector<std::string> out;
    out.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        char* s = rocke_diag_to_string(&diags[i]);
        if(s)
        {
            out.emplace_back(s);
            free(s);
        }
    }
    rocke_diags_free(diags, n);
    return out;
}

} // namespace rocke_bind

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
