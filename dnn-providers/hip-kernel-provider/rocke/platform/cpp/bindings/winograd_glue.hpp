// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/winograd_glue.hpp -- helpers shared by the Winograd forward family
 * bindings (rocke_engine_winograd_*.cpp): dict readers, the ConvProblem
 * filler and the lower / serialize / verify tails. Each Winograd family binds
 * in its own translation unit and registers through register_winograd_*().
 */
#ifndef ROCKE_BINDINGS_WINOGRAD_GLUE_HPP
#define ROCKE_BINDINGS_WINOGRAD_GLUE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "family_glue.hpp"
#include "llvm_flavor.hpp"

extern "C" {
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h" /* rocke_conv_problem_t */
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace rocke_wino_bind
{
namespace py = pybind11;

/* The dict readers default only an absent key: a key present with None is
 * malformed input (the Python specs take no None for these fields) and is
 * refused rather than read as the default. */
inline void reject_none(const py::dict& d, const char* key)
{
    if(d[key].is_none())
        throw py::value_error(std::string(key) + " must not be None");
}

inline int dict_int(const py::dict& d, const char* key, int dflt)
{
    if(!d.contains(key))
        return dflt;
    reject_none(d, key);
    return d[key].cast<int>();
}

/* A field the Python spec requires: absent or None is refused. */
inline int dict_required_int(const py::dict& d, const char* key, const char* what)
{
    if(!d.contains(key))
        throw py::value_error(std::string(what) + " requires " + key);
    reject_none(d, key);
    return d[key].cast<int>();
}

inline bool dict_bool(const py::dict& d, const char* key, bool dflt)
{
    if(!d.contains(key))
        return dflt;
    reject_none(d, key);
    return d[key].cast<bool>();
}

inline bool dict_str(const py::dict& d, const char* key, std::string& out)
{
    if(!d.contains(key))
        return false;
    reject_none(d, key);
    out = d[key].cast<std::string>();
    return true;
}

/* A string field kept alive for the lifetime of the spec. */
inline const char* keep(std::deque<std::string>& store, const std::string& s)
{
    store.push_back(s);
    return store.back().c_str();
}

/* ConvProblem from its dict (2-D; every shape field required). */
inline rocke_conv_problem_t conv_problem_from_dict(const py::dict& pd, const char* what)
{
    const char* required[] = {"N", "Hi", "Wi", "C", "K", "Y", "X"};
    for(const char* key : required)
    {
        if(!pd.contains(key))
            throw py::value_error(std::string(what) + ".problem requires " + key);
    }
    if(pd.contains("Di") && !pd["Di"].is_none())
        throw py::value_error(std::string(what) + ": 2-D convolutions only");
    rocke_conv_problem_t p = rocke_conv_problem_make(dict_int(pd, "N", 0),
                                                     dict_int(pd, "Hi", 0),
                                                     dict_int(pd, "Wi", 0),
                                                     dict_int(pd, "C", 0),
                                                     dict_int(pd, "K", 0),
                                                     dict_int(pd, "Y", 0),
                                                     dict_int(pd, "X", 0),
                                                     dict_int(pd, "sH", 1),
                                                     dict_int(pd, "sW", 1),
                                                     dict_int(pd, "pH", 0),
                                                     dict_int(pd, "pW", 0),
                                                     dict_int(pd, "dH", 1),
                                                     dict_int(pd, "dW", 1));
    p.groups = dict_int(pd, "groups", 1);
    return p;
}

inline std::string take_lowered(rocke_status_t st, char* llvm_text, const char* err, const char* fn)
{
    if(st != ROCKE_OK || !llvm_text)
    {
        if(llvm_text)
            free(llvm_text);
        std::string msg = std::string(fn) + " failed (status=" + std::to_string((int)st)
                          + "): " + (err && err[0] ? err : "unknown error");
        throw std::runtime_error(msg);
    }
    std::string out(llvm_text);
    free(llvm_text);
    return out;
}

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

/* Lower a built kernel (builder `b` owns it) to LLVM IR text; frees `b`. */
inline std::string lower_built(rocke_ir_builder_t* b,
                               rocke_kernel_def_t* k,
                               const std::string& arch,
                               const char* fn)
{
    if(!k || !rocke_ir_builder_ok(b))
    {
        std::string msg = std::string(fn) + " build failed: " + rocke_ir_builder_error(b);
        rocke_ir_builder_free(b);
        throw std::runtime_error(msg);
    }
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_lower_kernel_to_llvm_ex(
        k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof err);
    rocke_ir_builder_free(b);
    return take_lowered(st, ll, err, fn);
}

} // namespace rocke_wino_bind

#endif /* ROCKE_BINDINGS_WINOGRAD_GLUE_HPP */
