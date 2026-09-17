// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_img2col.cpp -- img2col family bindings for the rocke_engine
 * module. Kept in its own translation unit because the img2col helper header
 * defines its own struct rocke_conv_problem (a peer copy of the conv
 * implicit-GEMM problem with the same tag); compiling it alongside
 * instance_conv_implicit_gemm.h in the same TU is a C++ redefinition. Here the
 * only conv header included is instance_img2col.h, so the tag is unique.
 *
 * register_img2col(m) is called from the main module file, adding:
 *   img2col_lower_llvm(spec, arch) / img2col_serialize_ir(spec, arch)
 *   img2col_verify(spec, arch)
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

#include "family_glue.hpp"

extern "C" {
#include "rocke/instance_img2col.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"
}

namespace py = pybind11;

namespace
{

/* Serialize a built kernel to ck.dsl.ir/v1 text (raises on failure). */
std::string serialize_kernel(rocke_kernel_def_t* kernel, const char* fn)
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
std::vector<std::string> verify_kernel(rocke_kernel_def_t* kernel)
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

int i2c_dict_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<int>();
    return dflt;
}

bool i2c_dict_str(const py::dict& d, const char* key, std::string& out)
{
    if(d.contains(key) && !d[key].is_none())
    {
        out = d[key].cast<std::string>();
        return true;
    }
    return false;
}

const char* i2c_arch(const std::string& arch)
{
    return arch.empty() ? "gfx950" : arch.c_str();
}

/* Fill the img2col conv problem from a "problem" sub-dict. */
void fill_problem(rocke_conv_problem_t* p, const py::dict& d)
{
    p->N = i2c_dict_int(d, "N", p->N);
    p->Hi = i2c_dict_int(d, "Hi", p->Hi);
    p->Wi = i2c_dict_int(d, "Wi", p->Wi);
    p->C = i2c_dict_int(d, "C", p->C);
    p->K = i2c_dict_int(d, "K", p->K);
    p->Y = i2c_dict_int(d, "Y", p->Y);
    p->X = i2c_dict_int(d, "X", p->X);
    p->sH = i2c_dict_int(d, "sH", p->sH);
    p->sW = i2c_dict_int(d, "sW", p->sW);
    p->pH = i2c_dict_int(d, "pH", p->pH);
    p->pW = i2c_dict_int(d, "pW", p->pW);
    p->dH = i2c_dict_int(d, "dH", p->dH);
    p->dW = i2c_dict_int(d, "dW", p->dW);
}

rocke_img2col_spec_t build_spec(const py::dict& d, std::deque<std::string>& store)
{
    auto keep = [&](const std::string& s) -> const char* {
        store.push_back(s);
        return store.back().c_str();
    };
    rocke_img2col_spec_t s = rocke_img2col_spec_default();
    if(d.contains("problem") && py::isinstance<py::dict>(d["problem"]))
        fill_problem(&s.problem, d["problem"].cast<py::dict>());
    {
        std::string v;
        if(i2c_dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(i2c_dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_tile_m = i2c_dict_int(d, "block_tile_m", s.block_tile_m);
    s.block_tile_k = i2c_dict_int(d, "block_tile_k", s.block_tile_k);
    s.vec_k = i2c_dict_int(d, "vec_k", s.vec_k);
    return s;
}

std::string lower_llvm(const py::dict& d, const std::string& arch)
{
    std::deque<std::string> store;
    rocke_img2col_spec_t s = build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_img2col_lower_to_llvm(
        &s, i2c_arch(arch), ROCKE_LLVM_FLAVOR_AUTO, &ll, err, sizeof err);
    if(st != ROCKE_OK || !ll)
    {
        if(ll)
            free(ll);
        throw std::runtime_error("rocke_engine.img2col_lower_llvm failed (status="
                                 + std::to_string((int)st)
                                 + "): " + (err[0] ? err : "unknown error"));
    }
    std::string out(ll);
    free(ll);
    return out;
}

std::string serialize_ir(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.img2col_serialize_ir",
                                rocke_img2col_spec_t,
                                build_spec,
                                rocke_build_img2col_new(&b, &s, i2c_arch(arch)));
}

std::vector<std::string> verify(const py::dict& d, const std::string& arch)
{
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.img2col_verify",
                             rocke_img2col_spec_t,
                             build_spec,
                             rocke_build_img2col_new(&b, &s, i2c_arch(arch)));
}

} // namespace

void register_img2col(py::module_& m)
{
    m.def("img2col_lower_llvm", &lower_llvm, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("img2col_serialize_ir", &serialize_ir, py::arg("spec"), py::arg("arch") = "gfx950");
    m.def("img2col_verify", &verify, py::arg("spec"), py::arg("arch") = "gfx950");
}
