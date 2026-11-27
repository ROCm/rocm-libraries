// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * library/cpp/bindings/rocke_library_engine_img2col.cpp -- img2col family
 * bindings for the rocke_library_engine module. Kept in its own translation
 * unit because the img2col helper header
 * defines its own struct rocke_conv_problem (a peer copy of the conv
 * implicit-GEMM problem with the same tag); compiling it alongside
 * instance_conv_implicit_gemm.h in the same TU is a C++ redefinition. Here the
 * only conv header included is instance_img2col.h, so the tag is unique.
 *
 * register_img2col(m) is called from rocke_library_engine.cpp, adding:
 *   img2col_lower_llvm(spec, arch) / img2col_serialize_ir(spec, arch)
 *   img2col_verify(spec, arch)
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "llvm_flavor.hpp"

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

using rocke_bind::dict_str;
using rocke_bind::serialize_kernel;
using rocke_bind::verify_kernel;

/* Unlike rocke_bind::dict_int, a key set to None reads as the default. */
int i2c_dict_int(const py::dict& d, const char* key, int dflt)
{
    if(d.contains(key) && !d[key].is_none())
        return d[key].cast<int>();
    return dflt;
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
        if(dict_str(d, "dtype", v))
            s.dtype = keep(v);
        if(dict_str(d, "name", v))
            s.name = keep(v);
    }
    s.block_tile_m = i2c_dict_int(d, "block_tile_m", s.block_tile_m);
    s.block_tile_k = i2c_dict_int(d, "block_tile_k", s.block_tile_k);
    s.vec_k = i2c_dict_int(d, "vec_k", s.vec_k);
    return s;
}

std::string lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_img2col_spec_t s = build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_img2col_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    if(st != ROCKE_OK || !ll)
    {
        if(ll)
            free(ll);
        throw_if_arch_domain(st, err);
        throw std::runtime_error("rocke_library_engine.img2col_lower_llvm failed (status="
                                 + std::to_string((int)st)
                                 + "): " + (err[0] ? err : "unknown error"));
    }
    std::string out(ll);
    free(ll);
    return out;
}

std::string serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_library_engine.img2col_serialize_ir",
                                rocke_img2col_spec_t,
                                build_spec,
                                rocke_build_img2col_new(&b, &s, arch.c_str()));
}

std::vector<std::string> verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_library_engine.img2col_verify",
                             rocke_img2col_spec_t,
                             build_spec,
                             rocke_build_img2col_new(&b, &s, arch.c_str()));
}

} // namespace

void register_img2col(py::module_& m)
{
    m.def("img2col_lower_llvm", &lower_llvm, py::arg("spec"), py::arg("arch"));
    m.def("img2col_serialize_ir", &serialize_ir, py::arg("spec"), py::arg("arch"));
    m.def("img2col_verify", &verify, py::arg("spec"), py::arg("arch"));
}
