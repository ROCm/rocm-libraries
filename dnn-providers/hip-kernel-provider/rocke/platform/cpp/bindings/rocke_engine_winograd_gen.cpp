// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_winograd_gen.cpp -- bindings of the generalized
 * Winograd forward transform kernels (conv_winograd_gen; see
 * winograd_glue.hpp). Registered via register_winograd_gen():
 *   conv_winograd_gen_lower_llvm(spec, arch) / _serialize_ir / _verify
 *   conv_winograd_toom_cook_tables() -> {(m, r): {"at", "g", "bt"}}: the
 *     engine's literal Toom-Cook tables (read-only), each matrix a tuple of
 *     rows of exact (numerator, denominator) pairs, for the cross-check
 *     against conv_winograd_gen.toom_cook_gen.
 * The dict mirrors WinogradGenSpec with the problem and both axes nested
 * ({"m", "r", "seg"}) plus "stage": "K1" | "K2" | "K4".
 */
#include "winograd_glue.hpp"

extern "C" {
#include "rocke/instance_conv_winograd_gen.h"
}
#include "rocke/instance_conv_winograd_internal.h"

namespace py = pybind11;

namespace
{

using namespace rocke_wino_bind;

rocke_wino_axis_t axis_from(const py::dict& d, const char* key)
{
    if(!d.contains(key) || !py::isinstance<py::dict>(d[key]))
        throw py::value_error(std::string("WinogradGenSpec requires an axis dict ") + key);
    py::dict a = d[key].cast<py::dict>();
    if(a.contains("pts") && !a["pts"].is_none())
        throw py::value_error("the C++ engine supports the default Winograd points only");
    rocke_wino_axis_t ax;
    const std::string what = std::string("WinogradGenSpec.") + key;
    ax.m = dict_required_int(a, "m", what.c_str());
    ax.r = dict_required_int(a, "r", what.c_str());
    ax.seg = dict_int(a, "seg", 1);
    return ax;
}

rocke_winograd_gen_spec_t wgen_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_winograd_gen_spec_t s = rocke_winograd_gen_spec_default();
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("WinogradGenSpec requires a problem dictionary");
    s.problem = conv_problem_from_dict(d["problem"].cast<py::dict>(), "WinogradGenSpec");
    s.ax_y = axis_from(d, "ax_y");
    s.ax_x = axis_from(d, "ax_x");
    std::string v;
    if(dict_str(d, "io_dtype", v))
        s.io_dtype = keep(store, v);
    if(dict_str(d, "operand_dtype", v))
        s.operand_dtype = keep(store, v);
    if(dict_str(d, "name", v))
        s.name = keep(store, v);
    s.vec = dict_int(d, "vec", s.vec);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.nt_stores = dict_bool(d, "nt_stores", s.nt_stores);
    return s;
}

std::string wgen_stage(const py::dict& d)
{
    std::string st;
    if(!dict_str(d, "stage", st))
        throw py::value_error("conv_winograd_gen needs a 'stage' (K1, K2 or K4)");
    return st;
}

std::string conv_winograd_gen_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_winograd_gen_spec_t s = wgen_build_spec(d, store);
    std::string st = wgen_stage(d);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t rc = rocke_winograd_gen_lower_to_llvm(
        &s, st.c_str(), arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(rc, ll, err, "rocke_engine.conv_winograd_gen_lower_llvm");
}

std::string conv_winograd_gen_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::string st = wgen_stage(d);
    ROCKE_FAMILY_SERIALIZE_BODY(
        "rocke_engine.conv_winograd_gen_serialize_ir",
        rocke_winograd_gen_spec_t,
        wgen_build_spec,
        rocke_build_winograd_gen_stage_new(&b, &s, st.c_str(), arch.c_str()));
}

std::vector<std::string> conv_winograd_gen_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::string st = wgen_stage(d);
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.conv_winograd_gen_verify",
                             rocke_winograd_gen_spec_t,
                             wgen_build_spec,
                             rocke_build_winograd_gen_stage_new(&b, &s, st.c_str(), arch.c_str()));
}

py::tuple tc_matrix(const rocke_wino::Rational* v, int rows, int cols)
{
    py::tuple out(rows);
    for(int i = 0; i < rows; i++)
    {
        py::tuple row(cols);
        for(int j = 0; j < cols; j++)
        {
            const rocke_wino::Rational& q = v[i * cols + j];
            row[j] = py::make_tuple(q.num, q.den);
        }
        out[i] = row;
    }
    return out;
}

py::dict conv_winograd_toom_cook_tables()
{
    size_t count = 0;
    const rocke_wino::ToomCook* t = rocke_wino::toom_cook_tables(&count);
    py::dict out;
    for(size_t k = 0; k < count; k++)
    {
        const rocke_wino::ToomCook& tc = t[k];
        py::dict e;
        e["at"] = tc_matrix(tc.at, tc.m, tc.n);
        e["g"] = tc_matrix(tc.g, tc.n, tc.r);
        e["bt"] = tc_matrix(tc.bt, tc.n, tc.n);
        py::tuple key = py::make_tuple(tc.m, tc.r);
        if(out.contains(key))
            throw std::runtime_error("duplicate Toom-Cook table entry");
        out[key] = e;
    }
    return out;
}

} // namespace

void register_winograd_gen(py::module_& m)
{
    m.def("conv_winograd_toom_cook_tables", &conv_winograd_toom_cook_tables);
    m.def("conv_winograd_gen_lower_llvm",
          &conv_winograd_gen_lower_llvm,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_gen_serialize_ir",
          &conv_winograd_gen_serialize_ir,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_gen_verify", &conv_winograd_gen_verify, py::arg("spec"), py::arg("arch"));
}
