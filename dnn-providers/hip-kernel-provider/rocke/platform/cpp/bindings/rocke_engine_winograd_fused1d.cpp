// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_winograd_fused1d.cpp -- bindings of the fused 1-D
 * Winograd forward family (see winograd_glue.hpp). Registered via
 * register_winograd_fused1d(), adding:
 *   conv_winograd_fused_1d_wmma_lower_llvm(spec, arch)
 *   conv_winograd_fused_1d_wmma_serialize_ir(spec, arch)
 *   conv_winograd_fused_1d_wmma_verify(spec, arch)
 *
 * The dict mirrors FusedWino1DSpec with the problem nested and the W axis as
 * ax = {m, r, seg} (default interpolation points only).
 */
#include "winograd_glue.hpp"

extern "C" {
#include "rocke/instance_conv_winograd_fused_1d_wmma.h"
}

namespace py = pybind11;

namespace
{

using namespace rocke_wino_bind;

rocke_winograd_fused_1d_spec_t fu_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_winograd_fused_1d_spec_t s = rocke_winograd_fused_1d_spec_default();
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("FusedWino1DSpec requires a problem dictionary");
    s.problem = conv_problem_from_dict(d["problem"].cast<py::dict>(), "FusedWino1DSpec");
    if(!d.contains("ax") || !py::isinstance<py::dict>(d["ax"]))
        throw py::value_error("FusedWino1DSpec requires an ax dictionary");
    {
        py::dict ax = d["ax"].cast<py::dict>();
        if(ax.contains("pts") && !ax["pts"].is_none())
            throw py::value_error("FusedWino1DSpec: the C++ engine supports the default "
                                  "interpolation points only (ax.pts must be None)");
        s.ax_m = dict_int(ax, "m", s.ax_m);
        s.ax_r = dict_int(ax, "r", s.ax_r);
        s.ax_seg = dict_int(ax, "seg", s.ax_seg);
    }
    s.q = dict_int(d, "q", s.q);
    s.fk = dict_int(d, "fk", s.fk);
    s.grid_blocks = dict_int(d, "grid_blocks", s.grid_blocks);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.m_f16 = dict_bool(d, "m_f16", s.m_f16);
    {
        std::string v;
        if(dict_str(d, "xf", v))
            s.xf = keep(store, v);
        if(dict_str(d, "name", v))
            s.name = keep(store, v);
    }
    s.xf_batch = dict_int(d, "xf_batch", s.xf_batch);
    s.merge = dict_bool(d, "merge", s.merge);
    s.waves = dict_int(d, "waves", s.waves);
    s.waves_per_eu = dict_int(d, "waves_per_eu", s.waves_per_eu);
    return s;
}

std::string lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_winograd_fused_1d_spec_t s = fu_build_spec(d, store);
    char* ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = rocke_winograd_fused_1d_lower_to_llvm(
        &s, arch.c_str(), resolve_python_llvm_flavor(), &ll, err, sizeof err);
    return take_lowered(st, ll, err, "rocke_engine.conv_winograd_fused_1d_wmma_lower_llvm");
}

std::string serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.conv_winograd_fused_1d_wmma_serialize_ir",
                                rocke_winograd_fused_1d_spec_t,
                                fu_build_spec,
                                rocke_build_winograd_fused_1d_new(&b, &s, arch.c_str()));
}

std::vector<std::string> verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.conv_winograd_fused_1d_wmma_verify",
                             rocke_winograd_fused_1d_spec_t,
                             fu_build_spec,
                             rocke_build_winograd_fused_1d_new(&b, &s, arch.c_str()));
}

} // namespace

void register_winograd_fused1d(py::module_& m)
{
    m.def("conv_winograd_fused_1d_wmma_lower_llvm", &lower_llvm, py::arg("spec"), py::arg("arch"));
    m.def("conv_winograd_fused_1d_wmma_serialize_ir",
          &serialize_ir,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_fused_1d_wmma_verify", &verify, py::arg("spec"), py::arg("arch"));
}
