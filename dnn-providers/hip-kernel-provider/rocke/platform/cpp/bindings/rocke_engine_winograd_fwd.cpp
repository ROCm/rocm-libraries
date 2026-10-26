// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_winograd_fwd.cpp -- bindings of the Winograd forward
 * transform family (conv_winograd_fwd; see winograd_glue.hpp). Registered via
 * register_winograd_fwd(), adding:
 *   conv_winograd_fwd_lower_llvm(spec, arch) / _serialize_ir / _verify
 * The spec dict mirrors WinogradFwdSpec (problem nested) plus either
 * `stage` ("K1" | "K2a" | "K2" | "K4") or `merged` (a list of stage names).
 */
#include "winograd_glue.hpp"

extern "C" {
#include "rocke/instance_conv_winograd_fwd.h"
}

namespace py = pybind11;

namespace
{

using namespace rocke_wino_bind;

struct WfwdSel
{
    rocke_winograd_fwd_spec_t spec;
    rocke_winograd_fwd_stage_t stage = ROCKE_WINOGRAD_FWD_K1;
    std::vector<rocke_winograd_fwd_stage_t> merged;
};

rocke_winograd_fwd_stage_t wfwd_stage(const std::string& name)
{
    rocke_winograd_fwd_stage_t st;
    if(!rocke_winograd_fwd_stage_from_name(name.c_str(), &st))
        throw py::value_error("conv_winograd_fwd: unknown stage '" + name + "'");
    return st;
}

WfwdSel wfwd_build(const py::dict& d, std::deque<std::string>& store)
{
    WfwdSel out;
    rocke_winograd_fwd_spec_t& s = out.spec;
    s = rocke_winograd_fwd_spec_default();
    if(!d.contains("problem") || !py::isinstance<py::dict>(d["problem"]))
        throw py::value_error("WinogradFwdSpec requires a problem dictionary");
    s.problem = conv_problem_from_dict(d["problem"].cast<py::dict>(), "WinogradFwdSpec");
    std::string v;
    s.m = dict_int(d, "m", s.m);
    s.m_w = dict_int(d, "m_w", s.m_w);
    if(dict_str(d, "io_dtype", v))
        s.io_dtype = keep(store, v);
    if(dict_str(d, "operand_dtype", v))
        s.operand_dtype = keep(store, v);
    s.vec = dict_int(d, "vec", s.vec);
    s.block_size = dict_int(d, "block_size", s.block_size);
    s.prescale = dict_bool(d, "prescale", s.prescale);
    s.tiles_per_thread = dict_int(d, "tiles_per_thread", s.tiles_per_thread);
    if(dict_str(d, "prescale_mode", v))
        s.prescale_mode = keep(store, v);
    if(d.contains("input_block_tiles") && !d["input_block_tiles"].is_none())
    {
        auto t = d["input_block_tiles"].cast<std::vector<int>>();
        if(t.size() != 2)
            throw py::value_error("input_block_tiles must have two entries");
        s.input_block_tiles[0] = t[0];
        s.input_block_tiles[1] = t[1];
    }
    s.wg_tile_rows = dict_int(d, "wg_tile_rows", s.wg_tile_rows);
    s.nt_stores = dict_bool(d, "nt_stores", s.nt_stores);
    s.lds_lanes = dict_int(d, "lds_lanes", s.lds_lanes);
    s.filter_rows = dict_int(d, "filter_rows", s.filter_rows);
    s.filter_vec = dict_int(d, "filter_vec", s.filter_vec);
    if(dict_str(d, "name", v))
        s.name = keep(store, v);
    if(d.contains("merged") && !d["merged"].is_none())
    {
        for(const auto& nm : d["merged"].cast<std::vector<std::string>>())
            out.merged.push_back(wfwd_stage(nm));
        if(out.merged.empty())
            throw py::value_error("conv_winograd_fwd: empty merged stage list");
    }
    else if(dict_str(d, "stage", v))
        out.stage = wfwd_stage(v);
    else
        throw py::value_error("conv_winograd_fwd: the dict needs `stage` or `merged`");
    return out;
}

rocke_kernel_def_t* wfwd_kernel(rocke_ir_builder_t* b, const WfwdSel& sel, const std::string& arch)
{
    if(!sel.merged.empty())
        return rocke_build_winograd_merged_new(
            b, &sel.spec, sel.merged.data(), (int)sel.merged.size(), arch.c_str());
    return rocke_build_winograd_fwd_stage_new(b, &sel.spec, sel.stage, arch.c_str());
}

std::string conv_winograd_fwd_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    WfwdSel sel = wfwd_build(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = wfwd_kernel(&b, sel, arch);
    return lower_built(&b, k, arch, "rocke_engine.conv_winograd_fwd_lower_llvm");
}

std::string conv_winograd_fwd_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.conv_winograd_fwd_serialize_ir",
                                WfwdSel,
                                wfwd_build,
                                wfwd_kernel(&b, s, arch));
}

std::vector<std::string> conv_winograd_fwd_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY(
        "rocke_engine.conv_winograd_fwd_verify", WfwdSel, wfwd_build, wfwd_kernel(&b, s, arch));
}

} // namespace

void register_winograd_fwd(py::module_& m)
{
    m.def("conv_winograd_fwd_lower_llvm",
          &conv_winograd_fwd_lower_llvm,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_fwd_serialize_ir",
          &conv_winograd_fwd_serialize_ir,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_fwd_verify", &conv_winograd_fwd_verify, py::arg("spec"), py::arg("arch"));
}
