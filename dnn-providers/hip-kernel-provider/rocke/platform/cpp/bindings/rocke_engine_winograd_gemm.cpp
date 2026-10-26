// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * bindings/rocke_engine_winograd_gemm.cpp -- bindings of the Winograd batched
 * GEMM (K3, conv_winograd_gemm_wmma) and its split-K reduce (see
 * winograd_glue.hpp). Registered via register_winograd_gemm(), adding:
 *   conv_winograd_gemm_wmma_{lower_llvm,serialize_ir,verify}(spec, arch)
 *   conv_winograd_gemm_reduce_{lower_llvm,serialize_ir,verify}(spec, arch)
 * The dicts mirror WinogradGemmWmmaSpec / WinogradGemmReduceSpec field by
 * field; `sched` may be None.
 */
#include "winograd_glue.hpp"

extern "C" {
#include "rocke/instance_conv_winograd_gemm_wmma.h"
}

namespace py = pybind11;

namespace
{
using namespace rocke_wino_bind;

rocke_winograd_gemm_wmma_spec_t wgw_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_winograd_gemm_wmma_spec_t s = rocke_winograd_gemm_wmma_spec_default();
    std::string v;
    if(dict_str(d, "dtype", v))
        s.dtype = keep(store, v);
    if(dict_str(d, "name", v))
        s.name = keep(store, v);
    s.waves_m = dict_int(d, "waves_m", s.waves_m);
    s.waves_n = dict_int(d, "waves_n", s.waves_n);
    s.frags_m = dict_int(d, "frags_m", s.frags_m);
    s.frags_n = dict_int(d, "frags_n", s.frags_n);
    s.tile_k = dict_int(d, "tile_k", s.tile_k);
    s.stages = dict_int(d, "stages", s.stages);
    if(dict_str(d, "lds_pad", v))
        s.lds_pad = keep(store, v);
    if(dict_str(d, "epilogue", v))
        s.epilogue = keep(store, v);
    s.split_k = dict_bool(d, "split_k", s.split_k);
    s.prefetch = dict_bool(d, "prefetch", s.prefetch);
    s.xcd_remap = dict_bool(d, "xcd_remap", s.xcd_remap);
    s.num_xcds = dict_int(d, "num_xcds", s.num_xcds);
    if(d.contains("sched"))
        s.sched = d["sched"].is_none() ? nullptr : keep(store, d["sched"].cast<std::string>());
    s.wave_size = dict_int(d, "wave_size", s.wave_size);
    return s;
}

rocke_winograd_gemm_reduce_spec_t wgr_build_spec(const py::dict& d, std::deque<std::string>& store)
{
    rocke_winograd_gemm_reduce_spec_t s = rocke_winograd_gemm_reduce_spec_default();
    std::string v;
    if(dict_str(d, "dtype", v))
        s.dtype = keep(store, v);
    return s;
}

std::string gemm_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_winograd_gemm_wmma_spec_t s = wgw_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_winograd_gemm_wmma_new(&b, &s, arch.c_str());
    return lower_built(&b, k, arch, "rocke_engine.conv_winograd_gemm_wmma_lower_llvm");
}

std::string gemm_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.conv_winograd_gemm_wmma_serialize_ir",
                                rocke_winograd_gemm_wmma_spec_t,
                                wgw_build_spec,
                                rocke_build_winograd_gemm_wmma_new(&b, &s, arch.c_str()));
}

std::vector<std::string> gemm_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.conv_winograd_gemm_wmma_verify",
                             rocke_winograd_gemm_wmma_spec_t,
                             wgw_build_spec,
                             rocke_build_winograd_gemm_wmma_new(&b, &s, arch.c_str()));
}

std::string reduce_lower_llvm(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    std::deque<std::string> store;
    rocke_winograd_gemm_reduce_spec_t s = wgr_build_spec(d, store);
    rocke_ir_builder_t b;
    rocke_kernel_def_t* k = rocke_build_winograd_gemm_reduce_new(&b, &s, arch.c_str());
    return lower_built(&b, k, arch, "rocke_engine.conv_winograd_gemm_reduce_lower_llvm");
}

std::string reduce_serialize_ir(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_SERIALIZE_BODY("rocke_engine.conv_winograd_gemm_reduce_serialize_ir",
                                rocke_winograd_gemm_reduce_spec_t,
                                wgr_build_spec,
                                rocke_build_winograd_gemm_reduce_new(&b, &s, arch.c_str()));
}

std::vector<std::string> reduce_verify(const py::dict& d, const std::string& arch)
{
    require_arch(arch);
    ROCKE_FAMILY_VERIFY_BODY("rocke_engine.conv_winograd_gemm_reduce_verify",
                             rocke_winograd_gemm_reduce_spec_t,
                             wgr_build_spec,
                             rocke_build_winograd_gemm_reduce_new(&b, &s, arch.c_str()));
}

} // namespace

void register_winograd_gemm(py::module_& m)
{
    m.def("conv_winograd_gemm_wmma_lower_llvm", &gemm_lower_llvm, py::arg("spec"), py::arg("arch"));
    m.def("conv_winograd_gemm_wmma_serialize_ir",
          &gemm_serialize_ir,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_gemm_wmma_verify", &gemm_verify, py::arg("spec"), py::arg("arch"));
    m.def("conv_winograd_gemm_reduce_lower_llvm",
          &reduce_lower_llvm,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_gemm_reduce_serialize_ir",
          &reduce_serialize_ir,
          py::arg("spec"),
          py::arg("arch"));
    m.def("conv_winograd_gemm_reduce_verify", &reduce_verify, py::arg("spec"), py::arg("arch"));
}
