// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// torch.ops.hipdnn_vllm.* registrations and the hipdnn_vllm._C Python module.
//
// Ops are registered for the CUDA dispatch key (ROCm torch masquerades HIP as
// CUDA). The dispatcher runs them without the GIL. Fake (meta) impls live in
// hipdnn_vllm/fake_impls.py.

#include "graphs/graphs.h"
#include "plan_cache.h"
#include "runtime.h"
#include "stats.h"

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <hipdnn_frontend/version.h>
#include <torch/library.h>

#include <string>
#include <vector>

namespace py = pybind11;

namespace hipdnn_vllm
{
namespace
{

Runtime& runtimeFor(int64_t device)
{
    const c10::DeviceGuard guard(c10::Device(c10::DeviceType::CUDA, static_cast<c10::DeviceIndex>(device)));
    return Runtime::forDevice(static_cast<int>(device));
}

py::dict statsSnapshot()
{
    py::list counters;
    for(const auto& row : Stats::instance().rows())
    {
        py::dict d;
        d["op"] = row.op;
        d["route"] = row.route;
        d["reason"] = row.reason;
        d["phase"] = row.phase;
        d["count"] = row.count;
        counters.append(d);
    }
    const RuntimeTotals t = Stats::instance().totals();
    py::dict runtime;
    runtime["plan_builds"] = t.planBuilds;
    runtime["plan_build_declines"] = t.planBuildDeclines;
    runtime["plan_build_ns"] = t.planBuildNs;
    runtime["plan_cache_hits"] = t.planCacheHits;
    runtime["plan_cache_evictions"] = t.planCacheEvictions;
    runtime["plan_cache_size"] = planCache().size();
    runtime["plan_cache_capacity"] = planCache().capacity();
    runtime["plan_cache_pinned"] = planCache().pinnedCount();
    runtime["execute_calls"] = t.executeCalls;
    runtime["execute_host_ns"] = t.executeHostNs;
    py::dict out;
    out["counters"] = counters;
    out["runtime"] = runtime;
    return out;
}

} // namespace
} // namespace hipdnn_vllm

TORCH_LIBRARY(hipdnn_vllm, m)
{
    m.def("rms_norm(Tensor x, Tensor weight, float eps) -> Tensor");
    m.def("rms_norm_rows(Tensor x, Tensor weight, float eps, bool strict) -> Tensor");
    m.def("fused_add_rms_norm_rows(Tensor x, Tensor residual, Tensor weight, float eps, bool strict) "
          "-> (Tensor, Tensor)");
    m.def("linear(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor");
    m.def("sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor(a!) out, float scale, bool causal=False, "
          "bool bottom_right=True, int left_bound=-1, int right_bound=-1, "
          "Tensor? ragged_offset_q=None, Tensor? seq_len_q=None, int max_seq_len_q=0, "
          "Tensor? ragged_offset_kv=None, Tensor? page_table=None, Tensor? seq_len_kv=None, "
          "int max_seq_len_kv=0, Tensor? sink=None) -> ()");
    m.def("encoder_attention(Tensor q, Tensor k, Tensor v, Tensor? cu_seqlens, float scale) -> Tensor");
    m.def("layer_norm(Tensor x, Tensor? weight, Tensor? bias, float eps, bool strict) -> Tensor");
    m.def("conv_fwd(Tensor x, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, "
          "int groups, bool strict) -> Tensor");
}

TORCH_LIBRARY_IMPL(hipdnn_vllm, CUDA, m)
{
    m.impl("rms_norm", &hipdnn_vllm::rmsNorm);
    m.impl("rms_norm_rows", &hipdnn_vllm::rmsNormRows);
    m.impl("fused_add_rms_norm_rows", &hipdnn_vllm::fusedAddRmsNormRows);
    m.impl("linear", &hipdnn_vllm::linear);
    m.impl("sdpa_fwd", &hipdnn_vllm::sdpaFwd);
    m.impl("encoder_attention", &hipdnn_vllm::encoderAttention);
    m.impl("layer_norm", &hipdnn_vllm::layerNorm);
    m.impl("conv_fwd", &hipdnn_vllm::convFwd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace hipdnn_vllm;
    m.doc() = "hipdnn_vllm runtime (hipDNN C++ frontend over torch tensors)";

    m.def("build_backend_version", &buildBackendVersion);
    m.def("loaded_backend_version", &loadedBackendVersion);
    m.def("frontend_version", [] { return std::string(HIPDNN_FRONTEND_VERSION_STRING); });
    m.def("check_backend_version", &checkBackendVersion);
    m.def("set_backend_directory", &setBackendDirectory, py::arg("directory"));
    m.def("backend_library_path", &backendLibraryPath);
    m.def("backend_symbol_origin", &backendSymbolOrigin, py::arg("symbol"));
    m.def("decline_error_prefix", [](const std::string& op) { return declineErrorPrefix(op.c_str()); });

    m.def(
        "configure_plugins",
        [](const std::string& mode, const std::vector<std::string>& paths) {
            if(mode != "additive" && mode != "absolute")
            {
                throw std::invalid_argument("plugin mode must be 'additive' or 'absolute'");
            }
            configurePlugins(mode == "absolute" ? PluginMode::ABSOLUTE : PluginMode::ADDITIVE, paths);
        },
        py::arg("mode"),
        py::arg("paths"));
    m.def("plugins_configured", &pluginsConfigured);
    m.def("runtime_created", &Runtime::anyCreated);

    m.def(
        "engines",
        [](int64_t device) {
            std::vector<EngineInfo> engines;
            {
                const py::gil_scoped_release release;
                engines = runtimeFor(device).engines();
            }
            py::list out;
            for(const auto& e : engines)
            {
                py::dict d;
                d["id"] = e.id;
                d["name"] = e.name;
                d["plugin"] = e.plugin;
                d["version"] = e.version;
                d["type"] = e.type;
                out.append(d);
            }
            return out;
        },
        py::arg("device"));
    m.def(
        "loaded_plugin_paths",
        [](int64_t device) { return runtimeFor(device).loadedPluginPaths(); },
        py::arg("device"),
        py::call_guard<py::gil_scoped_release>());

    m.def(
        "set_plan_cache_size",
        [](int64_t n) { planCache().setCapacity(static_cast<size_t>(n < 1 ? 1 : n)); },
        py::arg("n"));
    m.def("plan_cache_clear", [] { planCache().clear(); });

    m.def("stats_snapshot", &statsSnapshot);
    m.def("stats_reset", [] { Stats::instance().reset(); });

    m.def("rms_norm_probe",
          &rmsNormProbe,
          py::arg("x"),
          py::arg("weight"),
          py::arg("eps"),
          py::call_guard<py::gil_scoped_release>());

    m.def("norm_set_exact_rows", &normSetExactRows, py::arg("rows"));
    m.def("norm_exact_rows", &normExactRows);
    m.def("norm_rows_reset", &normRowsReset);
    m.def("norm_prebuild_totals", [] {
        const NormPrebuildTotals t = normPrebuildTotals();
        py::dict d;
        d["signatures"] = t.signatures;
        d["plans"] = t.plans;
        d["declines"] = t.declines;
        d["ns"] = t.ns;
        return d;
    });
    m.def("fused_add_route",
          &fusedAddRoute,
          py::arg("x"),
          py::arg("residual"),
          py::arg("weight"),
          py::arg("eps"));
    m.def("linear_try",
          &linearTry,
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::call_guard<py::gil_scoped_release>());
    m.def("linear_probe",
          &linearProbe,
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::call_guard<py::gil_scoped_release>());
    // sdpa_fwd's arguments, shared by its probe and layout helpers.
    const auto defSdpa = [&m](const char* name, auto fn, auto... extra) {
        m.def(name,
              fn,
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("out"),
              py::arg("scale"),
              py::arg("causal") = false,
              py::arg("bottom_right") = true,
              py::arg("left_bound") = -1,
              py::arg("right_bound") = -1,
              py::arg("ragged_offset_q") = py::none(),
              py::arg("seq_len_q") = py::none(),
              py::arg("max_seq_len_q") = 0,
              py::arg("ragged_offset_kv") = py::none(),
              py::arg("page_table") = py::none(),
              py::arg("seq_len_kv") = py::none(),
              py::arg("max_seq_len_kv") = 0,
              py::arg("sink") = py::none(),
              extra...);
    };
    defSdpa("sdpa_fwd_probe", &sdpaFwdProbe, py::call_guard<py::gil_scoped_release>());
    defSdpa("sdpa_fwd_layout", &sdpaFwdLayout);
    m.def("layer_norm_probe",
          &layerNormProbe,
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps"),
          py::call_guard<py::gil_scoped_release>());
    m.def(
        "conv_fwd_probe",
        [](const at::Tensor& x,
           const at::Tensor& weight,
           const std::optional<at::Tensor>& bias,
           const std::vector<int64_t>& stride,
           const std::vector<int64_t>& padding,
           const std::vector<int64_t>& dilation,
           int64_t groups) { return convFwdProbe(x, weight, bias, stride, padding, dilation, groups); },
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("stride"),
        py::arg("padding"),
        py::arg("dilation"),
        py::arg("groups"),
        py::call_guard<py::gil_scoped_release>());
}
