// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <type_traits>

#include "builder_lifetime.hpp"
#include "family_glue.hpp"
#include "llvm_flavor.hpp"
#include "rocke/instance_gfx950_attention_dense_builder.h"

namespace py = pybind11;

namespace
{
// The Python repr of a nondefault codegen knob participates in kernel identity.
// In particular, 1, True and 1.0 must not be coerced to one another by pybind.
template <typename T>
T scalar(const char* key, py::handle value)
{
    if constexpr(std::is_same_v<T, bool>)
    {
        if(!PyBool_Check(value.ptr()))
            throw py::type_error(std::string(key) + " must be a bool");
    }
    else if constexpr(std::is_same_v<T, int>)
    {
        if(!PyLong_CheckExact(value.ptr()))
            throw py::type_error(std::string(key) + " must be an int");
    }
    else
    {
        static_assert(std::is_same_v<T, double>);
        if(!PyFloat_CheckExact(value.ptr()))
            throw py::type_error(std::string(key) + " must be a float");
    }
    return value.cast<T>();
}

struct DenseSpec
{
    rocke_gfx950_attention_dense_spec_t spec = rocke_gfx950_attention_dense_spec_default();
    // The C API borrows NUL-terminated UTF-8. Pin the source objects rather
    // than copying or interning the strings (including the original dtype).
    std::array<py::object, 5> strings;

    explicit DenseSpec(const py::dict& d)
    {
        size_t recognized = 0;
        auto get = [&](const char* key) {
            auto* v = PyDict_GetItemWithError(d.ptr(), py::str(key).ptr());
            if(!v && PyErr_Occurred())
                throw py::error_already_set();
            recognized += v != nullptr;
            return py::reinterpret_borrow<py::object>(v);
        };
        auto required = [&](const char* key) {
            auto v = get(key);
            if(!v)
                throw py::value_error(std::string("Gfx950AttentionDenseSpec requires ") + key);
            return scalar<int>(key, v);
        };
        auto field = [&]<typename T>(const char* key, T& out, bool* has = nullptr) {
            auto v = get(key);
            if(!v)
                return; // absent: retain the native dataclass default
            if(has)
            {
                *has = !v.is_none();
                if(!*has)
                    return;
            }
            out = scalar<T>(key, v);
        };
        auto text = [&](const char* key, const char*& out, bool nullable = false) -> py::object {
            auto v = get(key);
            if(!v || (nullable && v.is_none()))
                return {}; // optional strings default to nullptr
            if(!PyUnicode_CheckExact(v.ptr()))
                throw py::type_error(std::string(key) + " must be a str");
            Py_ssize_t n;
            out = PyUnicode_AsUTF8AndSize(v.ptr(), &n);
            if(!out)
                throw py::error_already_set();
            if(std::memchr(out, '\0', static_cast<size_t>(n)))
                throw py::value_error(std::string(key) + " contains an embedded NUL");
            return v;
        };

        auto& b = spec.base;
        b.batch = required("batch");
        b.seqlen_q = required("seqlen_q");
        b.seqlen_kv = required("seqlen_kv");
        b.num_query_heads = required("num_query_heads");
        b.num_kv_heads = required("num_kv_heads");
        b.head_size = required("head_size");
        strings = {text("dtype", b.dtype),
                   text("persist_decode", b.persist_decode),
                   text("nonpersist_decode", b.nonpersist_decode),
                   text("kv_storage_dtype", spec.kv_storage_dtype, true),
                   text("pv_loop_order", spec.pv_loop_order, true)};
        field("causal", b.causal);
        field("sliding_window", b.sliding_window);
        field("ragged", b.ragged);
        field("varlen", b.varlen);
        field("block_m", b.block_m);
        field("block_n", b.block_n);
        field("waves_per_eu", b.waves_per_eu);
        field("lds_k_group_pad", b.lds_k_group_pad);
        field("persistent", b.persistent);
        field("num_persistent", b.num_persistent);
        field("interleave", b.interleave);
        field("lazy_rescale", b.lazy_rescale);
        field("paged", b.paged);
        field("block_size", b.block_size);
        field("num_kv_blocks", b.num_kv_blocks);
        field("use_sinks", b.use_sinks);
        field("causal_bottom_right", b.causal_bottom_right);
        field("chiplet_num_xcds", b.chiplet_num_xcds);

        field("lds_v_row_pad", spec.lds_v_row_pad);
        field("wide_lds_dma", spec.wide_lds_dma);
        field("emit_lse", spec.emit_lse);
        field("right_bound", spec.right_bound);
        field("kv_segments", spec.kv_segments);
        field("lds_num_buffers", spec.lds_num_buffers);
        field("lds_k_row_pad", spec.lds_k_row_pad);
        field("lazy_rescale_threshold", spec.lazy_rescale_threshold);
        field("use_exp2_fast", spec.use_exp2_fast);
        field("exp_per_pv_step", spec.exp_per_pv_step, &spec.has_exp_per_pv_step);
        field("partial_vmcnt_prefetch", spec.partial_vmcnt_prefetch);
        field("pv_priority", spec.pv_priority);
        field("pv_sched_fence", spec.pv_sched_fence, &spec.has_pv_sched_fence);
        field("pv_sched_fence_mask", spec.pv_sched_fence_mask);
        field("pv_sched_group_template",
              spec.pv_sched_group_template,
              &spec.has_pv_sched_group_template);
        field("pv_sched_group_ds_read", spec.pv_sched_group_ds_read);
        field("iglp_mode", spec.iglp_mode, &spec.has_iglp_mode);
        field("causal_diag_split", spec.causal_diag_split);
        field("o_store_width", spec.o_store_width);
        if(recognized != d.size())
            throw py::type_error("unexpected field in Gfx950AttentionDenseSpec");
    }
};

// All three operations consume a freshly built native kernel. Own the builder
// through errors, including Python exceptions in the LLVM flavor resolver.
template <typename Consume>
auto with_kernel(const py::dict& d, const std::string& arch, const char* fn, Consume consume)
{
    const char* target = require_arch(arch);
    if(arch.find('\0') != std::string::npos)
        throw py::value_error("arch contains an embedded NUL");
    DenseSpec s(d);
    rocke_ir_builder_t b;
    auto* k = rocke_gfx950_build_attention_dense_new(&b, &s.spec, target);
    auto guard = rocke_bindings::own_builder(&b, rocke_ir_builder_free);
    if(!k || !rocke_ir_builder_ok(&b))
        throw std::runtime_error(std::string(fn) + " build failed: " + rocke_ir_builder_error(&b));
    return consume(k);
}

std::string lower_llvm(const py::dict& d, const std::string& arch)
{
    constexpr auto fn = "rocke_library_engine.gfx950_attention_dense_lower_llvm";
    return with_kernel(d, arch, fn, [&](rocke_kernel_def_t* k) {
        char* ll = nullptr;
        char err[ROCKE_ERR_MSG_CAP] = {};
        auto status = rocke_lower_kernel_to_llvm_ex(
            k, resolve_python_llvm_flavor(), arch.c_str(), &ll, err, sizeof(err));
        return rocke_bind::take_lowered(status, ll, err, fn);
    });
}

std::string serialize_ir(const py::dict& d, const std::string& arch)
{
    constexpr auto fn = "rocke_library_engine.gfx950_attention_dense_serialize_ir";
    return with_kernel(
        d, arch, fn, [&](rocke_kernel_def_t* k) { return rocke_bind::serialize_kernel(k, fn); });
}

std::vector<std::string> verify(const py::dict& d, const std::string& arch)
{
    constexpr auto fn = "rocke_library_engine.gfx950_attention_dense_verify";
    return with_kernel(d, arch, fn, rocke_bind::verify_kernel);
}
} // namespace

void register_gfx950_attention_dense(py::module_& m)
{
    m.def("gfx950_attention_dense_lower_llvm", &lower_llvm, py::arg("spec"), py::arg("arch"));
    m.def("gfx950_attention_dense_serialize_ir", &serialize_ir, py::arg("spec"), py::arg("arch"));
    m.def("gfx950_attention_dense_verify", &verify, py::arg("spec"), py::arg("arch"));
}
