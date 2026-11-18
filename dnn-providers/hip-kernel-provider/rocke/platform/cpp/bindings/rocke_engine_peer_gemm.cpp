// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Binding for the peer_gemm C ABI; only build/lower/verify, no runtime HIP calls. */
#include "rocke/instance_peer_gemm.h"

#include <cstdlib>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rocke/ir_serialize.h"
#include "rocke/verify.h"

namespace py = pybind11;

namespace
{
rocke_peer_gemm_spec_t peer_spec(const py::dict& values, std::deque<std::string>& strings)
{
    auto spec = rocke_peer_gemm_spec_default();
    const char* names[] = { "rows_per_source", "cols_per_rank", "k", "sources" };
    int* fields[] = { &spec.rows_per_source, &spec.cols_per_rank, &spec.k, &spec.sources };
    for(int i = 0; i < 4; ++i)
    {
        if(values.contains(names[i]))
        {
            py::handle value = values[names[i]];
            if(!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
            {
                throw py::value_error(std::string(names[i]) + " must be an integer");
            }
            *fields[i] = py::cast<int>(value);
        }
    }
    const char* string_names[] = { "dtype", "name" };
    const char** string_fields[] = { &spec.dtype, &spec.name };
    for(int i = 0; i < 2; ++i)
    {
        if(values.contains(string_names[i]))
        {
            py::handle value = values[string_names[i]];
            if(value.is_none())
            {
                *string_fields[i] = NULL;
            }
            else
            {
                strings.push_back(py::cast<std::string>(value));
                *string_fields[i] = strings.back().c_str();
            }
        }
    }
    return spec;
}

struct BuilderOwner
{
    rocke_ir_builder_t builder {};
    ~BuilderOwner()
    {
        rocke_ir_builder_free(&builder);
    }
};

rocke_kernel_def_t* build(
    BuilderOwner& owner, const rocke_peer_gemm_spec_t& spec, const std::string& arch)
{
    auto* kernel = rocke_build_peer_gemm_new(&owner.builder, &spec, arch.c_str());
    if(kernel == NULL)
    {
        throw std::runtime_error(std::string("peer GEMM build failed: ") + owner.builder.err);
    }
    return kernel;
}

std::string emit(const py::dict& values, const std::string& arch, bool llvm)
{
    std::deque<std::string> strings;
    auto spec = peer_spec(values, strings);
    BuilderOwner owner;
    auto* kernel = build(owner, spec, arch);
    char* text = NULL;
    auto status = llvm
        ? rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch.c_str(), &text)
        : rocke_ir_serialize(kernel, &text);
    std::unique_ptr<char, decltype(&std::free)> memory(text, &std::free);
    if(status != ROCKE_OK || text == NULL)
    {
        throw std::runtime_error("peer GEMM emission failed");
    }
    return std::string(text);
}

struct DiagnosticsOwner
{
    rocke_diag_t* diagnostics = NULL;
    size_t count = 0;
    ~DiagnosticsOwner()
    {
        rocke_diags_free(diagnostics, count);
    }
};

std::vector<std::string> verify(const py::dict& values, const std::string& arch)
{
    std::deque<std::string> strings;
    auto spec = peer_spec(values, strings);
    BuilderOwner owner;
    auto* kernel = build(owner, spec, arch);
    DiagnosticsOwner diagnostics;
    rocke_verify(kernel, &diagnostics.diagnostics, &diagnostics.count);
    std::vector<std::string> result;
    for(size_t i = 0; i < diagnostics.count; ++i)
    {
        char* text = rocke_diag_to_string(&diagnostics.diagnostics[i]);
        std::unique_ptr<char, decltype(&std::free)> memory(text, &std::free);
        if(text)
        {
            result.emplace_back(text);
        }
    }
    return result;
}
} // namespace

void register_peer_gemm(py::module_& module)
{
    module.def(
        "peer_gemm_lower_llvm",
        [](const py::dict& spec, const std::string& arch) { return emit(spec, arch, true); },
        py::arg("spec"),
        py::arg("arch") = "gfx950");
    module.def(
        "peer_gemm_serialize_ir",
        [](const py::dict& spec, const std::string& arch) { return emit(spec, arch, false); },
        py::arg("spec"),
        py::arg("arch") = "gfx950");
    module.def("peer_gemm_verify", &verify, py::arg("spec"), py::arg("arch") = "gfx950");
    module.def(
        "peer_gemm_is_valid",
        [](const py::dict& values, const std::string& arch)
        {
            std::deque<std::string> strings;
            auto spec = peer_spec(values, strings);
            char reason[ROCKE_ERR_MSG_CAP];
            bool valid = rocke_peer_gemm_is_valid_spec(&spec, arch.c_str(), reason, sizeof(reason));
            return py::make_tuple(valid, std::string(reason));
        },
        py::arg("spec"),
        py::arg("arch") = "gfx950");
    module.def(
        "peer_gemm_kernel_name",
        [](const py::dict& values)
        {
            std::deque<std::string> strings;
            auto spec = peer_spec(values, strings);
            char name[256];
            if(rocke_peer_gemm_kernel_name(&spec, name, sizeof(name)) != ROCKE_OK)
            {
                throw py::value_error("invalid peer GEMM kernel name/spec");
            }
            return std::string(name);
        },
        py::arg("spec"));
    module.def(
        "peer_gemm_grid",
        [](const py::dict& values)
        {
            std::deque<std::string> strings;
            auto spec = peer_spec(values, strings);
            int grid[3];
            if(rocke_peer_gemm_grid(&spec, grid) != ROCKE_OK)
            {
                throw py::value_error("invalid peer GEMM grid/spec");
            }
            return py::make_tuple(grid[0], grid[1], grid[2]);
        },
        py::arg("spec"));
}
