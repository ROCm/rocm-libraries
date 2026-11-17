// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Python binding for the instance_peer_tile_copy C ABI; no runtime HIP calls.
 */
#include "rocke/instance_peer_tile_copy.h"

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
rocke_peer_tile_copy_spec_t peer_spec(const py::dict& values, std::deque<std::string>& strings)
{
    rocke_peer_tile_copy_spec_t spec = rocke_peer_tile_copy_spec_default();
    const char* integer_names[] = { "tile_rows", "tile_cols", "block_size" };
    int* integer_fields[] = { &spec.tile_rows, &spec.tile_cols, &spec.block_size };
    for(int i = 0; i < 3; ++i)
    {
        if(values.contains(integer_names[i]))
        {
            py::handle value = values[integer_names[i]];
            if(!py::isinstance<py::int_>(value) || py::isinstance<py::bool_>(value))
            {
                throw py::value_error(std::string(integer_names[i]) + " must be an integer");
            }
            *integer_fields[i] = py::cast<int>(value);
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

struct PeerBuilder
{
    rocke_ir_builder_t builder {};
    ~PeerBuilder()
    {
        rocke_ir_builder_free(&builder);
    }
};

rocke_kernel_def_t* peer_kernel(
    PeerBuilder& owner, const rocke_peer_tile_copy_spec_t& spec, const std::string& arch)
{
    rocke_kernel_def_t* kernel
        = rocke_build_peer_tile_copy_new(&owner.builder, &spec, arch.c_str());
    if(kernel == NULL)
    {
        throw std::runtime_error(std::string("peer tile copy build failed: ") + owner.builder.err);
    }
    return kernel;
}

std::string peer_lower(const py::dict& values, const std::string& arch)
{
    std::deque<std::string> strings;
    auto spec = peer_spec(values, strings);
    PeerBuilder owner;
    auto* kernel = peer_kernel(owner, spec, arch);
    char* text = NULL;
    rocke_status_t status
        = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch.c_str(), &text);
    std::unique_ptr<char, decltype(&std::free)> output(text, &std::free);
    if(status != ROCKE_OK || text == NULL)
    {
        throw std::runtime_error("peer tile copy LLVM lowering failed");
    }
    return std::string(text);
}

std::string peer_serialize(const py::dict& values, const std::string& arch)
{
    std::deque<std::string> strings;
    auto spec = peer_spec(values, strings);
    PeerBuilder owner;
    auto* kernel = peer_kernel(owner, spec, arch);
    char* text = NULL;
    rocke_status_t status = rocke_ir_serialize(kernel, &text);
    std::unique_ptr<char, decltype(&std::free)> output(text, &std::free);
    if(status != ROCKE_OK || text == NULL)
    {
        throw std::runtime_error("peer tile copy IR serialization failed");
    }
    return std::string(text);
}

std::vector<std::string> peer_verify(const py::dict& values, const std::string& arch)
{
    std::deque<std::string> strings;
    auto spec = peer_spec(values, strings);
    PeerBuilder owner;
    auto* kernel = peer_kernel(owner, spec, arch);
    rocke_diag_t* diagnostics = NULL;
    size_t count = 0;
    rocke_verify(kernel, &diagnostics, &count);
    std::vector<std::string> output;
    for(size_t i = 0; i < count; ++i)
    {
        char* text = rocke_diag_to_string(&diagnostics[i]);
        std::unique_ptr<char, decltype(&std::free)> message(text, &std::free);
        if(text)
        {
            output.emplace_back(text);
        }
    }
    rocke_diags_free(diagnostics, count);
    return output;
}
} // namespace

void register_peer_tile_copy(py::module_& module)
{
    module.def(
        "peer_tile_copy_lower_llvm", &peer_lower, py::arg("spec"), py::arg("arch") = "gfx950");
    module.def("peer_tile_copy_serialize_ir",
        &peer_serialize,
        py::arg("spec"),
        py::arg("arch") = "gfx950");
    module.def("peer_tile_copy_verify", &peer_verify, py::arg("spec"), py::arg("arch") = "gfx950");
    module.def(
        "peer_tile_copy_kernel_name",
        [](const py::dict& values)
        {
            std::deque<std::string> strings;
            auto spec = peer_spec(values, strings);
            char name[512];
            char reason[256];
            if(!rocke_peer_tile_copy_is_valid_spec(&spec, NULL, reason, sizeof(reason)))
            {
                throw py::value_error(reason);
            }
            if(rocke_peer_tile_copy_kernel_name(&spec, name, sizeof(name)) != ROCKE_OK)
            {
                throw py::value_error("peer tile copy kernel name does not fit");
            }
            return std::string(name);
        },
        py::arg("spec"));
    module.def(
        "peer_tile_copy_grid",
        [](int64_t rows, int64_t cols, const py::dict& values)
        {
            std::deque<std::string> strings;
            auto spec = peer_spec(values, strings);
            int grid[3];
            if(rocke_peer_tile_copy_grid(rows, cols, &spec, grid) != ROCKE_OK)
            {
                throw py::value_error("invalid peer tile copy grid");
            }
            return py::make_tuple(grid[0], grid[1], grid[2]);
        },
        py::arg("rows"),
        py::arg("cols"),
        py::arg("spec"));
    module.def(
        "peer_tile_copy_is_valid",
        [](const py::dict& values, const std::string& arch)
        {
            std::deque<std::string> strings;
            auto spec = peer_spec(values, strings);
            char reason[256];
            bool valid
                = rocke_peer_tile_copy_is_valid_spec(&spec, arch.c_str(), reason, sizeof(reason));
            return py::make_tuple(valid, std::string(reason));
        },
        py::arg("spec"),
        py::arg("arch") = "gfx950");
}
