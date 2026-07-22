// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend
// (include/cudnn_frontend/graph_properties.h and graph_interface.h), used under
// the MIT license.

/**
 * @file unsupported_nodes.h
 * @brief Tier-2 fail-stub node surface for the hipDNN cuDNN-compatibility shim.
 *
 * cuDNN v9 defines 39 `*_attributes` node classes; roughly half have a 1:1
 * hipDNN equivalent (aliased in `cudnn_frontend/graph_properties.h`). The rest
 * have no hipDNN engine yet. So that any hipified v9 source still compiles and
 * fails *loudly* (never silently) on those nodes, this header declares each
 * missing attribute class and the macros the graph wrapper uses to stamp out a
 * matching `Graph::*` node method that records
 * `error_code_t::GRAPH_NOT_SUPPORTED` (RFC 0012 §4.4.2, §6.5).
 *
 * The error is recorded on the composition Graph and surfaces from the next
 * `validate()` / `build_operation_graph()` — node-adding methods return tensors,
 * not `error_t`, so they cannot report it directly (RFC §4.4.2).
 *
 * @note Internal-to-shim; pulled in by `detail/graph_wrapper.h`.
 */

#pragma once

#include <string>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming): these classes mirror cuDNN's
// snake_case public spelling for source compatibility.

namespace detail
{

// Common base for a Tier-2 fail-stub attribute class. It carries only the
// universal fluent setters every cuDNN attribute type shares (`set_name`,
// `set_compute_data_type`) so hipified source that chains them still compiles.
// Node-specific setters are intentionally omitted until a consumer asks for the
// node (RFC §7.3, consumer-driven landing order); adding one is how the node
// graduates from a stub to a real wrapper/alias.
template <typename Derived>
class UnsupportedAttributes
{
public:
    Derived& set_name(const std::string& name)
    {
        _name = name;
        return self();
    }

    const std::string& get_name() const
    {
        return _name;
    }

    Derived& set_compute_data_type(DataType_t type)
    {
        _computeDataType = type;
        return self();
    }

private:
    Derived& self()
    {
        return static_cast<Derived&>(*this);
    }

    std::string _name;
    DataType_t _computeDataType = DataType_t::NOT_SET;
};

} // namespace detail

// Stamp a Tier-2 fail-stub attribute class from an upstream cuDNN v9 class name.
// The class is default-constructible, copyable (node methods take it by value),
// and offers the universal setter surface via detail::UnsupportedAttributes.
#define HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(name)        \
    class name : public detail::UnsupportedAttributes<name> \
    {                                                       \
    }

/// @brief RFC 0012 §4.4.2 / §6.5 — unsupported node attribute.
/// hipDNN has no equivalent engine; the node compiles but reports
/// `error_code_t::GRAPH_NOT_SUPPORTED` at validate()/build().
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(BN_finalize_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Genstats_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(DBN_weight_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Matmul_fp8_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Instancenorm_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Instancenorm_backward_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(AdaLayernorm_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(AdaLayernorm_backward_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Rng_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Reshape_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Transpose_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(RoPE_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(RoPE_backward_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(SDPA_fp8_backward_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Softmax_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(DiagonalBandMask_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Slice_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(PagedCacheLoad_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Concatenate_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Moe_grouped_matmul_attributes);
/// @copydoc BN_finalize_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Moe_grouped_matmul_bwd_attributes);

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph

// Stamp a Tier-2 fail-stub node method body (RFC 0012 §6.5). Records
// GRAPH_NOT_SUPPORTED on the composition Graph — surfaced at the next
// validate()/build_operation_graph() — with a message pointing at the issue
// tracker, then returns a default-constructed result (null shared_ptr or an
// array of null shared_ptrs; both value-initialize empty via `return {};`).
//
// `name`   — the cuDNN v9 method name (stringized into the message)
// `params` — the parameter list, PARENTHESIZED so its commas are one macro arg
// `...`    — the return type (variadic so its commas, e.g. std::array<T, N>,
//            do not split it across arguments)
#define HIPDNN_CUDNN_SHIM_FAIL_NODE(name, params, ...)                                  \
    __VA_ARGS__ name params                                                             \
    {                                                                                   \
        recordError(error_code_t::GRAPH_NOT_SUPPORTED,                                  \
                    "cuDNN-shim node '" #name "' has no hipDNN equivalent yet; file a " \
                    "request at https://github.com/ROCm/rocm-libraries/issues");        \
        return {};                                                                      \
    }
