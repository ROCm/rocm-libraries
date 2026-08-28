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
 * `error_code_t::GRAPH_NOT_SUPPORTED`.
 *
 * The error is recorded on the composition Graph and surfaces from the next
 * `validate()` / `build_operation_graph()` — node-adding methods return tensors,
 * not `error_t`, so they cannot report it directly.
 *
 * @note Internal-to-shim; pulled in by `detail/graph_wrapper.h`.
 */

#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_helpers.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/graph_properties.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend/sdpa_attributes.h>
#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::graph
{
// NOLINTBEGIN(readability-identifier-naming): these classes mirror cuDNN's
// snake_case public spelling for source compatibility.

namespace detail
{

// Common base for a Tier-2 fail-stub attribute class. It carries only the
// universal accessors every cuDNN attribute type shares (`set_name`/`get_name`,
// `set_compute_data_type`/`get_compute_data_type`) so hipified source that
// chains or reads them still compiles, matching the hipDNN attribute types the
// Tier-1 aliases resolve to.
// Node-specific setters are intentionally omitted until a consumer asks for the
// node (consumer-driven landing order); adding one is how the node
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

    DataType_t get_compute_data_type() const
    {
        return _computeDataType;
    }

private:
    friend Derived;

    UnsupportedAttributes() = default;
    Derived& self()
    {
        return static_cast<Derived&>(*this);
    }

    std::string _name;
    DataType_t _computeDataType = DataType_t::NOT_SET;
};

// A Tier-2 fail-stub records GRAPH_NOT_SUPPORTED but must still hand back a live,
// graph-registered tensor: idiomatic cuDNN FE chains the result
// (`node(...)->set_output(true).set_uid(n)`), so a null return dereferences null
// before the error can surface at validate(). These helpers mint placeholder
// tensor(s) through the graph's public tensor() so they are tracked like any
// other tensor; they are never validated because the recorded error
// short-circuits validate()/build_operation_graph() first. Templated on the graph
// type to avoid a dependency cycle with the wrapper that includes this header.
template <typename Ptr>
struct is_shared_ptr : std::false_type
{
};

template <typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
{
};

// Build a fail-stub's return value: a single placeholder tensor, or an array of
// them for multi-output nodes. Result is the node method's declared return type
// (a std::shared_ptr<Tensor_attributes> or a std::array<..., N> of them).
template <typename Result, typename GraphT>
Result makeUnsupportedNodeResult(GraphT& graph)
{
    if constexpr(is_shared_ptr<Result>::value)
    {
        return graph.tensor(typename Result::element_type{});
    }
    else
    {
        Result result{};
        for(auto& element : result)
        {
            element = graph.tensor(typename Result::value_type::element_type{});
        }
        return result;
    }
}

} // namespace detail

// Stamp a Tier-2 fail-stub attribute class from an upstream cuDNN v9 class name.
// The user-provided constructor keeps C++17 brace-init from aggregate-initializing
// the private CRTP base constructor directly.
// NOLINTBEGIN(bugprone-macro-parentheses): name is a type token, not an expression.
#define HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(name)         \
    class name : public detail::UnsupportedAttributes<name>  \
    {                                                        \
    public:                                                  \
        name() {} /* NOLINT(modernize-use-equals-default) */ \
    }
// NOLINTEND(bugprone-macro-parentheses)

/// @brief Unsupported node attribute.
/// hipDNN has no equivalent engine; the node compiles but reports
/// `error_code_t::GRAPH_NOT_SUPPORTED` at validate()/build().
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Genstats_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(DBN_weight_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Matmul_fp8_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Instancenorm_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Instancenorm_backward_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Rng_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(RoPE_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(RoPE_backward_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(Softmax_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(DiagonalBandMask_attributes);
/// @copydoc Genstats_attributes
HIPDNN_CUDNN_SHIM_FAIL_STUB_ATTRIBUTES(PagedCacheLoad_attributes);

/// @brief Unsupported node attribute that still stores its node-specific
/// configuration, so consumer source which sets it compiles and can read it
/// back. The node reports `error_code_t::GRAPH_NOT_SUPPORTED` at
/// validate()/build() regardless — storing a value is not a claim to honor it.
class Reshape_attributes : public detail::UnsupportedAttributes<Reshape_attributes>
{
public:
    Reshape_attributes() {} // NOLINT(modernize-use-equals-default)

    Reshape_attributes& set_reshape_mode(ReshapeMode_t mode)
    {
        _mode = mode;
        return *this;
    }

    ReshapeMode_t get_reshape_mode() const
    {
        return _mode;
    }

private:
    // Upstream defaults the mode to VIEW_ONLY, not to the enum's NOT_SET.
    ReshapeMode_t _mode = ReshapeMode_t::VIEW_ONLY;
};

/// @copydoc Reshape_attributes
class Transpose_attributes : public detail::UnsupportedAttributes<Transpose_attributes>
{
public:
    Transpose_attributes() {} // NOLINT(modernize-use-equals-default)

    Transpose_attributes& set_permutation(const std::vector<int64_t>& permutation)
    {
        _permutation = permutation;
        return *this;
    }

    const std::vector<int64_t>& get_permutation() const
    {
        return _permutation;
    }

private:
    std::vector<int64_t> _permutation;
};

/// @copydoc Reshape_attributes
class Slice_attributes : public detail::UnsupportedAttributes<Slice_attributes>
{
public:
    Slice_attributes() {} // NOLINT(modernize-use-equals-default)

    Slice_attributes& set_slices(const std::vector<std::pair<int64_t, int64_t>>& slices)
    {
        _slices = slices;
        return *this;
    }

    const std::vector<std::pair<int64_t, int64_t>>& get_slices() const
    {
        return _slices;
    }

    Slice_attributes& set_strides(const std::vector<int64_t>& strides)
    {
        _strides = strides;
        return *this;
    }

    const std::vector<int64_t>& get_strides() const
    {
        return _strides;
    }

private:
    std::vector<std::pair<int64_t, int64_t>> _slices;
    std::vector<int64_t> _strides = {1};
};

/// @copydoc Reshape_attributes
class Concatenate_attributes : public detail::UnsupportedAttributes<Concatenate_attributes>
{
public:
    Concatenate_attributes() {} // NOLINT(modernize-use-equals-default)

    Concatenate_attributes& set_axis(int64_t axis)
    {
        _axis = axis;
        return *this;
    }

    int64_t get_axis() const
    {
        return _axis;
    }

    Concatenate_attributes& set_in_place_index(int64_t index)
    {
        _inPlaceIndex = index;
        return *this;
    }

    int64_t get_in_place_index() const
    {
        return _inPlaceIndex;
    }

private:
    int64_t _axis = 0;
    int64_t _inPlaceIndex = -1;
};

/// @copydoc Reshape_attributes
class AdaLayernorm_attributes : public detail::UnsupportedAttributes<AdaLayernorm_attributes>
{
public:
    AdaLayernorm_attributes() {} // NOLINT(modernize-use-equals-default)

    AdaLayernorm_attributes& set_forward_phase(NormFwdPhase_t phase)
    {
        _forwardPhase = phase;
        return *this;
    }

    NormFwdPhase_t get_forward_phase() const
    {
        return _forwardPhase;
    }

    AdaLayernorm_attributes& set_epsilon(std::shared_ptr<Tensor_attributes> epsilon)
    {
        _epsilon = std::move(epsilon);
        return *this;
    }

    const std::shared_ptr<Tensor_attributes>& get_epsilon() const
    {
        return _epsilon;
    }

private:
    NormFwdPhase_t _forwardPhase = NormFwdPhase_t::NOT_SET;
    std::shared_ptr<Tensor_attributes> _epsilon;
};

/// @copydoc Reshape_attributes
class BN_finalize_attributes : public detail::UnsupportedAttributes<BN_finalize_attributes>
{
public:
    BN_finalize_attributes() {} // NOLINT(modernize-use-equals-default)

    // Non-const references mirror upstream's signature; consumer source passes
    // lvalue shared_ptrs and an rvalue would not bind there either.
    BN_finalize_attributes& set_previous_running_stats(std::shared_ptr<Tensor_attributes>& mean,
                                                       std::shared_ptr<Tensor_attributes>& variance,
                                                       std::shared_ptr<Tensor_attributes>& momentum)
    {
        _previousRunningMean = mean;
        _previousRunningVariance = variance;
        _momentum = momentum;
        return *this;
    }

    const std::shared_ptr<Tensor_attributes>& get_previous_running_mean() const
    {
        return _previousRunningMean;
    }

    const std::shared_ptr<Tensor_attributes>& get_previous_running_variance() const
    {
        return _previousRunningVariance;
    }

    const std::shared_ptr<Tensor_attributes>& get_momentum() const
    {
        return _momentum;
    }

private:
    std::shared_ptr<Tensor_attributes> _previousRunningMean;
    std::shared_ptr<Tensor_attributes> _previousRunningVariance;
    std::shared_ptr<Tensor_attributes> _momentum;
};

/// @copydoc Reshape_attributes
class AdaLayernorm_backward_attributes
    : public detail::UnsupportedAttributes<AdaLayernorm_backward_attributes>
{
public:
    AdaLayernorm_backward_attributes() {} // NOLINT(modernize-use-equals-default)

    AdaLayernorm_backward_attributes&
        set_saved_mean_and_inv_variance(std::shared_ptr<Tensor_attributes> mean,
                                        std::shared_ptr<Tensor_attributes> invVariance)
    {
        _mean = std::move(mean);
        _invVariance = std::move(invVariance);
        return *this;
    }

    const std::shared_ptr<Tensor_attributes>& get_saved_mean() const
    {
        return _mean;
    }

    const std::shared_ptr<Tensor_attributes>& get_saved_inv_variance() const
    {
        return _invVariance;
    }

private:
    std::shared_ptr<Tensor_attributes> _mean;
    std::shared_ptr<Tensor_attributes> _invVariance;
};

// NOLINTEND(readability-identifier-naming)

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::graph

// Stamp a Tier-2 fail-stub node method body. Records GRAPH_NOT_SUPPORTED on the
// composition Graph — surfaced at the next validate()/build_operation_graph() —
// with a message pointing at the issue tracker, then returns a live,
// graph-registered placeholder result (a real Tensor_attributes, or an array of
// them for multi-output nodes) so idiomatic cuDNN FE chaining
// (`node(...)->set_output(true).set_uid(n)`) survives instead of dereferencing
// null before the recorded error can surface.
//
// Expands inside the shim graph wrapper (detail/graph_wrapper.h), so it relies on
// that class providing recordError(); the placeholder result is built by
// detail::makeUnsupportedNodeResult<Result>(graph) above.
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
        return detail::makeUnsupportedNodeResult<__VA_ARGS__>(*this);                   \
    }
