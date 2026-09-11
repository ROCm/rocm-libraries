// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/OperationMetadata.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

/// @file ArgumentResolver.hpp
/// @brief Turning a problem point into a builder's arguments (RFC 0019.13 §4.3.6).
///
/// The corpus is parameter dictionaries; benchmarking needs a graph. This is the half of that
/// gap that can be closed without a graph library: resolving each declared argument to a
/// concrete value, in declaration order, so `strides_of` can refer to a dims list already
/// resolved.
///
/// Kept separate from the dispatch that calls the builder because the two fail differently and
/// only one of them needs a device. A resolver bug -- a dims list assembled in the wrong order,
/// strides computed against the wrong argument -- produces a graph that builds, benchmarks, and
/// describes a different problem than the row says. That is worth testing against arithmetic,
/// which is what this seam allows.
namespace hipdnn_corpus_gen
{

/// A resolved argument, in the shapes the builders take: dims and strides lists, a dtype name,
/// or a scalar.
using ResolvedValue = std::variant<std::vector<int64_t>, std::string, int64_t, double, bool>;

struct ResolvedArgument
{
    std::string name;
    ResolvedValue value;
};

struct ArgumentResolution
{
    std::vector<ResolvedArgument> arguments;
    std::string error;

    bool ok() const
    {
        return error.empty();
    }

    const ResolvedArgument* find(const std::string& name) const
    {
        for(const auto& argument : arguments)
        {
            if(argument.name == name)
            {
                return &argument;
            }
        }
        return nullptr;
    }
};

namespace detail
{

/// Row-major contiguous strides for @p dims, which is what `strides_of` means (§4.3.6).
inline std::vector<int64_t> rowMajorStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size() - 1; i-- > 0;)
    {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    return strides;
}

/// Binds a problem point as `$q.*` variables for the §6.2 evaluator.
inline hipdnn_plugin_sdk::ingestor::uhd::VariableContext contextFor(const ProblemPoint& point)
{
    hipdnn_plugin_sdk::ingestor::uhd::VariableContext context;
    for(const auto& entry : point)
    {
        // entry rather than a structured binding: capturing one in a lambda is C++20.
        const auto& name = entry.first;
        std::visit([&context, &name](const auto& held) { context.bind("$q." + name, held); },
                   entry.second);
    }
    return context;
}

} // namespace detail

/// @brief Resolves every argument of @p spec against @p point, in declaration order.
///
/// Declaration order is load-bearing rather than stylistic: `strides_of` reads an argument
/// resolved earlier, so resolving out of order would silently produce strides for an empty
/// dims list -- a rank-zero tensor that fails much later, somewhere unrelated.
inline ArgumentResolution resolveArguments(const GraphBuilderSpec& spec, const ProblemPoint& point)
{
    ArgumentResolution resolution;

    const hipdnn_plugin_sdk::ingestor::uhd::JsonLogicEvaluator evaluator;
    const auto context = detail::contextFor(point);

    for(const auto& argument : spec.arguments)
    {
        ResolvedArgument resolved;
        resolved.name = argument.name;

        switch(argument.kind)
        {
        case BuilderArgument::Kind::DIRECT:
        {
            const auto reference = detail::queryReference(argument.source);
            const auto found = point.find(reference.empty() ? argument.source : reference);
            if(found == point.end())
            {
                resolution.error = "argument '" + argument.name + "': no value for '"
                                   + argument.source + "'";
                return resolution;
            }
            std::visit([&resolved](const auto& value) { resolved.value = value; }, found->second);
            break;
        }

        case BuilderArgument::Kind::EXPR:
        {
            // Each element is a §6.2 expression; an array of them is a dims list. Evaluated by
            // the shared interpreter rather than a local one, so a convolution's output extent
            // -- (H + 2*pad - dilation*(R-1) - 1)/stride + 1 -- is expressible without this
            // file growing an arithmetic evaluator of its own.
            std::vector<int64_t> dims;
            dims.reserve(argument.value.size());
            try
            {
                for(const auto& term : argument.value)
                {
                    // Floored, not rounded to nearest. §6.2's `/` is double division and the
                    // set offers `ceil_div` but no floor form, while dimension arithmetic
                    // almost always needs floor: a convolution's output extent is
                    // floor((H + 2*pad - dilation*(R-1) - 1)/stride) + 1. Rounding 111.5 up
                    // invents an output element the kernel will not produce, and the graph
                    // then disagrees with the shape the engine derives -- so the mismatch
                    // surfaces as an applicability refusal, far from its cause.
                    dims.push_back(static_cast<int64_t>(
                        std::floor(evaluator.evaluateDouble(term, context))));
                }
            }
            catch(const std::exception& error)
            {
                // Fails closed, per §6.2: an unknown symbol or a type error is a metadata bug,
                // and a substituted value would build a graph that disagrees with its label.
                resolution.error = "argument '" + argument.name + "': " + error.what();
                return resolution;
            }
            resolved.value = std::move(dims);
            break;
        }

        case BuilderArgument::Kind::STRIDES_OF:
        {
            const auto* source = resolution.find(argument.of);
            if(source == nullptr)
            {
                resolution.error = "argument '" + argument.name + "': strides_of '" + argument.of
                                   + "', which is not resolved yet";
                return resolution;
            }
            const auto* dims = std::get_if<std::vector<int64_t>>(&source->value);
            if(dims == nullptr || dims->empty())
            {
                resolution.error = "argument '" + argument.name + "': strides_of '" + argument.of
                                   + "', which is not a dims list";
                return resolution;
            }
            resolved.value = detail::rowMajorStrides(*dims);
            break;
        }

        case BuilderArgument::Kind::DTYPE_OF:
        {
            const auto reference = detail::queryReference(argument.source);
            const auto found = point.find(reference.empty() ? argument.source : reference);
            if(found == point.end())
            {
                resolution.error = "argument '" + argument.name + "': no value for '"
                                   + argument.source + "'";
                return resolution;
            }
            const auto* name = std::get_if<std::string>(&found->second);
            if(name == nullptr)
            {
                resolution.error = "argument '" + argument.name + "': '" + argument.source
                                   + "' is not a dtype name";
                return resolution;
            }
            // Left as the declared name; mapping to the FlatBuffers enumerator belongs with
            // the dispatch, which is the only place that knows the enum exists.
            resolved.value = *name;
            break;
        }

        case BuilderArgument::Kind::CONSTANT:
        {
            if(argument.constant.is_array())
            {
                std::vector<int64_t> values;
                for(const auto& term : argument.constant)
                {
                    values.push_back(term.get<int64_t>());
                }
                resolved.value = std::move(values);
            }
            else
            {
                std::visit([&resolved](const auto& value) { resolved.value = value; },
                           detail::jsonToValue(argument.constant));
            }
            break;
        }

        default:
            resolution.error = "argument '" + argument.name + "' has an unhandled kind";
            return resolution;
        }

        resolution.arguments.push_back(std::move(resolved));
    }

    return resolution;
}

} // namespace hipdnn_corpus_gen
