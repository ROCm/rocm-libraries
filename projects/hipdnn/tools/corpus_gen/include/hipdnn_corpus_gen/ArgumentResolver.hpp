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
/// Separate from builder dispatch so it can be tested without a device or graph library.
namespace hipdnn_corpus_gen
{

/// A dims/strides list, a dtype name, or a scalar.
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

/// Row-major contiguous strides for @p dims (`strides_of`, §4.3.6).
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
inline hipdnn_plugin_sdk::uhd::VariableContext contextFor(const ProblemPoint& point)
{
    hipdnn_plugin_sdk::uhd::VariableContext context;
    for(const auto& entry : point)
    {
        // Not a structured binding: capturing one in a lambda is C++20.
        const auto& name = entry.first;
        std::visit([&context, &name](const auto& held) { context.bind("$q." + name, held); },
                   entry.second);
    }
    return context;
}

} // namespace detail

/// @brief Resolves every argument of @p spec against @p point, in declaration order.
/// Order matters: `strides_of` reads an argument resolved earlier.
inline ArgumentResolution resolveArguments(const GraphBuilderSpec& spec, const ProblemPoint& point)
{
    ArgumentResolution resolution;

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
                resolution.error
                    = "argument '" + argument.name + "': no value for '" + argument.source + "'";
                return resolution;
            }
            std::visit([&resolved](const auto& value) { resolved.value = value; }, found->second);
            break;
        }

        case BuilderArgument::Kind::EXPR:
        {
            if(argument.scalarExpression)
            {
                try
                {
                    resolved.value = hipdnn_plugin_sdk::uhd::ExpressionSet::number(
                        argument.expressions.resolve(0, context));
                }
                catch(const std::exception& error)
                {
                    resolution.error = "argument '" + argument.name + "': " + error.what();
                    return resolution;
                }
                break;
            }

            std::vector<int64_t> dims;
            dims.reserve(argument.elementCount);
            try
            {
                for(size_t i = 0; i < argument.elementCount; ++i)
                {
                    if(i < argument.elementConditions.size()
                       && argument.elementConditions[i].has_value()
                       && !hipdnn_plugin_sdk::uhd::ExpressionSet::truth(
                           argument.expressions.resolve(*argument.elementConditions[i], context)))
                    {
                        continue;
                    }
                    const auto value = argument.expressions.resolve(i, context);
                    if(value.isInt())
                    {
                        dims.push_back(value.asInt());
                        continue;
                    }
                    const double number
                        = std::floor(hipdnn_plugin_sdk::uhd::ExpressionSet::number(value));
                    if(number < static_cast<double>(std::numeric_limits<int64_t>::min())
                       || number >= -static_cast<double>(std::numeric_limits<int64_t>::min()))
                    {
                        throw hipdnn_plugin_sdk::uhd::JsonLogicError(
                            "Dimension exceeds signed 64-bit range");
                    }
                    dims.push_back(static_cast<int64_t>(number));
                }
            }
            catch(const std::exception& error)
            {
                // Fail closed (§6.2): a substituted value would mislabel the graph.
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
                resolution.error
                    = "argument '" + argument.name + "': no value for '" + argument.source + "'";
                return resolution;
            }
            const auto* name = std::get_if<std::string>(&found->second);
            if(name == nullptr)
            {
                resolution.error = "argument '" + argument.name + "': '" + argument.source
                                   + "' is not a dtype name";
                return resolution;
            }
            // Kept as the declared name; the dispatch maps it to the FlatBuffers enum.
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
