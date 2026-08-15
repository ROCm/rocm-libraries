// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "launch/LaunchAbi.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "launch/PluginError.hpp"

namespace aot_catalog_engine::launch
{

using catalog::ArgKind;
using catalog::GridAxis;
using catalog::GridAxisKind;
using catalog::GridValue;
using catalog::ScalarType;
using catalog::WorkspaceExpr;
using catalog::WsOp;

namespace
{

int64_t evalGridValue(const GridValue& value, const SymbolTable& symbols)
{
    if(value.symbol.has_value())
    {
        auto it = symbols.find(*value.symbol);
        if(it == symbols.end())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: grid references undefined symbol '" + *value.symbol
                                 + "'");
        }
        return it->second;
    }
    return value.literal;
}

int64_t evalGridAxis(const GridAxis& axis, const SymbolTable& symbols)
{
    int64_t result = 0;
    switch(axis.kind)
    {
    case GridAxisKind::VALUE:
        result = evalGridValue(axis.value, symbols);
        break;
    case GridAxisKind::CEIL_DIV:
    {
        const int64_t num = evalGridValue(axis.numerator, symbols);
        const int64_t den = evalGridValue(axis.denominator, symbols);
        if(den == 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: grid ceil_div by zero");
        }
        result = (num + den - 1) / den;
        break;
    }
    case GridAxisKind::FLOOR_DIV:
    {
        const int64_t num = evalGridValue(axis.numerator, symbols);
        const int64_t den = evalGridValue(axis.denominator, symbols);
        if(den == 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: grid floor_div by zero");
        }
        result = num / den;
        break;
    }
    default:
        break;
    }

    if(axis.addend.has_value())
    {
        result += evalGridValue(*axis.addend, symbols);
    }

    if(result < 0)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog: grid axis evaluated to a negative value");
    }
    return result;
}

void appendBytes(std::vector<std::byte>& buffer, const void* src, size_t count)
{
    const auto* bytes = static_cast<const std::byte*>(src);
    buffer.insert(buffer.end(), bytes, bytes + count);
}

} // namespace

std::vector<ScalarValue> bindArgs(const std::vector<KernelArgument>& signature,
                                  const LaunchBindings& bindings,
                                  const PointerResolver& resolvePointer)
{
    std::vector<ScalarValue> bound;
    bound.reserve(signature.size());

    for(const auto& arg : signature)
    {
        if(arg.kind == ArgKind::POINTER)
        {
            // Prefer an already-known raw pointer value; otherwise resolve the
            // bound device-buffer uid to a pointer.
            if(auto valueIt = bindings.pointerValues.find(arg.name);
               valueIt != bindings.pointerValues.end())
            {
                bound.emplace_back(valueIt->second);
            }
            else if(auto uidIt = bindings.pointerUids.find(arg.name);
                    uidIt != bindings.pointerUids.end())
            {
                bound.emplace_back(resolvePointer(uidIt->second));
            }
            else
            {
                throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                 "aot-catalog: no pointer bound for argument '" + arg.name + "'");
            }
        }
        else // SCALAR
        {
            auto scalarIt = bindings.scalars.find(arg.name);
            if(scalarIt == bindings.scalars.end())
            {
                throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                 "aot-catalog: no scalar bound for argument '" + arg.name + "'");
            }
            bound.emplace_back(scalarIt->second);
        }
    }

    return bound;
}

std::vector<std::byte> packArgs(const std::vector<KernelArgument>& signature,
                                const std::vector<ScalarValue>& bound)
{
    if(signature.size() != bound.size())
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "aot-catalog: bound-argument count does not match signature");
    }

    std::vector<std::byte> buffer;
    size_t offset = 0;

    for(size_t i = 0; i < signature.size(); ++i)
    {
        const auto& arg = signature[i];
        const auto& value = bound[i];
        const size_t size = catalog::argSizeBytes(arg);
        if(size == 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: zero-width argument '" + arg.name + "' in signature");
        }

        // Natural-alignment padding: bring `offset` up to a multiple of `size`.
        const size_t padding = (size - (offset % size)) % size;
        buffer.insert(buffer.end(), padding, std::byte{0});
        offset += padding;

        if(arg.kind == ArgKind::POINTER)
        {
            const uint64_t raw = std::get<uint64_t>(value);
            appendBytes(buffer, &raw, sizeof(raw));
        }
        else if(arg.scalarType == ScalarType::F32)
        {
            const float raw = std::get<float>(value);
            appendBytes(buffer, &raw, sizeof(raw));
        }
        else if(arg.scalarType == ScalarType::I32)
        {
            const auto raw = static_cast<int32_t>(std::get<int64_t>(value));
            appendBytes(buffer, &raw, sizeof(raw));
        }
        else // I64
        {
            const int64_t raw = std::get<int64_t>(value);
            appendBytes(buffer, &raw, sizeof(raw));
        }

        offset += size;
    }

    return buffer;
}

Grid evalGrid(const GridFormula& formula, const SymbolTable& symbols)
{
    Grid grid;
    grid.x = static_cast<uint32_t>(evalGridAxis(formula.x, symbols));
    grid.y = static_cast<uint32_t>(evalGridAxis(formula.y, symbols));
    grid.z = static_cast<uint32_t>(evalGridAxis(formula.z, symbols));
    return grid;
}

int64_t evalWorkspace(const WorkspaceExpr& expr, const SymbolTable& symbols)
{
    switch(expr.op)
    {
    case WsOp::LITERAL:
        return expr.literal;
    case WsOp::SYMBOL:
    {
        auto it = symbols.find(expr.symbol);
        if(it == symbols.end())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: workspace references undefined symbol '" + expr.symbol
                                 + "'");
        }
        return it->second;
    }
    case WsOp::MUL:
    {
        int64_t result = 1;
        for(const auto& arg : expr.args)
        {
            result *= evalWorkspace(arg, symbols);
        }
        return result;
    }
    case WsOp::ADD:
    {
        int64_t result = 0;
        for(const auto& arg : expr.args)
        {
            result += evalWorkspace(arg, symbols);
        }
        return result;
    }
    case WsOp::MIN:
    {
        int64_t result = evalWorkspace(expr.args.front(), symbols);
        for(size_t i = 1; i < expr.args.size(); ++i)
        {
            result = std::min(result, evalWorkspace(expr.args[i], symbols));
        }
        return result;
    }
    case WsOp::MAX:
    {
        int64_t result = evalWorkspace(expr.args.front(), symbols);
        for(size_t i = 1; i < expr.args.size(); ++i)
        {
            result = std::max(result, evalWorkspace(expr.args[i], symbols));
        }
        return result;
    }
    case WsOp::SUB:
    {
        const int64_t lhs = evalWorkspace(expr.args[0], symbols);
        const int64_t rhs = evalWorkspace(expr.args[1], symbols);
        const int64_t result = lhs - rhs;
        if(result < 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: workspace 'sub' evaluated to a negative value");
        }
        return result;
    }
    case WsOp::CEIL_DIV:
    {
        const int64_t num = evalWorkspace(expr.args[0], symbols);
        const int64_t den = evalWorkspace(expr.args[1], symbols);
        if(den == 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: workspace ceil_div by zero");
        }
        return (num + den - 1) / den;
    }
    case WsOp::FLOOR_DIV:
    {
        const int64_t num = evalWorkspace(expr.args[0], symbols);
        const int64_t den = evalWorkspace(expr.args[1], symbols);
        if(den == 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: workspace floor_div by zero");
        }
        return num / den;
    }
    case WsOp::ALIGN_UP:
    {
        const int64_t value = evalWorkspace(expr.args[0], symbols);
        const int64_t align = evalWorkspace(expr.args[1], symbols);
        if(align == 0)
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "aot-catalog: workspace align_up by zero");
        }
        return ((value + align - 1) / align) * align;
    }
    default:
        break;
    }

    // Unreachable: parseWorkspaceExpr only emits the ops above. Fail closed if a
    // new op is ever added to the enum without a case here.
    throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                     "aot-catalog: unhandled workspace operator");
}

} // namespace aot_catalog_engine::launch
