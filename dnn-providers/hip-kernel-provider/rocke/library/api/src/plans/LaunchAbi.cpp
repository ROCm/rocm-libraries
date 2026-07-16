// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "plans/LaunchAbi.hpp"
#include "plans/PluginError.hpp"

namespace rocke_client::launch
{
namespace
{

std::int64_t evalGridValue(const dispatcher::GridValue& value,
                           const std::unordered_map<std::string, std::int64_t>& symbols)
{
    if(value.symbol.has_value())
    {
        const auto iter = symbols.find(*value.symbol);
        if(iter == symbols.end())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "unknown rocKE launch grid symbol: " + *value.symbol);
        }
        return iter->second;
    }
    return value.literal;
}

unsigned int evalGridAxis(const dispatcher::GridAxis& axis,
                          const std::unordered_map<std::string, std::int64_t>& symbols)
{
    if(axis.kind == dispatcher::GridAxis::Kind::VALUE)
    {
        return static_cast<unsigned int>(evalGridValue(axis.value, symbols));
    }

    const auto numerator = evalGridValue(axis.numerator, symbols);
    const auto denominator = evalGridValue(axis.denominator, symbols);
    if(denominator <= 0)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                         "rocKE launch grid ceil_div denominator must be positive");
    }
    return static_cast<unsigned int>((numerator + denominator - 1) / denominator);
}

void appendBytes(std::vector<std::byte>& packed, const void* source, std::size_t size)
{
    const auto* bytes = static_cast<const std::byte*>(source);
    packed.insert(packed.end(), bytes, bytes + size);
}

std::uint64_t resolveDevicePointer(const std::unordered_map<std::int64_t, void*>& ptrs,
                                   std::int64_t uid,
                                   const std::string& name)
{
    const auto iter = ptrs.find(uid);
    if(iter == ptrs.end() || iter->second == nullptr)
    {
        throwPluginError(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                         "missing device buffer for rocKE tensor " + name);
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(iter->second));
}

} // namespace

std::unordered_map<std::string, ScalarValue>
    bindArgs(const std::vector<dispatcher::KernelArgument>& signature,
             const dispatcher::LaunchBindings& bindings,
             const std::unordered_map<std::int64_t, void*>& devicePtrs)
{
    std::unordered_map<std::string, ScalarValue> values;
    values.reserve(signature.size());
    for(const auto& arg : signature)
    {
        if(arg.kind == dispatcher::ArgKind::POINTER)
        {
            const auto iter = bindings.pointerUids.find(arg.name);
            if(iter == bindings.pointerUids.end())
            {
                throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                 "no launch binding for rocKE pointer argument: " + arg.name);
            }
            values.emplace(arg.name, resolveDevicePointer(devicePtrs, iter->second, arg.name));
        }
        else
        {
            const auto iter = bindings.scalars.find(arg.name);
            if(iter == bindings.scalars.end())
            {
                throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                 "no launch binding for rocKE scalar argument: " + arg.name);
            }
            values.emplace(arg.name, iter->second);
        }
    }
    return values;
}

std::vector<std::byte> packArgs(const std::vector<dispatcher::KernelArgument>& signature,
                                const std::unordered_map<std::string, ScalarValue>& values)
{
    std::vector<std::byte> packed;
    std::size_t offset = 0;
    for(const auto& arg : signature)
    {
        const std::size_t size = dispatcher::argSizeBytes(arg);
        const std::size_t padding = (size - (offset % size)) % size;
        packed.insert(packed.end(), padding, std::byte{0});
        offset += padding;

        const auto valueIter = values.find(arg.name);
        if(valueIter == values.end())
        {
            throwPluginError(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                             "missing rocKE launch argument value for " + arg.name);
        }

        if(arg.kind == dispatcher::ArgKind::POINTER)
        {
            const auto value = std::get<std::uint64_t>(valueIter->second);
            appendBytes(packed, &value, sizeof(value));
        }
        else if(arg.scalarType == dispatcher::ScalarType::F32)
        {
            const auto value = std::get<float>(valueIter->second);
            appendBytes(packed, &value, sizeof(value));
        }
        else if(arg.scalarType == dispatcher::ScalarType::I32)
        {
            const auto value = static_cast<std::int32_t>(std::get<std::int64_t>(valueIter->second));
            appendBytes(packed, &value, sizeof(value));
        }
        else
        {
            const auto value = std::get<std::int64_t>(valueIter->second);
            appendBytes(packed, &value, sizeof(value));
        }
        offset += size;
    }
    return packed;
}

std::array<unsigned int, 3> evalGrid(const dispatcher::GridFormula& formula,
                                     const std::unordered_map<std::string, std::int64_t>& symbols)
{
    return {evalGridAxis(formula.x, symbols),
            evalGridAxis(formula.y, symbols),
            evalGridAxis(formula.z, symbols)};
}

} // namespace rocke_client::launch
