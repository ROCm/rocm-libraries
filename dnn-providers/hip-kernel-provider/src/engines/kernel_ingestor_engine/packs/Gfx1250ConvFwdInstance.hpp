// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdTypes.hpp"

/**
 * @file Gfx1250ConvFwdInstance.hpp
 * @brief An instance's catalog fields: its rocKE KernelIdentity, read once.
 *
 * The KMD carries the identity fields the packaged launch reads and every family knob,
 * each an int or a string (builders/gfx1250/conv_fwd_aot.py, Gfx1250ConvFwdAotSpec). A
 * knob an instance leaves unset is 0 (an int knob the list sets is positive); a bool knob
 * is 1 when set true and 2 when set false. The families read the fields by name through
 * the accessors here, which refuse a missing field rather than default it: the KMD gives
 * every kernel every field, so a missing one is a catalog that does not match this port.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
{

class InstanceGeometry
{
public:
    using Ints = std::map<std::string, int64_t, std::less<>>;
    using Strings = std::map<std::string, std::string, std::less<>>;

    InstanceGeometry(Ints ints, Strings strings)
        : _ints(std::move(ints))
        , _strings(std::move(strings))
    {
        _label = str("instance_id");
    }

    /// The instance's id (its identity's stable hash), for messages.
    const std::string& label() const
    {
        return _label;
    }
    std::string algorithm() const
    {
        return str("algorithm");
    }
    DataType dtype() const
    {
        return str("dtype") == "bf16" ? DataType::BF16 : DataType::FP16;
    }

    int64_t integer(const char* field) const
    {
        const auto it = _ints.find(field);
        if(it == _ints.end())
        {
            throw std::invalid_argument(_label + ": no int field '" + field + "'");
        }
        return it->second;
    }
    bool flag(const char* field) const
    {
        return integer(field) != 0;
    }
    std::string str(const char* field) const
    {
        const auto it = _strings.find(field);
        if(it == _strings.end())
        {
            throw std::invalid_argument(std::string("instance: no string field '") + field + "'");
        }
        return it->second;
    }

    bool hasKnob(const char* knob) const
    {
        const auto i = _ints.find(knob);
        if(i != _ints.end())
        {
            return i->second != 0;
        }
        const auto s = _strings.find(knob);
        return s != _strings.end() && !s->second.empty();
    }
    /// An int tuning knob; @p fallback when the instance leaves it unset (the spec's
    /// default).
    int64_t knob(const char* knob, int64_t fallback) const
    {
        const auto value = integer(knob);
        return value == 0 ? fallback : value;
    }
    /// A bool tuning knob: 1 set true, 2 set false, 0 unset (@p fallback).
    bool boolKnob(const char* knob, bool fallback) const
    {
        const auto value = integer(knob);
        if(value == 0)
        {
            return fallback;
        }
        if(value != 1 && value != 2)
        {
            throw std::invalid_argument(_label + ": bool knob '" + knob + "' is "
                                        + std::to_string(value));
        }
        return value == 1;
    }
    std::string knobString(const char* knob, const std::string& fallback) const
    {
        const auto value = str(knob);
        return value.empty() ? fallback : value;
    }

private:
    Ints _ints;
    Strings _strings;
    std::string _label;
};

/// The shape fields an instance may bake (KernelIdentity, checked by
/// kernel_identity.identity_supports_problem): 0 is "taken at run time", any other value
/// must match the problem the kernel runs.
struct BakedShape
{
    int64_t filterH, filterW, strideH, strideW, dilationH, dilationW, padH, padW, cpg, kpg;

    static BakedShape parse(const InstanceGeometry& g)
    {
        return {g.integer("filter_h"),
                g.integer("filter_w"),
                g.integer("stride_h"),
                g.integer("stride_w"),
                g.integer("dilation_h"),
                g.integer("dilation_w"),
                g.integer("pad_h"),
                g.integer("pad_w"),
                g.integer("cpg"),
                g.integer("kpg")};
    }

    /// Why a problem with these fields (in the member order) is not this instance's.
    Refusal mismatch(const std::array<int64_t, 10>& actual) const
    {
        static constexpr std::array<const char*, 10> NAMES = {"filter_h",
                                                              "filter_w",
                                                              "stride_h",
                                                              "stride_w",
                                                              "dilation_h",
                                                              "dilation_w",
                                                              "pad_h",
                                                              "pad_w",
                                                              "cpg",
                                                              "kpg"};
        const std::array<int64_t, 10> baked
            = {filterH, filterW, strideH, strideW, dilationH, dilationW, padH, padW, cpg, kpg};
        for(size_t i = 0; i < baked.size(); ++i)
        {
            if(baked[i] != 0 && baked[i] != actual[i])
            {
                return std::string(NAMES[i]) + "=" + std::to_string(baked[i]) + " but problem has "
                       + std::to_string(actual[i]);
            }
        }
        return std::nullopt;
    }
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx1250_conv_fwd
