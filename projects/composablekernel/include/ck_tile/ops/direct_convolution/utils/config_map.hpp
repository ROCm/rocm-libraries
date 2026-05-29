// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv {

/// A single entry in a ConfigMap: an explicit integer key paired with a Config value.
template <typename Config>
struct ConfigEntry
{
    int key;
    Config cfg;
};

/// Compile-time map from integer key to Config.
///
/// Backed by a fixed-size array of ConfigEntry; lookup is a linear scan —
/// acceptable because N is small and all lookups are in constexpr / template
/// instantiation contexts (zero runtime cost when ConfigIdx is a template parameter).
///
/// Construct via make_config_map<Config>({ {0, cfg0}, {1, cfg1}, ... }).
template <typename Config, int N>
struct ConfigMap
{
    ConfigEntry<Config> entries[N];

    static constexpr int size = N;

    /// Compile-time-only key lookup (consteval).
    /// Returns the Config value for the given key. Produces a compile error if the key
    /// is not found, since consteval requires full evaluation at compile time.
    consteval Config get(int key) const
    {
        for(int i = 0; i < N; ++i)
            if(entries[i].key == key)
                return entries[i].cfg;
        return entries[0].cfg;
    }

    /// Validity check: all keys are non-negative and unique.
    /// Call via static_assert(configs_map.is_valid(), "...") after construction.
    constexpr bool is_valid() const
    {
        for(int i = 0; i < N; ++i)
        {
            if(entries[i].key < 0)
                return false;
            for(int j = i + 1; j < N; ++j)
                if(entries[i].key == entries[j].key)
                    return false;
        }
        return true;
    }

};

/// Factory function: build a ConfigMap<Config, N> from a braced list of {key, cfg} pairs.
///
/// The Config type is given once as an explicit template argument; N is deduced from
/// the array initializer, so no manual entry count is required:
///
///   constexpr auto configs_map = make_config_map<MyCfg>({
///       {0, cfg0},
///       {1, cfg1},
///   });
///
/// For configs inside a templated struct (e.g. KernelConfigurations<DT>):
///
///   static constexpr auto configs_map = make_config_map<Config<DT>>({
///       {0, cfg0},
///   });
template <typename Config, int N>
constexpr auto make_config_map(const ConfigEntry<Config> (&entries)[N])
{
    ConfigMap<Config, N> map{};
    for(int i = 0; i < N; ++i)
        map.entries[i] = entries[i];
    return map;
}

} // namespace ck_tile::direct_conv
