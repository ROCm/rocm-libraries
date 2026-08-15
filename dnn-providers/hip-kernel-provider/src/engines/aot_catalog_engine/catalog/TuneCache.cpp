// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "catalog/TuneCache.hpp"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <variant>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <nlohmann/json.hpp>

namespace aot_catalog_engine::catalog
{

namespace fs = std::filesystem;
using nlohmann::json;

namespace
{

// Render one ShapeValue deterministically by variant type. bool -> 0/1, int64 ->
// decimal, double -> "%g" (compact, round-trip-stable enough for shape keys),
// string verbatim. Two problems with equal shapes always produce equal keys.
std::string renderShapeValue(const ShapeValue& value)
{
    return std::visit(
        [](const auto& v) -> std::string {
            using T = std::decay_t<decltype(v)>;
            if constexpr(std::is_same_v<T, bool>)
            {
                return v ? "1" : "0";
            }
            else if constexpr(std::is_same_v<T, int64_t>)
            {
                return std::to_string(v);
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                std::array<char, 32> buf{};
                const int n = std::snprintf(buf.data(), buf.size(), "%g", v);
                return (n > 0) ? std::string(buf.data(), static_cast<size_t>(n)) : std::string();
            }
            else // std::string
            {
                return v;
            }
        },
        value);
}

// Resolve the default persistence path: HIPDNN_AOT_TUNE_CACHE if set, else
// <temp_dir>/hipdnn_aot_tune_cache.json.
std::string defaultCachePath()
{
    // getEnv (data_sdk PlatformUtils) is cross-platform -- std::getenv trips
    // MSVC's -Wdeprecated-declarations under -Werror on the Windows superbuild.
    if(const std::string env = hipdnn_data_sdk::utilities::getEnv("HIPDNN_AOT_TUNE_CACHE");
       !env.empty())
    {
        return env;
    }
    std::error_code ec;
    const fs::path tmp = fs::temp_directory_path(ec);
    if(ec)
    {
        return {};
    }
    return (tmp / "hipdnn_aot_tune_cache.json").string();
}

} // namespace

std::string problemKey(const std::string& family, const ProblemShape& problem)
{
    std::string key = family;
    key += '|';
    bool first = true;
    // ProblemShape is a std::map -> iteration is key-ordered and deterministic.
    for(const auto& [name, value] : problem)
    {
        if(!first)
        {
            key += ',';
        }
        first = false;
        key += name;
        key += '=';
        key += renderShapeValue(value);
    }
    return key;
}

TuneCache::TuneCache()
    : _path(defaultCachePath())
{
    const std::lock_guard<std::mutex> lock(_mutex);
    loadLocked();
}

TuneCache::TuneCache(std::string path)
    : _path(std::move(path))
{
    const std::lock_guard<std::mutex> lock(_mutex);
    loadLocked();
}

std::optional<std::string> TuneCache::lookup(const std::string& key) const
{
    const std::lock_guard<std::mutex> lock(_mutex);
    auto it = _entries.find(key);
    if(it == _entries.end())
    {
        return std::nullopt;
    }
    return it->second.symbol;
}

void TuneCache::store(const std::string& key, const std::string& symbol, double ms)
{
    const std::lock_guard<std::mutex> lock(_mutex);
    _entries[key] = Entry{symbol, ms};
    saveLocked();
}

void TuneCache::loadLocked()
{
    if(_path.empty())
    {
        return;
    }
    std::ifstream in(_path);
    if(!in.is_open())
    {
        return; // no file yet -> empty cache (not an error)
    }

    json root;
    try
    {
        in >> root;
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_WARN("aot-catalog: ignoring unreadable tune cache '"
                               << _path << "': " << e.what());
        return;
    }

    if(!root.is_object() || !root.contains("entries") || !root.at("entries").is_object())
    {
        return;
    }

    for(const auto& [key, entry] : root.at("entries").items())
    {
        if(!entry.is_object() || !entry.contains("symbol") || !entry.at("symbol").is_string())
        {
            continue;
        }
        Entry parsed;
        parsed.symbol = entry.at("symbol").get<std::string>();
        if(entry.contains("ms") && entry.at("ms").is_number())
        {
            parsed.ms = entry.at("ms").get<double>();
        }
        _entries.emplace(key, std::move(parsed));
    }
}

void TuneCache::saveLocked() const
{
    if(_path.empty())
    {
        return;
    }

    json entries = json::object();
    for(const auto& [key, entry] : _entries)
    {
        entries[key] = json{{"symbol", entry.symbol}, {"ms", entry.ms}};
    }
    const json root = json{{"entries", std::move(entries)}};

    // Atomic-ish write: dump to a temp sibling then rename over the target so a
    // concurrent reader never sees a half-written file.
    const std::string tmpPath = _path + ".tmp";
    {
        std::ofstream out(tmpPath, std::ios::trunc);
        if(!out.is_open())
        {
            HIPDNN_PLUGIN_LOG_WARN("aot-catalog: could not open tune cache '" << tmpPath
                                                                              << "' for writing");
            return;
        }
        out << root.dump(2);
        if(!out.good())
        {
            HIPDNN_PLUGIN_LOG_WARN("aot-catalog: failed writing tune cache '" << tmpPath << "'");
            return;
        }
    }

    std::error_code ec;
    fs::rename(tmpPath, _path, ec);
    if(ec)
    {
        HIPDNN_PLUGIN_LOG_WARN("aot-catalog: could not commit tune cache '"
                               << _path << "': " << ec.message());
        std::error_code rmEc;
        fs::remove(tmpPath, rmEc);
    }
}

} // namespace aot_catalog_engine::catalog
