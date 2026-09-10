// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_backend.h>

#include "common/Utilities.hpp"
#include "harness/bundle/LoadedEngine.hpp"

namespace hipdnn_integration_tests::bundle
{

class LoadedEngineTable
{
public:
    static LoadedEngineTable& get()
    {
        static LoadedEngineTable s_instance;
        return s_instance;
    }

    LoadedEngineTable(const LoadedEngineTable&) = delete;
    LoadedEngineTable& operator=(const LoadedEngineTable&) = delete;
    LoadedEngineTable(LoadedEngineTable&&) = delete;
    LoadedEngineTable& operator=(LoadedEngineTable&&) = delete;

    void build(hipdnnHandle_t handle)
    {
        _engines.clear();
        _built = false;

        size_t numEngines = 0;
        if(hipdnnGetEngineCount_ext(handle, &numEngines) != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("[LoadedEngineTable] hipdnnGetEngineCount_ext failed");
        }

        if(numEngines == 0)
        {
            throw std::runtime_error(
                "[LoadedEngineTable] no engines loaded — check the plugin path");
        }

        _engines.reserve(numEngines);
        for(size_t i = 0; i < numEngines; ++i)
        {
            auto info = getEngineInfo(handle, i);
            _engines.push_back(LoadedEngine{info.engineId, std::move(info.engineName)});
        }

        _built = true;
    }

    void setForTesting(std::vector<LoadedEngine> engines)
    {
        _engines = std::move(engines);
        _built = true;
    }

    void reset()
    {
        _engines.clear();
        _built = false;
    }

    const std::vector<LoadedEngine>& all() const
    {
        requireBuilt();
        return _engines;
    }

    bool isBuilt() const
    {
        return _built;
    }

    bool isLoaded(std::string_view name) const
    {
        requireBuilt();
        return std::any_of(_engines.begin(), _engines.end(), [name](const LoadedEngine& e) {
            return e.name == name;
        });
    }

    /// The loaded engine with this name, or nullptr. Used to resolve --test-engine
    /// into the LoadedEngine injected into each harness, so the harness never has
    /// to reach back into a singleton to learn what it is running.
    const LoadedEngine* find(std::string_view name) const
    {
        requireBuilt();
        const auto it = std::find_if(_engines.begin(),
                                     _engines.end(),
                                     [name](const LoadedEngine& e) { return e.name == name; });
        return it == _engines.end() ? nullptr : &*it;
    }

private:
    LoadedEngineTable() = default;

    void requireBuilt() const
    {
        if(!_built)
        {
            throw std::runtime_error("[LoadedEngineTable] accessed before build() succeeded");
        }
    }

    std::vector<LoadedEngine> _engines;
    bool _built = false;
};

} // namespace hipdnn_integration_tests::bundle
