// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <optional>
#include <string>

namespace ck_dsl_plugin {

// Per-plan settings populated from engine-config knobs.
struct CkDslSettings {
    enum class SelectionMode { Builtin, Heuristic };

    SelectionMode selectionMode() const {
        return selection_mode_;
    }
    void setSelectionMode(SelectionMode m) {
        selection_mode_ = m;
    }

    const std::string& forcedCacheKey() const {
        return forced_cache_key_;
    }
    void setForcedCacheKey(const std::string& k) {
        forced_cache_key_ = k;
    }

    bool jitEnabled() const {
        return jit_enabled_;
    }
    void setJitEnabled(bool e) {
        jit_enabled_ = e;
    }

   private:
    SelectionMode selection_mode_ = SelectionMode::Builtin;
    std::string forced_cache_key_;
    bool jit_enabled_ = true;  // comgr-from-.ll fallback is in-process + cheap
};

}  // namespace ck_dsl_plugin
