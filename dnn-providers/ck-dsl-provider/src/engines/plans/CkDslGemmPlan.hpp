// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <memory>

#include "CkDslHandle.hpp"
#include "ck_dsl_runtime/kernel.hpp"
#include "engines/CkDslParamParser.hpp"

namespace ck_dsl_plugin {

// Holds a ck_dsl::Kernel already compiled at plan-build time (AOT). execute()
// maps variant-pack UIDs to device pointers and launches via the runtime.
// Direct kernel access is preserved: kernel().hsaco()/manifest()/cache_key().
class CkDslGemmPlan : public hipdnn_plugin_sdk::IPlan<CkDslHandle> {
   public:
    CkDslGemmPlan(CkDslParamParser::ParsedGemmParams params,
                  std::unique_ptr<ck_dsl::Kernel> kernel);

    size_t getWorkspaceSize(const CkDslHandle& handle) const override {
        return 0;
    }

    void execute(const CkDslHandle& handle, const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers, void* workspace) const override;

    const ck_dsl::Kernel& kernel() const {
        return *kernel_;
    }

   private:
    static void* findBuffer(int64_t uid, const hipdnnPluginDeviceBuffer_t* bufs, uint32_t count);

    CkDslParamParser::ParsedGemmParams params_;
    std::unique_ptr<ck_dsl::Kernel> kernel_;
};

}  // namespace ck_dsl_plugin
