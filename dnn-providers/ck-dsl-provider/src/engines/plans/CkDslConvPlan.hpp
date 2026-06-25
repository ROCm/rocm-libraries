// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <array>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>
#include <memory>

#include "CkDslHandle.hpp"
#include "ck_dsl_runtime/kernel.hpp"
#include "engines/CkDslConvParamParser.hpp"

namespace ck_dsl_plugin {

// Forward-conv plan (implicit-GEMM). The ck_dsl conv kernel bakes the conv
// geometry into its descriptor at build time, so its only runtime args are the
// three buffer pointers + their byte sizes (for the bounds-checked buffer
// resources). M = N*Ho*Wo, N_gemm = K, K_gemm = R*S*C.
class CkDslConvPlan : public hipdnn_plugin_sdk::IPlan<CkDslHandle> {
   public:
    CkDslConvPlan(CkDslConvParamParser::ParsedConvParams params,
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
    CkDslConvParamParser::ParsedConvParams params_;
    std::unique_ptr<ck_dsl::Kernel> kernel_;
};

}  // namespace ck_dsl_plugin
