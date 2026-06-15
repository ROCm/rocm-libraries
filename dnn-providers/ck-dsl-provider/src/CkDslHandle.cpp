// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CkDslHandle.hpp"

#include <cstdlib>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#include "CkDslContainer.hpp"

namespace {

std::string detect_gfx_arch() {
    hipDeviceProp_t prop;
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                                       "CkDslHandle: hipGetDevice failed");
    if (hipGetDeviceProperties(&prop, device) != hipSuccess)
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR, "CkDslHandle: hipGetDeviceProperties failed");
    return prop.gcnArchName;
}

}  // namespace

CkDslHandle::CkDslHandle() {
    gfx_arch_ = detect_gfx_arch();
    auto colon = gfx_arch_.find(':');
    if (colon != std::string::npos) gfx_arch_ = gfx_arch_.substr(0, colon);
    isa_ = ck_dsl::Compiler::isa_for(gfx_arch_);

    store_ = std::make_unique<ck_dsl::ArtifactStore>();

    // Per-arch bundle dir(s) from the environment. CK_DSL_KERNEL_LIB_PATH points
    // at a directory of shipped artifacts (manifest + .hsaco and/or .ll).
    size_t n = 0;
    if (const char* p = std::getenv("CK_DSL_KERNEL_LIB_PATH")) n += store_->add_bundle(p);

    dispatcher_ = std::make_unique<ck_dsl::Dispatcher>(*store_);

    // Trained-model kernel selection: when CK_DSL_ML_MODEL_DIR is set, rank
    // candidates by predicted TFLOPS instead of FirstFit.
    // Expected layout: <dir>/gemm/model_tflops.lgbm
    //                  <dir>/fmha/model_tflops.lgbm
    //                  <dir>/conv/model_tflops.lgbm
    const char* model = std::getenv("CK_DSL_ML_MODEL_DIR");
    bool ml = false;
    if (model != nullptr) {
        ml_heuristic_ = std::make_unique<ck_dsl::DslMlHeuristic>(model, store_.get());
        if (ml_heuristic_->is_loaded()) {
            auto* h = ml_heuristic_.get();
            dispatcher_->set_heuristic(
                [h](const ck_dsl::Problem& p, std::vector<ck_dsl::Dispatcher::Choice> c) {
                    return (*h)(p, std::move(c));
                });
            ml = true;
        } else {
            ml_heuristic_.reset();
        }
    }

    HIPDNN_PLUGIN_LOG_INFO("CkDslHandle: arch=" << gfx_arch_ << " kernels=" << n << " selection="
                                                << (ml ? "ml_heuristic" : "firstfit")
                                                << (ml && ml_heuristic_->has_conv() ? " conv_model=yes"
                                                                                    : ""));
}

CkDslHandle::~CkDslHandle() = default;

hipdnn_plugin_sdk::EngineManager<CkDslHandle, ck_dsl_plugin::CkDslSettings, CkDslContext>&
CkDslHandle::getEngineManager() {
    return container->getEngineManager();
}
