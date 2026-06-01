// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "KernelArtifact.hpp"

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <string>

namespace ck_dsl_provider {

ArgSchema::Kind parseArgKind(const std::string& kindStr) {
    // Keep the wire-string set tight on purpose: any new kind must be
    // added here AND in the Python side's `_smoke_arg_schema` /
    // `to_payload` emitter, so a typo cannot silently degrade to a
    // generic fallback that mis-aligns the arg buffer.
    if (kindStr == "Pointer") {
        return ArgSchema::Kind::Pointer;
    }
    if (kindStr == "I32") {
        return ArgSchema::Kind::I32;
    }
    if (kindStr == "I64") {
        return ArgSchema::Kind::I64;
    }
    if (kindStr == "F32") {
        return ArgSchema::Kind::F32;
    }
    if (kindStr == "F16") {
        return ArgSchema::Kind::F16;
    }
    throw hipdnn_plugin_sdk::HipdnnPluginException(
        HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
        std::string("KernelArtifact: unknown arg kind '") + kindStr +
            "' from compile_service. Extend parseArgKind to cover it.");
}

}  // namespace ck_dsl_provider
