// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "../runtime/KernelArtifact.hpp"
#include "CompilePayload.hpp"

namespace ck_dsl_provider {

/// Owns the embedded MicroPython interpreter's view of the frozen
/// ck_dsl_provider.compile_service module. One instance per process,
/// constructed by CkDslContainer after EmbeddedInterpreter::ensureInitialized().
///
/// ck_dsl is frozen into the plugin (or loaded from an on-disk bundle in the
/// dev build), so there is no sys.path baking. comgr is exposed to the
/// interpreter as the native ``comgr`` module (micropython/modcomgr.c), so
/// ck_dsl's runtime/comgr.py keeps its flow and compile() still returns a HSACO.
///
/// Every public method serialises on EmbeddedInterpreter::interpreterMutex()
/// (MicroPython has one global runtime state and no GIL) and translates any
/// Python-side exception into a HipdnnPluginException.
class CompileServiceBridge {
   public:
    CompileServiceBridge();
    ~CompileServiceBridge() noexcept;

    CompileServiceBridge(const CompileServiceBridge&) = delete;
    CompileServiceBridge& operator=(const CompileServiceBridge&) = delete;
    CompileServiceBridge(CompileServiceBridge&&) = delete;
    CompileServiceBridge& operator=(CompileServiceBridge&&) = delete;

    /// compile_service.compile_smoke(arch) -> KernelArtifact.
    KernelArtifact compileSmoke(std::string_view arch);

    /// compile_service.compile(op_kind, payload, arch) -> KernelArtifact.
    /// ``payload`` is the interpreter-neutral dict emitted by the matching
    /// per-op translator (e.g. convImplicitGemmSpecToPayload); the bridge
    /// marshals it to an mp_obj_t dict under the interpreter lock.
    KernelArtifact compile(std::string_view opKind, const PayloadDict& payload,
                           std::string_view arch);

    /// compile_service.is_applicable(op_kind, payload, arch) -> (ok, reason).
    std::pair<bool, std::string> isApplicable(std::string_view opKind, const PayloadDict& payload,
                                              std::string_view arch);

   private:
    /// Opaque mp_obj_t for the imported compile_service module. Kept alive by
    /// MicroPython's loaded-modules dict (a GC root), so storing the raw handle
    /// here is safe. Typed as void* to keep MicroPython headers out of this API.
    void* _module = nullptr;
};

}  // namespace ck_dsl_provider
