// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <string>
#include <string_view>
#include <utility>

#include "../runtime/KernelArtifact.hpp"

namespace ck_dsl_provider {

/// Owns the embedded interpreter's view of the provider-local
/// ck_dsl_provider.compile_service module. One instance per process,
/// constructed from CkDslContainer's ctor after
/// EmbeddedInterpreter::ensureInitialized() has returned.
///
/// Responsibilities:
///   * Idempotently prepend the two CMake-baked package paths
///     (CK_DSL_PYTHON_PACKAGE_PATH and CK_DSL_PROVIDER_PYTHON_PACKAGE_PATH)
///     to sys.path so the embedded interpreter can resolve
///     `import ck_dsl_provider.compile_service` and the cross-package
///     `import ck_dsl`.
///   * Cache the imported compile_service module on a member so per-
///     call hot paths (I-7's compile()) avoid the import lookup cost.
///   * Translate every py::error_already_set crossing the boundary into
///     a HipdnnPluginException via PythonError::raise.
///
/// For M1 step I-3 the only exposed call is noopSmoke(), which returns
/// the constant dict from ck_dsl_provider.compile_service.noop_smoke()
/// and is used by the provider's unit tests to confirm the cross-
/// package import path works inside a plugin context.
class CompileServiceBridge {
   public:
    CompileServiceBridge();

    /// Releases the held py::module_ with the GIL held. Plain
    /// `= default` here would let pybind11 dec_ref a PyObject* on
    /// whatever thread tears the container down — at process exit that
    /// thread does not hold the GIL, which is undefined behaviour and
    /// asserts in CPython debug builds.
    ~CompileServiceBridge() noexcept;

    CompileServiceBridge(const CompileServiceBridge&) = delete;
    CompileServiceBridge& operator=(const CompileServiceBridge&) = delete;
    CompileServiceBridge(CompileServiceBridge&&) = delete;
    CompileServiceBridge& operator=(CompileServiceBridge&&) = delete;

    /// Invoke ck_dsl_provider.compile_service.noop_smoke() and return
    /// the resulting dict. Acquires the GIL internally and translates
    /// any Python error into a HipdnnPluginException.
    pybind11::dict noopSmoke();

    /// Invoke ck_dsl_provider.compile_service.compile_smoke() and
    /// translate the returned dict into a KernelArtifact ready to hand
    /// to HipModule. Acquires the GIL internally; any
    /// py::error_already_set is translated via PythonError::raise.
    ///
    /// The smoke artifact is a gfx950 elementwise-copy HSACO (see
    /// the Python side's compile_smoke docstring) -- enough to prove
    /// the round-trip from compile-service to hipModuleLaunchKernel
    /// without depending on any per-op adapter logic. Production
    /// compiles go through :func:`compile`.
    KernelArtifact compileSmoke();

    /// POC verification entry point (ALMIOPEN-2002, Phase 4). Invokes
    /// ``ck_dsl_provider.compile_service.compile_sdpa_fwd_fake(arch)`` and
    /// translates the returned dict into a ``KernelArtifact``. The fake
    /// kernel has the EXACT 18-slot unified-SDPA ABI so the production
    /// ``SdpaFwdPlan`` packs + launches it unchanged, but its body does no
    /// attention math -- thread 0 writes an ABI-slot fingerprint into the
    /// output buffer. Used by the gfx90a fake-launch test to verify the
    /// real ``execute()`` arg binding + marshalling + launch plumbing on
    /// hardware without the gfx950 kernel's numerics.
    ///
    /// ``arch`` is the explicit target gfx token (the POC passes the
    /// detected device arch). Acquires the GIL internally; any
    /// ``py::error_already_set`` is translated via ``PythonError::raise``.
    KernelArtifact compileSdpaFwdFake(std::string_view arch);

    /// Production compile entry point. Invokes
    /// ck_dsl_provider.compile_service.compile(op_kind, payload, arch),
    /// translates the returned dict into a ``KernelArtifact``, and
    /// returns it. The payload is the on-wire dict emitted by the
    /// matching per-op ``*Payload::*SpecToPayload`` translator (for
    /// M1: ``convImplicitGemmSpecToPayload``).
    ///
    /// ``arch`` is the target gfx token (e.g. ``"gfx950"``), passed
    /// separately from the payload because it is an orthogonal compile
    /// target, not a spec field -- mirroring the DSL entry points
    /// (``build_implicit_gemm_conv(spec, arch)`` /
    /// ``compile_kernel(kernel, arch)``).
    ///
    /// Acquires the GIL internally; the caller must NOT already hold
    /// it. Any ``py::error_already_set`` from the Python side is
    /// translated via ``PythonError::raise``.
    ///
    /// The ``opKind`` string is the same identifier the JitCache key
    /// derivation uses; pick a stable value per op and never rename.
    KernelArtifact compile(std::string_view opKind, const pybind11::dict& payload,
                           std::string_view arch);

    /// Arch-aware applicability predicate. Invokes
    /// ck_dsl_provider.compile_service.is_applicable(op_kind, payload,
    /// arch), which builds the op's spec from the payload and consults
    /// the DSL's ``is_valid_spec`` for ``arch`` WITHOUT compiling.
    /// ``arch`` is passed separately from the payload -- the same shape
    /// as :func:`compile` -- because it is an orthogonal compile target,
    /// not a spec field.
    ///
    /// Returns ``{ok, reason}``: ``ok`` is the verdict, ``reason`` is a
    /// human-readable explanation when the spec is not valid for the
    /// arch (e.g. an MMA atom absent on the target, a wave-size
    /// mismatch, or an unknown arch).
    ///
    /// This is the authoritative gate the plan builder calls from
    /// ``isApplicable`` so a false positive (applicable here, fails at
    /// ``buildPlan`` compile time) cannot occur: it shares the exact
    /// validator ``build_implicit_gemm_conv`` runs internally.
    ///
    /// Acquires the GIL internally; the caller must NOT already hold it
    /// when entering, though holding it (e.g. to build ``payload``) is
    /// safe -- the acquire is reentrant. Any ``py::error_already_set``
    /// is translated via ``PythonError::raise``.
    std::pair<bool, std::string> isApplicable(std::string_view opKind,
                                              const pybind11::dict& payload, std::string_view arch);

    /// Test-only access to the imported compile_service module. Allows
    /// the unit suite to exercise the PythonError translation path by
    /// calling a deliberately-missing attribute. Production callers
    /// should use noopSmoke() or compile().
    pybind11::module_& moduleForTesting() noexcept {
        return _module;
    }

   private:
    /// Prepend a single path to sys.path iff it is not already present.
    /// Caller must hold the GIL. Returns true if the path was newly
    /// inserted, false if it was already on sys.path.
    ///
    /// Idempotence is exact-string-equality only; no normalisation,
    /// no symlink resolution. The CMake-baked literals never change
    /// at runtime, so this is sufficient for the current call sites.
    /// If a future caller passes paths from env/user input, normalise
    /// first.
    ///
    /// Thread safety: not designed for concurrent construction of
    /// multiple bridges; the container's per-process singleton model
    /// makes that irrelevant. If that ever changes, this scan-then-
    /// insert needs an external mutex.
    static bool prependSysPathIdempotent(pybind11::module_& sys, std::string_view path);

    pybind11::module_ _module;
};

}  // namespace ck_dsl_provider
