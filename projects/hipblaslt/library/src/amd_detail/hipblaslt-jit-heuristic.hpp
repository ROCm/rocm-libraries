// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <memory>

// What heuristic queries use when HIPBLASLT_JIT is 1 or 2.
namespace hipblaslt_jit
{
    // The generator side of the process's Jit.
    struct ProcessBackend
    {
        std::shared_ptr<const Backend>         backend;
        std::shared_ptr<const Predictor>       predictor; // iff the backend consumes predictions
        std::shared_ptr<const TuningKnowledge> knowledge; // iff predictor
    };

    struct ProcessBackendName
    {
        std::string id; // as HIPBLASLT_JIT_BACKENDS names the backend
        std::string name; // as reports name it
    };

    // Defined by the one provider the build links, which hipBLASLt's CMake
    // configuration selects. A provider that cannot generate returns a
    // Configure failure, which every query reports, and has an empty id.
    Status             makeDefaultProcessBackend(ProcessBackend& made);
    ProcessBackendName defaultProcessBackendName();

    // A backend heuristic queries use only when HIPBLASLT_JIT_BACKENDS names it.
    struct OptInProcessBackend
    {
        ProcessBackendName name;
        Status (*make)(ProcessBackend& made);
    };

    // Defined by the one opt-in provider the build links, which may list none.
    std::vector<OptInProcessBackend> optInProcessBackends();

    struct ProcessJit
    {
        ProcessBackendName         name;
        std::shared_ptr<const Jit> jit; // null when configuration failed
        Status                     configured; // the configuration failure
    };

    // The backends heuristic queries use, in order: those HIPBLASLT_JIT_BACKENDS
    // lists, or when it is unset or empty, the default one. Each is built on
    // first use with the comgr builder and the Tensile loader, publishing to
    // JitLibrary::process(). HIPBLASLT_JIT_BACKENDS names that this build does
    // not have are reported once and ignored.
    const std::vector<ProcessJit>& processJits();

    // The JIT solutions for one heuristic query.
    struct HeuristicFill
    {
        std::vector<int32_t> indices; // JIT library indices, grouped by backend, published first
        std::vector<Status>  failures; // in the order they happened
        // The backend's note when it generated; with several backends, what each returned.
        std::string summary;
        bool        repeated = false; // generation fell short before; not retried
        bool        skipped  = false; // short, and a LookupOnly prevented generation
    };

    // While one is alive, fillHeuristic on this thread returns only published
    // solutions: it starts no generator, build or publication.
    class LookupOnly
    {
    public:
        LookupOnly() noexcept;
        ~LookupOnly();
        LookupOnly(const LookupOnly&)            = delete;
        LookupOnly& operator=(const LookupOnly&) = delete;

        // The innermost one on this thread, or nullptr.
        static LookupOnly* active() noexcept;

        bool skipped = false; // a fillHeuristic call came up short and did not generate

    private:
        LookupOnly* m_outer;
    };

    // Up to count JIT library indices for request on device that need at most
    // workspaceLimit and use none of excludeKernels. Each processJits() backend
    // that accepts the request, in order, gets what is still missing less one
    // slot per later one: solutions it already published for exactly this
    // problem, then new ones it generates and publishes unless a LookupOnly is
    // active. One thread at a time generates a problem with a backend, and a
    // backend whose generation of a problem fell short does not generate it
    // again in this process. With several backends, failures name theirs.
    HeuristicFill fillHeuristic(const OperationRequest&         request,
                                int                             device,
                                size_t                          count,
                                size_t                          workspaceLimit,
                                const std::vector<std::string>& excludeKernels);

    enum class Severity
    {
        Warning,
        Error,
    };

    // Writes "hipblaslt <warning|error>: JIT <message>" to stderr the first time
    // this process reports message, and to the hipBLASLt log every time.
    void report(Severity severity, const std::string& message);

    // "<stage> failed for <problem>: <status.message>", followed by the log path
    // when the message does not name it.
    std::string describe(const Status& status, const std::string& problem);
}
