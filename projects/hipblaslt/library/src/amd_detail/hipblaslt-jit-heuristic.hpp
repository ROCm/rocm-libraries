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

    // Defined by the one provider the build links, which hipBLASLt's CMake
    // configuration selects. A provider that cannot generate returns a
    // Configure failure; processJit reports it for every query.
    Status makeDefaultProcessBackend(ProcessBackend& made);

    // The Jit for this process, built on first use from
    // makeDefaultProcessBackend, the comgr builder and the Tensile loader,
    // publishing to JitLibrary::process(). Null, with why set, when the
    // provider fails.
    std::shared_ptr<const Jit> processJit(Status& why);

    // The JIT solutions for one heuristic query.
    struct HeuristicFill
    {
        std::vector<int32_t> indices; // JIT library indices, published ones first
        std::vector<Status>  failures; // in the order they happened
        std::string          summary; // the backend's note, when it generated
        bool                 repeated = false; // generation fell short before; not retried
        bool                 skipped  = false; // short, and a LookupOnly prevented generation
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
    // workspaceLimit and use none of excludeKernels: solutions already published
    // for exactly this problem, then new ones processJit() generates and
    // publishes unless a LookupOnly is active. One thread at a time generates a
    // problem, and a problem whose generation fell short is not generated again
    // in this process.
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
