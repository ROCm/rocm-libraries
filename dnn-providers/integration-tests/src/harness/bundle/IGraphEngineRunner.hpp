// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>

#include "harness/Benchmark.hpp"
#include "harness/VariantPack.hpp"
#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/LoadedEngine.hpp"

namespace hipdnn_integration_tests::bundle
{

struct IntegrationTestBundle;

/// What the backend said to `execute()`, when it is something the harness acts on.
///
/// The frontend folds the backend's status into error text, so a status the harness needs is
/// recovered from that text by the runner that saw it, and carried here as a fact.
struct ExecuteStatus
{
    /// BAD_PARAM for a device pointer below the tensor's declared alignment: the uid of the
    /// tensor the backend rejected (EnginePluginResourceManager::executeOpGraph).
    std::optional<int64_t> alignmentRejectedUid;
};

/// The tensor the backend's alignment rejection names, or nullopt when `message` is not one.
///
/// The backend's text is "Tensor uid <N> device pointer is not aligned to the required <M>
/// bytes"; the frontend may wrap it. Both the uid and the sentence must be present, so an
/// unrelated error that mentions alignment, or one that names another uid, is not taken for it.
inline std::optional<int64_t> parseAlignmentRejectedUid(std::string_view message)
{
    constexpr std::string_view K_PREFIX = "Tensor uid ";
    constexpr std::string_view K_SENTENCE = " device pointer is not aligned to the required";

    const auto sentence = message.find(K_SENTENCE);
    if(sentence == std::string_view::npos)
    {
        return std::nullopt;
    }
    const auto prefix = message.rfind(K_PREFIX, sentence);
    if(prefix == std::string_view::npos)
    {
        return std::nullopt;
    }

    const auto digits
        = message.substr(prefix + K_PREFIX.size(), sentence - prefix - K_PREFIX.size());
    if(digits.empty())
    {
        return std::nullopt;
    }
    const std::size_t start = digits.front() == '-' ? 1 : 0;
    if(start == digits.size())
    {
        return std::nullopt;
    }
    int64_t value = 0;
    for(std::size_t i = start; i < digits.size(); ++i)
    {
        if(digits[i] < '0' || digits[i] > '9')
        {
            return std::nullopt;
        }
        const int64_t digit = digits[i] - '0';
        if(value > (std::numeric_limits<int64_t>::max() - digit) / 10)
        {
            return std::nullopt;
        }
        value = value * 10 + digit;
    }
    return digits.front() == '-' ? -value : value;
}

/// What one frontend operation did.
///
/// Returned rather than asserted so nothing below the harness owns a GTest
/// disposition: only the operation itself can tell an engine failure apart from any
/// other fatal failure in the same test, and only it knows how far the engine got.
struct EngineOpResult
{
    bool ok = false;
    /// The engine refused the graph. Distinct from `!ok`: a decline is the engine
    /// answering "not mine", which skips, while a failure is the engine breaking.
    bool declined = false;
    /// Plan compilation had already succeeded when this result was produced, so the
    /// engine reached BUILDABLE whatever happened next. Without it every engine
    /// error reads as "the engine never took the graph", and a bundle that only has
    /// to compile is denied the depth it actually reached.
    bool plansBuilt = false;
    std::string message;
    /// What execute() answered, when it was a status the harness acts on. Empty for every
    /// other result.
    ExecuteStatus executeStatus;

    static EngineOpResult succeeded()
    {
        EngineOpResult result;
        result.ok = true;
        result.plansBuilt = true;
        return result;
    }

    static EngineOpResult failed(std::string message)
    {
        EngineOpResult result;
        result.message = std::move(message);
        return result;
    }

    static EngineOpResult declinedBy(std::string message)
    {
        EngineOpResult result;
        result.declined = true;
        result.message = std::move(message);
        return result;
    }
};

/// Everything the harness does with a hipdnn_frontend::graph::Graph.
///
/// One seam, not three, because the three things the harness needs — load the
/// graph, compile plans for it, run it — all need the same shared handle and the
/// same device. Behind this interface a unit test needs neither, which is what
/// makes enforceAtLevel() and the whole verification body testable without an
/// engine plugin loaded.
///
/// Implementations MUST return a decline rather than throw
/// EngineNotApplicableError: "this engine will not take the graph" is an answer,
/// not an error.
class IGraphEngineRunner
{
public:
    IGraphEngineRunner() = default;
    virtual ~IGraphEngineRunner() = default;

    IGraphEngineRunner(const IGraphEngineRunner&) = delete;
    IGraphEngineRunner& operator=(const IGraphEngineRunner&) = delete;
    IGraphEngineRunner(IGraphEngineRunner&&) = delete;
    IGraphEngineRunner& operator=(IGraphEngineRunner&&) = delete;

    /// from_binary, plus the single get_ranked_engine_ids() query this test makes.
    /// A build failure comes back as GraphSession::buildFailed, not as a throw.
    virtual GraphSession openGraph(const IntegrationTestBundle& bundle,
                                   const std::optional<LoadedEngine>& engineUnderTest)
        = 0;

    /// create_execution_plans + check_support + build_plans, pinned to
    /// `engineUnderTest` when one was named.
    virtual EngineOpResult buildPlans(GraphSession& session,
                                      const std::optional<LoadedEngine>& engineUnderTest)
        = 0;

    /// buildPlans(), then allocate the workspace and execute into `variantPack`.
    /// Non-const because hipdnn_frontend::graph::Graph::execute() takes it that way.
    virtual EngineOpResult execute(GraphSession& session,
                                   const std::optional<LoadedEngine>& engineUnderTest,
                                   VariantPack& variantPack)
        = 0;

    /// Times the engine of the plan execute() built, on `variantPack`, with
    /// Graph::autotune(). Called only after the outputs of execute() were verified.
    virtual hipdnn_frontend::AutotuneResult benchmark(GraphSession& session,
                                                      VariantPack& variantPack,
                                                      const BenchmarkSettings& settings)
        = 0;
};

} // namespace hipdnn_integration_tests::bundle
