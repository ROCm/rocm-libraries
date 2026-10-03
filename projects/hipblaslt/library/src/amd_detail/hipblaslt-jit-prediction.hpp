// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

// Ranked candidates for the backends that consume a prediction.
namespace hipblaslt_jit
{
    // A named tuning parameter whose value is a JSON literal in the backend's vocabulary.
    struct TuningParameter
    {
        std::string name;
        std::string json;
    };

    // How a kernel covers the output tiles: an ordinary grid, or persistent
    // workgroups that process whole tiles or split them along K.
    struct ExecutionPolicy
    {
        enum class Strategy
        {
            None,
            DataParallel,
            StreamK,
        };
        enum class Assignment
        {
            StaticGrid,
            DynamicWorkQueue,
            Hybrid,
        };
        Strategy   strategy   = Strategy::None;
        Assignment assignment = Assignment::StaticGrid;
    };

    struct Candidate
    {
        uint32_t                     id              = 0; // stable within one Prediction
        double                       predictedCycles = 0;
        std::vector<TuningParameter> parameters; // forwarded to the backend
        std::vector<TuningParameter> modeled; // recorded, not forwarded
        std::string                  contract; // its modeled contract; empty: the prediction's
        int32_t                      seed = -1; // index of the seed it came from, or -1
        std::string                  provenance; // JSON the backend records, or empty
    };

    struct Prediction
    {
        std::string                  modeledContract; // of candidates that name none
        std::string                  model;
        std::vector<TuningParameter> hardware; // device facts the model used
        std::vector<TuningParameter> assumptions;
        std::vector<Candidate>       ranked; // best first
    };

    // A family of candidates for a predictor to rank.
    struct CandidateSeed
    {
        std::array<size_t, 4> tile{}; // macro tile M and N, waves M and N
        // Each {minimum, multiple} rule gives DepthU = max(minimum, multiple * instruction K).
        std::vector<std::array<size_t, 2>> depthRules;
        std::vector<std::array<int, 2>>    cacheHints; // {NonTemporalA, NonTemporalB}
        // A fixed seed names its instruction {M, N, K, blocks} and DepthU
        // instead of rules, and is one candidate.
        std::optional<std::array<size_t, 4>> instruction;
        size_t                               depthU = 0;
        std::vector<TuningParameter>         parameters; // forwarded verbatim
        std::string                          provenance; // JSON its candidates record
        size_t rank = 0; // fixed seeds go best rank first; the predictor orders equal ranks
        // The policies to expand; a fixed seed has exactly one.
        std::vector<ExecutionPolicy> policies;
    };

    class TuningKnowledge
    {
    public:
        virtual ~TuningKnowledge()                   = default;
        virtual std::string_view           id() const noexcept = 0;
        // Changes whenever seeds can change for the same request and target.
        virtual std::string                version() const = 0;
        virtual std::vector<CandidateSeed> seeds(const OperationRequest&,
                                                 const DeviceTarget&) const
            = 0;
        // Values for knobs the model does not predict. Empty leaves them to the backend.
        virtual std::vector<TuningParameter>
            defaults(const OperationRequest&, const DeviceTarget&, const Candidate&) const = 0;
    };

    struct PredictionRequest
    {
        const OperationRequest& request;
        const DeviceTarget&     target;
        size_t                  workspaceLimit = 0;
    };

    class Predictor
    {
    public:
        virtual ~Predictor()                                       = default;
        virtual std::string_view      id() const noexcept          = 0;
        virtual std::set<std::string> modeledContracts() const     = 0;
        virtual Status                predict(const PredictionRequest&,
                                              const TuningKnowledge&,
                                              Prediction&) const
            = 0;
    };

    std::shared_ptr<const Predictor>       makeOrigamiPredictor();
    std::shared_ptr<const TuningKnowledge> makeCatalogKnowledge();
}
