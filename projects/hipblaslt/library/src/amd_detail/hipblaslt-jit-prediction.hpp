// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <array>
#include <cstdint>
#include <memory>
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

    struct Candidate
    {
        uint32_t                     id              = 0; // stable within one Prediction
        double                       predictedCycles = 0;
        std::vector<TuningParameter> parameters; // forwarded to the backend
        std::vector<TuningParameter> modeled; // recorded, not forwarded
    };

    struct Prediction
    {
        std::string                  modeledContract;
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
    };

    class TuningKnowledge
    {
    public:
        virtual ~TuningKnowledge()                   = default;
        virtual std::string_view           id() const noexcept = 0;
        virtual std::vector<CandidateSeed> seeds(const OperationRequest&,
                                                 const DeviceTarget&) const
            = 0;
        // Values for knobs the model does not predict. Empty leaves them to the backend.
        virtual std::vector<TuningParameter>
            defaults(const OperationRequest&, const DeviceTarget&, const Candidate&) const = 0;
    };

    class Predictor
    {
    public:
        virtual ~Predictor()                                      = default;
        virtual std::string_view id() const noexcept              = 0;
        virtual std::string_view modeledContract() const noexcept = 0;
        virtual Status           predict(const OperationRequest&,
                                         const DeviceTarget&,
                                         const TuningKnowledge&,
                                         Prediction&) const
            = 0;
    };

    std::shared_ptr<const Predictor>       makeOrigamiPredictor();
    std::shared_ptr<const TuningKnowledge> makeCatalogKnowledge();
}
