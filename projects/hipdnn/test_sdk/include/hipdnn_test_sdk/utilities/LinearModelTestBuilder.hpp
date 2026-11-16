// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file LinearModelTestBuilder.hpp
 * @brief In-memory linear_data FlatBuffer builder shared by the UHD tests.
 */

#include <hipdnn_flatbuffers_sdk/data_objects/linear_model_generated.h>

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

/// Helper to build a LinearModel FlatBuffer in memory. Writes whatever it is given, so a
/// refusal test can build a malformed model.
class LinearModelTestBuilder
{
public:
    struct Term
    {
        int32_t feature = 0;
        uint8_t transform = 0;
        double mean = 0.0;
        double scale = 1.0;
        double lo = -3.0;
        double hi = 3.0;
    };

    struct Config
    {
        std::vector<double> key;
        std::vector<double> coefficients;
        double floor = 0.0;
    };

    LinearModelTestBuilder& setNumFeatures(int32_t n)
    {
        _numFeatures = n;
        return *this;
    }

    LinearModelTestBuilder& setFeaturesHash(const std::string& hash)
    {
        _featuresHash = hash;
        return *this;
    }

    LinearModelTestBuilder& setKeyFeatures(std::vector<int32_t> slots)
    {
        _keyFeatures = std::move(slots);
        return *this;
    }

    LinearModelTestBuilder& setFamilyFeatures(std::vector<int32_t> slots)
    {
        _familyFeatures = std::move(slots);
        return *this;
    }

    LinearModelTestBuilder& addTerm(Term term)
    {
        _basis.push_back(term);
        return *this;
    }

    LinearModelTestBuilder& addConfig(Config config)
    {
        _configs.push_back(std::move(config));
        return *this;
    }

    LinearModelTestBuilder& addFamily(Config family)
    {
        _families.push_back(std::move(family));
        return *this;
    }

    LinearModelTestBuilder& setGlobal(Config global)
    {
        _global = std::move(global);
        return *this;
    }

    /// A model with no global entry, which the loader refuses.
    LinearModelTestBuilder& omitGlobal()
    {
        _global.reset();
        return *this;
    }

    LinearModelTestBuilder& setTrainingArches(std::vector<std::string> arches)
    {
        _trainingArches = std::move(arches);
        return *this;
    }

    LinearModelTestBuilder& setIdentifier(bool present)
    {
        _identifier = present;
        return *this;
    }

    std::vector<uint8_t> build()
    {
        namespace fb = hipdnn_flatbuffers_sdk::data_objects;
        flatbuffers::FlatBufferBuilder fbb;

        const auto writeConfig = [&](const Config& config) {
            return fb::CreateLinearConfig(fbb,
                                          fbb.CreateVector(config.key),
                                          fbb.CreateVector(config.coefficients),
                                          config.floor);
        };

        std::vector<flatbuffers::Offset<fb::LinearBasisTerm>> terms;
        terms.reserve(_basis.size());
        for(const auto& term : _basis)
        {
            terms.push_back(fb::CreateLinearBasisTerm(
                fbb, term.feature, term.transform, term.mean, term.scale, term.lo, term.hi));
        }
        std::vector<flatbuffers::Offset<fb::LinearConfig>> configs;
        configs.reserve(_configs.size());
        for(const auto& config : _configs)
        {
            configs.push_back(writeConfig(config));
        }
        std::vector<flatbuffers::Offset<fb::LinearConfig>> families;
        families.reserve(_families.size());
        for(const auto& family : _families)
        {
            families.push_back(writeConfig(family));
        }
        const auto global
            = _global ? writeConfig(*_global) : flatbuffers::Offset<fb::LinearConfig>();
        std::vector<flatbuffers::Offset<flatbuffers::String>> arches;
        arches.reserve(_trainingArches.size());
        for(const auto& arch : _trainingArches)
        {
            arches.push_back(fbb.CreateString(arch));
        }

        const auto model = fb::CreateLinearModel(fbb,
                                                 _numFeatures,
                                                 fbb.CreateString(_featuresHash),
                                                 fbb.CreateVector(_keyFeatures),
                                                 fbb.CreateVector(_familyFeatures),
                                                 fbb.CreateVector(terms),
                                                 fbb.CreateVector(configs),
                                                 fbb.CreateVector(families),
                                                 global,
                                                 fbb.CreateString("rellin"),
                                                 fbb.CreateString("2026-01-01T00:00:00Z"),
                                                 100,
                                                 fbb.CreateVector(arches),
                                                 fbb.CreateString("1.0.0"));
        if(_identifier)
        {
            fbb.Finish(model, fb::LinearModelIdentifier());
        }
        else
        {
            fbb.Finish(model);
        }
        return {fbb.GetBufferPointer(), fbb.GetBufferPointer() + fbb.GetSize()};
    }

    /// Build and write to disk, for tests that need a model artifact path.
    bool buildToFile(const std::string& path)
    {
        const auto buffer = build();
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
        return static_cast<bool>(out);
    }

private:
    int32_t _numFeatures = 0;
    std::string _featuresHash;
    std::vector<int32_t> _keyFeatures;
    std::vector<int32_t> _familyFeatures;
    std::vector<Term> _basis;
    std::vector<Config> _configs;
    std::vector<Config> _families;
    std::optional<Config> _global = Config{{}, {0.0}, 0.0};
    std::vector<std::string> _trainingArches;
    bool _identifier = true;
};

} // namespace hipdnn_test_sdk::utilities
