// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include "IUhdAdapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <hipdnn_data_sdk/logging/Logger.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/linear_model_generated.h>
#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/Sha256.hpp>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_plugin_sdk::uhd
{

/// @brief Per-configuration linear ranker (RFC 0019 §7 "linear_data").
///
/// A within-catalog ranker: a candidate's score is its predicted log slowdown against the best
/// candidate for the same problem (linear_model.fbs), so LOWER IS BETTER and the UHD declares
/// `objective: min`. The score is not a physical estimate: the adapter serves no `metric` and no
/// calibrated role.
///
/// score(row) = max(c . [1, z], floor), where z are the clamped, standardized basis terms and
/// (c, floor) belong to the config whose key matches the row's key features, else the family
/// whose key matches its family features, else the global entry. A row costs one hash lookup and
/// one dot product; scoreBatch transforms a basis value only when it differs from the row
/// before, so a catalog's shared problem features are transformed once.
class LinearDataAdapter : public IUhdAdapter
{
public:
    /// Load a linear model from a FlatBuffer file.
    /// @param expectedModelHash Artifact SHA-256 hex; empty skips the check.
    /// @returns nullptr if loading or validation fails.
    static std::unique_ptr<LinearDataAdapter> load(const std::string& modelPath,
                                                   const std::string& expectedFeaturesHash,
                                                   const std::string& expectedModelHash = "");

    /// Load from an in-memory buffer; everything the model needs is copied out of it.
    /// @param expectedModelHash Artifact SHA-256 hex; empty skips the check.
    /// @returns nullptr if validation fails.
    static std::unique_ptr<LinearDataAdapter>
        loadFromBuffer(const uint8_t* buffer,
                       size_t size,
                       const std::string& expectedFeaturesHash,
                       const std::string& expectedModelHash = "");

    ~LinearDataAdapter() override = default;

    LinearDataAdapter(const LinearDataAdapter&) = delete;
    LinearDataAdapter& operator=(const LinearDataAdapter&) = delete;

    /// @returns The predicted log slowdown, or -infinity (declined) if @p features does not
    /// have the model's width.
    double score(const std::vector<double>& features) const override;

    BatchScores scoreBatch(const FeatureRows& rows) const override;

    size_t expectedFeatureCount() const override
    {
        return _numFeatures;
    }

    const std::string& getFeaturesHash() const override
    {
        return _featuresHash;
    }

    bool isTrainedForArch(const std::string& arch) const override;

private:
    /// One basis term, flattened from linear_model.fbs.
    struct Term
    {
        size_t feature;
        bool log1p;
        double mean;
        double scale;
        double lo;
        double hi;
    };

    /// Hash set of fixed-width double tuples (open addressing, load <= 1/2), keyed on feature
    /// row slots. Keys are compared as doubles with every NaN equal, so a knob that is absent
    /// matches another absent one. find() reads the key straight out of the row: no tuple is
    /// built and nothing is allocated.
    class KeyTable
    {
    public:
        /// @param keys n * slots.size() values, row-major.
        /// @returns The index of the first duplicate key, or -1 if all are unique.
        int64_t build(std::vector<uint32_t> slots, const std::vector<double>& keys, size_t count);

        /// @returns The index of the key equal to @p row at the table's slots, or -1.
        int32_t find(const double* row) const;

    private:
        static double canonical(double value);
        static uint64_t bitsOf(double value);
        uint64_t hashOf(const double* values, bool gathered, const double* row) const;

        std::vector<uint32_t> _slots;
        std::vector<double> _keys; // canonical values, row-major
        std::vector<int32_t> _table; // -1 empty, else a key index
        size_t _mask = 0;
    };

    LinearDataAdapter() = default;

    /// Checks what the verifier cannot (linear_model.fbs) and builds the lookup structures.
    /// @returns Why @p model is unusable, or empty with @p adapter filled.
    static std::string build(const hipdnn_flatbuffers_sdk::data_objects::LinearModel& model,
                             LinearDataAdapter& adapter);

    /// The entry scoring @p row: its config, else its family, else the global entry.
    size_t entryOf(const double* row) const;

    /// Basis term @p term's model input for value @p x: the clamped standardized transform,
    /// or NaN when @p x is missing, in which case the term adds nothing (the training mean).
    static double termInput(const Term& term, double x);

    /// `entry`'s score for the basis inputs @p inputs (one per term), floored.
    double scoreInputs(size_t entry, const double* inputs) const;

    std::string _featuresHash;
    size_t _numFeatures = 0;
    std::vector<std::string> _trainingArches;

    std::vector<Term> _basis;
    KeyTable _configs;
    KeyTable _families; // empty when the model has no family key

    /// One row of 1 + _basis.size() values per entry, then one floor per entry: the configs,
    /// then the families, then the global entry.
    std::vector<double> _coefficients;
    std::vector<double> _floors;
    size_t _familyBase = 0;
    size_t _globalEntry = 0;
};

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

inline double LinearDataAdapter::KeyTable::canonical(double value)
{
    if(std::isnan(value))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    // -0.0 == 0.0 as a double, so both must hash and compare alike.
    return value == 0.0 ? 0.0 : value;
}

inline uint64_t LinearDataAdapter::KeyTable::bitsOf(double value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline uint64_t LinearDataAdapter::KeyTable::hashOf(const double* values,
                                                    bool gathered,
                                                    const double* row) const
{
    // The values of one key, either contiguous (a stored key) or gathered from a row.
    uint64_t hash = 0x9e3779b97f4a7c15ULL;
    for(size_t i = 0; i < _slots.size(); ++i)
    {
        const double value = gathered ? row[_slots[i]] : values[i];
        hash ^= bitsOf(canonical(value)) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
        hash *= 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32;
    }
    return hash;
}

inline int64_t LinearDataAdapter::KeyTable::build(std::vector<uint32_t> slots,
                                                  const std::vector<double>& keys,
                                                  size_t count)
{
    _slots = std::move(slots);
    const size_t width = _slots.size();
    _keys.resize(count * width);
    for(size_t i = 0; i < _keys.size(); ++i)
    {
        _keys[i] = canonical(keys[i]);
    }

    size_t capacity = 2;
    while(capacity < 2 * count)
    {
        capacity *= 2;
    }
    _mask = capacity - 1;
    _table.assign(capacity, -1);

    for(size_t index = 0; index < count; ++index)
    {
        const double* key = _keys.data() + (index * width);
        size_t position = hashOf(key, false, nullptr) & _mask;
        while(_table[position] >= 0)
        {
            const double* other = _keys.data() + (static_cast<size_t>(_table[position]) * width);
            if(width == 0 || std::memcmp(other, key, width * sizeof(double)) == 0)
            {
                return static_cast<int64_t>(index);
            }
            position = (position + 1) & _mask;
        }
        _table[position] = static_cast<int32_t>(index);
    }
    return -1;
}

inline int32_t LinearDataAdapter::KeyTable::find(const double* row) const
{
    if(_table.empty())
    {
        return -1;
    }

    const size_t width = _slots.size();
    size_t position = hashOf(nullptr, true, row) & _mask;
    while(_table[position] >= 0)
    {
        const double* key = _keys.data() + (static_cast<size_t>(_table[position]) * width);
        size_t i = 0;
        // Stored keys are canonical, so equal doubles are equal bit patterns once the row's
        // value is canonical too.
        while(i < width && bitsOf(canonical(row[_slots[i]])) == bitsOf(key[i]))
        {
            ++i;
        }
        if(i == width)
        {
            return _table[position];
        }
        position = (position + 1) & _mask;
    }
    return -1;
}

inline std::unique_ptr<LinearDataAdapter>
    LinearDataAdapter::load(const std::string& modelPath,
                            const std::string& expectedFeaturesHash,
                            const std::string& expectedModelHash)
{
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if(!file)
    {
        return nullptr;
    }

    auto size = file.tellg();
    if(size <= 0 || size > static_cast<std::streamoff>(256 * 1024 * 1024))
    {
        return nullptr;
    }

    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    file.seekg(0);
    if(!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        return nullptr;
    }

    return loadFromBuffer(buffer.data(), buffer.size(), expectedFeaturesHash, expectedModelHash);
}

inline std::unique_ptr<LinearDataAdapter>
    LinearDataAdapter::loadFromBuffer(const uint8_t* buffer,
                                      size_t size,
                                      const std::string& expectedFeaturesHash,
                                      const std::string& expectedModelHash)
{
    if(buffer == nullptr || size < sizeof(flatbuffers::uoffset_t) + 4
       || size > size_t{256} * 1024 * 1024)
    {
        return nullptr;
    }

    // RFC 0019 §9.2: the digest identifies the model to persistent caches, so mismatched
    // bytes must not be scored under it.
    if(!expectedModelHash.empty())
    {
        const std::string actualHash = sha256(buffer, size);
        if(actualHash != expectedModelHash)
        {
            HIPDNN_SDK_LOG_ERROR(
                "LinearDataAdapter: model hash mismatch - expected='"
                << expectedModelHash << "' actual='" << actualHash
                << "'; the model is not used -- ranking degrades to static_order and an "
                   "engine estimate is reported as 0");
            return nullptr;
        }
    }

    if(!flatbuffers::BufferHasIdentifier(buffer, fb::LinearModelIdentifier()))
    {
        return nullptr;
    }

    flatbuffers::Verifier verifier(buffer, size);
    if(!fb::VerifyLinearModelBuffer(verifier))
    {
        return nullptr;
    }

    const auto* model = fb::GetLinearModel(buffer);
    if(model == nullptr)
    {
        return nullptr;
    }

    // RFC 0019 §6.3 check 3. ERROR, not WARN (§12), worded like the other adapters so one
    // log search finds all of them.
    const std::string modelHash
        = model->features_hash() != nullptr ? model->features_hash()->str() : "";
    if(!expectedFeaturesHash.empty() && modelHash != expectedFeaturesHash)
    {
        HIPDNN_SDK_LOG_ERROR("LinearDataAdapter: features hash mismatch - expected='"
                             << expectedFeaturesHash << "' actual='" << modelHash
                             << "'; the model is not used -- ranking degrades to static_order "
                                "and an engine estimate is reported as 0");
        return nullptr;
    }

    std::unique_ptr<LinearDataAdapter> adapter(new LinearDataAdapter());
    adapter->_featuresHash = modelHash;
    if(model->training_arches() != nullptr)
    {
        for(const auto* arch : *model->training_arches())
        {
            if(arch != nullptr)
            {
                adapter->_trainingArches.emplace_back(arch->str());
            }
        }
    }

    const auto reason = build(*model, *adapter);
    if(!reason.empty())
    {
        HIPDNN_SDK_LOG_ERROR("LinearDataAdapter: " << reason
                                                   << "; the model is not used -- ranking degrades "
                                                      "to static_order and an engine estimate is "
                                                      "reported as 0");
        return nullptr;
    }

    return adapter;
}

namespace linear_detail
{

/// Copy @p slots out of @p vector, each within [0, numFeatures).
/// @returns Why they are unusable, or empty.
inline std::string readSlots(const flatbuffers::Vector<int32_t>* vector,
                             const char* what,
                             int32_t numFeatures,
                             std::vector<uint32_t>& slots)
{
    if(vector == nullptr)
    {
        return {};
    }
    for(flatbuffers::uoffset_t i = 0; i < vector->size(); ++i)
    {
        const int32_t slot = vector->Get(i);
        if(slot < 0 || slot >= numFeatures)
        {
            return std::string(what) + " " + std::to_string(i) + " reads feature "
                   + std::to_string(slot) + " of a model with " + std::to_string(numFeatures);
        }
        slots.push_back(static_cast<uint32_t>(slot));
    }
    return {};
}

} // namespace linear_detail

inline std::string LinearDataAdapter::build(const fb::LinearModel& model,
                                            LinearDataAdapter& adapter)
{
    if(model.num_features() < 0)
    {
        return "the model declares " + std::to_string(model.num_features()) + " features";
    }
    const int32_t numFeatures = model.num_features();
    adapter._numFeatures = static_cast<size_t>(numFeatures);

    std::vector<uint32_t> keySlots;
    std::vector<uint32_t> familySlots;
    auto reason = linear_detail::readSlots(
        model.key_features(), "key_features entry", numFeatures, keySlots);
    if(reason.empty())
    {
        reason = linear_detail::readSlots(
            model.family_features(), "family_features entry", numFeatures, familySlots);
    }
    if(!reason.empty())
    {
        return reason;
    }

    if(model.basis() != nullptr)
    {
        adapter._basis.reserve(model.basis()->size());
        for(flatbuffers::uoffset_t i = 0; i < model.basis()->size(); ++i)
        {
            const auto* term = model.basis()->Get(i);
            const std::string name = "basis term " + std::to_string(i);
            if(term == nullptr)
            {
                return name + " is missing";
            }
            if(term->feature() < 0 || term->feature() >= numFeatures)
            {
                return name + " reads feature " + std::to_string(term->feature())
                       + " of a model with " + std::to_string(numFeatures);
            }
            if(term->transform() > 1)
            {
                return name + " has unknown transform " + std::to_string(term->transform());
            }
            if(!std::isfinite(term->mean()) || !std::isfinite(term->scale())
               || !std::isfinite(term->lo()) || !std::isfinite(term->hi()))
            {
                return name + " has a mean, scale, lo or hi that is not finite";
            }
            if(!(term->scale() > 0.0))
            {
                return name + "'s scale is not positive";
            }
            if(term->lo() > term->hi())
            {
                return name + "'s lo is above its hi";
            }
            adapter._basis.push_back({static_cast<size_t>(term->feature()),
                                      term->transform() == 1,
                                      term->mean(),
                                      term->scale(),
                                      term->lo(),
                                      term->hi()});
        }
    }

    const size_t stride = 1 + adapter._basis.size();

    // Appends @p entry's coefficients and floor; its key (when @p keyWidth > 0) goes to @p keys.
    const auto addEntry = [&](const fb::LinearConfig* entry,
                              const std::string& name,
                              size_t keyWidth,
                              std::vector<double>& keys) -> std::string {
        if(entry == nullptr)
        {
            return name + " is missing";
        }
        const size_t keyCount = entry->key() != nullptr ? entry->key()->size() : 0;
        if(keyCount != keyWidth)
        {
            return name + "'s key has " + std::to_string(keyCount) + " values, expected "
                   + std::to_string(keyWidth);
        }
        const size_t coefficientCount
            = entry->coefficients() != nullptr ? entry->coefficients()->size() : 0;
        if(coefficientCount != stride)
        {
            return name + " has " + std::to_string(coefficientCount) + " coefficients, expected "
                   + std::to_string(stride) + " (an intercept and one per basis term)";
        }
        for(flatbuffers::uoffset_t i = 0; i < coefficientCount; ++i)
        {
            if(!std::isfinite(entry->coefficients()->Get(i)))
            {
                return name + "'s coefficient " + std::to_string(i) + " is not finite";
            }
        }
        if(!std::isfinite(entry->floor()))
        {
            return name + "'s floor is not finite";
        }

        for(size_t i = 0; i < keyWidth; ++i)
        {
            keys.push_back(entry->key()->Get(static_cast<flatbuffers::uoffset_t>(i)));
        }
        adapter._coefficients.insert(adapter._coefficients.end(),
                                     entry->coefficients()->begin(),
                                     entry->coefficients()->end());
        adapter._floors.push_back(entry->floor());
        return {};
    };

    const size_t configCount = model.configs() != nullptr ? model.configs()->size() : 0;
    const size_t familyCount = model.families() != nullptr ? model.families()->size() : 0;
    if(familySlots.empty() && familyCount > 0)
    {
        return "the model has families but no family_features to key them";
    }

    adapter._coefficients.reserve((configCount + familyCount + 1) * stride);
    adapter._floors.reserve(configCount + familyCount + 1);

    std::vector<double> configKeys;
    configKeys.reserve(configCount * keySlots.size());
    for(size_t i = 0; i < configCount; ++i)
    {
        reason = addEntry(model.configs()->Get(static_cast<flatbuffers::uoffset_t>(i)),
                          "config " + std::to_string(i),
                          keySlots.size(),
                          configKeys);
        if(!reason.empty())
        {
            return reason;
        }
    }
    adapter._familyBase = configCount;

    std::vector<double> familyKeys;
    familyKeys.reserve(familyCount * familySlots.size());
    for(size_t i = 0; i < familyCount; ++i)
    {
        reason = addEntry(model.families()->Get(static_cast<flatbuffers::uoffset_t>(i)),
                          "family " + std::to_string(i),
                          familySlots.size(),
                          familyKeys);
        if(!reason.empty())
        {
            return reason;
        }
    }

    std::vector<double> noKey;
    reason = addEntry(model.global(), "the global entry", 0, noKey);
    if(!reason.empty())
    {
        return reason;
    }
    adapter._globalEntry = configCount + familyCount;

    if(const auto duplicate = adapter._configs.build(std::move(keySlots), configKeys, configCount);
       duplicate >= 0)
    {
        return "config " + std::to_string(duplicate) + " repeats an earlier config's key";
    }
    if(const auto duplicate
       = adapter._families.build(std::move(familySlots), familyKeys, familyCount);
       duplicate >= 0)
    {
        return "family " + std::to_string(duplicate) + " repeats an earlier family's key";
    }

    return {};
}

inline size_t LinearDataAdapter::entryOf(const double* row) const
{
    if(const int32_t config = _configs.find(row); config >= 0)
    {
        return static_cast<size_t>(config);
    }
    if(const int32_t family = _families.find(row); family >= 0)
    {
        return _familyBase + static_cast<size_t>(family);
    }
    return _globalEntry;
}

inline double LinearDataAdapter::termInput(const Term& term, double x)
{
    if(std::isnan(x))
    {
        return x;
    }
    const double t = term.log1p ? std::log1p(std::max(x, 0.0)) : x;
    return std::clamp((t - term.mean) / term.scale, term.lo, term.hi);
}

inline double LinearDataAdapter::scoreInputs(size_t entry, const double* inputs) const
{
    const double* coefficient = _coefficients.data() + (entry * (1 + _basis.size()));
    double sum = coefficient[0];
    for(size_t j = 0; j < _basis.size(); ++j)
    {
        // A missing feature stands at the training mean: it adds nothing.
        if(!std::isnan(inputs[j]))
        {
            sum += coefficient[1 + j] * inputs[j];
        }
    }
    return std::max(sum, _floors[entry]);
}

inline double LinearDataAdapter::score(const std::vector<double>& features) const
{
    if(features.size() != _numFeatures)
    {
        return -std::numeric_limits<double>::infinity();
    }
    std::vector<double> inputs(_basis.size());
    for(size_t j = 0; j < _basis.size(); ++j)
    {
        inputs[j] = termInput(_basis[j], features[_basis[j].feature]);
    }
    return scoreInputs(entryOf(features.data()), inputs.data());
}

inline BatchScores LinearDataAdapter::scoreBatch(const FeatureRows& rows) const
{
    BatchScores results;
    results.raw.assign(rows.count, -std::numeric_limits<double>::infinity());
    if(rows.width != _numFeatures)
    {
        return results;
    }
    // A catalog's rows share every problem feature, so a term whose value repeats the row
    // before keeps its input: the transform (often a log1p) runs once per distinct value.
    std::vector<double> inputs(_basis.size());
    for(size_t i = 0; i < rows.count; ++i)
    {
        const double* row = rows.row(i);
        const double* previous = i == 0 ? nullptr : rows.row(i - 1);
        for(size_t j = 0; j < _basis.size(); ++j)
        {
            const size_t feature = _basis[j].feature;
            if(previous == nullptr || !(previous[feature] == row[feature]))
            {
                inputs[j] = termInput(_basis[j], row[feature]);
            }
        }
        results.raw[i] = scoreInputs(entryOf(row), inputs.data());
    }
    return results;
}

inline bool LinearDataAdapter::isTrainedForArch(const std::string& arch) const
{
    if(_trainingArches.empty())
    {
        return true;
    }

    const auto target = stripArchFeatures(arch);
    return std::find(_trainingArches.begin(), _trainingArches.end(), target)
           != _trainingArches.end();
}

} // namespace hipdnn_plugin_sdk::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
