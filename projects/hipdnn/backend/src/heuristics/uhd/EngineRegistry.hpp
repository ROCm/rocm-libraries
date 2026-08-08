// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "adapters/IUhdAdapter.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn_backend::heuristics::uhd
{

/// @brief Kernel candidate metadata (mock for UKD).
///
/// Represents a single kernel variant with its KMD field values.
/// In full RFC 0017 integration, this maps to UKD::metadata.
struct KernelCandidate
{
    int64_t kernelId;
    int64_t priority = 0;
    std::unordered_map<std::string, double> metadata; // KMD field values
};

/// @brief UHD configuration for an engine (mock for UED-owned UHD).
///
/// Contains the heuristic configuration: features signature, adapter,
/// objective, and score metadata. In full RFC 0017 integration, this
/// is owned by the UED.
struct UhdConfig
{
    std::string uhdId;
    std::string name;
    std::vector<std::string> featuresSignature;
    std::string featuresHash;
    std::string objective = "max"; // "max" or "min"

    // Score metadata for cross-engine comparison (RFC §5, §12.3)
    std::string scoreUnits;           // e.g., "tflops", "ms"
    bool scoreCalibrated = false;     // cross-engine comparable?
    std::string scoreTransform;       // e.g., "log1p", "identity"

    // Adapter configuration
    std::string adapterType = "static_order"; // "static_order", "tree_data", etc.
    std::string modelArtifactPath;            // for tree_data/onnx/custom_library
    std::vector<std::string> staticOrderFields = {"priority", "id"}; // for static_order
};

/// @brief Engine registration entry (mock for UED).
///
/// Represents an engine with its UHD configuration and registered kernels.
/// In full RFC 0017 integration, this maps to the UED which owns the UHD
/// and KMD, with KDPs joining the engine and contributing child UKDs.
struct EngineEntry
{
    int64_t engineId;
    std::string engineName;
    UhdConfig uhdConfig;
    std::vector<KernelCandidate> candidates;

    // Loaded adapter (cached after first use)
    mutable std::shared_ptr<IUhdAdapter> cachedAdapter;
};

/// @brief Mock engine registry for UHD selection testing.
///
/// This registry provides a mock implementation of the UED/UKD lookup
/// that RFC 0017 will provide. It allows testing the complete UHD
/// selection flow without the full descriptor system.
///
/// Usage:
/// 1. Register engines with their UHD configs and kernel candidates
/// 2. UhdBuiltIn queries the registry during policyFinalize()
/// 3. The registry provides the adapter and candidate metadata for scoring
///
/// When RFC 0017 lands, this class will be replaced by real UED/UKD lookup.
class EngineRegistry
{
public:
    /// Get the singleton registry instance.
    static EngineRegistry& instance();

    /// Register an engine with its UHD configuration.
    ///
    /// Enforces the RFC 0019 §7.3 load-time contract, fail-closed: every $kernel.*
    /// reference in features_signature must be a known KMD field, and a declared
    /// features_hash must describe that signature.
    ///
    /// @param entry Complete engine entry including UHD config and candidates.
    /// @throws std::invalid_argument if either contract check fails.
    void registerEngine(EngineEntry entry);

    /// Look up an engine by ID.
    /// @returns Engine entry or nullopt if not found.
    std::optional<std::reference_wrapper<const EngineEntry>> getEngine(int64_t engineId) const;

    /// Get or create the adapter for an engine.
    /// @param engineId Engine to get adapter for.
    /// @returns Adapter or nullptr if engine not found or adapter creation fails.
    std::shared_ptr<IUhdAdapter> getOrCreateAdapter(int64_t engineId) const;

    /// Check if an engine is registered.
    bool hasEngine(int64_t engineId) const;

    /// Get all registered engine IDs.
    std::vector<int64_t> getAllEngineIds() const;

    /// Clear all registrations (for testing).
    void clear();

    /// Get the number of registered engines.
    size_t size() const { return _engines.size(); }

    EngineRegistry(const EngineRegistry&) = delete;
    EngineRegistry& operator=(const EngineRegistry&) = delete;

private:
    EngineRegistry() = default;

    /// Check a declared features_hash against its features_signature, and warn when a
    /// feature-bearing adapter ships without one.
    /// @throws std::invalid_argument on mismatch.
    static void validateFeaturesHash(const EngineEntry& entry);

    std::unordered_map<int64_t, EngineEntry> _engines;
    mutable std::mutex _mutex;
};

} // namespace hipdnn_backend::heuristics::uhd
