// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "FeatureExtractor.hpp"
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

    // RFC 0019 §6.4: Derived values (ordered map of name → JsonLogic expression)
    std::vector<std::pair<std::string, std::string>> derived; // name, expression (JSON string)

    std::vector<std::string> featuresSignature;
    std::string featuresHash;
    std::string objective = "max"; // "max" or "min"

    // Score metadata for cross-engine comparison (RFC §5, §12.3)
    std::string scoreUnits; // e.g., "tflops", "ms"
    bool scoreCalibrated = false; // cross-engine comparable?
    std::string scoreTransform; // e.g., "log1p", "identity"

    // Adapter configuration
    std::string adapterType = "static_order"; // "static_order", "tree_data", etc.
    std::string modelArtifactPath; // for tree_data/onnx/custom_library
    std::string modelHash; // checksum of model artifact for integrity validation
    std::vector<std::string> staticOrderFields = {"priority", "id"}; // for static_order
    std::string customLibrarySymbol; // for custom_library: symbol name in .so
};

/// @brief Engine registration entry (mock for UED).
///
/// Represents an engine with its UHD configuration and registered kernels.
/// In full RFC 0017 integration, this maps to the UED which owns the UHD
/// and KMD, with KDPs joining the engine and contributing child UKDs.
///
/// RFC 0019 §3.1: An engine names up to three role-scoped UHDs, each mapped
/// by architecture (gcnArchName). Resolution tries exact arch, then "default",
/// then nullopt.
struct EngineEntry
{
    int64_t engineId;
    std::string engineName;

    // RFC 0019 §3.1: Three role-scoped, arch-keyed UHDs
    std::unordered_map<std::string, UhdConfig> sortKernelCatalog; // main: ranks catalog
    std::unordered_map<std::string, UhdConfig> predictEngineTflops; // optional: engine estimate
    std::unordered_map<std::string, UhdConfig> predictApplicableKernels; // future: JIT candidate gen

    std::vector<KernelCandidate> candidates;

    // Loaded adapters (cached after first use, keyed by role + arch)
    mutable std::unordered_map<std::string, std::shared_ptr<IUhdAdapter>> cachedAdapters;

    // Cached feature extractors (keyed by role + arch)
    mutable std::unordered_map<std::string, std::shared_ptr<FeatureExtractor>> cachedExtractors;

    /// Resolve UHD by arch (RFC 0019 §8.3).
    /// Tries exact arch match, then "default", then returns nullopt.
    std::optional<UhdConfig> resolveUhd(const std::unordered_map<std::string, UhdConfig>& roleMap,
                                         const std::string& arch) const;

    /// Helper: resolve sort_kernel_catalog UHD (the main one).
    std::optional<UhdConfig> resolveSortKernelCatalog(const std::string& arch) const
    {
        return resolveUhd(sortKernelCatalog, arch);
    }

    /// Helper: resolve predict_engine_tflops UHD (optional).
    std::optional<UhdConfig> resolvePredictEngineTflops(const std::string& arch) const
    {
        return resolveUhd(predictEngineTflops, arch);
    }

    /// Helper: resolve predict_applicable_kernels UHD (future).
    std::optional<UhdConfig> resolvePredictApplicableKernels(const std::string& arch) const
    {
        return resolveUhd(predictApplicableKernels, arch);
    }

    // Backward compatibility: legacy single-UHD interface (deprecated)
    // Maps to sortKernelCatalog["default"] for existing code
    UhdConfig uhdConfig; // DEPRECATED: use sortKernelCatalog["default"] instead
    mutable std::shared_ptr<IUhdAdapter> cachedAdapter; // DEPRECATED
    mutable std::shared_ptr<FeatureExtractor> cachedExtractor; // DEPRECATED
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
    ///
    /// Returns a shared snapshot rather than a reference into the map. RFC 0019 §9.2
    /// makes descriptor replacement a supported operation ("drop a new engine
    /// descriptor set alongside the existing one... rollback restores the previous"),
    /// so a re-registration can land while a selection is in flight. A reference into
    /// the map would be assigned over or destroyed under the reader; holding a
    /// shared_ptr keeps the snapshot the selection started with alive until it
    /// finishes, and the next lookup picks up the new one.
    ///
    /// A snapshot is only self-consistent if everything derived from it is also
    /// resolved from it — use the snapshot overloads of getOrCreateAdapter and
    /// getOrCreateExtractor, not the by-ID ones, for the duration of a selection.
    ///
    /// @returns Engine entry snapshot, or nullptr if not found.
    std::shared_ptr<const EngineEntry> getEngine(int64_t engineId) const;

    /// Get or create the adapter for a specific UHD config (role + arch).
    ///
    /// Prefer this over the by-ID overload inside a selection. Resolving by ID reaches
    /// back into the live map, so a re-registration landing mid-selection would pair a
    /// new model with the old snapshot's config and candidates — silently, since the
    /// two disagree on things like `objective` and `score.transform`.
    ///
    /// @param entry Snapshot from getEngine().
    /// @param cfg UHD config (from resolveUhd).
    /// @param role UHD role ("sort_kernel_catalog", etc.).
    /// @param arch Architecture key ("gfx950", "default", etc.).
    /// @returns Adapter or nullptr if entry is null or adapter creation fails.
    std::shared_ptr<IUhdAdapter>
        getOrCreateAdapter(const std::shared_ptr<const EngineEntry>& entry,
                          const UhdConfig& cfg,
                          const std::string& role,
                          const std::string& arch) const;

    /// Get or create the feature extractor for a specific UHD config (role + arch).
    /// @param entry Snapshot from getEngine().
    /// @param cfg UHD config (from resolveUhd).
    /// @param role UHD role.
    /// @param arch Architecture key.
    /// @returns Extractor or nullptr if entry is null or its signature is empty.
    std::shared_ptr<FeatureExtractor>
        getOrCreateExtractor(const std::shared_ptr<const EngineEntry>& entry,
                            const UhdConfig& cfg,
                            const std::string& role,
                            const std::string& arch) const;

    /// Legacy: Get or create the adapter for an engine's default UHD.
    /// @param entry Snapshot from getEngine().
    /// @returns Adapter or nullptr if entry is null or adapter creation fails.
    /// @deprecated Use the (entry, cfg, role, arch) overload for multi-role support.
    std::shared_ptr<IUhdAdapter>
        getOrCreateAdapter(const std::shared_ptr<const EngineEntry>& entry) const;

    /// Legacy: Get or create the adapter for an engine, resolved by ID against the live map.
    /// @param engineId Engine to get adapter for.
    /// @returns Adapter or nullptr if engine not found or adapter creation fails.
    /// @deprecated Use the (entry, cfg, role, arch) overload for multi-role support.
    std::shared_ptr<IUhdAdapter> getOrCreateAdapter(int64_t engineId) const;

    /// Legacy: Get or create the feature extractor for an engine, resolved by ID.
    /// @param engineId Engine to get extractor for.
    /// @returns Extractor or nullptr if engine not found or signature is empty.
    /// @deprecated Use the (entry, cfg, role, arch) overload for multi-role support.
    std::shared_ptr<FeatureExtractor> getOrCreateExtractor(int64_t engineId) const;

    /// Check if an engine is registered.
    bool hasEngine(int64_t engineId) const;

    /// Get all registered engine IDs.
    std::vector<int64_t> getAllEngineIds() const;

    /// Clear all registrations (for testing).
    void clear();

    /// Get the number of registered engines.
    size_t size() const;

    EngineRegistry(const EngineRegistry&) = delete;
    EngineRegistry& operator=(const EngineRegistry&) = delete;

private:
    EngineRegistry() = default;

    /// Build cache key for adapter/extractor storage: "role:arch"
    static std::string makeCacheKey(const std::string& role, const std::string& arch)
    {
        return role + ":" + arch;
    }

    /// Check a declared features_hash against its features_signature, and warn when a
    /// feature-bearing adapter ships without one.
    /// @throws std::invalid_argument on mismatch.
    static void validateFeaturesHash(const UhdConfig& cfg,
                                      int64_t engineId,
                                      const std::string& role,
                                      const std::string& arch);

    /// Check that a declared score.transform is one this runtime can invert.
    /// @throws std::invalid_argument on an unsupported transform name.
    static void validateScoreTransform(const UhdConfig& cfg,
                                        int64_t engineId,
                                        const std::string& role,
                                        const std::string& arch);

    /// Check that a declared objective is one of the two RFC 0019 §5 values.
    /// @throws std::invalid_argument on any other value.
    static void validateObjective(const UhdConfig& cfg,
                                   int64_t engineId,
                                   const std::string& role,
                                   const std::string& arch);

    /// Entries are held by shared_ptr so a reader can outlive a re-registration that
    /// replaces the entry (see getEngine).
    std::unordered_map<int64_t, std::shared_ptr<EngineEntry>> _engines;
    mutable std::mutex _mutex;
};

} // namespace hipdnn_backend::heuristics::uhd
