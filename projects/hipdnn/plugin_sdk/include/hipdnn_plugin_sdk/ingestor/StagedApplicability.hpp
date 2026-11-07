// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// State shared by every engine asked about one graph in one GetApplicableEngineIds
/// call. Created and owned by the caller thread; engines fill it lazily during
/// beginApplicability (caller thread only), and it is read-only once any
/// finishApplicability has been dispatched. Pending checks borrow from it, so it must
/// outlive every finishApplicability of the call, as must the graph and handle.
class ApplicabilityScope
{
public:
    /// The device the first ingestor engine resolved, and the catalog key for it. The
    /// key carries the engine version, so only engines of `keyVersion` share it; an engine
    /// of another version builds its own. Written once per call, by the first engine, and
    /// never replaced: pending checks point into `properties`.
    struct Device
    {
        /// The resolver that answered; an engine with another resolver resolves its own
        /// device, and shares the key only if it lands on the same device id.
        const void* resolver = nullptr;
        DeviceId deviceId = NO_DEVICE;
        /// A copy, so a pending check does not depend on the resolver keeping its reference.
        DeviceProperties properties;
        /// nullopt when the graph yields no usable key; meaningful once keyBuilt.
        std::optional<CatalogKey> key;
        bool keyBuilt = false;
        /// The engine version `key` was built for; meaningful once keyBuilt.
        // Explicit init avoids -Wmissing-field-initializers on the partial Device{...} in
        // GenericPlanBuilder::beginApplicability.
        // NOLINTNEXTLINE(readability-redundant-member-init) - see above
        hipdnn_data_sdk::utilities::Version keyVersion{};
    };

    ApplicabilityScope() = default;
    ~ApplicabilityScope() = default;

    ApplicabilityScope(const ApplicabilityScope&) = delete;
    ApplicabilityScope& operator=(const ApplicabilityScope&) = delete;
    ApplicabilityScope(ApplicabilityScope&&) = delete;
    ApplicabilityScope& operator=(ApplicabilityScope&&) = delete;

    /// For the SDK's ingestor engines; empty until the first one resolves a device.
    std::optional<Device>& device() noexcept
    {
        return _device;
    }

private:
    std::optional<Device> _device;
};

/// Work left over after the caller-thread stage: for an ingestor engine, the per-variant
/// catalog loop. Each engine derives its own State and is the only one that reads it.
/// Pointer lifetime: a pending check holds raw pointers to the graph passed to
/// beginApplicability, to the ApplicabilityScope, and into the scope's device properties
/// and catalog key; it owns none of them. The caller keeps all of them alive, and does not
/// replace the scope's device, until every finishApplicability of the call has returned.
/// A pending check never outlives its call.
class PendingApplicability
{
public:
    struct State
    {
        virtual ~State() = default;
        /// Variants the finish stage will examine; the threading gate's cost estimate.
        std::size_t predictedVariants = 0;
        /// Identifies the engine whose begin produced this state, so its finish can
        /// recognise its own derived type without RTTI (plugins build with it off).
        const void* owner = nullptr;
    };

    PendingApplicability() noexcept = default;

    explicit PendingApplicability(std::unique_ptr<State> state) noexcept
        : _state(std::move(state))
    {
    }

    PendingApplicability(PendingApplicability&&) noexcept = default;
    PendingApplicability& operator=(PendingApplicability&&) noexcept = default;
    PendingApplicability(const PendingApplicability&) = delete;
    PendingApplicability& operator=(const PendingApplicability&) = delete;
    ~PendingApplicability() = default;

    bool valid() const noexcept
    {
        return _state != nullptr;
    }

    /// 0 when empty.
    std::size_t predictedVariants() const noexcept
    {
        return _state == nullptr ? 0 : _state->predictedVariants;
    }

    State* state() const noexcept
    {
        return _state.get();
    }

    /// Leaves this pending empty.
    std::unique_ptr<State> release() noexcept
    {
        return std::move(_state);
    }

private:
    std::unique_ptr<State> _state;
};

struct StagedApplicability
{
    /// true: `verdict` is final.
    bool decided = false;
    bool verdict = false;
    /// valid() iff !decided.
    PendingApplicability pending;
};

/// Applicability split into a caller-thread stage and a stage any thread may run, so a
/// caller can spread several engines' expensive halves across threads.
template <typename THandle>
class IStagedApplicability
{
public:
    virtual ~IStagedApplicability() = default;

    /// Caller thread only. Same exception behavior as isApplicable(). Decides cache hits,
    /// graph-level declines, schema-floor declines, unusable keys and any error
    /// isApplicable() catches.
    virtual StagedApplicability
        // NOLINTNEXTLINE(portability-template-virtual-member-function)
        beginApplicability(THandle& handle,
                           const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                           ApplicabilityScope& scope) const
        = 0;

    /// Any thread. Runs the per-variant loop, stores the catalog exactly as isApplicable()
    /// would, and returns its verdict. Makes no HIP runtime calls. Never throws
    /// std::exception: it catches and logs like isApplicable().
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    virtual bool finishApplicability(PendingApplicability&& pending) const = 0;
};

/// Builds @p graph's lazily initialised views on the calling thread, so engines reading
/// it concurrently afterwards only ever take const access. Swallows exceptions: each
/// engine reports its own error in its own stage.
inline void prewarmGraph(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph) noexcept
{
    // A view that fails to build is rethrown by whichever engine reads it, so the result is unused.
    const auto tryBuild = [](const auto& build) noexcept {
        try
        {
            build();
            return true;
        }
        catch(...)
        {
            return false;
        }
    };
    static_cast<void>(tryBuild([&graph] { static_cast<void>(graph.getTensorMap()); }));
    static_cast<void>(tryBuild([&graph] { static_cast<void>(graph.nodeWrappers()); }));
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
