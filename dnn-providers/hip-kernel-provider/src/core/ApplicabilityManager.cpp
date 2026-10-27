// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "ApplicabilityManager.hpp"

#include <algorithm>
#include <charconv>
#include <exception>
#include <optional>
#include <utility>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

namespace hip_kernel_provider::core
{

namespace
{

/// A non-negative integer spanning the whole of `text`, else nullopt.
std::optional<std::size_t> parseCount(const std::string& text)
{
    std::size_t value = 0;
    const auto* const end = text.data() + text.size();
    const auto [stop, error] = std::from_chars(text.data(), end, value);
    if(text.empty() || error != std::errc{} || stop != end)
    {
        return std::nullopt;
    }
    return value;
}

} // namespace

std::size_t parseMinOffloadVariants(const std::string& raw)
{
    if(raw.empty())
    {
        return DEFAULT_MIN_OFFLOAD_VARIANTS;
    }

    if(const auto value = parseCount(hipdnn_data_sdk::utilities::trim(raw)))
    {
        return *value;
    }

    HIPDNN_PLUGIN_LOG_WARN("ignoring unrecognized value for "
                           << APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME << ": '" << raw << "'");
    return DEFAULT_MIN_OFFLOAD_VARIANTS;
}

std::size_t minOffloadVariantsFromEnv()
{
    return parseMinOffloadVariants(
        hipdnn_data_sdk::utilities::getEnv(APPLICABILITY_MIN_OFFLOAD_VARIANTS_ENV_NAME));
}

std::size_t resolveTotalThreads(std::size_t pending, std::size_t cpus) noexcept
{
    return std::max<std::size_t>(1, std::min({pending, cpus, MAX_APPLICABILITY_THREADS}));
}

ApplicabilityManager::ApplicabilityManager([[maybe_unused]] ApplicabilityPool& pool)
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    : _pool(pool)
#endif
{
}

void ApplicabilityManager::addEngine(std::unique_ptr<Engine> engine, std::string_view name)
{
    Base::addEngine(std::move(engine), name);

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    // An insert may rehash, so the order is taken afresh from the base class.
    _entries.clear();
    _stagedCount = 0;
    for(const auto& [id, registered] : engines())
    {
        auto* staged = dynamic_cast<StagedEngine*>(registered.get());
        _entries.push_back({id, registered.get(), staged});
        _stagedCount += staged != nullptr ? 1 : 0;
    }
#endif
}

std::size_t ApplicabilityManager::stagedEngineCount() const noexcept
{
    return _stagedCount;
}

std::vector<int64_t> ApplicabilityManager::getApplicableEngineIds(Handle& handle,
                                                                  const IGraph& opGraph)
{
#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
    // Fewer than two staged engines can never have two pending, so they keep the base loop.
    if(_stagedCount >= 2)
    {
        return getApplicableEngineIdsStaged(handle, opGraph);
    }
#endif
    return Base::getApplicableEngineIds(handle, opGraph);
}

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

std::vector<int64_t> ApplicabilityManager::getApplicableEngineIdsStaged(Handle& handle,
                                                                        const IGraph& opGraph)
{
    struct Slot
    {
        explicit Slot(const Entry& entry)
            : id(entry.id)
            , engine(entry.engine)
            , staged(entry.staged)
        {
        }

        int64_t id;
        Engine* engine;
        StagedEngine* staged;
        char verdict = 0;
        std::exception_ptr error;
        hipdnn_plugin_sdk::ingestor::PendingApplicability pending;
    };

    // Every pending borrows the scope's catalog key, so the scope is declared before the slots
    // that hold them, and every finish, inline or on the pool (tryRun joins before it returns),
    // completes before either is destroyed. The device is resolved, and cached, by
    // beginApplicability on this thread; finishApplicability needs neither it nor the handle.
    hipdnn_plugin_sdk::ingestor::ApplicabilityScope scope;

    std::vector<Slot> slots;
    slots.reserve(_entries.size());
    for(const auto& entry : _entries)
    {
        slots.emplace_back(entry);
    }

    // Caller thread, in order. The first error ends the walk where the sequential loop would
    // have thrown, so no later engine runs.
    for(std::size_t index = 0; index < slots.size(); ++index)
    {
        auto& slot = slots[index];
        try
        {
            if(slot.staged != nullptr)
            {
                auto staged = slot.staged->beginApplicability(handle, opGraph, scope);
                slot.verdict = static_cast<char>(staged.verdict);
                slot.pending = std::move(staged.pending);
            }
            else
            {
                slot.verdict = static_cast<char>(slot.engine->isApplicable(handle, opGraph));
            }
        }
        catch(...)
        {
            slot.error = std::current_exception();
            slots.erase(slots.begin() + static_cast<std::ptrdiff_t>(index + 1), slots.end());
            break;
        }
    }

    std::vector<std::size_t> pendingSlots;
    std::size_t totalVariants = 0;
    std::size_t largest = 0;
    for(std::size_t index = 0; index < slots.size(); ++index)
    {
        if(slots[index].pending.valid())
        {
            const auto variants = slots[index].pending.predictedVariants();
            if(pendingSlots.empty() || variants > slots[largest].pending.predictedVariants())
            {
                largest = index;
            }
            pendingSlots.push_back(index);
            totalVariants += variants;
        }
    }

    // Any thread. finishApplicability makes no HIP runtime calls, so a pool thread needs no
    // device bound. Never throws, but a slot is the only way an error reaches the caller.
    auto finish = [&slots](std::size_t index) noexcept {
        auto& slot = slots[index];
        try
        {
            slot.verdict
                = static_cast<char>(slot.staged->finishApplicability(std::move(slot.pending)));
        }
        catch(...)
        {
            slot.error = std::current_exception();
        }
    };

    bool finished = false;
    if(pendingSlots.size() >= 2
       && totalVariants - slots[largest].pending.predictedVariants() >= minOffloadVariantsFromEnv())
    {
        // The largest goes first so it runs on the caller.
        std::vector<std::size_t> order;
        order.reserve(pendingSlots.size());
        order.push_back(largest);
        for(const auto index : pendingSlots)
        {
            if(index != largest)
            {
                order.push_back(index);
            }
        }

        const auto workers = resolveTotalThreads(pendingSlots.size(), affinityCpuCount()) - 1;
        if(workers > 0)
        {
            // Workers read the graph's lazily built views alongside this thread, so they are
            // built here first. Engines on this thread build them on demand, as the sequential
            // loop does, so a call that stays here never pays for them up front.
            hipdnn_plugin_sdk::ingestor::prewarmGraph(opGraph);
            finished = _pool.tryRun(order.size(), workers, [&finish, &order](std::size_t task) {
                finish(order[task]);
            });
        }
    }
    if(!finished)
    {
        for(const auto index : pendingSlots)
        {
            finish(index);
        }
    }

    std::vector<int64_t> applicable;
    for(const auto& slot : slots)
    {
        if(slot.error)
        {
            std::rethrow_exception(slot.error);
        }
        if(slot.verdict != 0)
        {
            applicable.push_back(slot.id);
        }
    }
    return applicable;
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR

} // namespace hip_kernel_provider::core
