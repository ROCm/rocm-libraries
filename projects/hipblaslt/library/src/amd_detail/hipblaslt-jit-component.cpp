// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include <cerrno>
#include <new>
#include <optional>
#include <stdexcept>
#include <system_error>
#ifdef _WIN32
#include <random>
#else
#include <stdlib.h>
#endif

namespace hipblaslt_jit
{
    namespace
    {
        namespace fs = std::filesystem;

        fs::path makeScratch()
        {
            const auto parent = fs::temp_directory_path();
#ifdef _WIN32
            std::random_device random;
            for(int attempt = 0; attempt != 100; ++attempt)
            {
                const auto path = parent
                                  / ("hipblaslt-jit-" + std::to_string(random())
                                     + std::to_string(random()));
                if(fs::create_directory(path))
                    return path;
            }
            throw std::runtime_error("Cannot create a JIT scratch directory in "
                                     + parent.u8string());
#else
            auto pattern = (parent / "hipblaslt-jit-XXXXXX").string();
            if(!mkdtemp(pattern.data()))
                throw std::system_error(errno,
                                        std::generic_category(),
                                        "Cannot create a JIT scratch directory in "
                                            + parent.string());
            return pattern;
#endif
        }

        // Removed when the call succeeds; kept after a failure unless it is empty.
        class Scratch
        {
        public:
            Scratch()
                : m_path(makeScratch())
            {
            }
            Scratch(const Scratch&)            = delete;
            Scratch& operator=(const Scratch&) = delete;
            ~Scratch()
            {
                std::error_code error;
                if(!m_keep || fs::is_empty(m_path, error))
                    fs::remove_all(m_path, error);
            }
            void keep() noexcept
            {
                m_keep = true;
            }
            const fs::path& path() const noexcept
            {
                return m_path;
            }

        private:
            fs::path m_path;
            bool     m_keep = false;
        };

        template <class F>
        Status guarded(F&& f)
        {
            try
            {
                return f();
            }
            catch(const std::bad_alloc&)
            {
                throw;
            }
            catch(const std::exception& e)
            {
                return {Status::Code::Failed, Stage::Configure, e.what()};
            }
            catch(...)
            {
                return {Status::Code::Failed, Stage::Configure, "Unknown exception"};
            }
        }
    }

    const char* toString(Stage stage) noexcept
    {
        switch(stage)
        {
        case Stage::Configure:
            return "configure";
        case Stage::Predict:
            return "predict";
        case Stage::Generate:
            return "generate";
        case Stage::Build:
            return "build";
        case Stage::Support:
            return "support";
        case Stage::Load:
            return "load";
        case Stage::Lookup:
            return "lookup";
        case Stage::Publish:
            return "publish";
        }
        return "unknown stage";
    }

    Jit::Jit(Components components)
        : m_components(std::move(components))
    {
        const auto& c = m_components;
        if(!c.backend || !c.builder || !c.loader)
            throw std::invalid_argument("Jit requires a backend, a builder and a loader");
        const auto& contract = c.backend->info().consumesPrediction;
        if(!contract.empty()
           && (!c.predictor || !c.knowledge || c.predictor->modeledContract() != contract))
            throw std::invalid_argument("Backend " + c.backend->info().id
                                        + " requires a predictor for " + contract
                                        + " and tuning knowledge");
    }

    Jit::Outcome Jit::generate(const OperationRequest&         request,
                               const DeviceTarget&             target,
                               size_t                          count,
                               size_t                          workspaceLimit,
                               const std::vector<std::string>& excludeKernels) const
    {
        const auto& c = m_components;
        Outcome     outcome;
        if(count == 0)
            return outcome;
        std::optional<debug::Generation> trace;
        if(debug::categories())
            trace.emplace(count);
        auto record = [&](Stage stage, Status status) {
            status.stage = stage;
            if(trace)
                trace->failure(toString(stage), status.message);
            outcome.failures.push_back(std::move(status));
        };

        Prediction prediction;
        const bool predicted = !c.backend->info().consumesPrediction.empty();
        if(predicted)
        {
            debug::Phase phase("predict");
            auto         status = guarded(
                [&] { return c.predictor->predict(request, target, *c.knowledge, prediction); });
            phase.stop();
            if(!status.ok())
            {
                record(Stage::Predict, std::move(status));
                return outcome;
            }
        }
        if(trace)
            trace->started(prediction.ranked.size());

        std::optional<Scratch> scratch;
        debug::Phase           scratchPhase("scratch");
        auto                   status = guarded([&] {
            scratch.emplace();
            return Status{};
        });
        scratchPhase.stop();
        if(!status.ok())
        {
            record(Stage::Configure, std::move(status));
            return outcome;
        }
        const GenerationRequest generation{request,
                                           target,
                                           predicted ? &prediction : nullptr,
                                           count,
                                           workspaceLimit,
                                           excludeKernels,
                                           scratch->path()};
        std::vector<GeneratedSolution> generated;
        debug::Phase                   backendPhase("backend");
        status = guarded([&] { return c.backend->generate(generation, generated); });
        backendPhase.stop();
        if(!status.ok())
        {
            record(status.code == Status::Code::TargetMismatch ? Stage::Configure
                                                               : Stage::Generate,
                   std::move(status));
            scratch->keep();
            return outcome;
        }
        outcome.summary = std::move(status.message);
        if(trace)
            trace->record().count("generated", static_cast<int64_t>(generated.size()));

        std::vector<BuiltSolution> supported;
        std::vector<size_t>        ranks; // of supported, in generated
        for(size_t rank = 0; rank < generated.size(); ++rank)
        {
            const auto& solution = generated[rank];
            if(supported.size() == count)
                break;
            debug::Scope  scope(trace ? &trace->solution(rank, solution.kernelName) : nullptr);
            BuiltSolution built;
            debug::Phase  buildPhase("build");
            status = guarded([&] { return c.builder->build(solution, generation, built); });
            buildPhase.stop();
            if(trace)
                trace->built(rank, status.ok() ? "built" : "build_failed", status.message);
            if(!status.ok())
            {
                record(Stage::Build, std::move(status));
                continue;
            }
            debug::Phase supportPhase("support");
            status = guarded(
                [&] { return c.loader->support(built, request, target, workspaceLimit); });
            supportPhase.stop();
            if(!status.ok())
            {
                if(trace)
                    trace->outcome(rank, "unsupported", status.message);
                record(Stage::Support, std::move(status));
                continue;
            }
            supported.push_back(std::move(built));
            ranks.push_back(rank);
        }

        bool load = true;
        if(c.store && !supported.empty())
        {
            std::vector<int32_t> indices;
            if(trace)
                trace->publishing(supported.size());
            debug::Phase publishPhase("publish");
            status = guarded([&] { return c.store->publish(request, target, supported, indices); });
            publishPhase.stop();
            if(status.ok() && indices.size() != supported.size())
                status = {Status::Code::Failed,
                          Stage::Publish,
                          "Solution store returned " + std::to_string(indices.size())
                              + " indices for " + std::to_string(supported.size())
                              + " solutions"};
            if(trace)
            {
                for(size_t i = 0; i < ranks.size(); ++i)
                    if(status.ok())
                    {
                        trace->outcome(ranks[i], "published");
                        trace->indexed(ranks[i], indices[i]);
                    }
                    else
                        trace->outcome(ranks[i], "publish_failed", status.message);
                if(status.ok())
                    trace->record().count("published", static_cast<int64_t>(indices.size()));
                trace->published(status.ok() ? "ok" : "failed", status.ok() ? indices.size() : 0);
            }
            if(status.ok())
            {
                outcome.indices = std::move(indices);
                load            = false;
            }
            else
                record(Stage::Publish, std::move(status));
        }
        if(load)
        {
            debug::Phase loadPhase("load");
            for(size_t i = 0; i < supported.size(); ++i)
            {
                const auto&                         built = supported[i];
                std::shared_ptr<const KernelBundle> bundle;
                status = guarded([&] {
                    return c.loader->load(built, request, target, workspaceLimit, bundle);
                });
                if(status.ok() && !bundle)
                    status = {Status::Code::Failed, Stage::Load, "Loader returned no bundle"};
                if(trace)
                    trace->outcome(ranks[i],
                                   status.ok() ? "loaded" : "load_failed",
                                   status.ok() ? std::string() : status.message);
                if(status.ok())
                    outcome.unpublished.push_back(std::move(bundle));
                else
                    record(Stage::Load, std::move(status));
            }
            loadPhase.stop();
            if(trace && !supported.empty())
            {
                trace->record().count("loaded", static_cast<int64_t>(outcome.unpublished.size()));
                trace->loaded(outcome.unpublished.size());
            }
        }

        if(!outcome.failures.empty())
            scratch->keep();
        return outcome;
    }
}
