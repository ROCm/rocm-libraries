// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
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
        auto        record = [&](Stage stage, Status status) {
            status.stage = stage;
            outcome.failures.push_back(std::move(status));
        };
        if(count == 0)
            return outcome;

        Prediction prediction;
        const bool predicted = !c.backend->info().consumesPrediction.empty();
        if(predicted)
        {
            auto status = guarded(
                [&] { return c.predictor->predict(request, target, *c.knowledge, prediction); });
            if(!status.ok())
            {
                record(Stage::Predict, std::move(status));
                return outcome;
            }
        }

        std::optional<Scratch> scratch;
        auto                   status = guarded([&] {
            scratch.emplace();
            return Status{};
        });
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
        status = guarded([&] { return c.backend->generate(generation, generated); });
        if(!status.ok())
        {
            record(status.code == Status::Code::TargetMismatch ? Stage::Configure
                                                               : Stage::Generate,
                   std::move(status));
            scratch->keep();
            return outcome;
        }
        outcome.summary = std::move(status.message);

        std::vector<BuiltSolution> supported;
        for(const auto& solution : generated)
        {
            if(supported.size() == count)
                break;
            BuiltSolution built;
            status = guarded([&] { return c.builder->build(solution, generation, built); });
            if(!status.ok())
            {
                record(Stage::Build, std::move(status));
                continue;
            }
            status = guarded(
                [&] { return c.loader->support(built, request, target, workspaceLimit); });
            if(!status.ok())
            {
                record(Stage::Support, std::move(status));
                continue;
            }
            supported.push_back(std::move(built));
        }

        bool load = true;
        if(c.store && !supported.empty())
        {
            std::vector<int32_t> indices;
            status = guarded([&] { return c.store->publish(request, target, supported, indices); });
            if(status.ok() && indices.size() != supported.size())
                status = {Status::Code::Failed,
                          Stage::Publish,
                          "Solution store returned " + std::to_string(indices.size())
                              + " indices for " + std::to_string(supported.size())
                              + " solutions"};
            if(status.ok())
            {
                outcome.indices = std::move(indices);
                load            = false;
            }
            else
                record(Stage::Publish, std::move(status));
        }
        if(load)
            for(const auto& built : supported)
            {
                std::shared_ptr<const KernelBundle> bundle;
                status = guarded([&] {
                    return c.loader->load(built, request, target, workspaceLimit, bundle);
                });
                if(status.ok() && !bundle)
                    status = {Status::Code::Failed, Stage::Load, "Loader returned no bundle"};
                if(status.ok())
                    outcome.unpublished.push_back(std::move(bundle));
                else
                    record(Stage::Load, std::move(status));
            }

        if(!outcome.failures.empty())
            scratch->keep();
        return outcome;
    }
}
