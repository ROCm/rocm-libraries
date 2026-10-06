// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-library.hpp"
#include <algorithm>
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
    std::shared_ptr<const SolutionLoader> makeTensileLoader();

    std::shared_ptr<const Jit> makeJit(std::shared_ptr<const Backend> backend)
    {
        const auto info  = backend->info();
        auto       store = makeLibraryStore(JitLibrary::process(), info, jitCodeObjectVersion);
        return std::make_shared<const Jit>(Jit::Components{
            std::move(backend), makeComgrBuilder(), makeTensileLoader(), std::move(store)});
    }

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
        auto record = [&](Stage stage, Status status) {
            status.stage = stage;
            outcome.failures.push_back(std::move(status));
        };

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
        const GenerationRequest generation{
            request, target, count, workspaceLimit, excludeKernels, scratch->path()};
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

        const BuildRequest build{target.targetId, generation.codeObjectVersion, scratch->path()};
        SupportedSolutions supported;
        size_t             solutions = 0;
        for(const auto& solution : generated)
        {
            if(solutions == count)
                break;
            BuiltSolution built;
            status = guarded([&] { return c.builder->build(solution, build, built); });
            if(!status.ok())
            {
                record(Stage::Build, std::move(status));
                continue;
            }
            std::vector<int> indices;
            status = guarded([&] {
                return c.loader->support(built, request, target, workspaceLimit, indices);
            });
            if(status.ok() && indices.empty())
                status = {Status::Code::Failed, Stage::Support, "Loader supported no solution"};
            if(!status.ok())
            {
                record(Stage::Support, std::move(status));
                continue;
            }
            indices.resize(std::min(indices.size(), count - solutions));
            solutions += indices.size();
            supported.emplace_back(std::move(built), std::move(indices));
        }

        bool load = true;
        if(c.store && !supported.empty())
        {
            std::vector<int32_t> indices;
            status = guarded([&] { return c.store->publish(request, target, supported, indices); });
            if(status.ok() && indices.size() != solutions)
                status = {Status::Code::Failed,
                          Stage::Publish,
                          "Solution store returned " + std::to_string(indices.size())
                              + " indices for " + std::to_string(solutions) + " solutions"};
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
            for(const auto& entry : supported)
            {
                const auto& built   = entry.first;
                const auto& indices = entry.second;
                std::vector<std::shared_ptr<const KernelBundle>> bundles;
                status = guarded([&] {
                    return c.loader->load(built, request, target, workspaceLimit, indices, bundles);
                });
                if(status.ok()
                   && (bundles.size() != indices.size()
                       || std::find(bundles.begin(), bundles.end(), nullptr) != bundles.end()))
                    status = {Status::Code::Failed,
                              Stage::Load,
                              "Loader did not return a bundle for each solution"};
                if(status.ok())
                    outcome.bundles.insert(outcome.bundles.end(), bundles.begin(), bundles.end());
                else
                    record(Stage::Load, std::move(status));
            }
        }

        if(!outcome.failures.empty())
            scratch->keep();
        return outcome;
    }
}
