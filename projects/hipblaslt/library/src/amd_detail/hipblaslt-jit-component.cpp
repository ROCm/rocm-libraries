// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-problem-type.hpp"
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
    std::shared_ptr<const Jit> makeJit(std::shared_ptr<const Backend> backend)
    {
        const auto info  = backend->info();
        auto       store = makeLibraryStore(processLibrary(), info, jitCodeObjectVersion);
        return std::make_shared<const Jit>(
            Jit::Components{std::move(backend), makeComgrBuilder(), std::move(store)});
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
        if(!c.backend || !c.builder || !c.store)
            throw std::invalid_argument("Jit requires a backend, a builder and a store");
    }

    Jit::Outcome Jit::generate(const GemmRequest&         request,
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
        std::vector<SourceCodeBundle> generated;
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

        std::optional<TensileLite::ContractionProblemGemm> problem;
        status = guarded([&] {
            if(!target.hardware)
                return Status{Status::Code::NotSupported,
                              Stage::Support,
                              "The device has no TensileLite hardware description"};
            problem.emplace(RocblasltToTensileContractionProblemGemm(request));
            return Status{};
        });
        if(!status.ok())
        {
            record(Stage::Support, std::move(status));
            scratch->keep();
            return outcome;
        }

        const auto libraryKey
            = CacheKey::make(target, c.backend->info(), generation.codeObjectVersion);
        for(const auto& source : generated)
        {
            if(outcome.indices.size() == count)
                break;
            std::vector<std::shared_ptr<TensileLite::ContractionSolution>> solutions;
            status = guarded([&] {
                solutions = supportingSolutions(
                    source.solutions, *problem, *target.hardware, workspaceLimit);
                if(solutions.empty())
                    return Status{Status::Code::Failed,
                                  Stage::Support,
                                  "No solution of the bundle supports the request"};
                return Status{};
            });
            if(!status.ok())
            {
                record(Stage::Support, std::move(status));
                continue;
            }
            solutions.resize(std::min(solutions.size(), count - outcome.indices.size()));

            SourceCodeBundle linked = source;
            Status           helperStatus;
            for(size_t i = 0; i < linked.helpers.size() && i < source.kernelNames.size(); ++i)
            {
                auto& helper = linked.helpers[i];
                if(helper.kind != Artifact::Kind::Hip)
                    continue;
                Artifact stored;
                helperStatus = processLibrary().acquireHelper(
                    libraryKey, target.device, source.kernelNames[i], [&](Artifact& out) {
                        SourceCodeBundle one;
                        one.mainKernel        = helper;
                        one.includes          = source.includes;
                        one.hipFlags          = source.hipFlags;
                        one.target            = source.target;
                        one.codeObjectVersion = source.codeObjectVersion;
                        auto built = c.builder->build(one, BuildKind::Relocatable, out);
                        keepBuildLog(built, scratch->path());
                        return built;
                    }, stored);
                if(!helperStatus.ok())
                    break;
                helper = std::move(stored);
            }
            if(!helperStatus.ok())
            {
                const auto stage = helperStatus.stage;
                record(stage, std::move(helperStatus));
                if(stage == Stage::Configure)
                {
                    scratch->keep();
                    return outcome;
                }
                continue;
            }
            Artifact executable;
            status = guarded(
                [&] { return c.builder->build(linked, BuildKind::Executable, executable); });
            keepBuildLog(status, scratch->path());
            if(!status.ok())
            {
                record(Stage::Build, std::move(status));
                continue;
            }

            std::vector<int32_t> indices;
            status = guarded(
                [&] { return c.store->publish(request, target, executable, solutions, indices); });
            if(status.ok() && indices.size() != solutions.size())
                status = {Status::Code::Failed,
                          Stage::Publish,
                          "Solution store returned " + std::to_string(indices.size())
                              + " indices for " + std::to_string(solutions.size())
                              + " solutions"};
            if(!status.ok())
            {
                record(Stage::Publish, std::move(status));
                continue;
            }
            outcome.indices.insert(outcome.indices.end(), indices.begin(), indices.end());
        }

        if(!outcome.failures.empty())
            scratch->keep();
        return outcome;
    }
}
