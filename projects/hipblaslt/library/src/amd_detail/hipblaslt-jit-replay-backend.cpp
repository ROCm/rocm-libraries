// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-replay.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <Tensile/ContractionProblemPredicates.hpp>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

namespace hipblaslt_ext::experimental::jit::replay
{
    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_jit::Processor;
        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;
        using Solution = std::shared_ptr<TensileLite::ContractionSolution>;

        // The processor of the assembly's `.amdgcn_target` directive, or gfx000.
        Processor declaredProcessor(const hipblaslt_jit::Artifact& assembly)
        {
            const std::string text(assembly.bytes.begin(), assembly.bytes.end());
            const std::string directive = ".amdgcn_target \"amdgcn-amd-amdhsa--";
            const auto        at        = text.find(directive);
            if(at == std::string::npos)
                return Processor::gfx000;
            const auto begin  = at + directive.size();
            const auto target = hipblaslt_jit::Target::fromTargetId(
                text.substr(begin, text.find('"', begin) - begin));
            return target ? target->processor : Processor::gfx000;
        }

        // A copy of solution named <kernel><suffix> with workGroupMapping and,
        // unless kMultiple is 0, a problem predicate that K is a multiple of it.
        Solution variant(const Solution&    solution,
                         const std::string& suffix,
                         int                workGroupMapping,
                         size_t             kMultiple)
        {
            using Gemm      = TensileLite::ContractionProblemGemm;
            using Predicate = TensileLite::Predicates::Predicate<Gemm>;
            using And       = TensileLite::Predicates::And<Gemm>;
            auto copy                          = hipblaslt_jit::copySolution(solution);
            copy->solutionName                 = solution->kernelName + suffix;
            copy->sizeMapping.workGroupMapping = workGroupMapping;
            if(kMultiple)
            {
                std::vector<std::shared_ptr<Predicate>> terms;
                if(const auto all = std::dynamic_pointer_cast<And>(solution->problemPredicate))
                    terms = all->value;
                else if(solution->problemPredicate)
                    terms.push_back(solution->problemPredicate);
                terms.push_back(
                    std::make_shared<TensileLite::Predicates::Contraction::BoundSizeMultiple>(
                        0, kMultiple));
                copy->problemPredicate = std::make_shared<And>(terms);
            }
            return copy;
        }

        struct Replayed
        {
            fs::path  directory;
            Processor processor = Processor::gfx000;
        };

        // Increment when a replayed solution can change.
        constexpr uint32_t replayVersion = 2;

        class ReplayBackend final : public hipblaslt_jit::Backend
        {
        public:
            explicit ReplayBackend(const Options& options)
                : m_info{"replay", "replay", replayVersion}
                , m_builder(hipblaslt_jit::makeComgrBuilder())
            {
                if(options.replay.empty())
                    throw std::invalid_argument("The replay backend has no bundle to replay");
                for(const auto& path : options.replay)
                {
                    Replayed   replayed{fs::u8path(path)};
                    const auto source = hipblaslt_jit::source_bundle::readSourceBundle(
                        replayed.directory);
                    if(source.mainKernel.kind == hipblaslt_jit::Artifact::Kind::Assembly)
                        replayed.processor = declaredProcessor(source.mainKernel);
                    if(replayed.processor == Processor::gfx000)
                        throw std::invalid_argument("The bundle " + path
                                                    + " has no assembly for a known target");
                    m_replayed.push_back(std::move(replayed));
                }
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status generate(const hipblaslt_jit::GenerationRequest&       request,
                            std::vector<hipblaslt_jit::SourceCodeBundle>& bundles) const override
            {
                bundles.clear();
                const auto problem
                    = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(request.request);
                bool   targeted = false, solves = false;
                size_t solving  = 0;
                for(const auto& replayed : m_replayed)
                {
                    if(solving >= request.count)
                        break;
                    if(replayed.processor != request.target.processor
                       || !request.target.hardware)
                        continue;
                    const auto cacheKey = std::make_tuple(replayed.directory.u8string(),
                                                           static_cast<int>(replayed.processor),
                                                           request.codeObjectVersion);
                    std::shared_ptr<const hipblaslt_jit::SourceCodeBundle> cached;
                    {
                        std::lock_guard<std::mutex> guard(m_bundleCacheMutex);
                        const auto                  found = m_bundleCache.find(cacheKey);
                        if(found != m_bundleCache.end())
                            cached = found->second;
                    }
                    if(!cached)
                    {
                        hipblaslt_jit::SourceCodeBundle source;
                        auto status = hipblaslt_jit::readTensileSourceBundle(*m_builder,
                                                                             replayed.directory,
                                                                             request.target,
                                                                             request.codeObjectVersion,
                                                                             source);
                        hipblaslt_jit::keepBuildLog(status, request.scratch);
                        if(!status.ok())
                            return status;
                        if(source.solutions.size() != 1)
                            return {Status::Code::Failed,
                                    Stage::Generate,
                                    "The kernel of " + replayed.directory.u8string()
                                        + " does not record one solution"};
                        const auto solution = source.solutions.front();
                        source.solutions    = {variant(solution, "_K512_WGM8", 8, 512),
                                               variant(solution, "_WGM1", 1, 0)};
                        cached = std::make_shared<const hipblaslt_jit::SourceCodeBundle>(
                            std::move(source));
                        std::lock_guard<std::mutex> guard(m_bundleCacheMutex);
                        m_bundleCache.emplace(cacheKey, cached);
                    }
                    const auto& source = *cached;

                    size_t bundleSolving = 0;
                    bool   fresh         = false;
                    for(const auto& replayedSolution : source.solutions)
                    {
                        if(!(*replayedSolution->hardwarePredicate)(*request.target.hardware))
                            continue;
                        targeted = true;
                        if(!(*replayedSolution->problemPredicate)(problem))
                            continue;
                        solves = true;
                        ++bundleSolving;
                        fresh = fresh
                                || !hipblaslt_jit::excludedKernel(request.excludeKernels,
                                                                  replayedSolution->kernelName);
                    }
                    if(!fresh)
                        continue;
                    solving += bundleSolving;
                    bundles.push_back(source);
                }
                if(!targeted)
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "No replayed solution targets " + request.target.processorName()};
                if(!solves)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "No replayed solution solves this problem"};
                return {Status::Code::Success,
                        Stage::Generate,
                        "Replayed " + std::to_string(bundles.size())
                            + (bundles.size() == 1 ? " bundle" : " bundles")};
            }

        private:
            hipblaslt_jit::BackendInfo                              m_info;
            std::shared_ptr<const hipblaslt_jit::CodeObjectBuilder> m_builder;
            std::vector<Replayed>                                   m_replayed;
            using BundleKey = std::tuple<std::string, int, int>;
            mutable std::mutex                                                      m_bundleCacheMutex;
            mutable std::map<BundleKey, std::shared_ptr<const hipblaslt_jit::SourceCodeBundle>>
                m_bundleCache;
        };
    }

    std::shared_ptr<const hipblaslt_jit::Backend> makeBackend(const Options& options)
    {
        return std::make_shared<const ReplayBackend>(options);
    }

    hipblasStatus_t
        createBackend(const Options&                              options,
                      std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                      Diagnostics&                                diagnostics)
    {
        jit         = {};
        diagnostics = {"replay", ""};
        try
        {
            jit = hipblaslt_jit::makeJit(makeBackend(options));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate the replay backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        catch(const std::exception& e)
        {
            diagnostics.message = e.what();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
    }
}
