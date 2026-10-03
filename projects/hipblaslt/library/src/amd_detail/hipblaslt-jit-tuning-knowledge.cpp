// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-json.hpp"
#include "hipblaslt-jit-knowledge.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "rocblaslt_secure_env.hpp"
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>

namespace hipblaslt_jit
{
    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_ext::experimental::jit::detail::GemmRequest;

        constexpr size_t seedCount = 8;
        const std::string prefix   = "hipblaslt-jit-knowledge-";
        const std::string suffix   = ".dat.zlib";

        // The fields a ProblemType must equal, as Tensile.JitKnowledge.coreKey writes them.
        std::string coreKey(const TensileLite::ContractionProblemGemm& problem,
                            const CanonicalGemm&                       gemm)
        {
            const auto type = [](rocisa::DataType value) { return static_cast<int>(value); };
            std::vector<std::pair<const char*, long long>> fields{
                {"DataTypeA", type(problem.a().dataType())},
                {"DataTypeB", type(problem.b().dataType())},
                {"MacDataTypeA", type(problem.computeInputTypeA())},
                {"MacDataTypeB", type(problem.computeInputTypeB())},
                {"DestDataType", type(problem.d().dataType())},
                {"ComputeDataType", type(problem.computeType())},
                {"F32XdlMathOp", type(problem.f32XdlMathOp())},
                {"HighPrecisionAccumulate", problem.highPrecisionAccumulate()},
                {"TransposeA", gemm.transA},
                {"TransposeB", gemm.transB},
                {"Sparse", problem.sparse()},
                {"SwizzleTensorA", problem.swizzleTensorA()},
                {"SwizzleTensorB", problem.swizzleTensorB()},
                {"MXBlockA", static_cast<long long>(problem.mxBlockA())},
                {"MXBlockB", static_cast<long long>(problem.mxBlockB())},
            };
            if(problem.mxBlockA())
                fields.emplace_back("DataTypeMXSA", type(problem.mxTypeA()));
            if(problem.mxBlockB())
                fields.emplace_back("DataTypeMXSB", type(problem.mxTypeB()));
            std::string key;
            for(const auto& [name, value] : fields)
                key += (key.empty() ? "" : ",") + std::string(name) + "=" + std::to_string(value);
            return key;
        }

        knowledge::Features features(const TensileLite::ContractionProblemGemm& problem)
        {
            knowledge::Features result;
            if(problem.useBias())
                result.bias = {static_cast<int>(problem.bias().dataType())};
            result.activation = problem.activationType() != TensileLite::ActivationType::None;
            result.scaleAB    = problem.useScaleAB();
            for(const auto& [name, on] : {std::pair{"UseScaleAlphaVec", problem.useScaleAlphaVec() != 0},
                                          std::pair{"UseScaleCD", problem.useScaleCD()},
                                          std::pair{"UseE", problem.useE()},
                                          std::pair{"Gradient", problem.useGradient()},
                                          std::pair{"OutputAmaxD", problem.outputAmaxD()},
                                          std::pair{"UseGateResidual", problem.useGateResidual()}})
                if(on)
                    result.flags.insert(name);
            return result;
        }

        bool satisfies(const TensileLite::ContractionProblemGemm& problem,
                       const CanonicalGemm&                       gemm,
                       const std::string&                         name,
                       int64_t                                    value)
        {
            const auto multiple = [&](size_t size) { return value > 0 && size % value == 0; };
            if(name == "AssertFree0ElementMultiple")
                return multiple(gemm.m);
            if(name == "AssertFree1ElementMultiple")
                return multiple(gemm.n);
            if(name == "AssertSummationElementMultiple")
                return multiple(gemm.k);
            if(name == "AssertAIGreaterThanEqual")
                return problem.arithmeticIntensity() >= value;
            if(name == "AssertAILessThanEqual")
                return problem.arithmeticIntensity() <= value;
            return false;
        }

        int jsonInteger(const std::vector<TuningParameter>& parameters, const char* name)
        {
            for(const auto& parameter : parameters)
                if(parameter.name == name)
                    return std::atoi(parameter.json.c_str());
            return 0;
        }

        struct Installed
        {
            std::shared_ptr<const knowledge::Database> database;
            std::string                                reason; // why there is none
        };

        // Tuned sets nearest to the request ahead of the catalog's seeds. The
        // headers of every installed file load once, for version(); a group's
        // block loads when a request first needs it.
        class TuningLibraryKnowledge final : public TuningKnowledge
        {
        public:
            TuningLibraryKnowledge(fs::path directory, bool perArchitecture, std::string off)
                : m_catalog(makeCatalogKnowledge())
                , m_directory(std::move(directory))
                , m_perArchitecture(perArchitecture)
                , m_off(std::move(off))
            {
                if(m_off.empty() && !m_directory.empty())
                    scan();
                for(const auto& [arch, installed] : m_installed)
                    if(installed.database)
                        m_version += (m_version.empty() ? "" : ";") + arch + "="
                                     + installed.database->contentHash();
                m_id = m_version.empty() ? std::string(m_catalog->id()) : "tensilelite-logic.v2";
                if(m_version.empty())
                    m_version = m_catalog->version();
            }

            std::string_view id() const noexcept override
            {
                return m_id;
            }
            std::string version() const override
            {
                return m_version;
            }
            std::vector<CandidateSeed> seeds(const OperationRequest& request,
                                             const DeviceTarget&     target) const override
            {
                auto result  = tuned(request, target);
                auto catalog = m_catalog->seeds(request, target);
                result.insert(result.end(),
                              std::make_move_iterator(catalog.begin()),
                              std::make_move_iterator(catalog.end()));
                return result;
            }
            std::vector<TuningParameter> defaults(const OperationRequest& request,
                                                  const DeviceTarget&     target,
                                                  const Candidate&        candidate) const override
            {
                return m_catalog->defaults(request, target, candidate);
            }

        private:
            fs::path path(const std::string& arch) const
            {
                if(m_directory.empty())
                    return {};
                return (m_perArchitecture ? m_directory / arch : m_directory)
                       / (prefix + arch + suffix);
            }

            void scan()
            {
                std::error_code error;
                const auto      open = [&](const std::string& arch) {
                    const auto file = path(arch);
                    if(!fs::is_regular_file(file, error))
                        return;
                    auto& installed = m_installed[arch];
                    try
                    {
                        auto database = std::make_shared<const knowledge::Database>(file);
                        if(database->libraryArch() != arch)
                            installed.reason = file.string() + " holds " + database->libraryArch()
                                               + " knowledge";
                        else
                            installed.database = std::move(database);
                    }
                    catch(const std::runtime_error& e)
                    {
                        installed.reason = e.what();
                    }
                };
                for(fs::directory_iterator it(m_directory, error), end; !error && it != end;
                    it.increment(error))
                {
                    const auto name = it->path().filename().string();
                    if(m_perArchitecture)
                        open(name);
                    else if(name.size() > prefix.size() + suffix.size()
                            && name.compare(0, prefix.size(), prefix) == 0
                            && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
                        open(name.substr(prefix.size(),
                                         name.size() - prefix.size() - suffix.size()));
                }
            }

            // The architecture's database, after one debug record per process and architecture.
            const knowledge::Database* database(const std::string& arch) const
            {
                const auto  found     = m_installed.find(arch);
                const auto* installed = found == m_installed.end() ? nullptr : &found->second;
                const auto* database  = installed ? installed->database.get() : nullptr;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if(!m_reported.insert(arch).second)
                        return database;
                }
                if(debug::on(debug::Knowledge))
                {
                    debug::Line line(debug::Knowledge, "load");
                    line.add("arch", arch).add("path", path(arch).string());
                    if(database)
                        line.add("status", "loaded")
                            .add("content_hash", database->contentHash())
                            .add("groups", database->groups().size());
                    else
                        line.add("status", "catalog")
                            .add("reason",
                                 !m_off.empty()             ? m_off
                                 : m_directory.empty()      ? "no Tensile library directory"
                                 : installed                ? installed->reason
                                                            : "no file at " + path(arch).string());
                    line.write();
                }
                return database;
            }

            std::vector<CandidateSeed> tuned(const OperationRequest& request,
                                             const DeviceTarget&     target) const
            {
                const auto* database = this->database(target.libraryArch);
                const auto* gemm     = dynamic_cast<const GemmRequest*>(&request);
                if(!database || !gemm)
                    return {};
                std::vector<CandidateSeed> seeds;
                try
                {
                    const auto problem = lowerForJit(*gemm);
                    const auto shape   = canonicalGemm(problem);
                    knowledge::Device device{target.cuCount, std::nullopt, {}};
                    if(const auto* gpu
                       = dynamic_cast<const TensileLite::AMDGPU*>(target.hardware.get()))
                    {
                        device.cuCount = gpu->computeUnitCount;
                        if(TensileLite::ChipIdRegistry::supportsChipIdPredicate(gpu->processor))
                            device.pciChipId = gpu->pciChipId();
                        if(device.pciChipId)
                            device.fallbackChipIds
                                = TensileLite::ChipIdRegistry::getFallbackChipIds(*device.pciChipId);
                    }
                    const auto match = database->nearest(
                        {coreKey(problem, shape), features(problem), {shape.m, shape.n, shape.batch, shape.k}},
                        device,
                        seedCount);
                    if(!match.corrupt.empty() && debug::on(debug::Knowledge))
                        debug::Line(debug::Knowledge, "corrupt")
                            .add("arch", target.libraryArch)
                            .add("path", database->path().string())
                            .add("reason", match.corrupt)
                            .write();
                    for(const auto& found : match.seeds)
                    {
                        CandidateSeed seed;
                        seed.tile = {found.macroTile[0], found.macroTile[1], found.waves[0], found.waves[1]};
                        seed.instruction = found.instruction;
                        seed.depthU      = found.depthU;
                        seed.parameters  = found.parameters;
                        for(const auto& [name, value] : found.asserts)
                            if(satisfies(problem, shape, name, value))
                                seed.parameters.push_back({name, std::to_string(value)});
                        seed.cacheHints = {{jsonInteger(found.parameters, "NonTemporalA"),
                                            jsonInteger(found.parameters, "NonTemporalB")}};
                        seed.policies   = {found.policy};
                        seed.rank       = found.rank;
                        seed.provenance = json::object({
                            {"branch", json::literal(found.branch)},
                            {"group", json::quote(match.group)},
                            {"source", json::quote(found.source)},
                            {"row", json::array(found.row)},
                            {"distance", json::literal(found.distance)},
                        });
                        seeds.push_back(std::move(seed));
                    }
                }
                catch(const std::runtime_error&)
                {
                    // Not a GEMM Tensile describes; the predictor reports why.
                    return {};
                }
                return seeds;
            }

            std::shared_ptr<const TuningKnowledge> m_catalog;
            fs::path                               m_directory;
            bool                                   m_perArchitecture;
            std::string                            m_off; // why knowledge is off, or empty
            std::map<std::string, Installed>       m_installed;
            std::string                            m_id, m_version;
            mutable std::mutex                     m_mutex;
            mutable std::set<std::string>          m_reported;
        };
    }

    std::shared_ptr<const TuningKnowledge> makeTuningLibraryKnowledge(std::filesystem::path directory,
                                                                      bool perArchitecture)
    {
        const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT_KNOWLEDGE");
        return std::make_shared<const TuningLibraryKnowledge>(
            std::move(directory),
            perArchitecture,
            value && std::string(value) == "none" ? "HIPBLASLT_JIT_KNOWLEDGE=none" : "");
    }
}
