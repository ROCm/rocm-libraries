// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/UtilsOrigami.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <origami/gemm.hpp>
#include <origami/origami.hpp>
#include <origami/streamk.hpp>
#include <stdexcept>

namespace hipblaslt_jit
{
    namespace
    {
        using hipblaslt_ext::experimental::jit::detail::GemmRequest;
        using json::literal;
        using json::quote;

        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error("JIT GEMM prediction: " + message);
        }

        constexpr const char* dataParallelContract = "origami.gemm.dp.v1";
        constexpr const char* persistentContract   = "origami.gemm.persistent.v1";
        constexpr const char* tunedContract        = "tensilelite.tuned.v1";

        struct Recipe
        {
            std::array<size_t, 9> matrixInstruction;
            origami::config_t     config;
            ExecutionPolicy       policy;
            int32_t               seed  = -1;
            bool                  fixed = false;
        };

        const char* toString(ExecutionPolicy::Strategy strategy)
        {
            using S = ExecutionPolicy::Strategy;
            return strategy == S::StreamK ? "StreamK" : strategy == S::DataParallel ? "DataParallel"
                                                                                     : "None";
        }

        const char* toString(ExecutionPolicy::Assignment assignment)
        {
            using A = ExecutionPolicy::Assignment;
            return assignment == A::Hybrid             ? "Hybrid"
                   : assignment == A::DynamicWorkQueue ? "DynamicWorkQueue"
                                                       : "StaticGrid";
        }

        // Stream-K and a fixed GlobalSplitU above 1 need workspace.
        bool needsWorkspace(const CandidateSeed& seed)
        {
            if(seed.policies.front().strategy == ExecutionPolicy::Strategy::StreamK)
                return true;
            for(const auto& parameter : seed.parameters)
                if(parameter.name == "GlobalSplitU")
                    return std::atoll(parameter.json.c_str()) > 1;
            return false;
        }

        origami::config_t config(size_t m, size_t n, size_t depth, const origami::dim3_t& mi)
        {
            origami::config_t config;
            config.mt              = {m, n, depth};
            config.mi              = mi;
            config.occupancy       = 1; // Conservative model input, not WG dimensions.
            config.stream_k        = 0; // Caller-selected data-parallel candidate domain.
            config.prediction_mode = origami::prediction_modes_t::estimation;
            config.target          = origami::target_t::tensilelite;
            return config;
        }

        std::vector<Recipe> candidates(const origami::hardware_t&        hardware,
                                       origami::data_type_t              dtype,
                                       const std::vector<CandidateSeed>& seeds,
                                       size_t                            workspaceLimit,
                                       bool                              persistent)
        {
            // Use the target's instruction catalog for every datatype. Tensile
            // subsequently validates the complete instruction and tile combination.
            auto instructions = hardware.get_valid_matrix_instructions(dtype);
            std::sort(instructions.begin(), instructions.end(), [](const auto& a, const auto& b) {
                return std::array{a.m, a.n, a.k} < std::array{b.m, b.n, b.k};
            });
            std::vector<Recipe> result;
            // A fixed seed is one candidate, ranked with its own Stream-K choice,
            // if the target has its instruction.
            for(size_t index = 0; index < seeds.size(); ++index)
            {
                const auto& seed = seeds[index];
                if(!seed.instruction)
                    continue;
                const auto& shape = seed.tile;
                const auto  mi    = std::find_if(
                    instructions.begin(), instructions.end(), [&](const auto& candidate) {
                        return std::array{candidate.m, candidate.n, candidate.k}
                               == std::array{(*seed.instruction)[0],
                                             (*seed.instruction)[1],
                                             (*seed.instruction)[2]};
                    });
                if(mi == instructions.end() || seed.policies.size() != 1 || !seed.depthU
                   || shape[0] % (mi->m * shape[2]) || shape[1] % (mi->n * shape[3])
                   || (!workspaceLimit && needsWorkspace(seed)))
                    continue;
                Recipe candidate;
                candidate.fixed           = true;
                candidate.config          = config(shape[0], shape[1], seed.depthU, *mi);
                candidate.config.stream_k = seed.policies.front().strategy
                                                    == ExecutionPolicy::Strategy::StreamK
                                                ? 1
                                                : 0;
                if(!seed.cacheHints.empty())
                {
                    candidate.config.cache_hints_a = seed.cacheHints.front()[0];
                    candidate.config.cache_hints_b = seed.cacheHints.front()[1];
                }
                candidate.config.index = result.size();
                candidate.seed         = static_cast<int32_t>(index);
                result.push_back(candidate);
            }
            // Data-parallel candidates come first: Origami's ranking is stable, so
            // one stays ahead of a Stream-K candidate of equal latency, which needs workspace.
            for(const bool streamK : {false, true})
                for(const auto& mi : instructions)
                    for(size_t index = 0; index < seeds.size(); ++index)
                    {
                        const auto& seed = seeds[index];
                        if(seed.instruction)
                            continue;
                        for(const auto& policy : seed.policies)
                        {
                            if((policy.strategy == ExecutionPolicy::Strategy::StreamK) != streamK
                               || (streamK && (!workspaceLimit || !persistent)))
                                continue;
                            for(const auto& rule : seed.depthRules)
                                for(const auto& hint : seed.cacheHints)
                                {
                                    const auto&  shape = seed.tile;
                                    const size_t depth = std::max(rule[0], rule[1] * mi.k);
                                    if(!mi.m || !mi.n || !mi.k || shape[0] % (mi.m * shape[2])
                                       || shape[1] % (mi.n * shape[3]) || depth % mi.k)
                                        continue;
                                    Recipe candidate;
                                    candidate.matrixInstruction = {mi.m,
                                                                   mi.n,
                                                                   mi.k,
                                                                   1,
                                                                   1,
                                                                   shape[0] / (mi.m * shape[2]),
                                                                   shape[1] / (mi.n * shape[3]),
                                                                   shape[2],
                                                                   shape[3]};
                                    candidate.config = config(shape[0], shape[1], depth, mi);
                                    candidate.config.stream_k      = streamK ? 5 : 0;
                                    candidate.config.cache_hints_a = hint[0];
                                    candidate.config.cache_hints_b = hint[1];
                                    candidate.config.index         = result.size();
                                    candidate.policy               = policy;
                                    candidate.seed                 = static_cast<int32_t>(index);
                                    result.push_back(candidate);
                                }
                        }
                    }
            return result;
        }
    }

    Prediction rankWithOrigami(const OperationRequest&                    operation,
                               const TensileLite::ContractionProblemGemm& problem,
                               const DeviceTarget&                        target,
                               size_t                                     workspaceLimit,
                               const TuningKnowledge&                     knowledge)
    {
        using Type         = rocisa::DataType;
        const auto* device = dynamic_cast<const TensileLite::hip::HipAMDGPU*>(target.hardware.get());
        require(device && device->analyticalHardware, "actual HIP device hardware is required");
        const auto& analytical = *device->analyticalHardware;
        using Arch             = origami::hardware_t::architecture_t;
        require(analytical.N_CU && analytical.NUM_XCD && analytical.lds_capacity
                    && analytical.rf_capacity && analytical.compute_clock_ghz > 0,
                "actual device resource limits are unavailable");
        CanonicalGemm gemm;
        try
        {
            gemm = canonicalGemm(problem);
        }
        catch(const std::runtime_error& e)
        {
            throw std::runtime_error("JIT GEMM prediction: " + std::string(e.what()));
        }
        const bool   transA = gemm.transA, transB = gemm.transB;
        const size_t m = gemm.m, n = gemm.n, k = gemm.k, batch = gemm.batch;
        require(batch, "a GEMM must describe at least one batch");

        origami::problem_t request;
        request.size        = {m, n, k};
        request.batch       = batch;
        request.num_cus     = problem.getParams().smCountTarget();
        request.a_transpose = transA ? origami::transpose_t::T : origami::transpose_t::N;
        request.b_transpose = transB ? origami::transpose_t::T : origami::transpose_t::N;
        request.a_dtype     = TensileLite::datatypeToAnalyticalDatatype(problem.a().dataType());
        request.b_dtype     = TensileLite::datatypeToAnalyticalDatatype(problem.b().dataType());
        request.c_dtype     = TensileLite::datatypeToAnalyticalDatatype(problem.c().dataType());
        request.d_dtype     = TensileLite::datatypeToAnalyticalDatatype(problem.d().dataType());
        request.mi_dtype    = TensileLite::datatypeToAnalyticalDatatype(
            problem.f32XdlMathOp() == Type::XFloat32 ? Type::XFloat32
                                                     : problem.computeInputTypeA());
        request.a_mx_block_size = problem.mxBlockA();
        request.b_mx_block_size = problem.mxBlockB();
        // Origami's instruction model describes one MAC input type. Mixed MAC
        // inputs and sparse instructions still go through Tensile validation.
        const bool modeled = m && n && k
                             && problem.computeInputTypeA() == problem.computeInputTypeB()
                             && !problem.sparse();
        const auto seeds   = modeled ? knowledge.seeds(operation, target)
                                     : std::vector<CandidateSeed>{};
        // TensileLite does not generate a Stream-K kernel with an auxiliary output.
        const auto recipes
            = candidates(analytical, request.mi_dtype, seeds, workspaceLimit, !problem.useE());
        // Both mapping selectors require at least one CU per XCD. Do not let
        // an unsupported budget reach their integer divisions or replace it.
        require(origami::resolve_num_cus(request.num_cus, analytical.N_CU) >= analytical.NUM_XCD,
                "Origami mapping requires a CU budget of at least the device XCD count");
        std::vector<origami::config_t> configs;
        for(const auto& recipe : recipes)
            configs.push_back(recipe.config);
        auto ranked
            = configs.empty()
                  ? std::vector<origami::prediction_result_t>{}
                  : origami::rank_configs(request, analytical, configs, origami::model_t::gemm);
        ranked.erase(std::remove_if(ranked.begin(),
                                    ranked.end(),
                                    [](const auto& result) {
                                        return !std::isfinite(result.latency) || result.latency <= 0
                                               || result.latency
                                                      == std::numeric_limits<double>::max();
                                    }),
                     ranked.end());
        require(!ranked.empty()
                    || std::any_of(recipes.begin(),
                                   recipes.end(),
                                   [](const auto& recipe) { return recipe.fixed; }),
                "No Origami ranking: no finite positive-latency candidates for this request");

        Prediction prediction;
        prediction.modeledContract = dataParallelContract;
        prediction.model           = "origami.gemm.estimation";
        prediction.hardware        = {
            {"device_id", literal(device->deviceId)},
            {"cu_count", literal(analytical.N_CU)},
            {"xcd_count", literal(analytical.NUM_XCD)},
            {"compute_clock_ghz", literal(analytical.compute_clock_ghz)},
            {"lds_capacity", literal(analytical.lds_capacity)},
            {"rf_capacity", literal(analytical.rf_capacity)},
        };
        prediction.assumptions = {
            {"occupancy", "1"},
            {"stream_k", quote("0, or 5 for the persistent contract")},
            {"stream_k_origin",
             quote("the knowledge's execution policies; Origami does not select enablement")},
            {"workgroup_mapping",
             quote("select_workgroup_mapping; unsupported transport rejects the candidate; "
                   "the runtime maps persistent kernels at each launch")},
            {"stagger",
             quote("select_staggerU; all three outputs including zeros are preserved; "
                   "the runtime staggers persistent kernels at each launch")},
            {"vector_widths", quote("not predicted by estimation; Tensile derives actual widths")},
            {"epilogue",
             quote("bias, activation, auxiliary outputs and scaling overhead are not modeled")},
            {"architecture_constants",
             quote(analytical.arch == Arch::gfx1250
                       ? "Origami gfx1250 provisional model: upstream memory constants "
                         "reuse gfx950 with gfx1250 overrides; not calibrated for gfx1250"
                       : "Origami native architecture model")},
        };
        std::vector<Candidate> tuned;
        std::vector<bool>      placed(recipes.size());
        // A tuned set travels as it is; only its tile and policy are modeled.
        const auto addTuned = [&](size_t index, double latency) {
            const auto& recipe = recipes[index];
            const auto& config = recipe.config;
            const auto& seed   = seeds[recipe.seed];
            const auto& policy = seed.policies.front();
            Candidate   candidate;
            candidate.id              = static_cast<uint32_t>(index);
            candidate.predictedCycles = latency;
            candidate.contract        = tunedContract;
            candidate.seed            = recipe.seed;
            candidate.parameters      = seed.parameters;
            candidate.provenance      = seed.provenance;
            candidate.modeled         = {
                {"macro_tile", json::array(std::array{config.mt.m, config.mt.n, config.mt.k})},
                {"execution",
                 json::object({{"strategy", quote(toString(policy.strategy))},
                               {"assignment", quote(toString(policy.assignment))}})},
            };
            tuned.push_back(std::move(candidate));
            placed[index] = true;
        };
        for(const auto& result : ranked)
        {
            require(result.config.index < recipes.size(), "Origami returned an unknown candidate");
            const auto& recipe = recipes[result.config.index];
            const auto& config = result.config;
            if(recipe.fixed)
            {
                addTuned(config.index, result.latency);
                continue;
            }
            const bool persistent = config.stream_k > 0;
            const auto [reduction, grid, activeCUs, timesteps, split]
                = origami::gemm::compute_launch_parameters(request, analytical, config,
                                                          config.grid_selection);
            require(persistent ? reduction == origami::reduction_t::tree
                                     || reduction == origami::reduction_t::parallel
                               : reduction == origami::reduction_t::none && split == 1,
                    "Origami returned a launch outside the candidate's contract");
            const auto mapping = origami::select_workgroup_mapping(request, analytical, config, grid);
            require(persistent || mapping.wgm != 0, "Origami returned a zero workgroup mapping");
            const auto stagger = origami::select_staggerU(request, analytical, config, grid, mapping.wgm);
            json::Members launch{{"stream_k", "0"},
                                 {"reduction", quote("none")},
                                 {"grid", literal(grid)},
                                 {"active_cus", literal(activeCUs)},
                                 {"timesteps", literal(timesteps)},
                                 {"split_factor", literal(split)}};
            Candidate     candidate;
            candidate.id              = static_cast<uint32_t>(config.index);
            candidate.predictedCycles = result.latency;
            candidate.contract        = dataParallelContract;
            candidate.seed            = recipe.seed;
            // The runtime chooses these again at each launch of a persistent kernel.
            if(persistent)
            {
                launch.erase(launch.begin());
                launch.front().second
                    = quote(reduction == origami::reduction_t::parallel ? "parallel" : "tree");
                launch.push_back({"hybrid_mode",
                                  quote(origami::hybrid_mode_to_string(
                                      origami::streamk::select_hybrid_mode(
                                          request, analytical, config, request.num_cus)))});
                candidate.contract = persistentContract;
            }
            candidate.parameters      = {
                {"MatrixInstruction", json::array(recipe.matrixInstruction)},
                {"DepthU", literal(config.mt.k)},
                {"NonTemporalA", literal(config.cache_hints_a)},
                {"NonTemporalB", literal(config.cache_hints_b)},
            };
            candidate.modeled = {
                {"macro_tile", json::array(std::array{config.mt.m, config.mt.n, config.mt.k})},
                {"workgroup_mapping",
                 json::object({{"wgm", literal(mapping.wgm)},
                               {"wgmxcc", literal(mapping.wgmxcc)},
                               {"wgmxccchunk", literal(mapping.wgmxccchunk)},
                               {"wgmxccsplitk", literal(mapping.wgmxccsplitk)}})},
                {"stagger",
                 json::object({{"staggerU", literal(stagger.staggerU)},
                               {"staggerUMapping", literal(stagger.staggerUMapping)},
                               {"staggerUStrideShift", literal(stagger.staggerUStrideShift)}})},
                {"launch", json::object(launch)},
            };
            if(persistent)
                candidate.modeled.insert(
                    candidate.modeled.begin() + 1,
                    {"execution",
                     json::object({{"strategy", quote(toString(recipe.policy.strategy))},
                                   {"assignment", quote(toString(recipe.policy.assignment))}})});
            for(auto& parameter : knowledge.defaults(operation, target, candidate))
                candidate.parameters.push_back(std::move(parameter));
            prediction.ranked.push_back(std::move(candidate));
        }
        // Origami's own rejections do not drop a tuned set: it follows the
        // ranked sets of its rank, without a latency.
        for(size_t index = 0; index < recipes.size(); ++index)
            if(recipes[index].fixed && !placed[index])
                addTuned(index, std::numeric_limits<double>::quiet_NaN());
        // Tuned seeds keep the knowledge's order; Origami's latency orders equal ranks.
        std::stable_sort(tuned.begin(), tuned.end(), [&](const auto& a, const auto& b) {
            return seeds[a.seed].rank < seeds[b.seed].rank;
        });
        prediction.ranked.insert(prediction.ranked.begin(),
                                 std::make_move_iterator(tuned.begin()),
                                 std::make_move_iterator(tuned.end()));
        return prediction;
    }

    namespace
    {
        class OrigamiPredictor final : public Predictor
        {
        public:
            std::string_view id() const noexcept override
            {
                return "origami";
            }
            std::set<std::string> modeledContracts() const override
            {
                return {dataParallelContract, persistentContract, tunedContract};
            }
            Status predict(const PredictionRequest& request,
                           const TuningKnowledge&   knowledge,
                           Prediction&              prediction) const override
            {
                prediction       = {};
                const auto* gemm = dynamic_cast<const GemmRequest*>(&request.request);
                if(!gemm)
                    return {Status::Code::NotSupported,
                            Stage::Predict,
                            "Origami does not model this operation"};
                try
                {
                    prediction = rankWithOrigami(request.request,
                                                 lowerForJit(*gemm),
                                                 request.target,
                                                 request.workspaceLimit,
                                                 knowledge);
                    return {};
                }
                catch(const std::bad_alloc&)
                {
                    throw;
                }
                catch(const std::exception& e)
                {
                    return {Status::Code::Failed, Stage::Predict, e.what()};
                }
            }
        };
    }

    std::shared_ptr<const Predictor> makeOrigamiPredictor()
    {
        return std::make_shared<const OrigamiPredictor>();
    }
}
