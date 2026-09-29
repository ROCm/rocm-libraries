// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
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

        struct Recipe
        {
            std::array<size_t, 9> matrixInstruction;
            origami::config_t     config;
        };

        std::vector<Recipe> candidates(const origami::hardware_t&        hardware,
                                       origami::data_type_t              dtype,
                                       const std::vector<CandidateSeed>& seeds)
        {
            // Use the target's instruction catalog for every datatype. Tensile
            // subsequently validates the complete instruction and tile combination.
            auto instructions = hardware.get_valid_matrix_instructions(dtype);
            std::sort(instructions.begin(), instructions.end(), [](const auto& a, const auto& b) {
                return std::array{a.m, a.n, a.k} < std::array{b.m, b.n, b.k};
            });
            std::vector<Recipe> result;
            for(const auto& mi : instructions)
                for(const auto& seed : seeds)
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
                            auto& config                = candidate.config;
                            config.mt                   = {shape[0], shape[1], depth};
                            config.mi                   = mi;
                            config.occupancy = 1; // Conservative model input, not WG dimensions.
                            config.stream_k  = 0; // Caller-selected data-parallel candidate domain.
                            config.cache_hints_a   = hint[0];
                            config.cache_hints_b   = hint[1];
                            config.prediction_mode = origami::prediction_modes_t::estimation;
                            config.target          = origami::target_t::tensilelite;
                            config.index           = result.size();
                            result.push_back(candidate);
                        }
            return result;
        }
    }

    static Prediction rank(const OperationRequest&                    operation,
                           const TensileLite::ContractionProblemGemm& problem,
                           const DeviceTarget&                        target,
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
        const auto                     recipes = m && n && k
                                     && problem.computeInputTypeA() == problem.computeInputTypeB()
                                     && !problem.sparse()
                                                     ? candidates(analytical, request.mi_dtype,
                                                                  knowledge.seeds(operation, target))
                                                     : std::vector<Recipe>{};
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
        require(!ranked.empty(),
                "No Origami ranking: no finite positive-latency candidates for this request");

        Prediction prediction;
        prediction.modeledContract = "origami.gemm.dp.v1";
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
            {"stream_k", "0"},
            {"stream_k_origin",
             quote("caller-selected data-parallel domain; Origami does not select enablement")},
            {"workgroup_mapping",
             quote("select_workgroup_mapping; unsupported transport rejects the candidate")},
            {"stagger", quote("select_staggerU; all three outputs including zeros are preserved")},
            {"vector_widths", quote("not predicted by estimation; Tensile derives actual widths")},
            {"epilogue",
             quote("bias, activation, auxiliary outputs and scaling overhead are not modeled")},
            {"architecture_constants",
             quote(analytical.arch == Arch::gfx1250
                       ? "Origami gfx1250 provisional model: upstream memory constants "
                         "reuse gfx950 with gfx1250 overrides; not calibrated for gfx1250"
                       : "Origami native architecture model")},
        };
        for(const auto& result : ranked)
        {
            require(result.config.index < recipes.size(), "Origami returned an unknown candidate");
            const auto& recipe = recipes[result.config.index];
            const auto& config = result.config;
            const auto [reduction, grid, activeCUs, timesteps, split]
                = origami::gemm::compute_launch_parameters(request, analytical, config,
                                                          config.grid_selection);
            require(config.stream_k == 0 && reduction == origami::reduction_t::none && split == 1,
                    "Origami returned a launch outside the data-parallel candidate domain");
            const auto mapping = origami::select_workgroup_mapping(request, analytical, config, grid);
            require(mapping.wgm != 0, "Origami returned a zero workgroup mapping");
            const auto stagger = origami::select_staggerU(request, analytical, config, grid, mapping.wgm);
            Candidate candidate;
            candidate.id              = static_cast<uint32_t>(config.index);
            candidate.predictedCycles = result.latency;
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
                {"launch",
                 json::object({{"stream_k", "0"},
                               {"reduction", quote("none")},
                               {"grid", literal(grid)},
                               {"active_cus", literal(activeCUs)},
                               {"timesteps", literal(timesteps)},
                               {"split_factor", literal(split)}})},
            };
            for(auto& parameter : knowledge.defaults(operation, target, candidate))
                candidate.parameters.push_back(std::move(parameter));
            prediction.ranked.push_back(std::move(candidate));
        }
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
            std::string_view modeledContract() const noexcept override
            {
                return "origami.gemm.dp.v1";
            }
            Status predict(const OperationRequest& request,
                           const DeviceTarget&     target,
                           const TuningKnowledge&  knowledge,
                           Prediction&             prediction) const override
            {
                prediction       = {};
                const auto* gemm = dynamic_cast<const GemmRequest*>(&request);
                if(!gemm)
                    return {Status::Code::NotSupported,
                            Stage::Predict,
                            "Origami does not model this operation"};
                try
                {
                    prediction = rank(request, lowerForJit(*gemm), target, knowledge);
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
