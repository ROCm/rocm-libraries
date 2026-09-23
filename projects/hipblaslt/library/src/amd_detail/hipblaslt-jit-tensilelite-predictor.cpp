// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-tensilelite-predictor.hpp"
#include <Tensile/UtilsOrigami.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <origami/origami.hpp>
#include <sstream>
#include <stdexcept>

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    namespace
    {
        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error("JIT GEMM prediction: " + message);
        }

        std::string jsonString(const std::string& value)
        {
            std::ostringstream out;
            out << '"';
            for(unsigned char c : value)
            {
                if(c == '"' || c == '\\')
                    out << '\\' << c;
                else if(c < 0x20)
                    out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(c);
                else
                    out << c;
            }
            out << '"';
            return out.str();
        }

        template <typename Values>
        void array(std::ostream& out, const Values& values)
        {
            out << '[';
            bool first = true;
            for(const auto value : values)
            {
                if(!first)
                    out << ',';
                out << value;
                first = false;
            }
            out << ']';
        }

        void writeFresh(const std::string& path, const std::string& contents)
        {
            const auto native = std::filesystem::u8path(path);
#ifdef _WIN32
            FILE* file = _wfopen(native.c_str(), L"wbx");
#else
            FILE* file = std::fopen(native.c_str(), "wbx");
#endif
            require(file, "cannot create " + path + ": " + std::strerror(errno));
            const bool written
                = std::fwrite(contents.data(), 1, contents.size(), file) == contents.size();
            const int closed = std::fclose(file);
            require(written && closed == 0, "cannot write request " + path);
        }

        struct Candidate
        {
            std::array<size_t, 9> matrixInstruction;
            origami::config_t     config;
        };

        std::vector<Candidate> candidates(const origami::hardware_t& hardware,
                                          origami::data_type_t       dtype)
        {
            // Use the target's instruction catalog for every datatype. Tensile
            // subsequently validates the complete instruction and tile combination.
            auto instructions = hardware.get_valid_matrix_instructions(dtype);
            std::sort(instructions.begin(), instructions.end(), [](const auto& a, const auto& b) {
                return std::array{a.m, a.n, a.k} < std::array{b.m, b.n, b.k};
            });

            // Declarative parameter shapes, independent of the installed solution library.
            // Each shape retains its wave topology; MT/MI alone cannot reconstruct it.
            const std::array<std::array<size_t, 4>, 11> shapes{{
                {32, 32, 2, 2},
                {64, 32, 2, 2},
                {32, 64, 2, 2},
                {64, 64, 2, 2},
                {128, 64, 2, 2},
                {64, 128, 2, 2},
                {128, 128, 2, 2},
                {256, 128, 2, 2},
                {128, 256, 2, 2},
                {256, 256, 2, 2},
                {128, 16, 4, 1},
            }};
            // gfx90a has no NT modifier; gfx1250 expresses NT via TemporalHint.
            // Keep both at default hints so the model matches emitted loads.
            std::vector<std::array<int, 2>> hints{{0, 0}};
            if(hardware.arch != origami::hardware_t::architecture_t::gfx90a
               && hardware.arch != origami::hardware_t::architecture_t::gfx1250)
            {
                hints.push_back({4, 0});
                hints.push_back({0, 4});
            }
            std::vector<Candidate> result;
            for(const auto& mi : instructions)
                for(const auto& shape : shapes)
                    for(size_t depth : {std::max(size_t(32), mi.k), std::max(size_t(64), 2 * mi.k)})
                        for(const auto& hint : hints)
                        {
                            if(!mi.m || !mi.n || !mi.k || shape[0] % (mi.m * shape[2])
                               || shape[1] % (mi.n * shape[3]) || depth % mi.k)
                                continue;
                            Candidate candidate;
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
                            config.stream_k  = 0; // Match the canonical Tensile default.
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

    PredictionPlan predictGemmPlan(const TensileLite::ContractionProblemGemm& problem,
                                   const TensileLite::Hardware&               hardware,
                                   const Options&                             options,
                                   const std::string&                         scaleModeA,
                                   const std::string&                         scaleModeB)
    {
        namespace fs       = std::filesystem;
        using Type         = rocisa::DataType;
        const auto* device = dynamic_cast<const TensileLite::hip::HipAMDGPU*>(&hardware);
        require(device && device->analyticalHardware, "actual HIP device hardware is required");
        const auto& analytical = *device->analyticalHardware;
        using Arch             = origami::hardware_t::architecture_t;
        require(analytical.N_CU && analytical.NUM_XCD && analytical.lds_capacity
                    && analytical.rf_capacity && analytical.compute_clock_ghz > 0,
                "actual device resource limits are unavailable");
        require(problem.stridedBatched() && !problem.groupedGemm(),
                "prediction requires a single strided GEMM; grouped GEMM is not implemented");
        require(problem.c().dataType() == problem.d().dataType(),
                "Tensile ProblemType requires matching C and D datatypes");
        const auto conjugate = [](const TensileLite::TensorOps& ops) {
            require(ops.empty()
                        || (ops.size() == 1
                            && ops.front().type == TensileLite::TensorOp::Type::ComplexConjugate),
                    "the input tensor operation cannot be represented by Tensile ProblemType");
            return !ops.empty();
        };
        const bool conjugateA = conjugate(problem.aOps());
        const bool conjugateB = conjugate(problem.bOps());
        require(problem.cOps().empty() && problem.dOps().empty(),
                "Tensile ProblemType does not describe C/D tensor operations");
        require(problem.freeIndicesA().size() == 1 && problem.freeIndicesB().size() == 1
                    && problem.boundIndices().size() == 1 && problem.batchIndices().size() == 1
                    && !problem.transposeC01(),
                "expected canonical batched GEMM indices");
        for(const auto* tensor : {&problem.a(), &problem.b(), &problem.c(), &problem.d()})
            require(tensor->dimensions() == 3 && tensor->strides().at(0) == 1,
                    "expected three-dimensional column-major tensor descriptors");
        const bool   transA = problem.freeIndicesA()[0].i == 1;
        const bool   transB = problem.freeIndicesB()[0].i == 0;
        const size_t m = problem.freeSizeA(0), n = problem.freeSizeB(0);
        const size_t k = problem.boundSize(0), batch = problem.batchSize(0);
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
                                                     ? candidates(analytical, request.mi_dtype)
                                                     : std::vector<Candidate>{};
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
        PredictionPlan    plan;
        const std::string output = fs::absolute(fs::u8path(options.outputPath)).u8string();
        require(!fs::exists(fs::u8path(output)) && !fs::exists(fs::u8path(output + ".yaml"))
                    && !fs::exists(fs::u8path(output + ".prediction.json")),
                "output artifacts already exist");

        std::ostringstream json;
        json << std::setprecision(17) << std::boolalpha;
        json << "{\n\"schema_version\":1,\"model\":" << jsonString("origami.gemm.estimation")
             << ",\"architecture\":" << jsonString(options.architecture)
             << ",\"problem_type\":{\"OperationType\":\"GEMM\",\"Batched\":true,"
                "\"StridedBatched\":true,\"TransposeA\":"
             << transA << ",\"TransposeB\":" << transB << ",\"ComplexConjugateA\":" << conjugateA
             << ",\"ComplexConjugateB\":" << conjugateB;
        const auto dataType = [&](const char* key, Type value) {
            json << "," << jsonString(key) << ':' << static_cast<int>(value);
        };
        dataType("DataType", problem.computeInputTypeA());
        dataType("DataTypeA", problem.a().dataType());
        dataType("DataTypeB", problem.b().dataType());
        dataType("MacDataTypeA", problem.computeInputTypeA());
        dataType("MacDataTypeB", problem.computeInputTypeB());
        dataType("DestDataType", problem.d().dataType());
        dataType("ComputeDataType", problem.computeType());
        dataType("F32XdlMathOp", problem.f32XdlMathOp());
        if(problem.mxBlockA())
            dataType("DataTypeMXSA", problem.mxTypeA());
        if(problem.mxBlockB())
            dataType("DataTypeMXSB", problem.mxTypeB());
        json << ",\"HighPrecisionAccumulate\":" << problem.highPrecisionAccumulate()
             << ",\"UseBias\":" << problem.useBias() << ",\"UseE\":" << problem.useE()
             << ",\"Gradient\":" << problem.useGradient()
             << ",\"UseScaleAB\":" << jsonString(problem.useScaleAB())
             << ",\"UseScaleCD\":" << problem.useScaleCD()
             << ",\"UseScaleAlphaVec\":" << problem.useScaleAlphaVec()
             << ",\"OutputAmaxD\":" << problem.outputAmaxD()
             << ",\"MXBlockA\":" << problem.mxBlockA() << ",\"MXBlockB\":" << problem.mxBlockB()
             << ",\"Sparse\":" << problem.sparse()
             << ",\"SwizzleTensorA\":" << problem.swizzleTensorA()
             << ",\"SwizzleTensorB\":" << problem.swizzleTensorB()
             << ",\"UseGateResidual\":" << problem.useGateResidual();
        if(problem.useBias())
            json << ",\"BiasDataTypeList\":[" << static_cast<int>(problem.bias().dataType())
                 << "],\"BiasSrc\":" << jsonString(std::string(1, 'A' + problem.biasSrc()));
        if(problem.useGateResidual())
            json << ",\"GateResidualDataTypeList\":["
                 << static_cast<int>(problem.gateResidual().dataType()) << ']';
        if(problem.useE())
            dataType("DataTypeE", problem.e().dataType());
        if(problem.outputAmaxD())
            dataType("DataTypeAmaxD", problem.amaxd().dataType());
        if(problem.activationType() != TensileLite::ActivationType::None)
        {
            json << ",\"Activation\":true,\"ActivationType\":\"hipblaslt_all\"";
            dataType("ActivationComputeDataType", problem.activationComputeType());
        }
        json << "},\"problem\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
             << ",\"batch\":" << batch << ",\"transpose_a\":" << transA
             << ",\"transpose_b\":" << transB << ",\"c_equals_d\":" << problem.cEqualsD()
             << ",\"num_cus\":" << request.num_cus;
        if(problem.mxBlockA() || problem.mxBlockB())
            json << ",\"scale_mode_a\":" << jsonString(scaleModeA)
                 << ",\"scale_mode_b\":" << jsonString(scaleModeB);
        const std::array<const TensileLite::TensorDescriptor*, 4> tensors{
            &problem.a(), &problem.b(), &problem.c(), &problem.d()};
        for(size_t i = 0; i != tensors.size(); ++i)
        {
            json << ",\"strides_" << char('a' + i) << "\":";
            array(json, tensors[i]->strides());
            json << ",\"sizes_" << char('a' + i) << "\":";
            array(json, tensors[i]->sizes());
        }
        for(const auto& entry :
            {std::make_pair("mxsa", &problem.mxsa()), std::make_pair("mxsb", &problem.mxsb())})
        {
            if(entry.second->empty())
                continue;
            json << ",\"strides_" << entry.first << "\":";
            array(json, entry.second->strides());
            json << ",\"sizes_" << entry.first << "\":";
            array(json, entry.second->sizes());
        }
        json << "},\"hardware\":{\"device_id\":" << device->deviceId
             << ",\"cu_count\":" << analytical.N_CU << ",\"xcd_count\":" << analytical.NUM_XCD
             << ",\"compute_clock_ghz\":" << analytical.compute_clock_ghz
             << ",\"lds_capacity\":" << analytical.lds_capacity
             << ",\"rf_capacity\":" << analytical.rf_capacity
             << "},\"model_assumptions\":{\"occupancy\":1,\"stream_k\":0,"
                "\"workgroup_mapping\":\"estimated internally; Tensile uses its default\","
                "\"vector_widths\":\"Origami defaults; Tensile derives actual widths\","
                "\"epilogue\":\"bias, activation, auxiliary outputs and scaling overhead are not "
                "modeled\","
                "\"architecture_constants\":"
             << jsonString(analytical.arch == Arch::gfx1250
                               ? "Origami gfx1250 provisional model: upstream memory constants "
                                 "reuse gfx950 with gfx1250 overrides; not calibrated for gfx1250"
                               : "Origami native architecture model")
             << "},\"candidates\":[";
        size_t count = 0;
        for(const auto& result : ranked)
        {
            require(result.config.index < recipes.size(), "Origami returned an unknown candidate");
            const auto& recipe = recipes[result.config.index];
            if(count++)
                json << ',';
            json << "{\"id\":" << result.config.index << ",\"predicted_cycles\":" << result.latency
                 << ",\"parameters\":{\"MatrixInstruction\":";
            array(json, recipe.matrixInstruction);
            json << ",\"DepthU\":" << recipe.config.mt.k
                 << ",\"NonTemporalA\":" << recipe.config.cache_hints_a
                 << ",\"NonTemporalB\":" << recipe.config.cache_hints_b << "}}";
        }
        json << "]}\n";
        const auto requestPath = output + ".request.json";
        writeFresh(requestPath, json.str());
        plan.requestPath = requestPath;
        plan.summary
            = "Origami ranked " + std::to_string(count)
              + " parameter candidates; the first candidate accepted by TensileLite was compiled";
        return plan;
    }
}
