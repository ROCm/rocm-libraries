// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-predictor.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <origami/origami.hpp>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

namespace hipblaslt_ext::experimental
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
            int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            require(fd >= 0, "cannot create " + path + ": " + std::strerror(errno));
            size_t offset = 0;
            while(offset < contents.size())
            {
                auto count = write(fd, contents.data() + offset, contents.size() - offset);
                if(count < 0 && errno == EINTR)
                    continue;
                if(count <= 0)
                {
                    auto reason = std::string(std::strerror(errno));
                    close(fd);
                    throw std::runtime_error("Cannot write JIT request " + path + ": " + reason);
                }
                offset += size_t(count);
            }
            require(close(fd) == 0, "cannot close request " + path);
        }

        struct Candidate
        {
            std::array<size_t, 9> matrixInstruction;
            origami::config_t     config;
        };

        std::vector<Candidate> candidates(const origami::hardware_t& hardware,
                                          origami::data_type_t       dtype)
        {
            const auto recommended = hardware.get_recommended_matrix_instruction(dtype);
            require(recommended.m == 16 && recommended.n == 16 && recommended.k != 0,
                    "no supported square MFMA recommendation for the requested type");
            std::vector<origami::dim3_t> instructions{recommended};
            if(dtype == origami::data_type_t::Half && recommended.k != 16)
                instructions.push_back({16, 16, 16});

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
            const std::array<std::array<int, 2>, 3>     hints{{{0, 0}, {4, 0}, {0, 4}}};
            std::vector<Candidate>                      result;
            for(const auto& mi : instructions)
                for(const auto& shape : shapes)
                    for(size_t depth : {size_t(32), size_t(64)})
                        for(const auto& hint : hints)
                        {
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

    std::string predictJitGemmConfig(const TensileLite::ContractionProblemGemm& problem,
                                     const TensileLite::Hardware&               hardware,
                                     const GenerateOptions&                     options,
                                     JitGemmInfo&                               info)
    {
        namespace fs       = std::filesystem;
        using Type         = rocisa::DataType;
        const auto* device = dynamic_cast<const TensileLite::hip::HipAMDGPU*>(&hardware);
        require(device && device->analyticalHardware, "actual HIP device hardware is required");
        const auto& analytical = *device->analyticalHardware;
        require(analytical.arch == origami::hardware_t::architecture_t::gfx950,
                "the initial predictor supports gfx950");
        const auto type = problem.a().dataType();
        require((type == Type::Half || type == Type::Float) && problem.b().dataType() == type
                    && problem.c().dataType() == type && problem.d().dataType() == type
                    && problem.computeType() == Type::Float,
                "expected F32/F32 or FP16/F32 accumulation with matching A/B/C/D types");
        require(type != Type::Half || problem.highPrecisionAccumulate(),
                "FP16 requires high precision accumulation");
        require(problem.f32XdlMathOp() == Type::Float && problem.computeInputTypeA() == type
                    && problem.computeInputTypeB() == type,
                "special compute-input conversions are not supported by the initial predictor");
        require(problem.stridedBatched() && !problem.groupedGemm() && !problem.sparse()
                    && !problem.swizzleTensorA() && !problem.swizzleTensorB() && !problem.mxBlockA()
                    && !problem.mxBlockB() && !problem.useBias() && !problem.useE()
                    && !problem.outputAmaxD() && !problem.useGradient()
                    && !problem.useGateResidual() && problem.useScaleAB().empty()
                    && !problem.useScaleCD() && !problem.useScaleAlphaVec()
                    && problem.activationType() == TensileLite::ActivationType::None,
                "expected a dense strided GEMM with the default epilogue");
        require(problem.aOps().empty() && problem.bOps().empty() && problem.cOps().empty()
                    && problem.dOps().empty(),
                "tensor operations are unsupported");
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
        require(m && n && k && batch, "zero-sized GEMMs do not require JIT generation");

        origami::problem_t request;
        request.size        = {m, n, k};
        request.batch       = batch;
        request.num_cus     = problem.getParams().smCountTarget();
        request.a_transpose = transA ? origami::transpose_t::T : origami::transpose_t::N;
        request.b_transpose = transB ? origami::transpose_t::T : origami::transpose_t::N;
        request.a_dtype = request.b_dtype = request.c_dtype = request.d_dtype = request.mi_dtype
            = type == Type::Half ? origami::data_type_t::Half : origami::data_type_t::Float;
        const auto                     recipes = candidates(analytical, request.mi_dtype);
        std::vector<origami::config_t> configs;
        for(const auto& recipe : recipes)
            configs.push_back(recipe.config);
        const auto ranked
            = origami::rank_configs(request, analytical, configs, origami::model_t::gemm);
        const std::string output = fs::absolute(options.outputPath).string();
        require(!fs::exists(output) && !fs::exists(output + ".yaml")
                    && !fs::exists(output + ".prediction.json"),
                "output artifacts already exist");
        info.configPath = output + ".yaml";

        std::ostringstream json;
        json << std::setprecision(17) << std::boolalpha;
        json << "{\n\"schema_version\":1,\"model\":\"origami.gemm.estimation\","
             << "\"architecture\":" << jsonString(options.architecture)
             << ",\"problem\":{\"m\":" << m << ",\"n\":" << n << ",\"k\":" << k
             << ",\"batch\":" << batch << ",\"transpose_a\":" << transA
             << ",\"transpose_b\":" << transB
             << ",\"data_type\":" << jsonString(type == Type::Half ? "h" : "s")
             << ",\"high_precision_accumulate\":" << problem.highPrecisionAccumulate()
             << ",\"c_equals_d\":" << problem.cEqualsD() << ",\"num_cus\":" << request.num_cus;
        const std::array<const TensileLite::TensorDescriptor*, 4> tensors{
            &problem.a(), &problem.b(), &problem.c(), &problem.d()};
        for(size_t i = 0; i != tensors.size(); ++i)
        {
            json << ",\"strides_" << char('a' + i) << "\":";
            array(json, tensors[i]->strides());
        }
        json << "},\"hardware\":{\"device_id\":" << device->deviceId
             << ",\"cu_count\":" << analytical.N_CU << ",\"xcd_count\":" << analytical.NUM_XCD
             << ",\"compute_clock_ghz\":" << analytical.compute_clock_ghz
             << "},\"model_assumptions\":{\"occupancy\":1,\"stream_k\":0,"
                "\"workgroup_mapping\":\"estimated internally; Tensile uses its default\","
                "\"vector_widths\":\"Origami defaults; Tensile derives actual widths\"},"
                "\"candidates\":[";
        size_t count = 0;
        for(const auto& result : ranked)
        {
            if(!std::isfinite(result.latency) || result.latency <= 0
               || result.latency == std::numeric_limits<double>::max())
                continue;
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
        require(count != 0, "Origami rejected every parameter candidate");
        json << "]}\n";
        const auto requestPath = output + ".request.json";
        writeFresh(requestPath, json.str());
        info.prediction = "Origami ranked " + std::to_string(count)
                          + " parameter candidates; Tensile validation pending";
        return requestPath;
    }
}
