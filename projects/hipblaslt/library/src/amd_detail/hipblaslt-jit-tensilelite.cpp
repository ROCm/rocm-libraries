// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-process.hpp"
#include "hipblaslt-jit-tensilelite-artifacts.hpp"
#include "hipblaslt-jit-tensilelite.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt-functions.h"
#include "rocblaslt.h"
#include "tensile_host.hpp"
#include "utility.hpp"
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace hipblaslt_ext::experimental::jit::tensilelite
{
    namespace
    {
        namespace fs = std::filesystem;
        using Tree   = std::map<std::string, std::string>;
        using Master = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

        using detail::artifacts::artifact;
        using detail::artifacts::field;
        using detail::artifacts::readArtifact;
        using detail::artifacts::readEnvelope;
        using detail::artifacts::readLibrary;

        bool targetMatchesDevice(const std::string& target, const std::string& device)
        {
            const auto targetColon = target.find(':');
            if(target.substr(0, targetColon) != device.substr(0, device.find(':')))
                return false;
            std::set<std::string> deviceFeatures;
            std::istringstream    deviceParts(device);
            std::string           part;
            std::getline(deviceParts, part, ':');
            while(std::getline(deviceParts, part, ':'))
                deviceFeatures.insert(part);
            std::istringstream targetParts(target);
            std::getline(targetParts, part, ':');
            while(std::getline(targetParts, part, ':'))
                if(deviceFeatures.count(part) == 0)
                    return false;
            return true;
        }

        void runGenerator(const Options& options, const char* module)
        {
            require(!options.pythonExecutable.empty() && !options.tensileSourceDirectory.empty()
                        && !options.configPath.empty() && !options.outputPath.empty()
                        && !options.architecture.empty(),
                    "Missing generation option");
            require(fs::is_directory(fs::u8path(options.tensileSourceDirectory)),
                    "Invalid Tensile source directory");
            require(!fs::exists(fs::u8path(options.outputPath)), "Output path already exists");
            auto toolPath = [](const std::string& tool) {
                return fs::u8path(tool).has_parent_path()
                           ? fs::absolute(fs::u8path(tool)).u8string()
                           : tool;
            };
            std::vector<std::string> args
                = {toolPath(options.pythonExecutable),
                   "-m",
                   module,
                   fs::absolute(fs::u8path(options.configPath)).u8string(),
                   fs::absolute(fs::u8path(options.outputPath)).u8string(),
                   "--architecture",
                   options.architecture,
                   "--cxx-compiler",
                   toolPath(options.cxxCompiler),
                   "--offload-bundler",
                   toolPath(options.offloadBundler),
                   "--library-format",
#ifdef TENSILE_YAML
                   "yaml"
#else
                   "msgpack"
#endif
                };
            auto cwd = fs::absolute(fs::u8path(options.outputPath));
            cwd += ".cwd";
            require(fs::create_directory(cwd), "Generator working directory already exists");
#ifdef _WIN32
            const char separator = ';';
#else
            const char separator = ':';
#endif
            hipblaslt_jit::process::Request request;
            request.argv             = std::move(args);
            request.workingDirectory = cwd;
            request.logPath          = fs::absolute(fs::u8path(options.outputPath));
            request.logPath += ".log";
            request.environment.emplace_back(
                "PYTHONPATH",
                fs::absolute(fs::u8path(options.tensileSourceDirectory)).u8string()
                    + (options.pythonPath.empty() ? "" : separator + options.pythonPath));
            const auto result = hipblaslt_jit::process::run(request);
            if(!result.succeeded())
            {
                std::ifstream log(request.logPath);
                std::string   line, detail = result.error;
                while(std::getline(log, line))
                    if(line.find("JIT GEMM build failed:") == 0
                       || line.find("Single-solution build failed:") == 0)
                        detail = line.substr(0, 8192);
                throw std::runtime_error("TensileLite generator: " + detail + "; see "
                                         + request.logPath.u8string());
            }
        }

        void requireRequest(bool condition, const std::string& message)
        {
            require(condition, "JIT GEMM prediction: " + message);
        }

        void writeFresh(const std::string& path, const std::string& contents)
        {
            const auto native = fs::u8path(path);
#ifdef _WIN32
            FILE* file = _wfopen(native.c_str(), L"wbx");
#else
            FILE* file = std::fopen(native.c_str(), "wbx");
#endif
            requireRequest(file, "cannot create " + path + ": " + std::strerror(errno));
            const bool written
                = std::fwrite(contents.data(), 1, contents.size(), file) == contents.size();
            const int closed = std::fclose(file);
            requireRequest(written && closed == 0, "cannot write request " + path);
        }

        // Writes <output>.request.json, the Tensile.JitGemm input, and returns its path.
        std::string writeJitGemmRequest(const jit::detail::GemmRequest& request,
                                        const Options&                  options,
                                        const hipblaslt_jit::Prediction& prediction)
        {
            namespace json     = hipblaslt_jit::json;
            const auto problem = hipblaslt_jit::lowerForJit(request);
            const auto gemm    = hipblaslt_jit::canonicalGemm(problem);
            const std::string output = fs::absolute(fs::u8path(options.outputPath)).u8string();
            requireRequest(!fs::exists(fs::u8path(output))
                               && !fs::exists(fs::u8path(output + ".yaml"))
                               && !fs::exists(fs::u8path(output + ".prediction.json")),
                           "output artifacts already exist");
            const auto members = [](const std::vector<hipblaslt_jit::TuningParameter>& values) {
                json::Members result;
                for(const auto& value : values)
                    result.emplace_back(value.name, value.json);
                return json::object(result);
            };
            std::ostringstream out;
            out << std::setprecision(17) << std::boolalpha;
            out << "{\n\"schema_version\":1,\"modeled_contract\":" << json::quote(prediction.modeledContract)
                << ",\"model\":" << json::quote(prediction.model)
                << ",\"architecture\":" << json::quote(options.architecture)
                << ",\"problem_type\":" << json::object(hipblaslt_jit::problemTypeFields(problem))
                << ",\"problem\":{\"m\":" << gemm.m << ",\"n\":" << gemm.n << ",\"k\":" << gemm.k
                << ",\"batch\":" << gemm.batch << ",\"transpose_a\":" << gemm.transA
                << ",\"transpose_b\":" << gemm.transB << ",\"c_equals_d\":" << problem.cEqualsD()
                << ",\"num_cus\":" << static_cast<size_t>(problem.getParams().smCountTarget());
            if(problem.mxBlockA() || problem.mxBlockB())
                out << ",\"scale_mode_a\":"
                    << json::quote(rocblaslt_scaling_format_to_string(request.problem.scaleAType))
                    << ",\"scale_mode_b\":"
                    << json::quote(rocblaslt_scaling_format_to_string(request.problem.scaleBType));
            const std::array<const TensileLite::TensorDescriptor*, 4> tensors{
                &problem.a(), &problem.b(), &problem.c(), &problem.d()};
            for(size_t i = 0; i != tensors.size(); ++i)
                out << ",\"strides_" << char('a' + i) << "\":" << json::array(tensors[i]->strides())
                    << ",\"sizes_" << char('a' + i) << "\":" << json::array(tensors[i]->sizes());
            for(const auto& entry : {std::make_pair("mxsa", &problem.mxsa()),
                                     std::make_pair("mxsb", &problem.mxsb())})
            {
                if(entry.second->empty())
                    continue;
                out << ",\"strides_" << entry.first << "\":"
                    << json::array(entry.second->strides()) << ",\"sizes_" << entry.first
                    << "\":" << json::array(entry.second->sizes());
            }
            out << "},\"hardware\":" << members(prediction.hardware)
                << ",\"model_assumptions\":" << members(prediction.assumptions) << ",\"candidates\":[";
            for(size_t i = 0; i != prediction.ranked.size(); ++i)
            {
                const auto& candidate = prediction.ranked[i];
                out << (i ? "," : "") << "{\"id\":" << candidate.id
                    << ",\"predicted_cycles\":" << candidate.predictedCycles
                    << ",\"parameters\":" << members(candidate.parameters)
                    << ",\"modeled\":" << members(candidate.modeled) << '}';
            }
            out << "]}\n";
            const auto path = output + ".request.json";
            writeFresh(path, out.str());
            return path;
        }
    }

    namespace
    {
        // Reads the TLJIT001 bundle the generator wrote as a library entry plus
        // prebuilt code objects.
        hipblaslt_jit::GeneratedSolution readGeneratedSolution(const Options&     options,
                                                               const std::string& deviceTarget)
        {
            using hipblaslt_jit::BuildUnit;
            hipblaslt_jit::GeneratedSolution result;
            const auto bundle = fs::canonical(fs::u8path(options.outputPath) / "bundle");
            std::vector<std::string> objectList;
            const auto               tree = readEnvelope(bundle / "loader.bin", objectList);
            require(field(tree, "schema_version") == "2" && field(tree, "counts.solutions") == "1"
                        && field(tree, "counts.main_kernels") == "1"
                        && field(tree, "solution.index") == "0",
                    "Unsupported single-solution manifest schema/counts/index");
            result.kernelName = field(tree, "main_kernel.name");
            require(result.kernelName == field(tree, "solution.kernel_name"),
                    "Manifest kernel names disagree");
            require(field(tree, "architecture.requested") == options.architecture,
                    "Bundle architecture differs from request");
            require(targetMatchesDevice(field(tree, "architecture.resolved"), deviceTarget)
                        && field(tree, "architecture.compiler_target")
                               == field(tree, "architecture.resolved"),
                    "Bundle architecture does not match device");
#ifdef TENSILE_YAML
            require(field(tree, "library.format") == "yaml", "Expected YAML library");
#else
            require(field(tree, "library.format") == "msgpack", "Expected MessagePack library");
#endif
            const auto codeObject = artifact(bundle, field(tree, "main_kernel.code_object"));
            std::set<fs::path> codeObjects;
            for(const auto& item : objectList)
                require(codeObjects.insert(artifact(bundle, item)).second,
                        "Duplicate code object artifact");
            require(codeObjects.count(codeObject) == 1,
                    "Main code object missing from code_objects");
            const auto physicalLibrary = artifact(bundle, field(tree, "library.path"));
            const auto logicalLibrary
                = artifact(bundle, field(tree, "library.logical_path"), false);
            require(physicalLibrary == logicalLibrary
                        || physicalLibrary == fs::path(logicalLibrary).concat(".zlib"),
                    "Manifest physical and logical library paths disagree");
            result.entry = readLibrary(physicalLibrary);
            const auto library
                = std::dynamic_pointer_cast<Master>(
                    TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                        result.entry));
            require(library && library->solutions.size() == 1 && library->solutions.count(0),
                    "Expected a non-lazy library containing only local solution 0");
            auto solution = library->solutions.at(0);
            require(solution && solution->index == 0 && solution->kernelName == result.kernelName
                        && solution->solutionName == field(tree, "solution.name"),
                    "Library solution identity does not match manifest");
            for(const auto& path : codeObjects)
                result.units.push_back({path == codeObject ? BuildUnit::Role::Main
                                                           : BuildUnit::Role::Helper,
                                        path.filename().u8string(),
                                        readArtifact(path)});
            return result;
        }

        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;

        // Runs the TensileLite generator and returns its bundle; it never loads code.
        class TensileLiteBackend final : public hipblaslt_jit::Backend
        {
        public:
            explicit TensileLiteBackend(const Options& options)
                : m_options(options)
                , m_info{"tensilelite",
                         "TensileLite",
                         options.configPath.empty() ? "origami.gemm.dp.v1" : ""}
            {
            }
            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }
            Status generate(const hipblaslt_jit::GenerationRequest&        request,
                            std::vector<hipblaslt_jit::GeneratedSolution>& solutions) const override
            {
                solutions.clear();
                const auto* gemm = dynamic_cast<const jit::detail::GemmRequest*>(&request.request);
                if(!gemm || request.request.kind() != jit::detail::GemmRequest::operation)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "TensileLite does not implement this operation"};
                auto configured = m_options;
                if(configured.architecture.empty())
                    configured.architecture = request.target.targetId;
                if(!targetMatchesDevice(configured.architecture, request.target.targetId))
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "Requested TensileLite architecture does not match device"};
                try
                {
                    std::string summary;
                    if(request.prediction)
                    {
                        configured.configPath
                            = writeJitGemmRequest(*gemm, configured, *request.prediction);
                        summary = "Origami ranked "
                                  + std::to_string(request.prediction->ranked.size())
                                  + " parameter candidates; the first candidate accepted by "
                                    "TensileLite was compiled";
                    }
                    runGenerator(configured,
                                 request.prediction ? "Tensile.JitGemm" : "Tensile.SingleSolution");
                    solutions.push_back(readGeneratedSolution(configured, request.target.targetId));
                    return {Status::Code::Success, Stage::Generate, std::move(summary)};
                }
                catch(const std::bad_alloc&)
                {
                    throw;
                }
                catch(const std::exception& e)
                {
                    Status failure{Status::Code::Failed, Stage::Generate, e.what()};
                    const auto log = fs::u8path(configured.outputPath + ".log");
                    std::error_code error;
                    if(fs::exists(log, error))
                        failure.logPath = fs::absolute(log, error).u8string();
                    return failure;
                }
            }

        private:
            Options                    m_options;
            hipblaslt_jit::BackendInfo m_info;
        };
    }

    hipblasStatus_t
        createBackend(const Options& options, jit::Backend& backend, jit::Diagnostics& diagnostics)
    {
        backend     = {};
        diagnostics = {"TensileLite", ""};
        if(options.pythonExecutable.empty() || options.tensileSourceDirectory.empty()
           || options.outputPath.empty())
        {
            diagnostics.message = "TensileLite requires Python, source and output paths";
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        try
        {
            backend = jit::detail::BackendAccess::make(
                std::make_shared<const hipblaslt_jit::Jit>(hipblaslt_jit::Jit::Components{
                    std::make_shared<const TensileLiteBackend>(options),
                    hipblaslt_jit::makeOrigamiPredictor(),
                    hipblaslt_jit::makeTensileLiteDefaults(),
                    hipblaslt_jit::makePrebuiltBuilder(),
                    hipblaslt_jit::makeTensileLoader(),
                    nullptr}));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate TensileLite backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
    }
}
