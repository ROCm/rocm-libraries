// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-process.hpp"
#include "hipblaslt-jit-tensilelite.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt-functions.h"
#include "rocblaslt.h"
#include "rocblaslt_secure_env.hpp"
#include "tensile_host.hpp"
#include "utility.hpp"
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
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace hipblaslt_ext::experimental::jit::tensilelite
{
    namespace
    {
        namespace fs = std::filesystem;

        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

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

        void runGenerator(const Options& options, const char* module, int codeObjectVersion)
        {
            namespace debug = hipblaslt_jit::debug;
            debug::Phase prepare("child_prepare");
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
                   "--source-only",
                   "--code-object-version",
                   std::to_string(codeObjectVersion),
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
            // The generator writes events.jsonl and timing.json there; the
            // scratch keeps them after a failure.
            const auto categories = debug::childCategories();
            const auto debugDir   = cwd / "jit-debug";
            if(!categories.empty())
            {
                debug::set("module", debug::Line::quote(module));
                request.argv.insert(request.argv.end(),
                                    {"--debug", categories, "--debug-dir", debugDir.u8string()});
            }
            prepare.stop();
            std::optional<debug::ChildObserver> observer;
            if(debug::on(debug::Progress))
            {
                debug::Line(debug::Progress, "child.start")
                    .add("module", module)
                    .add("log", request.logPath.u8string())
                    .write();
                try
                {
                    observer.emplace(debugDir / "events.jsonl");
                }
                catch(const std::system_error&)
                {
                }
            }
            debug::Phase child("child");
            const auto   result = hipblaslt_jit::process::run(request);
            child.stop();
            if(observer)
                observer->stop();
            if(!categories.empty())
            {
                const auto* record     = debug::innermost();
                const auto  childNanos = record ? record->nanoseconds("child") : 0;
                debug::set("exit",
                           "{\"started\":" + std::string(result.started ? "true" : "false")
                               + ",\"code\":" + std::to_string(result.exitCode)
                               + ",\"signal\":" + std::to_string(result.terminationSignal) + "}");
                if(debug::on(debug::Timing))
                {
                    std::string why;
                    auto timing = debug::childTiming(debugDir / "timing.json", childNanos, why);
                    if(why.empty())
                        debug::set("child", std::move(timing));
                    else
                        debug::set("child_timing", debug::Line::quote(why));
                }
                if(observer)
                {
                    debug::Line line(debug::Progress, "child.exit");
                    line.add("started", result.started)
                        .add("code", result.exitCode)
                        .add("signal", result.terminationSignal)
                        .add("events", observer->events())
                        .add("dropped", observer->dropped());
                    if(debug::on(debug::Timing))
                        line.json("ns", "{\"child\":" + std::to_string(childNanos) + "}");
                    line.write();
                }
            }
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
        std::string writeJitGemmRequest(const jit::detail::GemmRequest&  request,
                                        const Options&                   options,
                                        const hipblaslt_jit::Prediction& prediction,
                                        size_t                           count,
                                        const std::vector<std::string>&  excludeKernels)
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
            // Heuristic queries have no buffers, so C and D compare equal even when their
            // layouts differ; Tensile rejects C equal to D unless the strides match.
            const bool cEqualsD
                = problem.cEqualsD() && problem.c().strides() == problem.d().strides();
            std::ostringstream out;
            out << std::setprecision(17) << std::boolalpha;
            out << "{\n\"schema_version\":1,\"modeled_contract\":" << json::quote(prediction.modeledContract)
                << ",\"model\":" << json::quote(prediction.model)
                << ",\"architecture\":" << json::quote(options.architecture)
                << ",\"problem_type\":" << json::object(hipblaslt_jit::problemTypeFields(problem))
                << ",\"problem\":{\"m\":" << gemm.m << ",\"n\":" << gemm.n << ",\"k\":" << gemm.k
                << ",\"batch\":" << gemm.batch << ",\"transpose_a\":" << gemm.transA
                << ",\"transpose_b\":" << gemm.transB << ",\"c_equals_d\":" << cEqualsD
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
            out << ']';
            count = std::min(count, prediction.ranked.size());
            if(count > 1)
                out << ",\"requested_solutions\":" << count;
            if(!excludeKernels.empty())
            {
                out << ",\"exclude_kernel_names\":[";
                for(size_t i = 0; i != excludeKernels.size(); ++i)
                    out << (i ? "," : "") << json::quote(excludeKernels[i]);
                out << ']';
            }
            out << "}\n";
            const auto path = output + ".request.json";
            writeFresh(path, out.str());
            return path;
        }
    }

    namespace
    {
        // Reads the generator's source bundles, best first: <output>/bundle-<rank>
        // from Tensile.JitGemm, or <output>/bundle from Tensile.SingleSolution.
        std::vector<hipblaslt_jit::GeneratedSolution>
            readGeneratedSolutions(const Options&                  options,
                                   bool                            ranked,
                                   size_t                          count,
                                   const std::vector<std::string>& excludeKernels)
        {
            const auto            output = fs::u8path(options.outputPath);
            std::vector<fs::path> bundles;
            if(!ranked)
                bundles.push_back(output / "bundle");
            for(size_t rank = 0; ranked && rank < count; ++rank)
            {
                auto bundle = output / ("bundle-" + std::to_string(rank));
                if(!fs::exists(bundle))
                    break;
                bundles.push_back(std::move(bundle));
            }
            require(!bundles.empty(), "The generator published no bundle in " + output.u8string());
            std::vector<hipblaslt_jit::GeneratedSolution> solutions;
            for(const auto& bundle : bundles)
            {
                auto solution = hipblaslt_jit::readTensileSourceBundle(bundle);
                if(std::find(excludeKernels.begin(), excludeKernels.end(), solution.kernelName)
                   == excludeKernels.end())
                    solutions.push_back(std::move(solution));
            }
            require(!solutions.empty(), "Every generated kernel is excluded");
            return solutions;
        }

        std::string fileStamp(const fs::path& path)
        {
            std::error_code error;
            const auto      size = fs::file_size(path, error);
            if(error)
                return "missing";
            const auto modified = fs::last_write_time(path, error);
            return std::to_string(size) + ":"
                   + std::to_string(error ? 0 : modified.time_since_epoch().count());
        }

        // The file a process launch runs for program.
        fs::path findProgram(const std::string& program)
        {
            const auto      path = fs::u8path(program);
            std::error_code error;
            if(path.has_parent_path() || program.empty())
                return path;
#ifdef _WIN32
            constexpr char separator = ';';
#else
            constexpr char separator = ':';
#endif
            const char*        search = std::getenv("PATH");
            std::istringstream directories(search ? search : "");
            for(std::string directory; std::getline(directories, directory, separator);)
                if(!directory.empty() && fs::is_regular_file(fs::u8path(directory) / path, error))
                    return fs::u8path(directory) / path;
            return path;
        }

        void addFiles(hipblaslt_jit::Fnv1a& hash, const fs::path& root, bool recursive)
        {
            std::vector<std::pair<std::string, std::string>> files;
            std::error_code                                  error;
            for(fs::recursive_directory_iterator next(root, error), end; !error && next != end;
                next.increment(error))
            {
                const auto name = next->path().filename();
                if(next->is_directory(error))
                {
                    if(!recursive || name == "Tests" || name == "__pycache__")
                        next.disable_recursion_pending();
                }
                else if(next->is_regular_file(error))
                    files.emplace_back(next->path().lexically_relative(root).generic_u8string(),
                                       fileStamp(next->path()));
            }
            std::sort(files.begin(), files.end());
            hash.add(root.u8string());
            for(const auto& [name, stamp] : files)
                hash.add(name).add(stamp);
        }

        // Changes whenever the generator can produce different solutions: the
        // hipBLASLt build, the options, the programs they name, the recipe and
        // the Python sources the generator imports.
        std::string generatorVersion(const Options& options)
        {
            hipblaslt_jit::Fnv1a hash;
            hash.add(std::to_string(HIPBLASLT_VERSION_MAJOR) + "." + std::to_string(HIPBLASLT_VERSION_MINOR)
                     + "." + std::to_string(HIPBLASLT_VERSION_PATCH));
#ifndef _WIN32
            Dl_info library{};
            if(dladdr(reinterpret_cast<const void*>(&generatorVersion), &library) && library.dli_fname)
                hash.add(library.dli_fname).add(fileStamp(library.dli_fname));
#endif
            hash.add(options.architecture).add(options.pythonPath).add(options.configPath);
            for(const auto* program : {&options.pythonExecutable, &options.cxxCompiler})
            {
                const auto path = findProgram(*program);
                hash.add(path.u8string()).add(fileStamp(path));
            }
            if(!options.configPath.empty())
            {
                std::ifstream      config(fs::u8path(options.configPath), std::ios::binary);
                std::ostringstream content;
                content << config.rdbuf();
                hash.add(config ? content.str() : "missing");
            }
            addFiles(hash, fs::u8path(options.tensileSourceDirectory) / "Tensile", true);
#ifdef _WIN32
            constexpr char separator = ';';
#else
            constexpr char separator = ':';
#endif
            std::istringstream entries(options.pythonPath);
            for(std::string entry; std::getline(entries, entry, separator);)
                if(!entry.empty())
                    addFiles(hash, fs::u8path(entry) / "rocisa", false);
            return "tensilelite:" + hash.hex();
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
                         options.configPath.empty() ? "origami.gemm.dp.v1" : "",
                         generatorVersion(options)}
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
                if(configured.outputPath.empty())
                    configured.outputPath = (request.scratch / "tensilelite").u8string();
                if(configured.architecture.empty())
                    configured.architecture = request.target.targetId;
                if(!targetMatchesDevice(configured.architecture, request.target.targetId))
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "Requested TensileLite architecture does not match device"};
                try
                {
                    if(request.prediction)
                    {
                        hipblaslt_jit::debug::Phase phase("request_write");
                        configured.configPath = writeJitGemmRequest(*gemm,
                                                                    configured,
                                                                    *request.prediction,
                                                                    request.count,
                                                                    request.excludeKernels);
                    }
                    runGenerator(configured,
                                 request.prediction ? "Tensile.JitGemm" : "Tensile.SingleSolution",
                                 request.codeObjectVersion);
                    hipblaslt_jit::debug::Phase phase("bundle_read");
                    solutions = readGeneratedSolutions(configured,
                                                       request.prediction != nullptr,
                                                       request.count,
                                                       request.excludeKernels);
                    phase.stop();
                    std::string summary;
                    if(request.prediction)
                        summary = "Origami ranked "
                                  + std::to_string(request.prediction->ranked.size())
                                  + " parameter candidates; "
                                  + (solutions.size() == 1
                                         ? std::string("the first candidate accepted by "
                                                       "TensileLite was compiled")
                                         : "the first " + std::to_string(solutions.size())
                                               + " candidates accepted by TensileLite were "
                                                 "compiled");
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
                    hipblaslt_jit::makeCatalogKnowledge(),
                    hipblaslt_jit::makeComgrBuilder(),
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

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    namespace
    {
        std::string configured(const char* variable, const char* builtIn)
        {
            const char* value = rocblaslt_secure_getenv(variable);
            return value && *value ? value : builtIn;
        }

        Status missing(const char* what, const std::string& path, const char* variable)
        {
            return {Status::Code::Failed,
                    Stage::Configure,
                    std::string(what) + " not found at " + path + "; set " + variable};
        }
    }

    Status makeProcessBackend(hipblaslt_jit::ProcessBackend& made)
    {
        namespace debug = hipblaslt_jit::debug;
        debug::Phase toolCheck("tool_check");
        Options      options;
        options.pythonExecutable = configured("HIPBLASLT_JIT_PYTHON", HIPBLASLT_JIT_DEFAULT_PYTHON);
        options.tensileSourceDirectory
            = configured("HIPBLASLT_JIT_TENSILE_SOURCE", HIPBLASLT_JIT_DEFAULT_TENSILE_SOURCE);
        options.pythonPath
            = configured("HIPBLASLT_JIT_PYTHONPATH", HIPBLASLT_JIT_DEFAULT_PYTHONPATH);
        options.cxxCompiler = configured("HIPBLASLT_JIT_CXX", HIPBLASLT_JIT_DEFAULT_CXX);
        std::error_code error;
        if(!fs::is_regular_file(findProgram(options.pythonExecutable), error))
            return missing("Python", options.pythonExecutable, "HIPBLASLT_JIT_PYTHON");
        if(!fs::is_directory(fs::u8path(options.tensileSourceDirectory) / "Tensile", error))
            return missing("TensileLite source",
                           options.tensileSourceDirectory,
                           "HIPBLASLT_JIT_TENSILE_SOURCE");
        if(!fs::is_regular_file(findProgram(options.cxxCompiler), error))
            return missing("C++ compiler", options.cxxCompiler, "HIPBLASLT_JIT_CXX");
        toolCheck.stop();
        debug::set("python", debug::Line::quote(options.pythonExecutable));
        debug::set("tensile_source", debug::Line::quote(options.tensileSourceDirectory));
        debug::set("cxx", debug::Line::quote(options.cxxCompiler));
        // The backend hashes the generator sources for its version.
        debug::Phase backendPhase("backend");
        made.backend = std::make_shared<const TensileLiteBackend>(options);
        backendPhase.stop();
        made.predictor = hipblaslt_jit::makeOrigamiPredictor();
        made.knowledge = hipblaslt_jit::makeCatalogKnowledge();
        return {};
    }
}
