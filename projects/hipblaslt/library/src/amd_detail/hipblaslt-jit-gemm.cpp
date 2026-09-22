// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-gemm-predictor.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt-functions.h"
#include "rocblaslt.h"
#include "tensile_host.hpp"
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <algorithm>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
#include <set>
#include <spawn.h>
#include <sstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>

extern char** environ;

namespace hipblaslt_ext::experimental
{
    namespace
    {
        namespace fs = std::filesystem;
        using Tree   = boost::property_tree::ptree;
        using Master = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

        void checkHip(hipError_t status, const char* operation)
        {
            if(status != hipSuccess)
                throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
        }

        // PropertyTree parses JSON escapes; reject duplicate object keys before field lookup.
        void uniqueKeys(const Tree& tree)
        {
            std::set<std::string> names;
            const bool            isArray = !tree.empty() && tree.front().first.empty();
            for(const auto& child : tree)
            {
                require(isArray ? child.first.empty()
                                : (!child.first.empty() && names.insert(child.first).second),
                        "Manifest contains mixed array/object entries or duplicate object key");
                uniqueKeys(child.second);
            }
        }

        std::string field(const Tree& tree, const char* name)
        {
            const auto& child = tree.get_child(name);
            require(child.empty() && !child.data().empty()
                        && child.data().find('\0') == std::string::npos,
                    std::string("Invalid manifest field: ") + name);
            return child.data();
        }

        fs::path artifact(const fs::path& bundle, const std::string& name, bool mustExist = true)
        {
            fs::path relative(name);
            require(!relative.empty() && !relative.is_absolute(), "Artifact path must be relative");
            for(const auto& part : relative)
                require(part != "..", "Artifact path escapes bundle");
            const auto path   = fs::weakly_canonical(bundle / relative);
            const auto inside = path.lexically_relative(fs::canonical(bundle));
            require(!inside.empty() && *inside.begin() != "..", "Artifact symlink escapes bundle");
            if(mustExist)
                require(fs::is_regular_file(path), "Missing artifact: " + path.string());
            return path;
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

        void generate(const GenerateOptions& options, const char* module = "Tensile.SingleSolution")
        {
            require(!options.pythonExecutable.empty() && !options.tensileSourceDirectory.empty()
                        && !options.configPath.empty() && !options.outputPath.empty()
                        && !options.architecture.empty(),
                    "Missing generation option");
            require(fs::is_directory(options.tensileSourceDirectory),
                    "Invalid Tensile source directory");
            require(!fs::exists(options.outputPath), "Output path already exists");
            auto toolPath = [](const std::string& tool) {
                return fs::path(tool).has_parent_path() ? fs::absolute(tool).string() : tool;
            };
            std::vector<std::string> args = {toolPath(options.pythonExecutable),
                                             "-m",
                                             module,
                                             fs::absolute(options.configPath).string(),
                                             fs::absolute(options.outputPath).string(),
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
            std::vector<char*> argv;
            for(auto& arg : args)
            {
                require(arg.find('\0') == std::string::npos, "Generation argument contains NUL");
                argv.push_back(arg.data());
            }
            argv.push_back(nullptr);
            std::vector<std::string> environment;
            for(char** entry = environ; *entry; ++entry)
                if(std::strncmp(*entry, "PYTHONPATH=", 11) != 0)
                    environment.emplace_back(*entry);
            environment.push_back("PYTHONPATH="
                                  + fs::absolute(options.tensileSourceDirectory).string()
                                  + (options.pythonPath.empty() ? "" : ":" + options.pythonPath));
            std::vector<char*> envp;
            for(auto& value : environment)
                envp.push_back(value.data());
            envp.push_back(nullptr);

            // Isolate tools that may create relative temporary files without changing
            // the caller's process cwd. Retain this directory for diagnostics.
            const std::string cwd = fs::absolute(options.outputPath).string() + ".cwd";
            require(fs::create_directory(cwd), "Generator working directory already exists");
            const std::string logPath = options.outputPath + ".log";
            int log = open(logPath.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            require(log >= 0,
                    "Cannot create generator log: " + logPath + ": " + std::strerror(errno));
            posix_spawn_file_actions_t actions;
            int                        rc = posix_spawn_file_actions_init(&actions);
            if(rc != 0)
            {
                close(log);
                throw std::runtime_error("Cannot initialize spawn actions: "
                                         + std::string(std::strerror(rc)));
            }
            rc = posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
            if(rc == 0)
                rc = posix_spawn_file_actions_adddup2(&actions, log, STDOUT_FILENO);
            if(rc == 0)
                rc = posix_spawn_file_actions_adddup2(&actions, log, STDERR_FILENO);
            pid_t child = -1;
            if(rc == 0)
                rc = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), envp.data());
            posix_spawn_file_actions_destroy(&actions);
            close(log);
            require(rc == 0, "Cannot start generator: " + std::string(std::strerror(rc)));
            int   status = 0;
            pid_t waited;
            do
                waited = waitpid(child, &status, 0);
            while(waited < 0 && errno == EINTR);
            require(waited == child, "Cannot wait for generator");
            require(WIFEXITED(status) && WEXITSTATUS(status) == 0,
                    "Generator failed; see " + logPath);
        }
    }

    namespace
    {
        std::shared_ptr<detail::JitContext> loadGeneratedBundle(const GenerateOptions& options,
                                                                const hipDeviceProp_t& properties,
                                                                int                    device)
        {
            auto  context     = std::make_shared<detail::JitContext>();
            auto& p           = *context;
            p.device          = device;
            p.properties      = std::make_shared<hipDeviceProp_t>(properties);
            const auto bundle = fs::canonical(fs::path(options.outputPath) / "bundle");
            p.manifest        = (bundle / "manifest.json").string();
            Tree tree;
            boost::property_tree::read_json(p.manifest, tree);
            uniqueKeys(tree);
            require(field(tree, "schema_version") == "2" && field(tree, "counts.solutions") == "1"
                        && field(tree, "counts.main_kernels") == "1"
                        && field(tree, "solution.index") == "0",
                    "Unsupported single-solution manifest schema/counts/index");
            p.kernel = field(tree, "main_kernel.name");
            require(p.kernel == field(tree, "solution.kernel_name"),
                    "Manifest kernel names disagree");
            std::string architecture(properties.gcnArchName);
            require(field(tree, "architecture.requested") == options.architecture,
                    "Bundle architecture differs from request");
            require(targetMatchesDevice(field(tree, "architecture.resolved"), architecture)
                        && field(tree, "architecture.compiler_target")
                               == field(tree, "architecture.resolved"),
                    "Bundle architecture does not match device");
#ifdef TENSILE_YAML
            require(field(tree, "library.format") == "yaml", "Expected YAML library");
#else
            require(field(tree, "library.format") == "msgpack", "Expected MessagePack library");
#endif
            const auto  codeObject = artifact(bundle, field(tree, "main_kernel.code_object"));
            const auto& objectList = tree.get_child("code_objects");
            require(!objectList.empty() && objectList.data().empty(),
                    "Expected nonempty code_objects array");
            std::set<fs::path> codeObjects;
            for(const auto& item : objectList)
            {
                require(item.first.empty() && item.second.empty() && !item.second.data().empty()
                            && item.second.data().find('\0') == std::string::npos,
                        "Invalid code_objects entry");
                require(codeObjects.insert(artifact(bundle, item.second.data())).second,
                        "Duplicate code object artifact");
            }
            require(codeObjects.count(codeObject) == 1,
                    "Main code object missing from code_objects");
            const auto physicalLibrary = artifact(bundle, field(tree, "library.path"));
            const auto logicalLibrary
                = artifact(bundle, field(tree, "library.logical_path"), false);
            require(physicalLibrary == logicalLibrary
                        || physicalLibrary.string() == logicalLibrary.string() + ".zlib",
                    "Manifest physical and logical library paths disagree");
            p.hardware = TensileLite::hip::GetDevice(properties, p.device);
            p.library  = std::dynamic_pointer_cast<Master>(
                TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(
                    logicalLibrary.string()));
            require(p.library && p.library->solutions.size() == 1 && p.library->solutions.count(0),
                    "Expected a non-lazy library containing only local solution 0");
            auto solution = p.library->solutions.at(0);
            require(solution && solution->index == 0 && solution->kernelName == p.kernel
                        && solution->solutionName == field(tree, "solution.name"),
                    "Library solution identity does not match manifest");
            p.adapter = std::make_unique<TensileLite::hip::SolutionAdapter>(false, "jit-gemm");
            for(const auto& path : codeObjects)
                checkHip(p.adapter->loadCodeObjectFile(path.string()),
                         "Load generated code object");
            checkHip(p.adapter->initKernel(p.kernel), "Resolve generated kernel symbol");
            return context;
        }

        struct Registry
        {
            std::mutex                                                        mutex;
            std::unordered_map<uint64_t, std::shared_ptr<detail::JitContext>> entries;
            std::mt19937_64 random{std::random_device{}()};
        };

        Registry& registry()
        {
            // Intentionally retained: normal asynchronous launches and graph
            // executions may outlive any object holding a copied algorithm.
            static auto* instance = new Registry;
            return *instance;
        }

        uint64_t registerContext(std::shared_ptr<detail::JitContext> context)
        {
            auto&                       r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            uint64_t                    token;
            do
                token = r.random() & ((uint64_t{1} << 56) - 1);
            while(token == 0 || r.entries.count(token));
            r.entries.emplace(token, std::move(context));
            return token;
        }
    }

    std::shared_ptr<detail::JitContext> detail::resolveJitAlgo(const rocblaslt_matmul_algo& algo,
                                                               int                          device)
    {
        if(!isJitAlgo(algo))
            return {};
        int index;
        std::memcpy(&index, algo.data, sizeof(index));
        uint64_t token = 0;
        std::memcpy(&token, algo.data_pad, sizeof(algo.data_pad));
        auto&                       r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        auto                        found = r.entries.find(token);
        require(index == 0 && !algo.fallback && found != r.entries.end(),
                "Unknown or invalid process-local JIT algorithm");
        require(found->second->device == device, "JIT algorithm belongs to another device");
        int current;
        checkHip(hipGetDevice(&current), "hipGetDevice");
        require(current == device, "Use the device on which the JIT algorithm was generated");
        return found->second;
    }

    hipblasStatus_t getJitGemmAlgo(hipblasLtHandle_t                 handle,
                                   hipblasLtMatmulDesc_t             desc,
                                   const void*                       alpha,
                                   const void*                       A,
                                   hipblasLtMatrixLayout_t           layoutA,
                                   const void*                       B,
                                   hipblasLtMatrixLayout_t           layoutB,
                                   const void*                       beta,
                                   const void*                       C,
                                   hipblasLtMatrixLayout_t           layoutC,
                                   void*                             D,
                                   hipblasLtMatrixLayout_t           layoutD,
                                   const GenerateOptions&            requestedOptions,
                                   size_t                            maxWorkspaceBytes,
                                   hipblasLtMatmulHeuristicResult_t& result,
                                   JitGemmInfo&                      info)
    {
        result         = {};
        result.state   = HIPBLAS_STATUS_INVALID_VALUE;
        info           = {};
        uint64_t token = 0;
        try
        {
            require(handle && desc && layoutA && layoutB && layoutC && layoutD,
                    "JIT selection requires valid handle and descriptors");
            const auto rocHandle = reinterpret_cast<rocblaslt_handle>(handle);
            int        device;
            checkHip(hipGetDevice(&device), "hipGetDevice");
            require(device == rocHandle->device, "Use the device on which the handle was created");
            hipDeviceProp_t properties;
            checkHip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
            auto options = requestedOptions;
            if(options.architecture.empty())
                options.architecture = properties.gcnArchName;
            require(targetMatchesDevice(options.architecture, properties.gcnArchName),
                    "Requested architecture does not match device");
            std::shared_ptr<void>           opaque;
            size_t                          count = 0;
            rocblaslt::RocGemmProblemTypeV2 type;
            auto                            status = RocBlasLtStatusToHIPStatus(
                rocblaslt_gemm_create_cpp(rocHandle,
                                          reinterpret_cast<rocblaslt_matmul_desc>(desc),
                                          alpha,
                                          A,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutA),
                                          B,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutB),
                                          beta,
                                          C,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutC),
                                          D,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutD),
                                          type,
                                          opaque,
                                          count));
            require(status == HIPBLAS_STATUS_SUCCESS && count == 1 && opaque,
                    "Canonical GEMM descriptor translation failed");
            const bool predict = options.configPath.empty();
            if(predict)
            {
                const auto hardware = TensileLite::hip::GetDevice(properties, device);
                options.configPath
                    = predictJitGemmConfig(*ExtractProblemGemm(opaque), *hardware, options, info);
            }
            else
                info.configPath = fs::absolute(options.configPath).string();
            generate(options, predict ? "Tensile.JitGemm" : "Tensile.SingleSolution");
            if(predict)
            {
                Tree prediction;
                boost::property_tree::read_json(options.outputPath + ".prediction.json",
                                                prediction);
                uniqueKeys(prediction);
                info.prediction = field(prediction, "summary");
            }
            auto context      = loadGeneratedBundle(options, properties, device);
            info.manifestPath = context->manifest;
            info.kernelName   = context->kernel;
            token             = registerContext(context);
            rocblaslt_matmul_algo algo{};
            std::memcpy(
                algo.data + sizeof(int32_t), &detail::jitAlgoTag, sizeof(detail::jitAlgoTag));
            std::memcpy(algo.data_pad, &token, sizeof(algo.data_pad));
            algo.max_workspace_bytes          = maxWorkspaceBytes;
            size_t                  workspace = 0;
            rocblaslt::RocTuningV2* tuning    = nullptr;
            status                            = RocBlasLtStatusToHIPStatus(
                rocblaslt_is_algo_supported_cpp(rocHandle,
                                                rocblaslt::RocGemmType::ROCBLASLT_GEMM,
                                                opaque,
                                                algo,
                                                tuning,
                                                workspace));
            require(status == HIPBLAS_STATUS_SUCCESS,
                    "Generated solution does not support the problem or workspace limit");
            std::memcpy(&result.algo, &algo, sizeof(algo));
            result.workspaceSize = workspace;
            result.state         = HIPBLAS_STATUS_SUCCESS;
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::exception& e)
        {
            if(token)
            {
                auto&                       r = registry();
                std::lock_guard<std::mutex> lock(r.mutex);
                r.entries.erase(token); // Never submitted or exposed to the caller.
            }
            info.error = e.what();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
    }

    struct JitGemm::Impl
    {
        hipblasLtHandle_t                                  handle;
        int                                                device = -1;
        TensileLite::ContractionProblemGemm                problem;
        TensileLite::ContractionInputs                     inputs;
        std::vector<TensileLite::KernelInvocation>         kernels;
        bool                                               hasProblem = false;
        std::shared_ptr<TensileLite::Hardware>             hardware;
        std::shared_ptr<Master>                            library;
        std::shared_ptr<TensileLite::ContractionSolution>  solution;
        std::unique_ptr<TensileLite::hip::SolutionAdapter> adapter;
        hipEvent_t                                         completion        = nullptr;
        hipStream_t                                        stream            = nullptr;
        bool                                               submitted         = false;
        bool                                               drainRequired     = false;
        bool                                               prepared          = false;
        bool                                               initialized       = false;
        size_t                                             workspaceRequired = 0;
        size_t                                             generations       = 0;
        std::string                                        error, manifest, kernel;
        DispatchInfo                                       dispatch;
        void*                                              synchronizer      = nullptr;
        size_t                                             synchronizerBytes = 0;

        explicit Impl(hipblasLtHandle_t value)
            : handle(value)
        {
            if(handle)
                device = reinterpret_cast<rocblaslt_handle>(handle)->device;
        }

        ~Impl()
        {
            if(!adapter && !completion && !synchronizer)
                return;
            int  previous      = -1;
            auto status        = hipGetDevice(&previous);
            bool changedDevice = false;
            if(status == hipSuccess && previous != device)
            {
                status        = hipSetDevice(device);
                changedDevice = status == hipSuccess;
            }
            const bool correctDevice = status == hipSuccess;
            if(correctDevice && submitted)
            {
                status = drainRequired ? hipStreamSynchronize(stream)
                                       : hipEventSynchronize(completion);
                if(status != hipSuccess)
                    status = hipStreamSynchronize(stream);
            }
            if(status == hipSuccess)
            {
                adapter.reset();
                if(synchronizer)
                {
                    status = hipFree(synchronizer);
                    if(status != hipSuccess)
                        std::cerr << "JitGemm synchronizer cleanup: " << hipGetErrorString(status)
                                  << '\n';
                }
                if(completion)
                {
                    status = hipEventDestroy(completion);
                    if(status != hipSuccess)
                        std::cerr << "JitGemm event cleanup: " << hipGetErrorString(status) << '\n';
                }
            }
            else
            {
                // Completion/context could not be established. Leaking this private
                // module/event/synchronizer is safer than unloading code that may still be executing.
                adapter.release();
                std::cerr << "JitGemm retained module/event/synchronizer after cleanup failure: "
                          << hipGetErrorString(status) << '\n';
            }
            if(changedDevice)
            {
                status = hipSetDevice(previous);
                if(status != hipSuccess)
                    std::cerr << "JitGemm device restore: " << hipGetErrorString(status) << '\n';
            }
        }

        hipblasStatus_t fail(const std::string& message,
                             hipblasStatus_t    status = HIPBLAS_STATUS_INVALID_VALUE)
        {
            error = message;
            return status;
        }

        void checkDevice() const
        {
            int current;
            checkHip(hipGetDevice(&current), "hipGetDevice");
            require(handle && current == device, "Use the device on which the handle was created");
        }

        void wait()
        {
            if(submitted)
            {
                checkHip(drainRequired ? hipStreamSynchronize(stream)
                                       : hipEventSynchronize(completion),
                         "Wait for previous JIT GEMM");
                submitted     = false;
                drainRequired = false;
            }
        }

        void support(size_t workspaceBytes)
        {
            problem.setParams().resetInternalArgs();
            problem.setParams().setFallbackStatus(solution->isFallbackForHW(*hardware));
            problem.setWorkspaceSize(workspaceBytes);
            TensileLite::Task  task(*hardware, problem, *solution);
            std::ostringstream detail;
            bool               softwareMatch
                = problem.getParams().uniformSummationOrder()
                      ? TensileLite::softwarePredicate(
                            TensileLite::SolutionLibrarySearchType::DEFAULT,
                            task,
                            *hardware,
                            *solution,
                            problem)
                      : (*solution->problemPredicate)(problem) && (*solution->taskPredicate)(task);
            bool match = (*solution->hardwarePredicate)(*hardware) && softwareMatch;
            if(!match)
            {
                solution->hardwarePredicate->debugEval(*hardware, detail);
                solution->problemPredicate->debugEval(problem, detail);
                solution->taskPredicate->debugEval(task, detail);
            }
            require(match,
                    "Generated solution does not support this problem/device: " + detail.str());
        }
    };

    JitGemm::JitGemm(hipblasLtHandle_t handle)
        : m_impl(std::make_unique<Impl>(handle))
    {
    }
    JitGemm::~JitGemm() = default;

    hipblasStatus_t JitGemm::setProblem(hipblasLtMatmulDesc_t   desc,
                                        const void*             alpha,
                                        const void*             A,
                                        hipblasLtMatrixLayout_t layoutA,
                                        const void*             B,
                                        hipblasLtMatrixLayout_t layoutB,
                                        const void*             beta,
                                        const void*             C,
                                        hipblasLtMatrixLayout_t layoutC,
                                        void*                   D,
                                        hipblasLtMatrixLayout_t layoutD)
    {
        auto& p = *m_impl;
        p.error.clear();
        if(!p.prepared)
            p.hasProblem = false;
        if(p.prepared || !p.handle || !desc || !layoutA || !layoutB || !layoutC || !layoutD)
            return p.fail("setProblem requires valid descriptors and an unprepared object");
        try
        {
            p.checkDevice();
            std::shared_ptr<void>           opaque;
            size_t                          count = 0;
            rocblaslt::RocGemmProblemTypeV2 type;
            auto                            status = RocBlasLtStatusToHIPStatus(
                rocblaslt_gemm_create_cpp(reinterpret_cast<rocblaslt_handle>(p.handle),
                                          reinterpret_cast<rocblaslt_matmul_desc>(desc),
                                          alpha,
                                          A,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutA),
                                          B,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutB),
                                          beta,
                                          C,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutC),
                                          D,
                                          reinterpret_cast<rocblaslt_matrix_layout>(layoutD),
                                          type,
                                          opaque,
                                          count));
            if(status != HIPBLAS_STATUS_SUCCESS)
                return p.fail("Canonical GEMM descriptor translation failed", status);
            require(count == 1 && opaque, "Expected one GEMM problem");
            p.problem    = *ExtractProblemGemm(opaque);
            p.inputs     = *ExtractInputsGemm(opaque);
            p.hasProblem = true;
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::exception& e)
        {
            return p.fail(e.what());
        }
    }

    hipblasStatus_t JitGemm::prepare(const GenerateOptions& options, size_t& workspaceBytes)
    {
        auto& p = *m_impl;
        p.error.clear();
        workspaceBytes = 0;
        if(p.prepared || !p.hasProblem)
            return p.fail("prepare requires a bound problem and an unprepared object");
        try
        {
            p.checkDevice();
            hipDeviceProp_t properties;
            checkHip(hipGetDeviceProperties(&properties, p.device), "hipGetDeviceProperties");
            require(targetMatchesDevice(options.architecture, properties.gcnArchName),
                    "Requested architecture does not match device; this runtime accepts its HIP "
                    "architecture and feature qualifiers");
            ++p.generations;
            generate(options);
            auto context = loadGeneratedBundle(options, properties, p.device);
            p.manifest   = context->manifest;
            p.kernel     = context->kernel;
            p.hardware   = context->hardware;
            p.library    = context->library;
            p.solution   = p.library->solutions.at(0);
            // Stream-K has its own stream-private flag binding and queue setup in
            // the normal host path. This experiment currently supports non-Stream-K
            // solutions, including fixed/automatic GSU and adaptive accumulation.
            require(p.solution->sizeMapping.streamK == 0,
                    "JIT GEMM does not yet bind Stream-K stream-private state");
            require(!p.solution->problemType.outputAmaxD,
                    "JIT GEMM does not yet own the amax output synchronization state");
            p.support(std::numeric_limits<size_t>::max());
            p.workspaceRequired = p.solution->requiredWorkspaceSize(p.problem, *p.hardware);
            p.support(p.workspaceRequired);
            p.adapter = std::move(context->adapter);
            checkHip(hipEventCreateWithFlags(&p.completion, hipEventDisableTiming),
                     "Create completion event");
            p.prepared     = true;
            workspaceBytes = p.workspaceRequired;
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::exception& e)
        {
            p.adapter.reset();
            p.solution.reset();
            p.library.reset();
            return p.fail(e.what());
        }
    }

    hipblasStatus_t JitGemm::initialize(void* workspace, size_t workspaceBytes, hipStream_t stream)
    {
        auto& p = *m_impl;
        p.error.clear();
        p.initialized = false;
        p.dispatch    = {};
        if(!p.prepared || workspaceBytes < p.workspaceRequired
           || (p.workspaceRequired && !workspace))
            return p.fail("initialize requires a prepared object and sufficient workspace");
        try
        {
            p.checkDevice();
            hipDevice_t streamDevice;
            checkHip(hipStreamGetDevice(stream, &streamDevice), "Query stream device");
            require(streamDevice == p.device, "Stream belongs to a different device");
            hipStreamCaptureStatus capture;
            checkHip(hipStreamIsCapturing(stream, &capture), "Query stream capture");
            require(capture == hipStreamCaptureStatusNone,
                    "JitGemm does not support stream capture");
            p.wait();
            p.support(workspaceBytes);
            p.problem.setParams().setWGMXCC(p.solution->isFallbackForHW(*p.hardware) ? 1 : 0);
            p.inputs.ws            = workspace;
            p.inputs.workspaceSize = workspaceBytes;
            // The canonical inputs reference the handle's shared GSU counters. A
            // JIT owner can run concurrently with another owner on that handle,
            // so retain its own counters and preserve the normal kernel argument ABI.
            const auto syncBytes = p.solution->requiredSynchronizerSize(p.problem, *p.hardware);
            if(syncBytes > p.synchronizerBytes)
            {
                void* storage = nullptr;
                checkHip(hipMalloc(&storage, syncBytes), "Allocate private GSU synchronizer");
                if(p.synchronizer)
                {
                    const auto status = hipFree(p.synchronizer);
                    if(status != hipSuccess)
                    {
                        const auto cleanup = hipFree(storage);
                        checkHip(cleanup, "Release unused GSU synchronizer");
                        checkHip(status, "Resize private GSU synchronizer");
                    }
                }
                p.synchronizer      = storage;
                p.synchronizerBytes = syncBytes;
            }
            if(p.synchronizer)
                p.inputs.Synchronizer = p.synchronizer;
            auto invocations = p.solution->solve(p.problem, p.inputs, *p.hardware);
            require(std::count_if(
                        invocations.begin(),
                        invocations.end(),
                        [&](const auto& invocation) { return invocation.kernelName == p.kernel; })
                        == 1,
                    "Solution solve must emit its manifest main kernel exactly once");
            DispatchInfo dispatch;
            dispatch.configuredGlobalSplitU = p.solution->sizeMapping.globalSplitU;
            dispatch.globalSplitU           = p.problem.getParams().gsu() > 0
                                                  ? p.problem.getParams().gsu()
                                                  : p.solution->calculateAutoGSU(p.problem, p.hardware.get());
            const auto accumulation         = p.problem.getAccumulation(
                *p.hardware, p.solution->sizeMapping, dispatch.globalSplitU);
            dispatch.accumulation = accumulation == 3   ? "multiple-buffer-single-kernel"
                                    : accumulation == 2 ? "multiple-buffer"
                                    : accumulation == 1 ? "single-buffer"
                                                        : "direct";
            for(const auto& invocation : invocations)
            {
                require(!invocation.kernelName.empty(), "Solution emitted an unnamed invocation");
                checkHip(p.adapter->initKernel(invocation.kernelName),
                         "Resolve generated invocation symbol");
                dispatch.kernelNames.push_back(invocation.kernelName);
            }
            p.dispatch    = std::move(dispatch);
            p.kernels     = std::move(invocations);
            p.stream      = stream;
            p.initialized = true;
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::exception& e)
        {
            return p.fail(e.what());
        }
    }

    hipblasStatus_t JitGemm::run(hipStream_t stream)
    {
        auto& p = *m_impl;
        p.error.clear();
        if(!p.initialized || stream != p.stream)
            return p.fail("run requires initialization on this stream");
        try
        {
            p.checkDevice();
            hipStreamCaptureStatus capture;
            checkHip(hipStreamIsCapturing(stream, &capture), "Query stream capture");
            require(capture == hipStreamCaptureStatusNone,
                    "JitGemm does not support stream capture");
            // Mark submission before calling into the adapter: an exception may follow enqueue.
            p.drainRequired = true;
            p.submitted     = true;
            auto status     = hipSuccess;
            if(p.synchronizerBytes)
                status = hipMemsetAsync(p.synchronizer, 0, p.synchronizerBytes, stream);
            if(status == hipSuccess)
                status = p.adapter->launchKernels(p.kernels, stream, nullptr, nullptr, true);
            if(status == hipSuccess)
                status = hipEventRecord(p.completion, stream);
            if(status != hipSuccess)
            {
                // A launch may already have submitted work even when the event failed.
                const auto drained = hipStreamSynchronize(stream);
                p.submitted        = drained != hipSuccess;
                p.drainRequired    = drained != hipSuccess;
                p.initialized      = false;
                checkHip(drained, "Drain JIT GEMM after submission failure");
                checkHip(status, "Submit JIT GEMM / record completion");
            }
            p.submitted     = true;
            p.drainRequired = false;
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::exception& e)
        {
            p.initialized     = false;
            std::string error = e.what();
            if(p.drainRequired)
            {
                const auto status = hipStreamSynchronize(stream);
                p.submitted       = status != hipSuccess;
                p.drainRequired   = status != hipSuccess;
                if(status != hipSuccess)
                    error += std::string("; stream drain failed: ") + hipGetErrorString(status);
            }
            return p.fail(error, HIPBLAS_STATUS_EXECUTION_FAILED);
        }
    }

    const DispatchInfo& JitGemm::dispatchInfo() const
    {
        return m_impl->dispatch;
    }

    const std::string& JitGemm::lastError() const
    {
        return m_impl->error;
    }
    const std::string& JitGemm::manifestPath() const
    {
        return m_impl->manifest;
    }
    const std::string& JitGemm::kernelName() const
    {
        return m_impl->kernel;
    }
    size_t JitGemm::generationCount() const
    {
        return m_impl->generations;
    }
}
