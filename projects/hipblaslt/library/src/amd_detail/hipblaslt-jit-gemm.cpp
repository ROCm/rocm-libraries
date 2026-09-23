// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-internal.hpp"
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
#include <hipblaslt/hipblaslt-ext.hpp>
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

    int detail::jitMXScaleFormat(RocblasltContractionProblem::ScalingFormat scaleA,
                                 RocblasltContractionProblem::ScalingFormat scaleB,
                                 const std::string&                         architecture)
    {
        using Format        = RocblasltContractionProblem::ScalingFormat;
        int        expected = -1;
        const auto arch     = architecture.substr(0, architecture.find(':'));
        for(const auto mode : {scaleA, scaleB})
        {
            if(mode == Format::None || mode == Format::Scalar || mode == Format::Vector)
                continue;
            int format = -2;
            if(arch == "gfx950" && mode == Format::Block_32_UE8M0_32_8_EXT)
                format = 1;
            else if(arch == "gfx1250" && mode >= Format::Block_32_UE8M0
                    && mode <= Format::Block_16_UE5M3)
                format = 2;
            if(format == -2 || (expected != -1 && expected != format))
                return -2;
            expected = format;
        }
        return expected;
    }

    bool detail::jitScaleLayoutMatches(const JitContext&                          context,
                                       RocblasltContractionProblem::ScalingFormat scaleA,
                                       RocblasltContractionProblem::ScalingFormat scaleB)
    {
        const int expected = jitMXScaleFormat(scaleA, scaleB, context.properties->gcnArchName);
        return expected == -1
               || (expected >= 0
                   && context.library->solutions.at(0)->problemType.mxScaleFormat == expected);
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
            const auto outputLayout = reinterpret_cast<rocblaslt_matrix_layout>(layoutD);
            if(status == HIPBLAS_STATUS_SUCCESS && (!outputLayout->m || !outputLayout->n))
            {
                info.error   = "Empty output does not require a GEMM algorithm";
                result.state = HIPBLAS_STATUS_NOT_SUPPORTED;
                return HIPBLAS_STATUS_NOT_SUPPORTED;
            }
            require(status == HIPBLAS_STATUS_SUCCESS && count == 1 && opaque,
                    "Canonical GEMM descriptor translation failed");
            const auto rocDesc       = reinterpret_cast<rocblaslt_matmul_desc>(desc);
            const int  mxScaleFormat = detail::jitMXScaleFormat(
                rocDesc->scaleAType, rocDesc->scaleBType, properties.gcnArchName);
            require(mxScaleFormat != -2,
                    "Unsupported MX scale layout: gfx950 requires block32 UE8M0 pre-swizzled "
                    "scales (scale mode 1001); gfx1250 requires its natural block-scale modes");
            require(!options.configPath.empty(), "JIT selection requires an explicit YAML recipe");
            info.configPath = fs::absolute(options.configPath).string();
            generate(options);
            auto context = loadGeneratedBundle(options, properties, device);
            require(
                detail::jitScaleLayoutMatches(*context, rocDesc->scaleAType, rocDesc->scaleBType),
                "Generated solution MX scale layout differs from the supplied descriptors");
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
            if(status != HIPBLAS_STATUS_SUCCESS)
            {
                const auto&        problem  = *ExtractProblemGemm(opaque);
                const auto&        solution = *context->library->solutions.at(0);
                TensileLite::Task  task(*context->hardware, problem, solution);
                std::ostringstream diagnostics;
                diagnostics << "hipBLASLt status " << static_cast<int>(status) << std::boolalpha
                            << "; hardware=" << (*solution.hardwarePredicate)(*context->hardware)
                            << ", problem=" << (*solution.problemPredicate)(problem)
                            << ", task=" << (*solution.taskPredicate)(task) << '\n';
                solution.hardwarePredicate->debugEval(*context->hardware, diagnostics);
                solution.problemPredicate->debugEval(problem, diagnostics);
                solution.taskPredicate->debugEval(task, diagnostics);
                throw std::runtime_error(
                    "Generated solution does not support the problem or workspace limit: "
                    + diagnostics.str());
            }
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

}
