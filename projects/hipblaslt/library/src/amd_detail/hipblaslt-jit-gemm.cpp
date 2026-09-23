// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-process.hpp"
#include "hipblaslt-jit-tensilelite-artifacts.hpp"
#include "hipblaslt-jit-tensilelite-internal.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt-functions.h"
#include "rocblaslt.h"
#include "tensile_host.hpp"
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
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

        void checkHip(hipError_t status, const char* operation)
        {
            if(status != hipSuccess)
                throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
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

        void generate(const Options& options, const char* module = "Tensile.SingleSolution")
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
            require(result.succeeded(),
                    "TensileLite generator: " + result.error + "; see "
                        + request.logPath.u8string());
        }
    }

    namespace
    {
        std::shared_ptr<detail::Bundle> loadGeneratedBundle(const Options&         options,
                                                            const hipDeviceProp_t& properties,
                                                            int                    device)
        {
            auto  context     = std::make_shared<detail::Bundle>();
            auto& p           = *context;
            p.device          = device;
            p.properties      = properties;
            const auto bundle = fs::canonical(fs::u8path(options.outputPath) / "bundle");
            p.manifest        = (bundle / "manifest.json").u8string();
            std::vector<std::string> objectList;
            const auto               tree = readEnvelope(bundle / "loader.bin", objectList);
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
            p.hardware = TensileLite::hip::GetDevice(properties, p.device);
            p.library  = std::dynamic_pointer_cast<Master>(
                TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                    readLibrary(physicalLibrary)));
            require(p.library && p.library->solutions.size() == 1 && p.library->solutions.count(0),
                    "Expected a non-lazy library containing only local solution 0");
            auto solution = p.library->solutions.at(0);
            require(solution && solution->index == 0 && solution->kernelName == p.kernel
                        && solution->solutionName == field(tree, "solution.name"),
                    "Library solution identity does not match manifest");
            p.adapter = std::make_shared<TensileLite::hip::SolutionAdapter>(false, "jit-gemm");
            for(const auto& path : codeObjects)
                checkHip(p.adapter->loadCodeObjectBytes(readArtifact(path)),
                         "Load generated code object");
            checkHip(p.adapter->initKernel(p.kernel), "Resolve generated kernel symbol");
            return context;
        }

        struct Provider final : jit::detail::BackendImplementation
        {
            Options options;
            explicit Provider(const Options& value)
                : options(value)
            {
            }
            std::string_view name() const noexcept override
            {
                return "TensileLite";
            }
            hipblasStatus_t compile(const jit::detail::OperationRequest& request,
                                    const jit::detail::Target&           target,
                                    size_t                               workspaceLimit,
                                    std::shared_ptr<const jit::detail::KernelBundle>& bundle,
                                    Diagnostics& diagnostics) const override
            {
                bundle.reset();
                diagnostics.backend = "TensileLite";
                if(request.kind() != jit::detail::GemmRequest::operation
                   || !dynamic_cast<const jit::detail::GemmRequest*>(&request))
                {
                    diagnostics.message = "TensileLite does not implement this operation";
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
                auto configured = options;
                if(configured.architecture.empty())
                    configured.architecture = target.properties.gcnArchName;
                if(!targetMatchesDevice(configured.architecture, target.properties.gcnArchName))
                {
                    diagnostics.message
                        = "Requested TensileLite architecture does not match device";
                    return HIPBLAS_STATUS_ARCH_MISMATCH;
                }
                try
                {
                    generate(configured);
                    auto candidate
                        = loadGeneratedBundle(configured, target.properties, target.device);
                    size_t workspace = 0;
                    auto   status
                        = candidate->support(request, workspaceLimit, workspace, diagnostics);
                    if(status == HIPBLAS_STATUS_SUCCESS)
                        bundle = std::move(candidate);
                    return status;
                }
                catch(const std::bad_alloc&)
                {
                    throw;
                }
                catch(const std::exception& e)
                {
                    diagnostics.message = e.what();
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                }
            }
        };
    }

    hipblasStatus_t
        createBackend(const Options& options, jit::Backend& backend, Diagnostics& diagnostics)
    {
        backend     = {};
        diagnostics = {"TensileLite", ""};
        if(options.pythonExecutable.empty() || options.tensileSourceDirectory.empty()
           || options.configPath.empty() || options.outputPath.empty())
        {
            diagnostics.message = "TensileLite requires Python, source, recipe and output paths";
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
        try
        {
            backend = jit::detail::BackendAccess::make(std::make_shared<Provider>(options));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate TensileLite backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
    }
}
