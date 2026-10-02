// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-component.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace hj  = hipblaslt_jit;
namespace abi = hipblaslt_ext::experimental::jit::detail;
namespace fs  = std::filesystem;
using Code    = hj::Status::Code;
using hj::Stage;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    // Jit is operation-agnostic; none of these stages sees a GEMM.
    struct ProbeRequest final : abi::OperationRequest
    {
        std::string_view kind() const noexcept override
        {
            return "test.jit.probe.v1";
        }
    };

    struct ProbeBundle final : abi::KernelBundle
    {
        std::string kernel;
        explicit ProbeBundle(std::string name)
            : kernel(std::move(name))
        {
        }
        std::string_view operationKind() const noexcept override
        {
            return "test.jit.probe.v1";
        }
        std::string name() const override
        {
            return kernel;
        }
        std::string kernelNames() const override
        {
            return kernel;
        }
        hipblasStatus_t support(const abi::OperationRequest&,
                                size_t,
                                size_t&,
                                hipblaslt_ext::experimental::jit::Diagnostics&) const override
        {
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
        hipblasStatus_t prepare(const abi::OperationRequest&,
                                const abi::ExecutionContext&,
                                std::shared_ptr<const abi::PreparedLaunch>&,
                                hipblaslt_ext::experimental::jit::Diagnostics&) const override
        {
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
    };

    // Calls are recorded as "stage:kernel" in the order they happen.
    struct Log
    {
        mutable std::mutex       mutex;
        std::vector<std::string> calls;
        void                     add(const std::string& call)
        {
            std::lock_guard<std::mutex> lock(mutex);
            calls.push_back(call);
        }
        size_t count(const std::string& prefix) const
        {
            std::lock_guard<std::mutex> lock(mutex);
            return std::count_if(calls.begin(), calls.end(), [&](const auto& call) {
                return call.rfind(prefix, 0) == 0;
            });
        }
    };

    struct Backend final : hj::Backend
    {
        Log&                     log;
        hj::BackendInfo          information;
        std::vector<std::string> kernels; // generated in order
        hj::Status               result;
        bool                     throws = false, allocationFails = false, writes = true;
        mutable std::mutex       mutex;
        mutable size_t           count = 0, workspaceLimit = 0;
        mutable std::vector<std::string> excludeKernels;
        mutable std::vector<fs::path>    scratches;
        mutable const hj::DeviceTarget*  target = nullptr;

        Backend(Log& l, std::vector<std::string> k)
            : log(l)
            , information{"fake-backend", "Fake"}
            , kernels(std::move(k))
            , result{Code::Success, Stage::Generate, "generated"}
        {
        }
        const hj::BackendInfo& info() const noexcept override
        {
            return information;
        }
        hj::Status generate(const hj::GenerationRequest&       request,
                            std::vector<hj::GeneratedSolution>& solutions) const override
        {
            log.add("generate");
            {
                std::lock_guard<std::mutex> lock(mutex);
                count          = request.count;
                workspaceLimit = request.workspaceLimit;
                excludeKernels = request.excludeKernels;
                target         = &request.target;
                scratches.push_back(request.scratch);
            }
            require(dynamic_cast<const ProbeRequest*>(&request.request) != nullptr,
                    "Jit did not forward the request");
            require(fs::is_directory(request.scratch) && fs::is_empty(request.scratch),
                    "Scratch is not a fresh directory");
            if(writes)
                std::ofstream(request.scratch / "generator.log") << "log\n";
            if(allocationFails)
                throw std::bad_alloc();
            if(throws)
                throw std::runtime_error("deliberate generator exception");
            if(!result.ok())
                return result;
            for(const auto& kernel : kernels)
                solutions.push_back(
                    {{1, 2, 3}, kernel, {{hj::BuildUnit::Role::Main, kernel, {4}}}});
            return result;
        }
    };

    struct Builder final : hj::CodeObjectBuilder
    {
        Log&                                       log;
        std::set<std::string>                      failures;
        mutable std::mutex                         mutex;
        mutable std::vector<hj::GeneratedSolution> received; // in build order
        explicit Builder(Log& l, std::set<std::string> f = {})
            : log(l)
            , failures(std::move(f))
        {
        }
        hj::Status build(const hj::GeneratedSolution& solution,
                         const hj::GenerationRequest&,
                         hj::BuiltSolution& built) const override
        {
            log.add("build:" + solution.kernelName);
            {
                std::lock_guard<std::mutex> lock(mutex);
                received.push_back(solution);
            }
            if(failures.count(solution.kernelName))
                return {Code::Failed, Stage::Generate, "build failed " + solution.kernelName};
            built.generated    = solution;
            built.object.bytes = solution.units.at(0).bytes;
            return {};
        }
    };

    struct Loader final : hj::SolutionLoader
    {
        Log&                  log;
        std::set<std::string> rejects, throwing, failures, empty;
        explicit Loader(Log& l)
            : log(l)
        {
        }
        hj::Status support(const hj::BuiltSolution& built,
                           const hj::OperationRequest&,
                           const hj::DeviceTarget&,
                           size_t) const override
        {
            const auto& kernel = built.generated.kernelName;
            log.add("support:" + kernel);
            if(throwing.count(kernel))
                throw std::runtime_error("support threw " + kernel);
            if(rejects.count(kernel))
                return {Code::NotSupported, Stage::Load, "rejected " + kernel};
            return {};
        }
        hj::Status load(const hj::BuiltSolution& built,
                        const hj::OperationRequest&,
                        const hj::DeviceTarget&,
                        size_t,
                        std::shared_ptr<const hj::KernelBundle>& bundle) const override
        {
            const auto& kernel = built.generated.kernelName;
            log.add("load:" + kernel);
            if(failures.count(kernel))
                return {Code::Failed, Stage::Support, "load failed " + kernel};
            if(!empty.count(kernel))
                bundle = std::make_shared<ProbeBundle>(kernel);
            return {};
        }
    };

    struct Store final : hj::SolutionStore
    {
        Log&       log;
        hj::Status result;
        bool       shortIndices = false;
        explicit Store(Log& l)
            : log(l)
        {
        }
        hj::Status lookup(const hj::OperationRequest&,
                          const hj::DeviceTarget&,
                          size_t,
                          size_t,
                          const std::vector<std::string>&,
                          std::vector<int32_t>& indices) const override
        {
            log.add("lookup");
            indices.clear();
            return {};
        }
        hj::Status publish(const hj::OperationRequest&,
                           const hj::DeviceTarget&,
                           const std::vector<hj::BuiltSolution>& solutions,
                           std::vector<int32_t>&                 indices) const override
        {
            log.add("publish:" + std::to_string(solutions.size()));
            if(!result.ok())
                return result;
            for(size_t i = shortIndices ? 1 : 0; i < solutions.size(); ++i)
                indices.push_back(100 + static_cast<int32_t>(i));
            return {};
        }
    };

    // A Jit over fakes; tests adjust the fakes before calling run().
    struct Fixture
    {
        Log                      log;
        std::shared_ptr<Backend> backend;
        std::shared_ptr<Builder> builder = std::make_shared<Builder>(log);
        std::shared_ptr<Loader>  loader  = std::make_shared<Loader>(log);
        std::shared_ptr<Store>   store;
        ProbeRequest             request;
        hj::DeviceTarget         target;

        explicit Fixture(std::vector<std::string> kernels)
            : backend(std::make_shared<Backend>(log, std::move(kernels)))
        {
            target.device = 0;
            target.isa    = "gfx950";
        }
        hj::Jit jit() const
        {
            return hj::Jit({backend, builder, loader, store});
        }
        hj::Jit::Outcome run(size_t                          count   = 1,
                             const std::vector<std::string>& exclude = {}) const
        {
            return jit().generate(request, target, count, 4096, exclude);
        }
    };

    std::vector<std::string> names(const hj::Jit::Outcome& outcome)
    {
        std::vector<std::string> result;
        for(const auto& bundle : outcome.unpublished)
            result.push_back(bundle->name());
        return result;
    }

    void failure(const hj::Jit::Outcome& outcome,
                 Code                    code,
                 Stage                   stage,
                 const std::string&      message,
                 const std::string&      label)
    {
        require(!outcome.failures.empty(), label + ": no failure recorded");
        const auto& status = outcome.failures.front();
        require(status.code == code && status.stage == stage
                    && status.message.find(message) != std::string::npos,
                label + ": unexpected failure '" + status.message + "' at stage "
                    + std::to_string(static_cast<int>(status.stage)));
    }

    size_t scratchCount(const fs::path& parent)
    {
        size_t count = 0;
        for(const auto& entry : fs::directory_iterator(parent))
            count += entry.path().filename().string().rfind("hipblaslt-jit-", 0) == 0;
        return count;
    }

    void construction()
    {
        Log  log;
        auto backend  = std::make_shared<Backend>(log, std::vector<std::string>{});
        auto builder  = std::make_shared<Builder>(log);
        auto loader   = std::make_shared<Loader>(log);
        auto rejected = [](hj::Jit::Components components) {
            try
            {
                hj::Jit jit(std::move(components));
            }
            catch(const std::invalid_argument&)
            {
                return true;
            }
            return false;
        };
        require(rejected({nullptr, builder, loader, nullptr})
                    && rejected({backend, nullptr, loader, nullptr})
                    && rejected({backend, builder, nullptr, nullptr}),
                "Jit accepted a missing backend, builder or loader");
        hj::Jit plain({backend, builder, loader, nullptr});
        require(plain.components().backend == backend, "Jit did not keep its components");
        std::cout << "PASS Jit construction validates components\n";
    }

    void pipeline()
    {
        {
            Fixture f({"a"});
            require(f.run(0).unpublished.empty() && f.log.calls.empty(),
                    "A zero count reached the backend");
        }
        {
            Fixture f({"a", "b", "c", "d"});
            f.loader->rejects = {"b"};
            const auto outcome = f.run(2, {"x", "y"});
            require(names(outcome) == std::vector<std::string>{"a", "c"},
                    "Count limiting or support rejection dropped the wrong solutions");
            failure(outcome, Code::NotSupported, Stage::Support, "rejected b", "Support rejection");
            require(outcome.failures.size() == 1 && f.log.count("build:d") == 0,
                    "Jit built solutions beyond the requested count");
            require(f.backend->count == 2 && f.backend->workspaceLimit == 4096
                        && f.backend->excludeKernels == std::vector<std::string>{"x", "y"}
                        && f.backend->target == &f.target,
                    "Jit did not forward count, workspace, exclusions and target");
            require(outcome.summary == "generated", "Jit lost the backend's success note");
            const auto& received = f.builder->received;
            require(received.size() == 3
                        && std::all_of(received.begin(),
                                       received.end(),
                                       [](const hj::GeneratedSolution& solution) {
                                           const auto& units = solution.units;
                                           return solution.entry == std::vector<uint8_t>{1, 2, 3}
                                                  && units.size() == 1
                                                  && units[0].role == hj::BuildUnit::Role::Main
                                                  && units[0].name == solution.kernelName
                                                  && units[0].bytes == std::vector<uint8_t>{4};
                                       }),
                    "The builder did not receive exactly the generator's entry and units");
        }
        std::cout << "PASS count limiting, excludeKernels forwarding, the generator's units "
                     "reach the builder, one support rejection keeps the rest\n";

        {
            Fixture f({"a", "b"});
            f.store = std::make_shared<Store>(f.log);
            const auto outcome = f.run(2);
            require(outcome.indices == std::vector<int32_t>{100, 101} && outcome.unpublished.empty()
                        && outcome.failures.empty() && f.log.count("load:") == 0,
                    "A successful publish still loaded solutions");
        }
        {
            Fixture f({"a", "b"});
            const auto outcome = f.run(2);
            require(names(outcome) == std::vector<std::string>{"a", "b"}
                        && f.log.count("load:") == 2,
                    "Jit without a store did not load");
        }
        {
            Fixture f({"a", "b"});
            f.store         = std::make_shared<Store>(f.log);
            f.store->result = {Code::Failed, Stage::Configure, "store unavailable"};
            const auto outcome = f.run(2);
            failure(outcome, Code::Failed, Stage::Publish, "store unavailable", "Publish failure");
            require(outcome.indices.empty() && names(outcome) == std::vector<std::string>{"a", "b"},
                    "A failed publish did not fall back to loading");
        }
        {
            Fixture f({"a", "b"});
            f.store               = std::make_shared<Store>(f.log);
            f.store->shortIndices = true;
            const auto outcome    = f.run(2);
            failure(outcome, Code::Failed, Stage::Publish, "1 indices for 2", "Short publish");
            require(outcome.indices.empty() && f.log.count("load:") == 2,
                    "A short publish was accepted");
        }
        {
            Fixture f({"a"});
            f.store = std::make_shared<Store>(f.log);
            f.loader->rejects = {"a"};
            const auto outcome = f.run();
            require(f.log.count("publish:") == 0 && f.log.count("load:") == 0,
                    "Nothing supported, yet Jit published or loaded");
        }
        std::cout << "PASS load runs only without a store or after a publish failure\n";
    }

    void stages()
    {
        const std::pair<hj::Status, Stage> generated[]
            = {{{Code::TargetMismatch, Stage::Build, "other target"}, Stage::Configure},
               {{Code::NotSupported, Stage::Build, "not mine"}, Stage::Generate},
               {{Code::Failed, Stage::Load, "generator failed"}, Stage::Generate}};
        for(const auto& [status, stage] : generated)
        {
            Fixture f({"a"});
            f.backend->result  = status;
            const auto outcome = f.run();
            failure(outcome, status.code, stage, status.message, "Generate");
            require(f.log.count("build:") == 0, "Jit built after a failed generation");
        }
        {
            Fixture f({"a"});
            f.backend->throws  = true;
            const auto outcome = f.run();
            failure(outcome, Code::Failed, Stage::Generate, "deliberate generator", "Throw");
        }
        {
            Fixture f({"a", "b"});
            f.builder = std::make_shared<Builder>(f.log, std::set<std::string>{"a"});
            const auto outcome = f.run();
            failure(outcome, Code::Failed, Stage::Build, "build failed a", "Build");
            require(names(outcome) == std::vector<std::string>{"b"},
                    "A build failure dropped the other solutions");
        }
        {
            Fixture f({"a"});
            f.loader->throwing = {"a"};
            const auto outcome = f.run();
            failure(outcome, Code::Failed, Stage::Support, "support threw a", "Support");
        }
        {
            Fixture f({"a", "b"});
            f.loader->failures = {"a"};
            f.loader->empty    = {"b"};
            const auto outcome = f.run(2);
            failure(outcome, Code::Failed, Stage::Load, "load failed a", "Load");
            require(outcome.failures.size() == 2 && outcome.failures[1].stage == Stage::Load
                        && outcome.failures[1].message == "Loader returned no bundle"
                        && outcome.unpublished.empty(),
                    "A missing bundle was not a load failure");
        }
        {
            Fixture f({"a"});
            f.backend->allocationFails = true;
            bool thrown                = false;
            try
            {
                f.run();
            }
            catch(const std::bad_alloc&)
            {
                thrown = true;
            }
            require(thrown, "Allocation failure did not propagate");
        }
        std::cout << "PASS failures are attributed to the stage that produced them\n";
    }

    void scratch(const fs::path& parent)
    {
        const auto before = scratchCount(parent);
        {
            Fixture f({"a"});
            f.run();
            require(!fs::exists(f.backend->scratches.at(0))
                        && f.backend->scratches[0].parent_path() == parent,
                    "Scratch was not a removed directory under TMPDIR after success");
        }
        {
            Fixture f({"a"});
            f.builder = std::make_shared<Builder>(f.log, std::set<std::string>{"a"});
            f.run();
            require(fs::exists(f.backend->scratches.at(0) / "generator.log"),
                    "Scratch was not kept after a failure");
            fs::remove_all(f.backend->scratches[0]);
        }
        {
            Fixture f({"a"});
            f.backend->writes = false;
            f.backend->result = {Code::Failed, Stage::Generate, "failed"};
            f.run();
            require(!fs::exists(f.backend->scratches.at(0)), "An empty scratch was kept");
        }
        require(scratchCount(parent) == before, "Scratch directories leaked");
        std::cout << "PASS scratch removed on success, kept on failure unless empty\n";
    }

    void concurrency(const fs::path& parent)
    {
        Fixture                  f({"a", "b"});
        const auto               jit    = f.jit();
        const auto               before = scratchCount(parent);
        std::atomic<int>         passed{0};
        std::vector<std::thread> threads;
        for(int t = 0; t < 8; ++t)
            threads.emplace_back([&] {
                for(int i = 0; i < 25; ++i)
                {
                    const auto outcome = jit.generate(f.request, f.target, 2, 4096, {});
                    if(outcome.failures.empty()
                       && names(outcome) == std::vector<std::string>{"a", "b"})
                        ++passed;
                }
            });
        for(auto& thread : threads)
            thread.join();
        const std::set<fs::path> unique(f.backend->scratches.begin(), f.backend->scratches.end());
        require(passed == 200 && unique.size() == 200 && scratchCount(parent) == before,
                "Concurrent generation interfered: " + std::to_string(passed) + " passed, "
                    + std::to_string(unique.size()) + " scratch directories");
        std::cout << "PASS 8 threads generate concurrently with private scratch directories\n";
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " FRESH_SCRATCH_PARENT\n";
        return 2;
    }
    try
    {
        const fs::path parent = fs::absolute(argv[1]);
        require(fs::create_directories(parent), "Scratch parent already exists");
#ifdef _WIN32
        require(_putenv_s("TMP", parent.string().c_str()) == 0, "Cannot set TMP");
#else
        require(setenv("TMPDIR", parent.c_str(), 1) == 0, "Cannot set TMPDIR");
#endif
        require(fs::temp_directory_path() == parent, "Scratch parent is not the temp directory");
        construction();
        pipeline();
        stages();
        scratch(parent);
        concurrency(parent);
        std::cout << "ALL JIT COMPONENT CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
