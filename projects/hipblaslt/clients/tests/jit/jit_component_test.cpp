// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include "jit_test_child.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <sstream>
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

    struct Knowledge final : hj::TuningKnowledge
    {
        Log& log;
        explicit Knowledge(Log& l)
            : log(l)
        {
        }
        std::string_view id() const noexcept override
        {
            return "fake-knowledge";
        }
        std::string version() const override
        {
            return "3";
        }
        std::vector<hj::CandidateSeed> seeds(const hj::OperationRequest&,
                                             const hj::DeviceTarget&) const override
        {
            log.add("seeds");
            return {{{64, 64, 2, 2}, {{32, 1}}, {{0, 0}}}};
        }
        std::vector<hj::TuningParameter> defaults(const hj::OperationRequest&,
                                                  const hj::DeviceTarget&,
                                                  const hj::Candidate&) const override
        {
            return {};
        }
    };

    struct Predictor final : hj::Predictor
    {
        Log&               log;
        hj::Status         result;
        mutable size_t     workspaceLimit = 0;
        std::string        other; // the contract of a second candidate, when set
        Predictor(Log& l, hj::Status r = {})
            : log(l)
            , result(std::move(r))
        {
        }
        std::string_view id() const noexcept override
        {
            return "fake-model";
        }
        std::set<std::string> modeledContracts() const override
        {
            return {"fake.v1", "spare.v1"};
        }
        hj::Status predict(const hj::PredictionRequest& request,
                           const hj::TuningKnowledge&   knowledge,
                           hj::Prediction&              prediction) const override
        {
            log.add("predict");
            workspaceLimit = request.workspaceLimit;
            if(!result.ok())
                return result;
            const auto seeds = knowledge.seeds(request.request, request.target);
            prediction.modeledContract = "fake.v1";
            if(!other.empty())
            {
                prediction.ranked.push_back({6, 1.0, {{"DepthU", "0"}}, {}});
                prediction.ranked.back().contract = other;
            }
            prediction.ranked.push_back({7, 1.0, {{"DepthU", std::to_string(seeds.size())}}, {}});
            return {};
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
        mutable const hj::Prediction*    prediction = nullptr;
        mutable std::string              candidate;
        mutable std::vector<uint32_t>    ranked;
        mutable std::vector<fs::path>    scratches;
        mutable const hj::DeviceTarget*  target = nullptr;

        Backend(Log& l, std::vector<std::string> k, std::set<std::string> contracts = {})
            : log(l)
            , information{"fake-backend", "Fake", std::move(contracts)}
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
                prediction     = request.prediction;
                candidate      = request.prediction && !request.prediction->ranked.empty()
                                     ? request.prediction->ranked[0].parameters.at(0).json
                                     : "";
                ranked.clear();
                if(request.prediction)
                    for(const auto& c : request.prediction->ranked)
                        ranked.push_back(c.id);
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
        Log&                  log;
        std::set<std::string> failures;
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
        Log&        log;
        hj::Status  result;
        bool        shortIndices = false;
        std::string version; // the one the Jit made this store for
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
        Log                        log;
        std::shared_ptr<Backend>   backend;
        std::shared_ptr<Predictor> predictor;
        std::shared_ptr<Knowledge> knowledge = std::make_shared<Knowledge>(log);
        std::shared_ptr<Builder>   builder   = std::make_shared<Builder>(log);
        std::shared_ptr<Loader>    loader    = std::make_shared<Loader>(log);
        std::shared_ptr<Store>     store;
        ProbeRequest               request;
        hj::DeviceTarget           target;

        explicit Fixture(std::vector<std::string> kernels, std::set<std::string> contracts = {})
            : backend(std::make_shared<Backend>(log, std::move(kernels), std::move(contracts)))
            , predictor(std::make_shared<Predictor>(log))
        {
            target.device = 0;
            target.isa    = "gfx950";
        }
        hj::Jit jit() const
        {
            hj::Jit::StoreFactory factory;
            if(store)
                factory = [s = store](const hj::BackendInfo&, const std::string& version) {
                    s->version = version;
                    return s;
                };
            return hj::Jit({backend, predictor, knowledge, builder, loader, factory});
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
        auto backend   = std::make_shared<Backend>(log, std::vector<std::string>{});
        auto predicted = std::make_shared<Backend>(
            log, std::vector<std::string>{}, std::set<std::string>{"fake.v1", "other.v1"});
        auto other = std::make_shared<Backend>(
            log, std::vector<std::string>{}, std::set<std::string>{"other.v1"});
        auto predictor = std::make_shared<Predictor>(log);
        auto knowledge = std::make_shared<Knowledge>(log);
        auto builder   = std::make_shared<Builder>(log);
        auto loader    = std::make_shared<Loader>(log);
        auto rejected  = [](hj::Jit::Components components) {
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
        require(rejected({nullptr, nullptr, nullptr, builder, loader, nullptr})
                    && rejected({backend, nullptr, nullptr, nullptr, loader, nullptr})
                    && rejected({backend, nullptr, nullptr, builder, nullptr, nullptr}),
                "Jit accepted a missing backend, builder or loader");
        require(rejected({predicted, nullptr, knowledge, builder, loader, nullptr})
                    && rejected({predicted, predictor, nullptr, builder, loader, nullptr})
                    && rejected({other, predictor, knowledge, builder, loader, nullptr}),
                "Jit accepted a prediction contract it cannot satisfy");
        backend->information.version   = "b1";
        predicted->information.version = "b2";
        auto        store              = std::make_shared<Store>(log);
        std::string storeBackend;
        auto        factory = [&](const hj::BackendInfo& info, const std::string& version) {
            storeBackend   = info.id;
            store->version = version;
            return store;
        };
        hj::Jit plain({backend, nullptr, nullptr, builder, loader, nullptr});
        hj::Jit modeled({predicted, predictor, knowledge, builder, loader, factory});
        require(plain.components().backend == backend, "Jit did not keep its components");
        const std::string composed
            = "b2|predictor=fake-model;contracts=fake.v1|knowledge=fake-knowledge@3";
        require(plain.version() == "b1" && !plain.store() && modeled.version() == composed
                    && modeled.store() == store && store->version == composed
                    && storeBackend == "fake-backend",
                "Jit did not make its store under the composed version");
        std::cout << "PASS Jit construction validates components and prediction contracts, and "
                     "makes the store under the composed version\n";
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
        }
        std::cout << "PASS count limiting, excludeKernels forwarding, one support rejection "
                     "keeps the rest\n";

        {
            Fixture f({"a"});
            const auto outcome = f.run();
            require(f.log.count("predict") == 0 && f.backend->prediction == nullptr
                        && names(outcome) == std::vector<std::string>{"a"},
                    "Predict ran for a backend that consumes no prediction");
        }
        {
            Fixture f({"a"}, {"fake.v1"});
            const auto outcome = f.run();
            require(f.log.count("predict") == 1 && f.log.count("seeds") == 1
                        && f.backend->prediction != nullptr && f.backend->candidate == "1"
                        && f.predictor->workspaceLimit == 4096
                        && names(outcome) == std::vector<std::string>{"a"},
                    "The backend did not receive the prediction");
        }
        std::cout << "PASS Predict runs only for backends that consume a prediction\n";

        {
            Fixture f({"a"}, {"fake.v1"});
            f.predictor->other = "spare.v1";
            f.run();
            require(f.backend->ranked == std::vector<uint32_t>{7},
                    "A candidate the backend does not transport reached it");
            f.backend->information.contracts = {"spare.v1"};
            f.predictor->other               = "";
            failure(f.run(),
                    Code::NotSupported,
                    Stage::Predict,
                    "No predicted candidate",
                    "No transported candidate");
            require(f.log.count("generate") == 1, "Generation ran without a candidate");
        }
        std::cout << "PASS Jit keeps only candidates whose contract the backend transports\n";

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
        {
            Fixture f({"a"}, {"fake.v1"});
            f.predictor = std::make_shared<Predictor>(
                f.log, hj::Status{Code::NotSupported, Stage::Generate, "not modeled"});
            const auto outcome = f.run();
            failure(outcome, Code::NotSupported, Stage::Predict, "not modeled", "Predict");
            require(f.log.count("generate") == 0, "Generation ran after a failed prediction");
        }
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

    void defaults()
    {
        const auto        knowledge = hj::makeCatalogKnowledge();
        const ProbeRequest request;
        require(knowledge->id() == "catalog.v1", "Unexpected knowledge id");
        const std::vector<std::array<size_t, 4>> tiles{{32, 32, 2, 2},
                                                       {64, 32, 2, 2},
                                                       {32, 64, 2, 2},
                                                       {64, 64, 2, 2},
                                                       {128, 64, 2, 2},
                                                       {64, 128, 2, 2},
                                                       {128, 128, 2, 2},
                                                       {256, 128, 2, 2},
                                                       {128, 256, 2, 2},
                                                       {256, 256, 2, 2},
                                                       {128, 16, 4, 1}};
        const std::vector<std::array<size_t, 2>> depthRules{{32, 1}, {64, 2}};
        const std::vector<std::array<int, 2>>    defaultHints{{0, 0}};
        const std::vector<std::array<int, 2>>    allHints{{0, 0}, {4, 0}, {0, 4}};
        for(const auto* isa : {"gfx90a", "gfx942", "gfx950", "gfx1250"})
        {
            hj::DeviceTarget target;
            target.isa       = isa;
            const auto seeds = knowledge->seeds(request, target);
            const auto& hints
                = target.isa == "gfx90a" || target.isa == "gfx1250" ? defaultHints : allHints;
            require(seeds.size() == tiles.size(), std::string(isa) + ": wrong seed count");
            for(size_t i = 0; i < seeds.size(); ++i)
                require(seeds[i].tile == tiles[i] && seeds[i].depthRules == depthRules
                            && seeds[i].cacheHints == hints && !seeds[i].instruction
                            && seeds[i].policies.size() == 1
                            && seeds[i].policies[0].strategy == hj::ExecutionPolicy::Strategy::None,
                        std::string(isa) + ": wrong seed " + std::to_string(i));
            require(knowledge->defaults(request, target, {}).empty(),
                    "Catalog knowledge supplied parameter values");
        }
        std::cout << "PASS catalog knowledge seeds 11 tiles with per-architecture cache hints\n";
    }

    // Runs in a child with HIPBLASLT_JIT_DEBUG=all; each scenario's lines follow its name.
    void debugChild()
    {
        std::cerr << "scenario published" << std::endl;
        {
            Fixture f({"a", "b", "c"}, {"fake.v1"});
            f.builder = std::make_shared<Builder>(f.log, std::set<std::string>{"b"});
            f.store   = std::make_shared<Store>(f.log);
            require(f.run(3).indices == std::vector<int32_t>{100, 101}, "published: wrong indices");
        }
        std::cerr << "scenario loaded" << std::endl;
        {
            Fixture f({"a", "b"});
            f.loader->rejects = {"b"};
            require(names(f.run(2)) == std::vector<std::string>{"a"}, "loaded: wrong bundles");
        }
        std::cerr << "scenario failed" << std::endl;
        {
            Fixture f({"a"});
            f.backend->result = {Code::Failed, Stage::Generate, "no solution"};
            require(f.run().failures.size() == 1, "failed: no failure");
        }
        std::cerr << "scenario end" << std::endl;
    }

    std::string field(const std::string& line, const std::string& key)
    {
        const auto at = line.find("\"" + key + "\":");
        if(at == std::string::npos)
            return {};
        const auto first = at + key.size() + 3;
        if(line[first] == '"')
            return line.substr(first + 1, line.find('"', first + 1) - first - 1);
        return line.substr(first, line.find_first_of(",}", first) - first);
    }

    void debugLines(const fs::path& parent, const fs::path& self)
    {
        const auto tmp = parent / "debug-tmp";
        fs::create_directory(tmp);
        const auto log = parent / "debug-child.log";
        const bool ran = hipblaslt_jit_test::runChild({self.string(), "--debug-child"},
                                                      {{"HIPBLASLT_JIT", "1"},
                                                       {"HIPBLASLT_JIT_DEBUG", "all"},
                                                       {"HIPBLASLT_JIT_DEBUG_FILE", ""},
                                                       {"TMPDIR", tmp.string()}},
                                                      parent,
                                                      log);
        std::ifstream     in(log);
        const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
        require(ran, "The debug child failed: " + text);

        std::map<std::string, std::vector<std::string>> scenarios;
        std::string                                     scenario;
        std::istringstream                              lines(text);
        for(std::string line; std::getline(lines, line);)
            if(line.rfind("scenario ", 0) == 0)
                scenario = line.substr(9);
            else if(line.rfind("hipblaslt jit-debug {\"v\":1,", 0) == 0)
                scenarios[scenario].push_back(line);
        auto events = [&](const std::string& name) {
            std::vector<std::string> out;
            for(const auto& line : scenarios[name])
                out.push_back(field(line, "ev")
                              + (field(line, "ev") == "solution" || field(line, "ev") == "build.end"
                                     ? ":" + field(line, "kernel") + ":" + field(line, "outcome")
                                     : ""));
            return out;
        };
        auto find = [&](const std::string& name, const std::string& event) {
            for(const auto& line : scenarios[name])
                if(field(line, "ev") == event)
                    return line;
            throw std::runtime_error(name + ": no " + event + " line");
        };
        auto has = [](const std::string& line, const std::string& fragment) {
            return line.find(fragment) != std::string::npos;
        };

        const std::vector<std::string> published = {"process",
                                                    "generation.start",
                                                    "build.start",
                                                    "build.end:a:built",
                                                    "build.start",
                                                    "build.end:b:build_failed",
                                                    "failure",
                                                    "build.start",
                                                    "build.end:c:built",
                                                    "publish.start",
                                                    "publish.done",
                                                    "solution:a:published",
                                                    "solution:b:build_failed",
                                                    "solution:c:published",
                                                    "generation",
                                                    "generation.end"};
        require(events("published") == published, "published: wrong events or order");
        const auto generation = find("published", "generation");
        require(field(generation, "requested") == "3" && field(generation, "generated") == "3"
                    && field(generation, "published") == "2"
                    && field(generation, "failures") == "1"
                    && field(generation, "candidates") == "1"
                    && has(generation, "\"ns\":{\"total\":")
                    && has(generation, "\"predict\":") && has(generation, "\"scratch\":")
                    && has(generation, "\"backend\":") && has(generation, "\"build\":")
                    && has(generation, "\"support\":") && has(generation, "\"publish\":")
                    && has(generation, "\"other\":") && !has(generation, "\"load\":")
                    && field(generation, "gen") != "" && field(generation, "q") == "null",
                "published: wrong generation line " + generation);
        require(field(find("published", "failure"), "stage") == "build"
                    && field(find("published", "generation.end"), "outcome") == "partial"
                    && field(find("published", "publish.done"), "solutions") == "2",
                "published: wrong failure, outcome or publish");
        for(const auto& line : scenarios["published"])
            if(field(line, "ev") == "solution" && field(line, "kernel") != "b")
                require(field(line, "index") == (field(line, "kernel") == "a" ? "100" : "101")
                            && has(line, "\"build\":") && has(line, "\"support\":"),
                        "published: wrong solution line " + line);

        const std::vector<std::string> loaded = {"generation.start",
                                                 "build.start",
                                                 "build.end:a:built",
                                                 "build.start",
                                                 "build.end:b:built",
                                                 "failure",
                                                 "load.done",
                                                 "solution:a:loaded",
                                                 "solution:b:unsupported",
                                                 "generation",
                                                 "generation.end"};
        require(events("loaded") == loaded, "loaded: wrong events or order");
        require(has(find("loaded", "generation"), "\"load\":")
                    && !has(find("loaded", "generation"), "\"predict\":")
                    && field(find("loaded", "failure"), "stage") == "support"
                    && field(find("loaded", "generation.end"), "outcome") == "partial"
                    && field(find("loaded", "generation.end"), "loaded") == "1",
                "loaded: wrong generation lines");

        const std::vector<std::string> failed
            = {"generation.start", "failure", "generation", "generation.end"};
        require(events("failed") == failed
                    && field(find("failed", "failure"), "stage") == "generate"
                    && field(find("failed", "generation.end"), "outcome") == "failed",
                "failed: wrong events");
        std::cout << "PASS HIPBLASLT_JIT_DEBUG times each stage and reports progress per solution\n";
    }
}

int main(int argc, char** argv)
{
    if(argc == 2 && std::string(argv[1]) == "--debug-child")
    {
        try
        {
            debugChild();
            return 0;
        }
        catch(const std::exception& error)
        {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
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
        defaults();
#ifndef _WIN32
        debugLines(parent, fs::read_symlink("/proc/self/exe"));
#endif
        std::cout << "ALL JIT COMPONENT CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
