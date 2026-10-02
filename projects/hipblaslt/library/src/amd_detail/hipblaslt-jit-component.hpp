// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <hip/hip_runtime_api.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace TensileLite
{
    class Hardware;
}

// Compiled-in stages of JIT solution generation. This is not a plugin ABI.
namespace hipblaslt_jit
{
    using OperationRequest = hipblaslt_ext::experimental::jit::detail::OperationRequest;
    using KernelBundle     = hipblaslt_ext::experimental::jit::detail::KernelBundle;

    enum class Stage
    {
        Configure,
        Predict,
        Generate,
        Build,
        Support,
        Load,
        Lookup,
        Publish,
    };

    // "configure", "predict", ... as reports and debug lines name stages.
    const char* toString(Stage stage) noexcept;

    struct Status
    {
        enum class Code
        {
            Success,
            NotSupported, // outside the component's domain; not a defect
            TargetMismatch, // configured for another device target
            Failed,
        };
        Code        code  = Code::Success;
        Stage       stage = Stage::Configure; // assigned by Jit
        std::string message; // one line
        std::string logPath; // kept for diagnosis, or empty
        bool        ok() const noexcept
        {
            return code == Code::Success;
        }
    };

    struct DeviceTarget
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        std::string     targetId; // gcnArchName, e.g. "gfx950:sramecc+:xnack-"
        std::string     isa; // targetId up to the first ':'
        std::string     libraryArch; // GEMM library subtree, e.g. "gfx1250v0"
        int             wavefrontSize = 0;
        int             cuCount       = 0;
        std::shared_ptr<TensileLite::Hardware> hardware;

        static Status make(int device, DeviceTarget& target);
    };

    // A named tuning parameter whose value is a JSON literal in the backend's vocabulary.
    struct TuningParameter
    {
        std::string name;
        std::string json;
    };

    struct Candidate
    {
        uint32_t                     id              = 0; // stable within one Prediction
        double                       predictedCycles = 0;
        std::vector<TuningParameter> parameters; // forwarded to the backend
        std::vector<TuningParameter> modeled; // recorded, not forwarded
    };

    struct Prediction
    {
        std::string                  modeledContract;
        std::string                  model;
        std::vector<TuningParameter> hardware; // device facts the model used
        std::vector<TuningParameter> assumptions;
        std::vector<Candidate>       ranked; // best first
    };

    // A family of candidates for a predictor to rank.
    struct CandidateSeed
    {
        std::array<size_t, 4> tile{}; // macro tile M and N, waves M and N
        // Each {minimum, multiple} rule gives DepthU = max(minimum, multiple * instruction K).
        std::vector<std::array<size_t, 2>> depthRules;
        std::vector<std::array<int, 2>>    cacheHints; // {NonTemporalA, NonTemporalB}
    };

    class TuningKnowledge
    {
    public:
        virtual ~TuningKnowledge()                   = default;
        virtual std::string_view           id() const noexcept = 0;
        virtual std::vector<CandidateSeed> seeds(const OperationRequest&,
                                                 const DeviceTarget&) const
            = 0;
        // Values for knobs the model does not predict. Empty leaves them to the backend.
        virtual std::vector<TuningParameter>
            defaults(const OperationRequest&, const DeviceTarget&, const Candidate&) const = 0;
    };

    class Predictor
    {
    public:
        virtual ~Predictor()                                      = default;
        virtual std::string_view id() const noexcept              = 0;
        virtual std::string_view modeledContract() const noexcept = 0;
        virtual Status           predict(const OperationRequest&,
                                         const DeviceTarget&,
                                         const TuningKnowledge&,
                                         Prediction&) const
            = 0;
    };

    std::shared_ptr<const Predictor>       makeOrigamiPredictor();
    std::shared_ptr<const TuningKnowledge> makeTensileLiteDefaults();

    // A header a HIP unit includes by name.
    struct IncludeFile
    {
        std::string          name;
        std::vector<uint8_t> bytes;
    };

    // A source file the builder assembles or compiles.
    struct BuildUnit
    {
        enum class Role
        {
            Main,
            Helper,
        };
        enum class Kind
        {
            Assembly,
            Hip,
        };
        Role                     role = Role::Main;
        std::string              name;
        std::vector<uint8_t>     bytes;
        Kind                     kind = Kind::Assembly;
        std::vector<IncludeFile> includes; // Hip only
    };

    // One CustomKernel library entry plus the code that defines it.
    struct GeneratedSolution
    {
        // A one-solution TensileLite library; its solution carries the customKernel record.
        std::vector<uint8_t>   entry;
        std::string            kernelName; // the entry's main kernel
        std::vector<BuildUnit> units;
    };

    constexpr int jitCodeObjectVersion = 4;

    struct GenerationRequest
    {
        const OperationRequest&  request;
        const DeviceTarget&      target;
        const Prediction*        prediction = nullptr; // set only for backends that consume one
        size_t                   count      = 1;
        size_t                   workspaceLimit = 0;
        std::vector<std::string> excludeKernels; // kernels the caller already has
        std::filesystem::path    scratch; // private directory owned by this call
        int codeObjectVersion = jitCodeObjectVersion; // for generators and the builder
    };

    struct BackendInfo
    {
        std::string id;
        std::string name; // reported as Diagnostics::backend
        std::string consumesPrediction; // modeled contract, or empty
        std::string version; // changes whenever the generated solutions can change
    };

    class Backend
    {
    public:
        virtual ~Backend()                               = default;
        virtual const BackendInfo& info() const noexcept = 0;
        // Returns up to request.count solutions and loads nothing. NotSupported
        // means the request is outside the backend's domain.
        virtual Status generate(const GenerationRequest&, std::vector<GeneratedSolution>&) const
            = 0;
    };

    struct CodeObject
    {
        std::vector<uint8_t> bytes;
    };

    struct BuiltSolution
    {
        GeneratedSolution       generated;
        CodeObject              object;
        std::vector<CodeObject> helpers; // loaded next to the main object
    };

    class CodeObjectBuilder
    {
    public:
        virtual ~CodeObjectBuilder() = default;
        virtual Status
            build(const GeneratedSolution&, const GenerationRequest&, BuiltSolution&) const = 0;
    };

    // Builds every unit of a solution with comgr and links them into one code
    // object for GenerationRequest::target.
    std::shared_ptr<const CodeObjectBuilder> makeComgrBuilder();

    class SolutionLoader
    {
    public:
        virtual ~SolutionLoader() = default;
        // Evaluates the entry's predicates and workspace for the request. Loads no code.
        virtual Status support(const BuiltSolution&,
                               const OperationRequest&,
                               const DeviceTarget&,
                               size_t workspaceLimit) const
            = 0;
        // Loads the code objects into a process-local executable bundle.
        virtual Status load(const BuiltSolution&,
                            const OperationRequest&,
                            const DeviceTarget&,
                            size_t                               workspaceLimit,
                            std::shared_ptr<const KernelBundle>& bundle) const
            = 0;
    };

    class SolutionStore
    {
    public:
        virtual ~SolutionStore() = default;
        // Up to count indices of stored solutions for exactly this request that
        // need at most maxWorkspaceBytes, best first, without excluded kernels.
        virtual Status lookup(const OperationRequest&,
                              const DeviceTarget&,
                              size_t                          count,
                              size_t                          maxWorkspaceBytes,
                              const std::vector<std::string>& excludeKernels,
                              std::vector<int32_t>&           indices) const
            = 0;
        // Returns one library index per solution, in order.
        virtual Status publish(const OperationRequest&,
                               const DeviceTarget&,
                               const std::vector<BuiltSolution>&,
                               std::vector<int32_t>& indices) const
            = 0;
    };

    class Jit
    {
    public:
        struct Components
        {
            std::shared_ptr<const Backend>           backend;
            std::shared_ptr<const Predictor>         predictor; // iff the backend consumes predictions
            std::shared_ptr<const TuningKnowledge>   knowledge; // iff predictor
            std::shared_ptr<const CodeObjectBuilder> builder;
            std::shared_ptr<const SolutionLoader>    loader;
            std::shared_ptr<const SolutionStore>     store; // optional
        };

        struct Outcome
        {
            std::vector<int32_t>                             indices; // published
            std::vector<std::shared_ptr<const KernelBundle>> unpublished; // process-local
            std::vector<Status>                              failures; // in the order they happened
            std::string                                      summary; // the backend's success note
        };

        // Throws std::invalid_argument when a required component is missing or
        // the predictor's contract differs from the backend's.
        explicit Jit(Components components);

        // Predict, generate, build and check support, then publish, or load when
        // there is no store or publishing failed. Returns at most count
        // solutions that support the request. Thread-safe.
        Outcome generate(const OperationRequest&         request,
                         const DeviceTarget&             target,
                         size_t                          count,
                         size_t                          workspaceLimit,
                         const std::vector<std::string>& excludeKernels) const;

        const Components& components() const noexcept
        {
            return m_components;
        }

    private:
        Components m_components;
    };
}

namespace hipblaslt_ext::experimental::jit::detail
{
    struct CompiledSolution
    {
        hipblaslt_jit::DeviceTarget               target;
        std::shared_ptr<const OperationRequest>   request;
        std::shared_ptr<const hipblaslt_jit::Jit> jit;
        std::shared_ptr<const KernelBundle>       bundle;
        uint64_t                                  process        = 0;
        size_t                                    workspaceLimit = 0;
        size_t                                    workspaceBytes = 0;
    };
}
