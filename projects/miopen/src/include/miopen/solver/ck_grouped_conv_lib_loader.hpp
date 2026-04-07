// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/config.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if MIOPEN_BACKEND_HIP
#include <hip/hip_runtime_api.h>
#endif

struct CKKernelListHandle;

namespace miopen {
struct ExecutionContext;
namespace conv {
struct ProblemDescription;
} // namespace conv
namespace solver {
struct ConvSolution;

enum class CKSolverType
{
    GrpConvFwd           = 0,
    GrpConvBwd           = 1,
    GrpConvWrw           = 2,
    GrpConv3dFwd         = 3,
    GrpConv3dBwd         = 4,
    GrpConv3dWrw         = 5,
    FusedBiasActiv       = 6,
    FusedBiasResAddActiv = 7,
    FusedGrpActiv        = 8,
    FusedGrpBiasActiv    = 9,
    Count                = 10
};

/// Query the HIP runtime for the current device's architecture name.
/// Returns an empty string on failure or when not using the HIP backend.
inline std::string GetCurrentDeviceName()
{
#if MIOPEN_BACKEND_HIP
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
        return {};
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
        return {};
    return {props.gcnArchName};
#else
    return {};
#endif
}

class CKGroupedConvLibLoader
{
public:
    /// Thread-safe accessor returning a cached per-device singleton.
    MIOPEN_INTERNALS_EXPORT static const CKGroupedConvLibLoader&
    Get(const std::string& device_name);

    MIOPEN_INTERNALS_EXPORT bool IsLoaded() const { return loaded_; }

    // -- Solver-parameterized wrappers ----------------------------------------
    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    FillValidKernels(CKSolverType solver,
                     const miopen::conv::ProblemDescription& problem,
                     miopenDataType_t dtype,
                     bool use_tf32) const;

    /// Try tf32 first (if use_tf32 is true), then fall back to non-tf32.
    /// Updates use_tf32 to reflect whether tf32 was actually used.
    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    FillValidKernelsWithTf32Fallback(CKSolverType solver,
                                     const miopen::conv::ProblemDescription& problem,
                                     miopenDataType_t dtype,
                                     bool& use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool IsApplicable(CKSolverType solver,
                                              const miopen::conv::ProblemDescription& problem,
                                              miopenDataType_t dtype,
                                              bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT bool IsArgsSupported(CKSolverType solver,
                                                 const miopen::conv::ProblemDescription& problem,
                                                 const std::string& kernel_id,
                                                 miopenDataType_t dtype,
                                                 bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT size_t GetWorkspaceSize(CKSolverType solver,
                                                    const miopen::conv::ProblemDescription& problem,
                                                    miopenDataType_t dtype,
                                                    bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT ConvSolution
    GetSolution(CKSolverType solver,
                const ExecutionContext& ctx,
                const miopen::conv::ProblemDescription& problem,
                const std::string& kernel_id,
                bool use_tf32) const;

    MIOPEN_INTERNALS_EXPORT std::vector<std::string>
    GetAllKernelTypeStrings(CKSolverType solver) const;

    ~CKGroupedConvLibLoader();

    CKGroupedConvLibLoader(const CKGroupedConvLibLoader&)            = delete;
    CKGroupedConvLibLoader& operator=(const CKGroupedConvLibLoader&) = delete;

private:
    explicit CKGroupedConvLibLoader(const std::string& device_name);

    void OpenRuntimeLibraryForDevice(const std::string& device_name);
    bool LoadSymbols();
    void* ResolveRawSymbol(const char* symbol_name) const;
    void BindRequiredCommonSymbols(std::vector<std::string>& missing);
    void
    BindSolverSymbols(CKSolverType solver, const char* prefix, std::vector<std::string>& missing);
    void BindOptionalKernelTypeSymbols(std::vector<std::string>& missing);
    static constexpr int ToSolverIndex(CKSolverType solver) { return static_cast<int>(solver); }

    // Singleton cache
    static std::mutex& CacheMutex();
    static std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>>& Cache();

    void* lib_handle_ = nullptr;
    bool loaded_      = false;

    // -- Function pointer types -----------------------------------------------
    // CKKernelListHandle is declared at global scope in the interface header.

    using GetApiVersionFn = int (*)();

    using KernelListSizeFn = size_t (*)(const ::CKKernelListHandle*);
    using KernelListGetFn  = const char* (*)(const ::CKKernelListHandle*, size_t);
    using KernelListFreeFn = void (*)(::CKKernelListHandle*);

    using SolutionFreeFn = void (*)(ConvSolution*);

    using FillValidKernelsFn = ::CKKernelListHandle* (*)(const miopen::conv::ProblemDescription*,
                                                         miopenDataType_t,
                                                         bool);
    using IsApplicableFn     = bool (*)(const miopen::conv::ProblemDescription*,
                                    miopenDataType_t,
                                    bool);
    using IsArgsSupportedFn  = bool (*)(const miopen::conv::ProblemDescription*,
                                       const char*,
                                       miopenDataType_t,
                                       bool);
    using GetWorkspaceSizeFn = size_t (*)(const miopen::conv::ProblemDescription*,
                                          miopenDataType_t,
                                          bool);
    using GetSolutionFn      = ConvSolution* (*)(const ExecutionContext*,
                                            const miopen::conv::ProblemDescription*,
                                            const char*,
                                            bool);

    // -- Function pointers ----------------------------------------------------
    GetApiVersionFn get_api_version_fn_ = nullptr;

    KernelListSizeFn kernel_list_size_fn_ = nullptr;
    KernelListGetFn kernel_list_get_fn_   = nullptr;
    KernelListFreeFn kernel_list_free_fn_ = nullptr;

    SolutionFreeFn solution_free_fn_ = nullptr;

    using GetAllKernelTypeStringsFn = CKKernelListHandle* (*)();

    struct SolverFns
    {
        FillValidKernelsFn fill_valid_kernels          = nullptr;
        IsApplicableFn is_applicable                   = nullptr;
        IsArgsSupportedFn is_args_supported            = nullptr;
        GetWorkspaceSizeFn get_workspace_size          = nullptr;
        GetSolutionFn get_solution                     = nullptr;
        GetAllKernelTypeStringsFn get_all_kernel_types = nullptr;
    };

    SolverFns solver_fns_[static_cast<int>(CKSolverType::Count)];

    // Helper: extract kernel list from handle
    std::vector<std::string> ExtractKernelList(::CKKernelListHandle* handle) const;

    // Helper: extract ConvSolution from pointer
    ConvSolution ExtractSolution(ConvSolution* ptr) const;
};

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
/// Tokenize a CK kernel name string by extracting the template parameters
/// between '<' and '>', splitting on commas, and stripping whitespace.
inline std::vector<std::string> GetKernelAsTokens(const std::string& kernel)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(
        kernel.substr(kernel.find('<') + 1, kernel.find('>') - kernel.find('<') - 1));
    while(std::getline(tokenStream, token, ','))
    {
        token.erase(remove_if(token.begin(), token.end(), isspace),
                    token.end()); // strip whitespace
        tokens.push_back(token);
    }
    return tokens;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING

/// Check whether a kernel_id with embedded split_k is valid for deterministic
/// execution.  Returns false (invalid) if deterministic is requested and
/// split_k != 1.
inline bool IsDeterministicSplitKValid(const std::string& kernel_id, bool is_deterministic)
{
    if(!is_deterministic)
        return true;

    size_t plus_pos = kernel_id.find_last_of('+');
    if(plus_pos != std::string::npos)
    {
        try
        {
            int split_k_from_id = std::stoi(kernel_id.substr(plus_pos + 1));
            if(split_k_from_id != 1)
            {
                MIOPEN_LOG_I("Invalid configuration for deterministic mode: split_k="
                             << split_k_from_id << " (must be 1)");
                return false;
            }
        }
        catch(const std::exception&)
        {
            MIOPEN_LOG_E("Failed to parse split_k from kernel_id: " << kernel_id);
            return false;
        }
    }
    return true;
}

} // namespace solver
} // namespace miopen
