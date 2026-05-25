// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/config.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/solver/ck_impl_status.h>

#include <algorithm>
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
    DepthwiseFwd         = 10,
    Count                = 11
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

class CkImplLibLoader
{
public:
    /// Thread-safe accessor returning a cached per-device singleton.
    MIOPEN_INTERNALS_EXPORT static const CkImplLibLoader& Get(const std::string& device_name);

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

    ~CkImplLibLoader();

    CkImplLibLoader(const CkImplLibLoader&)            = delete;
    CkImplLibLoader& operator=(const CkImplLibLoader&) = delete;

private:
    explicit CkImplLibLoader(const std::string& device_name);

    void OpenRuntimeLibraryForDevice(const std::string& device_name);
    bool LoadSymbols();
    void* ResolveRawSymbol(const char* symbol_name) const;
    void BindRequiredCommonSymbols(std::vector<std::string>& missing);
    void
    BindSolverSymbols(CKSolverType solver, const char* prefix, std::vector<std::string>& missing);
    void BindOptionalKernelTypeSymbols();
    static constexpr int ToSolverIndex(CKSolverType solver) { return static_cast<int>(solver); }

    // Singleton cache
    static std::mutex& CacheMutex();
    static std::unordered_map<std::string, std::unique_ptr<CkImplLibLoader>>& Cache();

    void* lib_handle_ = nullptr;
    bool loaded_      = false;

    // -- Function pointer types -----------------------------------------------
    // CKKernelListHandle is declared at global scope in the interface header.

    using GetApiVersionFn = int (*)();

    using KernelListSizeFn = ck_impl_status_t (*)(const ::CKKernelListHandle*, size_t*);
    using KernelListGetFn = ck_impl_status_t (*)(const ::CKKernelListHandle*, size_t, const char**);
    using KernelListFreeFn = void (*)(::CKKernelListHandle*);

    using SolutionFreeFn = void (*)(ConvSolution*);

    using FillValidKernelsFn = ck_impl_status_t (*)(const miopen::conv::ProblemDescription*,
                                                    miopenDataType_t,
                                                    bool,
                                                    ::CKKernelListHandle**);
    using IsApplicableFn     = ck_impl_status_t (*)(const miopen::conv::ProblemDescription*,
                                                miopenDataType_t,
                                                bool,
                                                bool*);
    using IsArgsSupportedFn  = ck_impl_status_t (*)(
        const miopen::conv::ProblemDescription*, const char*, miopenDataType_t, bool, bool*);
    using GetWorkspaceSizeFn = ck_impl_status_t (*)(const miopen::conv::ProblemDescription*,
                                                    miopenDataType_t,
                                                    bool,
                                                    size_t*);
    using GetSolutionFn      = ck_impl_status_t (*)(const ExecutionContext*,
                                               const miopen::conv::ProblemDescription*,
                                               const char*,
                                               bool,
                                               ConvSolution**);

    // -- Function pointers ----------------------------------------------------
    GetApiVersionFn get_api_version_fn_ = nullptr;

    KernelListSizeFn kernel_list_size_fn_ = nullptr;
    KernelListGetFn kernel_list_get_fn_   = nullptr;
    KernelListFreeFn kernel_list_free_fn_ = nullptr;

    SolutionFreeFn solution_free_fn_ = nullptr;

    using GetLastErrorStringFn                     = void (*)(const char**);
    GetLastErrorStringFn get_last_error_string_fn_ = nullptr;

    using GetAllKernelTypeStringsFn = ck_impl_status_t (*)(::CKKernelListHandle**);

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

    // Check status code and throw on failure, including last-error context.
    void CheckStatus(ck_impl_status_t status, const char* operation) const;

    // Helper: check status and extract kernel list from handle
    std::vector<std::string> ExtractKernelList(ck_impl_status_t status,
                                               ::CKKernelListHandle* handle,
                                               const char* operation) const;

    // Helper: check status and extract ConvSolution from pointer
    ConvSolution
    ExtractSolution(ck_impl_status_t status, ConvSolution* ptr, const char* operation) const;
};

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
/// Tokenize a 2D CK kernel string by extracting parameters between < and >.
/// Unlike the 3D GetKernelAsTokens, this does NOT include the type name prefix
/// as a token — token[0] is the first numeric parameter.
inline std::vector<std::string> GetKernelAsTokens2D(const std::string& kernel)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(
        kernel.substr(kernel.find('<') + 1, kernel.find('>') - kernel.find('<') - 1));
    while(std::getline(tokenStream, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), isspace),
                    token.end()); // strip whitespace
        tokens.push_back(token);
    }
    return tokens;
}
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING

/// Create a CK split_k validator creator for use with RunAIHeuristics.
/// The returned callable takes a ProblemDescription and produces a validator
/// that checks whether a (kernel_id, split_k) pair is supported by CK.
inline auto MakeCKValidatorCreator(const CkImplLibLoader& loader,
                                   CKSolverType solver_type,
                                   miopenDataType_t data_type,
                                   bool use_tf32)
{
    return [&loader, solver_type, data_type, use_tf32](const miopen::conv::ProblemDescription& p) {
        return [&loader, solver_type, data_type, use_tf32, &p](const std::string& kid, int sk) {
            auto combined_id = kid + "+" + std::to_string(sk);
            return loader.IsArgsSupported(solver_type, p, combined_id, data_type, use_tf32);
        };
    };
}

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
