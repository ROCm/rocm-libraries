// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/solver/ck_impl_lib_loader.hpp>
#include <miopen/solver/ck_impl_error.hpp>
#include <miopen/solver/ck_impl_interface.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>

#include <miopen/filesystem.hpp>

#include <cstdlib>
#include <type_traits>

#ifdef _WIN32
// clang-format off
#include <windows.h>
// clang-format on
#else
#include <dlfcn.h>
#endif

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_CK_LIB_PATH)

namespace miopen {
namespace solver {

namespace {

/// Strip architecture-specific suffixes like ":sramecc+:xnack-" from a device
/// name, returning only the base GPU identifier (e.g. "gfx90a").
std::string StripDeviceSuffix(const std::string& device_name)
{
    auto pos = device_name.find(':');
    if(pos != std::string::npos)
        return device_name.substr(0, pos);
    return device_name;
}

/// Build the expected shared library filename for a given device.
std::string MakeLibraryFilename(const std::string& device_name)
{
    return std::string(miopen::library_prefix) + "MIOpenCKGroupedConv_" +
           StripDeviceSuffix(device_name) + std::string(miopen::dynamic_library_postfix);
}

/// Resolve the directory containing the loaded MIOpen shared library.
/// Canonicalizes symlinks so per-arch CK libraries are found even when
/// MIOpen is accessed through a symlink.
miopen::fs::path GetOwningModuleDirectory()
{
#ifdef _WIN32
    HMODULE hmod = nullptr;
    if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(miopenCreate),
                          &hmod) != 0)
    {
        std::wstring module_path(MAX_PATH, L'\0');
        while(true)
        {
            const auto len = GetModuleFileNameW(
                hmod, module_path.data(), static_cast<DWORD>(module_path.size()));
            if(len == 0)
                break;
            if(len < module_path.size())
            {
                module_path.resize(len);
                return miopen::weakly_canonical(miopen::fs::path{module_path}).parent_path();
            }
            module_path.resize(module_path.size() * 2);
        }
    }
#else
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(miopenCreate), &info) != 0 && info.dli_fname != nullptr &&
       info.dli_fname[0] != '\0')
    {
        return miopen::weakly_canonical(miopen::fs::path{info.dli_fname}).parent_path();
    }
#endif
    return {};
}

void* OpenDynamicLibrary(const miopen::fs::path& library_path)
{
#ifdef _WIN32
    const auto native_path = library_path.wstring();
    return static_cast<void*>(LoadLibraryW(native_path.c_str()));
#else
    // Keep symbols local to avoid accidental global symbol collisions and
    // preserve RTLD_NODELETE to match the existing unload behavior.
    constexpr int flags    = RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE;
    const auto native_path = library_path.string();
    return dlopen(native_path.c_str(), flags);
#endif
}

void CloseDynamicLibrary(void* handle)
{
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    // RTLD_NODELETE keeps the library mapped, so dlclose only decrements the
    // reference count without unmapping. We still call it for correctness.
    dlclose(handle);
#endif
}

std::string GetDynamicLoadError()
{
#ifdef _WIN32
    const DWORD err = GetLastError();
    char* msg       = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr,
                   err,
                   0,
                   reinterpret_cast<LPSTR>(&msg),
                   0,
                   nullptr);
    std::string result = msg ? msg : "unknown error";
    LocalFree(msg);
    return result;
#else
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const char* err = dlerror();
    return err != nullptr ? err : "unknown error";
#endif
}

void* ResolveDynamicSymbol(void* handle, const char* symbol_name)
{
    if(handle == nullptr)
        return nullptr;
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), symbol_name));
#else
    dlerror();                         // NOLINT(concurrency-mt-unsafe)
    return dlsym(handle, symbol_name); // NOLINT(concurrency-mt-unsafe)
#endif
}

miopenStatus_t toMiopenStatus(ck_impl_status_t status)
{
    switch(status)
    {
    case CK_IMPL_STATUS_SUCCESS: return miopenStatusSuccess;
    case CK_IMPL_STATUS_BAD_PARAM: return miopenStatusBadParm;
    case CK_IMPL_STATUS_INVALID_VALUE: return miopenStatusInvalidValue;
    case CK_IMPL_STATUS_INTERNAL_ERROR: return miopenStatusInternalError;
    case CK_IMPL_STATUS_ALLOC_FAILED: return miopenStatusAllocFailed;
    }

    return miopenStatusInternalError;
}

} // namespace

// -- Singleton infrastructure -------------------------------------------------

std::mutex& CkImplLibLoader::CacheMutex()
{
    static std::mutex mtx;
    return mtx;
}

std::unordered_map<std::string, std::unique_ptr<CkImplLibLoader>>& CkImplLibLoader::Cache()
{
    static std::unordered_map<std::string, std::unique_ptr<CkImplLibLoader>> cache;
    return cache;
}

const CkImplLibLoader& CkImplLibLoader::Get(const std::string& device_name)
{
    const auto key = StripDeviceSuffix(device_name);
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& cache = Cache();
    auto it     = cache.find(key);
    if(it == cache.end())
    {
        // Use new + reset instead of make_unique because the constructor is private.
        std::unique_ptr<CkImplLibLoader> ptr(new CkImplLibLoader(device_name));
        it = cache.emplace(key, std::move(ptr)).first;
    }
    return *it->second;
}

// -- Construction / Destruction -----------------------------------------------

CkImplLibLoader::CkImplLibLoader(const std::string& device_name)
{
    OpenRuntimeLibraryForDevice(device_name);
}

CkImplLibLoader::~CkImplLibLoader()
{
    if(lib_handle_ != nullptr)
        CloseDynamicLibrary(lib_handle_);
}

// -- Library loading ----------------------------------------------------------

void CkImplLibLoader::OpenRuntimeLibraryForDevice(const std::string& device_name)
{
    const miopen::fs::path filename{MakeLibraryFilename(device_name)};

    // 1. Try MIOPEN_CK_LIB_PATH environment variable
    const auto env_path = env::value(MIOPEN_CK_LIB_PATH);
    if(!env_path.empty())
    {
        const auto full_path = miopen::fs::path{env_path} / filename;
        lib_handle_          = OpenDynamicLibrary(full_path);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from env path: " << full_path.string());
        }
    }

    // 2. Try the directory containing the MIOpen shared library
    if(lib_handle_ == nullptr)
    {
        const auto lib_dir = GetOwningModuleDirectory();
        if(!lib_dir.empty())
        {
            const auto full_path = lib_dir / filename;
            lib_handle_          = OpenDynamicLibrary(full_path);
            if(lib_handle_ != nullptr)
            {
                MIOPEN_LOG_I2(
                    "Loaded CK grouped conv library from lib dir: " << full_path.string());
            }
        }
    }

    // 3. Fall back to default search path
    if(lib_handle_ == nullptr)
    {
        lib_handle_ = OpenDynamicLibrary(filename);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2(
                "Loaded CK grouped conv library from default path: " << filename.string());
        }
    }

    if(lib_handle_ == nullptr)
    {
        MIOPEN_LOG_W("CK grouped conv library not found for device "
                     << StripDeviceSuffix(device_name) << ": " << GetDynamicLoadError());
        return;
    }

    if(!LoadSymbols())
    {
        MIOPEN_LOG_W("Failed to resolve symbols in CK grouped conv library for device "
                     << StripDeviceSuffix(device_name));
        loaded_ = false;
        return;
    }

    // API version check
    const int lib_version = get_api_version_fn_();
    if(lib_version != CK_IMPL_API_VERSION)
    {
        MIOPEN_LOG_W("CK grouped conv API version mismatch for device "
                     << StripDeviceSuffix(device_name) << ": expected " << CK_IMPL_API_VERSION
                     << ", got " << lib_version);
        loaded_ = false;
        return;
    }

    loaded_ = true;
}

// -- Symbol resolution --------------------------------------------------------

void* CkImplLibLoader::ResolveRawSymbol(const char* symbol_name) const
{
    return ResolveDynamicSymbol(lib_handle_, symbol_name);
}

void CkImplLibLoader::BindRequiredCommonSymbols(std::vector<std::string>& missing)
{
    auto bind_symbol = [this, &missing](auto& member, const char* symbol_name) {
        using FnPtr = std::remove_reference_t<decltype(member)>;
        member      = reinterpret_cast<FnPtr>(ResolveRawSymbol(symbol_name));
        if(member == nullptr)
            missing.emplace_back(symbol_name);
    };

    bind_symbol(get_api_version_fn_, "ck_impl_get_api_version");
    bind_symbol(kernel_list_size_fn_, "ck_impl_kernel_list_size");
    bind_symbol(kernel_list_get_fn_, "ck_impl_kernel_list_get");
    bind_symbol(kernel_list_free_fn_, "ck_impl_kernel_list_free");
    bind_symbol(solution_free_fn_, "ck_impl_solution_free");
    bind_symbol(get_last_error_string_fn_, "ck_impl_get_last_error_string");
}

void CkImplLibLoader::BindSolverSymbols(CKSolverType solver,
                                        const char* prefix,
                                        std::vector<std::string>& missing)
{
    auto bind_symbol = [this, &missing](auto& member, const char* symbol_name) {
        using FnPtr = std::remove_reference_t<decltype(member)>;
        member      = reinterpret_cast<FnPtr>(ResolveRawSymbol(symbol_name));
        if(member == nullptr)
            missing.emplace_back(symbol_name);
    };

    auto& fns             = solver_fns_[ToSolverIndex(solver)];
    const std::string sym = std::string("ck_impl_") + prefix + "_";
    bind_symbol(fns.fill_valid_kernels, (sym + "fill_valid_kernels").c_str());
    bind_symbol(fns.is_applicable, (sym + "is_applicable").c_str());
    bind_symbol(fns.is_args_supported, (sym + "is_args_supported").c_str());
    bind_symbol(fns.get_workspace_size, (sym + "get_workspace_size").c_str());
    bind_symbol(fns.get_solution, (sym + "get_solution").c_str());
}

void CkImplLibLoader::BindOptionalKernelTypeSymbols()
{
    auto bind_optional = [this](auto& member, const char* symbol_name) {
        using FnPtr = std::remove_reference_t<decltype(member)>;
        member      = reinterpret_cast<FnPtr>(ResolveRawSymbol(symbol_name));
        if(member == nullptr)
            MIOPEN_LOG_I2("Optional CK symbol not found: " << symbol_name);
    };

    bind_optional(solver_fns_[ToSolverIndex(CKSolverType::GrpConvFwd)].get_all_kernel_types,
                  "ck_impl_fwd_get_all_kernel_type_strings");
    bind_optional(solver_fns_[ToSolverIndex(CKSolverType::GrpConvBwd)].get_all_kernel_types,
                  "ck_impl_bwd_get_all_kernel_type_strings");
    bind_optional(solver_fns_[ToSolverIndex(CKSolverType::GrpConvWrw)].get_all_kernel_types,
                  "ck_impl_wrw_get_all_kernel_type_strings");

    bind_optional(solver_fns_[ToSolverIndex(CKSolverType::GrpConv3dFwd)].get_all_kernel_types,
                  "ck_impl_3d_fwd_get_all_kernel_type_strings");
    bind_optional(solver_fns_[ToSolverIndex(CKSolverType::GrpConv3dBwd)].get_all_kernel_types,
                  "ck_impl_3d_bwd_get_all_kernel_type_strings");
    bind_optional(solver_fns_[ToSolverIndex(CKSolverType::GrpConv3dWrw)].get_all_kernel_types,
                  "ck_impl_3d_wrw_get_all_kernel_type_strings");
}

bool CkImplLibLoader::LoadSymbols()
{
    std::vector<std::string> missing;
    missing.reserve(32);

    BindRequiredCommonSymbols(missing);

    struct SolverSymbolBinding
    {
        CKSolverType solver;
        const char* prefix;
    };

    static constexpr SolverSymbolBinding solver_bindings[] = {
        {CKSolverType::GrpConvFwd, "fwd"},
        {CKSolverType::GrpConvBwd, "bwd"},
        {CKSolverType::GrpConvWrw, "wrw"},
        {CKSolverType::GrpConv3dFwd, "3d_fwd"},
        {CKSolverType::GrpConv3dBwd, "3d_bwd"},
        {CKSolverType::GrpConv3dWrw, "3d_wrw"},
        {CKSolverType::FusedBiasActiv, "fused_bias_activ"},
        {CKSolverType::FusedBiasResAddActiv, "fused_bias_res_add_activ"},
        {CKSolverType::FusedGrpActiv, "fused_grp_activ"},
        {CKSolverType::FusedGrpBiasActiv, "fused_grp_bias_activ"},
        {CKSolverType::DepthwiseFwd, "depthwise_fwd"}};

    for(const auto& binding : solver_bindings)
        BindSolverSymbols(binding.solver, binding.prefix, missing);

    BindOptionalKernelTypeSymbols();

    if(missing.empty())
        return true;

    std::ostringstream msg;
    for(std::size_t i = 0; i < missing.size(); ++i)
    {
        if(i != 0)
            msg << ", ";
        msg << missing[i];
    }

    MIOPEN_LOG_W("Failed to resolve CK grouped conv symbols (" << missing.size()
                                                               << " missing): " << msg.str());
    return false;
}

// -- Helpers ------------------------------------------------------------------

void CkImplLibLoader::CheckStatus(ck_impl_status_t status, const char* operation) const
{
    if(status == CK_IMPL_STATUS_SUCCESS)
        return;
    const char* error_msg = "";
    if(get_last_error_string_fn_ != nullptr)
        get_last_error_string_fn_(&error_msg);
    MIOPEN_THROW(toMiopenStatus(status),
                 std::string("CK grouped conv ") + operation +
                     " failed. Status: " + toString(status) + "(" + std::to_string(status) + ")" +
                     ", Error: " + (error_msg != nullptr ? error_msg : ""));
}

std::vector<std::string> CkImplLibLoader::ExtractKernelList(ck_impl_status_t status,
                                                            CKKernelListHandle* handle,
                                                            const char* operation) const
{
    // RAII guard: ensure handle is freed even if CheckStatus throws
    auto handle_guard =
        std::unique_ptr<CKKernelListHandle, KernelListFreeFn>(handle, kernel_list_free_fn_);

    CheckStatus(status, operation);
    if(handle_guard == nullptr)
        return {};
    std::vector<std::string> result;
    size_t n = 0;
    CheckStatus(kernel_list_size_fn_(handle_guard.get(), &n), "kernel_list_size");
    result.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        const char* s = nullptr;
        CheckStatus(kernel_list_get_fn_(handle_guard.get(), i, &s), "kernel_list_get");
        if(s != nullptr)
            result.emplace_back(s);
    }
    return result;
}

ConvSolution CkImplLibLoader::ExtractSolution(ck_impl_status_t status,
                                              ConvSolution* ptr,
                                              const char* operation) const
{
    // RAII guard: ensure ptr is freed even if CheckStatus throws
    auto ptr_guard = std::unique_ptr<ConvSolution, SolutionFreeFn>(ptr, solution_free_fn_);

    CheckStatus(status, operation);
    if(ptr_guard == nullptr)
        return ConvSolution{miopenStatusInternalError};
    return std::move(*ptr_guard);
}

// -- Solver-parameterized wrappers --------------------------------------------

std::vector<std::string> CkImplLibLoader::FillValidKernels(CKSolverType solver,
                                                           const conv::ProblemDescription& problem,
                                                           miopenDataType_t dtype,
                                                           bool use_tf32) const
{
    if(!IsLoaded())
        return {};
    CKKernelListHandle* handle = nullptr;
    auto status =
        solver_fns_[ToSolverIndex(solver)].fill_valid_kernels(&problem, dtype, use_tf32, &handle);
    return ExtractKernelList(status, handle, "fill_valid_kernels");
}

std::vector<std::string>
CkImplLibLoader::FillValidKernelsWithTf32Fallback(CKSolverType solver,
                                                  const conv::ProblemDescription& problem,
                                                  miopenDataType_t dtype,
                                                  bool& use_tf32) const
{
    auto result = FillValidKernels(solver, problem, dtype, use_tf32);
    if(result.empty() && use_tf32)
    {
        use_tf32 = false;
        result   = FillValidKernels(solver, problem, dtype, false);
    }
    return result;
}

bool CkImplLibLoader::IsApplicable(CKSolverType solver,
                                   const conv::ProblemDescription& problem,
                                   miopenDataType_t dtype,
                                   bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    bool result = false;
    auto status =
        solver_fns_[ToSolverIndex(solver)].is_applicable(&problem, dtype, use_tf32, &result);
    CheckStatus(status, "is_applicable");
    return result;
}

bool CkImplLibLoader::IsArgsSupported(CKSolverType solver,
                                      const conv::ProblemDescription& problem,
                                      const std::string& kernel_id,
                                      miopenDataType_t dtype,
                                      bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    bool result = false;
    auto status = solver_fns_[ToSolverIndex(solver)].is_args_supported(
        &problem, kernel_id.c_str(), dtype, use_tf32, &result);
    CheckStatus(status, "is_args_supported");
    return result;
}

size_t CkImplLibLoader::GetWorkspaceSize(CKSolverType solver,
                                         const conv::ProblemDescription& problem,
                                         miopenDataType_t dtype,
                                         bool use_tf32) const
{
    if(!IsLoaded())
        return 0;
    size_t result = 0;
    auto status =
        solver_fns_[ToSolverIndex(solver)].get_workspace_size(&problem, dtype, use_tf32, &result);
    CheckStatus(status, "get_workspace_size");
    return result;
}

ConvSolution CkImplLibLoader::GetSolution(CKSolverType solver,
                                          const ExecutionContext& ctx,
                                          const conv::ProblemDescription& problem,
                                          const std::string& kernel_id,
                                          bool use_tf32) const
{
    if(!IsLoaded())
        return ConvSolution{miopenStatusInternalError};
    ConvSolution* ptr = nullptr;
    auto status       = solver_fns_[ToSolverIndex(solver)].get_solution(
        &ctx, &problem, kernel_id.c_str(), use_tf32, &ptr);
    return ExtractSolution(status, ptr, "get_solution");
}

std::vector<std::string> CkImplLibLoader::GetAllKernelTypeStrings(CKSolverType solver) const
{
    if(!IsLoaded())
        return {};
    auto fn = solver_fns_[ToSolverIndex(solver)].get_all_kernel_types;
    if(fn == nullptr)
        return {};
    CKKernelListHandle* handle = nullptr;
    auto status                = fn(&handle);
    return ExtractKernelList(status, handle, "get_all_kernel_type_strings");
}

} // namespace solver
} // namespace miopen
