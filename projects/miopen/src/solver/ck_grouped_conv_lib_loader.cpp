// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/solver/ck_grouped_conv_lib_loader.hpp>
#include <miopen/solver/ck_grouped_conv_interface.hpp>
#include <miopen/conv_solution.hpp>
#include <miopen/execution_context.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/env.hpp>
#include <miopen/logger.hpp>

#include <miopen/filesystem.hpp>

#include <cstdlib>

#ifdef _WIN32
// clang-format off
#include <windows.h>
// clang-format on
// <windows.h> defines LoadLibrary as a macro, which collides with our member function name.
#undef LoadLibrary
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

/// Resolve the directory containing the MIOpen shared library.
/// On Linux uses dladdr + realpath; on Windows uses GetModuleHandleExA +
/// GetModuleFileNameA.  Canonicalizes symlinks so per-arch CK libraries
/// are found even when MIOpen is accessed through a symlink.
std::string GetMIOpenLibDir()
{
#ifdef _WIN32
    HMODULE hmod = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(miopenCreate),
                          &hmod) != 0)
    {
        char buf[MAX_PATH];
        DWORD len = GetModuleFileNameA(hmod, buf, MAX_PATH);
        if(len > 0 && len < MAX_PATH)
        {
            std::string path(buf, len);
            auto slash = path.find_last_of("\\/");
            if(slash != std::string::npos)
                return path.substr(0, slash);
        }
    }
#else
    Dl_info info;
    if(dladdr(reinterpret_cast<void*>(miopenCreate), &info) != 0)
    {
        std::unique_ptr<char, decltype(&free)> real(realpath(info.dli_fname, nullptr), free);
        std::string path(real != nullptr ? real.get() : info.dli_fname);
        auto slash = path.rfind('/');
        if(slash != std::string::npos)
            return path.substr(0, slash);
    }
#endif
    return {};
}

} // namespace

// -- Singleton infrastructure -------------------------------------------------

std::mutex& CKGroupedConvLibLoader::CacheMutex()
{
    static std::mutex mtx;
    return mtx;
}

std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>>&
CKGroupedConvLibLoader::Cache()
{
    static std::unordered_map<std::string, std::unique_ptr<CKGroupedConvLibLoader>> cache;
    return cache;
}

const CKGroupedConvLibLoader& CKGroupedConvLibLoader::Get(const std::string& device_name)
{
    const auto key = StripDeviceSuffix(device_name);
    std::lock_guard<std::mutex> lock(CacheMutex());
    auto& cache = Cache();
    auto it     = cache.find(key);
    if(it == cache.end())
    {
        // Use new + reset instead of make_unique because the constructor is private.
        std::unique_ptr<CKGroupedConvLibLoader> ptr(new CKGroupedConvLibLoader(device_name));
        it = cache.emplace(key, std::move(ptr)).first;
    }
    return *it->second;
}

// -- Construction / Destruction -----------------------------------------------

CKGroupedConvLibLoader::CKGroupedConvLibLoader(const std::string& device_name)
{
    LoadLibrary(device_name);
}

CKGroupedConvLibLoader::~CKGroupedConvLibLoader()
{
    if(lib_handle_ != nullptr)
    {
#ifdef _WIN32
        FreeLibrary(static_cast<HMODULE>(lib_handle_));
#else
        // RTLD_NODELETE keeps the library mapped, so dlclose only decrements the
        // reference count without unmapping.  We still call it for correctness.
        dlclose(lib_handle_);
#endif
    }
}

// -- Library loading ----------------------------------------------------------

void CKGroupedConvLibLoader::LoadLibrary(const std::string& device_name)
{
    const auto filename = MakeLibraryFilename(device_name);

    // Platform-specific load and error helpers
#ifdef _WIN32
    auto try_load = [](const std::string& path) -> void* {
        return static_cast<void*>(LoadLibraryA(path.c_str()));
    };
    auto get_load_error = []() -> std::string {
        DWORD err = GetLastError();
        char* msg = nullptr;
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
    };
#else
    constexpr int flags = RTLD_NOW | RTLD_NODELETE;
    auto try_load = [](const std::string& path) -> void* { return dlopen(path.c_str(), flags); };
    auto get_load_error = []() -> std::string {
        // NOLINTNEXTLINE(concurrency-mt-unsafe)
        const char* err = dlerror();
        return err ? err : "unknown error";
    };
#endif

    // 1. Try MIOPEN_CK_LIB_PATH environment variable
    const auto env_path = env::value(MIOPEN_CK_LIB_PATH);
    if(!env_path.empty())
    {
        auto full_path = env_path + "/" + filename;
        lib_handle_    = try_load(full_path);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from env path: " << full_path);
        }
    }

    // 2. Try the directory containing the MIOpen shared library
    if(lib_handle_ == nullptr)
    {
        auto lib_dir = GetMIOpenLibDir();
        if(!lib_dir.empty())
        {
            auto full_path = lib_dir + "/" + filename;
            lib_handle_    = try_load(full_path);
            if(lib_handle_ != nullptr)
            {
                MIOPEN_LOG_I2("Loaded CK grouped conv library from lib dir: " << full_path);
            }
        }
    }

    // 3. Fall back to default search path
    if(lib_handle_ == nullptr)
    {
        lib_handle_ = try_load(filename);
        if(lib_handle_ != nullptr)
        {
            MIOPEN_LOG_I2("Loaded CK grouped conv library from default path: " << filename);
        }
    }

    if(lib_handle_ == nullptr)
    {
        MIOPEN_LOG_W("CK grouped conv library not found for device "
                     << StripDeviceSuffix(device_name) << ": " << get_load_error());
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
    if(lib_version != CK_GROUPED_CONV_API_VERSION)
    {
        MIOPEN_LOG_W("CK grouped conv API version mismatch for device "
                     << StripDeviceSuffix(device_name) << ": expected "
                     << CK_GROUPED_CONV_API_VERSION << ", got " << lib_version);
        loaded_ = false;
        return;
    }

    loaded_ = true;
}

// -- Symbol resolution --------------------------------------------------------

bool CKGroupedConvLibLoader::LoadSymbols()
{
// Helper macro: resolve a symbol or return false on failure.
#ifdef _WIN32
#define LOAD_SYM(member, name)                                                             \
    do                                                                                     \
    {                                                                                      \
        member = reinterpret_cast<decltype(member)>(                                       \
            GetProcAddress(static_cast<HMODULE>(lib_handle_), #name));                     \
        if(member == nullptr)                                                              \
        {                                                                                  \
            MIOPEN_LOG_W("GetProcAddress failed for " #name ": error " << GetLastError()); \
            return false;                                                                  \
        }                                                                                  \
    } while(false)
#else
#define LOAD_SYM(member, name)                                                  \
    do                                                                          \
    {                                                                           \
        member = reinterpret_cast<decltype(member)>(dlsym(lib_handle_, #name)); \
        if(member == nullptr)                                                   \
        {                                                                       \
            MIOPEN_LOG_W("dlsym failed for " #name ": "                         \
                         << dlerror()); /* NOLINT(concurrency-mt-unsafe) */     \
            return false;                                                       \
        }                                                                       \
    } while(false)
#endif

    // Common
    LOAD_SYM(get_api_version_fn_, ckgrpconv_get_api_version);
    LOAD_SYM(kernel_list_size_fn_, ckgrpconv_kernel_list_size);
    LOAD_SYM(kernel_list_get_fn_, ckgrpconv_kernel_list_get);
    LOAD_SYM(kernel_list_free_fn_, ckgrpconv_kernel_list_free);
    LOAD_SYM(solution_free_fn_, ckgrpconv_solution_free);

    // Per-direction symbols
#define LOAD_DIR_SYMS(idx, prefix)                                                       \
    LOAD_SYM(dir_fns_[idx].fill_valid_kernels, ckgrpconv_##prefix##_fill_valid_kernels); \
    LOAD_SYM(dir_fns_[idx].is_applicable, ckgrpconv_##prefix##_is_applicable);           \
    LOAD_SYM(dir_fns_[idx].is_args_supported, ckgrpconv_##prefix##_is_args_supported);   \
    LOAD_SYM(dir_fns_[idx].get_workspace_size, ckgrpconv_##prefix##_get_workspace_size); \
    LOAD_SYM(dir_fns_[idx].get_solution, ckgrpconv_##prefix##_get_solution)

    LOAD_DIR_SYMS(0, fwd);
    LOAD_DIR_SYMS(1, bwd);
    LOAD_DIR_SYMS(2, wrw);
#undef LOAD_DIR_SYMS

#undef LOAD_SYM
    return true;
}

// -- Helpers ------------------------------------------------------------------

std::vector<std::string> CKGroupedConvLibLoader::ExtractKernelList(CKKernelListHandle* handle) const
{
    if(handle == nullptr)
        return {};
    std::vector<std::string> result;
    const size_t n = kernel_list_size_fn_(handle);
    result.reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        const char* s = kernel_list_get_fn_(handle, i);
        if(s != nullptr)
            result.emplace_back(s);
    }
    kernel_list_free_fn_(handle);
    return result;
}

ConvSolution CKGroupedConvLibLoader::ExtractSolution(ConvSolution* ptr) const
{
    if(ptr == nullptr)
        return ConvSolution{miopenStatusInternalError};
    ConvSolution result = std::move(*ptr);
    solution_free_fn_(ptr);
    return result;
}

// -- Direction-parameterized wrappers -----------------------------------------

std::vector<std::string>
CKGroupedConvLibLoader::FillValidKernels(CKConvDirection dir,
                                         const conv::ProblemDescription& problem,
                                         miopenDataType_t dtype,
                                         bool use_tf32) const
{
    if(!IsLoaded())
        return {};
    return ExtractKernelList(
        dir_fns_[static_cast<int>(dir)].fill_valid_kernels(&problem, dtype, use_tf32));
}

std::vector<std::string>
CKGroupedConvLibLoader::FillValidKernelsWithTf32Fallback(CKConvDirection dir,
                                                         const conv::ProblemDescription& problem,
                                                         miopenDataType_t dtype,
                                                         bool& use_tf32) const
{
    auto result = FillValidKernels(dir, problem, dtype, use_tf32);
    if(result.empty() && use_tf32)
    {
        use_tf32 = false;
        result   = FillValidKernels(dir, problem, dtype, false);
    }
    return result;
}

bool CKGroupedConvLibLoader::IsApplicable(CKConvDirection dir,
                                          const conv::ProblemDescription& problem,
                                          miopenDataType_t dtype,
                                          bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return dir_fns_[static_cast<int>(dir)].is_applicable(&problem, dtype, use_tf32);
}

bool CKGroupedConvLibLoader::IsArgsSupported(CKConvDirection dir,
                                             const conv::ProblemDescription& problem,
                                             const std::string& kernel_id,
                                             miopenDataType_t dtype,
                                             bool use_tf32) const
{
    if(!IsLoaded())
        return false;
    return dir_fns_[static_cast<int>(dir)].is_args_supported(
        &problem, kernel_id.c_str(), dtype, use_tf32);
}

size_t CKGroupedConvLibLoader::GetWorkspaceSize(CKConvDirection dir,
                                                const conv::ProblemDescription& problem,
                                                miopenDataType_t dtype,
                                                bool use_tf32) const
{
    if(!IsLoaded())
        return 0;
    return dir_fns_[static_cast<int>(dir)].get_workspace_size(&problem, dtype, use_tf32);
}

ConvSolution CKGroupedConvLibLoader::GetSolution(CKConvDirection dir,
                                                 const ExecutionContext& ctx,
                                                 const conv::ProblemDescription& problem,
                                                 const std::string& kernel_id,
                                                 bool use_tf32) const
{
    if(!IsLoaded())
        return ConvSolution{miopenStatusInternalError};
    return ExtractSolution(
        dir_fns_[static_cast<int>(dir)].get_solution(&ctx, &problem, kernel_id.c_str(), use_tf32));
}

} // namespace solver
} // namespace miopen
