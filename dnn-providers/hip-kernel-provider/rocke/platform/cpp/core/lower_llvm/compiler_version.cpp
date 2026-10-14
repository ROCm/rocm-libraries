// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native AUTO convenience adapter for runtime.comgr.loaded_compiler_info.
 * Paths locate candidate libraries; only the loaded compiler determines the
 * version. Retain the queried candidate for stable AUTO selection. A caller that owns
 * compilation must pass its compiler's flavor explicitly; this private handle
 * is not shared with an external compilation stage. */
#include "compiler_version.h"

#include <cstdint>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace
{
namespace fs = std::filesystem;

template <typename T>
T symbol(void* library, const char* name)
{
#ifdef _WIN32
    return reinterpret_cast<T>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return reinterpret_cast<T>(dlsym(library, name));
#endif
}

void* load(const std::string& path)
{
#ifdef _WIN32
    fs::path candidate(path);
    if(candidate.has_parent_path())
        return LoadLibraryExW(fs::absolute(candidate).c_str(),
                              nullptr,
                              LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    return LoadLibraryW(candidate.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

/* Query loaded-module provenance without opening or parsing version files. */
std::string library_for_symbol(void* address)
{
    if(!address)
        return {};
#ifdef _WIN32
    HMODULE module = nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                               | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(address),
                           &module))
        return {};
    std::vector<wchar_t> path(32768);
    DWORD size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    return size && size < path.size() ? fs::path(path.data()).string() : std::string();
#else
    Dl_info info{};
    return dladdr(address, &info) && info.dli_fname ? info.dli_fname : "";
#endif
}

/* Match Python's numeric ordering for discovered installation names. */
bool newer(const std::string& a, const std::string& b)
{
    auto numbers = [](const std::string& text) {
        std::vector<unsigned long> result;
        const char* p = text.c_str();
        while(*p)
        {
            if(*p >= '0' && *p <= '9')
            {
                char* end = nullptr;
                result.push_back(std::strtoul(p, &end, 10));
                p = end;
            }
            else
                ++p;
        }
        return result;
    };
    return numbers(a) > numbers(b);
}

std::vector<std::string> children(const fs::path& root, const char* prefix)
{
    std::vector<std::string> paths;
    std::error_code error;
    for(fs::directory_iterator it(root, error), end; !error && it != end; it.increment(error))
    {
        if(it->path().filename().string().starts_with(prefix))
            paths.push_back(it->path().string());
    }
    std::sort(paths.begin(), paths.end(), newer);
    return paths;
}

void* load_comgr(std::string& requested)
{
    std::vector<std::string> candidates;
    if(const char* path = std::getenv("ROCKE_COMGR_LIB"))
        candidates.emplace_back(path);
#ifdef _WIN32
    for(const char* name : {"HIP_PATH", "ROCM_PATH", "ROCM_HOME"})
    {
        if(const char* root = std::getenv(name))
        {
            auto paths = children(fs::path(root) / "bin", "amd_comgr");
            for(const auto& path : paths)
                if(fs::path(path).extension() == ".dll")
                    candidates.push_back(path);
        }
    }
    candidates.emplace_back("amd_comgr.dll");
#else
    auto add_libdir = [&](const fs::path& dir) {
        candidates.push_back((dir / "libamd_comgr.so").string());
        candidates.push_back((dir / "libamd_comgr.so.3").string());
    };
    for(const char* name : {"ROCM_PATH", "ROCM_HOME"})
        if(const char* root = std::getenv(name))
            add_libdir(fs::path(root) / "lib");
    auto roots = children("/opt", "rocm");
    std::vector<std::string> cores;
    for(const auto& root : roots)
    {
        auto paths = children(root, "core-");
        cores.insert(cores.end(), paths.begin(), paths.end());
    }
    std::sort(cores.begin(), cores.end(), newer);
    for(const auto& core : cores)
        add_libdir(fs::path(core) / "lib");
    for(const auto& root : roots)
        add_libdir(fs::path(root) / "lib");
    candidates.emplace_back("libamd_comgr.so");
#endif
    for(const auto& candidate : candidates)
        if(void* library = load(candidate))
        {
            requested = candidate;
            return library;
        }
    return nullptr;
}

/* COMGR's public ABI uses single-word opaque structs. Bind only the actions
 * needed to preprocess Clang's built-in macros; no HIP/LLVM headers or GPU
 * runtime are required by the engine build or the query. */
struct handle
{
    uint64_t value = 0;
};

bool probe(void* library, unsigned* major, unsigned* minor, unsigned* patch)
{
    auto create_set = symbol<int (*)(handle*)>(library, "amd_comgr_create_data_set");
    auto destroy_set = symbol<int (*)(handle)>(library, "amd_comgr_destroy_data_set");
    auto create_data = symbol<int (*)(int, handle*)>(library, "amd_comgr_create_data");
    auto release_data = symbol<int (*)(handle)>(library, "amd_comgr_release_data");
    auto set_data = symbol<int (*)(handle, size_t, const char*)>(library, "amd_comgr_set_data");
    auto set_name = symbol<int (*)(handle, const char*)>(library, "amd_comgr_set_data_name");
    auto add_data = symbol<int (*)(handle, handle)>(library, "amd_comgr_data_set_add");
    auto create_info = symbol<int (*)(handle*)>(library, "amd_comgr_create_action_info");
    auto destroy_info = symbol<int (*)(handle)>(library, "amd_comgr_destroy_action_info");
    auto get_isa = symbol<int (*)(size_t, const char**)>(library, "amd_comgr_get_isa_name");
    auto set_isa
        = symbol<int (*)(handle, const char*)>(library, "amd_comgr_action_info_set_isa_name");
    auto set_language = symbol<int (*)(handle, int)>(library, "amd_comgr_action_info_set_language");
    auto action = symbol<int (*)(int, handle, handle, handle)>(library, "amd_comgr_do_action");
    auto get_output
        = symbol<int (*)(handle, int, size_t, handle*)>(library, "amd_comgr_action_data_get_data");
    auto get_data = symbol<int (*)(handle, size_t*, char*)>(library, "amd_comgr_get_data");
    if(!create_set || !destroy_set || !create_data || !release_data || !set_data || !set_name
       || !add_data || !create_info || !destroy_info || !get_isa || !set_isa || !set_language
       || !action || !get_output || !get_data)
        return false;

    struct resource
    {
        handle h;
        int (*destroy)(handle);
        ~resource()
        {
            if(h.value)
                destroy(h);
        }
    };
    resource inputs{{}, destroy_set}, outputs{{}, destroy_set};
    resource source{{}, release_data}, info{{}, destroy_info}, output{{}, release_data};
    const char* isa = nullptr;
    const char payload[]
        = "ROCKE_LLVM_VERSION __clang_major__ __clang_minor__ __clang_patchlevel__\n";
    // SOURCE=1, OPENCL_1_2=1, SOURCE_TO_PREPROCESSOR=0 in the COMGR ABI.
    if(create_set(&inputs.h) || create_set(&outputs.h) || create_data(1, &source.h)
       || create_info(&info.h) || set_data(source.h, sizeof(payload) - 1, payload)
       || set_name(source.h, "rocke_compiler_version.cl") || add_data(inputs.h, source.h)
       || get_isa(0, &isa) || set_isa(info.h, isa) || set_language(info.h, 1)
       || action(0, info.h, inputs.h, outputs.h) || get_output(outputs.h, 1, 0, &output.h))
        return false;
    size_t size = 0;
    if(get_data(output.h, &size, nullptr))
        return false;
    std::vector<char> text(size + 1, '\0');
    if(get_data(output.h, &size, text.data()))
        return false;
    const char* line = text.data();
    do
    {
        if(std::sscanf(line, "ROCKE_LLVM_VERSION %u %u %u", major, minor, patch) == 3 && *major)
            return true;
        line = std::strchr(line, '\n');
    } while(line && *++line);
    return false;
}
const ckc::CompilerInfo* loaded_info()
{
    static std::mutex mutex;
    // Retained for process lifetime, matching Python's lazy COMGR handle.
    static void* library = nullptr;
    static ckc::CompilerInfo info{};
    static std::string requested, comgr_path, query_path;
    static bool queried = false;
    std::lock_guard<std::mutex> lock(mutex);
    // Retain success; a failed load may recover after library availability changes.
    if(!library)
        library = load_comgr(requested);
    if(!library)
        return nullptr;
    if(!queried)
    {
        info.source = "unavailable";
        if(auto query
           = symbol<void (*)(unsigned*, unsigned*, unsigned*)>(library, "LLVMGetVersion"))
        {
            query(&info.llvm_major, &info.llvm_minor, &info.llvm_patch);
            if(info.llvm_major)
            {
                info.source = "LLVMGetVersion";
                query_path = library_for_symbol(reinterpret_cast<void*>(query));
            }
        }
        if(!info.llvm_major)
        {
            if(probe(library, &info.llvm_major, &info.llvm_minor, &info.llvm_patch))
            {
                info.source = "COMGR preprocessing";
                query_path = library_for_symbol(symbol<void*>(library, "amd_comgr_do_action"));
            }
            else
                info.llvm_major = info.llvm_minor = info.llvm_patch = 0;
        }
        comgr_path = library_for_symbol(symbol<void*>(library, "amd_comgr_get_version"));
        info.requested_comgr = requested.c_str();
        info.comgr_path = comgr_path.empty() ? nullptr : comgr_path.c_str();
        info.query_library_path = query_path.empty() ? nullptr : query_path.c_str();
        queried = true;
    }
    return &info;
}
} // namespace

const ckc::CompilerInfo* ckc::candidate_compiler_info()
{
    try
    {
        return loaded_info();
    }
    catch(const std::exception&)
    {
        // Candidate introspection is best-effort; the caller decides how to handle unavailable.
        return nullptr;
    }
}
