// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native AUTO convenience adapter for runtime.comgr.loaded_compiler_info.
 * Paths locate candidate libraries; only the loaded compiler determines the
 * version. Retain the queried candidate for stable AUTO selection. A caller
 * that owns compilation must pass its compiler's flavor explicitly; this
 * private handle is not shared with an external compilation stage. */
#include "compiler_version.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>

#include <tlhelp32.h>
#else
#include <dlfcn.h>
#ifdef __linux__
#include <link.h>
#endif
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

bool comgr_name(const fs::path& path, bool mapped = false)
{
    auto name = path.filename().string();
#ifdef _WIN32
    static const std::regex pattern("amd_comgr[0-9]*\\.dll");
    if(mapped)
        std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
#else
    static const std::regex pattern("libamd_comgr\\.so(\\.[0-9]+)*");
    (void)mapped;
#endif
    return std::regex_match(name, pattern);
}

std::vector<fs::path> children(const fs::path& root)
{
    std::vector<fs::path> paths;
    std::error_code error;
    for(fs::directory_iterator it(root, error), end; !error && it != end; it.increment(error))
        paths.push_back(it->path());
    std::sort(paths.begin(), paths.end());
    return paths;
}

/* Wheel RECORD paths use CSV quoting, including commas in payload names. */
std::string record_path(const std::string& line)
{
    if(line.empty() || line[0] != '"')
        return line.substr(0, line.find(','));
    std::string result;
    for(size_t i = 1; i < line.size(); ++i)
    {
        if(line[i] != '"')
            result += line[i];
        else if(i + 1 < line.size() && line[i + 1] == '"')
        {
            result += '"';
            ++i;
        }
        else
            return result;
    }
    return {};
}

/* Mirrors runtime._sdk_comgr: distribution ownership, not package imports or
 * guessed payload globs. Native installations place the SDK wheel in the same
 * prefix as the engine/executable. Resolve at runtime so moving the whole
 * prefix works, without retaining a build-machine SDK path or requiring Python.
 */
std::optional<std::string> sdk_comgr_in(const fs::path& site)
{
    for(const auto& dist : children(site))
    {
        if(dist.extension() != ".dist-info")
            continue;
        std::ifstream metadata(dist / "METADATA");
        std::string line;
        bool owned = false;
        while(std::getline(metadata, line))
        {
            if(!line.starts_with("Name: "))
                continue;
            auto name = line.substr(6);
            if(!name.empty() && name.back() == '\r')
                name.pop_back();
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
                return c == '_' || c == '.' ? '-' : static_cast<char>(std::tolower(c));
            });
            owned = name == "rocm-sdk-core";
            break;
        }
        if(!owned)
            continue;
        std::ifstream record(dist / "RECORD");
        std::vector<std::string> matches;
        while(std::getline(record, line))
        {
            fs::path relative(record_path(line));
            if(relative.is_absolute())
                continue;
            std::vector<std::string> parts;
            for(const auto& part : relative)
                parts.push_back(part.string());
            if(parts.size() != 3
               || !(parts[0] == "_rocm_sdk_core"
                    || (parts[0].starts_with("_rocm_sdk_core_") && parts[0].size() > 15)))
                continue;
#ifdef _WIN32
            const char* library_dir = "bin";
#else
            const char* library_dir = "lib";
#endif
            std::error_code error;
            if(parts[1] == library_dir && comgr_name(relative)
               && fs::is_regular_file(site / relative, error))
                matches.push_back(fs::absolute(site / relative).lexically_normal().string());
        }
        if(!matches.empty())
        {
            std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
                auto an = fs::path(a).filename().string(), bn = fs::path(b).filename().string();
                return an.size() != bn.size() ? an.size() < bn.size() : a < b;
            });
            return matches[0];
        }
        // Like importlib.metadata.distribution(), use the selected distribution.
        return std::string();
    }
    return std::nullopt;
}

std::string executable_path()
{
#ifdef _WIN32
    std::vector<wchar_t> buffer(32768);
    auto size = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
    return size && size < buffer.size() ? fs::path(buffer.data()).string() : std::string();
#else
    std::error_code error;
    auto path = fs::read_symlink("/proc/self/exe", error);
    return error ? std::string() : path.string();
#endif
}

std::string discover_sdk_comgr()
{
    std::vector<fs::path> visited;
    for(const auto& anchor :
        {library_for_symbol(reinterpret_cast<void*>(&discover_sdk_comgr)), executable_path()})
    {
        if(anchor.empty())
            continue;
        for(auto root = fs::absolute(anchor).parent_path(); !root.empty();
            root = root.parent_path())
        {
            if(std::find(visited.begin(), visited.end(), root) == visited.end())
            {
                visited.push_back(root);
                if(auto selected = sdk_comgr_in(root); selected)
                    return *selected;
                if(auto selected = sdk_comgr_in(root / "Lib" / "site-packages"); selected)
                    return *selected;
                for(const char* lib : {"lib", "lib64"})
                    for(const auto& python : children(root / lib))
                        if(python.filename().string().starts_with("python"))
                            if(auto selected = sdk_comgr_in(python / "site-packages"); selected)
                                return *selected;
                // Configure-time selected Python site, for build-tree consumers
                // only. The site sidecar is deliberately not installed.
                std::ifstream descriptor(root / "rocke_sdk_site.txt");
                std::string marker, location;
                if(std::getline(descriptor, marker))
                {
                    const bool absolute = marker == "rocke-sdk-site-absolute-v1";
                    if((!absolute && marker != "rocke-sdk-site-v1")
                       || !std::getline(descriptor, location) || location.empty())
                        return {};
                    const fs::path site(location);
                    if(site.is_absolute() != absolute || (!absolute && site.has_root_path()))
                        return {};
                    auto selected = sdk_comgr_in(absolute ? site : root / site);
                    return selected ? *selected : std::string();
                }
            }
            if(root == root.parent_path())
                break;
        }
    }
    return {};
}

/* Check mapped files before dlopen/LoadLibrary. Distinct COMGR copies can bring
 * distinct LLVM command-line registries; RTLD_LOCAL does not isolate them. */
bool compatible_with_mapped(const fs::path& candidate)
{
    std::vector<fs::path> mapped;
#ifdef _WIN32
    HANDLE snapshot
        = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
    if(snapshot == INVALID_HANDLE_VALUE)
        return false;
    {
        MODULEENTRY32W entry{};
        entry.dwSize = sizeof(entry);
        if(!Module32FirstW(snapshot, &entry))
        {
            CloseHandle(snapshot);
            return false;
        }
        do
        {
            if(comgr_name(fs::path(entry.szExePath), true))
                mapped.emplace_back(entry.szExePath);
        } while(Module32NextW(snapshot, &entry));
        const auto status = GetLastError();
        CloseHandle(snapshot);
        if(status != ERROR_NO_MORE_FILES)
            return false;
    }
#else
#ifdef __linux__
    struct mapped_module
    {
        uintptr_t address;
        bool found = false;
    };
    std::vector<mapped_module> modules;
    // Join retained loader names to actual mapped files. NFS renames unlinked
    // libraries to .nfs*, and the original name may already name a new inode.
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* state) {
            fs::path path(info->dlpi_name ? info->dlpi_name : "");
            if(comgr_name(path))
                static_cast<std::vector<mapped_module>*>(state)->push_back(
                    {static_cast<uintptr_t>(info->dlpi_addr)});
            return 0;
        },
        &modules);
#endif
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while(std::getline(maps, line))
    {
        std::istringstream fields(line);
        std::string range, field, path;
        fields >> range;
        for(int i = 0; i < 4; ++i)
            fields >> field;
        std::getline(fields >> std::ws, path);
        auto name = path.ends_with(" (deleted)") ? path.substr(0, path.size() - 10) : path;
        if(comgr_name(fs::path(name)))
            mapped.emplace_back(path);
#ifdef __linux__
        auto separator = range.find('-');
        if(separator != std::string::npos)
        {
            auto start = std::strtoull(range.c_str(), nullptr, 16);
            auto end = std::strtoull(range.c_str() + separator + 1, nullptr, 16);
            for(auto& module : modules)
                if(!path.empty() && start <= module.address && module.address < end)
                {
                    mapped.emplace_back(path);
                    module.found = true;
                }
        }
#endif
    }
#ifdef __linux__
    for(const auto& module : modules)
        if(!module.found)
            return false;
#endif
#endif
    for(const auto& path : mapped)
    {
        std::error_code error;
        if(!fs::equivalent(candidate, path, error) || error)
            return false;
    }
    return true;
}

void* load_comgr(std::string& requested)
{
    const char* override = std::getenv("ROCKE_COMGR_LIB");
    std::string candidate = override && *override ? override : discover_sdk_comgr();
    if(!candidate.empty() && compatible_with_mapped(candidate))
    {
        if(void* library = load(candidate))
        {
            requested = candidate;
            return library;
        }
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
