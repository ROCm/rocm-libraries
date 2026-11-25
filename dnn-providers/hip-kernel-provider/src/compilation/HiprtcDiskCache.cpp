// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HiprtcDiskCache.hpp"

#include "utilities/Checksum.hpp"
#include "utilities/Digest.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <system_error>

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hipdnn_data_sdk/utilities/CacheRoot.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#endif

namespace hip_kernel_provider::compilation::hiprtc_cache
{

namespace
{

/// Bumped when the key's inputs or the file format change, so entries of an older format are
/// never looked up.
constexpr std::string_view CACHE_FORMAT = "hip-kernel-provider-hiprtc-v3";

/// Every entry starts with this, then the payload length and the payload's XXH64 checksum (8
/// bytes each, little-endian). The checksum catches accidental damage; the key, a SHA-256 of the
/// inputs, is what identifies the content.
constexpr std::string_view ENTRY_MAGIC{"hkpco\0\0\3", 8};
constexpr std::size_t LENGTH_BYTES = 8;
constexpr std::size_t CHECKSUM_BYTES = 8;
constexpr std::size_t HEADER_BYTES = ENTRY_MAGIC.size() + LENGTH_BYTES + CHECKSUM_BYTES;

/// Appends @p field with its length first, so no two field sequences hash alike.
void appendField(std::string& key, std::string_view field)
{
    key.append(std::to_string(field.size()));
    key.push_back(':');
    key.append(field);
}

/// The versioned file name and size of @p path, independent of where it is installed.
std::string fileIdentity(const std::filesystem::path& path)
{
    std::error_code error;
    auto resolved = std::filesystem::canonical(path, error);
    if(error)
    {
        resolved = path;
    }
    const auto size = std::filesystem::file_size(resolved, error);
    return "file:" + resolved.filename().string() + ":" + (error ? "?" : std::to_string(size));
}

#if !defined(_WIN32)

struct BuildIdSearch
{
    std::uintptr_t address = 0;
    bool found = false;
    std::string buildId;
};

/// The GNU build id among the notes of one PT_NOTE segment, in hex, or empty.
std::string buildIdInNotes(const unsigned char* notes, std::size_t size)
{
    std::size_t offset = 0;
    const auto align4 = [](std::size_t value) { return (value + 3) & ~std::size_t{3}; };
    while(offset + sizeof(ElfW(Nhdr)) <= size)
    {
        ElfW(Nhdr) header;
        std::memcpy(&header, notes + offset, sizeof(header));
        const std::size_t nameOffset = offset + sizeof(header);
        const std::size_t descOffset = nameOffset + align4(header.n_namesz);
        const std::size_t next = descOffset + align4(header.n_descsz);
        if(next > size || next <= offset)
        {
            return {};
        }
        if(header.n_type == NT_GNU_BUILD_ID && header.n_namesz == 4
           && std::memcmp(notes + nameOffset, "GNU", 4) == 0)
        {
            static constexpr std::array<char, 16> DIGITS
                = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
            std::string hex;
            hex.reserve(std::size_t{2} * header.n_descsz);
            for(std::size_t i = 0; i < header.n_descsz; ++i)
            {
                const auto byte = notes[descOffset + i];
                hex.push_back(DIGITS[byte >> 4]);
                hex.push_back(DIGITS[byte & 0xF]);
            }
            return hex;
        }
        offset = next;
    }
    return {};
}

int findBuildId(dl_phdr_info* info, std::size_t /*size*/, void* data)
{
    auto* search = static_cast<BuildIdSearch*>(data);
    bool contains = false;
    for(ElfW(Half) i = 0; i < info->dlpi_phnum; ++i)
    {
        const auto& segment = info->dlpi_phdr[i];
        const std::uintptr_t start = info->dlpi_addr + segment.p_vaddr;
        if(segment.p_type == PT_LOAD && search->address >= start
           && search->address < start + segment.p_memsz)
        {
            contains = true;
            break;
        }
    }
    if(!contains)
    {
        return 0;
    }
    search->found = true;
    for(ElfW(Half) i = 0; i < info->dlpi_phnum && search->buildId.empty(); ++i)
    {
        const auto& segment = info->dlpi_phdr[i];
        if(segment.p_type == PT_NOTE)
        {
            const std::uintptr_t address = info->dlpi_addr + segment.p_vaddr;
            // The loader reports where the segment is mapped as an address.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            const auto* notes = reinterpret_cast<const unsigned char*>(address);
            search->buildId = buildIdInNotes(notes, segment.p_memsz);
        }
    }
    return 1;
}

#endif

/// The build identity of the loaded library that holds @p address: its ELF build id where it
/// has one, otherwise its versioned file name and size. "unknown" when no library holds it.
std::string libraryIdentity(const void* address)
{
    if(address == nullptr)
    {
        return "unknown";
    }
#if defined(_WIN32)
    HMODULE module = nullptr;
    if(GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          static_cast<LPCWSTR>(address),
                          &module)
       == 0)
    {
        return "unknown";
    }
    std::wstring name(32768, L'\0');
    const auto length = GetModuleFileNameW(module, name.data(), static_cast<DWORD>(name.size()));
    if(length == 0)
    {
        return "unknown";
    }
    name.resize(length);
    return fileIdentity(std::filesystem::path(name));
#else
    BuildIdSearch search;
    search.address = reinterpret_cast<std::uintptr_t>(address);
    dl_iterate_phdr(&findBuildId, &search);
    if(!search.buildId.empty())
    {
        return "build-id:" + search.buildId;
    }
    Dl_info info{};
    if(dladdr(address, &info) == 0 || info.dli_fname == nullptr)
    {
        return "unknown";
    }
    return fileIdentity(info.dli_fname);
#endif
}

std::string computeCompilerIdentity()
{
    std::string identity;

    int major = 0;
    int minor = 0;
    if(hiprtcVersion(&major, &minor) != HIPRTC_SUCCESS)
    {
        major = -1;
    }
    identity += "hiprtc=" + std::to_string(major) + "." + std::to_string(minor);

    int runtime = 0;
    identity += ";hip-runtime="
                + (hipRuntimeGetVersion(&runtime) == hipSuccess ? std::to_string(runtime)
                                                                : std::string("unknown"));

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    identity += ";hiprtc-library="
                + libraryIdentity(reinterpret_cast<const void*>(&hiprtcCompileProgram));

#if defined(_WIN32)
    identity += ";comgr=unknown";
#else
    // comgr is a dependency of both HIPRTC and the HIP runtime, so it is loaded whenever this
    // runs; looked up rather than linked so the provider does not depend on its headers.
    void* getVersion = dlsym(RTLD_DEFAULT, "amd_comgr_get_version");
    if(getVersion != nullptr)
    {
        std::size_t comgrMajor = 0;
        std::size_t comgrMinor = 0;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        reinterpret_cast<void (*)(std::size_t*, std::size_t*)>(getVersion)(&comgrMajor,
                                                                           &comgrMinor);
        identity += ";comgr=" + std::to_string(comgrMajor) + "." + std::to_string(comgrMinor)
                    + ";comgr-library=" + libraryIdentity(getVersion);
    }
    else
    {
        identity += ";comgr=unknown";
    }
#endif
    return identity;
}

void appendLittleEndian(std::vector<char>& out, std::uint64_t value)
{
    for(std::size_t i = 0; i < LENGTH_BYTES; ++i)
    {
        out.push_back(static_cast<char>((value >> (8 * i)) & 0xFF));
    }
}

std::uint64_t readLittleEndian(const char* bytes)
{
    std::uint64_t value = 0;
    for(std::size_t i = 0; i < LENGTH_BYTES; ++i)
    {
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return value;
}

} // namespace

const std::string& compilerIdentity()
{
    static const std::string s_identity = computeCompilerIdentity();
    return s_identity;
}

std::string codeObjectKey(std::string_view compilerIdentity,
                          std::string_view programName,
                          std::string_view source,
                          const std::vector<std::string_view>& includeTexts,
                          const std::vector<const char*>& includeNames,
                          const std::vector<std::string>& options)
{
    std::string key;
    appendField(key, CACHE_FORMAT);
    appendField(key, compilerIdentity);
    appendField(key, programName);
    appendField(key, source);
    for(size_t i = 0; i < includeTexts.size(); ++i)
    {
        appendField(key, includeNames[i]);
        appendField(key, includeTexts[i]);
    }
    for(const auto& option : options)
    {
        appendField(key, option);
    }
    return utilities::sha256Hex(key.data(), key.size());
}

std::filesystem::path codeObjectPath(const std::string& key)
{
    const auto root = hipdnn_data_sdk::utilities::cacheRoot();
    if(root.empty())
    {
        return {};
    }
    return root / "hip-kernel-provider" / "hiprtc" / (key + ".co");
}

std::vector<char> encodeEntry(const std::vector<char>& codeObject)
{
    std::vector<char> entry;
    entry.reserve(HEADER_BYTES + codeObject.size());
    entry.insert(entry.end(), ENTRY_MAGIC.begin(), ENTRY_MAGIC.end());
    appendLittleEndian(entry, codeObject.size());
    appendLittleEndian(entry, utilities::xxh64(codeObject.data(), codeObject.size()));
    entry.insert(entry.end(), codeObject.begin(), codeObject.end());
    return entry;
}

std::vector<char> decodeEntry(const std::vector<char>& entry, std::string& problem)
{
    if(entry.size() < HEADER_BYTES
       || std::string_view(entry.data(), ENTRY_MAGIC.size()) != ENTRY_MAGIC)
    {
        problem = "has no cache entry header";
        return {};
    }
    const auto length = readLittleEndian(entry.data() + ENTRY_MAGIC.size());
    const auto actual = entry.size() - HEADER_BYTES;
    if(length != actual)
    {
        problem = "holds " + std::to_string(actual) + " payload bytes where its header records "
                  + std::to_string(length);
        return {};
    }
    const auto recorded = readLittleEndian(entry.data() + ENTRY_MAGIC.size() + LENGTH_BYTES);
    const char* payload = entry.data() + HEADER_BYTES;
    if(utilities::xxh64(payload, actual) != recorded)
    {
        problem = "does not match the checksum its header records";
        return {};
    }
    problem.clear();
    return {payload, payload + actual};
}

ReadResult readCodeObject(const std::filesystem::path& path)
{
    ReadResult result;
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        return result;
    }
    const std::vector<char> entry((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    result.codeObject = decodeEntry(entry, result.problem);
    result.outcome = result.problem.empty() ? ReadOutcome::VALID : ReadOutcome::CORRUPT;
    return result;
}

bool writeCodeObject(const std::filesystem::path& path, const std::vector<char>& codeObject)
{
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if(error)
    {
        HIPDNN_PLUGIN_LOG_INFO("hipRTC cache: cannot create " << path.parent_path().string() << ": "
                                                              << error.message());
        return false;
    }

    const auto entry = encodeEntry(codeObject);
    std::random_device random;
    auto temporary = path;
    temporary += "." + std::to_string(random()) + std::to_string(random()) + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        file.write(entry.data(), static_cast<std::streamsize>(entry.size()));
        file.flush();
        if(!file)
        {
            HIPDNN_PLUGIN_LOG_INFO("hipRTC cache: cannot write " << temporary.string());
            file.close();
            std::filesystem::remove(temporary, error);
            return false;
        }
    }
    // Rename replaces any entry already there, which on every supported platform is atomic for
    // a reader: it sees the old file or the new one, never a mix.
    std::filesystem::rename(temporary, path, error);
    if(error)
    {
        HIPDNN_PLUGIN_LOG_INFO("hipRTC cache: cannot place " << path.string() << ": "
                                                             << error.message());
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

} // namespace hip_kernel_provider::compilation::hiprtc_cache
