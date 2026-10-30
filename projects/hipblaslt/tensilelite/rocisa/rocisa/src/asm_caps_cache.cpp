// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "asm_caps_cache.hpp"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <sstream>
#include <system_error>
#include <thread>

// Identifies the probe sources (CMakeLists.txt hashes them): editing a probe
// must not return what an earlier rocisa cached.
#ifndef ROCISA_ASM_CAPS_PROBE_HASH
#error "ROCISA_ASM_CAPS_PROBE_HASH must be defined"
#endif

namespace fs = std::filesystem;

namespace
{
    constexpr const char* kFormat = "rocisa-asm-caps 1";

    // Unset and empty are the same: no value.
    std::optional<std::string> envVar(const char* name)
    {
#ifdef _WIN32
        // MSVC's CRT deprecates getenv in favour of _dupenv_s.
        char*  value = nullptr;
        size_t len   = 0;
        if(_dupenv_s(&value, &len, name) != 0 || !value)
            return std::nullopt;
        std::string s(value);
        std::free(value);
#else
        const char* value = std::getenv(name);
        if(!value)
            return std::nullopt;
        std::string s(value);
#endif
        if(s.empty())
            return std::nullopt;
        return s;
    }

    bool envFlag(const char* name)
    {
        auto value = envVar(name);
        if(!value)
            return false;
        for(auto& c : *value)
            c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return *value == "1" || *value == "YES" || *value == "ON" || *value == "TRUE";
    }

    std::optional<fs::path> cacheDir()
    {
        if(envFlag("TENSILE_DISABLE_ASM_CAPS_CACHE"))
            return std::nullopt;
        if(auto dir = envVar("TENSILE_ASM_CAPS_CACHE_DIR"))
            return fs::path(*dir);
#ifdef _WIN32
        auto home = envVar("USERPROFILE");
#else
        auto home = envVar("HOME");
#endif
        if(!home)
            return std::nullopt;
        return fs::path(*home) / ".tensile" / "asm_caps_cache";
    }

    // Only names the entry's file: the whole key is stored in it and compared on load.
    std::uint64_t fnv1a(const std::string& s)
    {
        std::uint64_t h = 14695981039346656037ull;
        for(unsigned char c : s)
        {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }

    struct Entry
    {
        fs::path    file;
        std::string key;
    };

    // The assembler is identified the way ccache identifies a compiler by
    // default: its resolved path, size and modification time.
    std::optional<Entry> entryFor(const IsaVersion& isaVersion, const std::string& assemblerPath)
    {
        auto dir = cacheDir();
        if(!dir || assemblerPath.empty())
            return std::nullopt;
        std::error_code ec;
        fs::path        assembler = fs::canonical(assemblerPath, ec);
        if(ec)
            return std::nullopt;
        auto size = fs::file_size(assembler, ec);
        if(ec)
            return std::nullopt;
        auto mtime = fs::last_write_time(assembler, ec);
        if(ec)
            return std::nullopt;

        std::string        isa = getGfxNameTuple(isaVersion);
        std::ostringstream key;
        key << kFormat << "\n"
            << "probes " << ROCISA_ASM_CAPS_PROBE_HASH << "\n"
            << "isa " << isa << "\n"
            << "assembler " << assembler.string() << "\n"
            << "size " << size << "\n"
            << "mtime " << mtime.time_since_epoch().count() << "\n";
        std::ostringstream name;
        name << isa << "-" << std::hex << fnv1a(key.str()) << ".txt";
        return Entry{*dir / name.str(), key.str()};
    }
} // namespace

namespace rocisa
{
    std::optional<std::map<std::string, int>> loadAsmCaps(const IsaVersion&  isaVersion,
                                                          const std::string& assemblerPath)
    {
        try
        {
            auto entry = entryFor(isaVersion, assemblerPath);
            if(!entry)
                return std::nullopt;
            std::ifstream in(entry->file, std::ios::binary);
            if(!in)
                return std::nullopt;
            std::string text((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
            if(text.compare(0, entry->key.size(), entry->key) != 0)
                return std::nullopt;

            std::istringstream         body(text.substr(entry->key.size()));
            std::map<std::string, int> caps;
            std::string                line;
            while(std::getline(body, line))
            {
                // Writers end every entry with this line, so a truncated file is a miss.
                if(line == "end")
                    return caps;
                std::istringstream fields(line);
                std::string        name, extra;
                int                value;
                if(!(fields >> name >> value) || (fields >> extra))
                    return std::nullopt;
                caps[name] = value;
            }
            return std::nullopt;
        }
        catch(const std::exception&)
        {
            return std::nullopt;
        }
    }

    void storeAsmCaps(const IsaVersion&                 isaVersion,
                      const std::string&                assemblerPath,
                      const std::map<std::string, int>& asmCaps)
    {
        try
        {
            auto entry = entryFor(isaVersion, assemblerPath);
            if(!entry)
                return;
            std::error_code ec;
            fs::create_directories(entry->file.parent_path(), ec);
            if(ec)
                return;

            // Unique per writer, so concurrent builds never write the same file.
            std::ostringstream tmpName;
            tmpName << entry->file.filename().string() << ".tmp." << std::hex
                    << std::hash<std::thread::id>{}(std::this_thread::get_id()) << "."
                    << std::random_device{}();
            fs::path tmp = entry->file.parent_path() / tmpName.str();
            {
                std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
                out << entry->key;
                for(const auto& [name, value] : asmCaps)
                    out << name << " " << value << "\n";
                out << "end\n";
                if(!out)
                {
                    out.close();
                    fs::remove(tmp, ec);
                    return;
                }
            }
            // Readers see either no entry or a complete one.
            fs::rename(tmp, entry->file, ec);
            if(ec)
                fs::remove(tmp, ec);
        }
        catch(const std::exception&)
        {
        }
    }
} // namespace rocisa
