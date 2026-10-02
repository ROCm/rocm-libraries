// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace hipblaslt_jit::source_bundle
{
    namespace fs = std::filesystem;
    inline void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    inline fs::path artifact(const fs::path& bundle, const std::string& name, bool mustExist = true)
    {
        auto relative = fs::u8path(name);
        require(!relative.empty() && !relative.has_root_name() && !relative.has_root_directory(),
                "Artifact path must be relative");
        for(const auto& part : relative)
            require(part != "..", "Artifact path escapes bundle");
        const auto path   = fs::weakly_canonical(bundle / relative);
        const auto inside = path.lexically_relative(fs::canonical(bundle));
        require(!inside.empty() && *inside.begin() != "..", "Artifact symlink escapes bundle");
        if(mustExist)
            require(fs::is_regular_file(path), "Missing artifact: " + path.string());
        return path;
    }

    inline std::vector<uint8_t> readArtifact(const fs::path& path)
    {
        constexpr size_t limit = 64 * 1024 * 1024;
        const auto       size  = fs::file_size(path);
        require(size > 0 && size <= limit, "Artifact has an invalid size: " + path.u8string());
        std::ifstream        input(path, std::ios::binary);
        std::vector<uint8_t> bytes(size);
        require(bool(input.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
                    && input.peek() == std::char_traits<char>::eof(),
                "Cannot read complete artifact: " + path.u8string());
        return bytes;
    }

    inline std::vector<uint8_t> readLibrary(const fs::path& path)
    {
        auto bytes = readArtifact(path);
        if(path.extension() != ".zlib")
            return bytes;
        struct Inflater
        {
            z_stream stream{};
            Inflater()
            {
                require(inflateInit(&stream) == Z_OK, "Cannot initialize library decoder");
            }
            ~Inflater()
            {
                inflateEnd(&stream);
            }
        } decoder;
        auto& stream               = decoder.stream;
        stream.next_in             = bytes.data();
        stream.avail_in            = static_cast<uInt>(bytes.size());
        constexpr size_t     chunk = 64 * 1024, limit = 64 * 1024 * 1024;
        std::vector<uint8_t> decoded;
        int                  status;
        do
        {
            require(decoded.size() < limit, "Decoded solution library is too large");
            const auto offset         = decoded.size();
            const auto inputRemaining = stream.avail_in;
            decoded.resize(offset + chunk);
            stream.next_out  = decoded.data() + offset;
            stream.avail_out = chunk;
            status           = inflate(&stream, Z_NO_FLUSH);
            require(status == Z_OK || status == Z_STREAM_END,
                    "Invalid compressed solution library");
            decoded.resize(offset + chunk - stream.avail_out);
            require(status == Z_STREAM_END || decoded.size() > offset
                        || stream.avail_in < inputRemaining,
                    "Truncated compressed solution library");
        } while(status != Z_STREAM_END);
        require(stream.avail_in == 0 && !decoded.empty(),
                "Trailing or empty solution library data");
        return decoded;
    }

    struct SourceFile
    {
        std::string          name;
        std::vector<uint8_t> bytes;
    };

    // The build inputs of a TensileLite source bundle, found by directory
    // convention: library/TensileLibrary.* is the solution library entry,
    // sources/*.s are the main kernels, sources/Kernels.cpp holds the helper
    // kernels, and every other file in sources/ is a header they include.
    struct SourceBundle
    {
        std::vector<uint8_t>    library; // decoded
        std::vector<SourceFile> assembly; // by name
        std::vector<SourceFile> helpers;
        std::vector<SourceFile> headers; // by name
    };

    inline SourceBundle readSourceBundle(const fs::path& bundle)
    {
        constexpr size_t fileLimit = 1024, byteLimit = 256 * 1024 * 1024;
        SourceBundle     result;
        std::vector<fs::path> libraries;
        for(const char* name : {"library/TensileLibrary.dat.zlib",
                                "library/TensileLibrary.dat",
                                "library/TensileLibrary.yaml"})
            if(fs::exists(bundle / name))
                libraries.push_back(artifact(bundle, name));
        require(libraries.size() == 1,
                "Expected one solution library in " + (bundle / "library").u8string());
        result.library = readLibrary(libraries.front());

        const auto sources = artifact(bundle, "sources", false);
        require(fs::is_directory(sources), "Missing source directory: " + sources.u8string());
        std::vector<std::string> names;
        for(const auto& entry : fs::directory_iterator(sources))
        {
            require(names.size() < fileLimit, "Too many files in " + sources.u8string());
            names.push_back(entry.path().filename().u8string());
            require(entry.is_regular_file(), "Unexpected source entry: " + names.back());
        }
        std::sort(names.begin(), names.end());
        size_t total = 0;
        for(const auto& name : names)
        {
            SourceFile file{name, readArtifact(artifact(bundle, "sources/" + name))};
            total += file.bytes.size();
            require(total <= byteLimit, "Source bundle is too large: " + sources.u8string());
            auto& files = fs::u8path(name).extension() == ".s" ? result.assembly
                          : name == "Kernels.cpp"             ? result.helpers
                                                              : result.headers;
            files.push_back(std::move(file));
        }
        require(!result.assembly.empty(), "No main kernel assembly in " + sources.u8string());
        return result;
    }
}
