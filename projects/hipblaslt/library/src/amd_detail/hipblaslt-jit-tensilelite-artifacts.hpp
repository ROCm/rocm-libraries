// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <zlib.h>

namespace hipblaslt_ext::experimental::jit::tensilelite::detail::artifacts
{
    namespace fs = std::filesystem;
    using Tree   = std::map<std::string, std::string>;
    inline void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }
    inline uint32_t readCount(std::istream& input)
    {
        unsigned char bytes[4];
        require(bool(input.read(reinterpret_cast<char*>(bytes), 4)), "Truncated loader envelope");
        return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16
               | uint32_t(bytes[3]) << 24;
    }

    inline std::string readString(std::istream& input)
    {
        auto size = readCount(input);
        require(size > 0 && size <= 1048576, "Invalid loader field length");
        std::string value(size, '\0');
        require(bool(input.read(value.data(), size)) && value.find('\0') == std::string::npos,
                "Invalid or truncated loader field");
        return value;
    }

    inline Tree readEnvelope(const fs::path& path, std::vector<std::string>& objects)
    {
        require(fs::file_size(path) <= 16 * 1024 * 1024, "Loader envelope is too large");
        std::ifstream input(path, std::ios::binary);
        char          magic[8];
        require(bool(input.read(magic, 8)) && std::string(magic, 8) == "TLJIT001",
                "Unsupported loader envelope version");
        const char* fields[] = {"schema_version",
                                "counts.solutions",
                                "counts.main_kernels",
                                "solution.index",
                                "main_kernel.name",
                                "solution.kernel_name",
                                "solution.name",
                                "architecture.requested",
                                "architecture.resolved",
                                "architecture.compiler_target",
                                "library.format",
                                "main_kernel.code_object",
                                "library.path",
                                "library.logical_path"};
        require(readCount(input) == std::size(fields), "Invalid loader field count");
        Tree tree;
        for(auto name : fields)
            tree.emplace(name, readString(input));
        auto count = readCount(input);
        require(count > 0 && count <= 4096, "Invalid loader code object count");
        for(uint32_t i = 0; i < count; ++i)
            objects.push_back(readString(input));
        require(input.peek() == std::char_traits<char>::eof(), "Trailing loader envelope data");
        return tree;
    }

    inline const std::string& field(const Tree& tree, const char* name)
    {
        return tree.at(name);
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

}
