// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipblaslt_jit::source_bundle
{
    namespace fs = std::filesystem;
    inline void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    // True when path, already canonical, stays inside canonical root.
    inline bool pathStaysInside(const fs::path& root, const fs::path& path)
    {
        const auto inside = path.lexically_relative(root);
        return !inside.empty() && *inside.begin() != "..";
    }

    inline fs::path artifact(const fs::path& bundle, const std::string& name, bool mustExist = true)
    {
        auto relative = fs::u8path(name);
        require(!relative.empty() && !relative.has_root_name() && !relative.has_root_directory(),
                "Artifact path must be relative");
        for(const auto& part : relative)
            require(part != "..", "Artifact path escapes bundle");
        const auto path = fs::weakly_canonical(bundle / relative);
        require(pathStaysInside(fs::canonical(bundle), path), "Artifact symlink escapes bundle");
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

    // The sources of a TensileLite bundle, found by directory convention.
    // sources/*.s is the assembly main kernel (at most one). sources/*.hip and
    // sources/*.cpp are HIP: the main kernel when there is no assembly, and
    // helpers otherwise. Every other file in sources/ is a header. A library/
    // entry is rejected: a kernel's solutions come from the kernel itself.
    inline hipblaslt_jit::SourceCodeBundle readSourceBundle(const fs::path& bundle)
    {
        constexpr size_t fileLimit = 1024, byteLimit = 256 * 1024 * 1024;
        const auto       sources = artifact(bundle, "sources", false);
        require(fs::is_directory(sources), "Missing source directory: " + sources.u8string());
        require(!fs::exists(fs::symlink_status(bundle / "library")),
                "A source bundle holds no library entry: " + bundle.u8string());
        std::vector<std::string> names;
        for(const auto& entry : fs::directory_iterator(sources))
        {
            require(names.size() < fileLimit, "Too many files in " + sources.u8string());
            names.push_back(entry.path().filename().u8string());
            require(entry.is_regular_file(), "Unexpected source entry: " + names.back());
        }
        std::sort(names.begin(), names.end());
        size_t                               total = 0;
        std::vector<hipblaslt_jit::Artifact> assembly, hip, headers;
        for(const auto& name : names)
        {
            hipblaslt_jit::Artifact file;
            file.name  = name;
            file.bytes = readArtifact(artifact(bundle, "sources/" + name));
            total += file.bytes.size();
            require(total <= byteLimit, "Source bundle is too large: " + sources.u8string());
            const auto extension = fs::u8path(name).extension();
            if(extension == ".s")
            {
                file.kind = hipblaslt_jit::Artifact::Kind::Assembly;
                assembly.push_back(std::move(file));
            }
            else if(extension == ".hip" || extension == ".cpp")
            {
                file.kind = hipblaslt_jit::Artifact::Kind::Hip;
                hip.push_back(std::move(file));
            }
            else
                headers.push_back(std::move(file));
        }
        require(!assembly.empty() || !hip.empty(), "No kernel source in " + sources.u8string());
        require(assembly.size() <= 1, "A source bundle has one main kernel");
        hipblaslt_jit::SourceCodeBundle result;
        result.includes = std::move(headers);
        if(!assembly.empty())
        {
            result.mainKernel = std::move(assembly.front());
            result.helpers    = std::move(hip);
        }
        else
        {
            result.mainKernel = std::move(hip.front());
            hip.erase(hip.begin());
            result.helpers = std::move(hip);
        }
        return result;
    }
}
