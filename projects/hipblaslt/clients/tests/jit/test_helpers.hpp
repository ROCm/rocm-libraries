// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>

// Helpers the JIT tests share. Assertions go through Catch2.
namespace hipblaslt_jit_test
{
    // Requires that f throws.
    inline void reject(const std::function<void()>& f, const std::string& message)
    {
        bool failed = false;
        try
        {
            f();
        }
        catch(const std::exception&)
        {
            failed = true;
        }
        {
            INFO((message));
            REQUIRE((failed));
        }
    }

    inline std::string readFile(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        {
            INFO(("Cannot read " + path.u8string()));
            REQUIRE((bool(input)));
        }
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    inline void writeFile(const std::filesystem::path& path, const std::string& bytes)
    {
        std::ofstream file(path, std::ios::binary);
        {
            INFO(("Cannot write " + path.u8string()));
            REQUIRE((bool(file.write(bytes.data(), bytes.size()))));
        }
    }

    // The written bundles of the processor in a device's gcnArchName, such as
    // root/gfx942 for "gfx942:sramecc+:xnack-".
    inline std::filesystem::path deviceBundles(const std::filesystem::path& root,
                                               const std::string&           gcnArchName)
    {
        const auto bundles = root / gcnArchName.substr(0, gcnArchName.find(':'));
        {
            INFO(("No JIT test bundles for " + gcnArchName + " in " + root.u8string()));
            REQUIRE((std::filesystem::is_directory(bundles)));
        }
        return bundles;
    }
}
