// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>

// REQUIRE that reports `message` when it fails. The message applies to this
// assertion only, and the condition may contain commas.
#define JIT_REQUIRE(message, ...) \
    do                            \
    {                             \
        INFO((message));          \
        REQUIRE((__VA_ARGS__));   \
    } while(false)

// Helpers the JIT tests share. Assertions go through Catch2.
namespace hipblaslt_jit_test
{
    // Requires that f throws.
    inline void reject(const std::function<void()>& f,
                       const std::string&           message = "Malformed artifact was accepted")
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
        JIT_REQUIRE(message, failed);
    }

    inline std::string readFile(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        JIT_REQUIRE("Cannot read " + path.u8string(), bool(input));
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    inline void writeFile(const std::filesystem::path& path, const std::string& bytes)
    {
        std::ofstream file(path, std::ios::binary);
        JIT_REQUIRE("Cannot write " + path.u8string(),
                    bool(file.write(bytes.data(), bytes.size())));
    }

    inline void checkHip(hipError_t status, const char* expression)
    {
        JIT_REQUIRE(std::string(expression) + ": " + hipGetErrorString(status),
                    status == hipSuccess);
    }

    struct Device
    {
        void* pointer{};
        explicit Device(size_t bytes)
        {
            checkHip(hipMalloc(&pointer, bytes), "hipMalloc");
        }
        ~Device()
        {
            static_cast<void>(hipFree(pointer));
        }
        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;
    };
}
