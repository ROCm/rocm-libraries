/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "ir/IntrinsicRegistry.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>

#ifndef _WIN32
#include <linux/limits.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace stinkytofu
{
    IntrinsicRegistry::IntrinsicRegistry()
        : initialized_(false)
    {
        // Automatically find and load intrinsics.st.bc on construction
        std::string path = findIntrinsicsBitcode();
        if(!path.empty())
        {
            library_ = IntrinsicLibrary::loadFromFile(path);
            if(library_)
            {
                loadedPath_  = path;
                initialized_ = true;
                std::cout << "IntrinsicRegistry: Loaded " << library_->size() << " intrinsics from "
                          << path << "\n";
            }
            else
            {
                std::cerr << "IntrinsicRegistry: Failed to load intrinsics from " << path << "\n";
            }
        }
        else
        {
            std::cerr << "IntrinsicRegistry: intrinsics.st.bc not found in search paths\n";
        }
    }

    IntrinsicRegistry& IntrinsicRegistry::instance()
    {
        static IntrinsicRegistry registry;
        return registry;
    }

    bool IntrinsicRegistry::hasIntrinsic(const std::string& name) const
    {
        return library_ && library_->hasIntrinsic(name);
    }

    const Pattern* IntrinsicRegistry::lookup(const std::string& name) const
    {
        return library_ ? library_->lookup(name) : nullptr;
    }

    std::vector<std::string> IntrinsicRegistry::getIntrinsicNames() const
    {
        return library_ ? library_->getIntrinsicNames() : std::vector<std::string>();
    }

    bool IntrinsicRegistry::isInitialized() const
    {
        return initialized_;
    }

    std::string IntrinsicRegistry::getLoadedPath() const
    {
        return loadedPath_;
    }

    bool IntrinsicRegistry::reload(const std::string& path)
    {
        library_ = IntrinsicLibrary::loadFromFile(path);
        if(library_)
        {
            loadedPath_  = path;
            initialized_ = true;
            return true;
        }
        return false;
    }

    std::shared_ptr<IntrinsicLibrary> IntrinsicRegistry::getLibrary() const
    {
        return library_;
    }

    std::string IntrinsicRegistry::findIntrinsicsBitcode()
    {
        auto paths = getSearchPaths();

        for(const auto& path : paths)
        {
            std::ifstream f(path);
            if(f.good())
            {
                return path;
            }
        }

        return "";
    }

    std::vector<std::string> IntrinsicRegistry::getSearchPaths()
    {
        std::vector<std::string> paths;

        // 1. Environment variable override
        const char* envPath = std::getenv("STINKYTOFU_INTRINSICS_PATH");
        if(envPath)
        {
            paths.push_back(envPath);
        }

        // 2. Current directory
        paths.push_back("intrinsics.st.bc");
        paths.push_back("./intrinsics.st.bc");

        // 3. Build directory (relative to current location)
        paths.push_back("../intrinsics.st.bc");
        paths.push_back("../../intrinsics.st.bc");
        paths.push_back("../build/intrinsics.st.bc");
        paths.push_back("../../build/intrinsics.st.bc");

        // 4. Get executable/library path and search relative to it
        std::string execPath = getExecutablePath();
        if(!execPath.empty())
        {
            // Assuming libstinkytofu.so is in <install>/lib/
            // intrinsics.st.bc should be in <install>/lib/stinkytofu/
            size_t lastSlash = execPath.find_last_of("/\\");
            if(lastSlash != std::string::npos)
            {
                std::string libDir = execPath.substr(0, lastSlash);
                paths.push_back(libDir + "/stinkytofu/intrinsics.st.bc");
                paths.push_back(libDir + "/intrinsics.st.bc");

                // Also try going up one level (lib/ -> prefix/)
                size_t secondLastSlash = libDir.find_last_of("/\\");
                if(secondLastSlash != std::string::npos)
                {
                    std::string prefix = libDir.substr(0, secondLastSlash);
                    paths.push_back(prefix + "/lib/stinkytofu/intrinsics.st.bc");
                    paths.push_back(prefix + "/share/stinkytofu/intrinsics.st.bc");
                }
            }
        }

        // 5. Standard installation paths (Linux)
        paths.push_back("/usr/local/lib/stinkytofu/intrinsics.st.bc");
        paths.push_back("/usr/lib/stinkytofu/intrinsics.st.bc");
        paths.push_back("/opt/rocm/lib/stinkytofu/intrinsics.st.bc");

        return paths;
    }

    std::string IntrinsicRegistry::getExecutablePath()
    {
#ifndef _WIN32
        char    result[PATH_MAX];
        ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
        if(count != -1)
        {
            return std::string(result, count);
        }
#else
        char  result[MAX_PATH];
        DWORD count = GetModuleFileNameA(NULL, result, MAX_PATH);
        if(count != 0)
        {
            return std::string(result);
        }
#endif
        return "";
    }

} // namespace stinkytofu
