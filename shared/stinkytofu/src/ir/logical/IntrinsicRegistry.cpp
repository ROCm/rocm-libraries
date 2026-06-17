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

#include "stinkytofu/ir/logical/IntrinsicRegistry.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>

#ifndef _WIN32
#include <dlfcn.h>
#include <linux/limits.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

namespace stinkytofu {
IntrinsicRegistry::IntrinsicRegistry() : initialized_(false) {
    // Automatically find and load intrinsics.st.bc on construction
    std::string path = findIntrinsicsBitcode();
    if (!path.empty()) {
        library_ = IntrinsicLibrary::loadFromFile(path);
        if (library_) {
            loadedPath_ = path;
            initialized_ = true;
            std::cout << "IntrinsicRegistry: Loaded " << library_->size() << " intrinsics from "
                      << path << "\n";
        } else {
            std::cerr << "IntrinsicRegistry: Failed to load intrinsics from " << path << "\n";
        }
    } else {
        std::cerr << "IntrinsicRegistry: intrinsics.st.bc not found in search paths\n";
    }
}

IntrinsicRegistry& IntrinsicRegistry::instance() {
    static IntrinsicRegistry registry;
    return registry;
}

bool IntrinsicRegistry::hasIntrinsic(const std::string& name) const {
    return library_ && library_->hasIntrinsic(name);
}

const Pattern* IntrinsicRegistry::lookup(const std::string& name) const {
    return library_ ? library_->lookup(name) : nullptr;
}

std::vector<std::string> IntrinsicRegistry::getIntrinsicNames() const {
    return library_ ? library_->getIntrinsicNames() : std::vector<std::string>();
}

bool IntrinsicRegistry::isInitialized() const {
    return initialized_;
}

std::string IntrinsicRegistry::getLoadedPath() const {
    return loadedPath_;
}

bool IntrinsicRegistry::reload(const std::string& path) {
    library_ = IntrinsicLibrary::loadFromFile(path);
    if (library_) {
        loadedPath_ = path;
        initialized_ = true;
        return true;
    }
    return false;
}

std::shared_ptr<IntrinsicLibrary> IntrinsicRegistry::getLibrary() const {
    return library_;
}

std::string IntrinsicRegistry::findIntrinsicsBitcode() {
    auto paths = getSearchPaths();

    for (const auto& path : paths) {
        std::ifstream f(path);
        if (f.good()) {
            return path;
        }
    }

    return "";
}

std::vector<std::string> IntrinsicRegistry::getSearchPaths() {
    std::vector<std::string> paths;

    // 1. Environment variable override (highest priority)
#ifdef _WIN32
    char* envPath = nullptr;
    size_t envLen = 0;
    _dupenv_s(&envPath, &envLen, "STINKYTOFU_INTRINSICS_PATH");
#else
    const char* envPath = std::getenv("STINKYTOFU_INTRINSICS_PATH");
#endif
    if (envPath) {
        paths.push_back(envPath);
#ifdef _WIN32
        free(envPath);
#endif
    }

    // 2. Relative to the loaded shared library (PRIMARY)
    std::string libPath = getLibraryPath();
    if (!libPath.empty()) {
        size_t lastSlash = libPath.find_last_of("/\\");
        if (lastSlash != std::string::npos) {
            std::string libDir = libPath.substr(0, lastSlash);

            // FIRST: Search same directory and child directories
            // This is the typical install layout:
            //   /usr/local/lib/libstinkytofu.so
            //   /usr/local/lib/stinkytofu/intrinsics.st.bc  <- child directory
            paths.push_back(libDir + "/intrinsics.st.bc");             // Same dir
            paths.push_back(libDir + "/stinkytofu/intrinsics.st.bc");  // Child dir
            paths.push_back(libDir + "/Intrinsics/intrinsics.st.bc");  // Child dir

            // SECOND: Walk up directory tree
            // This handles the build directory layout:
            //   build/lib/stinkytofu/_stinkytofu.so
            //   build/intrinsics.st.bc  <- 3 levels up
            std::string current = libDir;
            for (int level = 0; level < 5; ++level) {
                // Go up one level
                size_t slash = current.find_last_of("/\\");
                if (slash == std::string::npos) break;
                current = current.substr(0, slash);

                // Check at this parent level
                paths.push_back(current + "/intrinsics.st.bc");
                paths.push_back(current + "/Intrinsics/intrinsics.st.bc");
                paths.push_back(current + "/lib/Intrinsics/intrinsics.st.bc");
                paths.push_back(current + "/lib/stinkytofu/intrinsics.st.bc");
                paths.push_back(current + "/share/stinkytofu/intrinsics.st.bc");
            }
        }
    }

    // 3. Current working directory (fallback)
    paths.push_back("intrinsics.st.bc");

    // 4. Standard system installation paths (fallback)
    paths.push_back("/usr/local/lib/stinkytofu/intrinsics.st.bc");
    paths.push_back("/usr/local/share/stinkytofu/intrinsics.st.bc");
    paths.push_back("/usr/lib/stinkytofu/intrinsics.st.bc");
    paths.push_back("/opt/rocm/lib/stinkytofu/intrinsics.st.bc");

    return paths;
}

std::string IntrinsicRegistry::getLibraryPath() {
    // Get the path of the loaded shared library (not the Python interpreter!)
#ifndef _WIN32
    Dl_info info;
    // Use dladdr with a known symbol from this library
    if (dladdr((void*)&IntrinsicRegistry::instance, &info) && info.dli_fname) {
        char resolved[PATH_MAX];
        if (realpath(info.dli_fname, resolved)) {
            return std::string(resolved);
        }
        return std::string(info.dli_fname);
    }
#else
    HMODULE hModule;
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&IntrinsicRegistry::instance, &hModule)) {
        char result[MAX_PATH];
        DWORD count = GetModuleFileNameA(hModule, result, MAX_PATH);
        if (count != 0) {
            return std::string(result);
        }
    }
#endif
    return "";
}

std::string IntrinsicRegistry::getExecutablePath() {
#ifndef _WIN32
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    if (count != -1) {
        return std::string(result, count);
    }
#else
    char result[MAX_PATH];
    DWORD count = GetModuleFileNameA(NULL, result, MAX_PATH);
    if (count != 0) {
        return std::string(result);
    }
#endif
    return "";
}

}  // namespace stinkytofu
