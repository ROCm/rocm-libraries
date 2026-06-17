/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/pipeline/PassBuilder.hpp"

#include "stinkytofu/Version.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <filesystem>
#include <iostream>

#ifndef _WIN32
#define STINKYTOFU_SONAME "libstinkytofu.so." STINKYTOFU_STRINGIFY_(STINKYTOFU_VERSION_MAJOR)
#define STINKYTOFU_STRINGIFY_(x) STINKYTOFU_STRINGIFY__(x)
#define STINKYTOFU_STRINGIFY__(x) #x
#endif

namespace stinkytofu {

void PassBuilder::registerAtExtensionPoint(PipelineExtensionPoint EP, ExtensionCallback CB) {
    callbacks_[static_cast<int>(EP)].push_back(std::move(CB));
}

void PassBuilder::applyExtensionPoint(PipelineExtensionPoint EP, PassManager& PM,
                                      StinkyAsmModule& module) const {
    auto it = callbacks_.find(static_cast<int>(EP));
    if (it == callbacks_.end()) return;
    for (const auto& cb : it->second) {
        cb(PM, module);
    }
}

struct PassBuilder::FactoryRegistry {
    std::unordered_map<std::string, PassFactory> factories;
    std::vector<void*> loadedPlugins;
    std::mutex mu;

    ~FactoryRegistry() = default;
};

PassBuilder::FactoryRegistry& PassBuilder::getFactoryRegistry() {
    static FactoryRegistry registry;
    return registry;
}

void PassBuilder::registerNamedPassFactory(const std::string& name, PassFactory factory) {
    auto& reg = getFactoryRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.factories[name] = std::move(factory);
}

std::unique_ptr<Pass> PassBuilder::createPassByName(const std::string& name,
                                                    StinkyAsmModule& module) {
    PassFactory factory;
    {
        auto& reg = getFactoryRegistry();
        std::lock_guard<std::mutex> lock(reg.mu);
        auto it = reg.factories.find(name);
        if (it == reg.factories.end()) {
            return nullptr;
        }
        factory = it->second;
    }
    return factory(module);
}

bool PassBuilder::loadPlugin(const std::string& path) {
#ifdef _WIN32
    HMODULE handle = LoadLibraryA(path.c_str());
    if (!handle) {
        std::cerr << "PassBuilder: failed to load plugin '" << path << "': error code "
                  << GetLastError() << '\n';
        return false;
    }

    using RegisterFn = void (*)();
    auto* registerFn = reinterpret_cast<RegisterFn>(GetProcAddress(handle, "registerPlugin"));
    if (!registerFn) {
        std::cerr << "PassBuilder: plugin '" << path << "' does not export registerPlugin()"
                  << '\n';
        FreeLibrary(handle);
        return false;
    }
#else
    // Promote the already-loaded libstinkytofu to RTLD_GLOBAL so the plugin
    // resolves stinkytofu symbols from the host's copy, not a second one.
    // RTLD_NOLOAD prevents loading a new copy — it only changes visibility.
    // dlclose the returned handle to avoid leaking a reference count.
    if (void* self = dlopen(STINKYTOFU_SONAME, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL)) dlclose(self);

    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::cerr << "PassBuilder: failed to load plugin '" << path << "': " << dlerror() << '\n';
        return false;
    }

    using RegisterFn = void (*)();
    auto* registerFn = reinterpret_cast<RegisterFn>(dlsym(handle, "registerPlugin"));
    if (!registerFn) {
        std::cerr << "PassBuilder: plugin '" << path
                  << "' does not export registerPlugin(): " << dlerror() << '\n';
        dlclose(handle);
        return false;
    }
#endif

    registerFn();

    auto& reg = getFactoryRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    reg.loadedPlugins.push_back(reinterpret_cast<void*>(handle));
    return true;
}

void PassBuilder::loadPluginsFromDirectory(const std::string& dirPath) {
    namespace fs = std::filesystem;
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) return;

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext == ".so" || ext == ".dll") {
            loadPlugin(entry.path().string());
        }
    }
}

void PassBuilder::unloadPlugins() {
    std::vector<void*> handles;
    {
        auto& reg = getFactoryRegistry();
        std::lock_guard<std::mutex> lock(reg.mu);
        reg.factories.clear();
        handles.swap(reg.loadedPlugins);
    }
    for (auto* handle : handles) {
#ifdef _WIN32
        FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
        dlclose(handle);
#endif
    }
}

}  // namespace stinkytofu
