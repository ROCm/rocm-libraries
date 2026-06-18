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
#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {

class StinkyAsmModule;

/// Named injection points in the optimization pipeline where plugin
/// passes can be inserted.  Mirrors LLVM's PassBuilder extension points.
enum class PipelineExtensionPoint {
    /// Whole-kernel scope, before the region ScopeAdaptor.
    BeforeRegionPasses,
    /// Inside the per-region PassManager, before the DAG scheduler.
    InnerRegionBegin,
    /// Inside the per-region PassManager, after the DAG scheduler
    /// but before waitcnt insertion.
    InnerRegionEnd,
    /// Whole-kernel scope, after the region ScopeAdaptor.
    AfterRegionPasses,
};

/// LLVM-style pass builder that collects plugin-registered callbacks
/// and applies them at named extension points during pipeline construction.
///
/// Usage from a pipeline builder (e.g. Gfx1250Backend):
/// @code
///   PB.applyExtensionPoint(PipelineExtensionPoint::InnerRegionEnd, innerPM, module);
/// @endcode
///
/// Usage from a plugin (e.g. TensileLite barrier insertion pass):
/// @code
///   module.getPassBuilder().registerAtExtensionPoint(
///       PipelineExtensionPoint::InnerRegionEnd,
///       [](PassManager& PM, StinkyAsmModule&) {
///           PM.addPass(createBarrierInsertionPass());
///       });
/// @endcode
class STINKYTOFU_EXPORT PassBuilder {
   public:
    using ExtensionCallback = std::function<void(PassManager&, StinkyAsmModule&)>;

    /// Register a callback to be invoked at the given extension point.
    void registerAtExtensionPoint(PipelineExtensionPoint EP, ExtensionCallback CB);

    /// Invoke all callbacks registered at the given extension point.
    void applyExtensionPoint(PipelineExtensionPoint EP, PassManager& PM,
                             StinkyAsmModule& module) const;

    /// Factory type: creates a pass given a module reference.
    /// Plugins that need module data (e.g., pluginDataI64/Str) use this signature.
    using PassFactory = std::function<std::unique_ptr<Pass>(StinkyAsmModule&)>;

    /// Register a named pass factory.  Plugins call this at static init
    /// or module init time so that Python can refer to passes by name.
    static void registerNamedPassFactory(const std::string& name, PassFactory factory);

    /// Create a pass by name.  Returns nullptr if the name is not registered.
    static std::unique_ptr<Pass> createPassByName(const std::string& name, StinkyAsmModule& module);

    /// Load all plugin shared libraries (.so/.dll) from the given directory.
    /// Each plugin must export `extern "C" void registerPlugin()`.
    /// Loaded plugins call registerNamedPassFactory() during their
    /// registerPlugin() to make their passes available by name.
    static void loadPluginsFromDirectory(const std::string& dirPath);

    /// Load a single plugin shared library (.so on Linux, .dll on Windows).
    static bool loadPlugin(const std::string& path);

    /// Absolute path to StinkyTofu's own bundled example plugin, or "" if it was
    /// not built (STINKYTOFU_BUILD_EXAMPLES=OFF) or cannot be located on disk.
    ///
    /// This is stinkytofu answering "where is *my* plugin", computed relative to
    /// the loaded libstinkytofu — it is not a general plugin lookup. Consumers
    /// that ship their own plugins (rocisa, TensileLite) know their own install
    /// location and pass it straight to loadPlugin(); they do not call this.
    static std::string examplePluginPath();

    /// Explicitly close all loaded plugin handles.
    /// C++ callers should invoke this before shutdown.
    /// Python callers should NOT call this — plugin handles are
    /// cleaned up by the OS when the process exits.
    static void unloadPlugins();

   private:
    std::unordered_map<int, std::vector<ExtensionCallback>> callbacks_;

    struct FactoryRegistry;
    static FactoryRegistry& getFactoryRegistry();
};

}  // namespace stinkytofu
