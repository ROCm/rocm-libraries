/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

#include <array>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adt/IntrusiveList.hpp"

namespace stinkytofu
{
    // Forward declarations
    class PassContext;

    struct StinkyKernelInfo
    {
        std::array<int, 3> arch{0, 0, 0};
        uint32_t           TileA0;
        uint32_t           TileB0;
        uint32_t           TileM0;
        uint32_t           NumGRA;
        uint32_t           NumGRB;
        uint32_t           NumGRM;
        uint32_t           WavefrontSize;
        uint32_t           NumWaves;
    };

    struct StinkyOptInfo
    {
        bool unrollGemmMovableBarrier = false;
        bool unrollGemm               = false;
        bool distributeGlobalRead     = false;
    };

    class IRBase : public IntrusiveListNode<IRBase>
    {
    public:
        // Stinky framework could support multiple IR types in the future,
        // conceptually similar to MLIR but in much simpler framework.
        enum class IRType
        {
            StinkyTofu,
        };

        IRBase(IRType type)
            : irType(type)
        {
        }

        virtual ~IRBase() = default;

        virtual void dump(std::ostream& out) const = 0;

        void dump();

        IRType getType() const
        {
            return irType;
        }

    private:
        const IRType irType;
    };

    struct IRListProperties
    {
        bool                          containsLoop = false;
        IntrusiveListIterator<IRBase> loopBegin;
        IntrusiveListIterator<IRBase> loopEnd;
    };

    using IRList = IntrusiveList<IRBase>;

    // Base IR Builder.
    //
    // Each IR type should have its own derived IRBuilder. All creation and
    // deletion of derived IRBase should be handled through the derived
    // IRBuilder.
    //
    // Note that PassContext owns the IRList, PassContext will delete all
    // IRs when PassContext is destructed.
    class IRBuilder
    {
    protected:
        IRList* irlist = nullptr;

    public:
        using ID = const void*;

        IRBuilder(IRList& irlist)
            : irlist(&irlist)
        {
        }

        virtual ~IRBuilder() = default;
    };

    // Base class for all passes.
    //
    // A pass operates on an IRList and performs either analysis or
    // transformation.
    class Pass
    {
    public:
        using ID     = const void*;
        using PassID = ID;

    public:
        virtual ~Pass() = default;

        virtual ID          getPassID() const = 0;
        virtual const char* getName() const   = 0;

        virtual void run(IRList& irlist, PassContext& passCtx) = 0;
    };

    // Base class for all analysis passes.
    //
    // An analysis pass computes information from the IR.
    // Its results can be queried by other passes through
    // the AnalysisManager.
    class AnalysisPass : public Pass
    {
    };

    // AnalysisManager manages analysis passes and their results.
    //
    // It runs analysis passes on demand, and each pass caches its own result.
    // If a result is invalidated by another pass, the manager will rerun the
    // analysis pass when the result is requested.
    class AnalysisManager
    {
    public:
        enum class PassState
        {
            NotRun,
            Completed,

            // The result of the pass is invalidated by another pass.
            // If the other pass wants to use the result, PassManager needs to
            // rerun the pass.
            Invalidated,
        };

        ~AnalysisManager();

        void registerAnalysisPass(std::unique_ptr<AnalysisPass> pass)
        {
            // Release ownership, AnalysisManager will manage the lifetime of the pass.
            AnalysisPass* passPtr = pass.release();

            analysisPasses[passPtr->getPassID()] = {passPtr, PassState::NotRun};
        }

        template <class AnalysisPassType, class IRUnitType>
        AnalysisPassType& getResult(IRUnitType& irUnit, PassContext& passCtx)
        {
            Pass::ID pid = AnalysisPassType::ID;
            auto     it  = analysisPasses.find(pid);
            assert(it != analysisPasses.end() && "Analysis pass not registered");

            std::pair<AnalysisPass*, PassState>& entry = it->second;

            PassState&        state = entry.second;
            AnalysisPassType* pass  = static_cast<AnalysisPassType*>(entry.first);
            if(state != PassState::Completed)
            {
                assert(state == PassState::NotRun || state == PassState::Invalidated);
                pass->run(irUnit, passCtx);
                state = PassState::Completed;
            }
            return *pass;
        }

        void invalidate(Pass::ID pid)
        {
            auto it = analysisPasses.find(pid);
            assert(it != analysisPasses.end() && "Analysis pass not registered");
            it->second.second = PassState::Invalidated;
        }

    private:
        // Note that AnalysisManager owns the AnalysisPass pointers and will delete them in
        // destructor.
        std::unordered_map<Pass::ID, std::pair<AnalysisPass*, PassState>> analysisPasses;
    };

    // The PassContext serves as the central state and resource manager for
    // the StinkyTofu pass execution framework.
    //
    // It acts as a shared context that provides passes with access to
    // essential services and data during IR transformation and analysis.
    class PassContext
    {
        AnalysisManager  analysisMgr;
        StinkyKernelInfo kernel;
        StinkyOptInfo    optInfo;
        IRListProperties properties;

        std::unordered_map<IRBuilder::ID, std::unique_ptr<IRBuilder>> irBuilders;

        // Note: Even though StinkyInstruction is currently the only IR
        //       type, there could be more IR types (levels) like MLIR in
        //       the future.
        IRList irlist;

        void cleanup();

    public:
        PassContext() = default;
        ~PassContext()
        {
            cleanup();
        }

        // FIXME: This should be removed, as PassContext should not expose IRList directly.
        //        But if we don't expose it, the derived PassManager classes cannot access it.
        IRList& getIRList()
        {
            return irlist;
        }

        AnalysisManager& getAnalysisManager()
        {
            return analysisMgr;
        }

        void addKernelInfo(StinkyKernelInfo kernelCfg)
        {
            kernel = kernelCfg;
        }

        const StinkyKernelInfo& getKernelInfo() const
        {
            return kernel;
        }

        void setOptInfo(const StinkyOptInfo& opt)
        {
            optInfo = opt;
        }

        const StinkyOptInfo& getOptInfo() const
        {
            return optInfo;
        }

        void setLoopProperties(bool                          containsLoop,
                               IntrusiveListIterator<IRBase> loopBegin,
                               IntrusiveListIterator<IRBase> loopEnd)
        {
            properties.containsLoop = containsLoop;
            properties.loopBegin    = loopBegin;
            properties.loopEnd      = loopEnd;
        }

        const IRListProperties& getProperties() const
        {
            return properties;
        }

        template <class IRBuilderType, class... Args>
        IRBuilderType& getOrCreateIRBuilder(IRList& irlist, Args&&... args)
        {
            IRBuilder::ID id = IRBuilderType::ID;
            auto          it = irBuilders.find(id);

            IRBuilderType* builder = nullptr;
            if(it == irBuilders.end())
            {
                std::unique_ptr<IRBuilderType> builderPtr
                    = std::make_unique<IRBuilderType>(irlist, std::forward<Args>(args)...);

                builder = builderPtr.get();

                irBuilders[id] = std::move(builderPtr);
            }
            else
            {
                builder = static_cast<IRBuilderType*>(it->second.get());
            }
            return *builder;
        }
    };

    bool isDebugOnlyEnabled(const char* TYPE);

#define DEBUG_WITH_TYPE(TYPE, X)     \
    do                               \
    {                                \
        if(isDebugOnlyEnabled(TYPE)) \
        {                            \
            X;                       \
        }                            \
    } while(0)

    // PASS_DEBUG is used by each pass to print its internal debug information.
    // A pass can be enabled for debug output by adding its DEBUG_TYPE name to
    // PassManagerDebugConfig::addDebugOnly
#define PASS_DEBUG(X) DEBUG_WITH_TYPE(DEBUG_TYPE, X)

    // Configuration for PassManager debug output.
    //
    // Users can configure which passes to print the IR before/after running,
    // and where to dump the output (stdout or file).
    //
    // Users can also configure a global debug-only list of passes to print.
    class PassManagerDebugConfig final
    {
        friend class PassManager;

        unsigned printAfterAll : 1;
        unsigned printBeforeAll : 1;

        std::unique_ptr<std::ostream> dumpStreamBefore;
        std::unique_ptr<std::ostream> dumpStreamAfter;

        std::unordered_set<std::string> onlyPrintBefore;
        std::unordered_set<std::string> onlyPrintAfter;

    public:
        PassManagerDebugConfig();
        ~PassManagerDebugConfig();

        void setPrintAfterAll(bool v = true);
        void setPrintBeforeAll(bool v = true);
        void addOnlyPrintBefore(const std::string& passName);
        void addOnlyPrintAfter(const std::string& passName);
        void setDumpToFileInBefore(const std::string& filename);
        void setDumpToFileInAfter(const std::string& filename);

        bool shouldPrintBefore(const std::string& passName) const;
        bool shouldPrintAfter(const std::string& passName) const;

        std::ostream& getOutputStreamInBefore() const;
        std::ostream& getOutputStreamInAfter() const;

    public:
        // Note: The debug only functions will use internal global static
        // variables, that's why they are static.
        static void addDebugOnly(const std::string& passName);
        static void clearDebugOnly();
    };

    // PassManager manages a list of passes to run on a module.
    // It also manages the analysis passes and their results through PassContext.
    //
    // Note: The module is using physical registers, so it is no longer in SSA form.
    //
    // Note: Even though StinkyInstruction is currently the only IR
    //       type, there could be more IR types (levels) like MLIR in
    //       the future.
    class PassManager
    {
    public:
        void run();

        // Add a pass to the list of passes to run.
        // The passes will be run in the order they are added.
        //
        // By default, the PassManager does not have any pass to run.
        void addPass(std::unique_ptr<Pass> pass)
        {
            passes.push_back(std::move(pass));
        }

        void registerAnalysisPass(std::unique_ptr<AnalysisPass> pass)
        {
            passCtx.getAnalysisManager().registerAnalysisPass(std::move(pass));
        }

        PassManager()  = default;
        ~PassManager() = default;

    public:
        void setDebugConfig(std::unique_ptr<PassManagerDebugConfig> cfg);
        void setKernelConfig(std::array<int, 3> arch,
                             uint32_t           ta0,
                             uint32_t           tb0,
                             uint32_t           tm0,
                             uint32_t           nGRA,
                             uint32_t           nGRB,
                             uint32_t           nGRM,
                             uint32_t           wavefrontSz,
                             uint32_t           numWaves);
        void setOptConfig(const StinkyOptInfo& opt);

    protected:
        PassContext passCtx;

        // List of passes to run.
        //
        // Users can add passes to this list through 'addPass' method.
        // The passes will be run in the order they are added.
        std::vector<std::unique_ptr<Pass>> passes;

        std::unique_ptr<PassManagerDebugConfig> dbgCfg;
    };
}

#include "stinkypasses.hpp"
