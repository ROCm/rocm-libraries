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
#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "adt/IntrusiveList.hpp"

namespace stinkytofu
{
    // Forward declarations
    class PassContext;
    enum class GfxArchID : uint32_t;

    // Error codes for StinkyIRConverter operations
    enum class StinkyErrorCode : int
    {
        SUCCESS       = 0,
        PASSCTX_EMPTY = 1,
    };

    /// GEMM-specific tile configuration
    /// This configuration is specific to GEMM kernels and their tiling strategy
    /// Note: WavefrontSize is NOT included here as it's derived from architecture,
    ///       not a user-configurable parameter. Use getWaveFrontSize(arch) to query it.
    struct GemmTileConfig
    {
        std::array<int, 3> arch{0, 0, 0}; ///< GPU architecture [gfx, major, minor]
        uint32_t           TileA0; ///< Tile size for A dimension 0
        uint32_t           TileB0; ///< Tile size for B dimension 0
        uint32_t           TileM0; ///< Tile size for M dimension 0
        uint32_t           NumGRA; ///< Number of global read A
        uint32_t           NumGRB; ///< Number of global read B
        uint32_t           NumGRM; ///< Number of global read M
        uint32_t           NumWaves; ///< Number of wavefronts
    };

    /// Pass-specific feature configuration
    /// Categorizes optimization behaviors into semantics, properties, and features
    struct PassFeatureConfig
    {
        /// Barrier semantics and unrolling behavior
        /// These are code structure PROPERTIES (not optional features)
        struct BarrierConfig
        {
            bool unrollMovableBarrier = false; ///< Whether GEMM barriers can be moved during unroll
        };

        /// Loop structure and unrolling properties
        /// These are code structure PROPERTIES (not optional features)
        struct LoopConfig
        {
            bool unrollGemm = false; ///< Whether GEMM loops are unrolled
        };

        /// DAG scheduler feature switches
        /// These are OPTIONAL FEATURES that can be enabled/disabled
        struct DagFeatures
        {
            bool distributeGlobalRead = false; ///< Enable global read distribution optimization
        };

        BarrierConfig barrierConfig;
        LoopConfig    loopConfig;
        DagFeatures   dagFeatures;
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

    using IRList = IntrusiveList<IRBase>;

    // Forward declarations for BasicBlock and Function
    class Function;

    // BasicBlock holds a list of IR and CFG edges.
    //
    // BasicBlocks are organized in an intrusive list within a Function.
    // Each BasicBlock contains an IRList and maintains
    // predecessor/successor relationships for control flow.
    class BasicBlock : public IntrusiveListNode<BasicBlock>
    {
    private:
        Function*                parent = nullptr;
        std::string              label; // Optional label for the block
        IRList                   ir;
        std::vector<BasicBlock*> predecessors;
        std::vector<BasicBlock*> successors;

    public:
        explicit BasicBlock(Function* parent, const std::string& label = "")
            : parent(parent)
            , label(label)
        {
        }

        Function* getParent()
        {
            return parent;
        }
        const Function* getParent() const
        {
            return parent;
        }
        void setParent(Function* p)
        {
            parent = p;
        }

        const std::string& getLabel() const
        {
            return label;
        }
        void setLabel(const std::string& l)
        {
            label = l;
        }

        // Access to IR
        IRList& getIR()
        {
            return ir;
        }
        const IRList& getIR() const
        {
            return ir;
        }

        // CFG navigation
        void addSuccessor(BasicBlock* bb)
        {
            successors.push_back(bb);
        }

        void addPredecessor(BasicBlock* bb)
        {
            predecessors.push_back(bb);
        }

        void removeSuccessor(BasicBlock* bb)
        {
            successors.erase(std::remove(successors.begin(), successors.end(), bb),
                             successors.end());
        }

        void removePredecessor(BasicBlock* bb)
        {
            predecessors.erase(std::remove(predecessors.begin(), predecessors.end(), bb),
                               predecessors.end());
        }

        const std::vector<BasicBlock*>& getSuccessors() const
        {
            return successors;
        }
        const std::vector<BasicBlock*>& getPredecessors() const
        {
            return predecessors;
        }

        std::vector<BasicBlock*>& getSuccessors()
        {
            return successors;
        }
        std::vector<BasicBlock*>& getPredecessors()
        {
            return predecessors;
        }

        // Utilities
        IRBase* getTerminator()
        {
            if(ir.empty())
                return nullptr;
            return &ir.back();
        }

        const IRBase* getTerminator() const
        {
            if(ir.empty())
                return nullptr;
            return &ir.back();
        }

        bool empty() const
        {
            return ir.empty();
        }
        size_t size() const
        {
            size_t count = 0;
            for(auto it = ir.begin(); it != ir.end(); ++it)
                ++count;
            return count;
        }

        // Clear all IR in this BasicBlock
        // Deletes all IRBase objects and clears the IR list
        void clearAllIR()
        {
            while(!ir.empty())
            {
                IRBase* irNode = ir.begin().getNodePtr();
                ir.erase(ir.begin());
                delete irNode;
            }
        }

        void dump(std::ostream& out) const;
    };

    using BasicBlockList = IntrusiveList<BasicBlock>;

    // Function holds a list of BasicBlocks.
    //
    // This represents a function/kernel in the StinkyTofu IR.
    // BasicBlocks are organized as an intrusive list and can be
    // traversed in program order.
    class Function
    {
    private:
        std::string    name;
        BasicBlockList basicBlocks;
        BasicBlock*    entryBlock = nullptr;
        GemmTileConfig gemmConfig;

    public:
        explicit Function(const std::string& name = "")
            : name(name)
        {
        }

        ~Function()
        {
            // Clean up all basic blocks
            while(!basicBlocks.empty())
            {
                BasicBlock* bb = &basicBlocks.front();
                basicBlocks.remove(bb);
                delete bb;
            }
        }

        const std::string& getName() const
        {
            return name;
        }
        void setName(const std::string& n)
        {
            name = n;
        }

        // BasicBlock management
        BasicBlock* createBasicBlock(const std::string& label = "")
        {
            BasicBlock* bb = new BasicBlock(this, label);
            basicBlocks.push_back(bb);
            if(!entryBlock)
                entryBlock = bb;
            return bb;
        }

        BasicBlock* createBasicBlockBefore(BasicBlock* before, const std::string& label = "")
        {
            BasicBlock* bb = new BasicBlock(this, label);
            basicBlocks.insert(BasicBlockList::iterator(before), bb);
            return bb;
        }

        void removeBasicBlock(BasicBlock* bb)
        {
            if(bb == entryBlock)
                entryBlock = nullptr;
            basicBlocks.remove(bb);
            delete bb;
        }

        // Delete all BasicBlocks and their IR
        // Clears all IR in each BasicBlock, then deletes all BasicBlocks
        void deleteAllBasicBlocks()
        {
            while(!basicBlocks.empty())
            {
                BasicBlock* bb = &basicBlocks.front();
                bb->clearAllIR();
                basicBlocks.remove(bb);
                delete bb;
            }
            entryBlock = nullptr;
        }

        BasicBlockList& getBasicBlocks()
        {
            return basicBlocks;
        }
        const BasicBlockList& getBasicBlocks() const
        {
            return basicBlocks;
        }

        void setEntryBlock(BasicBlock* bb)
        {
            entryBlock = bb;
        }
        BasicBlock* getEntryBlock()
        {
            return entryBlock;
        }
        const BasicBlock* getEntryBlock() const
        {
            return entryBlock;
        }

        // GEMM tile configuration
        void setGemmTileConfig(const GemmTileConfig& config)
        {
            gemmConfig = config;
        }
        const GemmTileConfig& getGemmTileConfig() const
        {
            return gemmConfig;
        }

        // Iteration over basic blocks
        auto begin()
        {
            return basicBlocks.begin();
        }
        auto end()
        {
            return basicBlocks.end();
        }
        auto begin() const
        {
            return basicBlocks.begin();
        }
        auto end() const
        {
            return basicBlocks.end();
        }

        bool empty() const
        {
            return basicBlocks.empty();
        }
        size_t size() const
        {
            size_t count = 0;
            for(auto it = basicBlocks.begin(); it != basicBlocks.end(); ++it)
                ++count;
            return count;
        }

        void clear()
        {
            while(!basicBlocks.empty())
            {
                BasicBlock* bb = &basicBlocks.front();
                basicBlocks.remove(bb);
                delete bb;
            }
        }

        void dump(std::ostream& out) const;
    };

    // Base IR Builder.
    //
    // Each IR type should have its own derived IRBuilder. All creation and
    // deletion of derived IRBase should be handled through the derived
    // IRBuilder.
    //
    // Note that PassContext owns the Function (which owns BasicBlocks and IRLists),
    // PassContext will delete all IRs when PassContext is destructed.
    /// IRBuilder base class for constructing IR.
    ///
    /// Follows LLVM/MLIR's IRBuilder pattern as a lightweight value object.
    /// Builders are typically created on the stack and passed an insertion
    /// point (IRList) where new instructions should be inserted.
    ///
    /// Example usage (LLVM/MLIR style):
    ///   auto builder = passCtx.getIRBuilder<StinkyInstIRBuilder>(bb.getIR(), arch);
    ///   builder.createStinkyInstBefore(...);
    ///
    /// Derived builders (e.g., StinkyInstIRBuilder) provide type-specific
    /// construction methods.
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

        /// Set the insertion point to a different IRList.
        /// This allows the builder to be reused across different BasicBlocks.
        void setInsertionPoint(IRList& newIRList)
        {
            irlist = &newIRList;
        }

        /// Get the current insertion point.
        IRList* getInsertionPoint() const
        {
            return irlist;
        }
    };

    //----------------------------------------------------------------------
    // BasicBlock Filter Support
    //----------------------------------------------------------------------

    // Forward declaration
    class BasicBlock;

    // BasicBlock filter predicate type
    using BasicBlockFilter = std::function<bool(const BasicBlock&)>;

    // Filter builder for easy construction of BasicBlock filters
    class BasicBlockFilterBuilder
    {
    public:
        // Filter by label prefix
        static BasicBlockFilter byLabelPrefix(const std::string& prefix)
        {
            return [prefix](const BasicBlock& bb) { return bb.getLabel().rfind(prefix, 0) == 0; };
        }

        // Filter by exact label names
        static BasicBlockFilter byLabels(const std::set<std::string>& labels)
        {
            return [labels](const BasicBlock& bb) { return labels.count(bb.getLabel()) > 0; };
        }

        // Filter by custom predicate
        static BasicBlockFilter byPredicate(std::function<bool(const BasicBlock&)> pred)
        {
            return pred;
        }

        // Combine filters with AND logic
        static BasicBlockFilter combine(const BasicBlockFilter& f1, const BasicBlockFilter& f2)
        {
            return [f1, f2](const BasicBlock& bb) { return f1(bb) && f2(bb); };
        }

        // Exclude certain BasicBlocks (NOT logic)
        static BasicBlockFilter exclude(const BasicBlockFilter& filter)
        {
            return [filter](const BasicBlock& bb) { return !filter(bb); };
        }

        // Process all BasicBlocks (default)
        static BasicBlockFilter all()
        {
            return [](const BasicBlock&) { return true; };
        }
    };

    //----------------------------------------------------------------------
    // Pass Infrastructure
    //----------------------------------------------------------------------

    // Base class for all passes.
    //
    // A pass operates on a Function (which contains BasicBlocks with IRLists)
    // and performs either analysis or transformation.
    class Pass
    {
    public:
        using ID     = const void*;
        using PassID = ID;

    public:
        virtual ~Pass() = default;

        virtual ID          getPassID() const = 0;
        virtual const char* getName() const   = 0;

        virtual void run(Function& func, PassContext& passCtx) = 0;
    };

    // Base class for all analysis passes.
    //
    // An analysis pass computes information from the IR.
    // Its results can be queried by other passes through
    // the AnalysisManager.
    class AnalysisPass : public Pass
    {
    };

    //----------------------------------------------------------------------
    // High-Level IR Pass Infrastructure
    //----------------------------------------------------------------------

    // Forward declaration for IR instruction type
    class IRInstruction;
    class StinkyInstruction;

    /**
     * @brief Base class for all high-level IR passes
     *
     * IRInstPass operates on IRInstruction (high-level IR) as opposed to
     * Pass which operates on StinkyInstruction (assembly IR).
     *
     * Passes can be composed into pipelines using IRInstPassManager for
     * optimization and lowering sequences.
     *
     * Naming convention:
     * - IRInstPass - operates on IRInstruction (high-level IR)
     * - Pass       - operates on StinkyInstruction (assembly IR)
     */
    class IRInstPass
    {
    public:
        virtual ~IRInstPass() = default;

        /**
         * @brief Get the name of this pass (for debugging/logging)
         */
        virtual const char* getName() const = 0;

        /**
         * @brief Check if this pass produces assembly instructions
         *
         * Most passes produce IRInstruction*, but the final lowering pass
         * (ToStinkyAsmPass) produces StinkyInstruction*.
         */
        virtual bool producesAsm() const
        {
            return false;
        }
    };

    /**
     * @brief Pass that transforms IRInstructions to other IRInstructions
     *
     * Example: CompositeInstructionLoweringPass expands VAddPKF32 -> VAddF32
     */
    class IRInstTransformPass : public IRInstPass
    {
    public:
        /**
         * @brief Transform an IR instruction
         * @param irInst Input IR instruction
         * @param arch Target architecture for capability queries
         * @return Vector of output IR instructions (may be 0, 1, or many)
         */
        virtual std::vector<IRInstruction*> transform(IRInstruction* irInst, GfxArchID arch) = 0;

        bool producesAsm() const override
        {
            return false;
        }
    };

    /**
     * @brief Pass that converts IRInstructions to assembly instructions
     *
     * This is the final lowering pass in the pipeline.
     * Example: ToStinkyAsmPass converts VAddF32 (IRInstruction) -> v_add_f32 (StinkyInstruction)
     */
    class IRInstToAsmPass : public IRInstPass
    {
    public:
        /**
         * @brief Lower an IR instruction to assembly
         * @param irInst Input IR instruction
         * @param arch Target architecture for mnemonic mapping
         * @return Vector of output assembly instructions (typically 1)
         */
        virtual std::vector<StinkyInstruction*> lower(IRInstruction* irInst, GfxArchID arch) = 0;

        bool producesAsm() const override
        {
            return true;
        }
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
        AnalysisManager   analysisMgr;
        GemmTileConfig    gemmConfig;
        PassFeatureConfig passConfig;
        uint32_t          wavefrontSize = 0; ///< Computed from gemmConfig.arch

        // Function holds BasicBlocks which contain IRLists.
        // This is the primary IR container for the pass infrastructure.
        std::unique_ptr<Function> function;

        // Global BasicBlock filter applied to all StinkyInstPass instances.
        // By default, all BasicBlocks are processed.
        BasicBlockFilter globalBBFilter;

        void cleanup();

    public:
        PassContext()
            : function(std::make_unique<Function>("kernel"))
            , globalBBFilter(BasicBlockFilterBuilder::all())
        {
        }

        ~PassContext()
        {
            cleanup();
        }

        // Access to the Function
        Function& getFunction()
        {
            return *function;
        }

        const Function& getFunction() const
        {
            return *function;
        }

        AnalysisManager& getAnalysisManager()
        {
            return analysisMgr;
        }

        // ========== New API (preferred) ==========

        void setGemmTileConfig(const GemmTileConfig& config);

        const GemmTileConfig& getGemmTileConfig() const
        {
            return gemmConfig;
        }

        /// Get wavefront size (derived from architecture, not user-configurable)
        uint32_t getWavefrontSize() const
        {
            return wavefrontSize;
        }

        void setPassFeatureConfig(const PassFeatureConfig& config)
        {
            passConfig = config;
        }

        const PassFeatureConfig& getPassFeatureConfig() const
        {
            return passConfig;
        }

        /// Set global BasicBlock filter for all StinkyInstPass instances.
        /// This filter determines which BasicBlocks should be processed by passes.
        void setBasicBlockFilter(BasicBlockFilter filter)
        {
            globalBBFilter = std::move(filter);
        }

        /// Check if a BasicBlock should be processed according to the global filter.
        bool shouldProcessBasicBlock(const BasicBlock& bb) const
        {
            return globalBBFilter(bb);
        }

        /// Get an IRBuilder of the specified type.
        ///
        /// The builder is returned by value and can be used as a local variable.
        /// No heap allocation or lifetime management is needed.
        ///
        /// Example:
        ///   for (BasicBlock& bb : func) {
        ///     auto builder = passCtx.getIRBuilder<StinkyInstIRBuilder>(
        ///                         bb.getIR(), arch);
        ///     // Use builder to create instructions in bb
        ///     builder.createStinkyInstBefore(...);
        ///   }
        template <class IRBuilderType, class... Args>
        IRBuilderType getIRBuilder(IRList& irlist, Args&&... args)
        {
            // Construct and return builder by value (LLVM/MLIR style)
            return IRBuilderType(irlist, std::forward<Args>(args)...);
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

        // Set GEMM tile configuration (wavefront size automatically determined from architecture)
        void setGemmTileConfig(const GemmTileConfig& config);

        // Deprecated: Use setGemmTileConfig instead
        void setKernelConfig(std::array<int, 3> arch,
                             uint32_t           ta0,
                             uint32_t           tb0,
                             uint32_t           tm0,
                             uint32_t           nGRA,
                             uint32_t           nGRB,
                             uint32_t           nGRM,
                             uint32_t           numWaves);

        // Set pass feature configuration
        void setPassFeatureConfig(const PassFeatureConfig& config);

        void setBasicBlockFilter(BasicBlockFilter filter)
        {
            passCtx.setBasicBlockFilter(filter);
        }

        // Set the Function to operate on (transfers ownership from external Function to PassContext)
        void setFunction(Function& externalFunc);

        // Get access to the PassContext (for advanced usage)
        PassContext& getPassContext()
        {
            return passCtx;
        }

        const PassContext& getPassContext() const
        {
            return passCtx;
        }

    protected:
        PassContext passCtx;

        // List of passes to run.
        //
        // Users can add passes to this list through 'addPass' method.
        // The passes will be run in the order they are added.
        std::vector<std::unique_ptr<Pass>> passes;

        std::unique_ptr<PassManagerDebugConfig> dbgCfg;
    };

    /**
     * StinkyIRConverter - A utility class for converting raw instruction strings to a Function.
     *
     * This class provides a simple interface to parse MLIR-style StinkyTofu IR text and
     * convert it to a Function that can be used with the StinkyTofu pass infrastructure.
     *
     * Example usage:
     * ```cpp
     * std::string raw_stinky_inst = R"(
     *   v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
     *   v[4:7] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
     * )";
     *
     * StinkyIRConverter converter;
     * Function* func = converter.convertToFunction(raw_stinky_inst);
     *
     * // Use the function with passes...
     * // Access the entry BasicBlock and its IR:
     * BasicBlock* entryBB = func->getEntryBlock();
     * IRList& irlist = entryBB->getIR();
     *
     * // Don't forget to clean up when done
     * converter.cleanup();
     * ```
     */
    class StinkyIRConverter
    {
    public:
        /**
         * Constructor with default architecture (gfx942).
         */
        StinkyIRConverter();

        /**
         * Constructor with specified architecture.
         *
         * @param targetArch The target GPU architecture (e.g., {9, 4, 2} for gfx942)
         */
        StinkyIRConverter(const std::array<int, 3>& targetArch);

        /**
         * Convert a raw instruction string to a Function.
         *
         * The string should be in MLIR-style StinkyTofu IR format:
         * destRegs = "st.mnemonic"(srcRegs) { attributes }
         *
         * @param rawInstructions The raw instruction string to parse
         * @return Pointer to the Function owned by the internal PassContext, or nullptr if conversion fails
         */
        Function* convertToFunction(const std::string& rawInstructions);

        /**
         * Static helper to populate a Function from an IR text string.
         * This is the core conversion logic shared by both StinkyIRConverter and stinkytofu-opt.
         *
         * @param irText The IR source text to parse
         * @param func The Function to populate with a BasicBlock containing instructions
         * @param passCtx The PassContext for resource management
         * @param arch The target GPU architecture
         * @return StinkyErrorCode indicating success or failure
         */
        static StinkyErrorCode populateFunctionFromString(const std::string& irText,
                                                          Function&          func,
                                                          PassContext&       passCtx,
                                                          GfxArchID          arch);

        /**
         * Get the PassContext associated with the last conversion.
         * This is useful if you need to access the context for running passes.
         *
         * @return Pointer to the PassContext, or nullptr if not available
         */
        PassContext* getPassContext();

        /**
         * Cleanup resources. Call this when done with the IRList.
         * The PassContext destructor will handle cleanup of all IR objects.
         */
        void cleanup();

        /**
         * Destructor - automatically cleans up resources.
         */
        ~StinkyIRConverter();

    private:
        std::array<int, 3>                       arch;
        std::unique_ptr<stinkytofu::PassContext> passCtx;
    };
}

#include "stinkypasses.hpp"
