/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

/**
 * @file MemoryTracer.cpp
 * @author rocRoller Developers
 * @brief Memory tracer for the rocRoller kernel graph.
 *
 * This file implements a memory tracer that simulates memory accesses
 * in a kernel graph.
 *
 * The general idea is:
 *
 * 1. Instantiate a `MemoryTracer()` object with the kernel graph.
 *
 * 2. Call `trace()` to walk the control graph and generate a list of
 *    memory events.  Each memory event roughly corresponds to a
 *    memory instruction that the code-generator will emit.
 *
 *    This step is done once.
 *
 * 3. For each memory effect that you want to simulate, instantiate a
 *    "model".
 *
 *    For example, the `LDSBankModel()` focuses on LDS read/writes,
 *    and tries to predict LDS bank conflicts.
 *
 *    a. Call the tracer's `simulateLaunch()` and provide your model.
 *
 *    b. The `simulateLaunch()` method will "blow up" all memory
 *       events by evaluating the indexing expression for a collection
 *       of `Workgroup` and `Workitem` values into a large collection
 *       of `MemoryEventSimulated` objects.
 *
 *    c. Each of these simulated memory events will be passed to your
 *       model through the `simulate()` method.
 */

#include <map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/TopoVisitor.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    namespace MemoryTracer
    {
        namespace Expression = rocRoller::Expression;
        using ExpressionPtr  = Expression::ExpressionPtr;

        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        using namespace CoordinateGraph;
        using namespace ControlGraph;

        enum Direction
        {
            GlobalLoad,
            GlobalStore,
            LDSLoad,
            LDSStore
        };

        /**
         * @brief Memory event expression.
         *
         * This structure roughly corresponds to memory instruction
         * that the code-generator will emit.
         */
        struct MemoryEventExpression
        {
            int           operationTag; //< Operation tag
            int           sourceTag; //< Source coordinate tag
            int           destinationTag; //< Destination coordinate tag
            Direction     direction; //< Memory access type
            ExpressionPtr index; //< Index expression
            uint          instructionNumber; //< Instruction number
            uint          bytesRequested; //< Number of bytes requested
        };

        /**
         * @brief Memory event simulated.
         *
         * This is a "blown up" version of `MemoryEventExpression`.
         *
         * Note that each `MemoryEventExpression` has an index
         * expression that may contain `Workgroup` and/or `Workitem`
         * coordinates.
         *
         * The `MemoryTracer` will evaluate the index expression in
         * `MemoryEventSimulated` for a collection of `Workgroup` and
         * `Workitem` values and create a "blown up" version of the
         * memory event that contains the actual byte offset.
         */
        struct MemoryEventSimulated
        {
            int       operationTag; //< Operation tag
            int       sourceTag; //< Source coordinate tag
            int       destinationTag; //< Destination coordinate tag
            uint      instructionNumber; //< Instruction number
            Direction direction; //< Memory access type: GlobalLoad, GlobalStore, LDSLoad, LDSStore
            uint      byteOffset; //< Buffer offset in bytes
            uint      bytesRequested; //< Number of bytes requested
            uint      workGroup; //< Workgroup index
            uint      workItem; //<Workitem index

            // XXX Consider adding SMEM vs VMEM, ie, if VMEM, this has a Workitem dependency
            //
            // If VMEM, possibly remove workItem and just keep a stride?
        };

        std::string toString(Direction const& direction)
        {
            switch(direction)
            {
            case Direction::GlobalLoad:
                return "GlobalLoad";
            case Direction::GlobalStore:
                return "GlobalStore";
            case Direction::LDSLoad:
                return "LDSLoad";
            case Direction::LDSStore:
                return "LDSStore";
            default:
                return "UnknownDirection";
            }
        }

        std::string toString(MemoryEventSimulated const& event)
        {
            return fmt::format(
                "Event(tag: {}, direction: {}, bufferOffset: {}, bytesRequested: {}, "
                "workGroup: {}, workItem: {})",
                event.operationTag,
                toString(event.direction),
                event.byteOffset,
                event.bytesRequested,
                event.workGroup,
                event.workItem);
        }

        /**
         * LDS bank model
         */
        struct LDSBankModel
        {
            struct LDSBankAccess
            {
                int       operationTag;
                int       ldsTag;
                Direction direction;
                uint      workitem;
                uint      bankIndex;
            };

            struct Summary
            {
                struct Banks
                {
                    uint   bankIndex;
                    size_t workitemsAccessed;
                    bool   imbalanced;
                };
                struct Access
                {
                    int                           tag;
                    uint                          instruction;
                    int                           ldsTag;
                    std::vector<Banks>            accessedBanks;
                    std::vector<std::vector<int>> banksToWorkitems;
                };

                std::vector<Access>     accesses;
                std::unordered_set<int> imbalancedTags;

                std::string toString()
                {
                    std::stringstream ss;
                    for(auto const& [tag, instruction, ldsTag, accessedBanks, banksToWorkitems] :
                        this->accesses)
                    {
                        ss << fmt::format("Operation tag {} instruction {} accesses LDS {}:\n",
                                          tag,
                                          instruction,
                                          ldsTag);
                        for(auto const& [bankIndex, workitemsAccessed, imbalanced] : accessedBanks)
                        {
                            ss << fmt::format("  Bank {}: {} workitems {}\n",
                                              bankIndex,
                                              workitemsAccessed,
                                              imbalanced ? "(imbalanced)" : "");
                        }
                        for(size_t bankIndex = 0; bankIndex < banksToWorkitems.size(); ++bankIndex)
                        {
                            ss << fmt::format("  Bank {:2d}: ", bankIndex);
                            for(auto workitem : banksToWorkitems[bankIndex])
                            {
                                ss << fmt::format("{:2d} ", workitem);
                            }
                            ss << '\n';
                        }
                        return ss.str();
                    }
                    ss << fmt::format("  Imbalanced tags: {}\n", this->imbalancedTags);
                    return ss.str();
                }
            };

            /**
             * @brief Construct a new LDSBankModel object.
             *
             * @param sizeInBytes Size of the LDS in bytes.
             * @param numBanks Number of banks in the LDS.
             */
            LDSBankModel(uint entryWidthInBytes, uint numBanks, uint numEntriesPerBank)
                : m_entryWidthInBytes(entryWidthInBytes)
                , m_numBanks(numBanks)
                , m_numEntriesPerBank(numEntriesPerBank)
            {
            }

            bool filter(MemoryEventExpression event)
            {
                return event.direction == Direction::LDSLoad
                       || event.direction == Direction::LDSStore;
            }

            void simulate(MemoryEventSimulated event)
            {
                if(event.operationTag == 4722)
                {
                    Log::info("LDSBankModel::simulate({})", toString(event));
                }

                for(int i = 0; i < event.bytesRequested; i += m_entryWidthInBytes)
                {
                    auto ldsAddressInBytes = event.byteOffset + i;
                    auto bankIndex         = (ldsAddressInBytes / m_entryWidthInBytes) % m_numBanks;

                    auto ldsTag = event.direction == Direction::LDSStore ? event.destinationTag
                                                                         : event.sourceTag;

                    // XXX When we break this down by instruction, need to add to LDSBankAccess struct
                    m_bankAccesses[{event.operationTag, event.instructionNumber}].push_back(
                        LDSBankAccess{event.operationTag,
                                      ldsTag,
                                      event.direction,
                                      event.workItem,
                                      bankIndex});
                }
            }

            std::string summary()
            {
                std::stringstream ss;
                ss << "LDS Bank Model: " << m_entryWidthInBytes * m_numEntriesPerBank * m_numBanks
                   << " bytes, " << m_entryWidthInBytes << "byte bank width, " << m_numBanks
                   << " banks" << std::endl;

                Summary summary;

                // For each operation tag and instruction...
                for(auto const& [key, accesses] : m_bankAccesses)
                {
                    auto [tag, instruction] = key;

                    auto ldsTag = accesses[0].ldsTag;

                    //
                    // Do different workitems touch the same bank?
                    //
                    std::map<uint, std::unordered_set<uint>> bankWorkitems;
                    for(auto access : accesses)
                    {
                        bankWorkitems[access.bankIndex].insert(access.workitem);
                    }

                    uint minWorkitemsPerBank = 0;
                    for(int bankIndex = 0; bankIndex < m_numBanks; ++bankIndex)
                    {
                        if(bankWorkitems.contains(bankIndex))
                            minWorkitemsPerBank
                                = std::min(minWorkitemsPerBank,
                                           static_cast<uint>(bankWorkitems[bankIndex].size()));
                    }

                    bool anyImbalance = false;
                    for(auto const& [bankIndex, workitems] : bankWorkitems)
                        anyImbalance |= workitems.size() > minWorkitemsPerBank;

                    if(anyImbalance)
                        summary.imbalancedTags.insert(tag);

                    const auto workitemsInfo = [&]() {
                        decltype(Summary::Access::accessedBanks) workitemsInfo;
                        for(auto const& [bankIndex, workitems] : bankWorkitems)
                        {
                            auto imbalanced = workitems.size() > minWorkitemsPerBank;
                            workitemsInfo.emplace_back(bankIndex, workitems.size(), imbalanced);
                        }
                        return workitemsInfo;
                    }();

                    // For each bank, find the workitems that accessed it
                    std::vector<std::vector<int>> banksToWorkitems;
                    const auto                    maxWorkitems = 256;
                    for(int bankIndex = 0; bankIndex < m_numBanks; ++bankIndex)
                    {
                        if(bankWorkitems.contains(bankIndex))
                        {
                            banksToWorkitems.emplace_back([&]() {
                                std::vector<int> workitems;
                                for(int workitem = 0; workitem < maxWorkitems; ++workitem)
                                {
                                    if(bankWorkitems[bankIndex].contains(workitem))
                                    {
                                        workitems.emplace_back(workitem);
                                    }
                                }
                                return workitems;
                            }());
                        }
                        else
                        {
                            banksToWorkitems.emplace_back();
                        }
                    }

                    summary.accesses.emplace_back(
                        tag, instruction, ldsTag, workitemsInfo, banksToWorkitems);
                }

                return summary.toString();
            }

        private:
            uint m_entryWidthInBytes;
            uint m_numBanks;
            uint m_numEntriesPerBank;

            std::map<std::pair<int, uint>, std::vector<LDSBankAccess>> m_bankAccesses;
        };

        /**
         * @brief Memory tracer for the kernel graph.
         *
         * This class walks the control graph and builds a list of
         * MemoryEventExpression objects.  These objects represent
         * instructions that the code-generator will emit.
	 *
	 * Note that the base LDS allocation address is assumed to be
	 * zero.  If you are comparing the bank indexes reported here
	 * vs those computed by, eg, inspecting register values, you
	 * may see a discrepancy.  However, the number of bank
	 * conflicts should be the same.
         */
        //struct MemoryTracer : TopoControlGraphVisitor<MemoryTracer>
        struct MemoryTracer
        {
            // MemoryTracer(KernelGraph const&      graph,
            //              KernelInvocation const& invocation,
            //              KernelArguments const&  arguments)
            //     : TopoControlGraphVisitor<MemoryTracer>::TopoControlGraphVisitor(graph)
            //     , m_graph(graph)
            //     , m_invocation(invocation)
            //     , m_arguments(arguments)
            MemoryTracer(KernelGraph const&      graph,
                         KernelInvocation const& invocation,
                         KernelArguments const&  arguments)
                : m_graph(graph)
                , m_invocation(invocation)
                , m_arguments(arguments)
            {
                for(int i = 0; i < 3; ++i)
                {
                    m_workgroupOffset[i] = m_arguments.size();
                    auto wg_name         = concatenate("WG", i);
                    auto wg_carg         = CommandArgument(nullptr,
                                                   DataType::UInt32,
                                                   m_workgroupOffset[i],
                                                   DataDirection::ReadOnly,
                                                   wg_name);
                    auto wg              = std::make_shared<CommandArgument>(wg_carg);
                    m_arguments.appendUnbound<uint>(wg_name);

                    m_workitemOffset[i] = m_arguments.size();
                    auto wi_name        = concatenate("WI", i);
                    auto wi_carg        = CommandArgument(nullptr,
                                                   DataType::UInt32,
                                                   m_workitemOffset[i],
                                                   DataDirection::ReadOnly,
                                                   wi_name);
                    auto wi             = std::make_shared<CommandArgument>(wi_carg);
                    m_arguments.appendUnbound<uint>(wi_name);

                    m_kernelWorkgroupIndexes[i] = std::make_shared<Expression::Expression>(wg);
                    m_kernelWorkitemIndexes[i]  = std::make_shared<Expression::Expression>(wi);
                }
            }

            /**
             * @brief Walk the control graph and ...
             */
            void trace()
            {
                Log::debug("MemoryTracer::trace()");
                auto coordinateGraph
                    = std::make_shared<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph>(
                        m_graph.coordinates);

                auto coords = Transformer(coordinateGraph.get());
                coords.fillExecutionCoordinates(
                    nullptr, m_kernelWorkgroupIndexes, m_kernelWorkitemIndexes);

                auto candidates = m_graph.control.roots().to<std::set>();
                generate(candidates, coords);
            }

            /*
             * Tracing...
             */

            bool hasGeneratedInputs(int const& tag)
            {
                auto inputs = m_graph.control.getInputNodeIndices<Sequence>(tag);
                for(auto const& input : inputs)
                {
                    if(m_completedControlNodes.find(input) == m_completedControlNodes.end())
                        return false;
                }
                return true;
            }

            // TODO Use Scott's helper, but need Transformer coords!
            void generate(std::set<int> candidates, Transformer coords)
            {
                while(!candidates.empty())
                {
                    std::set<int> nodes;

                    // Find all candidate nodes whose inputs have been satisfied
                    for(auto const& tag : candidates)
                        if(hasGeneratedInputs(tag))
                            nodes.insert(tag);

                    // If there are none, we have a problem.
                    AssertFatal(!nodes.empty(),
                                "Invalid control graph!",
                                ShowValue(m_graph.control),
                                ShowValue(candidates));

                    // Visit all the nodes we found.
                    for(auto const& tag : nodes)
                    {
                        auto op = std::get<Operation>(m_graph.control.getElement(tag));
                        call(tag, op, coords);
                    }

                    // Add output nodes to candidates.
                    for(auto const& tag : nodes)
                    {
                        auto outTags = m_graph.control.getOutputNodeIndices<Sequence>(tag);
                        candidates.insert(outTags.begin(), outTags.end());
                    }

                    // Delete generated nodes from candidates.
                    for(auto const& node : nodes)
                        candidates.erase(node);
                }
            }

            void call(int tag, Operation const& op, Transformer coords)
            {
                auto opName = toString(op);
                Log::debug("MemoryTracer::{}({})", opName, tag);
                std::visit(*this, std::variant<int>(tag), op, std::variant<Transformer>(coords));
                m_completedControlNodes.insert(tag);
            }

            void operator()(int tag, AssertOp const& op, Transformer coords) {}

            void operator()(int tag, Assign const& op, Transformer coords) {}

            void operator()(int tag, Barrier const& op, Transformer coords) {}

            void operator()(int tag, Block const& op, Transformer coords)
            {
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                generate(body, coords);
            }

            void operator()(int tag, ComputeIndex const& op, Transformer coords) {}

            void operator()(int tag, ConditionalOp const& op, Transformer coords)
            {
                auto trueBody = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                generate(trueBody, coords);
                auto elseBody = m_graph.control.getOutputNodeIndices<Else>(tag).to<std::set>();
                if(!elseBody.empty())
                {
                    generate(elseBody, coords);
                }
            }

            void operator()(int tag, Deallocate const& op, Transformer coords) {}

            void operator()(int tag, DoWhileOp const& op, Transformer coords) {}

            void operator()(int tag, Exchange const& op, Transformer coords) {}

            void operator()(int tag, ForLoopOp const& op, Transformer coords)
            {
                auto loopIncrTag = m_graph.mapper.get(tag, NaryArgument::DEST);
                auto loopDims = m_graph.coordinates.getOutputNodeIndices<DataFlowEdge>(loopIncrTag);
                for(auto const& dim : loopDims)
                {
                    // XXX this is a hack, we should have a way to set the coordinate
                    Log::warn("Setting coordinate {} to 0 for ForLoop", dim);
                    coords.setCoordinate(dim, Expression::literal(0));
                }

                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                generate(body, coords);
            }

            void operator()(int tag, Kernel const& op, Transformer coords)
            {
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                generate(body, coords);
            }

            void operator()(int tag, LoadLDSTile const& load, Transformer coords)
            {
                auto [ldsTag, lds]   = m_graph.getDimension<LDS>(tag);
                auto [tileTag, tile] = m_graph.getDimension<MacroTile>(tag);

                auto maybeParentLDS
                    = only(m_graph.coordinates.getOutputNodeIndices(ldsTag, CT::isEdge<Duplicate>));
                if(maybeParentLDS)
                    ldsTag = *maybeParentLDS;

                if(tile.memoryType == MemoryType::WAVE)
                {
                    auto [waveTileTag, waveTile] = m_graph.getDimension<WaveTile>(tag);
                    auto [vgprTag, vgpr]         = m_graph.getDimension<VGPR>(tag);

                    auto dataTypeInfo = DataTypeInfo::Get(load.varType);
                    auto numBits
                        = static_cast<uint>(dataTypeInfo.elementBits / dataTypeInfo.packing);
                    auto numElements = getUnsignedInt(evaluate(vgpr.size));
                    auto numBytes    = (numBits * numElements) / 8u;

                    coords.setCoordinate(vgprTag, Expression::literal(0));
                    auto index = coords.reverse({ldsTag})[0];

                    Log::info("LDS WAVE LOAD: tag {}, numBits {}, numElements {}, numBytes {}",
                              tag,
                              numBits,
                              numElements,
                              numBytes);

                    m_events.push_back({tag,
                                        ldsTag,
                                        tileTag,
                                        Direction::LDSLoad,
                                        index * Expression::literal(numBits),
                                        0, // XXX
                                        numBytes});
                }
#if 0		
                else if(tile.memoryType == MemoryType::VGPR)
                {
                    auto [elemXTag, elemX] = m_graph.getDimension<ElementNumber>(tag, 0);
                    auto [elemYTag, elemY] = m_graph.getDimension<ElementNumber>(tag, 1);

                    auto m = getUnsignedInt(evaluate(elemX.size));
                    auto n = getUnsignedInt(evaluate(elemY.size));

                    Log::info("LDS VGPR LOAD: tag {}, m {}, n {}", tag, m, n);

                    // auto packing = DataTypeInfo::Get(load.varType).packing;
                    // n /= packing;

                    auto elementBits
                        = DataTypeInfo::Get(load.varType.getDereferencedType()).elementBits;
                    // auto numBytes = (elementBits * m * n) / 8u;

                    auto numBytes = elementBits / 8u;

                    for(auto i = 0; i < m; ++i)
                    {
                        coords.setCoordinate(elemXTag, Expression::literal(i));
                        for(auto j = 0; j < n; ++j)
                        {
                            coords.setCoordinate(elemYTag, Expression::literal(j));

                            // XXX do this once, just above
                            auto index = coords.reverse({ldsTag})[0];

                            m_events.push_back({tag,
                                                ldsTag,
                                                tileTag,
                                                Direction::LDSLoad,
                                                index * Expression::literal(numBytes),
                                                0, // XXX
                                                numBytes});

                            Log::debug("LDSLoad: tag {}, index {}, numBytes {}",
                                       tag,
                                       toString(index),
                                       numBytes);
                        }
                    }
                }
                else
                {
                    // XXX
                }
#endif
            }

            void operator()(int tag, LoadTileDirect2LDS const& op, Transformer coords) {}

            void operator()(int tag, LoadLinear const& op, Transformer coords) {}

            void operator()(int tag, LoadTiled const& load, Transformer coords)
            {
                return; // XXX

                auto [userTag, user] = m_graph.getDimension<User>(tag);
                auto [tileTag, tile] = m_graph.getDimension<MacroTile>(tag);

                if(tile.memoryType != MemoryType::VGPR)
                    return;

                // This seems wrong?
                auto numBytes
                    = DataTypeInfo::Get(load.varType.getDereferencedType()).elementBits / 8u;

                auto m = tile.sizes[0];
                auto n = tile.sizes[1];

                auto elemX = m_graph.mapper.get<ElementNumber>(tag, 0);
                auto elemY = m_graph.mapper.get<ElementNumber>(tag, 1);

                // XXX Can we we create dummy variables for these?  Then
                // only create the expression once and evaluate it
                // multiple times?

                for(auto i = 0; i < m; ++i)
                {
                    coords.setCoordinate(elemX, Expression::literal(i));
                    for(auto j = 0; j < n; ++j)
                    {
                        coords.setCoordinate(elemY, Expression::literal(j));

                        auto index = coords.reverse({userTag})[0];

                        m_events.push_back({tag,
                                            userTag,
                                            tileTag,
                                            Direction::GlobalLoad,
                                            index * Expression::literal(numBytes),
                                            0, // XXX
                                            numBytes});
                    }
                }
            }

            void operator()(int tag, LoadVGPR const& load, Transformer coords)
            {
                auto [userTag, user] = m_graph.getDimension<User>(tag);
                auto [vgprTag, vgpr] = m_graph.getDimension<VGPR>(tag);

                // Only one?
                auto numBytes
                    = DataTypeInfo::Get(load.varType.getDereferencedType()).elementBits / 8u;
                ExpressionPtr index
                    = load.scalar ? Expression::literal(0u) : coords.reverse({userTag})[0];

                m_events.push_back({tag,
                                    userTag,
                                    vgprTag,
                                    Direction::GlobalLoad,
                                    index * Expression::literal(numBytes),
                                    0, // XXX
                                    numBytes});
            }

            void operator()(int tag, LoadSGPR const& load, Transformer coords) {}

            void operator()(int tag, Multiply const& op, Transformer coords) {}

            void operator()(int tag, NOP const& op, Transformer coords) {}

            void operator()(int tag, Scope const& op, Transformer coords)
            {
                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                generate(body, coords);
            }

            void operator()(int tag, SeedPRNG const& op, Transformer coords) {}

            void operator()(int tag, SetCoordinate const& setCoordinate, Transformer coords)
            {
                auto connections = m_graph.mapper.getConnections(tag);
                coords.setCoordinate(connections[0].coordinate, setCoordinate.value);

                auto init = m_graph.control.getOutputNodeIndices<Initialize>(tag).to<std::set>();
                generate(init, coords);

                auto body = m_graph.control.getOutputNodeIndices<Body>(tag).to<std::set>();
                generate(body, coords);
            }

            void operator()(int tag, StoreLDSTile const& op, Transformer coords) {}

            void operator()(int tag, StoreLinear const& op, Transformer coords) {}

            void operator()(int tag, StoreTiled const& op, Transformer coords) {}

            void operator()(int tag, StoreVGPR const& op, Transformer coords) {}

            void operator()(int tag, StoreSGPR const& op, Transformer coords) {}

            void operator()(int tag, TensorContraction const& op, Transformer coords) {}

            void operator()(int tag, UnrollOp const& op, Transformer coords) {}

            void operator()(int tag, WaitZero const& op, Transformer coords) {}

            void simulateLaunch(auto& model, uint numWorkgroups, uint numWorkitems)
            {
                auto rawArguments     = m_arguments.dataVector();
                auto runtimeArguments = RuntimeArguments(rawArguments.data(), rawArguments.size());

                auto setWorkgroup = [&](uint i, uint v) {
                    *((uint*)(rawArguments.data() + m_workgroupOffset[i])) = v;
                };
                auto setWorkitem = [&](uint i, uint v) {
                    *((uint*)(rawArguments.data() + m_workitemOffset[i])) = v;
                };

                for(auto const& event : m_events)
                {
                    if(not model.filter(event))
                        continue;

                    for(uint wg = 0; wg < numWorkgroups; ++wg)
                    {
                        setWorkgroup(0, wg);
                        for(uint wi = 0; wi < numWorkitems; ++wi)
                        {
                            setWorkitem(0, wi);

                            // Might want to cache these

                            auto offsetValue = Expression::evaluate(event.index, runtimeArguments);
                            auto offset = std::visit([](auto x) { return (size_t)x; }, offsetValue);
                            auto simulated = MemoryEventSimulated{event.operationTag,
                                                                  event.sourceTag,
                                                                  event.destinationTag,
                                                                  event.instructionNumber,
                                                                  event.direction,
                                                                  static_cast<uint>(offset),
                                                                  event.bytesRequested,
                                                                  wg,
                                                                  wi};
                            model.simulate(simulated);
                        }
                    }
                }
            }

        private:
            KernelGraph   m_graph;
            std::set<int> m_completedControlNodes;

            std::list<MemoryEventExpression> m_events;

            KernelArguments  m_arguments;
            KernelInvocation m_invocation;

            std::array<uint, 3>          m_workgroupOffset, m_workitemOffset;
            std::array<ExpressionPtr, 3> m_kernelWorkgroupIndexes, m_kernelWorkitemIndexes;
        };
    }

    void memoryTrace(KernelGraph const&      original,
                     KernelInvocation const& invocation,
                     KernelArguments const&  arguments)
    {
        // XXX REMOVE THIS
        {
            std::ofstream dfile;
            dfile.open("SIMULATED.yaml", std::ofstream::out | std::ofstream::trunc);
            dfile << toYAML(original);
            dfile.close();
        }

        Log::info("MemoryTracer::memoryTrace()");

        auto graph  = original;
        auto tracer = MemoryTracer::MemoryTracer(graph, invocation, arguments);
        tracer.trace();

        Log::info("MemoryTracer::LDSBankModel()");
        // 64KiB bank model: 4 bytes per bank entry, 32 banks, 512 entries per bank
        auto model = MemoryTracer::LDSBankModel(4, 32, 512);

        // For LDS, just simulate using 1 workgroup
        auto workgroups            = 1;
        auto workitemsPerWorkgroup = product(invocation.workgroupSize);
        tracer.simulateLaunch(model, workgroups, workitemsPerWorkgroup);

        std::cout << model.summary() << std::endl;
    }

}
