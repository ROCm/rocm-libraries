
#include <iostream>
#include <memory>
#include <set>
#include <variant>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;

        LoadStoreTileGenerator::LoadStoreTileGenerator(std::shared_ptr<KernelGraph> graph,
                                                       ContextPtr                   context,
                                                       unsigned int workgroupSizeTotal)
            : m_graph(graph)
            , m_context(context)
            , m_fastArith(context)
            , m_workgroupSizeTotal(workgroupSizeTotal)
        {
        }

        inline Generator<Instruction> LoadStoreTileGenerator::generate(auto&         dest,
                                                                       ExpressionPtr expr) const
        {
            co_yield Expression::generate(dest, expr, m_context);
        }

        inline ExpressionPtr L(auto const& x)
        {
            return Expression::literal(x);
        }

        inline Register::ValuePtr LoadStoreTileGenerator::getBufferSrd(int tag)
        {
            auto bufferTag = m_graph->mapper.get<Buffer>(tag);
            return m_context->registerTagManager()->getRegister(bufferTag);
        }

        /**
             * @brief Build unrolled offset expression.
             *
             * Offsets inside unrolled loops look like:
             *
             *    offset = offset + unroll-iteration * stride
             *
             * where the additional piece is a local/independent
             * expression.
             *
             * When requesting an Offset register, this routines looks
             * nearby for Stride expressions connected to Unroll
             * coordinates, and returns the
             *
             *     + unroll-iteration * stride
             *
             * part of the offset above.
             */
        ExpressionPtr LoadStoreTileGenerator::getOffsetExpr(int                offsetTag,
                                                            Transformer const& coords)
        {
            // Find storage node connected to Offset edge.
            auto maybeTargetTag = findStorageNeighbour(offsetTag, *m_graph);
            if(!maybeTargetTag)
                return nullptr;
            auto [targetTag, direction] = *maybeTargetTag;

            // Find all required coordinates for the storage node,
            // and filter out Unroll coordinates.
            auto [required, path] = findRequiredCoordinates(targetTag, direction, *m_graph);
            auto unrolls          = filterCoordinates<Unroll>(required, *m_graph);

            if(unrolls.size() == 0)
                return nullptr;

            ExpressionPtr result = Expression::literal(0u);

            for(auto const& unroll : unrolls)
            {
                // Find the neighbour of the Unroll that:
                // 1. is in the load/store coordinate transform path
                // 2. has a Stride edge connected to it
                std::optional<int> maybeStrideTag;
                std::vector<int>   neighbourNodes;
                if(direction == Graph::Direction::Downstream)
                    neighbourNodes = m_graph->coordinates.parentNodes(unroll).to<std::vector>();
                else
                    neighbourNodes = m_graph->coordinates.childNodes(unroll).to<std::vector>();
                for(auto neighbourNode : neighbourNodes)
                {
                    if(path.contains(neighbourNode))
                    {
                        auto neighbourEdges = m_graph->coordinates.getNeighbours(
                            neighbourNode, Graph::opposite(direction));
                        for(auto neighbourEdge : neighbourEdges)
                        {
                            auto maybeStride = m_graph->coordinates.get<Stride>(neighbourEdge);
                            if(maybeStride
                               && m_context->registerTagManager()->hasExpression(neighbourEdge))
                            {
                                maybeStrideTag = neighbourEdge;
                            }
                        }
                    }
                }

                if(!maybeStrideTag)
                    continue;

                auto [strideExpr, _dtype]
                    = m_context->registerTagManager()->getExpression(*maybeStrideTag);

                result = result + coords.getCoordinate(unroll) * strideExpr;
            }

            return m_fastArith(result);
        }

        Generator<Instruction> LoadStoreTileGenerator::getOffset(Register::ValuePtr& dst,
                                                                 ExpressionPtr&      expr,
                                                                 Transformer         coords,
                                                                 int                 tag,
                                                                 int                 dimension)
        {
            auto offsetTag = m_graph->mapper.get<Offset>(tag, dimension);
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::getOffset(tag:"
                                               " {}, dimension: {}, offsetTag: {})",
                                               tag,
                                               dimension,
                                               offsetTag);
            if(offsetTag < 0)
                co_return;

            if(m_context->registerTagManager()->hasRegister(offsetTag))
            {
                dst  = m_context->registerTagManager()->getRegister(offsetTag);
                expr = getOffsetExpr(offsetTag, coords);
                co_return;
            }

            if(m_baseOffsets.count(offsetTag) > 0)
            {
                auto baseTag = m_baseOffsets[offsetTag];
                auto base    = m_context->registerTagManager()->getRegister(baseTag);

                dst = base->placeholder();
                co_yield m_context->copier()->copy(dst, base);
                dst->setName(concatenate("offset", offsetTag));

                m_context->getScopeManager()->addRegister(offsetTag);
                m_context->registerTagManager()->addRegister(offsetTag, dst);
                expr = getOffsetExpr(offsetTag, coords);
                co_return;
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::generateStride(Register::ValuePtr& stride,
                                                                      int                 tag,
                                                                      int                 dimension)
        {
            auto strideTag = m_graph->mapper.get<Stride>(tag, dimension);
            if(strideTag >= 0)
            {
                auto [strideExpr, strideDataType]
                    = m_context->registerTagManager()->getExpression(strideTag);
                strideExpr = m_fastArith(strideExpr);

                // If stride can be evaluated at compile time, return a literal. Otherwise,
                // create a new register.
                if(Expression::evaluationTimes(strideExpr)[EvaluationTime::Translate])
                {
                    stride = Register::Value::Literal(Expression::evaluate(strideExpr));
                }
                else
                {
                    stride = Register::Value::Placeholder(
                        m_context, Register::Type::Scalar, strideDataType, 1);
                    co_yield generate(stride, strideExpr);
                }
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genComputeIndex(int                 tag,
                                                                       ComputeIndex const& ci,
                                                                       Transformer         coords)
        {
            auto tagger = m_context->registerTagManager();

            auto base = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BASE});
            auto offset = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::OFFSET});
            auto stride = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});
            auto target = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::TARGET});
            auto increment = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::INCREMENT});

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::ComputeIndex({}): "
                "target {} increment {} base {} offset {} stride {}",
                tag,
                target,
                increment,
                base,
                offset,
                stride);

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = m_graph->coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of computing indexes,
                // use the parent LDS as the target instead.
                namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                auto maybeParentLDS = only(
                    m_graph->coordinates.getInputNodeIndices(target, CT::isEdge<PassThrough>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }

            auto scope    = m_context->getScopeManager();
            uint numBytes = DataTypeInfo::Get(ci.valueType).elementSize;

            coords.setCoordinate(increment, L(0u));
            for(int idx = 0;; ++idx)
            {
                auto zeroTag = m_graph->mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::ZERO, idx});
                if(zeroTag < 0)
                    break;
                coords.setCoordinate(zeroTag, L(0u));
            }

            if(base < 0)
            {
                // no base coordinate to copy offset from, so need
                // to explicity compute our own offset

                auto offsetReg
                    = tagger->getRegister(offset, Register::Type::Vector, ci.offsetType, 1);
                offsetReg->setName(concatenate("offset", tag));
                scope->addRegister(offset);

                auto indexExpr
                    = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

                rocRoller::Log::getLogger()->debug("  Offset({}): {}", offset, toString(indexExpr));

                co_yield generate(offsetReg, indexExpr * L(numBytes));
            }
            else
            {
                m_baseOffsets.insert_or_assign(offset, base);
            }

            if(stride > 0)
            {
                auto indexExpr = ci.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                            : coords.reverseStride(increment, L(1), {target})[0];
                rocRoller::Log::getLogger()->debug("  Stride({}): {}", stride, toString(indexExpr));
                tagger->addExpression(stride, indexExpr * L(numBytes), ci.strideType);
                scope->addRegister(stride);
            }

            auto buffer = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});
            if(buffer > 0)
            {
                auto user = m_graph->coordinates.get<User>(target);
                if(user && !tagger->hasRegister(buffer))
                {
                    auto bufferReg = tagger->getRegister(
                        buffer, Register::Type::Scalar, {DataType::None, PointerType::Buffer}, 1);
                    bufferReg->setName(concatenate("buffer", tag));
                    if(bufferReg->allocationState() == Register::AllocationState::Unallocated)
                    {
                        co_yield Register::AllocateIfNeeded(bufferReg);
                        Register::ValuePtr basePointer;
                        auto               bufDesc = BufferDescriptor(bufferReg, m_context);
                        co_yield m_context->argLoader()->getValue(user->argumentName(),
                                                                  basePointer);
                        co_yield bufDesc.setBasePointer(basePointer);
                        co_yield bufDesc.setDefaultOpts();
                    }
                    scope->addRegister(buffer);
                }
            }
        }

        /**
             * @brief Load a tile from memory into registers
             *
             * @param kind The kind of memory instruction to use
             * @param m Number of rows in the tile
             * @param n Number of columns in the tile
             * @param dataType The type of the data being loaded
             * @param pack Whether to pack smaller types into a single register
             * @param tag The tag of the control graph node generating the load
             * @param vgpr The registers to store the data in
             * @param offset Offset from the starting index
             * @return Generator<Instruction>
             */
        Generator<Instruction> LoadStoreTileGenerator::loadTile(MemoryInstructions::MemoryKind kind,
                                                                uint64_t                       m,
                                                                uint64_t                       n,
                                                                VariableType       dataType,
                                                                int                tag,
                                                                Register::ValuePtr offset,
                                                                Transformer&       coords)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::loadTile({})",
                                               tag);

            auto macTileTag = m_graph->mapper.get<MacroTile>(tag);

            Register::ValuePtr tmpl;
            unsigned int       packedAmount = 1;
            if(dataType == DataType::Half && n > 1)
            {
                tmpl = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Halfx2, m * n / 2);
                packedAmount = 2;
            }
            else
            {
                tmpl = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, dataType, m * n);
            }

            if(!offset)
            {
                offset = Register::Value::Literal(0u);
            }

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag, tmpl);
            co_yield Register::AllocateIfNeeded(vgpr);

            // Get the values from the associated ComputeIndex node
            Register::ValuePtr rowOffsetReg;
            ExpressionPtr      rowOffsetExpr;
            co_yield getOffset(rowOffsetReg, rowOffsetExpr, coords, tag, 0);
            auto colOffsetReg = rowOffsetReg->placeholder();

            if(rowOffsetExpr
               && Expression::evaluationTimes(rowOffsetExpr)[EvaluationTime::Translate]
               && offset->regType() == Register::Type::Literal
               && getUnsignedInt(evaluate(rowOffsetExpr)) > 0)
            {
                offset = Register::Value::Literal(getUnsignedInt(evaluate(rowOffsetExpr))
                                                  + getUnsignedInt(offset->getLiteralValue()));
            }
            else if(rowOffsetExpr)
            {
                auto unrolledRowOffsetExpr
                    = m_fastArith(rowOffsetReg->expression() + rowOffsetExpr);
                auto tmp = rowOffsetReg->placeholder();
                co_yield generate(tmp, unrolledRowOffsetExpr);
                rowOffsetReg = tmp;
            }

            AssertFatal(rowOffsetReg, "Invalid row offset register.");
            AssertFatal(colOffsetReg, "Invalid col offset register.");

            std::shared_ptr<BufferDescriptor> bufDesc;
            if(kind == MemoryInstructions::MemoryKind::Buffer)
            {
                auto bufferSrd = getBufferSrd(tag);
                bufDesc        = std::make_shared<BufferDescriptor>(bufferSrd, m_context);
            }

            Register::ValuePtr rowStrideReg, colStrideReg;
            if(m > 1)
                co_yield generateStride(rowStrideReg, tag, 0);
            else
                rowStrideReg = Register::Value::Literal(0u);
            co_yield generateStride(colStrideReg, tag, 1);

            AssertFatal(rowStrideReg, "Invalid row stride register.");
            AssertFatal(colStrideReg, "Invalid col stride register.");

            auto elementSize        = (uint)DataTypeInfo::Get(dataType).elementSize;
            bool colStrideIsLiteral = (colStrideReg->regType() == Register::Type::Literal);
            bool colStrideIsOne
                = colStrideIsLiteral
                  && (getUnsignedInt(colStrideReg->getLiteralValue()) == elementSize);

            auto proc      = Settings::getInstance()->get(Settings::Scheduler);
            auto cost      = Settings::getInstance()->get(Settings::SchedulerCost);
            auto scheduler = Component::GetNew<Scheduling::Scheduler>(proc, cost, m_context);
            std::vector<Generator<Instruction>> generators;

            // Load a tile of Half precision values where each register will hold
            // two half precision values.
            if(vgpr->variableType() == DataType::Halfx2 && !colStrideIsOne)
            {
                if(rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && offset->regType() == Register::Type::Literal)
                {
                    // If all of the strides are literals, we can load everything using offsets
                    // without using a runtime counter
                    auto offsetValue = getUnsignedInt(offset->getLiteralValue());
                    auto rowStride   = getUnsignedInt(rowStrideReg->getLiteralValue());
                    auto colStride   = getUnsignedInt(colStrideReg->getLiteralValue());
                    for(uint64_t i = 0; i < m; ++i)
                    {
                        for(uint64_t j = 0; j < n; j += 2)
                        {
                            uint a = i * n + j;

                            generators.push_back(m_context->mem()->loadAndPack(
                                kind,
                                vgpr->element({static_cast<int>(a / 2)}),
                                rowOffsetReg,
                                Register::Value::Literal(offsetValue + j * colStride),
                                rowOffsetReg,
                                Register::Value::Literal(offsetValue + (j + 1) * colStride),
                                "",
                                bufDesc));
                        }
                        offsetValue += rowStride;
                    }
                    co_yield (*scheduler)(generators);
                }
                else
                {
                    auto gen = [ctx = m_context,
                                rowOffsetReg,
                                rowStrideReg,
                                colStrideReg,
                                vgpr,
                                kind,
                                offset,
                                bufDesc](uint64_t i, uint64_t j, uint a) -> Generator<Instruction> {
                        FastArithmetic     fa(ctx);
                        Register::ValuePtr offset1;
                        Register::ValuePtr offset2;

                        co_yield Expression::generate(
                            offset1,
                            fa(rowOffsetReg->expression()
                               + colStrideReg->expression() * Expression::literal(j)),
                            ctx);
                        co_yield Expression::generate(
                            offset2, fa(offset1->expression() + colStrideReg->expression()), ctx);

                        co_yield ctx->mem()->loadAndPack(kind,
                                                         vgpr->element({static_cast<int>(a / 2)}),
                                                         offset1,
                                                         offset,
                                                         offset2,
                                                         offset,
                                                         "",
                                                         bufDesc);
                    };

                    for(uint64_t i = 0; i < m; ++i)
                    {
                        for(uint64_t j = 0; j < n; j += 2)
                        {
                            uint a = i * n + j;
                            generators.push_back(gen(i, j, a));
                        }
                        co_yield (*scheduler)(generators);
                        generators.clear();
                        if(i < m - 1)
                        {
                            co_yield generate(rowOffsetReg,
                                              rowOffsetReg->expression()
                                                  + rowStrideReg->expression());
                        }
                    }
                }
            }
            else
            {
                if(rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && offset->regType() == Register::Type::Literal)
                {
                    // If all of the strides are literals, we can load everything using offsets
                    // without using a runtime counter
                    auto offsetValue = getUnsignedInt(offset->getLiteralValue());
                    auto rowStride   = getUnsignedInt(rowStrideReg->getLiteralValue());
                    auto colStride   = getUnsignedInt(colStrideReg->getLiteralValue());
                    if(colStrideIsOne)
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            auto start = (i * n) / packedAmount;
                            auto stop  = (i * n + n) / packedAmount;
                            co_yield m_context->mem()->load(
                                kind,
                                vgpr->element(Generated(iota(start, stop))),
                                rowOffsetReg,
                                Register::Value::Literal(offsetValue),
                                elementSize * n,
                                "",
                                false,
                                bufDesc);
                            offsetValue += rowStride;
                        }
                    }
                    else
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            for(uint64_t j = 0; j < n; ++j)
                            {
                                co_yield m_context->mem()->load(
                                    kind,
                                    vgpr->element({static_cast<int>(i * n + j)}),
                                    rowOffsetReg,
                                    Register::Value::Literal(offsetValue + j * colStride),
                                    elementSize,
                                    "",
                                    false,
                                    bufDesc);
                            }
                            offsetValue += rowStride;
                        }
                    }
                }
                else
                {
                    if(colStrideIsOne)
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            auto start = (i * n) / packedAmount;
                            auto stop  = (i * n + n) / packedAmount;
                            co_yield m_context->mem()->load(
                                kind,
                                vgpr->element(Generated(iota(start, stop))),
                                rowOffsetReg->subset({0}),
                                offset,
                                elementSize * n,
                                "",
                                false,
                                bufDesc);

                            if(i < m - 1)
                            {
                                co_yield generate(rowOffsetReg,
                                                  rowOffsetReg->expression()
                                                      + rowStrideReg->expression());
                            }
                        }
                    }
                    else
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            co_yield m_context->copier()->copy(colOffsetReg, rowOffsetReg);

                            for(uint64_t j = 0; j < n; ++j)
                            {
                                co_yield m_context->mem()->load(
                                    kind,
                                    vgpr->element({static_cast<int>(i * n + j)}),
                                    colOffsetReg->subset({0}),
                                    offset,
                                    elementSize,
                                    "",
                                    false,
                                    bufDesc);
                                if(j < n - 1)
                                {
                                    co_yield generate(colOffsetReg,
                                                      colOffsetReg->expression()
                                                          + colStrideReg->expression());
                                }
                            }

                            if(i < m - 1)
                            {
                                co_yield generate(rowOffsetReg,
                                                  rowOffsetReg->expression()
                                                      + rowStrideReg->expression());
                            }
                        }
                    }
                }
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileVGPRCI(int              tag,
                                                                           LoadTiled const& load,
                                                                           Transformer      coords,
                                                                           int              sdim)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileVGPRCI()");
            co_yield Instruction::Comment("GEN: loadMacroTileVGPRCI");

            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

            co_yield loadTile(
                MemoryInstructions::MemoryKind::Buffer, m, n, load.vtype, tag, nullptr, coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileVGPR(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileVGPR({})", tag);
            co_yield Instruction::Comment("GEN: loadMacroTileVGPR");

            auto [userTag, user]       = m_graph->getDimension<User>(tag);
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            auto const m = macTile.subTileSizes[0];
            auto const n = macTile.subTileSizes[1];

            AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

            rocRoller::Log::getLogger()->debug(
                "  macro tile: {}; sub tile size: {}x{}", macTileTag, m, n);

            co_yield loadTile(
                MemoryInstructions::MemoryKind::Buffer, m, n, load.vtype, tag, nullptr, coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileLDS(int                tag,
                                                                        LoadLDSTile const& load,
                                                                        Transformer        coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileLDS()");
            co_yield_(Instruction::Comment("GEN: loadMacroTileLDS"));

            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto const m = tile.subTileSizes[0];
            auto const n = tile.subTileSizes[1];

            co_yield loadTile(
                MemoryInstructions::MemoryKind::Local, m, n, load.vtype, tag, ldsOffset, coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVELDSCI(
            int tag, LoadLDSTile const& load, Transformer coords, int sdim)
        {
            co_yield_(Instruction::Comment("GEN: loadMacroTileWAVELDSCI"));

            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVELDSCI({}, {})",
                ldsTag,
                waveTileTag);

            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto vtype = ldsAllocation->variableType();

            auto nWaveTag = m_graph->mapper.get<WaveTileNumber>(tag, sdim);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;
            co_yield loadTile(MemoryInstructions::MemoryKind::Local,
                              1,
                              numVgpr,
                              load.vtype,
                              tag,
                              ldsOffset,
                              coords);
        }

        // CI : compute index
        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVECI(int              tag,
                                                                           LoadTiled const& load,
                                                                           Transformer      coords,
                                                                           int              sdim)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVECI({})", tag);
            co_yield Instruction::Comment("GEN: loadMacroTileWAVECI");

            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            co_yield loadTile(MemoryInstructions::MemoryKind::Buffer,
                              1,
                              numVgpr,
                              load.vtype,
                              tag,
                              nullptr,
                              coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVECIACCUM(
            int tag, LoadTiled const& load, Transformer coords)

        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVECIACCUM({})", tag);
            co_yield Instruction::Comment("GEN: loadMacroTileWAVECIACCUM");

            auto [userTag, user]         = m_graph->getDimension<User>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            co_yield loadTile(MemoryInstructions::MemoryKind::Buffer,
                              numVgpr / 4,
                              4,
                              load.vtype,
                              tag,
                              nullptr,
                              coords);
        }

        Generator<Instruction>
            LoadStoreTileGenerator::genLoadTile(int tag, LoadTiled const& load, Transformer coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                    co_yield loadMacroTileVGPRCI(tag, load, coords, 1);
                    break;
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileVGPRCI(tag, load, coords, 0);
                    break;
                default:
                    co_yield loadMacroTileVGPR(tag, load, coords);
                    break;
                }
            }
            break;
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                    co_yield loadMacroTileWAVECI(tag, load, coords, 1);
                    break;
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileWAVECI(tag, load, coords, 0);
                    break;
                case LayoutType::MATRIX_ACCUMULATOR:
                    co_yield loadMacroTileWAVECIACCUM(tag, load, coords);
                    break;
                default:
                    Throw<FatalError>("Layout type not supported yet for LoadTiled.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for LoadTiled.");
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genLoadLDSTile(int                tag,
                                                                      LoadLDSTile const& load,
                                                                      Transformer        coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
                co_yield loadMacroTileLDS(tag, load, coords);
                break;
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                    co_yield loadMacroTileWAVELDSCI(tag, load, coords, 1);
                    break;
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileWAVELDSCI(tag, load, coords, 0);
                    break;
                default:
                    Throw<FatalError>("Layout type not supported yet for LoadLDSTile.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for LoadLDSTile.");
            }
        }

        /**
             * @brief Store a tile from registers into memory
             *
             * @param kind The kind of memory instruction to use
             * @param m Number of rows in the tile
             * @param n Number of columns in the tile
             * @param dataType The type of the data being stored
             * @param tag The tag of the control graph node generating the store
             * @param vgpr The registers containing the data
             * @param offset Offset from the starting index
             * @return Generator<Instruction>
             */
        Generator<Instruction>
            LoadStoreTileGenerator::storeTile(MemoryInstructions::MemoryKind kind,
                                              uint64_t                       m,
                                              uint64_t                       n,
                                              VariableType                   dataType,
                                              int                            tag,
                                              Register::ValuePtr             vgpr,
                                              Register::ValuePtr             offset,
                                              Transformer&                   coords)
        {
            auto elementSize = DataTypeInfo::Get(dataType).elementSize;

            if(!offset)
            {
                offset = Register::Value::Literal(0u);
            }

            Register::ValuePtr rowOffsetReg;
            ExpressionPtr      rowOffsetExpr;
            co_yield getOffset(rowOffsetReg, rowOffsetExpr, coords, tag, 0);
            auto colOffsetReg = rowOffsetReg->placeholder();

            AssertFatal(rowOffsetReg, "Invalid row offset register.");
            AssertFatal(colOffsetReg, "Invalid col offset register.");

            if(rowOffsetExpr
               && Expression::evaluationTimes(rowOffsetExpr)[EvaluationTime::Translate]
               && offset->regType() == Register::Type::Literal
               && getUnsignedInt(evaluate(rowOffsetExpr)) > 0)
            {
                offset = Register::Value::Literal(getUnsignedInt(evaluate(rowOffsetExpr))
                                                  + getUnsignedInt(offset->getLiteralValue()));
            }
            else if(rowOffsetExpr)
            {
                auto unrolledRowOffsetExpr = simplify(rowOffsetReg->expression() + rowOffsetExpr);
                auto tmp                   = rowOffsetReg->placeholder();
                co_yield generate(tmp, unrolledRowOffsetExpr);
                rowOffsetReg = tmp;
            }

            std::shared_ptr<BufferDescriptor> bufDesc;
            if(kind == MemoryInstructions::MemoryKind::Buffer)
            {
                auto bufferSrd = getBufferSrd(tag);
                bufDesc        = std::make_shared<BufferDescriptor>(bufferSrd, m_context);
            }

            Register::ValuePtr rowStrideReg, colStrideReg;
            co_yield generateStride(rowStrideReg, tag, 0);
            co_yield generateStride(colStrideReg, tag, 1);

            AssertFatal(rowStrideReg, "Invalid row stride register.");
            AssertFatal(colStrideReg, "Invalid col stride register.");

            if(!m_context->targetArchitecture().HasCapability(GPUCapability::ArchAccUnifiedRegs))
            {
                co_yield m_context->copier()->ensureType(vgpr, vgpr, Register::Type::Vector);
            }

            // Convert the data to the expected datatype
            Register::ValuePtr converted;
            if(DataTypeInfo::Get(vgpr->variableType()).segmentVariableType != dataType)
            {
                co_yield m_context->copier()->ensureType(vgpr, vgpr, Register::Type::Vector);
                converted = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, dataType, vgpr->valueCount());
                co_yield converted->allocate();
                for(int i = 0; i < vgpr->valueCount(); ++i)
                {
                    Register::ValuePtr tmp = converted->element({i});
                    co_yield Expression::generate(
                        tmp,
                        convert(dataType.dataType,
                                std::make_shared<Expression::Expression>(vgpr->element({i}))),
                        m_context);
                }
            }
            else
            {
                converted = vgpr;
            }

            unsigned int packedAmount = DataTypeInfo::Get(converted->variableType()).packing;

            bool colStrideIsLiteral = (colStrideReg->regType() == Register::Type::Literal);

            bool colStrideIsOne
                = colStrideIsLiteral
                  && (getUnsignedInt(colStrideReg->getLiteralValue()) == elementSize);
            // Load a tile of Half precision values where each register will hold
            // two half precision values.
            if(converted->variableType() == DataType::Halfx2 && !colStrideIsOne)
            {
                if(rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && offset->regType() == Register::Type::Literal)
                {
                    // If all of the strides are literals, we can load everything using offsets
                    // without using a runtime counter
                    auto offsetValue = getUnsignedInt(offset->getLiteralValue());
                    auto rowStride   = getUnsignedInt(rowStrideReg->getLiteralValue());
                    auto colStride   = getUnsignedInt(colStrideReg->getLiteralValue());
                    for(uint64_t i = 0; i < m; ++i)
                    {
                        for(uint64_t j = 0; j < n; ++j)
                        {
                            uint a = (i * n + j) / 2;

                            co_yield m_context->mem()->store(
                                kind,
                                rowOffsetReg,
                                converted->element({static_cast<int>(a)}),
                                Register::Value::Literal(offsetValue + j * colStride),
                                elementSize,
                                "",
                                j % 2 == 1,
                                bufDesc);
                        }
                        offsetValue += rowStride;
                    }
                }
                else
                {
                    for(uint64_t i = 0; i < m; ++i)
                    {
                        co_yield m_context->copier()->copy(colOffsetReg, rowOffsetReg);
                        for(uint64_t j = 0; j < n; ++j)
                        {
                            uint a = (i * n + j) / 2;

                            co_yield m_context->mem()->store(
                                kind,
                                colOffsetReg->subset({0}),
                                converted->element({static_cast<int>(a)}),
                                offset,
                                elementSize,
                                "",
                                j % 2 == 1,
                                bufDesc);

                            if(j < n - 1)
                            {
                                co_yield generate(colOffsetReg,
                                                  colOffsetReg->expression()
                                                      + colStrideReg->expression());
                            }
                        }
                        if(i < m - 1)
                            co_yield generate(rowOffsetReg,
                                              rowOffsetReg->expression()
                                                  + rowStrideReg->expression());
                    }
                }
            }
            else
            {
                if(rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && offset->regType() == Register::Type::Literal)
                {
                    // If all of the strides are literals, we can store everything using offsets
                    // without using a runtime counter
                    auto offsetValue = getUnsignedInt(offset->getLiteralValue());
                    auto rowStride   = getUnsignedInt(rowStrideReg->getLiteralValue());
                    auto colStride   = getUnsignedInt(colStrideReg->getLiteralValue());
                    if(colStrideIsOne)
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            auto start = (i * n) / packedAmount;
                            auto stop  = (i * n + n) / packedAmount;
                            co_yield m_context->mem()->store(
                                kind,
                                rowOffsetReg,
                                converted->element(Generated(iota(start, stop))),
                                Register::Value::Literal(offsetValue),
                                elementSize * n,
                                "",
                                false,
                                bufDesc);
                            offsetValue += rowStride;
                        }
                    }
                    else
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            for(uint64_t j = 0; j < n; ++j)
                            {
                                uint a = i * n + j;
                                co_yield m_context->mem()->store(
                                    kind,
                                    rowOffsetReg,
                                    converted->element({static_cast<int>(a)}),
                                    Register::Value::Literal(offsetValue + j * colStride),
                                    elementSize,
                                    "",
                                    false,
                                    bufDesc);
                            }
                            offsetValue += rowStride;
                        }
                    }
                }
                else
                {
                    if(colStrideIsOne)
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            auto start = (i * n) / packedAmount;
                            auto stop  = (i * n + n) / packedAmount;
                            co_yield m_context->mem()->store(
                                kind,
                                rowOffsetReg->subset({0}),
                                converted->element(Generated(iota(start, stop))),
                                offset,
                                elementSize * n,
                                "",
                                false,
                                bufDesc);

                            if(i < m - 1)
                            {
                                co_yield generate(rowOffsetReg,
                                                  rowOffsetReg->expression()
                                                      + rowStrideReg->expression());
                            }
                        }
                    }
                    else
                    {
                        for(uint64_t i = 0; i < m; ++i)
                        {
                            co_yield m_context->copier()->copy(colOffsetReg, rowOffsetReg);
                            for(int j = 0; j < n; ++j)
                            {
                                uint a = i * n + j;

                                co_yield m_context->mem()->store(
                                    kind,
                                    colOffsetReg->subset({0}),
                                    converted->element({static_cast<int>(a)}),
                                    offset,
                                    elementSize,
                                    "",
                                    false,
                                    bufDesc);
                                if(j < n - 1)
                                {
                                    co_yield generate(colOffsetReg,
                                                      colOffsetReg->expression()
                                                          + colStrideReg->expression());
                                }
                            }

                            if(i < m - 1)
                            {
                                co_yield generate(rowOffsetReg,
                                                  rowOffsetReg->expression()
                                                      + rowStrideReg->expression());
                            }
                        }
                    }
                }
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileLDS(int                 tag,
                                                                         StoreLDSTile const& store,
                                                                         Transformer         coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileLDS()");
            co_yield Instruction::Comment("GEN: storeMacroTileLDS");

            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            // Temporary register(s) that is used to copy the data from global memory to
            // local memory.
            auto vgpr  = m_context->registerTagManager()->getRegister(tileTag);
            auto vtype = store.dataType;

            Register::ValuePtr rowOffsetReg;
            ExpressionPtr      rowOffsetExpr;
            co_yield getOffset(rowOffsetReg, rowOffsetExpr, coords, tag, 0);
            AssertFatal(rowOffsetReg, "Invalid row offset register.");

            auto numElements = product(tile.subTileSizes) * m_workgroupSizeTotal;
            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(m_context, vtype, numElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto const n           = getUnsignedInt(evaluate(elemY.size));

            // saving the offsets to be restored for each macrotile in LDS
            // TODO : Need more design thought (how to seed an offset register)
            auto resetOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            co_yield m_context->copier()->copy(resetOffset, rowOffsetReg);

            co_yield storeTile(
                MemoryInstructions::MemoryKind::Local, m, n, vtype, tag, vgpr, ldsOffset, coords);

            // TODO : Need more design thought (how to seed an offset register)
            co_yield m_context->copier()->copy(rowOffsetReg, resetOffset);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileVGPR(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            auto [userTag, user]       = m_graph->getDimension<User>(tag);
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileVGPR({})", tag);
            co_yield Instruction::Comment("GEN: storeMacroTileVGPR");

            rocRoller::Log::getLogger()->debug("  user {}; tile {}", userTag, macTileTag);

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto const m = macTile.subTileSizes[0];
            auto const n = macTile.subTileSizes[1];

            co_yield storeTile(MemoryInstructions::MemoryKind::Buffer,
                               m,
                               n,
                               store.dataType,
                               tag,
                               vgpr,
                               nullptr,
                               coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVELDS(
            int tag, StoreLDSTile const& store, Transformer coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileWAVELDS()");
            co_yield Instruction::Comment("GEN: storeMacroTileWAVELDS");

            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto macrotileNumElements    = product(macTile.sizes);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);
            uint waveTileNumElements     = waveTile.sizes[0] * waveTile.sizes[1];
            auto vtype                   = store.dataType;

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation
                    = Register::Value::AllocateLDS(m_context, vtype, macrotileNumElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            uint wfs     = m_context->kernel()->wavefront_size();
            uint numVgpr = waveTileNumElements / wfs;
            auto agpr    = m_context->registerTagManager()->getRegister(macTileTag);
            AssertFatal(agpr->registerCount() == numVgpr);

            co_yield storeTile(MemoryInstructions::MemoryKind::Local,
                               numVgpr / 4,
                               4,
                               vtype,
                               tag,
                               agpr,
                               ldsOffset,
                               coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVECI(int               tag,
                                                                            StoreTiled const& store,
                                                                            Transformer coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileWAVE()");
            co_yield Instruction::Comment("GEN: storeMacroTileWAVE");

            auto [userTag, user]         = m_graph->getDimension<User>(tag);
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            uint numElements = waveTile.sizes[0] * waveTile.sizes[1];
            uint wfs         = m_context->kernel()->wavefront_size();
            uint numVgpr     = numElements / wfs;

            auto agpr = m_context->registerTagManager()->getRegister(macTileTag);

            AssertFatal(agpr->registerCount() == numVgpr);

            co_yield storeTile(MemoryInstructions::MemoryKind::Buffer,
                               numVgpr / 4,
                               4,
                               store.dataType,
                               tag,
                               agpr,
                               nullptr,
                               coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::genStoreTile(int               tag,
                                                                    StoreTiled const& store,
                                                                    Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
                co_yield storeMacroTileVGPR(tag, store, coords);
                break;
            case MemoryType::WAVE:
                co_yield storeMacroTileWAVECI(tag, store, coords);
                break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for StoreTiled.");
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genStoreLDSTile(int                 tag,
                                                                       StoreLDSTile const& store,
                                                                       Transformer         coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
                co_yield storeMacroTileLDS(tag, store, coords);
                break;
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_ACCUMULATOR:
                    co_yield storeMacroTileWAVELDS(tag, store, coords);
                    break;
                default:
                    Throw<FatalError>("Layout type not supported yet for StoreLDSTile.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for StoreLDSTile.");
            }
        }
    }
}
