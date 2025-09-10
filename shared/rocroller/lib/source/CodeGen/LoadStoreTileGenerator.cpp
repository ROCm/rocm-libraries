/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#include <memory>

#include <rocRoller/CodeGen/Annotate.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Buffer.hpp>
#include <rocRoller/CodeGen/BufferInstructionOptions.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/LoadStoreTileGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile_details.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace CT         = rocRoller::KernelGraph::CoordinateGraph;
        namespace Expression = rocRoller::Expression;
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        using namespace Expression;
        using namespace LowerTileDetails;

        std::string toString(LoadStoreTileGenerator::LoadStoreTileInfo const& info)
        {
#define ShowReg(reg)                                              \
    (reg ? concatenate("\t" #reg " = ", reg->description(), "\n") \
         : concatenate("\t" #reg " = (nullptr)\n"))
#define ShowBuf(buf)                                                              \
    (buf && buf->allRegisters()                                                   \
         ? concatenate("\t" #buf " = ", buf->allRegisters()->description(), "\n") \
         : concatenate("\t" #buf " = (nullptr)\n"))

            return concatenate("LSTInfo {\n",
                               ShowValue(info.kind),
                               ShowValue(info.m),
                               ShowValue(info.n),
                               ShowValue(info.elementBits),
                               ShowValue(info.packedAmount),
                               ShowValue(info.ldsWriteStride),
                               ShowReg(info.data),
                               ShowReg(info.rowOffsetReg),
                               ShowReg(info.rowStrideReg),
                               ShowValue(info.rowStrideAttributes),
                               ShowReg(info.colStrideReg),
                               ShowValue(info.colStrideAttributes),
                               ShowReg(info.offset),
                               ShowBuf(info.bufDesc),
                               ShowValue(info.bufOpts),
                               ShowValue(info.isTransposedTile),
                               "}");
        }

        LoadStoreTileGenerator::LoadStoreTileGenerator(KernelGraphPtr graph,
                                                       ContextPtr     context,
                                                       unsigned int   workgroupSizeTotal)
            : m_graph(graph)
            , m_context(context)
            , m_fastArith(FastArithmetic(context))
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

        inline std::shared_ptr<BufferDescriptor> LoadStoreTileGenerator::getBufferDesc(int tag)
        {
            auto bufferTag = m_graph->mapper.get<Buffer>(tag);
            auto bufferSrd = m_context->registerTagManager()->getRegister(bufferTag);
            return std::make_shared<BufferDescriptor>(bufferSrd, m_context);
        }

        static std::pair<uint, uint>
            getElementBlockValues(KernelGraph const& graph, int target, const bool isTransposed)
        {
            uint elementBlockNumber = 0;
            uint elementBlockIndex  = 0;

            std::unordered_set<int> tileTags;
            using OpsAndTilesType
                = std::tuple<std::pair<int, Operation>, std::pair<int, MacroTile>, DataType>;
            std::vector<OpsAndTilesType> targetOpsAndTiles;

            for(auto conn : graph.mapper.getCoordinateConnections(target))
            {
                auto     opTag = conn.control;
                auto     op    = std::get<Operation>(graph.control.getElement(opTag));
                DataType dataType;
                if(std::visit(rocRoller::overloaded{[&](LoadTiled& load) {
                                                        dataType = load.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](LoadLDSTile& load) {
                                                        dataType = load.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](StoreTiled& store) {
                                                        dataType = store.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](StoreLDSTile& store) {
                                                        dataType = store.varType.dataType;
                                                        return true;
                                                    },
                                                    [&](auto& other) { return false; }},
                              op))
                {
                    auto [macTileTag, macTile] = graph.getDimension<MacroTile>(opTag);

                    auto maybeParentTile = only(
                        graph.coordinates.getOutputNodeIndices(macTileTag, CT::isEdge<Duplicate>));
                    if(maybeParentTile)
                    {
                        macTileTag = *maybeParentTile;
                        macTile    = *graph.coordinates.get<MacroTile>(macTileTag);
                    }

                    if(!tileTags.count(macTileTag))
                    {
                        targetOpsAndTiles.push_back({{opTag, op}, {macTileTag, macTile}, dataType});
                    }
                }
            }

            auto [tagAndOp, tagAndTile, dataType] = [](auto opsAndTiles) -> OpsAndTilesType {
                for(OpsAndTilesType& elem : opsAndTiles)
                {
                    auto memType = std::get<1>(elem).second.memoryType;
                    if(memType == MemoryType::WAVE || memType == MemoryType::WAVE_SWIZZLE)
                    {
                        return elem;
                    }
                }
                return opsAndTiles[0];
            }(targetOpsAndTiles);
            auto [opTag, op]           = tagAndOp;
            auto [macTileTag, macTile] = tagAndTile;

            if(macTile.memoryType == MemoryType::VGPR
               || (macTile.layoutType == LayoutType::MATRIX_ACCUMULATOR
                   && macTile.memoryType == MemoryType::WAVE_SPLIT))
            {
                auto [elementNumberXTag, elementNumberX]
                    = graph.getDimension<ElementNumber>(opTag, 0);
                AssertFatal(
                    Expression::evaluationTimes(elementNumberX.size)[EvaluationTime::Translate],
                    "Could not determine ElementNumberX size at translate-time.\n",
                    ShowValue(elementNumberX));

                auto [elementNumberYTag, elementNumberY]
                    = graph.getDimension<ElementNumber>(opTag, 1);
                AssertFatal(
                    Expression::evaluationTimes(elementNumberY.size)[EvaluationTime::Translate],
                    "Could not determine ElementNumber size at translate-time.\n",
                    ShowValue(elementNumberY));

                elementBlockNumber = getUnsignedInt(evaluate(elementNumberX.size));
                elementBlockIndex  = getUnsignedInt(evaluate(elementNumberY.size));
            }
            else if(macTile.memoryType == MemoryType::WAVE
                    || macTile.memoryType == MemoryType::WAVE_SWIZZLE)
            {
                auto [vgprBlockNumberTag, vgprBlockNumber]
                    = graph.getDimension<VGPRBlockNumber>(opTag, 0);
                AssertFatal(
                    Expression::evaluationTimes(vgprBlockNumber.size)[EvaluationTime::Translate],
                    "Could not determine VGPRBlockNumber size at translate-time.\n",
                    ShowValue(vgprBlockNumber));

                auto [vgprBlockIndexTag, vgprBlockIndex]
                    = graph.getDimension<VGPRBlockIndex>(opTag, 0);
                AssertFatal(
                    Expression::evaluationTimes(vgprBlockIndex.size)[EvaluationTime::Translate],
                    "Could not determine VGPRBlockIndex size at translate-time.\n",
                    ShowValue(vgprBlockIndex));

                elementBlockNumber = getUnsignedInt(evaluate(vgprBlockNumber.size));
                elementBlockIndex  = getUnsignedInt(evaluate(vgprBlockIndex.size));
                if(isScaleType(dataType))
                {
                    // Scales are another special case here. For Scales we need
                    // to get VGPR coordinate instead of VGPRBlockNumber/Index
                    // (see addLoadSwizzleTileCT).
                    auto [vgprTag, vgpr] = graph.getDimension<VGPR>(opTag, 0);
                    AssertFatal(Expression::evaluationTimes(vgpr.size)[EvaluationTime::Translate],
                                "Could not determine VGPR size at translate-time.\n",
                                ShowValue(vgpr));
                    // Multiplying by elementBlockNumber here forces the use
                    // of the widest load/store possible
                    elementBlockIndex = elementBlockNumber * getUnsignedInt(evaluate(vgpr.size));
                }

                if((!isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(dataType,
                                                                     {.m = macTile.subTileSizes[0],
                                                                      .n = macTile.subTileSizes[1],
                                                                      .k = macTile.subTileSizes[2]})
                    || isScaleType(dataType))
                   && !isTransposed)
                {
                    // For Scales and other kinds of tiles, VGPRBlockIndex holds
                    // number of VGPR per block and not elements per VGPRBlock.
                    elementBlockIndex *= packingFactorForDataType(dataType);
                }
            }
            else
            {
                Throw<FatalError>(
                    "Could not find ElementNumber or VGPRBlockNumber/Index coordinates.\n",
                    ShowValue(op),
                    ShowValue(macTile));
            }

            AssertFatal(elementBlockNumber > 0 && elementBlockIndex > 0,
                        "elemementBlockNumber & elementBlockIndex must be greater than zero. ",
                        ShowValue(elementBlockNumber),
                        ShowValue(elementBlockIndex));
            return {elementBlockNumber, elementBlockIndex};
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
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::getOffsetExpr(offsetTag: {})", offsetTag);

            // Find storage node connected to Offset edge.
            auto maybeTargetTag = findStorageNeighbour(offsetTag, *m_graph);
            if(!maybeTargetTag)
            {
                return nullptr;
            }

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

                auto [strideExpr, strideAttrs]
                    = m_context->registerTagManager()->getExpression(*maybeStrideTag);

                Log::debug(
                    "  unroll coord {} value: {}", unroll, toString(coords.getCoordinate(unroll)));

                result = result + coords.getCoordinate(unroll) * strideExpr;
            }

            return result;
        }

        DataType getOffsetTypeFromComputeIndex(const KernelGraph& graph, int offsetTag)
        {
            for(auto const& conn : graph.mapper.getCoordinateConnections(offsetTag))
            {
                if(auto computeIndex = graph.control.get<ComputeIndex>(conn.control);
                   computeIndex.has_value())
                {
                    return computeIndex->offsetType;
                }
            }
            Throw<FatalError>("No ComputeIndex found for Offset tag.", ShowValue(offsetTag));
        }

        Generator<Instruction> LoadStoreTileGenerator::getOffset(LoadStoreTileInfo& info,
                                                                 Transformer        coords,
                                                                 int                tag,
                                                                 bool               preserveOffset,
                                                                 bool               direct2LDS)
        {
            auto offsetTag = m_graph->mapper.get<Offset>(tag, direct2LDS ? 2 : 0);
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::getOffset(tag:"
                                               " {}, offsetTag: {})",
                                               tag,
                                               offsetTag);

            AssertFatal(offsetTag >= 0, "No Offset found");

            ExpressionPtr rowOffsetExpr;

            if(m_context->registerTagManager()->hasRegister(offsetTag))
            {
                if(direct2LDS)
                {
                    auto tmp  = m_context->registerTagManager()->getRegister(offsetTag);
                    auto expr = info.data->expression() + tmp->expression();

                    if(info.data->regType() == Register::Type::Literal)
                        info.data = nullptr;

                    co_yield generate(info.data, expr);
                }
                else
                {
                    info.rowOffsetReg = m_context->registerTagManager()->getRegister(offsetTag);
                }

                rowOffsetExpr = getOffsetExpr(offsetTag, coords);
            }
            else if(m_baseOffsets.count(offsetTag) > 0)
            {
                auto baseTag = m_baseOffsets[offsetTag];
                if(direct2LDS)
                {
                    auto tmp = m_context->registerTagManager()->getRegister(baseTag);
                    co_yield generate(info.data, info.data->expression() + tmp->expression());
                    m_context->getScopeManager()->addRegister(offsetTag);
                    m_context->registerTagManager()->addRegister(offsetTag, info.data);
                }
                else
                {
                    info.rowOffsetReg = m_context->registerTagManager()->getRegister(
                        offsetTag,
                        Register::Type::Vector,
                        getOffsetTypeFromComputeIndex(*m_graph, offsetTag),
                        1);
                    info.rowOffsetReg->setName(concatenate("offset", offsetTag));
                    m_context->getScopeManager()->addRegister(offsetTag);

                    // Copy base to new offset register
                    auto baseReg = m_context->registerTagManager()->getRegister(baseTag);
                    co_yield m_context->copier()->copy(info.rowOffsetReg, baseReg);
                }

                rowOffsetExpr = getOffsetExpr(offsetTag, coords);
            }
            else
            {
                Throw<FatalError>("Base offset not found");
            }

            if(direct2LDS)
            {
                co_return;
            }

            if(rowOffsetExpr)
                rocRoller::Log::getLogger()->debug("  rowOffsetExpr: {}", toString(rowOffsetExpr));

            if(rowOffsetExpr
               && Expression::evaluationTimes(rowOffsetExpr)[EvaluationTime::Translate]
               && info.offset->regType() == Register::Type::Literal)
            {
                info.offset
                    = Register::Value::Literal(getUnsignedInt(evaluate(rowOffsetExpr))
                                               + getUnsignedInt(info.offset->getLiteralValue()));
                rowOffsetExpr.reset();
            }

            if(rowOffsetExpr)
            {
                auto unrolledRowOffsetExpr = info.rowOffsetReg->expression() + rowOffsetExpr;
                auto tmp = info.rowOffsetReg->placeholder(Register::Type::Vector, {});
                co_yield generate(
                    tmp, convert(info.rowOffsetReg->variableType(), unrolledRowOffsetExpr));
                info.rowOffsetReg = tmp;
            }
            else if(preserveOffset)
            {
                auto tmp = info.rowOffsetReg->placeholder(Register::Type::Vector, {});
                co_yield m_context->copier()->copy(tmp, info.rowOffsetReg);
                info.rowOffsetReg = tmp;
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::generateStride(
            Register::ValuePtr& stride, RegisterExpressionAttributes& attrs, int tag, int dimension)
        {
            auto strideTag = m_graph->mapper.get<Stride>(tag, dimension);
            if(strideTag >= 0)
            {
                auto [strideExpr, strideAttributes]
                    = m_context->registerTagManager()->getExpression(strideTag);

                attrs = strideAttributes;

                if(!Expression::evaluationTimes(strideExpr)[EvaluationTime::Translate])
                {
                    stride = Register::Value::Placeholder(
                        m_context, Register::Type::Vector, strideAttributes.dataType, 1);
                    stride->setName("Stride");
                }
                else
                {
                    stride = nullptr;
                }

                co_yield generate(stride, strideExpr);
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
            auto buffer = m_graph->mapper.get(
                tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});

            auto info = fmt::format("KernelGraph::LoadStoreTileGenerator::ComputeIndex({}): "
                                    "target {} increment {} base {} offset {} stride {} buffer {}",
                                    tag,
                                    target,
                                    increment,
                                    base,
                                    offset,
                                    stride,
                                    buffer);
            Log::debug(info);
            co_yield Instruction::Comment(info);

            // TODO: Design a better way of binding storage to coordinates
            auto maybeLDS = m_graph->coordinates.get<LDS>(target);
            if(maybeLDS)
            {
                // If target is LDS; it might be a duplicated LDS
                // node.  For the purposes of computing indexes,
                // use the parent LDS as the target instead.
                namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                auto maybeParentLDS = only(
                    m_graph->coordinates.getOutputNodeIndices(target, CT::isEdge<Duplicate>));
                if(maybeParentLDS)
                    target = *maybeParentLDS;
            }
            maybeLDS = m_graph->coordinates.get<LDS>(target);

            auto isTransposed
                = m_graph->coordinates
                      .findNodes(target,
                                 [&](int tag) -> bool {
                                     auto maybeAdhoc = m_graph->coordinates.get<Adhoc>(tag);
                                     return maybeAdhoc
                                            && maybeAdhoc->name() == "Adhoc.transpose.simdsPerWave";
                                 })
                      .to<std::vector>()
                      .size()
                  == 1;

            auto scope = m_context->getScopeManager();

            auto toBytes = [&](ExpressionPtr expr) -> ExpressionPtr {
                uint numBits = DataTypeInfo::Get(ci.valueType).elementBits;

                // TODO: This would be a good place to add a GPU
                // assert.  If numBits is not a multiple of 8, assert
                // that (expr * numBits) is a multiple of 8.
                Log::debug("  toBytes: {}: numBits {}", toString(ci.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            // Set the zero-coordinates to zero
            auto fullStop  = [&](int tag) { return tag == increment; };
            auto direction = ci.forward ? Graph::Direction::Upstream : Graph::Direction::Downstream;
            auto [required, path] = findRequiredCoordinates(target, direction, fullStop, *m_graph);

            for(auto tag : required)
                if((tag != increment) && (!coords.hasCoordinate(tag)))
                    coords.setCoordinate(tag, L(0u));

            // Set the increment coordinate to zero if it doesn't
            // already have a value
            bool initializeIncrement = !coords.hasPath({target}, ci.forward);
            if(initializeIncrement)
            {
                coords.setCoordinate(increment, L(0u));
            }

            // Compute an offset address if we don't have an
            // associated base address to inherit from
            if(base < 0 && offset > 0)
            {
                auto offsetType = Register::Type::Vector;
                if(ci.isDirect2LDS)
                    offsetType = Register::Type::Scalar;
                auto offsetReg = tagger->getRegister(offset, offsetType, ci.offsetType, 1);
                offsetReg->setName(concatenate("Offset", tag));
                scope->addRegister(offset);

                auto indexExpr
                    = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

                auto const& typeInfo = DataTypeInfo::Get(ci.valueType);
                auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

                auto const& arch = m_context->targetArchitecture();
                const auto  needsPadding
                    = numBits == 6 && isTransposed
                      && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

                ExpressionPtr paddingBytes{L(0u)};
                if(needsPadding && maybeLDS)
                {
                    uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                    auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                    paddingBytes           = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
                }

                co_yield Instruction::Comment(
                    fmt::format("  Offset({}): indexExpr: {}", offset, toString(indexExpr)));
                co_yield Instruction::Comment(
                    fmt::format("  Offset({}): paddingBytes: {}", offset, toString(paddingBytes)));

                auto expr = toBytes(indexExpr) + paddingBytes;

                if(ci.isDirect2LDS)
                    expr = makeScalar(expr);

                co_yield generate(offsetReg, convert(offsetReg->variableType(), expr));
                offsetReg->setReadOnly();
            }
            else
            {
                m_baseOffsets.insert_or_assign(offset, base);
            }

            // Compute a stride
            if(stride > 0)
            {
                auto indexExpr = ci.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                            : coords.reverseStride(increment, L(1), {target})[0];

                // We have to manually invoke m_fastArith here since it can't traverse into the
                // RegisterTagManager.
                // TODO: Revisit storing expressions in the RegisterTagManager.
                bool unitStride = false;
                if(Expression::evaluationTimes(indexExpr)[EvaluationTime::Translate])
                {
                    if(getUnsignedInt(evaluate(indexExpr)) == 1u)
                        unitStride = true;
                }

                uint          elementBlockSize = 0;
                ExpressionPtr elementBlockStride;
                ExpressionPtr trLoadPairStride;
                ExpressionPtr elementBlockStridePaddingBytes{L(0u)};
                ExpressionPtr trLoadPairStridePaddingBytes{L(0u)};
                ExpressionPtr indexExprPaddingBytes{L(0u)};

                auto const& typeInfo = DataTypeInfo::Get(ci.valueType);
                auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

                if(numBits == 16 || numBits == 8 || numBits == 6 || numBits == 4)
                {
                    auto [elementBlockNumber, elementBlockIndex]
                        = getElementBlockValues(*m_graph, target, isTransposed);

                    elementBlockSize = elementBlockIndex;

                    auto const& arch = m_context->targetArchitecture();
                    if(isTransposed)
                    {
                        // See addLoadWaveTileCTF8F6F4 in LowerTile.cpp
                        const auto wfs = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
                        uint const numVBlocks
                            = wfs == 64 ? (numBits == 8 ? 2 : 1) : (numBits == 8 ? 4 : 2);
                        elementBlockSize = (elementBlockNumber / numVBlocks) * elementBlockSize;
                    }
                    AssertFatal(
                        elementBlockSize > 0, "Invalid elementBlockSize: ", elementBlockSize);

                    const auto needsPadding
                        = numBits == 6 && isTransposed
                          && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

                    // Padding is added after every 16 elements, thus for F6 datatypes that will
                    // be transpose loaded from LDS elementBlockSize is set to 16 instead of 32.
                    if(needsPadding)
                    {
                        elementBlockSize = 16;
                    }

                    elementBlockStride
                        = ci.forward
                              ? coords.forwardStride(increment, L(elementBlockSize), {target})[0]
                              : coords.reverseStride(increment, L(elementBlockSize), {target})[0];

                    uint elementsPerTrLoad = elementBlockIndex;
                    trLoadPairStride
                        = ci.forward
                              ? coords.forwardStride(increment, L(elementsPerTrLoad), {target})[0]
                              : coords.reverseStride(increment, L(elementsPerTrLoad), {target})[0];

                    if(needsPadding && maybeLDS)
                    {
                        uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                        auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                        elementBlockStridePaddingBytes
                            = elementBlockStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                        trLoadPairStridePaddingBytes
                            = trLoadPairStride / L(elementsPerTrLoad) * L(extraLdsBytes);
                        indexExprPaddingBytes = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
                    }
                }

                co_yield Instruction::Comment(
                    fmt::format("  Stride({}): indexExpr: {}", stride, toString(indexExpr)));
                co_yield Instruction::Comment(fmt::format("  Stride({}): indexExprPaddingBytes: {}",
                                                          stride,
                                                          toString(indexExprPaddingBytes)));
                co_yield Instruction::Comment(
                    fmt::format("  Stride({}): unitStride: {} vgprBlockSize: {}",
                                stride,
                                unitStride,
                                elementBlockSize));
                co_yield Instruction::Comment(fmt::format(
                    "  Stride({}): elementBlockStride: {} elementBlockStridePaddingBytes: {}",
                    stride,
                    toString(elementBlockStride),
                    toString(elementBlockStridePaddingBytes)));
                co_yield Instruction::Comment(fmt::format("  Stride({}): trLoadPairStride:  {} "
                                                          "trLoadPairStridePaddingBytes: {}",
                                                          stride,
                                                          toString(trLoadPairStride),
                                                          toString(trLoadPairStridePaddingBytes)));

                tagger->addExpression(stride,
                                      m_fastArith(toBytes(indexExpr) + indexExprPaddingBytes),
                                      {ci.strideType,
                                       unitStride,
                                       elementBlockSize,
                                       toBytes(elementBlockStride) + elementBlockStridePaddingBytes,
                                       toBytes(trLoadPairStride) + trLoadPairStridePaddingBytes});
                scope->addRegister(stride);
            }

            // Create a buffer descriptor
            if(buffer > 0)
            {
                auto user = m_graph->coordinates.get<User>(target);
                if(user && !tagger->hasRegister(buffer))
                {
                    AssertFatal(
                        user->size, "Invalid User dimension: missing size.", ShowValue(target));

                    auto bufferReg = tagger->getRegister(
                        buffer, Register::Type::Scalar, {DataType::None, PointerType::Buffer}, 1);
                    bufferReg->setName(concatenate("Buffer", buffer));
                    if(bufferReg->allocationState() == Register::AllocationState::Unallocated)
                    {
                        Register::ValuePtr basePointer;
                        auto               bufDesc = BufferDescriptor(bufferReg, m_context);
                        co_yield m_context->argLoader()->getValue(user->argumentName, basePointer);
                        if(user->offset && !Expression::canEvaluateTo(0u, user->offset))
                        {
                            Register::ValuePtr tmpRegister;
                            co_yield generate(tmpRegister,
                                              simplify(basePointer->expression() + user->offset));
                            co_yield bufDesc.setBasePointer(tmpRegister);
                        }
                        else
                        {
                            co_yield bufDesc.setBasePointer(basePointer);
                        }

                        co_yield bufDesc.setDefaultOpts();
                        Register::ValuePtr limitValue;
                        co_yield generate(limitValue, toBytes(user->size));
                        // TODO: Handle sizes larger than 32 bits
                        auto limit = (limitValue->regType() == Register::Type::Literal)
                                         ? limitValue
                                         : limitValue->subset({0});
                        limit->setVariableType(DataType::UInt32);
                        co_yield bufDesc.setSize(limit);
                    }
                    scope->addRegister(buffer);
                }
            }
        }

        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction> LoadStoreTileGenerator::moveTileDirect2LDS(
            LoadStoreTileInfo& info, int numBytes, bool setM0, Register::ValuePtr readAddr)
        {
            //TODO: enable to load 12 bytes
            if(m_context->targetArchitecture().HasCapability(GPUCapability::HasWiderDirectToLds))
                AssertFatal(numBytes == 1 || numBytes == 2 || numBytes == 4 || numBytes == 16,
                            ShowValue(numBytes));
            else
                AssertFatal(numBytes == 1 || numBytes == 2 || numBytes == 4, ShowValue(numBytes));
            auto m0 = m_context->getM0();
            AssertFatal(info.ldsWriteStride == m_workgroupSizeTotal * numBytes);

            if(setM0)
            {
                co_yield generate(m0, info.data->expression());
            }
            else
            {
                co_yield generate(m0, m0->expression() + Expression::literal(info.ldsWriteStride));
            }

            if(info.offset->regType() == Register::Type::Literal
               && !m_context->targetArchitecture().isSupportedConstantValue(info.offset))
            {
                const auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());
                info.offset            = Register::Value::Placeholder(
                    m_context, Register::Type::Scalar, DataType::UInt32, 1);
                info.offset->setName("Offset");
                co_yield generate(info.offset, Expression::literal(offsetValue))
                    .map(AddComment(fmt::format("{} is not a supported value!", offsetValue)));
            }

            co_yield m_context->mem()->moveData<Dir>(info.kind,
                                                     readAddr,
                                                     nullptr,
                                                     info.offset,
                                                     numBytes,
                                                     "",
                                                     false,
                                                     info.bufDesc,
                                                     info.bufOpts);
        }

        /**
         * @brief Load or Store a tile where all of the strides are literal values.
         *
         * @tparam Dir
         * @param info
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction>
            LoadStoreTileGenerator::moveTileLiteralStrides(LoadStoreTileInfo& info)
        {
            Log::debug("KernelGraph::LoadStoreTileGenerator::moveTileLiteralStrides<{}>",
                       toString(Dir));

            co_yield Instruction::Comment(toString(info));

            auto typeInfo = DataTypeInfo::Get(info.data->variableType().dataType);

            // If all of the strides are literals, we can load everything using offsets
            // without using a runtime counter
            auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());
            auto rowStride   = getUnsignedInt(info.rowStrideReg->getLiteralValue());
            auto colStride   = getUnsignedInt(info.colStrideReg->getLiteralValue());

            if(!info.isTransposedTile && info.colStrideAttributes.unitStride)
            {
                uint numVGPRBlocks = 1;
                if(info.colStrideAttributes.elementBlockSize > 0
                   && (info.n * info.packedAmount) > info.colStrideAttributes.elementBlockSize)
                {
                    AssertFatal((info.n * info.packedAmount)
                                    % info.colStrideAttributes.elementBlockSize
                                == 0);
                    numVGPRBlocks
                        = (info.n * info.packedAmount) / info.colStrideAttributes.elementBlockSize;
                }

                // Segmented
                auto bitsPerMove  = info.n * info.elementBits / numVGPRBlocks;
                auto bytesPerMove = bitsPerMove / 8u;

                // packed
                auto elementsPerMove = bitsPerMove / typeInfo.elementBits;

                auto elementBlockStride
                    = (numVGPRBlocks > 1)
                          ? getUnsignedInt(evaluate(info.colStrideAttributes.elementBlockStride))
                          : 0;

                auto msg = concatenate(ShowValue(info.m),
                                       ShowValue(info.n),
                                       ShowValue(elementsPerMove),
                                       ShowValue(bytesPerMove),
                                       ShowValue(rowStride),
                                       ShowValue(colStride),
                                       ShowValue(info.colStrideAttributes.elementBlockSize),
                                       ShowValue(numVGPRBlocks),
                                       ShowValue(elementBlockStride));

                co_yield Instruction::Comment(msg);
                Log::debug(msg);

                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint r = 0; r < numVGPRBlocks; ++r)
                    {
                        auto start = (i * numVGPRBlocks + r) * elementsPerMove;
                        auto stop  = (i * numVGPRBlocks + r + 1) * elementsPerMove;
                        if(info.bufOpts.lds)
                        {
                            co_yield moveTileDirect2LDS<Dir>(
                                info, bytesPerMove, (i == 0 && r == 0), info.rowOffsetReg);
                        }
                        else
                        {
                            co_yield m_context->mem()->moveData<Dir>(
                                info.kind,
                                info.rowOffsetReg,
                                info.data->element(Generated(iota(start, stop))),
                                Register::Value::Literal(offsetValue + r * elementBlockStride),
                                bytesPerMove,
                                "",
                                false,
                                info.bufDesc,
                                info.bufOpts);
                        }
                    }
                    offsetValue += rowStride;
                }
            }
            else if(info.isTransposedTile)
            {
                auto packedVarType = typeInfo.packedVariableType().value_or(typeInfo.variableType);
                auto packedInfo    = DataTypeInfo::Get(packedVarType);

                auto individualBits = info.elementBits / info.packedAmount;

                auto const& arch              = m_context->targetArchitecture();
                const auto  bitsPerTrLoad     = bitsPerTransposeLoad(arch, individualBits);
                const auto  extraLDSBytes     = extraLDSBytesPerElementBlock(arch, individualBits);
                const auto  bytesPerTrLoad    = bitsPerTrLoad / 8;
                const auto  elementsPerTrLoad = bitsPerTrLoad / individualBits;
                const auto  numVGPRsPerLoad   = bitsPerTrLoad / Register::bitsPerRegister;
                const auto  numVGPRBlocks     = numVGPRsPerLoad / packedInfo.registerCount;
                const auto  numTrLoads        = (info.n * info.packedAmount) / elementsPerTrLoad;
                const auto  elementBlockStride
                    = getUnsignedInt(evaluate(info.colStrideAttributes.elementBlockStride));
                const auto trLoadPairStride
                    = getUnsignedInt(evaluate(info.colStrideAttributes.trLoadPairStride));

                AssertFatal((info.n * info.packedAmount) % elementsPerTrLoad == 0,
                            "WaveTileN must be multiple of the number of elements loaded by each "
                            "transpose load!",
                            ShowValue(info.n),
                            ShowValue(info.packedAmount),
                            ShowValue(elementsPerTrLoad));
                AssertFatal(numTrLoads % 2 == 0, "Transpose loads must be executed in pairs!");

                Log::debug("  M {} N {} elementsPerTrLoad {} bytesPerTrLoad {} extraLDSBytes {} "
                           "elementBits {} "
                           "rowStride {} colStride {} vgprPerLoad {} numTrLoads {}",
                           info.m,
                           info.n,
                           elementsPerTrLoad,
                           bytesPerTrLoad,
                           extraLDSBytes,
                           individualBits,
                           rowStride,
                           colStride,
                           numVGPRsPerLoad,
                           numTrLoads);

                auto msg = concatenate(ShowValue(info.m),
                                       ShowValue(info.n),
                                       ShowValue(elementsPerTrLoad),
                                       ShowValue(bytesPerTrLoad),
                                       ShowValue(rowStride),
                                       ShowValue(colStride),
                                       ShowValue(info.colStrideAttributes.elementBlockSize),
                                       ShowValue(numVGPRBlocks),
                                       ShowValue(numTrLoads),
                                       ShowValue(trLoadPairStride),
                                       ShowValue(elementBlockStride));

                co_yield Instruction::Comment(msg);
                Log::debug(msg);

                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint64_t j = 0; j < numTrLoads; ++j)
                    {
                        auto start = (i * numTrLoads + (j + 0)) * numVGPRBlocks;
                        auto stop  = (i * numTrLoads + (j + 1)) * numVGPRBlocks;
                        auto trLoadOffset
                            = (j % 2) * trLoadPairStride + (j / 2) * elementBlockStride;
                        co_yield m_context->mem()->transposeLoadLocal(
                            info.data->element(Generated(iota(start, stop))),
                            info.rowOffsetReg,
                            offsetValue + trLoadOffset,
                            bytesPerTrLoad + extraLDSBytes,
                            individualBits);
                    }
                    offsetValue += rowStride;
                }
            }
            else
            {
                for(uint64_t i = 0; i < info.m; ++i)
                {
                    for(uint64_t j = 0; j < info.n; ++j)
                    {
                        if(info.bufOpts.lds)
                        {
                            co_yield moveTileDirect2LDS<Dir>(info,
                                                             CeilDivide(info.elementBits, 8u),
                                                             (i == 0 && j == 0),
                                                             info.rowOffsetReg);
                        }
                        else
                        {
                            co_yield m_context->mem()->moveData<Dir>(
                                info.kind,
                                info.rowOffsetReg,
                                info.data->element(
                                    {static_cast<int>((i * info.n + j) / info.packedAmount)}),
                                Register::Value::Literal(offsetValue + j * colStride),
                                CeilDivide(info.elementBits, 8u),
                                "",
                                j % info.packedAmount == 1,
                                info.bufDesc);
                        }
                    }
                    offsetValue += rowStride;
                }
            }
        }

        /**
         * @brief Load or store a tile where the column stride is known to be a single element, but
         *        the row stride is only known at runtime.
         *
         * @tparam Dir
         * @param info
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction> LoadStoreTileGenerator::moveTileColStrideOne(LoadStoreTileInfo& info)
        {
            Log::debug("KernelGraph::LoadStoreTileGenerator::moveTileColStrideOne<{}>",
                       toString(Dir));

            auto unsegmentedDataType = DataTypeInfo::Get(info.data->variableType().dataType);
            auto segmentTypeInfo     = DataTypeInfo::Get(unsegmentedDataType.segmentVariableType);

            AssertFatal(info.offset->regType() == Register::Type::Literal,
                        ShowValue(info.offset->description()));
            auto offsetValue = getUnsignedInt(info.offset->getLiteralValue());

            uint numVGPRBlocks = 1;
            if(info.colStrideAttributes.elementBlockSize > 0
               && (info.n * unsegmentedDataType.packing)
                      > info.colStrideAttributes.elementBlockSize)
            {
                AssertFatal((info.n * unsegmentedDataType.packing)
                                % info.colStrideAttributes.elementBlockSize
                            == 0);
                numVGPRBlocks = (info.n * unsegmentedDataType.packing)
                                / info.colStrideAttributes.elementBlockSize;
            }

            // Segmented
            auto bitsPerMove  = info.n * info.elementBits / numVGPRBlocks;
            auto bytesPerMove = bitsPerMove / 8u;

            // Unsegmented
            auto elementsPerMove = bitsPerMove / unsegmentedDataType.elementBits;

            co_yield Instruction::Comment(
                fmt::format("  M {} N {} elementsPerMove {} bytesPerMove {} rowStride {} colStride "
                            "{} vgprBlockSize {} numVGPRBlocks {}",
                            info.m,
                            info.n,
                            elementsPerMove,
                            bytesPerMove,
                            toString(info.rowStrideReg->expression()),
                            toString(info.colStrideReg->expression()),
                            info.colStrideAttributes.elementBlockSize,
                            numVGPRBlocks));

            for(uint64_t i = 0; i < info.m; ++i)
            {
                for(uint r = 0; r < numVGPRBlocks; ++r)
                {
                    auto start = (i * numVGPRBlocks + r) * elementsPerMove;
                    auto stop  = (i * numVGPRBlocks + r + 1) * elementsPerMove;
                    if(info.bufOpts.lds)
                    {
                        if(r * bytesPerMove > 0)
                        {
                            const auto newOffsetValue = offsetValue + r * bytesPerMove;
                            co_yield generate(info.offset, Expression::literal(newOffsetValue));
                        }

                        co_yield moveTileDirect2LDS<Dir>(
                            info, bytesPerMove, (i == 0 && r == 0), info.rowOffsetReg);
                    }
                    else
                    {
                        co_yield m_context->mem()->moveData<Dir>(
                            info.kind,
                            info.rowOffsetReg,
                            info.data->element(Generated(iota(start, stop))),
                            Register::Value::Literal(offsetValue + r * bytesPerMove),
                            bytesPerMove,
                            "",
                            false,
                            info.bufDesc,
                            info.bufOpts);
                    }
                }

                if(i < info.m - 1)
                {
                    co_yield generate(info.rowOffsetReg,
                                      info.rowOffsetReg->expression()
                                          + info.rowStrideReg->subset({0})->expression());
                }
            }
        }

        /**
         * @brief Load or store a tile where the strides are only known at runtime.
         *
         * @tparam Dir
         * @param info
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction>
            LoadStoreTileGenerator::moveTileRuntimeStrides(LoadStoreTileInfo& info)
        {
            Log::debug("KernelGraph::LoadStoreTileGenerator::moveTileRuntimeStrides<{}>",
                       toString(Dir));

            AssertFatal(!info.bufOpts.lds);

            auto colOffsetReg = info.rowOffsetReg->placeholder();

            for(uint64_t i = 0; i < info.m; ++i)
            {
                co_yield m_context->copier()->copy(colOffsetReg, info.rowOffsetReg);

                for(uint64_t j = 0; j < info.n; ++j)
                {
                    co_yield m_context->mem()->moveData<Dir>(
                        info.kind,
                        colOffsetReg->subset({0}),
                        info.data->element(
                            {static_cast<int>((i * info.n + j) / info.packedAmount)}),
                        info.offset,
                        CeilDivide(info.elementBits, 8u),
                        "",
                        j % info.packedAmount == 1,
                        info.bufDesc,
                        info.bufOpts);

                    if(j < info.n - 1)
                    {
                        co_yield generate(colOffsetReg,
                                          colOffsetReg->expression()
                                              + info.colStrideReg->subset({0})->expression());
                    }
                }

                if(i < info.m - 1)
                {
                    co_yield generate(info.rowOffsetReg,
                                      info.rowOffsetReg->expression()
                                          + info.rowStrideReg->subset({0})->expression());
                }
            }
        }

        /**
         * @brief Load or store a tile
         *
         * @param kind The kind of memory instruction to use
         * @param m Number of rows in the tile
         * @param n Number of columns in the tile
         * @param dataType The type of the data being loaded
         * @param isTransposedTile if tile needs to be transposed
         * @param tag The tag of the control graph node generating the load or store
         * @param vgpr The registers to store the data in (null is loading)
         * @param offset Offset from the starting index
         * @param coords Transformer object
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction> LoadStoreTileGenerator::moveTile(MemoryInstructions::MemoryKind kind,
                                                                uint64_t                       m,
                                                                uint64_t                       n,
                                                                VariableType             varType,
                                                                int                      tag,
                                                                Register::ValuePtr       vgpr,
                                                                Register::ValuePtr       offset,
                                                                Transformer&             coords,
                                                                BufferInstructionOptions bufOpts,
                                                                bool isTransposedTile,
                                                                bool isPadded)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::moveTile<{}>({}) {}x{}",
                toString(Dir),
                tag,
                m,
                n);

            LoadStoreTileInfo info;
            info.kind             = kind;
            info.m                = m;
            info.n                = n;
            info.offset           = offset;
            info.bufOpts          = bufOpts;
            info.isTransposedTile = isTransposedTile;

            Register::ValuePtr finalVGPR;

            if(!info.offset)
            {
                info.offset = Register::Value::Literal(0u);
            }

            if(kind == MemoryInstructions::MemoryKind::Buffer
               || kind == MemoryInstructions::MemoryKind::Buffer2LDS)
            {
                info.bufDesc = getBufferDesc(tag);
            }

            auto const& varTypeInfo = DataTypeInfo::Get(varType);

            co_yield Instruction::Comment(
                concatenate(ShowValue(varTypeInfo.elementBits), ShowValue(varTypeInfo.packing)));

            info.elementBits = (uint)varTypeInfo.elementBits;

            if(m > 1)
                co_yield generateStride(info.rowStrideReg, info.rowStrideAttributes, tag, 0);
            else
                info.rowStrideReg = Register::Value::Literal(0u);
            co_yield generateStride(info.colStrideReg, info.colStrideAttributes, tag, 1);

            AssertFatal(info.rowStrideReg, "Invalid row stride register.");
            AssertFatal(info.colStrideReg, "Invalid col stride register.");

            bool colStrideIsLiteral = (info.colStrideReg->regType() == Register::Type::Literal);

            bool allStridesAreLiteral
                = (info.rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && info.offset->regType() == Register::Type::Literal);
            bool colStrideIsOne = colStrideIsLiteral && info.colStrideAttributes.unitStride;

            if(Dir == MemoryInstructions::MemoryDirection::Load)
            {
                auto macTileTag = m_graph->mapper.get<MacroTile>(tag);

                macTileTag
                    = only(m_graph->coordinates.getInputNodeIndices(macTileTag, CT::isEdge<View>))
                          .value_or(macTileTag);

                auto packedVariableType = varTypeInfo.packedVariableType();

                auto allocOptions = Register::AllocationOptions::FullyContiguous();

                auto elementBits = DataTypeInfo::Get(varTypeInfo.segmentVariableType).elementBits;
                if(elementBits == 6 && isPadded && !info.isTransposedTile)
                {
                    auto registerCount = varTypeInfo.registerCount;
                    allocOptions = {.contiguousChunkWidth = int(registerCount), .alignment = 2};
                    co_yield Instruction::Comment(
                        concatenate("Allocation options: ", allocOptions));
                }

                auto tmpl = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, varType, m * n, allocOptions);
                tmpl->setName("tmpl");

                if(kind == MemoryInstructions::MemoryKind::Buffer2LDS)
                {
                    info.data        = vgpr;
                    info.bufOpts.lds = 1;

                    // get lds write stride
                    Register::ValuePtr           ldsWriteStrideRegister;
                    RegisterExpressionAttributes _ignore;
                    co_yield generateStride(ldsWriteStrideRegister, _ignore, tag, 2);
                    auto ldsStride      = getUnsignedInt(ldsWriteStrideRegister->getLiteralValue());
                    info.ldsWriteStride = ldsStride;
                }
                else if(m_context->registerTagManager()->hasRegister(macTileTag))
                {
                    auto reg = m_context->registerTagManager()->getRegister(macTileTag);

                    if(!m_context->targetArchitecture().HasCapability(
                           GPUCapability::ArchAccUnifiedRegs)
                       && reg->regType() != Register::Type::Vector)
                    {
                        // If no unified acc/vgpr registers, create a temporary vgpr register.
                        // The result of the load will be copied into finalVGPR after the load
                        // has been performed.
                        info.data = tmpl;
                        finalVGPR = reg;
                    }
                    else
                    {
                        info.data = reg;
                    }
                }
                else
                {
                    info.data = m_context->registerTagManager()->getRegister(macTileTag, tmpl);

                    auto viewTileTags
                        = m_graph->coordinates.getOutputNodeIndices(macTileTag, CT::isEdge<View>);
                    for(auto viewTileTag : viewTileTags)
                    {
                        m_context->registerTagManager()->addRegister(viewTileTag, info.data);
                    }
                }

                Log::debug("  tag {} tile coord {} registers {}",
                           tag,
                           macTileTag,
                           info.data->description());
            }
            else
            {
                if(!m_context->targetArchitecture().HasCapability(
                       GPUCapability::ArchAccUnifiedRegs))
                {
                    co_yield m_context->copier()->ensureType(vgpr, vgpr, Register::Type::Vector);
                }

                // Convert the data to the expected datatype
                auto existingVarType = vgpr->variableType();
                if(existingVarType != varType
                   && DataTypeInfo::Get(existingVarType).segmentVariableType != varType)
                {
                    co_yield Instruction::Comment("Convert in LSTGen");
                    co_yield m_context->copier()->ensureType(vgpr, vgpr, Register::Type::Vector);
                    Register::ValuePtr result;
                    co_yield generate(result, convert(varType, vgpr->expression()));
                    info.data = result;
                }
                else
                {
                    info.data = vgpr;
                }
            }

            info.packedAmount = DataTypeInfo::Get(info.data->variableType()).packing;

            co_yield Instruction::Comment(
                ShowValue(Dir) + toString(info)
                + concatenate(ShowValue(allStridesAreLiteral), ShowValue(colStrideIsOne)));

            // Get the values from the associated ComputeIndex node
            co_yield getOffset(info, coords, tag, !allStridesAreLiteral && info.m > 1);
            AssertFatal(info.rowOffsetReg, "Invalid row offset register.");

            if(kind == MemoryInstructions::MemoryKind::Buffer2LDS)
            {
                co_yield getOffset(
                    info, coords, tag, false /* preserveOffset */, true /* direct2LDS */);

                // set global read offset
            }

            if(allStridesAreLiteral)
            {
                co_yield moveTileLiteralStrides<Dir>(info);
            }
            else if(colStrideIsOne)
            {
                co_yield moveTileColStrideOne<Dir>(info);
            }
            else
            {
                co_yield moveTileRuntimeStrides<Dir>(info);
            }

            if(finalVGPR)
            {
                co_yield m_context->copier()->copy(finalVGPR, info.data);
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileVGPR(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileVGPR()");
            co_yield Instruction::Comment("GEN: loadMacroTileVGPRCI " + toString(load.varType));

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(n % packing == 0,
                        ShowValue(m),
                        ShowValue(n),
                        ShowValue(packing),
                        ShowValue(load.varType));
            n /= packing;

            AssertFatal(m > 0 && n > 0, "Invalid/unknown subtile size dimensions");

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                load.varType,
                tag,
                nullptr,
                nullptr,
                coords,
                {},
                /*isTransposedTile=*/false,
                /*isPadded=*/tile.paddingBytes() > 0);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileLDS(int                tag,
                                                                        LoadLDSTile const& load,
                                                                        Transformer        coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileLDS: OP {} LDS {} MacroTile {}",
                tag,
                ldsTag,
                tileTag);
            co_yield_(Instruction::Comment(concatenate(
                "GEN: loadMacroTileLDS OP ", tag, " LDS ", ldsTag, "MacroTile ", tileTag)));

            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Local,
                m,
                n,
                load.varType,
                tag,
                nullptr,
                ldsOffset,
                coords)
                .map(MemoryInstructions::addExtraSrc(ldsAllocation));
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileDirect2LDS(
            int tag, LoadTileDirect2LDS const& load, Transformer coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);
            auto dataType        = load.varType;

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "loadMacroTileDirect2LDS: OP {} LDS {} MacroTile {}",
                                               tag,
                                               ldsTag,
                                               tileTag);
            co_yield Instruction::Comment(concatenate(
                "GEN: loadMacroTileDirect2LDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag));

            auto numElements = product(tile.subTileSizes) * m_workgroupSizeTotal;

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(m_context, dataType, numElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto offsetValue = ldsAllocation->getLDSAllocation()->offset();

            auto ldsOffset = Register::Value::Literal(offsetValue);

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(n % packing == 0,
                        ShowValue(m),
                        ShowValue(n),
                        ShowValue(packing),
                        ShowValue(load.varType));
            n /= packing;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer2LDS,
                m,
                n,
                dataType,
                tag,
                ldsOffset,
                nullptr,
                coords,
                {})
                .map(MemoryInstructions::addExtraDst(ldsAllocation));
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVELDS(int                tag,
                                                                            LoadLDSTile const& load,
                                                                            Transformer coords)
        {
            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "loadMacroTileWAVELDS: OP {} LDS {} WaveTile {}",
                                               tag,
                                               ldsTag,
                                               waveTileTag);
            co_yield_(Instruction::Comment(concatenate(
                "GEN: loadMacroTileWAVELDS OP ", tag, " LDS ", ldsTag, " WaveTile ", waveTileTag)));

            ldsTag = only(m_graph->coordinates.getOutputNodeIndices(ldsTag, CT::isEdge<View>))
                         .value_or(ldsTag);
            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            uint numElements       = waveTile.sizes[0] * waveTile.sizes[1];
            auto [_, lane]         = m_graph->getDimension<Lane>(tag);
            auto activeLanesInWave = getUnsignedInt(evaluate(lane.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(numElements % (activeLanesInWave * packing) == 0,
                        ShowValue(numElements),
                        ShowValue(activeLanesInWave),
                        ShowValue(packing),
                        ShowValue(activeLanesInWave * packing));
            uint numVgpr = numElements / (activeLanesInWave * packing);
            AssertFatal(numVgpr > 0, "Invalid load dimensions.");

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Local,
                1,
                numVgpr,
                load.varType,
                tag,
                nullptr,
                ldsOffset,
                coords,
                {},
                load.isTransposedTile)
                .map(MemoryInstructions::addExtraSrc(ldsAllocation));
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVE(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVE: OP {} WaveTile {}",
                tag,
                waveTileTag);
            co_yield Instruction::Comment(
                concatenate("GEN: loadMacroTileWAVE OP", tag, " WaveTile ", waveTileTag));

            uint numElements       = waveTile.sizes[0] * waveTile.sizes[1];
            auto [_, lane]         = m_graph->getDimension<Lane>(tag);
            auto activeLanesInWave = getUnsignedInt(evaluate(lane.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(numElements % (activeLanesInWave * packing) == 0,
                        ShowValue(numElements),
                        ShowValue(activeLanesInWave),
                        ShowValue(packing),
                        ShowValue(activeLanesInWave * packing),
                        ShowValue(load.varType));
            uint numVgpr = numElements / (activeLanesInWave * packing);
            AssertFatal(numVgpr > 0, "Invalid load dimensions.");

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                1,
                numVgpr,
                load.varType,
                tag,
                nullptr,
                nullptr,
                coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVECIACCUM(
            int tag, LoadTiled const& load, Transformer coords)

        {
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileWAVECIACCUM: OP {} WaveTile {}",
                tag,
                waveTileTag);
            co_yield Instruction::Comment(
                concatenate("GEN: loadMacroTileWAVECIACCUM OP ", tag, " WaveTile ", waveTileTag));

            uint numElements       = waveTile.sizes[0] * waveTile.sizes[1];
            auto [_, lane]         = m_graph->getDimension<Lane>(tag);
            auto activeLanesInWave = getUnsignedInt(evaluate(lane.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(numElements % (activeLanesInWave * packing) == 0,
                        ShowValue(numElements),
                        ShowValue(activeLanesInWave),
                        ShowValue(packing),
                        ShowValue(activeLanesInWave * packing));
            uint numVgpr = numElements / activeLanesInWave;
            AssertFatal(numVgpr > 0, "Invalid load dimensions.");

            auto [vgprBlockNumberTag, vgprBlockNumber]
                = m_graph->getDimension<VGPRBlockNumber>(tag, 0);
            auto [vgprBlockIndexTag, vgprBlockIndex]
                = m_graph->getDimension<VGPRBlockIndex>(tag, 0);

            AssertFatal(
                Expression::evaluationTimes(vgprBlockNumber.size)[EvaluationTime::Translate],
                "Could not determine VGPRBlockNumber size at translate-time.");
            AssertFatal(Expression::evaluationTimes(vgprBlockIndex.size)[EvaluationTime::Translate],
                        "Could not determine VGPRBlockIndex size at translate-time.");

            const auto m = getUnsignedInt(evaluate(vgprBlockNumber.size));
            auto       n = getUnsignedInt(evaluate(vgprBlockIndex.size));
            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                load.varType,
                tag,
                nullptr,
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
                co_yield loadMacroTileVGPR(tag, load, coords);
                break;
            case MemoryType::WAVE:
            case MemoryType::WAVE_SWIZZLE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileWAVE(tag, load, coords);
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
            case MemoryType::WAVE_SPLIT:
            case MemoryType::VGPR:
            case MemoryType::LDS:
                co_yield loadMacroTileLDS(tag, load, coords);
                break;
            case MemoryType::WAVE:
            case MemoryType::WAVE_SWIZZLE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                case LayoutType::MATRIX_B:
                    co_yield loadMacroTileWAVELDS(tag, load, coords);
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

        Generator<Instruction> LoadStoreTileGenerator::genLoadTileDirect2LDS(
            int tag, LoadTileDirect2LDS const& load, Transformer coords)
        {
            co_yield loadMacroTileDirect2LDS(tag, load, coords);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileLDS(int                 tag,
                                                                         StoreLDSTile const& store,
                                                                         Transformer         coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileLDS: OP {} LDS {} MacroTile {} "
                "layoutType {} memoryType {} paddingBytes {}",
                tag,
                ldsTag,
                tileTag,
                toString(tile.layoutType),
                toString(tile.memoryType),
                tile.paddingBytes());
            co_yield Instruction::Comment(concatenate(
                "GEN: storeMacroTileLDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag));

            auto packing = DataTypeInfo::Get(store.varType).packing;

            // Temporary register(s) that is used to copy the data from global memory to
            // local memory.
            auto vgpr    = m_context->registerTagManager()->getRegister(tileTag);
            auto varType = store.varType;

            auto numElements  = product(tile.subTileSizes) * m_workgroupSizeTotal;
            auto paddingBytes = tile.paddingBytes();
            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(
                    m_context, varType, numElements / packing, /*alignment*/ 4, paddingBytes);
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
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Local,
                m,
                n,
                varType,
                tag,
                vgpr,
                ldsOffset,
                coords,
                {},
                /*isTransposedTile*/ false,
                /*isPadded*/ paddingBytes > 0)
                .map(MemoryInstructions::addExtraDst(ldsAllocation));
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileVGPR(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileVGPR: OP {} MacroTile {}",
                tag,
                macTileTag);
            co_yield Instruction::Comment(
                concatenate("GEN: storeMacroTileVGPR OP ", tag, " MacroTile ", macTileTag));

            macTileTag
                = only(m_graph->coordinates.getOutputNodeIndices(macTileTag, CT::isEdge<View>))
                      .value_or(macTileTag);

            auto vgpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            auto packing = DataTypeInfo::Get(store.varType).packing;
            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                store.varType,
                tag,
                vgpr,
                nullptr,
                coords,
                store.bufOpts);
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVELDS(
            int tag, StoreLDSTile const& store, Transformer coords)
        {
            auto [ldsTag, lds]           = m_graph->getDimension<LDS>(tag);
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto macrotileNumElements    = product(macTile.sizes);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);
            uint waveTileNumElements     = waveTile.sizes[0] * waveTile.sizes[1];
            auto varType                 = store.varType;

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::storeMacroTileWAVELDS: OP {} LDS {} "
                "MacroTile {} WaveTile {}",
                tag,
                ldsTag,
                macTileTag,
                waveTileTag);
            co_yield Instruction::Comment(concatenate("GEN: storeMacroTileWAVELDS OP ",
                                                      tag,
                                                      " LDS ",
                                                      ldsTag,
                                                      " MacroTile ",
                                                      macTileTag,
                                                      " WaveTile ",
                                                      waveTileTag));

            auto packing = DataTypeInfo::Get(store.varType).packing;

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(
                    m_context, varType, macrotileNumElements / packing);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            auto ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto [_, lane]         = m_graph->getDimension<Lane>(tag);
            auto activeLanesInWave = getUnsignedInt(evaluate(lane.size));

            AssertFatal(waveTileNumElements % (activeLanesInWave * packing) == 0,
                        ShowValue(waveTileNumElements),
                        ShowValue(activeLanesInWave),
                        ShowValue(packing),
                        ShowValue(activeLanesInWave * packing));

            co_yield Instruction::Comment(concatenate(ShowValue(waveTile),
                                                      ShowValue(waveTileNumElements),
                                                      ShowValue(activeLanesInWave),
                                                      ShowValue(packing),
                                                      ShowValue(activeLanesInWave * packing),
                                                      ShowValue(store.varType)));

            uint numVgpr = waveTileNumElements / activeLanesInWave;

            auto agpr = m_context->registerTagManager()->getRegister(macTileTag);

            co_yield Instruction::Comment(concatenate(ShowValue(agpr->description()),
                                                      ShowValue(waveTile),
                                                      ShowValue(waveTileNumElements),
                                                      ShowValue(activeLanesInWave),
                                                      ShowValue(packing),
                                                      ShowValue(activeLanesInWave * packing)));

            auto [vgprBlockNumberTag, vgprBlockNumber]
                = m_graph->getDimension<VGPRBlockNumber>(tag, 0);
            auto [vgprBlockIndexTag, vgprBlockIndex]
                = m_graph->getDimension<VGPRBlockIndex>(tag, 0);

            AssertFatal(
                Expression::evaluationTimes(vgprBlockNumber.size)[EvaluationTime::Translate],
                "Could not determine VGPRBlockNumber size at translate-time.");
            AssertFatal(Expression::evaluationTimes(vgprBlockIndex.size)[EvaluationTime::Translate],
                        "Could not determine VGPRBlockIndex size at translate-time.");

            const auto m = getUnsignedInt(evaluate(vgprBlockNumber.size));
            auto       n = getUnsignedInt(evaluate(vgprBlockIndex.size));

            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Local, m, n, varType, tag, agpr, ldsOffset, coords)
                .map(MemoryInstructions::addExtraDst(ldsAllocation));
        }

        Generator<Instruction> LoadStoreTileGenerator::storeMacroTileWAVE(int               tag,
                                                                          StoreTiled const& store,
                                                                          Transformer       coords)
        {
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "storeMacroTileWAVE: OP {} MacroTile {} WaveTile {}",
                                               tag,
                                               macTileTag,
                                               waveTileTag);
            co_yield Instruction::Comment(concatenate("GEN: storeMacroTileWAVE OP ",
                                                      tag,
                                                      " MacroTile ",
                                                      macTileTag,
                                                      " WaveTile ",
                                                      waveTileTag));

            uint numElements       = waveTile.sizes[0] * waveTile.sizes[1];
            auto [_, lane]         = m_graph->getDimension<Lane>(tag);
            auto activeLanesInWave = getUnsignedInt(evaluate(lane.size));

            auto packing = DataTypeInfo::Get(store.varType).packing;
            AssertFatal(numElements % (activeLanesInWave * packing) == 0,
                        ShowValue(numElements),
                        ShowValue(activeLanesInWave),
                        ShowValue(packing),
                        ShowValue(activeLanesInWave * packing));
            uint numValues = numElements / activeLanesInWave;
            AssertFatal(numValues > 0, "Invalid store dimensions.");

            auto [vgprBlockNumberTag, vgprBlockNumber]
                = m_graph->getDimension<VGPRBlockNumber>(tag, 0);
            auto [vgprBlockIndexTag, vgprBlockIndex]
                = m_graph->getDimension<VGPRBlockIndex>(tag, 0);

            AssertFatal(
                Expression::evaluationTimes(vgprBlockNumber.size)[EvaluationTime::Translate],
                "Could not determine VGPRBlockNumber size at translate-time.");
            AssertFatal(Expression::evaluationTimes(vgprBlockIndex.size)[EvaluationTime::Translate],
                        "Could not determine VGPRBlockIndex size at translate-time.");

            const auto m = getUnsignedInt(evaluate(vgprBlockNumber.size));
            auto       n = getUnsignedInt(evaluate(vgprBlockIndex.size));

            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            auto agpr = m_context->registerTagManager()->getRegister(macTileTag);

            auto packedType = DataTypeInfo::Get(store.varType).packedVariableType();
            if(packedType && agpr->variableType() == packedType.value())
            {
                auto packing = DataTypeInfo::Get(packedType.value()).packing;
                AssertFatal(agpr->registerCount() == (numValues / packing));
            }
            else
            {
                AssertFatal(agpr->registerCount() == numValues);
            }

            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(
                MemoryInstructions::MemoryKind::Buffer,
                m,
                n,
                store.varType,
                tag,
                agpr,
                nullptr,
                coords,
                store.bufOpts);
        }

        Generator<Instruction> LoadStoreTileGenerator::genStoreTile(int               tag,
                                                                    StoreTiled const& store,
                                                                    Transformer       coords)
        {
            auto [macTileTag, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::WAVE_SPLIT:
            case MemoryType::VGPR:
                co_yield storeMacroTileVGPR(tag, store, coords);
                break;
            case MemoryType::WAVE:
                co_yield storeMacroTileWAVE(tag, store, coords);
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
