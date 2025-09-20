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

#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AssignComputeIndex.hpp>
#include <rocRoller/KernelGraph/Transforms/LowerTile_details.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;
        namespace Expression = rocRoller::Expression;

        inline Expression::ExpressionPtr L(auto const& x)
        {
            return Expression::literal(x);
        }

        static std::pair<uint, uint>
            getElementBlockValues(KernelGraph const& graph, int target, const bool isTransposed)
        {
            namespace CT            = rocRoller::KernelGraph::CoordinateGraph;
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
                AssertFatal(Expression::evaluationTimes(
                                elementNumberX.size)[Expression::EvaluationTime::Translate],
                            "Could not determine ElementNumberX size at translate-time.\n",
                            ShowValue(elementNumberX));

                auto [elementNumberYTag, elementNumberY]
                    = graph.getDimension<ElementNumber>(opTag, 1);
                AssertFatal(Expression::evaluationTimes(
                                elementNumberY.size)[Expression::EvaluationTime::Translate],
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
                AssertFatal(Expression::evaluationTimes(
                                vgprBlockNumber.size)[Expression::EvaluationTime::Translate],
                            "Could not determine VGPRBlockNumber size at translate-time.\n",
                            ShowValue(vgprBlockNumber));

                auto [vgprBlockIndexTag, vgprBlockIndex]
                    = graph.getDimension<VGPRBlockIndex>(opTag, 0);
                AssertFatal(Expression::evaluationTimes(
                                vgprBlockIndex.size)[Expression::EvaluationTime::Translate],
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
                    AssertFatal(Expression::evaluationTimes(
                                    vgpr.size)[Expression::EvaluationTime::Translate],
                                "Could not determine VGPR size at translate-time.\n",
                                ShowValue(vgpr));
                    // Multiplying by elementBlockNumber here forces the use
                    // of the widest load/store possible
                    elementBlockIndex = elementBlockNumber * getUnsignedInt(evaluate(vgpr.size));
                }

                if((!LowerTileDetails::isTileOfSubDwordTypeWithNonContiguousVGPRBlocks(
                        dataType,
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

        int makeAssignBase(KernelGraph&        graph,
                           ComputeIndex const& ci,
                           const int           target,
                           const int           offset,
                           const bool          maybeLDS,
                           const bool          isTransposed,
                           const ContextPtr    context,
                           Transformer&        coords)
        {
            auto toBytes = [&](Expression::ExpressionPtr expr) -> Expression::ExpressionPtr {
                uint numBits = DataTypeInfo::Get(ci.valueType).elementBits;

                // TODO: This would be a good place to add a GPU
                // assert.  If numBits is not a multiple of 8, assert
                // that (expr * numBits) is a multiple of 8.
                Log::debug("  toBytes: {}: numBits {}", toString(ci.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            auto offsetRegisterType = Register::Type::Vector;
            if(ci.isDirect2LDS)
                offsetRegisterType = Register::Type::Scalar;

            auto indexExpr = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

            auto const& typeInfo = DataTypeInfo::Get(ci.valueType);
            auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

            auto const& arch = context->targetArchitecture();
            const auto  needsPadding
                = numBits == 6 && isTransposed
                  && arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes);

            Expression::ExpressionPtr paddingBytes{L(0u)};
            if(needsPadding && maybeLDS)
            {
                uint elementsPerTrLoad = bitsPerTransposeLoad(arch, numBits) / numBits;
                auto extraLdsBytes     = extraLDSBytesPerElementBlock(arch, numBits);
                paddingBytes           = indexExpr / L(elementsPerTrLoad) * L(extraLdsBytes);
            }

            auto expr = toBytes(indexExpr) + paddingBytes;

            if(ci.isDirect2LDS)
            {
                expr = std::make_shared<Expression::Expression>(Expression::ToScalar{expr});
            }

            auto assignNode         = Assign{offsetRegisterType, convert(ci.offsetType, expr)};
            assignNode.variableType = ci.offsetType;
            auto assignTag          = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, offset, NaryArgument::DEST);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::makeAssignBase: assign {} expression {} to offset {}",
                assignTag,
                toString(assignNode.expression),
                offset);

            return assignTag;
        }

        int makeAssignStride(KernelGraph&        graph,
                             ComputeIndex const& ci,
                             const int           target,
                             const int           stride,
                             const int           increment,
                             bool                maybeLDS,
                             const bool          isTransposed,
                             const ContextPtr    context,
                             Transformer&        coords)
        {
            auto toBytes = [&](Expression::ExpressionPtr expr) -> Expression::ExpressionPtr {
                uint numBits = DataTypeInfo::Get(ci.valueType).elementBits;

                // TODO: This would be a good place to add a GPU
                // assert.  If numBits is not a multiple of 8, assert
                // that (expr * numBits) is a multiple of 8.
                Log::debug("  toBytes: {}: numBits {}", toString(ci.valueType), numBits);

                if(numBits % 8u == 0)
                    return expr * L(numBits / 8u);
                return (expr * L(numBits)) / L(8u);
            };

            auto indexExpr = ci.forward ? coords.forwardStride(increment, L(1), {target})[0]
                                        : coords.reverseStride(increment, L(1), {target})[0];

            // We have to manually invoke m_fastArith here since it can't traverse into the
            // RegisterTagManager.
            // TODO: Revisit storing expressions in the RegisterTagManager.
            bool unitStride = false;
            if(Expression::evaluationTimes(indexExpr)[Expression::EvaluationTime::Translate])
            {
                if(getUnsignedInt(evaluate(indexExpr)) == 1u)
                    unitStride = true;
            }

            uint                      elementBlockSize = 0;
            Expression::ExpressionPtr elementBlockStride;
            Expression::ExpressionPtr trLoadPairStride;
            Expression::ExpressionPtr elementBlockStridePaddingBytes{L(0u)};
            Expression::ExpressionPtr trLoadPairStridePaddingBytes{L(0u)};
            Expression::ExpressionPtr indexExprPaddingBytes{L(0u)};

            auto const& typeInfo = DataTypeInfo::Get(ci.valueType);
            auto        numBits  = DataTypeInfo::Get(typeInfo.segmentVariableType).elementBits;

            if(numBits == 16 || numBits == 8 || numBits == 6 || numBits == 4)
            {
                auto [elementBlockNumber, elementBlockIndex]
                    = getElementBlockValues(graph, target, isTransposed);

                elementBlockSize = elementBlockIndex;

                auto const& arch = context->targetArchitecture();
                if(isTransposed)
                {
                    // See addLoadWaveTileCTF8F6F4 in LowerTile.cpp
                    const auto wfs = arch.GetCapability(GPUCapability::DefaultWavefrontSize);
                    uint const numVBlocks
                        = wfs == 64 ? (numBits == 8 ? 2 : 1) : (numBits == 8 ? 4 : 2);
                    elementBlockSize = (elementBlockNumber / numVBlocks) * elementBlockSize;
                }
                AssertFatal(elementBlockSize > 0, "Invalid elementBlockSize: ", elementBlockSize);

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

            auto assignNode
                = Assign{Register::Type::Vector, toBytes(indexExpr) + indexExprPaddingBytes};
            assignNode.variableType = ci.strideType;
            assignNode.strideExpressionAttributes
                = {ci.strideType,
                   unitStride,
                   elementBlockSize,
                   toBytes(elementBlockStride) + elementBlockStridePaddingBytes,
                   toBytes(trLoadPairStride) + trLoadPairStridePaddingBytes};
            auto assignTag = graph.control.addElement(assignNode);
            graph.mapper.connect(assignTag, stride, NaryArgument::DEST);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::makeAssignStride: assign {} expression {} to stride {}",
                assignTag,
                toString(assignNode.expression),
                stride);
            return assignTag;
        }

        KernelGraph AssignComputeIndex::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddComputeIndex");
            auto kgraph = original;

            auto isComputeIndexPredicate
                = [&kgraph](int x) { return kgraph.control.get<ComputeIndex>(x).has_value(); };

            // search candidates
            auto candidates
                = kgraph.control.findNodes(*kgraph.control.roots().begin(), isComputeIndexPredicate)
                      .to<std::vector>();

            std::vector<std::tuple<int, int, int>> ciAndAssign;

            // commit changes
            for(const auto& tag : candidates)
            {
                auto base = kgraph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BASE});
                auto offset = kgraph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::OFFSET});
                auto stride = kgraph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});
                auto target = kgraph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::TARGET});
                auto increment = kgraph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::INCREMENT});
                auto buffer = kgraph.mapper.get(
                    tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BUFFER});

                rocRoller::Log::getLogger()->debug(
                    "KernelGraph::commit: computeindex {} base {} offset {} stride {}, target {} "
                    "increment {} buffer {}",
                    tag,
                    base,
                    offset,
                    stride,
                    target,
                    increment,
                    buffer);

                auto maybeLDS = kgraph.coordinates.get<LDS>(target).has_value();
                if(maybeLDS)
                {
                    // If target is LDS; it might be a duplicated LDS
                    // node.  For the purposes of computing indexes,
                    // use the parent LDS as the target instead.
                    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

                    auto maybeParentLDS = only(
                        kgraph.coordinates.getOutputNodeIndices(target, CT::isEdge<Duplicate>));
                    if(maybeParentLDS)
                        target = *maybeParentLDS;
                }
                maybeLDS = kgraph.coordinates.get<LDS>(target).has_value();

                auto isTransposed
                    = kgraph.coordinates
                          .findNodes(target,
                                     [&](int tag) -> bool {
                                         auto maybeAdhoc = kgraph.coordinates.get<Adhoc>(tag);
                                         return maybeAdhoc
                                                && maybeAdhoc->name()
                                                       == "Adhoc.transpose.simdsPerWave";
                                     })
                          .to<std::vector>()
                          .size()
                      == 1;

                auto ci = kgraph.control.get<ComputeIndex>(tag).value();
                auto direction
                    = ci.forward ? Graph::Direction::Upstream : Graph::Direction::Downstream;

                // 1. Set register coordinates
                auto xform = kgraph.buildTransformer(tag);

                auto const maybeForLoop = findContainingOperation<ForLoopOp>(tag, kgraph);
                auto       fullStop     = [&](int tag) { return tag == increment; };
                auto [required, path]
                    = findRequiredCoordinates(target, direction, fullStop, kgraph);

                std::map<int, Expression::ExpressionPtr> regCoords;
                auto isRegisterDim = [&maybeForLoop](auto dim) -> bool {
                    using T = std::decay_t<decltype(dim)>;
                    if(maybeForLoop)
                        return CIsAnyOf<T, Wavefront, Workitem, Workgroup, ForLoop>;
                    else
                        return CIsAnyOf<T, Wavefront, Workitem, Workgroup>;
                };
                for(auto coord : required)
                {
                    if(std::visit(isRegisterDim, kgraph.coordinates.getNode(coord)))
                    {
                        auto registerType = Register::Type::Vector;
                        auto coordDF      = std::make_shared<Expression::Expression>(
                            Expression::DataFlowTag{coord, registerType, DataType::UInt32});
                        regCoords[coord] = coordDF;
                    }
                }
                for(auto const& [coord, expr] : regCoords)
                {
                    if(!xform.hasCoordinate(coord))
                        xform.setCoordinate(coord, expr);
                }

                // 2. Set remaining coordinates
                for(auto coord : required)
                    if((coord != increment) && (!xform.hasCoordinate(coord)))
                        xform.setCoordinate(coord, L(0u));

                // 3. Set the increment coordinate to zero if it doesn't
                // already have a value
                bool initializeIncrement
                    = !xform.hasPath({target}, direction == Graph::Direction::Upstream);
                if(initializeIncrement)
                {
                    xform.setCoordinate(increment, L(0u));
                }

                auto assignStrideTag = -1, assignBaseTag = -1;

                if(base < 0 && offset > 0)
                {
                    assignBaseTag = makeAssignBase(
                        kgraph, ci, target, offset, maybeLDS, isTransposed, m_context, xform);
                }

                if(stride > 0)
                {
                    assignStrideTag = makeAssignStride(kgraph,
                                                       ci,
                                                       target,
                                                       stride,
                                                       increment,
                                                       maybeLDS,
                                                       isTransposed,
                                                       m_context,
                                                       xform);
                }

                if(assignStrideTag != -1 || assignBaseTag != -1)
                {
                    ciAndAssign.push_back({tag, assignBaseTag, assignStrideTag});
                }
            }

            for(auto const& tags : ciAndAssign)
            {
                auto [ciTag, assignBaseTag, assignStrideTag] = tags;
                if(assignStrideTag != -1)
                    insertAfter(kgraph, ciTag, assignStrideTag, assignStrideTag);
                if(assignBaseTag != -1)
                    insertAfter(kgraph, ciTag, assignBaseTag, assignBaseTag);
            }

            return kgraph;
        }
    }
}
