// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
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

        std::string toString(LoadStoreTileGenerator::LoadStoreTileInfo const& info)
        {
#define ShowReg(reg)                                              \
    (reg ? concatenate("\t" #reg " = ", reg->description(), "\n") \
         : concatenate("\t" #reg " = (nullptr)\n"))

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
                               ShowReg(info.bufDesc),
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

        inline Register::ValuePtr LoadStoreTileGenerator::getBufferDesc(int tag)
        {
            auto bufferTag = m_graph->mapper.get<Buffer>(tag);
            auto bufferSrd = m_context->registerTagManager()->getRegister(bufferTag);
            return bufferSrd;
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
        ExpressionPtr LoadStoreTileGenerator::getOffsetExpr(int  opTag,
                                                            bool isStorePartOfGlobalToLDS,
                                                            Transformer const& coords)
        {
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::getOffsetExpr("
                                               "operationTag: {}, isStorePartOfGlobalToLDS: {})",
                                               opTag,
                                               isStorePartOfGlobalToLDS);

            auto [target, direction]
                = getOperationTarget(opTag, *m_graph, isStorePartOfGlobalToLDS);
            auto [required, path] = findRequiredCoordinates(target, direction, *m_graph);

            auto unrolls = filterCoordinates<Unroll>(required, *m_graph);
            if(unrolls.size() == 0)
                return nullptr;

            ExpressionPtr result = Expression::literal(0u);

            auto getUnrollStrideCoordinate = [this](int opTag, int subdimension) {
                for(auto const& c : m_graph->mapper.getConnections(opTag))
                {
                    if(std::holds_alternative<Connections::UnrollStride>(c.connection))
                    {
                        auto curConnection = std::get<Connections::UnrollStride>(c.connection);
                        auto unrollDim     = curConnection.unrollDimension;
                        if(unrollDim != subdimension)
                            continue;
                        return c.coordinate;
                    }
                }
                return -1;
            };

            for(auto const& unroll : unrolls)
            {
                auto const subDimension = m_graph->mapper.getConnectionSubdimension(opTag, unroll);

                if(subDimension == -1)
                    continue;

                auto strideTag = getUnrollStrideCoordinate(opTag, subDimension);
                if(strideTag == -1)
                    continue;

                auto [strideExpr, strideAttrs]
                    = m_context->registerTagManager()->getExpression(strideTag);

                Log::debug(
                    "  unroll coord {} value: {}", unroll, toString(coords.getCoordinate(unroll)));
                Log::debug("  stride coord {} expr: {}", strideTag, toString(strideExpr));
                result = result + coords.getCoordinate(unroll) * strideExpr;
            }

            return result;
        }

        DataType getOffsetDataTypeFromGraph(int                op,
                                            KernelGraph const& graph,
                                            bool               isStorePartOfGlobalToLDSOp)
        {
            DataType rv = DataType::UInt64;
            auto     s  = graph.control.get<StoreTiled>(op);
            auto     l  = graph.control.get<LoadTiled>(op);
            auto     ll = graph.control.get<LoadLDSTile>(op);
            auto     sl = graph.control.get<StoreLDSTile>(op);

            auto isGlobalLoad = false;
            if(l)
            {
                auto [_, macTile] = graph.getDimension<MacroTile>(op);
                if(macTile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
                {
                    isGlobalLoad = true;
                }
            }

            if(s || (l and not isGlobalLoad) || ll || sl || isStorePartOfGlobalToLDSOp)
            {
                rv = DataType::UInt32;
            }
            return rv;
        }

        Generator<Instruction> LoadStoreTileGenerator::getOffset(LoadStoreTileInfo& info,
                                                                 Transformer        coords,
                                                                 bool               preserveOffset,
                                                                 bool isStorePartOfGlobalToLDS)
        {
            auto offsetTag
                = m_graph->mapper.get<Offset>(info.tag, isStorePartOfGlobalToLDS ? 2 : 0);
            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::getOffset(tag:"
                                               " {}, offsetTag: {})",
                                               info.tag,
                                               offsetTag);

            AssertFatal(offsetTag >= 0, "No Offset found");

            ExpressionPtr rowOffsetExpr;

            if(m_context->registerTagManager()->hasRegister(offsetTag))
            {
                if(isStorePartOfGlobalToLDS)
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

                rowOffsetExpr = getOffsetExpr(info.tag, isStorePartOfGlobalToLDS, coords);
            }
            else
            {
                auto baseTag = -1;
                for(auto const& c : m_graph->mapper.getConnections(info.tag))
                {
                    if(!std::holds_alternative<Connections::BaseOffset>(c.connection))
                        continue;
                    auto curConnection = std::get<Connections::BaseOffset>(c.connection);
                    auto subdim        = curConnection.subdimension;

                    if(subdim != 0)
                        continue;
                    baseTag = c.coordinate;
                }
                if(baseTag == -1)
                {
                    Throw<FatalError>("Base offset not found");
                }

                if(isStorePartOfGlobalToLDS)
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
                        getOffsetDataTypeFromGraph(info.tag, *m_graph, isStorePartOfGlobalToLDS),
                        1);
                    info.rowOffsetReg->setName(concatenate("Offset", offsetTag));
                    m_context->getScopeManager()->addRegister(offsetTag);

                    // Copy base to new offset register
                    auto baseReg = m_context->registerTagManager()->getRegister(baseTag);
                    co_yield m_context->copier()->copy(info.rowOffsetReg, baseReg);
                }

                rowOffsetExpr = getOffsetExpr(info.tag, isStorePartOfGlobalToLDS, coords);
            }

            if(isStorePartOfGlobalToLDS)
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
                    stride->setName(concatenate("Stride", strideTag));
                }
                else
                {
                    stride = nullptr;
                }

                co_yield generate(stride, strideExpr);
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

            // When padding, the ldsWriteStride might be larger than m_workgroupSizeTotal * numBytes
            AssertFatal(info.ldsWriteStride >= m_workgroupSizeTotal * numBytes,
                        ShowValue(info.ldsWriteStride));

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
                info.offset->setName("OffsetD2L");
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
                            info.offset
                                = Register::Value::Literal(offsetValue + r * elementBlockStride);
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

                    if(i < info.m - 1)
                    {
                        if(info.isMacroTileRowStride)
                        {
                            const auto rowElementBlockStride
                                = info.rowStrideAttributes.elementBlockStride;
                            AssertFatal(Expression::evaluationTimes(
                                            rowElementBlockStride)[EvaluationTime::Translate],
                                        "Could not determine "
                                        "rowStrideAttributes.ElementBlockStride at translate-time.",
                                        ShowValue(rowElementBlockStride));
                            offsetValue += getUnsignedInt(evaluate(rowElementBlockStride));
                        }
                        else
                        {
                            offsetValue += rowStride;
                        }
                    }
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

                const auto wfs = arch.GetCapability(GPUCapability::DefaultWavefrontSize);

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
                            = (wfs == 32 && isF16(info.data->variableType().dataType))
                                  ? j * trLoadPairStride
                                  : (j % 2) * trLoadPairStride + (j / 2) * elementBlockStride;
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
                        colOffsetReg,
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
         * @param info detailed information need to generate loads and stores
         * @param coords Transformer object
         * @return Generator<Instruction>
         */
        template <MemoryInstructions::MemoryDirection Dir>
        Generator<Instruction> LoadStoreTileGenerator::moveTile(LoadStoreTileInfo& info,
                                                                Transformer&       coords)
        {
            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::moveTile<{}>({}) {}x{}",
                toString(Dir),
                info.tag,
                info.m,
                info.n);

            Register::ValuePtr finalVGPR;

            if(!info.offset)
            {
                info.offset = Register::Value::Literal(0u);
            }

            if(info.kind == MemoryInstructions::MemoryKind::Buffer
               || info.kind == MemoryInstructions::MemoryKind::Buffer2LDS)
            {
                info.bufDesc = getBufferDesc(info.tag);
            }

            auto const& varTypeInfo = DataTypeInfo::Get(info.varType);

            co_yield Instruction::Comment(
                concatenate(ShowValue(varTypeInfo.elementBits), ShowValue(varTypeInfo.packing)));

            info.elementBits = (uint)varTypeInfo.elementBits;

            if(info.m > 1)
                co_yield generateStride(info.rowStrideReg, info.rowStrideAttributes, info.tag, 0);
            else
                info.rowStrideReg = Register::Value::Literal(0u);
            co_yield generateStride(info.colStrideReg, info.colStrideAttributes, info.tag, 1);

            AssertFatal(info.rowStrideReg, "Invalid row stride register.");
            AssertFatal(info.colStrideReg, "Invalid col stride register.");

            bool colStrideIsLiteral = (info.colStrideReg->regType() == Register::Type::Literal);

            bool allStridesAreLiteral
                = (info.rowStrideReg->regType() == Register::Type::Literal && colStrideIsLiteral
                   && info.offset->regType() == Register::Type::Literal);
            bool colStrideIsOne = colStrideIsLiteral && info.colStrideAttributes.unitStride;

            if(Dir == MemoryInstructions::MemoryDirection::Load)
            {
                auto macTileTag = m_graph->mapper.get<MacroTile>(info.tag);

                macTileTag
                    = only(m_graph->coordinates.getInputNodeIndices(macTileTag, CT::isEdge<View>))
                          .value_or(macTileTag);

                auto packedVariableType = varTypeInfo.packedVariableType();

                auto allocOptions = Register::AllocationOptions::FullyContiguous();

                auto elementBits = DataTypeInfo::Get(varTypeInfo.segmentVariableType).elementBits;
                const auto& arch = m_context->targetArchitecture();
                auto        macTile = m_graph->coordinates.getNode<MacroTile>(macTileTag);
                if(macTile.memoryType == MemoryType::VGPR && elementBits == 6
                   && (!arch.HasCapability(GPUCapability::DSReadTransposeB6PaddingBytes)
                       || info.isPadded)
                   && !info.isTransposedTile)
                {
                    // FIXME: fix contiguousChunkWidth calculation
                    auto registerCount = arch.target().gfx == GPUArchitectureGFX::GFX1250
                                             ? info.n * varTypeInfo.registerCount
                                             : varTypeInfo.registerCount;
                    allocOptions = {.contiguousChunkWidth = int(registerCount), .alignment = 2};
                    co_yield Instruction::Comment(
                        concatenate("Allocation options: ", allocOptions));
                }

                if(arch.HasCapability(GPUCapability::HasVGPRIndexing)
                   and isScaleType(varTypeInfo.variableType.dataType))
                {
                    allocOptions.forceReservedRegion = true;
                }

                auto tmpl = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, info.varType, info.m * info.n, allocOptions);
                tmpl->setName("tmpl");

                if(info.kind == MemoryInstructions::MemoryKind::Buffer2LDS
                   || info.kind == MemoryInstructions::MemoryKind::TDMToLDS)
                {
                    info.bufOpts.lds = 1;

                    // get lds write stride
                    Register::ValuePtr           ldsWriteStrideRegister;
                    RegisterExpressionAttributes _ignore;
                    co_yield generateStride(ldsWriteStrideRegister, _ignore, info.tag, 2);
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
                           info.tag,
                           macTileTag,
                           info.data->description());
            }
            else
            {
                if(!m_context->targetArchitecture().HasCapability(
                       GPUCapability::ArchAccUnifiedRegs))
                {
                    co_yield m_context->copier()->ensureType(
                        info.data, info.data, Register::Type::Vector);
                }

                // Convert the data to the expected datatype
                auto existingVarType = info.data->variableType();
                if(existingVarType != info.varType
                   && DataTypeInfo::Get(existingVarType).segmentVariableType != info.varType)
                {
                    co_yield Instruction::Comment("Convert in LSTGen");
                    co_yield m_context->copier()->ensureType(
                        info.data, info.data, Register::Type::Vector);
                    Register::ValuePtr result;
                    co_yield generate(result, convert(info.varType, info.data->expression()));
                    info.data = result;
                }
            }

            info.packedAmount = DataTypeInfo::Get(info.data->variableType()).packing;

            co_yield Instruction::Comment(
                ShowValue(Dir) + toString(info)
                + concatenate(ShowValue(allStridesAreLiteral), ShowValue(colStrideIsOne)));

            // Get the offset values set by AssignIndexExpressions
            co_yield getOffset(info, coords, !allStridesAreLiteral && info.m > 1);
            AssertFatal(info.rowOffsetReg, "Invalid row offset register.");

            if(info.kind == MemoryInstructions::MemoryKind::Buffer2LDS
               || info.kind == MemoryInstructions::MemoryKind::TDMToLDS)
            {
                co_yield getOffset(
                    info, coords, /*preserveOffset=*/false, /*isStorePartOfGlobalToLDS=*/true);

                // set global read offset
            }

            if(info.kind == MemoryInstructions::MemoryKind::TDMToLDS)
            {
                auto tdmExpr = info.tdmDesc->expression();

                const auto ldsAddressExpr    = info.data->expression();
                const auto globalAddressExpr = info.rowOffsetReg->expression();

                tdmExpr = TDMDescriptor::SetLDSAddress(tdmExpr, ldsAddressExpr);
                tdmExpr = TDMDescriptor::SetGlobalAddress(tdmExpr, globalAddressExpr);

                const auto tileDim0 = info.n;
                const auto tileDim1 = info.m;
                Log::debug(fmt::format("  TileDim0 {} TileDim1 {}", tileDim0, tileDim1));

                const auto tileDim0Expr = Expression::literal(tileDim0);
                const auto tileDim1Expr = Expression::literal(tileDim1);
                tdmExpr = TDMDescriptor::SetTileDims(tdmExpr, tileDim0Expr, tileDim1Expr);

                co_yield Expression::generate(info.tdmDesc, tdmExpr, m_context);

                co_yield m_context->mem()->loadTensorToLDS(info.tdmDesc);
            }
            else if(allStridesAreLiteral)
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

            LoadStoreTileInfo info{.tag              = tag,
                                   .kind             = MemoryInstructions::MemoryKind::Buffer,
                                   .m                = m,
                                   .n                = n,
                                   .data             = nullptr,
                                   .varType          = load.varType,
                                   .isTransposedTile = false,
                                   .isPadded         = tile.paddingBytes() > 0};
            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords);
        }

        LoadStoreTileGenerator::LoadStoreTileInfo
            LoadStoreTileGenerator::loadMacroTileLDSInfo(int tag, LoadLDSTile const& load)
        {
            auto [ldsTag, _lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, _tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug(
                "KernelGraph::LoadStoreTileGenerator::loadMacroTileLDS: OP {} LDS {} MacroTile {}",
                tag,
                ldsTag,
                tileTag);

            LoadStoreTileInfo result;
            result.comments = {concatenate(
                "GEN: loadMacroTileLDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag)};

            if(m_context && m_context->registerTagManager()->hasRegister(ldsTag))
            {
                auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
                result.offset
                    = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());
            }

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            auto packing = DataTypeInfo::Get(load.varType).packing;
            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            result.tag     = tag;
            result.kind    = MemoryInstructions::MemoryKind::Local;
            result.m       = m;
            result.n       = n;
            result.data    = nullptr;
            result.varType = load.varType;
            return result;
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileDirect2LDS(
            int tag, LoadTileDirect2LDS const& load, Transformer coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);
            auto varType         = load.varType;

            auto packing = DataTypeInfo::Get(varType).packing;

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "loadMacroTileDirect2LDS: OP {} LDS {} MacroTile {}",
                                               tag,
                                               ldsTag,
                                               tileTag);
            co_yield Instruction::Comment(concatenate(
                "GEN: loadMacroTileDirect2LDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag));

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                auto numElements = GetNumLDSElements(*m_graph, ldsTag) / packing;

                ldsAllocation = Register::Value::AllocateLDS(m_context, varType, numElements);
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

            AssertFatal(n % packing == 0,
                        ShowValue(m),
                        ShowValue(n),
                        ShowValue(packing),
                        ShowValue(load.varType));
            n /= packing;

            LoadStoreTileInfo info{.tag     = tag,
                                   .kind    = MemoryInstructions::MemoryKind::Buffer2LDS,
                                   .m       = m,
                                   .n       = n,
                                   .data    = ldsOffset,
                                   .varType = varType,
                                   .bufDesc = nullptr,
                                   .bufOpts = {}};
            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords)
                .map(MemoryInstructions::addExtraDst(ldsAllocation));
        }

        LoadStoreTileGenerator::LoadStoreTileInfo
            LoadStoreTileGenerator::loadMacroTileWAVELDSInfo(int tag, LoadLDSTile const& load)
        {
            auto [ldsTag, _lds]          = m_graph->getDimension<LDS>(tag);
            auto [waveTileTag, waveTile] = m_graph->getDimension<WaveTile>(tag);

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "loadMacroTileWAVELDS: OP {} LDS {} WaveTile {}",
                                               tag,
                                               ldsTag,
                                               waveTileTag);

            LoadStoreTileInfo result;
            result.comments = {concatenate(
                "GEN: loadMacroTileWAVELDS OP ", tag, " LDS ", ldsTag, " WaveTile ", waveTileTag)};

            ldsTag = only(m_graph->coordinates.getOutputNodeIndices(ldsTag, CT::isEdge<View>))
                         .value_or(ldsTag);
            // Find the LDS allocation that contains the tile and store
            // the offset of the beginning of the allocation into ldsOffset.
            if(m_context && m_context->registerTagManager()->hasRegister(ldsTag))
            {
                auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
                result.offset
                    = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());
            }

            uint numVGPRBlockSets = GetVGPRBlockSetDimSize(*m_graph, tag).value_or(1);

            uint numElements       = waveTile.sizes[0] * waveTile.sizes[1] / numVGPRBlockSets;
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

            result.tag                  = tag;
            result.kind                 = MemoryInstructions::MemoryKind::Local;
            result.m                    = numVGPRBlockSets;
            result.n                    = numVgpr;
            result.data                 = nullptr;
            result.varType              = load.varType;
            result.isTransposedTile     = load.isTransposedTile;
            result.isMacroTileRowStride = numVGPRBlockSets > 1;
            return result;
        }

        Generator<Instruction> LoadStoreTileGenerator::loadMacroTileWAVE(int              tag,
                                                                         LoadTiled const& load,
                                                                         Transformer      coords)
        {
            auto [macTileTag, macTile]   = m_graph->getDimension<MacroTile>(tag);
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

            auto memoryKind = MemoryInstructions::MemoryKind::Buffer;
            if(macTile.memoryType == MemoryType::WAVE_FROM_GLOBAL)
            {
                memoryKind = MemoryInstructions::MemoryKind::Global;
            }

            LoadStoreTileInfo info{.tag     = tag,
                                   .kind    = memoryKind,
                                   .m       = 1,
                                   .n       = numVgpr,
                                   .data    = nullptr,
                                   .varType = load.varType};

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords);
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

            LoadStoreTileInfo info{.tag     = tag,
                                   .kind    = MemoryInstructions::MemoryKind::Buffer,
                                   .m       = m,
                                   .n       = n,
                                   .data    = nullptr,
                                   .varType = load.varType};
            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords);
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
            case MemoryType::WAVE_FROM_GLOBAL:
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
                    Throw<FatalError>(ShowValue(macTile.layoutType),
                                      " Layout type not supported yet for LoadTiled.");
                }
            }
            break;
            default:
                Throw<FatalError>(ShowValue(macTile.memoryType),
                                  " Tile affinity type not supported yet for LoadTiled.");
            }
        }

        LoadStoreTileGenerator::LoadStoreTileInfo
            LoadStoreTileGenerator::getLoadLDSTileInfo(int tag, LoadLDSTile const& load)
        {
            auto [_, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::WAVE_SPLIT:
            case MemoryType::VGPR:
            case MemoryType::LDS:
                return loadMacroTileLDSInfo(tag, load);
            case MemoryType::WAVE:
            case MemoryType::WAVE_SWIZZLE:
            case MemoryType::WAVE_FROM_GLOBAL:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_A:
                case LayoutType::MATRIX_B:
                    return loadMacroTileWAVELDSInfo(tag, load);
                default:
                    Throw<FatalError>("Layout type not supported yet for LoadLDSTile.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for LoadLDSTile.");
            }
        }

        Generator<Instruction> LoadStoreTileGenerator::genLoadLDSTile(int                tag,
                                                                      LoadLDSTile const& load,
                                                                      Transformer        coords)
        {
            auto info = getLoadLDSTileInfo(tag, load);

            for(auto const& comment : info.comments)
                co_yield Instruction::Comment(comment);

            auto modelledAddresses = m_graph->modelledAddresses.find(tag);
            auto applyAddresses    = [&](Instruction inst) {
                if(modelledAddresses != m_graph->modelledAddresses.end())
                {
                    inst.setModelledAddresses(modelledAddresses->second);
                    if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                        inst.addComment(fmt::format("Modelled addresses (normalized): {}",
                                                    modelledAddresses->second));
                }
                return inst;
            };

            auto [ldsTag, _lds] = m_graph->getDimension<LDS>(tag);
            ldsTag = only(m_graph->coordinates.getOutputNodeIndices(ldsTag, CT::isEdge<View>))
                         .value_or(ldsTag);

            if(m_context && m_context->registerTagManager()->hasRegister(ldsTag))
            {
                auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
                co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords)
                    .map(MemoryInstructions::addExtraSrc(ldsAllocation))
                    .map(applyAddresses);
            }
            else
                co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords)
                    .map(applyAddresses);
        }

        Generator<Instruction> LoadStoreTileGenerator::genLoadTileDirect2LDS(
            int tag, LoadTileDirect2LDS const& load, Transformer coords)
        {
            co_yield loadMacroTileDirect2LDS(tag, load, coords);
        }

        LoadStoreTileGenerator::LoadStoreTileInfo
            LoadStoreTileGenerator::getStoreLDSTileInfo(int tag, StoreLDSTile const& store)
        {
            auto [_, macTile] = m_graph->getDimension<MacroTile>(tag);

            switch(macTile.memoryType)
            {
            case MemoryType::VGPR:
            case MemoryType::LDS:
                return storeMacroTileLDSInfo(tag, store);
            case MemoryType::WAVE:
            {
                switch(macTile.layoutType)
                {
                case LayoutType::MATRIX_ACCUMULATOR:
                    return storeMacroTileWAVELDSInfo(tag, store);
                default:
                    Throw<FatalError>("Layout type not supported yet for StoreLDSTile.");
                }
            }
            break;
            default:
                Throw<FatalError>("Tile affinity type not supported yet for StoreLDSTile.");
            }
        }

        LoadStoreTileGenerator::LoadStoreTileInfo
            LoadStoreTileGenerator::storeMacroTileLDSInfo(int tag, StoreLDSTile const& store)
        {
            auto [ldsTag, _]     = m_graph->getDimension<LDS>(tag);
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
            auto infoComment = concatenate(
                "GEN: storeMacroTileLDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag);

            auto packing = DataTypeInfo::Get(store.varType).packing;

            // Temporary register(s) that is used to copy the data from global memory to
            // local memory.
            auto varType = store.varType;

            auto paddingBytes = tile.paddingBytes();
            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation = nullptr;
            Register::ValuePtr ldsOffset     = nullptr;

            if(m_context)
            {
                if(!m_context->registerTagManager()->hasRegister(ldsTag))
                {
                    auto numElements = GetNumLDSElements(*m_graph, ldsTag);
                    ldsAllocation    = Register::Value::AllocateLDS(
                        m_context, varType, numElements / packing, /*alignment*/ 4, paddingBytes);
                    m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
                }
                else
                {
                    ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
                }

                ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());
            }

            auto [elemXTag, elemX] = m_graph->getDimension<ElementNumber>(tag, 0);
            auto [elemYTag, elemY] = m_graph->getDimension<ElementNumber>(tag, 1);
            auto const m           = getUnsignedInt(evaluate(elemX.size));
            auto       n           = getUnsignedInt(evaluate(elemY.size));

            AssertFatal(n % packing == 0, ShowValue(n), ShowValue(packing));
            n /= packing;

            LoadStoreTileInfo info{
                .tag     = tag,
                .kind    = MemoryInstructions::MemoryKind::Local,
                .m       = m,
                .n       = n,
                .data    = (m_context && m_context->registerTagManager()->hasRegister(tileTag))
                               ? m_context->registerTagManager()->getRegister(tileTag)
                               : nullptr,
                .varType = varType,
                .offset  = ldsOffset,
                .isTransposedTile = false,
                .isPadded         = paddingBytes > 0};
            info.comments = {infoComment};
            return info;
        }

        Generator<Instruction> LoadStoreTileGenerator::genLoadTiledTDMToLDS(
            int tag, LoadTiledTDMToLDS const& load, Transformer coords)
        {
            auto [ldsTag, lds]   = m_graph->getDimension<LDS>(tag);
            auto [tileTag, tile] = m_graph->getDimension<MacroTile>(tag);

            rocRoller::Log::getLogger()->debug("KernelGraph::LoadStoreTileGenerator::"
                                               "genLoadTDMToLDS: OP {} LDS {} MacroTile {}",
                                               tag,
                                               ldsTag,
                                               tileTag);
            co_yield Instruction::Comment(concatenate(
                "GEN: LoadTiledTDMToLDS OP ", tag, " LDS ", ldsTag, " MacroTile ", tileTag));

            auto numElements = product(tile.subTileSizes) * m_workgroupSizeTotal;

            // Allocate LDS memory
            Register::ValuePtr ldsAllocation;
            if(!m_context->registerTagManager()->hasRegister(ldsTag))
            {
                ldsAllocation = Register::Value::AllocateLDS(m_context, load.varType, numElements);
                m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
            }
            else
            {
                ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
            }

            // base offset of the allocation
            const auto ldsOffset
                = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());

            auto [elemXTag, elemX]           = m_graph->getDimension<ElementNumber>(tag, 0);
            const auto  swapTileDimensions   = isSwappedLayout(*m_graph, elemXTag, elemX);
            const auto& workgroupSizeSlowDim = m_context->kernel()->workgroupSize()[0];
            const auto  wavefrontSize        = m_context->targetArchitecture().GetCapability(
                GPUCapability::DefaultWavefrontSize);
            const auto wavesPerWorkgroup = workgroupSizeSlowDim / wavefrontSize;

            const uint64_t m
                = (swapTileDimensions ? tile.sizes[1] : tile.sizes[0]) / wavesPerWorkgroup;
            const uint64_t n = (swapTileDimensions ? tile.sizes[0] : tile.sizes[1]);

            auto toBytes = [&](auto dimSize) -> uint32_t {
                const auto bitsInAByte   = 8;
                const auto elementBits   = DataTypeInfo::Get(load.varType).elementBits;
                const auto dimSizeInBits = dimSize * elementBits;

                AssertFatal(dimSizeInBits % bitsInAByte == 0,
                            "Dimension size in bytes must be an integer.",
                            ShowValue(dimSize),
                            ShowValue(elementBits),
                            ShowValue(dimSizeInBits));
                return dimSizeInBits / bitsInAByte;
            };

            auto tdmTag  = m_graph->mapper.get<TDM>(tag);
            auto tdmRegs = m_context->registerTagManager()->getRegister(tdmTag);

            LoadStoreTileInfo info{.tag     = tag,
                                   .kind    = MemoryInstructions::MemoryKind::TDMToLDS,
                                   .m       = m,
                                   .n       = toBytes(n),
                                   .data    = ldsOffset,
                                   .varType = load.varType,
                                   .tdmDesc = tdmRegs};

            co_yield moveTile<MemoryInstructions::MemoryDirection::Load>(info, coords)
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

            LoadStoreTileInfo info{
                .tag     = tag,
                .kind    = MemoryInstructions::MemoryKind::Buffer,
                .m       = m,
                .n       = n,
                .data    = vgpr,
                .varType = store.varType,
                .bufOpts = store.bufOpts,
            };
            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(info, coords);
        }

        LoadStoreTileGenerator::LoadStoreTileInfo
            LoadStoreTileGenerator::storeMacroTileWAVELDSInfo(int tag, StoreLDSTile const& store)
        {
            auto [ldsTag, _lds]          = m_graph->getDimension<LDS>(tag);
            auto [macTileTag, _macTile]  = m_graph->getDimension<MacroTile>(tag);
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
            auto infoComment = concatenate("GEN: storeMacroTileWAVELDS OP ",
                                           tag,
                                           " LDS ",
                                           ldsTag,
                                           " MacroTile ",
                                           macTileTag,
                                           " WaveTile ",
                                           waveTileTag);

            auto packing = DataTypeInfo::Get(store.varType).packing;

            // Allocate LDS memory, and store the offset of the beginning of the allocation
            // into ldsOffset.
            Register::ValuePtr ldsAllocation = nullptr;
            Register::ValuePtr ldsOffset     = nullptr;
            Register::ValuePtr agpr          = nullptr;

            if(m_context)
            {
                if(!m_context->registerTagManager()->hasRegister(ldsTag))
                {
                    auto numElements = GetNumLDSElements(*m_graph, ldsTag);

                    ldsAllocation
                        = Register::Value::AllocateLDS(m_context, varType, numElements / packing);
                    m_context->registerTagManager()->addRegister(ldsTag, ldsAllocation);
                }
                else
                {
                    ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
                }

                ldsOffset = Register::Value::Literal(ldsAllocation->getLDSAllocation()->offset());
                agpr      = m_context->registerTagManager()->getRegister(macTileTag);
            }

            auto [_, lane]         = m_graph->getDimension<Lane>(tag);
            auto activeLanesInWave = getUnsignedInt(evaluate(lane.size));

            AssertFatal(waveTileNumElements % (activeLanesInWave * packing) == 0,
                        ShowValue(waveTileNumElements),
                        ShowValue(activeLanesInWave),
                        ShowValue(packing),
                        ShowValue(activeLanesInWave * packing));

            LoadStoreTileInfo info{.tag     = tag,
                                   .kind    = MemoryInstructions::MemoryKind::Local,
                                   .m       = 0,
                                   .n       = 0,
                                   .data    = agpr,
                                   .varType = varType,
                                   .offset  = ldsOffset};
            info.comments = {infoComment};

            if(m_context)
            {
                info.comments.push_back(concatenate(ShowValue(waveTile),
                                                    ShowValue(waveTileNumElements),
                                                    ShowValue(activeLanesInWave),
                                                    ShowValue(packing),
                                                    ShowValue(activeLanesInWave * packing),
                                                    ShowValue(store.varType)));

                info.comments.push_back(concatenate(ShowValue(agpr->description()),
                                                    ShowValue(waveTile),
                                                    ShowValue(waveTileNumElements),
                                                    ShowValue(activeLanesInWave),
                                                    ShowValue(packing),
                                                    ShowValue(activeLanesInWave * packing)));
            }

            uint numVgpr = waveTileNumElements / activeLanesInWave;

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

            info.m = m;
            info.n = n;
            return info;
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

            LoadStoreTileInfo info{.tag     = tag,
                                   .kind    = MemoryInstructions::MemoryKind::Buffer,
                                   .m       = m,
                                   .n       = n,
                                   .data    = agpr,
                                   .varType = store.varType,
                                   .bufOpts = store.bufOpts};
            co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(info, coords);
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
            case MemoryType::WAVE_SWIZZLE:
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
            auto info = getStoreLDSTileInfo(tag, store);

            for(auto const& comment : info.comments)
                co_yield Instruction::Comment(comment);

            auto modelledAddresses = m_graph->modelledAddresses.find(tag);
            auto applyAddresses    = [&](Instruction inst) {
                if(modelledAddresses != m_graph->modelledAddresses.end())
                {
                    inst.setModelledAddresses(modelledAddresses->second);
                    if(GPUInstructionInfo::isLDS(inst.getOpCode()))
                        inst.addComment(fmt::format("Modelled addresses (normalized): {}",
                                                    modelledAddresses->second));
                }
                return inst;
            };

            auto [ldsTag, _lds] = m_graph->getDimension<LDS>(tag);

            if(m_context && m_context->registerTagManager()->hasRegister(ldsTag))
            {
                auto ldsAllocation = m_context->registerTagManager()->getRegister(ldsTag);
                co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(info, coords)
                    .map(MemoryInstructions::addExtraDst(ldsAllocation))
                    .map(applyAddresses);
            }
            else
                co_yield moveTile<MemoryInstructions::MemoryDirection::Store>(info, coords)
                    .map(applyAddresses);
        }
    }
}
