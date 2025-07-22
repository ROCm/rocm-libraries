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

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AssignComputeIndex.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/ExpressionTransformations.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        inline Expression::ExpressionPtr L(auto const& x)
        {
            return Expression::literal(x);
        }

        int makeAssignBase(KernelGraph& graph,
                        ComputeIndex const& ci,
                       int          target,
                       int          offset,
                       bool         maybeLDS,
                       bool         isTransposed,
                       ContextPtr   context,
                       Transformer& coords)
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

            auto indexExpr
                    = ci.forward ? coords.forward({target})[0] : coords.reverse({target})[0];

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
                    expr = makeScalar(expr);

            // auto assignNode         = Assign{offsetRegisterType, convert(ci.offsetType, expr)};
            // assignNode.variableType = ci.offsetType;
            // auto assignTag          = graph.control.addElement(assignNode);
            // graph.mapper.connect(assignTag, offset, NaryArgument::DEST);

            // rocRoller::Log::getLogger()->debug(
            //     "KernelGraph::makeAssignBase: assign {} expression {} to offset {}",
            //     assignTag,
            //     toString(assignNode.expression),
            //     offset);

            // return assignTag;

            std::cout << "YL: makeAssignBase (target, offset, expression) " << target << ", " << offset << ", " << toString(expr) << std::endl;
            return 0;

        }

        KernelGraph AssignComputeIndex::apply(KernelGraph const& original)
        {
            TIMER(t, "KernelGraph::AddComputeIndex");
            auto kgraph = original;

            auto isComputeIndexPredicate = [&kgraph](int x) {
                return kgraph.control.get<ComputeIndex>(x).has_value();
            };

            // search candidates
            auto candidates = kgraph.control.findNodes( *kgraph.control.roots().begin(), isComputeIndexPredicate).to<std::vector>();
            std::cout << "Number of ComputeIndex nodes " << candidates.size() << std::endl;

            // commit changes
            for (const auto& tag : candidates)
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

                auto ci = kgraph.control.get<ComputeIndex>(tag).value();

                // get Transformer
                Transformer coords(&kgraph.coordinates);

                // check maybeLDS
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

                // check isTransposed
                auto isTransposed
                = kgraph.coordinates
                      .findNodes(target,
                                 [&](int tag) -> bool {
                                     auto maybeAdhoc = kgraph.coordinates.get<Adhoc>(tag);
                                     return maybeAdhoc
                                            && maybeAdhoc->name() == "Adhoc.transpose.simdsPerWave";
                                 })
                      .to<std::vector>()
                      .size()
                  == 1;

                // Set the zero-coordinates to zero
                auto fullStop  = [&](int tag) { return tag == increment; };
                auto direction = ci.forward ? Graph::Direction::Upstream : Graph::Direction::Downstream;
                auto [required, path] = findRequiredCoordinates(target, direction, fullStop, kgraph);

                for(auto requiredTag : required)
                    if((requiredTag  != increment) && (!coords.hasCoordinate(requiredTag )))
                        coords.setCoordinate(requiredTag , L(0u));

                // Set the increment coordinate to zero if it doesn't
                // already have a value
                bool initializeIncrement = !coords.hasPath({target}, ci.forward);
                if(initializeIncrement)
                {
                    coords.setCoordinate(increment, L(0u));
                }


                // if (stride > 0)
                // {
                //     auto assignStrideTag = makeAssignStride();
                //     // insertAfter(kgraph, tag, assignStrideTag);                    
                // }

                if (base < 0 && offset > 0)
                {
                    auto assignBaseTag = makeAssignBase(kgraph, ci, target, offset, maybeLDS, isTransposed, m_context, coords);
                    // insertAfter(kgraph, tag, assignBaseTag);
                }
            }

            return kgraph;
        }
    }
}
