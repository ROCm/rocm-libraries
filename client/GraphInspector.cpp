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

#include "include/GraphInspector.hpp"

namespace rocRoller
{
    namespace Client
    {

        GraphInspector::GraphInspector(CommandPtr              command,
                                       CommandKernel&          kernel,
                                       CommandArguments const& runtimeArgs)
            : m_command(command)
            , m_kernel(kernel)
            , m_runtimeArgs(runtimeArgs)
            , m_kgraph(kernel.getKernelGraph())
            , m_coords(std::make_shared<KernelGraph::CoordinateGraph::CoordinateGraph>(
                  m_kgraph.coordinates))
            , m_tx(m_coords.get())
            , m_invocation(kernel.getKernelInvocation(runtimeArgs.runtimeArguments()))
        {
            m_tx.fillExecutionCoordinates(kernel.getContext());
            assignLiteralSizesAndStrides();
        }

        void GraphInspector::assignLiteralSizesAndStrides()
        {
            m_argValues = m_command->readArguments(m_runtimeArgs.runtimeArguments());

            auto pred = isNode<KernelGraph::CoordinateGraph::SubDimension>(m_coords);
            for(auto const& idx : m_coords->findElements(pred))
            {
                auto& el
                    = const_cast<typename KernelGraph::CoordinateGraph::CoordinateGraph::Element&>(
                        m_coords->getElement(idx));

                auto& subdim = std::get<KernelGraph::CoordinateGraph::SubDimension>(
                    std::get<KernelGraph::CoordinateGraph::Dimension>(el));

                auto sizeName = toString(subdim.size);

                if(m_argValues.count(sizeName) > 0)
                    subdim.size = Expression::literal(m_argValues.at(sizeName));

                auto strideName = toString(subdim.stride);
                if(m_argValues.count(strideName) > 0)
                    subdim.stride = Expression::literal(m_argValues.at(strideName));
            }
        }

        int GraphInspector::findLoadStoreTile(std::string const& argName)
        {
            auto predicate = [this, &argName](int idx) {
                if(m_coords->getElementType(idx) != Graph::ElementType::Node)
                    return false;

                auto user = m_coords->get<KernelGraph::CoordinateGraph::User>(idx);

                if(!user)
                    return false;

                return user->argumentName == argName;
            };

            auto coords = m_coords->findElements(predicate).to<std::vector>();
            AssertFatal(coords.size() == 1, ShowValue(coords.size()), ShowValue(argName));

            return coords[0];
        }

        int GraphInspector::findTensor(int userTag)
        {
            return findLoadStoreTile(concatenate("Tensor_", userTag, "_pointer"));
        }

        void GraphInspector::inventExecutionCoordinates()
        {
            setCoordinate<KernelGraph::CoordinateGraph::ForLoop>(5);
            setCoordinate<KernelGraph::CoordinateGraph::Workgroup>(0);
        }

        size_t GraphInspector::getLoadIndex(int coord)
        {
            std::vector<int> coords{coord};
            auto             exps = m_tx.reverse(coords);
            AssertFatal(exps.size() == 1);

            auto val = Expression::evaluate(exps[0], m_runtimeArgs.runtimeArguments());

            return std::visit(to_size_t, val);
        }

        size_t GraphInspector::getStoreIndex(int coord)
        {
            std::vector<int> coords{coord};
            auto             exps = m_tx.forward(coords);
            AssertFatal(exps.size() == 1);

            auto val = Expression::evaluate(exps[0], m_runtimeArgs.runtimeArguments());

            return std::visit(to_size_t, val);
        }

        std::tuple<int, int, int> GraphInspector::getMacroTileSizes() const
        {
            int macM = -1;
            int macN = -1;
            int macK = -1;

            for(auto const& macroTileTag :
                m_kernel.getKernelGraph()
                    .coordinates.getNodes<KernelGraph::CoordinateGraph::MacroTile>()
                    .to<std::vector>())
            {
                auto macroTile = m_kernel.getKernelGraph()
                                     .coordinates.getNode<KernelGraph::CoordinateGraph::MacroTile>(
                                         macroTileTag);
                if(macroTile.layoutType == LayoutType::MATRIX_A)
                {
                    macM = macroTile.sizes[0];
                    AssertFatal(macK == macroTile.sizes[1] || macK == -1);
                    macK = macroTile.sizes[1];
                }
                else if(macroTile.layoutType == LayoutType::MATRIX_B)
                {
                    macN = macroTile.sizes[1];
                    AssertFatal(macK == macroTile.sizes[0] || macK == -1);
                    macK = macroTile.sizes[0];
                }
                if(macM > 0 && macK > 0 && macN > 0)
                {
                    break;
                }
            }
            AssertFatal(macM > 0);
            AssertFatal(macN > 0);
            AssertFatal(macK > 0);

            return std::tuple<int, int, int>{macM, macN, macK};
        }

    }
}
