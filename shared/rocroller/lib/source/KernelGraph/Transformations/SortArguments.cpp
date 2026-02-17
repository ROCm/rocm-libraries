/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2026 AMD ROCm(TM) Software
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

#include "rocRoller/AssemblyKernelArgument.hpp"
#include <rocRoller/KernelGraph/Transforms/SortArguments.hpp>
#include <rocRoller/KernelGraph/Transforms/SortArguments_detail.hpp>

#include <rocRoller/KernelGraph/TopoVisitor.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer.hpp>

namespace rocRoller::KernelGraph
{
    namespace SortArguments_detail
    {
        class ArgumentFirstUseVisitor : public TopoControlGraphVisitor<ArgumentFirstUseVisitor>
        {
        public:
            ArgumentFirstUseVisitor(KernelGraph const& graph, AssemblyKernelPtr kernel)
                : TopoControlGraphVisitor(graph)
                , m_kernel(kernel)
                , m_argTracer(graph, kernel)
            {
            }

            void operator()(int node, auto const& op)
            {
                auto args = m_argTracer.referencedArguments(node);
                for(auto const& arg : args)
                {
                    if(!m_argumentFirstUse.contains(arg))
                        m_argumentFirstUse[arg] = m_nextArgumentIndex++;
                }
            }

            int argumentFirstUse(std::string const& arg) const
            {
                return m_argumentFirstUse.at(arg);
            }

        private:
            AssemblyKernelPtr         m_kernel;
            ControlFlowArgumentTracer m_argTracer;
            int                       m_nextArgumentIndex = 0;

            std::unordered_map<std::string, int> m_argumentFirstUse;
        };

        void sortArgumentsByFirstUse(KernelGraph const&                   graph,
                                     AssemblyKernelPtr                    kernel,
                                     std::vector<AssemblyKernelArgument>& arguments)
        {
            ArgumentFirstUseVisitor visitor(graph, kernel);
            visitor.walk();

            std::ranges::stable_sort(arguments, [&](auto const& a, auto const& b) {
                return visitor.argumentFirstUse(a.getName()) < visitor.argumentFirstUse(b.getName());
            });
        }
    }

    KernelGraph SortArguments::apply(KernelGraph const& graph)
    {
        auto kernel    = m_context->kernel();
        auto arguments = kernel->arguments();

        Log::debug("SortArguments: Before sorting by first use:");
        for(auto const& arg : arguments)
        {
            Log::debug("Argument: {} ({})", arg.getName(), arg.getSize());
        }

        SortArguments_detail::sortArgumentsByFirstUse(graph, kernel, arguments);

        Log::debug("SortArguments: After sorting by first use:");
        for(auto const& arg : arguments)
        {
            Log::debug("Argument: {} ({})", arg.getName(), arg.getSize());
        }

        m_context->argLoader()->decidePreloadedKernargs(arguments);

        Log::debug("SortArguments: After deciding preloaded kernargs:");
        for(auto const& arg : arguments)
        {
            Log::debug("Argument: {} ({})", arg.getName(), arg.getSize());
        }

        kernel->resetArguments();

        for(auto& arg : arguments)
        {
            arg.setOffset(-1);
            m_context->kernel()->addArgument(arg);
        }

        return graph;
    }
}