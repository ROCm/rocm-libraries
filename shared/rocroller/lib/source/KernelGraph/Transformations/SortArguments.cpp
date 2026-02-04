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

#include <rocRoller/KernelGraph/TopoVisitor.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowArgumentTracer.hpp>
// #include <rocRoller/CommandSolution.hpp>
// #include <rocRoller/Expression.hpp>
// #include <rocRoller/ExpressionTransformations.hpp>

// #include <rocRoller/KernelGraph/KernelGraph.hpp>
// #include <rocRoller/KernelGraph/Visitors.hpp>

// #include <rocRoller/Operations/Command.hpp>

namespace rocRoller
{
    namespace KernelGraph
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

        std::vector<AssemblyKernelArgument>
            FillAlignmentGaps(std::deque<AssemblyKernelArgument> arguments,
                              std::set<std::string>              launchTimeOnlyArguments)
        {
            std::vector<AssemblyKernelArgument> newArguments;

            int offset = 0;
            while(!arguments.empty())
            {
                auto argToAppend = arguments.begin();

                auto nextOffset = RoundUpToMultiple<int>(offset, argToAppend->size);

                if(nextOffset > offset)
                {
                    auto gapSize = nextOffset - offset;
                    Log::critical("Gap size: {}", gapSize);
                    for(; argToAppend != arguments.end(); ++argToAppend)
                    {
                        if(launchTimeOnlyArguments.contains(argToAppend->name))
                        {
                            argToAppend = arguments.end();
                            break;
                        }

                        if(argToAppend->size == gapSize)
                        {
                            Log::critical(
                                "Found argument: {} ({})", argToAppend->name, argToAppend->size);
                            break;
                        }
                    }

                    if(argToAppend == arguments.end())
                        argToAppend = arguments.begin();
                }

                Log::critical("Adding argument: {} ({})", argToAppend->name, argToAppend->size);
                newArguments.push_back(*argToAppend);
                offset = RoundUpToMultiple<int>(offset, argToAppend->size) + argToAppend->size;

                arguments.erase(argToAppend);
            }

            return newArguments;
        }

        KernelGraph SortArguments::apply(KernelGraph const& graph)
        {

            ArgumentFirstUseVisitor visitor(graph, m_context->kernel());

            visitor.walk();

            auto launchTimeOnlyArguments = m_context->kernel()->launchTimeOnlyArguments();

            auto arguments = [&]() -> std::deque<AssemblyKernelArgument> {
                auto tmp = m_context->kernel()->resetArguments();
                return {tmp.begin(), tmp.end()};
            }();

            Log::critical("Arguments Before:");
            for(auto const& arg : arguments)
            {
                Log::critical("Argument: {} ({})", arg.name, arg.size);
            }

            std::ranges::sort(arguments, [&](auto const& a, auto const& b) {
                auto aLaunchTimeOnly = launchTimeOnlyArguments.contains(a.name);
                auto bLaunchTimeOnly = launchTimeOnlyArguments.contains(b.name);
                if(aLaunchTimeOnly && !bLaunchTimeOnly)
                    return false;
                if(!aLaunchTimeOnly && bLaunchTimeOnly)
                    return true;

                return visitor.argumentFirstUse(a.name) < visitor.argumentFirstUse(b.name);
            });

            auto newArguments = FillAlignmentGaps(arguments, launchTimeOnlyArguments);
            // std::vector<AssemblyKernelArgument> newArguments{arguments.begin(), arguments.end()};

            Log::critical("Arguments After:");
            for(auto const& arg : newArguments)
            {
                std::string ltOnly = launchTimeOnlyArguments.contains(arg.name) ? " (LT only)" : "";
                Log::critical("Argument: {} ({}){}", arg.name, arg.size, ltOnly);
            }

            for(auto& arg : newArguments)
            {
                if(!launchTimeOnlyArguments.contains(arg.name))
                {
                    arg.offset = -1;
                    m_context->kernel()->addArgument(arg);
                }
            }

            return graph;
        }
    }
}