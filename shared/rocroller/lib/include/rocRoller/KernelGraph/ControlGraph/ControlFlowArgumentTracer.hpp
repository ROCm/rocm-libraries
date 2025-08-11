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

#pragma once

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

#include <rocRoller/KernelGraph/RegisterTagManager_fwd.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller::KernelGraph
{
    class ControlFlowArgumentTracer
    {
    public:
        ControlFlowArgumentTracer(KernelGraph const& kgraph, AssemblyKernelPtr const& kernel);

        std::unordered_set<std::string> const& referencedArguments(int controlNode) const;

        /**
         * Map of control node -> arguments referenced by that control node.
         */
        std::unordered_map<int, std::unordered_set<std::string>> const& referencedArguments() const;

        /**
         * Any arguments that we know will never be referenced in the kernel.
         */
        std::set<std::string> const& neverReferencedArguments() const;

    private:
        std::unordered_map<int, std::unordered_set<std::string>> m_referencedArguments;

        std::set<std::string> m_neverReferencedArguments;
    };
}
