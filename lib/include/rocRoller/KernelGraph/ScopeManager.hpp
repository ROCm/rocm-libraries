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

#include <memory>
#include <unordered_set>
#include <vector>

#include <rocRoller/Context_fwd.hpp>

namespace rocRoller::KernelGraph
{

    /**
     * @brief Manage registers added to Scope operations in the
     * control graph.
     *
     * When the code-generator visits the Kernel operation in the
     * control graph, it creates a ScopeManager and attaches it to the
     * Transformer.
     *
     * When the code-generator visits a Scope operation, it pushes a
     * new scope.  Child operations can add registers (tags) to the
     * active (top-most) scope.  When the scope is finished visiting
     * all child operations, it pops the current scope and releases
     * registers from the popped scope.
     *
     * This facility implies that multiple scopes can exist in the
     * control graph.  When adding a tag to the current scope, the
     * ScopeManager checks all other scopes in the stack to see if the
     * register already exists.  If not, the register is added to the
     * current scope.
     */
    class ScopeManager
    {
    public:
        ScopeManager() = delete;
        ScopeManager(ContextPtr context, KernelGraphPtr graph)
            : m_context(context)
            , m_graph(graph){};

        /**
         * @brief Create a new Scope and push it onto the scope stack.
         */
        void pushNewScope();

        /**
         * @brief Add a register to the top-most scope.
         */
        void addRegister(int tag);

        /**
         * @brief Release all registers in the top-most scope and pop
         * it.
         */
        void popAndReleaseScope();

    private:
        ContextPtr                           m_context;
        KernelGraphPtr                       m_graph;
        std::vector<std::unordered_set<int>> m_tags;
    };

}
