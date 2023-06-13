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
        ScopeManager(ContextPtr context)
            : m_context(context)
        {
        }

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
        std::vector<std::unordered_set<int>> m_tags;
    };

}
