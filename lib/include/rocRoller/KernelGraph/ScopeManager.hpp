#pragma once

#include <memory>
#include <set>

#include <rocRoller/Context_fwd.hpp>

namespace rocRoller::KernelGraph
{

    /**
     * @brief Manage registers added to Scope operations in the
     * control graph.
     *
     * When the code-generator visits a Scope operation in the control
     * graph, it creates a ScopeManager and attaches it to the
     * Transformer.  Child operations can add registers (tags) to the
     * active scope.  When the scope is finished visiting all child
     * operations, it releases tracked registers.
     *
     * Recall that the Transformer is passed *by value* to subsequent
     * code-generation visitors: if the Transformer is manipulated,
     * child operations see the change, but parent operations do not.
     *
     * This facility implies that multiple scopes can exist in the
     * control graph, but that only a single scope is active at a
     * time.  Note that scopes do not "inherit" parent scopes.
     */
    class ScopeManager
    {
    public:
        ScopeManager() = delete;
        ScopeManager(std::shared_ptr<Context> context)
            : m_context(context)
        {
        }

        void add(int tag);
        void release();

    private:
        std::shared_ptr<Context> m_context;
        std::set<int>            m_tags;
    };

}
