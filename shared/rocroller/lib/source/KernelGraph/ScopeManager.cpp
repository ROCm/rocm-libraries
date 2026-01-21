// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{

    void ScopeManager::pushNewScope()
    {
        m_tags.push_back({});
    }

    void ScopeManager::addRegister(int tag)
    {
        // Do not add if the tag is already in scope.
        for(auto const& s : m_tags)
        {
            if(s.count(tag) > 0)
                return;
        }
        m_tags.back().insert(tag);
    }

    void ScopeManager::popAndReleaseScope()
    {
        for(auto tag : m_tags.back())
        {
            if(m_context->registerTagManager()->hasRegister(tag))
            {
                if(!hasDeallocate(*m_graph, tag))
                    m_context->registerTagManager()->deleteTag(tag);
            }
        }
        m_tags.pop_back();
    }

}
