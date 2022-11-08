#include <rocRoller/Context.hpp>
#include <rocRoller/KernelGraph/RegisterTagManager.hpp>
#include <rocRoller/KernelGraph/ScopeManager.hpp>

namespace rocRoller::KernelGraph
{
    void ScopeManager::add(int tag)
    {
        m_tags.insert(tag);
    }

    void ScopeManager::release()
    {
        for(auto tag : m_tags)
        {
            m_context->registerTagManager()->deleteRegister(tag);
        }
        m_tags.clear();
    }

}
