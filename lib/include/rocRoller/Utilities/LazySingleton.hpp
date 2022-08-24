#pragma once

#include <memory>

namespace rocRoller
{
    template <typename Class>
    class LazySingleton
    {
    public:
        static std::shared_ptr<Class> getInstance()
        {
            if(!m_instance)
            {
                m_instance = std::make_shared<Class>();
            }
            return m_instance;
        }

        static void reset()
        {
            m_instance = nullptr;
        }

    private:
        static std::shared_ptr<Class> m_instance;
    };

    template <typename Class>
    std::shared_ptr<Class> LazySingleton<Class>::m_instance = nullptr;
}