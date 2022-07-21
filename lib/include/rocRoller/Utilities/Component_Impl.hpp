/**
 * @file Component_Impl.hpp
 *
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 *
 */
#pragma once

#include "Component.hpp"
#include "Error.hpp"
#include "Logging.hpp"
#include "Utils.hpp"

namespace rocRoller
{
    namespace Component
    {

        template <ComponentBase Base>
        requires(!CSingleUse<Base>) std::shared_ptr<Base> Get(typename Base::Argument const& arg)
        {
            using Factory = ComponentFactory<Base>;
            auto& factory = Factory::Instance();
            return factory.get(arg);
        }

        template <ComponentBase Base>
        requires(!CSingleUse<Base>) std::shared_ptr<Base> Get(typename Base::Argument&& arg)
        {
            using Factory = ComponentFactory<Base>;
            auto& factory = Factory::Instance();
            return factory.get(arg);
        }

        // clang-format off
        template <ComponentBase Base, typename Arg0, typename... Args>
        requires(!CSingleUse<Base>
              && !std::same_as<typename Base::Argument, Arg0>
              && std::constructible_from<typename Base::Argument, Arg0, Args...>)
        std::shared_ptr<Base> Get(Arg0 const& arg0, Args const&... args)
        {
            return Get<Base>(typename Base::Argument(arg0, args...));
        }
        // clang-format on

        template <ComponentBase Base>
        std::shared_ptr<Base> GetNew(typename Base::Argument const& arg)
        {
            using Factory = ComponentFactory<Base>;
            auto& factory = Factory::Instance();
            return factory.getNew(arg);
        }

        template <ComponentBase Base>
        std::shared_ptr<Base> GetNew(typename Base::Argument&& arg)
        {
            using Factory = ComponentFactory<Base>;
            auto& factory = Factory::Instance();
            return factory.getNew(arg);
        }

        // clang-format off
        template <ComponentBase Base, typename Arg0, typename... Args>
        requires(!std::same_as<typename Base::Argument, Arg0>
              && std::constructible_from<typename Base::Argument, Arg0, Args...>)
        std::shared_ptr<Base> GetNew(Arg0 const& arg0, Args const&... args)
        {
            return GetNew<Base>(typename Base::Argument(arg0, args...));
        }
        // clang-format on

        template <Component Comp>
        bool RegisterComponentImpl()
        {
            using Factory = ComponentFactory<typename Comp::Base>;
            auto& factory = Factory::Instance();
            return factory.template registerComponent<Comp>();
        }

        template <ComponentBase Base>
        ComponentFactory<Base>& ComponentFactory<Base>::Instance()
        {
            static ComponentFactory instance;

            return instance;
        }

        template <ComponentBase Base>
        template <typename T>
        std::shared_ptr<Base> ComponentFactory<Base>::get(T&& arg) const
        {
            auto iter = m_instanceCache.find(std::forward<T>(arg));

            if(iter != m_instanceCache.end())
            {
                return iter->second;
            }

            auto newInstance                      = getNew(std::forward<T>(arg));
            m_instanceCache[std::forward<T>(arg)] = newInstance;

            return newInstance;
        }

        template <ComponentBase Base>
        template <typename T>
        std::shared_ptr<Base> ComponentFactory<Base>::getNew(T&& arg) const
        {
            auto const& entry = getEntry(std::forward<T>(arg));

            return entry.builder(std::forward<T>(arg));
        }

        template <ComponentBase Base>
        template <typename T, bool Debug>
        typename ComponentFactory<Base>::Entry const&
            ComponentFactory<Base>::getEntry(T&& arg) const
        {
            auto iter = m_entryCache.find(std::forward<T>(arg));

            if(iter != m_entryCache.end())
            {
                return iter->second;
            }

            Entry const& e = findEntry<T, Debug>(std::forward<T>(arg));

            return m_entryCache[std::forward<T>(arg)] = e;
        }

        template <ComponentBase Base>
        template <typename T, bool Debug>
        typename ComponentFactory<Base>::Entry const&
            ComponentFactory<Base>::findEntry(T&& arg) const
        {
            auto foundIter = m_entries.end();

            for(auto iter = m_entries.begin(); iter != m_entries.end(); iter++)
            {
                if(iter->matcher(std::forward<T>(arg)))
                {
                    if(!Debug)
                        return *iter;

                    if(foundIter == m_entries.end())
                    {
                        foundIter = iter;
                    }
                    else
                    {
                        Log::warn("Multiple suitable components found for {}: {}, {}",
                                  Base::Name,
                                  foundIter->name,
                                  iter->name);
                    }
                }
            }

            AssertFatal(foundIter != m_entries.end(),
                        Base::Name,
                        ": No valid component found: ",
                        ShowValue(arg));

            return *foundIter;
        }

        template <ComponentBase Base>
        template <Component Comp>
        bool ComponentFactory<Base>::registerComponent()
        {
            return registerComponent(Comp::Name, Comp::Match, Comp::Build);
        }

        template <ComponentBase Base>
        bool ComponentFactory<Base>::registerComponent(std::string const& name,
                                                       Matcher<Base>      matcher,
                                                       Builder<Base>      builder)
        {
            for(auto const& entry : m_entries)
            {
                if(entry.name == name)
                    throw std::runtime_error(
                        concatenate("Duplicate ", Base::Name, " component names: '", name, "'"));
            }

            RegisterBase(this);

            m_entries.push_back({name, matcher, builder});

            return true;
        }

        template <ComponentBase Base>
        template <typename T>
        void ComponentFactory<Base>::emptyCache(T&& arg)
        {
            m_entryCache.erase(std::forward<T>(arg));
            m_instanceCache.erase(std::forward<T>(arg));
        }

        template <ComponentBase Base>
        void ComponentFactory<Base>::emptyCache()
        {
            m_entryCache.clear();
            m_instanceCache.clear();
        }

        inline void ComponentFactoryBase::ClearAllCaches()
        {
            for(auto ins : m_instances)
                ins->emptyCache();
        }

        inline void ComponentFactoryBase::RegisterBase(ComponentFactoryBase* base)
        {
            m_instances.insert(base);
        }

        // std::unordered_set<ComponentFactoryBase *> ComponentFactoryBase::m_instances;
    }
}
