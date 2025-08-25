/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#include <rocRoller/Utilities/Component.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

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
            using Factory       = ComponentFactory<Base>;
            auto const& factory = Factory::Instance();
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
            auto iter = m_instanceCache.find(arg);

            if(iter != m_instanceCache.end())
            {
                return iter->second;
            }

            auto newInstance                      = getNew(arg);
            m_instanceCache[std::forward<T>(arg)] = newInstance;

            return newInstance;
        }

        template <ComponentBase Base>
        template <typename T>
        std::shared_ptr<Base> ComponentFactory<Base>::getNew(T&& arg) const
        {
            auto const& entry = getEntry(arg);

            return entry.builder(std::forward<T>(arg));
        }

        template <ComponentBase Base>
        template <typename T, bool Debug>
        typename ComponentFactory<Base>::Entry const&
            ComponentFactory<Base>::getEntry(T&& arg) const
        {
            auto iter = m_entryCache.find(arg);

            if(iter != m_entryCache.end())
            {
                return iter->second;
            }

            Entry const& e = findEntry<T, Debug>(arg);

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
                        rocRoller::Log::warn("Multiple suitable components found for {}: {}, {}",
                                             Base::Basename,
                                             foundIter->name,
                                             iter->name);
                    }
                }
            }

            // TODO Investigate why cppcheck isn't ensuring that this macro
            // is expanded correctly, requiring the derefInvalidIterator
            // suppression.
            AssertFatal(foundIter != m_entries.end(),
                        Base::Basename,
                        ": No valid component found: ",
                        ShowValue(arg));

            // cppcheck-suppress derefInvalidIterator
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
            auto sameName = [&name](auto const& entry) { return entry.name == name; };
            if(std::any_of(m_entries.begin(), m_entries.end(), sameName))
                throw std::runtime_error(
                    concatenate("Duplicate ", Base::Basename, " component names: '", name, "'"));

            RegisterBase(this);

            m_entries.push_back({name, matcher, builder});

            return true;
        }

        template <ComponentBase Base>
        template <typename T>
        void ComponentFactory<Base>::emptyCache(T&& arg)
        {
            m_entryCache.erase(arg);
            m_instanceCache.erase(arg);
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
