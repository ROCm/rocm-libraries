/**
 * @file Component.hpp
 *
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 *
 */

#pragma once

#include <concepts>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace rocRoller
{
    namespace Component
    {
        template <typename T>
        concept CSingleUse = requires()
        {
            requires static_cast<bool>(T::SingleUse) == true;
        };

        /**
         * @brief A `ComponentBase` is a base class for a category of components.
         *
         *  - It defines the interface for accessing the implementations.
         *  - All subclasses should provide interchangeable functionality.
         *
         */
        template <typename T>
        concept ComponentBase = requires(T& a)
        {
            /**
             * The Argument type should be either a single type, or a tuple of types needed to
             * pick the correct component implementation.
             */
            typename T::Argument;
            requires std::default_initializable<std::hash<typename T::Argument>>;
            requires std::copy_constructible<typename T::Argument>;

            //{ a.name() } -> std::convertible_to<std::string>;
            //{ T::Match(c) } -> std::convertible_to<bool>;

            {
                T::Name
                } -> std::convertible_to<std::string>;
        };

        /**
         * A function to match whether a given component is appropriate for the situation.
         *
         * Only one subclass should match any given situation.
         */
        template <ComponentBase Base>
        using Matcher = std::function<bool(typename Base::Argument)>;

        /**
         * A factory function to create an instance of a particular subclass.
         */
        template <ComponentBase Base>
        using Builder = std::function<std::shared_ptr<Base>(typename Base::Argument)>;

        /**
         * A concrete subclass which fulfils the required functionality in a subset of situations.
         */
        template <typename T>
        concept Component = requires(T a)
        {
            typename T::Base;
            typename T::Base::Argument;

            // Single-use status can't depend on implementation.
            requires CSingleUse<T>
            == CSingleUse<typename T::Base>;

            requires std::derived_from<T, typename T::Base>;
            requires ComponentBase<typename T::Base>;

            {
                T::Match
                } -> std::convertible_to<Matcher<typename T::Base>>;
            {
                T::Build
                } -> std::convertible_to<Builder<typename T::Base>>;

            {
                T::Name
                } -> std::convertible_to<std::string>;
            {
                T::Basename
                } -> std::convertible_to<std::string>;
        };

        /**
         * @brief Returns an object of the appropriate subclass of `Base`, based on `ctx`.
         * May be the same object from one call to the next.
         */
        template <ComponentBase Base>
        requires(!CSingleUse<Base>) std::shared_ptr<Base> Get(typename Base::Argument const& ctx);

        template <ComponentBase Base>
        requires(!CSingleUse<Base>) std::shared_ptr<Base> Get(typename Base::Argument&& ctx);

        /**
         * @brief Convenience function which will construct a tuple and call the above implementation of
         * May be the same object from one call to the next.
         */
        // clang-format off
        template <ComponentBase Base, typename Arg0, typename... Args>
        requires(!CSingleUse<Base>
              && !std::same_as<typename Base::Argument, Arg0>
              && std::constructible_from<typename Base::Argument, Arg0, Args...>)
        std::shared_ptr<Base> Get(Arg0 const& arg0, Args const&... args);
        // clang-format on

        /**
         * @brief Returns a new instance object of the appropriate subclass of `Base`, based
         * on `ctx`.
         */
        template <ComponentBase Base>
        std::shared_ptr<Base> GetNew(typename Base::Argument const& ctx);

        template <ComponentBase Base>
        std::shared_ptr<Base> GetNew(typename Base::Argument&& ctx);

        // clang-format off
        template <ComponentBase Base, typename Arg0, typename... Args>
        requires(!std::same_as<typename Base::Argument, Arg0>
              && std::constructible_from<typename Base::Argument, Arg0, Args...>)
        std::shared_ptr<Base> GetNew(Arg0 const& arg0, Args const&... args);
        // clang-format on

        template <Component Comp>
        bool RegisterComponentImpl();

#define RegisterComponentBaseCustom(base, name) const std::string base::Name = #name

#define RegisterComponentBase(base) RegisterComponentBaseCustom(base, #base)

#define VAR_CAT2(a, b) a##b
#define VAR_CAT(a, b) VAR_CAT2(a, b)
#define RegisterComponentCustom(component, name)                        \
    const std::string component::Name     = name;                       \
    const std::string component::Basename = component::Base::Name;      \
    namespace                                                           \
    {                                                                   \
        auto VAR_CAT(_component_, __LINE__)                             \
            = rocRoller::Component::RegisterComponentImpl<component>(); \
    }

#define RegisterComponent(component) RegisterComponentCustom(component, #component)

        class ComponentFactoryBase
        {
        public:
            static void ClearAllCaches();

            virtual void emptyCache() = 0;

            static void RegisterBase(ComponentFactoryBase* base);

        private:
            inline static std::unordered_set<ComponentFactoryBase*> m_instances;
        };

        template <ComponentBase Base>
        class ComponentFactory : public ComponentFactoryBase
        {
        public:
            using Argument = typename Base::Argument;

            struct Entry
            {
                std::string   name;
                Matcher<Base> matcher;
                Builder<Base> builder;
            };

            static ComponentFactory& Instance();

            template <typename T>
            std::shared_ptr<Base> get(T&& arg) const;
            template <typename T>
            std::shared_ptr<Base> getNew(T&& arg) const;

            template <typename T, bool Debug = false>
            Entry const& getEntry(T&& arg) const;

            template <Component Comp>
            bool registerComponent();

            bool registerComponent(std::string const& name,
                                   Matcher<Base>      matcher,
                                   Builder<Base>      builder);

            template <typename T>
            void emptyCache(T&& arg);

            virtual void emptyCache() override;

        private:
            std::vector<Entry> m_entries;

            /**
             * Finds an entry among the registered entries (classes).  This is the fallback for if there
             * is no entry in the cache.
             */
            template <typename T, bool Debug = false>
            Entry const& findEntry(T&& arg) const;

            mutable std::unordered_map<Argument, Entry>                 m_entryCache;
            mutable std::unordered_map<Argument, std::shared_ptr<Base>> m_instanceCache;
        };

    }
}

#include "Component_Impl.hpp"
