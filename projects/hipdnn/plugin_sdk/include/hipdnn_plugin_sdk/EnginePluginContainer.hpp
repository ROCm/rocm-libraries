// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>
#include <type_traits>

#include <hipdnn_plugin_sdk/EngineManager.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Compile-time checks for engine plugin container requirements.
 *
 * These type traits verify that a container class meets the requirements
 * for use with DECLARE_ENGINE_PLUGIN_DEFAULT_IMPL macro.
 *
 * Required methods:
 * 1. static uint32_t copyEngineIds(int64_t*, uint32_t, uint32_t&)
 * 2. EngineManager& getEngineManager()
 */

// Check for getEngineManager() method
template <typename T, typename = void>
struct HasGetEngineManager : std::false_type
{
};

template <typename T>
struct HasGetEngineManager<T, std::void_t<decltype(std::declval<T&>().getEngineManager())>>
    : std::true_type
{
};

// Check for static copyEngineIds method
template <typename T, typename = void>
struct HasCopyEngineIds : std::false_type
{
};

template <typename T>
struct HasCopyEngineIds<
    T,
    std::void_t<decltype(T::copyEngineIds(
        std::declval<int64_t*>(), std::declval<uint32_t>(), std::declval<uint32_t&>()))>>
    : std::true_type
{
};

/**
 * @brief Validates that a container type meets all requirements.
 *
 * This function uses static_assert to provide clear error messages if
 * a container is missing required methods.
 */
template <typename ContainerType>
constexpr void validateContainerType()
{
    static_assert(HasGetEngineManager<ContainerType>::value,
                  "Container type must have a 'EngineManager& getEngineManager()' method");

    static_assert(HasCopyEngineIds<ContainerType>::value,
                  "Container type must have a 'static uint32_t copyEngineIds(int64_t*, uint32_t, "
                  "uint32_t&)' method");
}

/**
 * @brief Helper template for managing a shared plugin container instance.
 *
 * This template provides the shared_ptr/weak_ptr pattern described in the RFC
 * for managing a single container instance that is shared across multiple
 * plugin handles.
 *
 * ## Usage
 *
 * ```cpp
 * // In your plugin implementation:
 * class MyContainer : public EnginePluginContainer { ... };
 *
 * static SharedContainerManager<MyContainer> containerManager;
 *
 * // In hipdnnEnginePluginCreate:
 * auto container = containerManager.getOrCreate();
 * handle->container = container;
 *
 * // When all handles are destroyed, the container is automatically cleaned up.
 * ```
 *
 * @tparam ContainerType The derived container type to manage.
 */
template <typename ContainerType>
class SharedContainerManager
{
public:
    SharedContainerManager() = default;

    /**
     * @brief Gets the existing container or creates a new one.
     *
     * Thread-safe. If a container already exists and hasn't been destroyed,
     * returns a shared_ptr to it. Otherwise, creates a new container.
     *
     * @return Shared pointer to the container.
     */
    std::shared_ptr<ContainerType> getOrCreate()
    {
        auto containerPtr = _weakContainer.lock();
        if(containerPtr != nullptr)
        {
            return containerPtr;
        }

        std::lock_guard<std::mutex> lock(_mutex);

        // if we do have a race condition that results in threads getting locked, we want to
        // ensure that we only create one instance.  Therefore, the second thread to get
        // through will just read from the weak pointer rather than create a new instance.
        containerPtr = _weakContainer.lock();
        if(containerPtr != nullptr)
        {
            return containerPtr;
        }

        containerPtr = std::make_shared<ContainerType>();
        _weakContainer = containerPtr;
        return containerPtr;
    }

private:
    std::weak_ptr<ContainerType> _weakContainer;
    std::mutex _mutex;
};

} // namespace hipdnn_plugin_sdk
