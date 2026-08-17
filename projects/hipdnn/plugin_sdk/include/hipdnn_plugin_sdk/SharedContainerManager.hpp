// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>

namespace hipdnn_plugin_sdk
{

/**
 * @file SharedContainerManager.hpp
 * @brief Helper template for managing shared plugin container lifecycle.
 *
 * This file provides the SharedContainerManager class which implements the
 * shared_ptr/weak_ptr pattern for managing a single container instance that
 * is shared across multiple plugin handles.
 *
 * @see EnginePluginImpl.inl for the plugin implementation
 * @see EnginePluginTypeTraits.hpp for type validation
 */

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
 * class MyContainer { ... };
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
 * @tparam ContainerType The container type to manage.
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
        const std::lock_guard<std::mutex> lock(_mutex);

        auto containerPtr = _weakContainer.lock();
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
