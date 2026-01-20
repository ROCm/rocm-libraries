// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <mutex>

#include <hipdnn_plugin_sdk/EngineManager.hpp>

namespace hipdnn_plugin_sdk
{

/**
 * @brief Base class for engine plugin containers.
 *
 * The EnginePluginContainer encapsulates the lifecycle of an engine plugin
 * from creation to destruction. It acts as the root object that manages
 * the EngineManager and provides the bridge between the C API and the
 * C++ implementation.
 *
 * The default implementation uses a shared_ptr/weak_ptr pattern to allow
 * the plugin to be loaded multiple times while sharing resources. When the
 * last reference drops, the container is cleaned up.
 *
 * Plugin developers can extend this class to add custom initialization,
 * resources, or behavior.
 *
 * ## Usage Pattern
 *
 * 1. Derive from EnginePluginContainer
 * 2. Override registerEngines() to add your engines to the manager
 * 3. Use getOrCreateShared() to get/create the singleton instance
 *
 * @note This class is designed to be shared across multiple plugin handles.
 *       Implementations should be thread-safe.
 */
class EnginePluginContainer
{
public:
    EnginePluginContainer()
        : _engineManager(std::make_unique<EngineManager>())
    {
    }

    virtual ~EnginePluginContainer() = default;

    // Disallow copy
    EnginePluginContainer(const EnginePluginContainer&) = delete;
    EnginePluginContainer& operator=(const EnginePluginContainer&) = delete;

    /**
     * @brief Gets the engine manager for this container.
     *
     * @return Reference to the engine manager.
     */
    EngineManager& getEngineManager()
    {
        return *_engineManager;
    }

    /**
     * @brief Gets the engine manager for this container (const version).
     *
     * @return Const reference to the engine manager.
     */
    const EngineManager& getEngineManager() const
    {
        return *_engineManager;
    }

protected:
    /**
     * @brief Override this method to register engines with the manager.
     *
     * This method is called during container initialization. Derived classes
     * should create their engines and plan builders here and add them to
     * the engine manager.
     *
     * @note This method is called once when the container is first created.
     */
    virtual void registerEngines()
    {
        // Default implementation does nothing.
        // Derived classes should override to register their engines.
    }

    /**
     * @brief Gets the engine manager for modification during initialization.
     *
     * Use this in registerEngines() to add engines to the manager.
     *
     * @return Reference to the engine manager.
     */
    EngineManager& engineManager()
    {
        return *_engineManager;
    }

private:
    std::unique_ptr<EngineManager> _engineManager;
};

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
template<typename ContainerType>
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

        // Double-check after acquiring lock
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
