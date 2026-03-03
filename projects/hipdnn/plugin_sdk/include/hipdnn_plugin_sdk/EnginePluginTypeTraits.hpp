// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <type_traits>

#include <hip/hip_runtime.h>

namespace hipdnn_plugin_sdk
{

/**
 * @file EnginePluginTypeTraits.hpp
 * @brief Compile-time type traits for validating engine plugin types.
 *
 * This file provides type traits and validation functions to verify that
 * container and handle types meet the requirements for use with the
 * EnginePluginImpl.inl implementation.
 *
 * @see EnginePluginImpl.inl for the plugin implementation
 * @see SharedContainerManager.hpp for container lifecycle management
 */

// =============================================================================
// Container Type Traits
// =============================================================================

/**
 * @brief Check for getEngineManager() method.
 */
template <typename T, typename = void>
struct HasGetEngineManager : std::false_type
{
};

template <typename T>
struct HasGetEngineManager<T, std::void_t<decltype(std::declval<T&>().getEngineManager())>>
    : std::true_type
{
};

/**
 * @brief Check for static copyEngineIds method.
 */
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
 *
 * Required methods:
 * - EngineManager& getEngineManager()
 * - static uint32_t copyEngineIds(int64_t*, uint32_t, uint32_t&)
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

// =============================================================================
// Handle Type Traits
// =============================================================================

/**
 * @brief Check for container member of correct type.
 */
template <typename HandleType, typename ContainerType, typename = void>
struct HasContainerMember : std::false_type
{
};

template <typename HandleType, typename ContainerType>
struct HasContainerMember<HandleType,
                          ContainerType,
                          std::void_t<decltype(std::declval<HandleType&>().container)>>
    : std::is_same<decltype(std::declval<HandleType&>().container), std::shared_ptr<ContainerType>>
{
};

/**
 * @brief Check for setStream(hipStream_t) method.
 */
template <typename T, typename = void>
struct HasSetStream : std::false_type
{
};

template <typename T>
struct HasSetStream<
    T,
    std::void_t<decltype(std::declval<T&>().setStream(std::declval<hipStream_t>()))>>
    : std::true_type
{
};

/**
 * @brief Check for removeEngineDetailsDetachedBuffer(const void*) method.
 */
template <typename T, typename = void>
struct HasRemoveEngineDetailsDetachedBuffer : std::false_type
{
};

template <typename T>
struct HasRemoveEngineDetailsDetachedBuffer<
    T,
    std::void_t<decltype(std::declval<T&>().removeEngineDetailsDetachedBuffer(
        std::declval<const void*>()))>> : std::true_type
{
};

/**
 * @brief Validates that a handle type meets all requirements.
 *
 * This function uses static_assert to provide clear error messages if
 * a handle is missing required members or methods.
 *
 * Required:
 * - std::shared_ptr<ContainerType> container
 * - void setStream(hipStream_t)
 * - EngineManager& getEngineManager()
 * - void removeEngineDetailsDetachedBuffer(const void*)
 */
template <typename HandleType, typename ContainerType>
constexpr void validateHandleType()
{
    static_assert(HasContainerMember<HandleType, ContainerType>::value,
                  "Handle type must have a 'std::shared_ptr<ContainerType> container' member");

    static_assert(HasSetStream<HandleType>::value,
                  "Handle type must have a 'void setStream(hipStream_t)' method");

    static_assert(HasGetEngineManager<HandleType>::value,
                  "Handle type must have a 'EngineManager& getEngineManager()' method");

    static_assert(HasRemoveEngineDetailsDetachedBuffer<HandleType>::value,
                  "Handle type must have a 'void removeEngineDetailsDetachedBuffer(const void*)' "
                  "method");
}

} // namespace hipdnn_plugin_sdk
