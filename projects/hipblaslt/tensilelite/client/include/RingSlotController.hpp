// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>

namespace TensileLite
{
namespace Client
{
    class RingSlotController
    {
    public:
        explicit RingSlotController(size_t activeBufferCount = 2)
            : m_activeBufferCount(activeBufferCount)
        {
            if(m_activeBufferCount == 0 || m_activeBufferCount > 3)
            {
                throw std::invalid_argument(
                    "RingSlotController: activeBufferCount must be in [1, 3]");
            }
        }

        size_t activeBufferCount() const noexcept
        {
            return m_activeBufferCount;
        }

        size_t activeSlot() const noexcept
        {
            return m_activeSlot;
        }

        size_t availableSlots() const noexcept
        {
            return m_availableSlots;
        }

        bool needsCopyBarrier() const noexcept
        {
            return m_needsCopyBarrier;
        }

        bool hasAvailableSlot() const noexcept
        {
            return m_availableSlots > 0;
        }

        bool hasPendingWork() const noexcept
        {
            return hasAvailableSlot() || needsCopyBarrier();
        }

        std::optional<size_t> nextPrimeSlot() const noexcept
        {
            if(m_availableSlots >= m_activeBufferCount - 1)
                return std::nullopt;

            return (m_activeSlot + m_availableSlots + 1) % m_activeBufferCount;
        }

        void markSlotPrimed() noexcept
        {
            if(m_availableSlots < m_activeBufferCount - 1)
                ++m_availableSlots;
        }

        std::optional<size_t> advance() noexcept
        {
            if(m_availableSlots == 0)
                return std::nullopt;

            m_activeSlot = (m_activeSlot + 1) % m_activeBufferCount;
            --m_availableSlots;
            m_needsCopyBarrier = true;
            return m_activeSlot;
        }

        void markBarrierWaited() noexcept
        {
            m_needsCopyBarrier = false;
        }

        void cancel() noexcept
        {
            m_activeSlot       = 0;
            m_availableSlots   = 0;
            m_needsCopyBarrier = false;
        }

    private:
        size_t m_activeBufferCount = 2;
        size_t m_activeSlot        = 0;
        size_t m_availableSlots    = 0;
        bool   m_needsCopyBarrier  = false;
    };
}
}
