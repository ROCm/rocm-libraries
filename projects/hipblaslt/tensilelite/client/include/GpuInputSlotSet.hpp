// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/Tensile_fwd.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace TensileLite::Client
{
    struct GpuInputSlot
    {
        std::vector<void*>                    ptrs;
        std::vector<void**>                   batchPtrs;
        std::shared_ptr<TensileLite::ProblemInputs> cachedInputs;

        bool populated() const noexcept
        {
            return !ptrs.empty();
        }

        void clear() noexcept
        {
            ptrs.clear();
            batchPtrs.clear();
            cachedInputs.reset();
        }
    };

    template <size_t MaxSlots>
    class GpuInputSlotSet
    {
    public:
        using slot_type = GpuInputSlot;

        GpuInputSlot& at(size_t slot)
        {
            return m_slots.at(slot);
        }

        GpuInputSlot const& at(size_t slot) const
        {
            return m_slots.at(slot);
        }

        bool populated(size_t slot) const
        {
            return at(slot).populated();
        }

        void clear(size_t slot)
        {
            at(slot).clear();
        }

        void clearFrom(size_t firstSlot)
        {
            if(firstSlot > MaxSlots)
                throw std::out_of_range("GpuInputSlotSet clearFrom slot out of range.");

            for(size_t slot = firstSlot; slot < MaxSlots; ++slot)
            {
                clear(slot);
            }
        }

    private:
        std::array<slot_type, MaxSlots> m_slots{};
    };
} // namespace TensileLite::Client
