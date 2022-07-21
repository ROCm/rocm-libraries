
#pragma once

#include "Register.hpp"
#include "RegisterAllocator.hpp"

#include "../Utilities/Error.hpp"

// Used for std::iota
#include <numeric>

namespace rocRoller
{
    namespace Register
    {
        inline Allocator::Allocator(Type regType, int count)
            : m_regType(regType)
            , m_registers(count)
        {
        }

        inline Type Allocator::regType() const
        {
            return m_regType;
        }

        inline void Allocator::allocate(std::shared_ptr<Allocation> alloc)
        {
            auto registers = findFree(alloc->registerCount(), alloc->options());

            if(registers.empty())
                throw std::runtime_error("No more registers!");

            allocate(alloc, std::move(registers));
        }

        inline void Allocator::allocate(std::shared_ptr<Allocation> alloc,
                                        std::vector<int> const&     registers)
        {
            for(auto idx : registers)
            {
                m_registers[idx] = alloc;
                m_maxUsed        = std::max(m_maxUsed, idx);
            }

            alloc->setAllocation(shared_from_this(), registers);
        }

        inline void Allocator::allocate(std::shared_ptr<Allocation> alloc,
                                        std::vector<int>&&          registers)
        {
            for(auto idx : registers)
            {
                m_registers[idx] = alloc;
                m_maxUsed        = std::max(m_maxUsed, idx);
            }

            alloc->setAllocation(shared_from_this(), std::move(registers));
        }

        inline bool Allocator::canAllocate(std::shared_ptr<const Allocation> alloc) const
        {
            auto theRegisters = findFree(alloc->registerCount(), alloc->options());
            return !theRegisters.empty();
        }

        inline std::vector<int> Allocator::findFree(int                        count,
                                                    Allocation::Options const& options) const
        {
            if(options.contiguous)
            {
                auto loc = findContiguousRange(0, count, options);
                if(loc >= 0)
                {
                    // auto range = std::ranges::iota_view{loc, loc+count};
                    // return {range.begin(), range.end()};
                    std::vector<int> rv(count);
                    std::iota(rv.begin(), rv.end(), loc);
                    return rv;
                }
                else
                    return {};
            }

            std::vector<int> rv;
            rv.reserve(count);

            int idx = 0;

            while(rv.size() < count)
            {
                idx = findContiguousRange(idx, 1, options);

                if(idx < 0)
                    return {};

                rv.push_back(idx);

                idx++;
            }

            return rv;
        }

        inline int Allocator::findContiguousRange(int                        start,
                                                  int                        count,
                                                  Allocation::Options const& options) const
        {
            if(options.alignment > 1)
                start = Align(start, options);

            while(start + count < m_registers.size())
            {
                bool good = true;
                for(int i = start; i < start + count; i++)
                {
                    if(!isFree(i))
                    {
                        good = false;
                        break;
                    }
                }

                if(good)
                    return start;

                start = Align(start + 1, options);
            }

            return -1;
        }

        constexpr inline int Allocator::Align(int start, Allocation::Options const& options)
        {
            AssertFatal(options.alignment <= 1 || options.alignmentPhase < options.alignment,
                        ShowValue(options.alignment),
                        ShowValue(options.alignmentPhase));

            if(options.alignment <= 1)
                return start;

            while((start % options.alignment) != options.alignmentPhase)
                start++;

            return start;
        }

        inline int Allocator::size() const
        {
            return m_registers.size();
        }

        inline int Allocator::maxUsed() const
        {
            return m_maxUsed;
        }

        inline int Allocator::useCount() const
        {
            return maxUsed() + 1;
        }

        inline bool Allocator::isFree(int idx) const
        {
            return m_registers[idx].expired();
        }

        inline void Allocator::free(std::vector<int> const& registers)
        {
            for(int idx : registers)
                m_registers[idx].reset();
        }
    }
}
