
#pragma once

#include "Register.hpp"

class RegisterTest_RegisterToString_Test;

namespace rocRoller
{
    namespace Register
    {
        class Allocator : public std::enable_shared_from_this<Allocator>
        {
        public:
            Allocator(Type regType, int count);

            Type regType() const;

            size_t size() const;
            int    maxUsed() const;
            int    useCount() const;
            int    currentlyFree() const;

            void allocate(AllocationPtr alloc);

            /**
             * Reassigns the registers in `indices` to a new `Allocation` that is returned.
             * The indices must all be currently assigned to the same `Allocation`.
             */
            template <std::ranges::forward_range T>
            AllocationPtr reassign(T const& indices);

            bool canAllocate(std::shared_ptr<const Allocation> alloc) const;

            std::vector<int> findFree(int count, Allocation::Options const& options) const;

            //> Returns the first free range matching the criteria. Returns -1 if no such range exists.
            int findContiguousRange(int start, int count, Allocation::Options const& options) const;

            constexpr static int Align(int start, Allocation::Options const& options);

            bool isFree(int idx) const;

            std::string toString() const;

            /**
             *  Normally, registers are implicitly freed when their allocation goes out of scope.
             *  This is for manual control only.
             */
            void free(std::vector<int> const& registers);

        private:
            friend class ::RegisterTest_RegisterToString_Test;

            //> Allocate these specific registers.
            void allocate(AllocationPtr alloc, std::vector<int> const& registers);
            //> Allocate these specific registers.
            void allocate(AllocationPtr alloc, std::vector<int>&& registers);

            Type m_regType;

            int m_maxUsed = -1;

            std::vector<std::weak_ptr<Allocation>> m_registers;
        };
    }
}

#include "RegisterAllocator_impl.hpp"
