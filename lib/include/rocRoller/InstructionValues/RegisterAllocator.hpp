
#pragma once

#include "Register.hpp"

namespace rocRoller
{
    namespace Register
    {
        class Allocator : public std::enable_shared_from_this<Allocator>
        {
        public:
            Allocator(Type regType, int count);

            Type regType() const;

            int size() const;
            int maxUsed() const;
            int useCount() const;

            void allocate(std::shared_ptr<Allocation> alloc);

            //> Allocate these specific registers.
            void allocate(std::shared_ptr<Allocation> alloc, std::vector<int> const& registers);
            //> Allocate these specific registers.
            void allocate(std::shared_ptr<Allocation> alloc, std::vector<int>&& registers);

            bool canAllocate(std::shared_ptr<const Allocation> alloc) const;

            std::vector<int> findFree(int count, Allocation::Options const& options) const;

            //> Returns the first free range matching the criteria. Returns -1 if no such range exists.
            int findContiguousRange(int start, int count, Allocation::Options const& options) const;

            constexpr static int Align(int start, Allocation::Options const& options);

            bool isFree(int idx) const;

            /**
             *  Normally, registers are implicitly freed when their allocation goes out of scope.
             *  This is for manual control only.
             */
            void free(std::vector<int> const& registers);

        private:
            Type m_regType;

            int m_maxUsed = -1;

            std::vector<std::weak_ptr<Allocation>> m_registers;
        };
    }
}

#include "RegisterAllocator_impl.hpp"
