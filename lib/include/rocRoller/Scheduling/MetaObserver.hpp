
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Scheduling.hpp"

#include "../CodeGen/Instruction.hpp"
#include "../CodeGen/WaitCount.hpp"
#include "../Context.hpp"
#include "../InstructionValues/Register.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        template <CObserver... Types>
        class MetaObserver : public IObserver
        {
        public:
            using Tup = std::tuple<Types...>;
            enum
            {
                Size = std::tuple_size<Tup>::value
            };

            MetaObserver();
            MetaObserver(Tup const& tup);

            ~MetaObserver();

            //> Speculatively predict stalls if this instruction were scheduled now.
            virtual InstructionStatus peek(Instruction const& inst) const override;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            virtual void modify(Instruction& inst) const override;

            //> This instruction _will_ be scheduled now, record any side effects.
            // TODO: Should this return void? Is the InstructionStatus object useful here?
            virtual InstructionStatus observe(Instruction const& inst) override;

        private:
#if 0
            template <size_t Index> requires CanIndexTup
            InstructionStatus peek_at(Instruction const& inst);

            template <size_t Index> requires (Index < Size)
            void modify_at(Instruction & inst);

            template <size_t Index> requires (Index < Size)
            InstructionStatus observe_at(Instruction const& inst);

            template <size_t Index> requires (Index < Size)
            InstructionStatus peek_above(Instruction const& inst);

            template <size_t Index> requires (Index < (Size-1))
            void modify_above(Instruction & inst);

            template <size_t Index> requires (Index < (Size-1))
            InstructionStatus observe_above(Instruction const& inst);
#endif

            Tup m_tuple;
        };

    }
}

#include "MetaObserver_impl.hpp"
