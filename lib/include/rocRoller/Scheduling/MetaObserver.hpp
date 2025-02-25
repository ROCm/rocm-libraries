
#pragma once

#include <concepts>
#include <string>
#include <tuple>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/WaitCount.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/Scheduling/Scheduling.hpp>

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
            MetaObserver(ContextPtr ctx);
            MetaObserver(Tup const& tup);

            ~MetaObserver();

            //> Speculatively predict stalls if this instruction were scheduled now.
            virtual InstructionStatus peek(Instruction const& inst) const override;

            //> Add any waitcnt or nop instructions needed before `inst` if it were to be scheduled now.
            //> Throw an exception if it can't be scheduled now.
            virtual void modify(Instruction& inst) const override;

            //> This instruction _will_ be scheduled now, record any side effects.
            virtual void observe(Instruction const& inst) override;

        private:
            Tup m_tuple;
        };

    }
}

#include <rocRoller/Scheduling/MetaObserver_impl.hpp>
