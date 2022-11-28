
#pragma once

#include <concepts>
#include <string>
#include <tuple>
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
            MetaObserver(ContextPtr ctx);
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

            static bool required(std::shared_ptr<Context> ctx);

        private:
            Tup m_tuple;
        };

    }
}

#include "MetaObserver_impl.hpp"
