#pragma once

#include <rocRoller/Scheduling/Scheduling.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        class VALUWriteFollowedByMFMARead
        {
        public:
            VALUWriteFollowedByMFMARead(std::shared_ptr<Context> context)
                : m_context(context){};

            InstructionStatus peek(Instruction const& inst) const;
            void              modify(Instruction& inst) const;
            InstructionStatus observe(Instruction const& inst);

            static bool required(std::shared_ptr<Context> context)
            {
                return true;
                // TODO: specialize this rule to 90a when MFMA rules for 908 implemented
                // return context->targetArchitecture().target().getVersionString() == "gfx90a";
            }

        private:
            std::weak_ptr<Context> m_context;
            int const              nops = 2;

            int getNops(Instruction const& inst) const;
        };

        static_assert(CObserver<VALUWriteFollowedByMFMARead>);
    }
}
