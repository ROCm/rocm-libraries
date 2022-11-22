#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/GenericWaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rule for VALU Write followed by an MFMA Read requiring 2 NOPs
         *
         */
        class VALUWriteFollowedByMFMARead : public GenericWaitStateObserver
        {
        public:
            VALUWriteFollowedByMFMARead(std::shared_ptr<Context> context)
                : GenericWaitStateObserver(context){};

            static bool required(std::shared_ptr<Context> context)
            {
                return true;
                // TODO: specialize this rule to 90a when MFMA rules for 908 implemented
                // return context->targetArchitecture().target().getVersionString() == "gfx90a";
            }

        protected:
            virtual int  getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            virtual bool trigger(std::shared_ptr<InstructionRef> inst) const;
            virtual bool writeTrigger() const;
            virtual int  getNops(Instruction const& inst) const;
        };

        static_assert(CObserver<VALUWriteFollowedByMFMARead>);
    }
}
