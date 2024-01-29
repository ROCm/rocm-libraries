#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rule for DL Op Writes
         *
         * | Arch | 1st Inst     | 2nd Inst                    | NOPs |
         * | ---- | ------------ | --------------------------- | ---- |
         * | 90a  | v_dot* write | Same opcode, read SrcC same | 0    |
         * | 90a  | v_dot* write | Same opcode, read SrcA/B    | 3    |
         * | 90a  | v_dot* write | Different opcode            | 3    |
         * | 94x  | v_dot* write | Same opcode, read SrcC same | 0    |
         * | 94x  | v_dot* write | Same opcode, read SrcA/B    | 3    |
         * | 94x  | v_dot* write | Different opcode            | 3    |
         *
         */
        class DLWrite : public WaitStateObserver<DLWrite>
        {
        public:
            DLWrite() {}
            DLWrite(ContextPtr context)
                : WaitStateObserver<DLWrite>(context){};

            void observe(Instruction const& inst)
            {
                observe_base(inst);
            }

            static bool required(ContextPtr context)
            {
                auto arch = context->targetArchitecture().target().getVersionString();
                return arch == "gfx90a" || arch == "gfx940" || arch == "gfx941" || arch == "gfx942";
            }

            int         getMaxNops(std::shared_ptr<InstructionRef> inst) const;
            bool        trigger(std::shared_ptr<InstructionRef> inst) const;
            bool        writeTrigger() const;
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "DL Write Hazard";
            }

        private:
            bool determineHazard(Register::RegisterId const& regId,
                                 InstructionRef const&       instRef) const;

            int const m_maxNops = 3;
        };

        static_assert(CWaitStateObserver<DLWrite>);
    }
}
