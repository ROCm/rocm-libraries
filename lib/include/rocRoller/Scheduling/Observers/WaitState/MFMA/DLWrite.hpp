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

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA2GPU() || target.isCDNA3GPU() || target.isCDNA35GPU();
            }

            /**
             * Overriden to cache inst opcode
             */
            void observeHazard(Instruction const& inst) override;

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "DL Write Hazard";
            }

        private:
            int const   m_maxNops    = 3;
            std::string m_prevOpCode = "";
        };

        static_assert(CWaitStateObserver<DLWrite>);
    }
}
