#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 908/90a/94x rule for V_CMPX Write to EXEC followed by an MFMA requiring 4 NOPs
         *
         * | Arch | 1st Inst           | 2nd Inst        | NOPs |
         * | ---- | ------------------ | --------------- | ---- |
         * | 908  | v_cmpx* write EXEC | v_mfma*         | 4    |
         * | 908  | v_cmpx* write EXEC | v_accvgpr_write | 4    |
         * | 90a  | v_cmpx* write EXEC | v_mfma*         | 4    |
         * | 94x  | v_cmpx* write EXEC | v_mfma*         | 4    |
         *
         */
        class CMPXWriteExec : public WaitStateObserver<CMPXWriteExec>
        {
        public:
            CMPXWriteExec() {}
            CMPXWriteExec(ContextPtr context)
                : WaitStateObserver<CMPXWriteExec>(context)
            {
                m_checkACCVGPR = context->targetArchitecture().target().isCDNA1GPU();
            };

            /**
             * Overriden as we need to target Exec
             */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNAGPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            int                   getNops(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            std::string getComment() const
            {
                return "EXEC Write Hazard";
            }

        private:
            bool      m_checkACCVGPR;
            int const m_maxNops = 4;
        };

        static_assert(CWaitStateObserver<CMPXWriteExec>);
    }
}
