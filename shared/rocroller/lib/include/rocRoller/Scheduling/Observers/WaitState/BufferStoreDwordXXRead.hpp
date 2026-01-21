// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief CDNA rule for buffer_store_dwordx3/4 followed by a VALU or non-VALU instruction
         *
         * | Arch | 1st Inst                    | 2nd Inst      | NOPs |
         * | ---- | --------------------------- | ------------- | ---- |
         * | 908  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 908  | buffer_store_dwordx3/4 read | v_* read      | 1    |
         * | 90a  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 90a  | buffer_store_dwordx3/4 read | v_* read      | 1    |
         * | 94x  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 94x  | buffer_store_dwordx3/4 read | v_* read      | 2    |
         * | 950  | buffer_store_dwordx3/4 read | non-VALU read | 1    |
         * | 950  | buffer_store_dwordx3/4 read | v_* read      | 2    |
         *
         * NOTE: If soffset argument is an SGPR, no NOPs required
         * NOTE: Only affects writedata from the first instruction
         *
         */
        class BufferStoreDwordXXRead : public WaitStateObserver<BufferStoreDwordXXRead>
        {
        public:
            BufferStoreDwordXXRead() {}
            BufferStoreDwordXXRead(ContextPtr context)
                : WaitStateObserver<BufferStoreDwordXXRead>(context)
            {
                auto const& target = context->targetArchitecture().target();
                m_isCDNA1orCDNA2   = target.isCDNA1GPU() || target.isCDNA2GPU();
            };

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNAGPU();
            }

            /**
             * Overriden as we need to target vdata
             */
            void observeHazard(Instruction const& inst) override;

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return false;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "Buffer Store Read Hazard";
            }

        private:
            bool      m_isCDNA1orCDNA2;
            int const m_maxNops = 2;
        };

        static_assert(CWaitStateObserver<BufferStoreDwordXXRead>);
    }
}
