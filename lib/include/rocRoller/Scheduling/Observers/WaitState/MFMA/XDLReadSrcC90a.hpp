#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rules for XDLOP Reads of Src C (WAR)
         *
         * | Arch | 1st Inst                    | 2nd Inst  | NOPs |
         * | ---- | --------------------------- | --------- | ---- |
         * | 90a  | v_mfma* read SrcC (2 pass)  | v_* write | 1    |
         * | 90a  | v_mfma* read SrcC (8 pass)  | v_* write | 11   |
         * | 90a  | v_mfma* read SrcC (16 pass) | v_* write | 19   |
         *
         */
        class XDLReadSrcC90a : public WaitStateObserver<XDLReadSrcC90a>
        {
        public:
            XDLReadSrcC90a() {}
            XDLReadSrcC90a(ContextPtr context)
                : WaitStateObserver<XDLReadSrcC90a>(context){};

            /**
             * Overriden as we need to target src C only
             */
            void observeHazard(Instruction const& inst) override;

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA2GPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return false;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "XDL Read Hazard";
            }

        private:
            std::unordered_map<int, int> m_latencyAndNops = {{2, 1}, {8, 11}, {16, 19}};
        };

        static_assert(CWaitStateObserver<XDLReadSrcC90a>);
    }
}
