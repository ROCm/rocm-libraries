#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rules for XDL Write Hazards.
         *
         * Note: Excludes DGEMM cases.
         *
         * | Arch | 1st Inst                | 2nd Inst                           | NOPs |
         * | ---- | ----------------------- | ---------------------------------- | ---- |
         * | 90a  | v_mfma* write           | v_mfma* read SrcC same             | 0    |
         * | 90a  | v_mfma* write (2 pass)  | v_mfma* read SrcC overlapped       | 2    |
         * | 90a  | v_mfma* write (8 pass)  | v_mfma* read SrcC overlapped       | 8    |
         * | 90a  | v_mfma* write (16 pass) | v_mfma* read SrcC overlapped       | 16   |
         * | 90a  | v_mfma* write (2 pass)  | v_mfma_*_*f64 read SrcC overlapped | 3    |
         * | 90a  | v_mfma* write (8 pass)  | v_mfma_*_*f64 read SrcC overlapped | 9    |
         * | 90a  | v_mfma* write (16 pass) | v_mfma_*_*f64 read SrcC overlapped | 17   |
         * | 90a  | v_mfma* write (2 pass)  | v_mfma* read SrcA/B                | 5    |
         * | 90a  | v_mfma* write (8 pass)  | v_mfma* read SrcA/B                | 11   |
         * | 90a  | v_mfma* write (16 pass) | v_mfma* read SrcA/B                | 19   |
         * | 90a  | v_mfma* write (2 pass)  | buffer* read overlapped            | 5    |
         * | 90a  | v_mfma* write (8 pass)  | buffer* read overlapped            | 11   |
         * | 90a  | v_mfma* write (16 pass) | buffer* read overlapped            | 19   |
         * | 90a  | v_mfma* write (2 pass)  | ds* read overlapped                | 5    |
         * | 90a  | v_mfma* write (8 pass)  | ds* read overlapped                | 11   |
         * | 90a  | v_mfma* write (16 pass) | ds* read overlapped                | 19   |
         * | 90a  | v_mfma* write (2 pass)  | flat* read overlapped              | 5    |
         * | 90a  | v_mfma* write (8 pass)  | flat* read overlapped              | 11   |
         * | 90a  | v_mfma* write (16 pass) | flat* read overlapped              | 19   |
         * | 90a  | v_mfma* write (2 pass)  | v_* read/write                     | 5    |
         * | 90a  | v_mfma* write (8 pass)  | v_* read/write                     | 11   |
         * | 90a  | v_mfma* write (16 pass) | v_* read/write                     | 19   |
         *
         */
        class XDLWrite90a : public WaitStateObserver<XDLWrite90a>
        {
        public:
            XDLWrite90a() {}
            XDLWrite90a(ContextPtr context)
                : WaitStateObserver<XDLWrite90a>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.isCDNA2GPU();
            }

            int                   getMaxNops(Instruction const& inst) const;
            bool                  trigger(Instruction const& inst) const;
            static constexpr bool writeTrigger()
            {
                return true;
            }
            int         getNops(Instruction const& inst) const;
            std::string getComment() const
            {
                return "XDL Write Hazard";
            }

        private:
            // Excluded as these are handled in other observers
            std::vector<std::string> m_excludedOpCodes
                = {"v_mfma_f64_4x4x4f64", "v_mfma_f64_16x16x4f64"};

            std::unordered_map<int, int> m_latencyAndNops = {{2, 5}, {8, 11}, {16, 19}};
        };

        static_assert(CWaitStateObserver<XDLWrite90a>);
    }
}
