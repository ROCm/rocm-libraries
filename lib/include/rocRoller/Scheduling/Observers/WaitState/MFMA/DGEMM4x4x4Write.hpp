#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rules for v_mfma_f64_4x4x4f64 Write Hazards
         *
         * | Arch | 1st Inst                  | 2nd Inst                           | NOPs |
         * | ---- | ------------------------- | ---------------------------------- | ---- |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_f64_4x4x4f64 read SrcC same | 4    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcC overlapped | 4    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma* read SrcC overlapped       | 0    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcA/B          | 6    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | v_* read/write                     | 6    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | buffer* read overlapped            | 9    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | ds* read overlapped                | 9    |
         * | 90a  | v_mfma_f64_4x4x4f64 write | flat* read overlapped              | 9    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma_f64_4x4x4f64 read SrcC same | 4    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcC overlapped | 4    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma* read SrcC overlapped       | 0    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_mfma_*_*f64 read SrcA/B          | 6    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | v_* read/write                     | 6    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | buffer* read overlapped            | 9    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | ds* read overlapped                | 9    |
         * | 94x  | v_mfma_f64_4x4x4f64 write | flat* read overlapped              | 9    |
         *
         */
        class DGEMM4x4x4Write : public WaitStateObserver<DGEMM4x4x4Write>
        {
        public:
            DGEMM4x4x4Write() {}
            DGEMM4x4x4Write(ContextPtr context)
                : WaitStateObserver<DGEMM4x4x4Write>(context){};

            constexpr static bool required(GPUArchitectureTarget const& target)
            {
                return target.is90aGPU() || target.is94XGPU();
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
                return "DGEMM Write Hazard";
            }

        private:
            std::string m_targetOpCode = "v_mfma_f64_4x4x4f64";
            int const   m_maxNops      = 9;
        };

        static_assert(CWaitStateObserver<DGEMM4x4x4Write>);
    }
}
