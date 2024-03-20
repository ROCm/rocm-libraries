#pragma once

#include <rocRoller/Scheduling/Observers/WaitState/WaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief 90a rules for v_mfma_f64_16x16x4f64 Write Hazards
         *
         * | Arch | 1st Inst                    | 2nd Inst                             | NOPs |
         * | ---- | --------------------------- | ------------------------------------ | ---- |
         * | 90a  | v_mfma_f64_16x16x4f64 write | v_mfma_f64_16x16x4f64 read SrcC same | 0    |
         * | 90a  | v_mfma_f64_16x16x4f64 write | v_mfma_*_*f64 read SrcC overlapped   | 9    |
         * | 90a  | v_mfma_f64_16x16x4f64 write | v_mfma* read SrcC overlapped         | 0    |
         * | 90a  | v_mfma_f64_16x16x4f64 write | v_mfma_*_*f64 read SrcA/B            | 11   |
         * | 90a  | v_mfma_f64_16x16x4f64 write | v_mfma* read SrcA/B                  | 11   |
         * | 90a  | v_mfma_f64_16x16x4f64 write | v_* read/write                       | 11   |
         * | 90a  | v_mfma_f64_16x16x4f64 write | buffer* read overlapped              | 18   |
         * | 90a  | v_mfma_f64_16x16x4f64 write | ds* read overlapped                  | 18   |
         * | 90a  | v_mfma_f64_16x16x4f64 write | flat* read overlapped                | 18   |
         * | 94x  | v_mfma_f64_16x16x4f64 write | v_mfma_f64_16x16x4f64 read SrcC same | 0    |
         * | 94x  | v_mfma_f64_16x16x4f64 write | v_mfma_*_*f64 read SrcC overlapped   | 9    |
         * | 94x  | v_mfma_f64_16x16x4f64 write | v_mfma* read SrcC overlapped         | 0    |
         * | 94x  | v_mfma_f64_16x16x4f64 write | v_mfma_*_*f64 read SrcA/B            | 11   |
         * | 94x  | v_mfma_f64_16x16x4f64 write | v_mfma* read SrcA/B                  | 11   |
         * | 94x  | v_mfma_f64_16x16x4f64 write | v_* read/write                       | 11   |
         * | 94x  | v_mfma_f64_16x16x4f64 write | buffer* read overlapped              | 18   |
         * | 94x  | v_mfma_f64_16x16x4f64 write | ds* read overlapped                  | 18   |
         * | 94x  | v_mfma_f64_16x16x4f64 write | flat* read overlapped                | 18   |
         *
         * Note: v_mfma_f64_16x16x4_f64 is an equivalent variant for 94x
         *
         */
        class DGEMM16x16x4Write : public WaitStateObserver<DGEMM16x16x4Write>
        {
        public:
            DGEMM16x16x4Write() {}
            DGEMM16x16x4Write(ContextPtr context)
                : WaitStateObserver<DGEMM16x16x4Write>(context){};

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
                return "DGEMM Write Hazard";
            }

        private:
            std::vector<std::string> m_targetOpCodes
                = {"v_mfma_f64_16x16x4f64", "v_mfma_f64_16x16x4_f64"};
            int const m_maxNops = 18;
        };

        static_assert(CWaitStateObserver<DGEMM16x16x4Write>);
    }
}
