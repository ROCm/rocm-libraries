#pragma once

#include "../Scheduling.hpp"

#include "../../CodeGen/InstructionReference.hpp"
#include "../../Context.hpp"
#include "../../GPUArchitecture/GPUInstructionInfo.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief Observer for wait state hazards specific to GFX 90a MFMA instructions
         *
         * @deprecated
         * TODO: Remove when all rules here are implemented with smaller observers
         */
        class MFMA90aObserver
        {
        public:
            MFMA90aObserver(std::shared_ptr<Context> context)
                : m_context(context){};

            InstructionStatus peek(Instruction const& inst) const
            {
                return InstructionStatus::Nops(getNopMin(inst));
            };

            void modify(Instruction& inst) const
            {
                inst.setNopMin(getNopMin(inst));
            }

            InstructionStatus observe(Instruction const& inst)
            {
                // If instruction is not a comment or empty
                if(!inst.getOpCode().empty())
                {
                    m_prevInst = InstructionReference(inst);
                }
                return InstructionStatus::Nops(inst.getNopCount());
            }

            static bool required(std::shared_ptr<Context> context)
            {
                // TODO: This should be fixed once we have a 908 hazard observer.
                return true; //context->targetArchitecture().target().getVersionString() == "gfx90a";
            }

        private:
            std::weak_ptr<Context> m_context;
            InstructionReference   m_prevInst;

            int const m_CMPXExecWriteFollowedByMFMANops     = 4;
            int const m_VALUWriteFollowedByMFMAReadNops     = 2;
            int const m_XDLWriteFollowedByMFMAExactReadNops = 0;
            int const m_DLWriteFollowedBySameReadCNops      = 0;
            int const m_DLWriteFollowedBySameReadABNops     = 3;
            int const m_DLWriteFollowedByDifferentNops      = 3;

            // Latency of the XDLOP instruction paried with the required NOPs
            std::map<int, int> const m_XDLWriteFollowedByMFMAOverlapReadNops
                = {{2, 2}, {8, 8}, {16, 16}};
            std::map<int, int> const m_XDLWriteFollowedByDGEMMOverlapReadNops
                = {{2, 3}, {8, 9}, {16, 17}};
            std::map<int, int> const m_XDLWriteFollowedByMFMAReadNops = {{2, 5}, {8, 11}, {16, 19}};
            std::map<int, int> const m_XDLWriteFollowedByMemNops      = {{2, 5}, {8, 11}, {16, 19}};
            std::map<int, int> const m_XDLWriteFollowedByVALUReadWriteNops
                = m_XDLWriteFollowedByMemNops;
            std::map<int, int> const m_XDLReadFollowedByVALUWriteNops = {{2, 1}, {8, 11}, {16, 19}};

            int getNopMin(Instruction const& inst) const
            {
                InstructionReference currentInst(inst);
                int                  minNop = 0;

                if(m_prevInst.isDLOP())
                {
                    if(m_prevInst.getOpCode() != currentInst.getOpCode())
                    {
                        minNop = std::max(minNop, m_DLWriteFollowedByDifferentNops);
                    }
                    else
                    {
                        AssertFatal(currentInst.getSrcs().size() == 3,
                                    "DLOP instruction srcs wrong size");
                        AssertFatal(m_prevInst.getSrcs().size() == 3,
                                    "DLOP instruction srcs wrong size");
                        AssertFatal(m_prevInst.getDsts().size() >= 1,
                                    "DLOP instruction dsts wrong size");
                        if(m_prevInst.getDsts()[0].exactlyOverlaps(currentInst.getSrcs()[2]))
                        {
                            minNop = std::max(minNop, m_DLWriteFollowedBySameReadCNops);
                        }
                        else
                        {
                            minNop = std::max(minNop, m_DLWriteFollowedBySameReadABNops);
                        }
                    }
                }
                if(currentInst.isDGEMM())
                {
                    // TODO: Support DGEMM operations
                    Throw<FatalError>("DGEMM not supported due to wait state hazard");
                }

                // XDLOP
                if(m_prevInst.isXDLOP())
                {
                    AssertFatal(m_prevInst.getSrcs().size() == 3,
                                "MFMA instruction srcs wrong size");
                    AssertFatal(m_prevInst.getDsts().size() >= 1,
                                "MFMA instruction dsts wrong size");

                    // XDLOP follows
                    if(currentInst.isXDLOP())
                    {
                        AssertFatal(currentInst.getSrcs().size() == 3,
                                    "MFMA instruction srcs wrong size");

                        // MFMA reads VGPR as SrcC exactly
                        if(currentInst.isMFMA()
                           && currentInst.getSrcs()[2].exactlyOverlaps(m_prevInst.getDsts()[0]))
                        {
                            minNop = std::max(minNop, m_XDLWriteFollowedByMFMAExactReadNops);
                        }
                        // Check SrcC overlap with first Vdst
                        else if(currentInst.getSrcs()[2].intersects(m_prevInst.getDsts()[0]))
                        {
                            if(currentInst.isDGEMM())
                            {
                                minNop = std::max(
                                    minNop,
                                    getNopFromLatency(m_prevInst.getOpCode(),
                                                      m_XDLWriteFollowedByDGEMMOverlapReadNops));
                            }
                            else
                            {
                                minNop = std::max(
                                    minNop,
                                    getNopFromLatency(m_prevInst.getOpCode(),
                                                      m_XDLWriteFollowedByMFMAOverlapReadNops));
                            }
                        }

                        // MFMA reads output as SrcA or SrcB
                        if(currentInst.isMFMA()
                           && (currentInst.getSrcs()[0].intersects(m_prevInst.getDsts()[0])
                               || currentInst.getSrcs()[1].intersects(m_prevInst.getDsts()[0])))
                        {
                            minNop = std::max(minNop,
                                              getNopFromLatency(m_prevInst.getOpCode(),
                                                                m_XDLWriteFollowedByMFMAReadNops));
                        }
                    }
                    else if(currentInst.isVMEM() || currentInst.isFlat() || currentInst.isLDS())
                    {
                        for(auto const& src : currentInst.getSrcs())
                        {
                            // Read after write
                            if(src.intersects(m_prevInst.getDsts()[0]))
                            {
                                minNop = std::max(minNop,
                                                  getNopFromLatency(m_prevInst.getOpCode(),
                                                                    m_XDLWriteFollowedByMemNops));
                            }
                        }
                        for(auto const& dst : currentInst.getDsts())
                        {
                            // Write after write
                            if(dst.intersects(m_prevInst.getDsts()[0]))
                            {
                                minNop = std::max(minNop,
                                                  getNopFromLatency(m_prevInst.getOpCode(),
                                                                    m_XDLWriteFollowedByMemNops));
                            }
                        }
                    }
                    else if(currentInst.isVALU())
                    {
                        for(auto const& src : currentInst.getSrcs())
                        {
                            // Read after write
                            if(src.intersects(m_prevInst.getDsts()[0]))
                            {
                                minNop = std::max(
                                    minNop,
                                    getNopFromLatency(m_prevInst.getOpCode(),
                                                      m_XDLWriteFollowedByVALUReadWriteNops));
                            }
                        }

                        for(auto const& dst : currentInst.getDsts())
                        {
                            // Write after write
                            if(dst.intersects(m_prevInst.getDsts()[0]))
                            {
                                minNop = std::max(
                                    minNop,
                                    getNopFromLatency(m_prevInst.getOpCode(),
                                                      m_XDLWriteFollowedByVALUReadWriteNops));
                            }

                            // Write after reading SrcC
                            if(dst.intersects(m_prevInst.getSrcs()[2]))
                            {
                                minNop
                                    = std::max(minNop,
                                               getNopFromLatency(m_prevInst.getOpCode(),
                                                                 m_XDLReadFollowedByVALUWriteNops));
                            }
                        }
                    }
                }
                // V_CMPX* writes exec followed by MFMA
                if(m_prevInst.isCMPX() && currentInst.isMFMA())
                {
                    AssertFatal(m_prevInst.getDsts().size() >= 1,
                                "CMPX instruction dsts wrong size");
                    for(auto const& prevDst : m_prevInst.getDsts())
                    {
                        if(prevDst.isExec())
                        {
                            minNop = std::max(minNop, m_CMPXExecWriteFollowedByMFMANops);
                        }
                    }
                }
                // VALU write followed by MFMA read instruction covered by the observer VALUWriteFollowedByMFMARead
                return minNop;
            }

            int getNopFromLatency(std::string const&        opCode,
                                  std::map<int, int> const& latencyAndNops) const
            {
                auto        context      = m_context.lock();
                auto const& architecture = context->targetArchitecture();
                int         passes       = architecture.GetInstructionInfo(opCode).getLatency();

                AssertFatal(latencyAndNops.contains(passes), "Unexpected number of MFMA passes");

                return latencyAndNops.at(passes);
            }
        };

        static_assert(CObserver<MFMA90aObserver>);
    }
}
