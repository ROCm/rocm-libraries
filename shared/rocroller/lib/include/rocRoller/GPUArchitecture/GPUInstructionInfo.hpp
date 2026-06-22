// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    /**
     * These represent the individual wait queues that exist on a GPU.
     */
    enum class GPUWaitQueueType : int
    {
        LoadQueue = 0,
        StoreQueue,
        SendMsgQueue,
        SMemQueue,
        DSQueue,
        EXPQueue,
        VSQueue,
        TensorQueue,
        FinalInstruction,
        None,
        Count,
    };

    std::string toString(GPUWaitQueueType);

    /**
     * These represent an individual register that a s_waitcnt instruction can target.
     */
    enum class GPUWaitQueue : int
    {
        LoadQueue = 0,
        StoreQueue,
        KMQueue,
        DSQueue,
        EXPQueue,
        VSQueue,
        TensorQueue,
        None,
        Count,
    };

    constexpr GPUWaitQueue fromWaitQueueType(GPUWaitQueueType input);

    std::string toString(GPUWaitQueue input);

    enum class CoexecCategory : int
    {
        NotAnInstruction = 0,
        Scalar,
        VMEM,
        VALU,
        VALU_Trans,
        XDL,
        XDL_Scale,
        LDS,
        Count
    };

    std::string toString(CoexecCategory cat);

    class GPUInstructionInfo
    {
    public:
        GPUInstructionInfo() = default;
        GPUInstructionInfo(std::string const&                   instruction,
                           int                                  waitcnt,
                           std::vector<GPUWaitQueueType> const& waitQueues,
                           int                                  latency        = 0,
                           bool                                 implicitAccess = false,
                           bool                                 branch         = false,
                           unsigned int                         maxOffsetValue = 0);

        std::string                   getInstruction() const;
        int                           getWaitCount() const;
        std::vector<GPUWaitQueueType> getWaitQueues() const;
        int                           getLatency() const;
        bool                          hasImplicitAccess() const;
        bool                          isBranch() const;
        unsigned int                  maxOffsetValue() const;

        friend std::ostream& operator<<(std::ostream& os, const GPUInstructionInfo& d);

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        /**
         * Static functions below are for checking instruction type.
         * The input to these functions is the op name.
         * @{
         */
        static bool isDLOP(std::string const& inst);
        static bool isMFMA(std::string const& inst);
        static bool isWMMA(std::string const& inst);
        static bool isSWMMAC(std::string const& inst);
        static bool isVCMPX(std::string const& inst);
        static bool isVCMP(std::string const& inst);

        static bool isScalar(std::string const& inst);
        static bool isSMEM(std::string const& inst);
        static bool isSBarrier(std::string const& opCode);
        static bool isSControl(std::string const& inst);
        static bool isSALU(std::string const& inst);
        static bool isIntInst(std::string const& inst);
        static bool isUIntInst(std::string const& inst);

        static bool isVector(std::string const& inst);
        static bool isVALU(std::string const& inst);
        static bool isVALUTrans(std::string const& inst);
        static bool isDGEMM(std::string const& inst);
        static bool isSGEMM(std::string const& inst);
        static bool isVMEM(std::string const& inst);
        static bool isVMEMRead(std::string const& inst);
        static bool isVMEMWrite(std::string const& inst);
        static bool isFlat(std::string const& inst);
        static bool isLDS(std::string const& inst);
        static bool isLDSRead(std::string const& inst);
        static bool isLDSWrite(std::string const& inst);
        static bool isTensor(std::string const& inst);

        static bool isACCVGPRRead(std::string const& inst);
        static bool isACCVGPRWrite(std::string const& inst);
        static bool isVAddInst(std::string const& inst);
        static bool isVAddCarryInst(std::string const& inst);
        static bool isVSubInst(std::string const& inst);
        static bool isVSubCarryInst(std::string const& inst);
        static bool isVReadlane(std::string const& inst);
        static bool isVWritelane(std::string const& inst);
        static bool isVPermlane(std::string const& inst);
        static bool isVDivScale(std::string const& inst);
        static bool isVDivFmas(std::string const& inst);

        static CoexecCategory getCoexecCategory(std::string const& opCode);
        /** @} */

    private:
        std::string                   m_instruction = "";
        int                           m_waitCount   = -1;
        std::vector<GPUWaitQueueType> m_waitQueues;
        int                           m_latency        = -1;
        bool                          m_implicitAccess = false;
        bool                          m_isBranch       = false;
        unsigned int                  m_maxOffsetValue = 0;
    };

    std::string toString(GPUWaitQueue);
    std::string toString(GPUWaitQueueType);
}

#include <rocRoller/GPUArchitecture/GPUInstructionInfo_impl.hpp>
