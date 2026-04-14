// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

namespace rocRoller
{
    inline std::string toString(GPUWaitQueueType input)
    {
        switch(input)
        {
        case GPUWaitQueueType::LoadQueue:
            return "LoadQueue";
        case GPUWaitQueueType::StoreQueue:
            return "StoreQueue";
        case GPUWaitQueueType::SendMsgQueue:
            return "SendMsgQueue";
        case GPUWaitQueueType::SMemQueue:
            return "SMemQueue";
        case GPUWaitQueueType::DSQueue:
            return "DSQueue";
        case GPUWaitQueueType::EXPQueue:
            return "EXPQueue";
        case GPUWaitQueueType::VSQueue:
            return "VSQueue";
        case GPUWaitQueueType::FinalInstruction:
            return "FinalInstruction";
        case GPUWaitQueueType::None:
            return "None";
        case GPUWaitQueueType::Count:
            return "Count";
        }

        throw std::invalid_argument("Invalid GPUWaitQueueType!");
        return "";
    }

    constexpr inline GPUWaitQueue fromWaitQueueType(GPUWaitQueueType input)
    {
        switch(input)
        {
        case GPUWaitQueueType::LoadQueue:
            return GPUWaitQueue::LoadQueue;

        case GPUWaitQueueType::StoreQueue:
            return GPUWaitQueue::StoreQueue;

        case GPUWaitQueueType::DSQueue:
            return GPUWaitQueue::DSQueue;

        case GPUWaitQueueType::SendMsgQueue:
        case GPUWaitQueueType::SMemQueue:
            return GPUWaitQueue::KMQueue;

        case GPUWaitQueueType::EXPQueue:
            return GPUWaitQueue::EXPQueue;

        case GPUWaitQueueType::VSQueue:
            return GPUWaitQueue::VSQueue;

        case GPUWaitQueueType::FinalInstruction:
        case GPUWaitQueueType::None:
            return GPUWaitQueue::None;

        case GPUWaitQueueType::Count:
            return GPUWaitQueue::Count;
        }
    }

    inline std::string toString(GPUWaitQueue input)
    {
        switch(input)
        {
        case GPUWaitQueue::LoadQueue:
            return "LoadQueue";
        case GPUWaitQueue::StoreQueue:
            return "StoreQueue";
        case GPUWaitQueue::KMQueue:
            return "KMQueue";
        case GPUWaitQueue::DSQueue:
            return "DSQueue";
        case GPUWaitQueue::EXPQueue:
            return "EXPQueue";
        case GPUWaitQueue::VSQueue:
            return "VSQueue";
        case GPUWaitQueue::Count:
            return "Count";
        case GPUWaitQueue::None:
            return "None";
        }
    }

    inline GPUInstructionInfo::GPUInstructionInfo(std::string const&                   instruction,
                                                  int                                  waitcnt,
                                                  std::vector<GPUWaitQueueType> const& waitQueues,
                                                  int                                  latency,
                                                  bool         implicitAccess,
                                                  bool         branch,
                                                  unsigned int maxOffsetValue)
        : m_instruction(instruction)
        , m_waitCount(waitcnt)
        , m_waitQueues(waitQueues)
        , m_latency(latency)
        , m_implicitAccess(implicitAccess)
        , m_isBranch(branch)
        , m_maxOffsetValue(maxOffsetValue)
    {
    }

    inline std::string GPUInstructionInfo::getInstruction() const
    {
        return m_instruction;
    }
    inline int GPUInstructionInfo::getWaitCount() const
    {
        return m_waitCount;
    }

    inline std::vector<GPUWaitQueueType> GPUInstructionInfo::getWaitQueues() const
    {
        return m_waitQueues;
    }

    inline int GPUInstructionInfo::getLatency() const
    {
        return m_latency;
    }

    inline bool GPUInstructionInfo::hasImplicitAccess() const
    {
        return m_implicitAccess;
    }

    inline bool GPUInstructionInfo::isBranch() const
    {
        return m_isBranch;
    }

    inline unsigned int GPUInstructionInfo::maxOffsetValue() const
    {
        return m_maxOffsetValue;
    }
}
