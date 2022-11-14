
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocRoller
{
    inline std::string ToString(GPUWaitQueueType input)
    {
        return input.ToString();
    }

    inline GPUInstructionInfo::GPUInstructionInfo(std::string const&                   instruction,
                                                  int                                  waitcnt,
                                                  std::vector<GPUWaitQueueType> const& waitQueues,
                                                  int                                  latency,
                                                  bool implicitAccess,
                                                  bool branch)
        : m_instruction(instruction)
        , m_waitCount(waitcnt)
        , m_waitQueues(waitQueues)
        , m_latency(latency)
        , m_implicitAccess(implicitAccess)
        , m_isBranch(branch)
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

    //--GPUWaitQueue
    inline std::string GPUWaitQueue::ToString() const
    {
        return GPUWaitQueue::ToString(m_value);
    }

    inline std::string GPUWaitQueue::ToString(GPUWaitQueue::Value m_value)
    {
        for(const auto& mapping : m_stringMap)
        {
            if(mapping.second == m_value)
            {
                return mapping.first;
            }
        }
        return "";
    }

    inline std::unordered_map<std::string, GPUWaitQueue::Value> GPUWaitQueue::m_stringMap = {
        {"None", Value::None},
        {"VMQueue", Value::VMQueue},
        {"LGKMQueue", Value::LGKMQueue},
        {"EXPQueue", Value::EXPQueue},
        {"VSQueue", Value::VSQueue},
        {"Count", Value::Count},
    };

    //--GPUWaitQueueType
    inline std::string GPUWaitQueueType::ToString() const
    {
        return GPUWaitQueueType::ToString(m_value);
    }

    inline std::string GPUWaitQueueType::ToString(GPUWaitQueueType::Value m_value)
    {
        for(const auto& mapping : m_stringMap)
        {
            if(mapping.second == m_value)
            {
                return mapping.first;
            }
        }
        return "";
    }

    inline const std::unordered_map<std::string, GPUWaitQueueType::Value>
        GPUWaitQueueType::m_stringMap = {
            {"None", Value::None},
            {"VMQueue", Value::VMQueue},
            {"LGKMSendMsgQueue", Value::LGKMSendMsgQueue},
            {"LGKMDSQueue", Value::LGKMDSQueue},
            {"LGKMSmemQueue", Value::LGKMSmemQueue},
            {"EXPQueue", Value::EXPQueue},
            {"VSQueue", Value::VSQueue},
            {"FinalInstruction", Value::FinalInstruction},
            {"Count", Value::Count},
    };
}
