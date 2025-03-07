/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <algorithm>
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
    inline std::string toString(GPUWaitQueueType input)
    {
        return input.toString();
    }

    inline std::string toString(GPUWaitQueue input)
    {
        return input.toString();
    }

    inline std::ostream& operator<<(std::ostream& stream, GPUWaitQueueType::Value const& v)
    {
        return stream << GPUWaitQueueType::toString(v);
    }

    inline std::ostream& operator<<(std::ostream& stream, GPUWaitQueue::Value const& v)
    {
        return stream << GPUWaitQueue::toString(v);
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

    //--GPUWaitQueue
    inline std::string GPUWaitQueue::toString() const
    {
        return GPUWaitQueue::toString(m_value);
    }

    inline std::string GPUWaitQueue::toString(GPUWaitQueue::Value value)
    {
        auto it = std::find_if(m_stringMap.begin(),
                               m_stringMap.end(),
                               [&value](auto const& mapping) { return value == mapping.second; });

        if(it == m_stringMap.end())
            return "";
        return it->first;
    }

    inline std::unordered_map<std::string, GPUWaitQueue::Value> GPUWaitQueue::m_stringMap = {
        {"None", Value::None},
        {"LoadQueue", Value::LoadQueue},
        {"StoreQueue", Value::StoreQueue},
        {"DSQueue", Value::DSQueue},
        {"KMQueue", Value::KMQueue},
        {"EXPQueue", Value::EXPQueue},
        {"VSQueue", Value::VSQueue},
        {"Count", Value::Count},
    };

    //--GPUWaitQueueType
    inline std::string GPUWaitQueueType::toString() const
    {
        return GPUWaitQueueType::toString(m_value);
    }

    inline std::string GPUWaitQueueType::toString(GPUWaitQueueType::Value value)
    {
        auto it = std::find_if(m_stringMap.begin(),
                               m_stringMap.end(),
                               [&value](auto const& mapping) { return value == mapping.second; });

        if(it == m_stringMap.end())
            return "";
        return it->first;
    }

    inline const std::unordered_map<std::string, GPUWaitQueueType::Value>
        GPUWaitQueueType::m_stringMap = {
            {"None", Value::None},
            {"LoadQueue", Value::LoadQueue},
            {"StoreQueue", Value::StoreQueue},
            {"SendMsgQueue", Value::SendMsgQueue},
            {"SMemQueue", Value::SMemQueue},
            {"DSQueue", Value::DSQueue},
            {"EXPQueue", Value::EXPQueue},
            {"VSQueue", Value::VSQueue},
            {"FinalInstruction", Value::FinalInstruction},
            {"Count", Value::Count},
    };
}
