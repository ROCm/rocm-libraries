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

#include <array>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    class GPUWaitQueueType
    {
    public:
        enum Value : uint8_t
        {
            LoadQueue = 0,
            StoreQueue,
            SendMsgQueue,
            SMemQueue,
            DSQueue,
            EXPQueue,
            VSQueue,
            FinalInstruction,
            Count,
            None = Count,
        };

        GPUWaitQueueType() = default;
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUWaitQueueType(Value input)
            : m_value(input)
        {
        }
        explicit GPUWaitQueueType(std::string const& input)
            : m_value(m_stringMap.at(input))
        {
        }
        explicit constexpr GPUWaitQueueType(uint8_t input)
            : m_value(static_cast<Value>(input))
        {
        }

        constexpr operator uint8_t() const
        {
            return static_cast<uint8_t>(m_value);
        }

        std::string toString() const;

        static std::string toString(Value);

        struct Hash
        {
            std::size_t operator()(const GPUWaitQueueType& input) const
            {
                return std::hash<uint8_t>()((uint8_t)input.m_value);
            };
        };

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

        template <typename T1, typename T2>
        friend struct rocRoller::Serialization::EnumTraits;

    private:
        Value                                               m_value = Value::Count;
        static const std::unordered_map<std::string, Value> m_stringMap;
    };

    std::ostream& operator<<(std::ostream&, GPUWaitQueueType::Value const& v);

    class GPUWaitQueue
    {
    public:
        enum Value : uint8_t
        {
            LoadQueue = 0,
            StoreQueue,
            KMQueue,
            DSQueue,
            EXPQueue,
            VSQueue,
            Count,
            None = Count,
        };

        GPUWaitQueue() = default;
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUWaitQueue(Value input)
            : m_value(input)
        {
        }

        explicit GPUWaitQueue(std::string const& input)
            : m_value(GPUWaitQueue::m_stringMap[input])
        {
        }
        explicit constexpr GPUWaitQueue(uint8_t input)
            : m_value(static_cast<Value>(input))
        {
        }
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUWaitQueue(GPUWaitQueueType input)
        {
            switch(input)
            {
            case GPUWaitQueueType::LoadQueue:
                m_value = Value::LoadQueue;
                break;
            case GPUWaitQueueType::StoreQueue:
                m_value = Value::StoreQueue;
                break;
            case GPUWaitQueueType::DSQueue:
                m_value = Value::DSQueue;
                break;
            case GPUWaitQueueType::SendMsgQueue:
            case GPUWaitQueueType::SMemQueue:
                m_value = Value::KMQueue;
                break;
            case GPUWaitQueueType::EXPQueue:
                m_value = Value::EXPQueue;
                break;
            case GPUWaitQueueType::VSQueue:
                m_value = Value::VSQueue;
                break;
            default:
                m_value = Value::None;
            }
        }

        constexpr operator uint8_t() const
        {
            return static_cast<uint8_t>(m_value);
        }

        std::string toString() const;

        static std::string toString(Value);

        struct Hash
        {
            std::size_t operator()(const GPUWaitQueue& input) const
            {
                return std::hash<uint8_t>()((uint8_t)input.m_value);
            };
        };

    private:
        Value                                         m_value = Value::Count;
        static std::unordered_map<std::string, Value> m_stringMap;
    };

    std::ostream& operator<<(std::ostream&, GPUWaitQueue::Value const& v);

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
