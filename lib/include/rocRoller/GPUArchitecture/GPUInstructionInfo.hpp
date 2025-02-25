
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
            VMQueue = 0,
            LGKMSendMsgQueue,
            LGKMDSQueue,
            LGKMSmemQueue,
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

    class GPUWaitQueue
    {
    public:
        enum Value : uint8_t
        {
            VMQueue = 0,
            LGKMQueue,
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
            case GPUWaitQueueType::VMQueue:
                m_value = Value::VMQueue;
                break;
            case GPUWaitQueueType::LGKMSendMsgQueue:
            case GPUWaitQueueType::LGKMDSQueue:
            case GPUWaitQueueType::LGKMSmemQueue:
                m_value = Value::LGKMQueue;
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

    class GPUInstructionInfo
    {
    public:
        GPUInstructionInfo() = default;
        GPUInstructionInfo(std::string const&                   instruction,
                           int                                  waitcnt,
                           std::vector<GPUWaitQueueType> const& waitQueues,
                           int                                  latency         = 0,
                           bool                                 implicitAccess  = false,
                           bool                                 branch          = false,
                           unsigned int                         maxLiteralValue = 0);

        std::string                   getInstruction() const;
        int                           getWaitCount() const;
        std::vector<GPUWaitQueueType> getWaitQueues() const;
        int                           getLatency() const;
        bool                          hasImplicitAccess() const;
        bool                          isBranch() const;
        unsigned int                  maxLiteralValue() const;

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
        static bool isVCMPX(std::string const& inst);
        static bool isVCMP(std::string const& inst);

        static bool isScalar(std::string const& inst);
        static bool isSMEM(std::string const& inst);
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
        static bool isVSubInst(std::string const& inst);
        static bool isVReadlane(std::string const& inst);
        static bool isVWritelane(std::string const& inst);
        static bool isVDivScale(std::string const& inst);
        static bool isVDivFmas(std::string const& inst);
        /** @} */

    private:
        std::string                   m_instruction = "";
        int                           m_waitCount   = -1;
        std::vector<GPUWaitQueueType> m_waitQueues;
        int                           m_latency         = -1;
        bool                          m_implicitAccess  = false;
        bool                          m_isBranch        = false;
        unsigned int                  m_maxLiteralValue = 0;
    };

    std::string toString(GPUWaitQueue);
    std::string toString(GPUWaitQueueType);
}

#include "GPUInstructionInfo_impl.hpp"
