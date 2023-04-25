
#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    template <typename IO>
    struct Serialization::
        MappingTraits<Scheduling::Weights, IO, rocRoller::Serialization::EmptyContext>
    {
        static const bool flow = false;
        using iot              = IOTraits<IO>;

        static void mapping(IO& io, Scheduling::Weights& weights)
        {
            iot::mapRequired(io, "nops", weights.nops);
            iot::mapRequired(io, "vmcnt", weights.vmcnt);
            iot::mapRequired(io, "lgkmcnt", weights.lgkmcnt);
            iot::mapRequired(io, "vectorQueueSat", weights.vectorQueueSat);
            iot::mapRequired(io, "vmQueueLen", weights.vmQueueLen);
            iot::mapRequired(io, "ldsQueueSat", weights.ldsQueueSat);
            iot::mapRequired(io, "lgkmQueueLen", weights.lgkmQueueLen);
            iot::mapRequired(io, "newSGPRs", weights.newSGPRs);
            iot::mapRequired(io, "newVGPRs", weights.newVGPRs);
            iot::mapRequired(io, "highWaterMarkSGPRs", weights.highWaterMarkSGPRs);
            iot::mapRequired(io, "highWaterMarkVGPRs", weights.highWaterMarkVGPRs);
            iot::mapRequired(io, "notMFMA", weights.notMFMA);
            iot::mapRequired(io, "fractionOfSGPRs", weights.fractionOfSGPRs);
            iot::mapRequired(io, "fractionOfVGPRs", weights.fractionOfVGPRs);
            iot::mapRequired(io, "outOfRegisters", weights.outOfRegisters);
            iot::mapRequired(io, "zeroFreeBarriers", weights.zeroFreeBarriers);
        }

        static void mapping(IO& io, Scheduling::Weights& weights, EmptyContext& ctx)
        {
            mapping(io, weights);
        }
    };

    namespace Scheduling
    {
        Weights::Weights()
            : fractionOfSGPRs(6332.39f)
            , fractionOfVGPRs(3348.02f)
            , highWaterMarkSGPRs(301.49f)
            , highWaterMarkVGPRs(63.41f)
            , ldsQueueSat(429.44f)
            , lgkmQueueLen(37)
            , lgkmcnt(113.88f)
            , newSGPRs(1015.01f)
            , newVGPRs(620.96f)
            , nops(106.54f)
            , notMFMA(886.65f)
            , outOfRegisters(1e9f)
            , vectorQueueSat(474.53f)
            , vmQueueLen(12)
            , vmcnt(313.34f)
            , zeroFreeBarriers(true)
        {
        }

        RegisterComponent(LinearWeightedCost);
        static_assert(Component::Component<LinearWeightedCost>);

        inline LinearWeightedCost::LinearWeightedCost(std::shared_ptr<Context> ctx)
            : Cost{ctx}
        {
            auto settingsFile = Settings::getInstance()->get(Settings::SchedulerWeights);
            if(!settingsFile.empty())
            {
                m_weights = Serialization::readYAMLFile<Weights>(settingsFile);
            }
        }

        inline bool LinearWeightedCost::Match(Argument arg)
        {
            return std::get<0>(arg) == CostFunction::LinearWeighted;
        }

        inline std::shared_ptr<Cost> LinearWeightedCost::Build(Argument arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<LinearWeightedCost>(std::get<1>(arg));
        }

        inline std::string LinearWeightedCost::name() const
        {
            return Name;
        }

        inline float LinearWeightedCost::cost(Instruction const&       inst,
                                              InstructionStatus const& status) const
        {

            auto nops = static_cast<float>(status.nops);

            auto const& arch = m_ctx.lock()->targetArchitecture();

            auto maxVmcnt   = arch.GetCapability(GPUCapability::MaxVmcnt);
            auto maxLgkmcnt = arch.GetCapability(GPUCapability::MaxLgkmcnt);

            float vmcnt = 0;
            if(status.waitCount.vmcnt() >= 0)
                vmcnt = 1 + maxVmcnt - status.waitCount.vmcnt();

            float lgkmcnt = 0;
            if(status.waitCount.lgkmcnt() >= 0)
                lgkmcnt = 1 + maxLgkmcnt - status.waitCount.lgkmcnt();

            float vectorQueueSat = std::max(
                status.waitLengths.at(GPUWaitQueueType::VMQueue) - m_weights.vmQueueLen, 0);
            float ldsQueueSat = std::max(
                status.waitLengths.at(GPUWaitQueueType::LGKMDSQueue) - m_weights.lgkmQueueLen, 0);

            float newSGPRs
                = status.allocatedRegisters.at(static_cast<size_t>(Register::Type::Scalar));
            float newVGPRs
                = status.allocatedRegisters.at(static_cast<size_t>(Register::Type::Vector));
            float highWaterMarkSGPRs = status.highWaterMarkRegistersDelta.at(
                static_cast<size_t>(Register::Type::Scalar));
            float highWaterMarkVGPRs = status.highWaterMarkRegistersDelta.at(
                static_cast<size_t>(Register::Type::Vector));

            float notMFMA = inst.getOpCode().find("mfma") == std::string::npos ? 1.0f : 0.0f;

            float fractionOfSGPRs
                = status.allocatedRegisters.at(static_cast<size_t>(Register::Type::Scalar));
            float remainingSGPRs
                = status.remainingRegisters.at(static_cast<size_t>(Register::Type::Scalar));
            if(remainingSGPRs > 0)
                fractionOfSGPRs /= remainingSGPRs;

            float fractionOfVGPRs
                = status.allocatedRegisters.at(static_cast<size_t>(Register::Type::Vector));
            float remainingVGPRs
                = status.remainingRegisters.at(static_cast<size_t>(Register::Type::Vector));
            if(remainingVGPRs > 0)
                fractionOfVGPRs /= remainingVGPRs;

            float outOfRegisters = status.outOfRegisters.count();

            if(m_weights.zeroFreeBarriers && inst.getOpCode() == "s_barrier"
               && status.waitCount == WaitCount())
                return 0;

            return m_weights.nops * nops //
                   + m_weights.vmcnt * vmcnt //
                   + m_weights.lgkmcnt * lgkmcnt //
                   + m_weights.vectorQueueSat * vectorQueueSat //
                   + m_weights.ldsQueueSat * ldsQueueSat //
                   + m_weights.newSGPRs * newSGPRs //
                   + m_weights.newVGPRs * newVGPRs //
                   + m_weights.highWaterMarkSGPRs * highWaterMarkSGPRs //
                   + m_weights.highWaterMarkVGPRs * highWaterMarkVGPRs //
                   + m_weights.notMFMA * notMFMA //
                   + m_weights.fractionOfSGPRs * fractionOfSGPRs //
                   + m_weights.fractionOfVGPRs * fractionOfVGPRs //
                   + m_weights.outOfRegisters * outOfRegisters //
                ;
        }
    }
}
