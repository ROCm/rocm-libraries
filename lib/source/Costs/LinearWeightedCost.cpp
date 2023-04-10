
#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        RegisterComponent(LinearWeightedCost);
        static_assert(Component::Component<LinearWeightedCost>);

        inline LinearWeightedCost::LinearWeightedCost(std::shared_ptr<Context> ctx)
            : Cost{ctx}
        {
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
            // Heuristics #1: What's the ideal number of in-flight memory instructions
            // before we start prioritizing other things?
            int lgkmQueueSat = 6;
            int vmQueueSat   = 16;

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

            float vectorQueueSat
                = std::max(status.waitLengths.at(GPUWaitQueueType::VMQueue) - vmQueueSat, 0);
            float ldsQueueSat
                = std::max(status.waitLengths.at(GPUWaitQueueType::LGKMDSQueue) - lgkmQueueSat, 0);

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

            // Heuristics #2: What's the best relative value of each of these coefficients?

            if(inst.getOpCode() == "s_barrier" && status.waitCount == WaitCount())
                return 0;

            return 25.f * nops //
                   + 2.0f * vmcnt //
                   + 0.3f * lgkmcnt //
                   + 0.0f * vectorQueueSat //
                   + 8.0f * ldsQueueSat //
                   + 0.0f * newSGPRs //
                   + 1.0f * newVGPRs //
                   + 0.0f * highWaterMarkSGPRs //
                   + 0.0f * highWaterMarkVGPRs //
                   + 15.0f * notMFMA //
                   + 1.0f * fractionOfSGPRs //
                   + 1.0f * fractionOfVGPRs //
                   + 1e6f * outOfRegisters //
                ;
        }
    }
}
