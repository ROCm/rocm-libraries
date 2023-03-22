
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

        inline float LinearWeightedCost::cost(const InstructionStatus& inst) const
        {
            // Heuristics #1: What's the ideal number of in-flight memory instructions
            // before we start prioritizing other things?
            int lgkmQueueSat = 3;
            int vmQueueSat   = 16;

            auto nops = static_cast<float>(inst.nops);

            auto const& arch = m_ctx.lock()->targetArchitecture();

            auto maxVmcnt   = arch.GetCapability(GPUCapability::MaxVmcnt);
            auto maxLgkmcnt = arch.GetCapability(GPUCapability::MaxLgkmcnt);

            float vmcnt = 0;
            if(inst.waitCount.vmcnt() >= 0)
                vmcnt = 1 + maxVmcnt - inst.waitCount.vmcnt();

            float lgkmcnt = 0;
            if(inst.waitCount.lgkmcnt() >= 0)
                lgkmcnt = 1 + maxLgkmcnt - inst.waitCount.vmcnt();

            float vectorQueueSat
                = std::max(inst.waitLengths.at(GPUWaitQueueType::VMQueue) - vmQueueSat, 0);
            float ldsQueueSat
                = std::max(inst.waitLengths.at(GPUWaitQueueType::LGKMDSQueue) - lgkmQueueSat, 0);

            float newSGPRs
                = inst.allocatedRegisters.at(static_cast<size_t>(Register::Type::Scalar));
            float newVGPRs
                = inst.allocatedRegisters.at(static_cast<size_t>(Register::Type::Vector));
            float highWaterMarkSGPRs
                = inst.highWaterMarkRegistersDelta.at(static_cast<size_t>(Register::Type::Scalar));
            float highWaterMarkVGPRs
                = inst.highWaterMarkRegistersDelta.at(static_cast<size_t>(Register::Type::Vector));

            // Heuristics #2: What's the best relative value of each of these coefficients?
            return 1.5f * nops //
                   + 2.0f * vmcnt //
                   + 0.3f * lgkmcnt //
                   + 0.0f * vectorQueueSat //
                   + 4.0f * ldsQueueSat //
                   + 0.0f * newSGPRs //
                   + 1.0f * newVGPRs //
                   + 0.0f * highWaterMarkSGPRs //
                   + 4.5f * highWaterMarkVGPRs;
        }
    }
}
