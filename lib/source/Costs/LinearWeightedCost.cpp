
#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>
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
            iot::mapRequired(io, "stallCycles", weights.stallCycles);
            iot::mapRequired(io, "newSGPRs", weights.newSGPRs);
            iot::mapRequired(io, "newVGPRs", weights.newVGPRs);
            iot::mapRequired(io, "highWaterMarkSGPRs", weights.highWaterMarkSGPRs);
            iot::mapRequired(io, "highWaterMarkVGPRs", weights.highWaterMarkVGPRs);
            iot::mapRequired(io, "notMFMA", weights.notMFMA);
            iot::mapRequired(io, "isMFMA", weights.isMFMA);
            iot::mapRequired(io, "fractionOfSGPRs", weights.fractionOfSGPRs);
            iot::mapRequired(io, "fractionOfVGPRs", weights.fractionOfVGPRs);
            iot::mapRequired(io, "outOfRegisters", weights.outOfRegisters);
            iot::mapRequired(io, "zeroFreeBarriers", weights.zeroFreeBarriers);

            iot::mapRequired(io, "isSMEM", weights.isSMEM);
            iot::mapRequired(io, "isSControl", weights.isSControl);
            iot::mapRequired(io, "isSALU", weights.isSALU);

            iot::mapRequired(io, "isVMEMRead", weights.isVMEMRead);
            iot::mapRequired(io, "isVMEMWrite", weights.isVMEMWrite);
            iot::mapRequired(io, "isLDSRead", weights.isLDSRead);
            iot::mapRequired(io, "isLDSWrite", weights.isLDSWrite);
            iot::mapRequired(io, "isVALU", weights.isVALU);

            iot::mapRequired(io, "isACCVGPRWrite", weights.isACCVGPRWrite);
            iot::mapRequired(io, "isACCVGPRRead", weights.isACCVGPRRead);
        }

        static void mapping(IO& io, Scheduling::Weights& weights, EmptyContext& ctx)
        {
            mapping(io, weights);
        }
    };

    namespace Scheduling
    {
        Weights::Weights()
            : fractionOfSGPRs(37.92553601045985)
            , fractionOfVGPRs(414.6763363284178)
            , highWaterMarkSGPRs(220.1029922503964)
            , highWaterMarkVGPRs(449.3631986674793)
            , isACCVGPRRead(402.9938370234925)
            , isACCVGPRWrite(50.53821969417193)
            , isLDSRead(83.807985884837559)
            , isLDSWrite(245.03596536366308)
            , isMFMA(263.77891651105875)
            , isSALU(2340.968619137546)
            , isSControl(102.74380989849855)
            , isSMEM(95.26408460062744)
            , isVALU(136.65680476426198)
            , isVMEMRead(1355.0851385340522)
            , isVMEMWrite(303.11297867836936)
            , ldsQueueSat(71.26604926389385)
            , lgkmQueueLen(31)
            , lgkmcnt(114.51288868009384)
            , newSGPRs(697.4677932661201)
            , newVGPRs(439.31252236152045)
            , nops(89.76419353739078)
            , notMFMA(451.04459769663833)
            , outOfRegisters(1000000000.0)
            , stallCycles(1000.0)
            , vectorQueueSat(773.1735010508048)
            , vmQueueLen(11)
            , vmcnt(169.05013905680107)
            , zeroFreeBarriers(true)
        {
        }

        RegisterComponent(LinearWeightedCost);
        static_assert(Component::Component<LinearWeightedCost>);

        inline LinearWeightedCost::LinearWeightedCost(ContextPtr ctx)
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
                   + m_weights.stallCycles * status.stallCycles //
                   + m_weights.newSGPRs * newSGPRs //
                   + m_weights.newVGPRs * newVGPRs //
                   + m_weights.highWaterMarkSGPRs * highWaterMarkSGPRs //
                   + m_weights.highWaterMarkVGPRs * highWaterMarkVGPRs //
                   + m_weights.notMFMA * notMFMA //
                   //+ m_weights.isMFMA * (1.0f - notMFMA) //
                   + m_weights.fractionOfSGPRs * fractionOfSGPRs //
                   + m_weights.fractionOfVGPRs * fractionOfVGPRs //
                   + m_weights.outOfRegisters * outOfRegisters //

                   + m_weights.isSMEM * InstructionRef::isSMEM(inst) //
                   + m_weights.isSControl * InstructionRef::isSControl(inst) //
                   + m_weights.isSALU * InstructionRef::isSALU(inst) //

                   + m_weights.isVMEMRead * InstructionRef::isVMEMRead(inst) //
                   + m_weights.isVMEMWrite * InstructionRef::isVMEMWrite(inst) //
                   + m_weights.isLDSRead * InstructionRef::isLDSRead(inst) //
                   + m_weights.isLDSWrite * InstructionRef::isLDSWrite(inst) //
                   + m_weights.isVALU * InstructionRef::isVALU(inst) //

                   + m_weights.isACCVGPRWrite * InstructionRef::isACCVGPRWrite(inst) //
                   + m_weights.isACCVGPRRead * InstructionRef::isACCVGPRRead(inst) //
                ;
        }
    }
}
