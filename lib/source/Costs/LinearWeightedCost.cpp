
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
            : fractionOfSGPRs(355.3108286906086)
            , fractionOfVGPRs(39.0772209608651)
            , highWaterMarkSGPRs(589.3984679981991)
            , highWaterMarkVGPRs(737.0723385770781)
            , isACCVGPRRead(215.29900460719165)
            , isACCVGPRWrite(112.53226566450687)
            , isLDSRead(1065.9125784709306)
            , isLDSWrite(109.01025909512389)
            , isMFMA(343.6368908511061)
            , isSALU(86.76155051253933)
            , isSControl(56.651303809189926)
            , isSMEM(19.69694529564257)
            , isVALU(161.4533731992491)
            , isVMEMRead(79.29458852723246)
            , isVMEMWrite(23.57073489778686)
            , ldsQueueSat(65.95446246070229)
            , lgkmQueueLen(35)
            , lgkmcnt(96.55017800679398)
            , newSGPRs(1143.7733673732694)
            , newVGPRs(118.1844904455429)
            , nops(2196.4873607526833)
            , notMFMA(155.24701615203398)
            , outOfRegisters(1000000000.0)
            , stallCycles(1000.0)
            , vectorQueueSat(52.88961947171189)
            , vmQueueLen(6)
            , vmcnt(230.33153533090928)
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
