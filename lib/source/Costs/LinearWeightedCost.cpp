
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
        constexpr Weights GFX90A_WEIGHTS = {.nops               = 102.16064235921411,
                                           .vmcnt              = 175.18015151123478,
                                           .lgkmcnt            = 57.42789675890664,
                                           .vmQueueLen         = 4,
                                           .vectorQueueSat     = 28.444961163165864,
                                           .ldsQueueSat        = 514.0030224624215,
                                           .lgkmQueueLen       = 8,
                                           .stallCycles        = 1000.0,
                                           .notMFMA            = 1126.0097472441112,
                                           .isMFMA             = 2064.675633601412,
                                           .isSMEM             = 51.73435897493539,
                                           .isSControl         = 52.20563014565612,
                                           .isSALU             = 187.61842343063546,
                                           .isVMEMRead         = 132.51539911592974,
                                           .isVMEMWrite        = 211.778830039103,
                                           .isLDSRead          = 305.80350680401983,
                                           .isLDSWrite         = 374.8770913083329,
                                           .isVALU             = 34.88704839783154,
                                           .isACCVGPRWrite     = 66.94359848246074,
                                           .isACCVGPRRead      = 518.2179267123564,
                                           .newSGPRs           = 93.71800320620157,
                                           .newVGPRs           = 199.44918165517117,
                                           .highWaterMarkSGPRs = 96.88502683536836,
                                           .highWaterMarkVGPRs = 53.526584801082066,
                                           .fractionOfSGPRs    = 12042.440418691136,
                                           .fractionOfVGPRs    = 95.71453942518029,
                                           .outOfRegisters     = 1000000000.0,
                                           .zeroFreeBarriers   = true};
        constexpr Weights GFX908_WEIGHTS = {.nops               = 620.4481910898397,
                                           .vmcnt              = 203.68521471982095,
                                           .lgkmcnt            = 318.37307256604777,
                                           .vmQueueLen         = 12,
                                           .vectorQueueSat     = 35.208369484081274,
                                           .ldsQueueSat        = 26.945053555005625,
                                           .lgkmQueueLen       = 15,
                                           .stallCycles        = 1000.0,
                                           .notMFMA            = 117.91373534215671,
                                           .isMFMA             = 59.46037213518759,
                                           .isSMEM             = 82.61122691405645,
                                           .isSControl         = 159.0019357312576,
                                           .isSALU             = 86.74764292348273,
                                           .isVMEMRead         = 161.7542358360216,
                                           .isVMEMWrite        = 128.64699076145752,
                                           .isLDSRead          = 144.1002026966658,
                                           .isLDSWrite         = 66.83668795251201,
                                           .isVALU             = 204.34498440996558,
                                           .isACCVGPRWrite     = 59.743294671257594,
                                           .isACCVGPRRead      = 90.37656910299161,
                                           .newSGPRs           = 4626.184607642332,
                                           .newVGPRs           = 73.03434870351303,
                                           .highWaterMarkSGPRs = 162.64312780347115,
                                           .highWaterMarkVGPRs = 21.300679968689767,
                                           .fractionOfSGPRs    = 105.34086248489358,
                                           .fractionOfVGPRs    = 308.3901444061333,
                                           .outOfRegisters     = 1000000000.0,
                                           .zeroFreeBarriers   = true};
        constexpr Weights GFX942_WEIGHTS = GFX90A_WEIGHTS;

        RegisterComponent(LinearWeightedCost);
        static_assert(Component::Component<LinearWeightedCost>);

        inline LinearWeightedCost::LinearWeightedCost(ContextPtr ctx)
            : Cost{ctx}
        {
            auto settingsFile = Settings::getInstance()->get(Settings::SchedulerWeights);
            if(!settingsFile.empty())
            {
                try
                {
                    m_weights = Serialization::readYAMLFile<Weights>(settingsFile);
                }
                catch(const std::exception& e)
                {
                    Throw<FatalError>(e.what(),
                                      " parsing linear weighted costs file `",
                                      settingsFile,
                                      "` specified by ",
                                      Settings::SchedulerWeights.help());
                }
            }
            else
            {
                const auto arch = ctx->targetArchitecture().target().getVersionString();
                // TODO: consolidate with other arch version discriminators
                if(arch == "gfx908")
                    m_weights = GFX908_WEIGHTS;
                else if(arch == "gfx90a")
                    m_weights = GFX90A_WEIGHTS;
                else if(arch == "gfx940" || arch == "gfx941" || arch == "gfx942")
                    m_weights = GFX942_WEIGHTS;
                else
                {
                    Log::warn("Unsupported architecture {} for linear weighted cost; defaulting to "
                              "GFX90A weights",
                              arch);
                    m_weights = GFX90A_WEIGHTS;
                }
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
