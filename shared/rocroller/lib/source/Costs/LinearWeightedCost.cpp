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

#include <rocRoller/Scheduling/Costs/LinearWeightedCost.hpp>

#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/Settings.hpp>

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>

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

            iot::mapOptional(io, "vmemCycles", weights.vmemCycles);
            iot::mapOptional(io, "vmemQueueSize", weights.vmemQueueSize);
            iot::mapOptional(io, "dsmemCycles", weights.dsmemCycles);
            iot::mapOptional(io, "dsmemQueueSize", weights.dsmemQueueSize);
        }

        static void mapping(IO& io, Scheduling::Weights& weights, EmptyContext& ctx)
        {
            mapping(io, weights);
        }
    };

    namespace Scheduling
    {
        constexpr Weights GFX950_WEIGHTS = {.nops               = 1001.4279088984798,
                                            .vmcnt              = 526.093932290615,
                                            .lgkmcnt            = 885.6074246484375,
                                            .vmQueueLen         = 35,
                                            .vectorQueueSat     = 1435.1211330900899,
                                            .ldsQueueSat        = 134.22254126254612,
                                            .lgkmQueueLen       = 35,
                                            .stallCycles        = 1000.0,
                                            .notMFMA            = 10986.395913365237,
                                            .isMFMA             = 81.12435660463446,
                                            .isSMEM             = 199.72403310006476,
                                            .isSControl         = 137.84120970739187,
                                            .isSALU             = 30.872084606099698,
                                            .isVMEMRead         = 85.9588364354184,
                                            .isVMEMWrite        = 251.36047887498077,
                                            .isLDSRead          = 48.2195248263684,
                                            .isLDSWrite         = 84.55911735600286,
                                            .isVALU             = 89.04421879077165,
                                            .isACCVGPRWrite     = 71.32846565329866,
                                            .isACCVGPRRead      = 58.89401902273831,
                                            .newSGPRs           = 94.44006642730224,
                                            .newVGPRs           = 49.879984390857786,
                                            .highWaterMarkSGPRs = 265.4234682031597,
                                            .highWaterMarkVGPRs = 113.2530825463529,
                                            .fractionOfSGPRs    = 45.527998466975674,
                                            .fractionOfVGPRs    = 258.19016732766454,
                                            .outOfRegisters     = 1000000000.0,
                                            .zeroFreeBarriers   = true,
                                            .vmemCycles         = 480,
                                            .vmemQueueSize      = 3,
                                            .dsmemCycles        = 95,
                                            .dsmemQueueSize     = 3};
        constexpr Weights GFX942_WEIGHTS = {.nops               = 194.43894526982916,
                                            .vmcnt              = 71.87967451224605,
                                            .lgkmcnt            = 57.99317255314543,
                                            .vmQueueLen         = 6,
                                            .vectorQueueSat     = 194.99142473663127,
                                            .ldsQueueSat        = 70.66603652975317,
                                            .lgkmQueueLen       = 5,
                                            .stallCycles        = 1000.0,
                                            .notMFMA            = 33.41526718654649,
                                            .isMFMA             = 291.18776903110466,
                                            .isSMEM             = 12.84171818450304,
                                            .isSControl         = 110.20159698039858,
                                            .isSALU             = 159.1603833670879,
                                            .isVMEMRead         = 187.42330152238435,
                                            .isVMEMWrite        = 122.30683584898141,
                                            .isLDSRead          = 35.02310763944675,
                                            .isLDSWrite         = 103.59301022907425,
                                            .isVALU             = 81.08374289168657,
                                            .isACCVGPRWrite     = 113.173666431442,
                                            .isACCVGPRRead      = 605.5787621618031,
                                            .newSGPRs           = 182.32244316409327,
                                            .newVGPRs           = 97.53552053492548,
                                            .highWaterMarkSGPRs = 167.9777964063783,
                                            .highWaterMarkVGPRs = 132.954375228908,
                                            .fractionOfSGPRs    = 73.89647742758,
                                            .fractionOfVGPRs    = 195.11069104040754,
                                            .outOfRegisters     = 1000000000.0,
                                            .zeroFreeBarriers   = true};
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

        RegisterComponent(LinearWeightedCost);
        static_assert(Component::Component<LinearWeightedCost>);

        inline LinearWeightedCost::LinearWeightedCost(ContextPtr ctx)
            : Cost{ctx}
            , m_weights(loadWeights(ctx))
        {
        }

        Weights LinearWeightedCost::loadWeights(ContextPtr ctx) const
        {
            auto settingsFile = Settings::getInstance()->get(Settings::SchedulerWeights);
            if(!settingsFile.empty())
            {
                try
                {
                    return Serialization::readYAMLFile<Weights>(settingsFile);
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
                auto const& arch = ctx->targetArchitecture().target();

                if(arch.isCDNA1GPU())
                    return GFX908_WEIGHTS;
                else if(arch.isCDNA2GPU())
                    return GFX90A_WEIGHTS;
                else if(arch.isCDNA3GPU())
                    return GFX942_WEIGHTS;
                else if(arch.isCDNA35GPU())
                    return GFX950_WEIGHTS;
                else
                {
                    Log::warn("Unsupported architecture {} for linear weighted cost; defaulting to "
                              "GFX90A weights",
                              arch.toString());
                    return GFX90A_WEIGHTS;
                }
            }
        }

        Weights const& LinearWeightedCost::getWeights() const
        {
            return m_weights;
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
            if(status.waitCount.loadcnt() >= 0 || status.waitCount.storecnt() >= 0)
            {
                auto loadcnt  = status.waitCount.loadcnt();
                auto storecnt = status.waitCount.storecnt();
                auto cnt      = WaitCount::CombineValues(loadcnt, storecnt);
                vmcnt         = 1 + maxVmcnt - cnt;
            }

            float lgkmcnt = 0;
            if(status.waitCount.kmcnt() >= 0 || status.waitCount.dscnt() >= 0)
            {
                auto kmcnt = status.waitCount.kmcnt();
                auto dscnt = status.waitCount.dscnt();
                auto cnt   = WaitCount::CombineValues(kmcnt, dscnt);
                lgkmcnt    = 1 + maxLgkmcnt - cnt;
            }

            int vmQueueLen = status.waitLengths.at(GPUWaitQueueType::LoadQueue)
                             + status.waitLengths.at(GPUWaitQueueType::StoreQueue);
            float vectorQueueSat = std::max(vmQueueLen - m_weights.vmQueueLen, 0);
            int   kmQueueLen     = status.waitLengths.at(GPUWaitQueueType::SMemQueue)
                             + status.waitLengths.at(GPUWaitQueueType::DSQueue);
            float ldsQueueSat = std::max(kmQueueLen - m_weights.lgkmQueueLen, 0);

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

            const auto& gpu            = arch.target();
            const auto& barrierLiteral = gpu.isRDNA4GPU() ? "s_barrier_wait" : "s_barrier";
            if(m_weights.zeroFreeBarriers && inst.getOpCode().rfind(barrierLiteral, 0) == 0
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

                   + m_weights.isSMEM * GPUInstructionInfo::isSMEM(inst.getOpCode()) //
                   + m_weights.isSControl * GPUInstructionInfo::isSControl(inst.getOpCode()) //
                   + m_weights.isSALU * GPUInstructionInfo::isSALU(inst.getOpCode()) //

                   + m_weights.isVMEMRead * GPUInstructionInfo::isVMEMRead(inst.getOpCode()) //
                   + m_weights.isVMEMWrite * GPUInstructionInfo::isVMEMWrite(inst.getOpCode()) //
                   + m_weights.isLDSRead * GPUInstructionInfo::isLDSRead(inst.getOpCode()) //
                   + m_weights.isLDSWrite * GPUInstructionInfo::isLDSWrite(inst.getOpCode()) //
                   + m_weights.isVALU * GPUInstructionInfo::isVALU(inst.getOpCode()) //

                   + m_weights.isACCVGPRWrite
                         * GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode()) //
                   + m_weights.isACCVGPRRead
                         * GPUInstructionInfo::isACCVGPRRead(inst.getOpCode()) //
                ;
        }
    }
}
