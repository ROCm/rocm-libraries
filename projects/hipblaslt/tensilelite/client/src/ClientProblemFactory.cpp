/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"

#include <cstddef>

namespace TensileLite
{
    namespace Client
    {
        ClientProblemFactory::ClientProblemFactory(po::variables_map const& args)
            : m_problemSizes(args["problem-size"].as<std::vector<std::vector<size_t>>>())
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_groupedGemm(args["grouped-gemm"].as<bool>())
            , m_sparse(args["sparse"].as<int>())
            , m_highPrecisionAccumulate(args["high-precision-accumulate"].as<bool>())
            , m_kernelLanguage(args["kernel-language"].as<KernelLanguage>())
            , m_performanceMetric(args["performance-metric"].as<PerformanceMetric>())
            , m_deterministicMode(args["deterministic-mode"].as<bool>())
            , m_cEqualsD(args["c-equal-d"].as<bool>())
            , m_biasTypeArgs(std::vector<rocisa::DataType>(1, rocisa::DataType::Float))
            , m_gateTypeArgs(std::vector<rocisa::DataType>(1, rocisa::DataType::Float))
            , m_factorDimArgs(std::vector<int>(1, 0))
            , m_activationType(ActivationType::None)
            , m_activationNoGuard(false)
            , m_activationEnumArg(std::vector<ActivationType>(1, ActivationType::None))
            , m_streamKHybridMode(std::vector<int>(1, 0))
            , m_computeInputTypeA(rocisa::DataType::Float)
            , m_computeInputTypeB(rocisa::DataType::Float)
            , m_f32XdlMathOp(rocisa::DataType::Float)
            , m_activationComputeType(rocisa::DataType::Float)
            , m_useUserArgs(false)
            , m_mxBlockA(args["mx-a-block"].as<int>())
            , m_mxBlockB(args["mx-b-block"].as<int>())
            , m_padMXScaleTensorFreeDim(false)
            , m_swizzleTensorA(false)
            , m_swizzleTensorB(false)
            , m_metadataLayout(args["metadata-layout"].as<int>())
            , m_aOps(args["a-ops"].as<TensorOps>())
            , m_bOps(args["b-ops"].as<TensorOps>())
            , m_cOps(args["c-ops"].as<TensorOps>())
            , m_dOps(args["d-ops"].as<TensorOps>())
        {
            using std::static_pointer_cast;

            if(m_mxBlockA || m_mxBlockB)
            {
                hipDeviceProp_t prop;
                int deviceIdx = args.count("device-idx") ? args["device-idx"].as<int>() : 0;
                HIP_CHECK_EXC(hipGetDeviceProperties(&prop, deviceIdx));
                std::string archName(prop.gcnArchName);
                m_padMXScaleTensorFreeDim = (archName.find("gfx950") != std::string::npos);
            }

            std::vector<bool> isComplex;
            if(args.count("problem-identifier"))
            {
                ContractionProblemGemm::IdentifierToIndices(
                    args["problem-identifier"].as<std::string>(),
                    m_freeIndices,
                    m_batchIndices,
                    m_boundIndices,
                    m_aOps,
                    m_bOps,
                    m_cOps,
                    m_dOps);

                for(size_t i = 0; i < isComplex.size(); i++)
                {
                    if(isComplex[i])
                    {
                        std::runtime_error("Complex is not supported.");
                    }
                }
            }
            else
            {
                std::runtime_error("Currently only accepts identifier as input.");
            }

            // Default datatype
            rocisa::DataType type = rocisa::DataType::None;
            if(args.count("type"))
            {
                type = args["type"].as<rocisa::DataType>();
            }

            // Should add problem type in ClientParamters.ini
            auto dummy     = ContractionProblemGemm::GetDummy();
            auto tensors   = dummy.tensors();
            auto constants = dummy.constants();
            m_tensorTypes.resize(tensors.size());
            m_tensorStrides.resize(tensors.size());
            m_constantTypes.resize(constants.size());
            m_constantValues.resize(constants.size());
            // Get types and values from the information from ContractionProblem
            // May contain useless information for ClientProblemFactory
            // Get tensor types
            for(size_t i = 0; i < tensors.size(); i++)
            {
                std::string typeName = tensors[i].getName() + "-type";
                if(args.count(typeName))
                {
                    m_tensorTypes[i] = args[typeName].as<rocisa::DataType>();
                }
                else
                {
                    m_tensorTypes[i] = type;
                }
                std::string strideName = tensors[i].getName() + "-strides";
                if(args.count(strideName))
                {
                    m_tensorStrides[i] = args[strideName].as<std::vector<std::vector<size_t>>>();
                }
                else
                {
                    m_tensorStrides[i] = std::vector<std::vector<size_t>>();
                }
            }

            // MX scale element types: use dedicated options (see main.cpp mx-a-type / mx-b-type).
            // Do not rely on the generic tensor loop alone — args.count("mx-a-type") is often false
            // when the value only comes from program_options default_value or from the INI merge.
            m_tensorTypes[ContractionProblemGemm::TENSOR::MXSA]
                = args["mx-a-type"].as<rocisa::DataType>();
            m_tensorTypes[ContractionProblemGemm::TENSOR::MXSB]
                = args["mx-b-type"].as<rocisa::DataType>();

            // Get constant types
            for(size_t i = 0; i < constants.size(); i++)
            {
                std::string typeName = constants[i].name + "-type";
                if(args.count(typeName))
                {
                    m_constantTypes[i] = args[typeName].as<rocisa::DataType>();
                }
                else
                {
                    m_constantTypes[i] = type;
                }
                std::string valueName = "init-" + constants[i].name;
                if(args.count(valueName))
                {
                    m_constantValues[i]
                        = DataInitialization::getValue<double>(args[valueName].as<InitMode>());
                }
                else
                {
                    m_constantValues[i] = 0;
                }
            }

            if(args.count("activation-compute-type"))
                m_activationComputeType = args["activation-compute-type"].as<rocisa::DataType>();

            if(args.count("use-e"))
                m_useE = args["use-e"].as<bool>();

            if(args.count("use-gradient"))
                m_useGradient = args["use-gradient"].as<bool>();

            if(args.count("output-amaxD"))
                m_outputAmaxD = args["output-amaxD"].as<bool>();

            if(args.count("dquant-type"))
            {
                std::string dq = args["dquant-type"].as<std::string>();
                if(dq == "tile")
                    m_dquantType = DQuantType::Tile;
                else if(dq == "mxfp8")
                    m_dquantType = DQuantType::MXFP8;
                else
                    m_dquantType = DQuantType::None;
            }
            if(args.count("dquant-size-0"))
                m_dquantSize0Override = static_cast<int>(args["dquant-size-0"].as<size_t>());
            if(args.count("dquant-size-1"))
                m_dquantSize1Override = static_cast<int>(args["dquant-size-1"].as<size_t>());
            if(args.count("use-partial-rms"))
                m_usePartialRMS = args["use-partial-rms"].as<bool>();
            if(args.count("partial-rms-residual-add"))
                m_partialRMSResidualAdd = args["partial-rms-residual-add"].as<bool>();
            if(args.count("partial-rms-quant"))
                m_partialRMSQuant = args["partial-rms-quant"].as<bool>();
            if(args.count("partial-rms-store-bf16-d"))
                m_partialRMSStoreBf16D = args["partial-rms-store-bf16-d"].as<bool>();
            if(args.count("partial-rms-mt0"))
                m_partialRMSMT0Override = static_cast<int>(args["partial-rms-mt0"].as<size_t>());
            if(args.count("partial-rms-mt1"))
                m_partialRMSMT1Override = static_cast<int>(args["partial-rms-mt1"].as<size_t>());
            if(args.count("partial-rms-gamma-type"))
                m_partialRMSGammaType = args["partial-rms-gamma-type"].as<rocisa::DataType>();
            if(args.count("partial-rms-residual-type"))
                m_partialRMSResidualType = args["partial-rms-residual-type"].as<rocisa::DataType>();
            if(args.count("use-deepseek-scale-a"))
                m_useDeepseekScaleA = args["use-deepseek-scale-a"].as<bool>();
            if(args.count("use-deepseek-scale-b"))
                m_useDeepseekScaleB = args["use-deepseek-scale-b"].as<bool>();
            if(args.count("deepseek-scale-aq0"))
                m_deepseekScaleAq0 = static_cast<int>(args["deepseek-scale-aq0"].as<size_t>());
            if(args.count("deepseek-scale-aq1"))
                m_deepseekScaleAq1 = static_cast<int>(args["deepseek-scale-aq1"].as<size_t>());
            if(args.count("deepseek-scale-bq0"))
                m_deepseekScaleBq0 = static_cast<int>(args["deepseek-scale-bq0"].as<size_t>());
            if(args.count("deepseek-scale-bq1"))
                m_deepseekScaleBq1 = static_cast<int>(args["deepseek-scale-bq1"].as<size_t>());

            if(args.count("bias-type-args"))
                m_biasTypeArgs = args["bias-type-args"].as<std::vector<rocisa::DataType>>();
            if(args.count("factor-dim-args"))
                m_factorDimArgs = args["factor-dim-args"].as<std::vector<int>>();
            if(args.count("activation-type"))
                m_activationType = args["activation-type"].as<ActivationType>();
            if(args.count("activation-no-guard"))
                m_activationNoGuard = args["activation-no-guard"].as<bool>();
            if(args.count("activation-enum-args"))
                m_activationEnumArg
                    = args["activation-enum-args"].as<std::vector<ActivationType>>();
            if(args.count("streamk-hybrid-mode"))
            {
                auto raw = args["streamk-hybrid-mode"].as<std::vector<int>>();
                if(!raw.empty())
                    m_streamKHybridMode = std::move(raw);
            }
            if(args.count("use-bias"))
                m_useBias = args["use-bias"].as<int>();
            if(args.count("bias-source"))
                m_biasSrc = args["bias-source"].as<int>();
            if(args.count("use-gate-residual"))
                m_useGateResidual = args["use-gate-residual"].as<bool>();
            if(args.count("gate-type-args"))
                m_gateTypeArgs = args["gate-type-args"].as<std::vector<rocisa::DataType>>();
            if(args.count("use-scaleAB"))
                m_useScaleAB = args["use-scaleAB"].as<std::string>();
            if(args.count("use-scaleCD"))
                m_useScaleCD = args["use-scaleCD"].as<bool>();
            if(args.count("use-scaleAlphaVec"))
                m_useScaleAlphaVec = args["use-scaleAlphaVec"].as<int>();
            if(args.count("max-workspace-size"))
                m_maxWorkspaceSize = args["max-workspace-size"].as<size_t>();

            if(args.count("compute-input-type-A"))
            {
                //accept mix-types (i.g. Float8BFloat8); there no need to set m_computeInputTypeA and m_computeInputTypeB
                m_computeInputTypeA = args["compute-input-type-A"].as<rocisa::DataType>();
            }

            if(args.count("compute-input-type-B"))
            {
                //accept mix-types (i.g. Float8BFloat8); there no need to set m_computeInputTypeA and m_computeInputTypeB
                m_computeInputTypeB = args["compute-input-type-B"].as<rocisa::DataType>();
            }

            if(args.count("f32-xdl-math-op"))
            {
                m_f32XdlMathOp = args["f32-xdl-math-op"].as<rocisa::DataType>();
            }

            if(args.count("swizzle-tensor-a"))
            {
                m_swizzleTensorA = args["swizzle-tensor-a"].as<bool>();
            }

            if(args.count("swizzle-tensor-b"))
            {
                m_swizzleTensorB = args["swizzle-tensor-b"].as<bool>();
            }

            if(args.count("use-user-args"))
            {
                m_useUserArgs = args["use-user-args"].as<bool>();
            }

            if(m_groupedGemm)
            {
                auto problems = std::make_shared<ContractionProblemGroupedGemm>();
                createProblems(problems->gemms);
                m_problems.push_back(static_pointer_cast<ContractionProblem>(problems));
            }
            else
            {
                std::vector<ContractionProblemGemm> v;
                createProblems(v);
                for(auto& it : v)
                {
                    auto problem     = std::make_shared<ContractionProblemGemm>();
                    (*problem.get()) = it;
                    m_problems.push_back(static_pointer_cast<ContractionProblem>(problem));
                }
            }
        }

        ClientProblemFactory::~ClientProblemFactory() = default;

        std::vector<std::shared_ptr<ContractionProblem>> const&
            ClientProblemFactory::problems() const
        {
            return m_problems;
        }

        void ClientProblemFactory::createProblems(std::vector<ContractionProblemGemm>& rv)
        {
            rv.clear();
            int biasSize       = std::max(1, (int)m_biasTypeArgs.size());
            int gateSize       = std::max(1, (int)m_gateTypeArgs.size());
            int activationSize = std::max(1, (int)m_activationEnumArg.size());
            int factorDimSize  = std::max(
                1, m_useScaleAlphaVec == 3 || m_useBias == 3 ? (int)m_factorDimArgs.size() : 1);
            // StreamK=5 hybrid-mode toggle variants. When the YAML sets
            // StreamKHybridMode: [0, 1] each base problem is replayed
            // twice (one static pass, one dynamic pass) so a single
            // tlrun invocation covers both code paths of an SK5 kernel.
            int streamKHybridModeSize = std::max(1, (int)m_streamKHybridMode.size());
            rv.reserve(m_problemSizes.size() * activationSize * biasSize * gateSize * factorDimSize
                       * streamKHybridModeSize);

            std::vector<size_t> aStrides, bStrides, cStrides, dStrides, eStrides, biasStrides,
                gateStrides;

            if(m_tensorStrides[ContractionProblemGemm::TENSOR::A].size() == 1)
                aStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::A][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::B].size() == 1)
                bStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::B][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::C].size() == 1)
                cStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::C][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::D].size() == 1)
                dStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::D][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::E].size() == 1)
                eStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::E][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::BIAS].size() == 1)
                biasStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::BIAS][0];
            if(m_tensorStrides[ContractionProblemGemm::TENSOR::GATE_RESIDUAL].size() == 1)
                gateStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::GATE_RESIDUAL][0];

            // Outer loop is intentionally kept at the same indentation
            // as the inner factor/bias/activation/problem-size loops to
            // avoid re-indenting ~200 lines of unrelated body code.
            for(int m = 0; m < streamKHybridModeSize; m++)
            {
            for(int l = 0; l < factorDimSize; l++)
            {
                for(int k = 0; k < biasSize; k++)
                {
                    for(int g = 0; g < gateSize; g++)
                    {
                        for(int j = 0; j < activationSize; j++)
                        {
                            for(int i = 0; i < m_problemSizes.size(); i++)
                            {
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::A].size()
                                == m_problemSizes.size())
                                    aStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::A][i];
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::B].size()
                                == m_problemSizes.size())
                                    bStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::B][i];
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::C].size()
                                == m_problemSizes.size())
                                    cStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::C][i];
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::D].size()
                                == m_problemSizes.size())
                                    dStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::D][i];
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::E].size()
                                == m_problemSizes.size())
                                    eStrides = m_tensorStrides[ContractionProblemGemm::TENSOR::E][i];
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::BIAS].size()
                                == m_problemSizes.size())
                                    biasStrides
                                        = m_tensorStrides[ContractionProblemGemm::TENSOR::BIAS][i];
                                if(m_tensorStrides[ContractionProblemGemm::TENSOR::GATE_RESIDUAL].size()
                                == m_problemSizes.size())
                                    gateStrides
                                        = m_tensorStrides[ContractionProblemGemm::TENSOR::GATE_RESIDUAL][i];

                                if(m_useBias && m_useScaleAlphaVec && m_useBias != m_useScaleAlphaVec)
                                    continue;

                                int factorDim = (m_useScaleAlphaVec == 1 || m_useBias == 1)   ? 0
                                                : (m_useScaleAlphaVec == 2 || m_useBias == 2) ? 1
                                                : (m_useScaleAlphaVec == 3 || m_useBias == 3)
                                                    ? m_factorDimArgs[l]
                                                    : 0;
                                rv.push_back(ContractionProblemGemm::FromIndexSizes(
                                    m_freeIndices,
                                    m_batchIndices,
                                    m_boundIndices,
                                    m_problemSizes[i],
                                    m_tensorTypes[ContractionProblemGemm::TENSOR::A],
                                    aStrides,
                                    m_aOps,
                                    m_tensorTypes[ContractionProblemGemm::TENSOR::B],
                                    bStrides,
                                    m_bOps,
                                    m_tensorTypes[ContractionProblemGemm::TENSOR::C],
                                    cStrides,
                                    m_cOps,
                                    m_tensorTypes[ContractionProblemGemm::TENSOR::D],
                                    dStrides,
                                    m_dOps,
                                    m_constantValues[ContractionProblemGemm::CONST::BETA]));

                                rv.back().setComputeInputTypeA(m_computeInputTypeA);
                                rv.back().setComputeInputTypeB(m_computeInputTypeB);
                                rv.back().setAlphaRestriction(toScalarValueEnum(
                                    m_constantValues[ContractionProblemGemm::CONST::ALPHA]));
                                rv.back().setCEqualsD(m_cEqualsD);
                                rv.back().setAlphaType(
                                    m_constantTypes[ContractionProblemGemm::CONST::ALPHA]);
                                rv.back().setBetaType(
                                    m_constantTypes[ContractionProblemGemm::CONST::BETA]);
                                rv.back().setStridedBatched(m_stridedBatched);
                                rv.back().setHighPrecisionAccumulate(m_highPrecisionAccumulate);
                                rv.back().setUseGradient(m_useGradient);
                                rv.back().setUseBias(m_useBias);
                                rv.back().setUseGateResidual(m_useGateResidual);
                                rv.back().setUseE(m_useE);
                                rv.back().setOutputAmaxD(m_outputAmaxD);
                                rv.back().setDquantType(m_dquantType);
                                rv.back().setUseDeepseekScaleA(m_useDeepseekScaleA);
                                rv.back().setUseDeepseekScaleB(m_useDeepseekScaleB);
                                rv.back().setDeepseekScaleAq0(m_deepseekScaleAq0);
                                rv.back().setDeepseekScaleAq1(m_deepseekScaleAq1);
                                rv.back().setDeepseekScaleBq0(m_deepseekScaleBq0);
                                rv.back().setDeepseekScaleBq1(m_deepseekScaleBq1);
                                rv.back().setUsePartialRMS(m_usePartialRMS);
                                rv.back().setPartialRMSResidualAdd(m_partialRMSResidualAdd);
                                rv.back().setKernelLanguage(m_kernelLanguage);
                                rv.back().setPerformanceMetric(m_performanceMetric);
                                rv.back().setDeterministicMode(m_deterministicMode);
                                rv.back().setSparse(m_sparse, m_metadataLayout);
                                rv.back().setActivationType(m_activationType);
                                rv.back().setWorkspaceSize(m_maxWorkspaceSize);
                                rv.back().setSwizzleTensorA(m_swizzleTensorA);
                                rv.back().setSwizzleTensorB(m_swizzleTensorB);
                                if(k < m_biasTypeArgs.size())
                                {
                                    auto length
                                        = (m_biasSrc == ContractionProblemGemm::TENSOR::B)
                                            ? rv.back().d().sizes()[1]
                                        : (m_useBias == 1
                                            || (m_biasSrc != ContractionProblemGemm::TENSOR::D))
                                            ? rv.back().d().sizes()[0]
                                            : rv.back().d().sizes()[factorDim];
                                    bool isBiasOutput = m_useGradient ? true : false;
                                    auto biasStride   = biasStrides.size() < 2 ? 0 : biasStrides[2];
                                    rv.back().setBias(
                                        m_biasTypeArgs[k],
                                        length,
                                        biasStride,
                                        isBiasOutput,
                                        static_cast<ContractionProblemGemm::TENSOR>(m_biasSrc),
                                        factorDim);
                                }
                                else
                                {
                                    rv.back().setBias(rocisa::DataType::None, 0, 0);
                                }
                                if(m_useGateResidual)
                                {
                                    auto const& d     = rv.back().d();
                                    auto const& gs    = gateStrides.empty() ? d.strides() : gateStrides;
                                    rocisa::DataType gType = (g < (int)m_gateTypeArgs.size())
                                                                ? m_gateTypeArgs[g]
                                                                : rocisa::DataType::None;
                                    rv.back().setGateResidual(gType, d.sizes(), gs);
                                }
                                if(m_useE)
                                {
                                    bool isEOutput = true;
                                    if(m_useGradient)
                                        isEOutput = false;
                                    rv.back().setE(m_tensorTypes[ContractionProblemGemm::TENSOR::E],
                                                rv.back().d().sizes(),
                                                eStrides,
                                                isEOutput);
                                }
                                if(m_outputAmaxD)
                                {
                                    bool isOutput = true;
                                    rv.back().setAmaxD(
                                        m_tensorTypes[ContractionProblemGemm::TENSOR::AMAXD], isOutput);
                                    rv.back().setSynchronizer(rocisa::DataType::Int32, 1);
                                }
                                else
                                {
                                    rv.back().setSynchronizer(
                                        m_constantTypes[ContractionProblemGemm::CONST::ALPHA], 409600);
                                }
                                if(m_usePartialRMS)
                                {
                                    // PartialRMSAxis=0: free0=N_hidden tiles with MT0,
                                    // free1=M_tokens padded with MT1.
                                    // d.sizes()[0]=N_hidden (free0), d.sizes()[1]=M_tokens (free1).
                                    // partialBuf[token, t_free0]: shape [M_tokens_padded, n_d].
                                    size_t nHidden  = rv.back().d().sizes()[0];  // free0
                                    size_t mTokens  = rv.back().d().sizes()[1];  // free1

                                    int mt0      = m_partialRMSMT0Override > 0 ? m_partialRMSMT0Override : 16;
                                    int mt1      = m_partialRMSMT1Override > 0 ? m_partialRMSMT1Override : 16;
                                    size_t mPadded  = ((mTokens  + static_cast<size_t>(mt1) - 1) / static_cast<size_t>(mt1)) * static_cast<size_t>(mt1);
                                    size_t nTilesN  = (nHidden   + static_cast<size_t>(mt0) - 1) / static_cast<size_t>(mt0);

                                    rv.back().setPartialRMSMT0(mt0);
                                    rv.back().setPartialRMSMT1(mt1);
                                    rv.back().setRMSGamma(m_partialRMSGammaType, nHidden);
                                    rv.back().setPartialRMSQuant(m_partialRMSQuant);
                                    // Double the row count so both halves fit: first half = Σx²,
                                    // second half = amax(|D|)/448.
                                    size_t pbRows = m_partialRMSQuant ? 2 * mPadded : mPadded;
                                    rv.back().setPartialBuf(pbRows, nTilesN);
                                    rv.back().setPartialRMSResidualAdd(m_partialRMSResidualAdd);
                                    if(m_partialRMSResidualAdd)
                                        rv.back().setResidual(m_partialRMSResidualType, mTokens, nHidden);
                                    if(m_partialRMSStoreBf16D)
                                    {
                                        // ResidualOut: same shape as D (fp8 output), bf16 elements.
                                        auto const& dSizes   = rv.back().d().sizes();
                                        auto const& dStrides = rv.back().d().strides();
                                        rv.back().setResidualOut(rocisa::DataType::BFloat16,
                                                                 dSizes, dStrides);
                                    }
                                }
                                if(m_dquantType != DQuantType::None)
                                {
                                    size_t M  = rv.back().d().sizes()[0];
                                    size_t N  = rv.back().d().sizes()[1];
                                    int    q0 = m_dquantSize0Override > 0 ? m_dquantSize0Override : static_cast<int>(M);
                                    int    q1 = m_dquantSize1Override > 0 ? m_dquantSize1Override : static_cast<int>(N);
                                    rv.back().setDquantSize0(q0);
                                    rv.back().setDquantSize1(q1);
                                    rv.back().setQuantScale((M + q0 - 1) / q0, (N + q1 - 1) / q1);
                                    rv.back().setMxScale((N + q1 - 1) / q1, (M + q0 - 1) / q0);
                                }
                                if(m_useDeepseekScaleA)
                                {
                                    size_t M    = rv.back().d().sizes()[0];
                                    size_t K    = rv.back().boundSize(0);
                                    int    aq1  = m_deepseekScaleAq1 > 0 ? m_deepseekScaleAq1 : 128;
                                    // Device buffer: [ceil(M/64), nKBlocks, 64] fp32.
                                    // 64 = WavefrontSize; each slot broadcasts one value per lane.
                                    size_t nRowGroups = (M + 63) / 64;
                                    size_t nKBlocks   = (K + aq1 - 1) / aq1;
                                    rv.back().setScaleADeepseek(nRowGroups * nKBlocks * 64);
                                }
                                if(m_useDeepseekScaleB)
                                {
                                    size_t K    = rv.back().boundSize(0);
                                    size_t N    = rv.back().d().sizes()[1];
                                    int    aq1  = m_deepseekScaleAq1 > 0 ? m_deepseekScaleAq1 : 128;
                                    int    bq1  = m_deepseekScaleBq1 > 0 ? m_deepseekScaleBq1 : 128;
                                    // Device buffer: [nNBlocks, nKBlocks, 64] fp32.
                                    size_t nNBlocks = (N + bq1 - 1) / bq1;
                                    size_t nKBlocks = (K + aq1 - 1) / aq1;
                                    rv.back().setScaleBDeepseek(nNBlocks * nKBlocks * 64);
                                }
                                if(j < m_activationEnumArg.size())
                                {
                                    rv.back().setParams().setActivationEnum(m_activationEnumArg[j]);
                                }
                                else
                                {
                                    rv.back().setActivationType(m_activationType);
                                }
                                rv.back().setActivationNoGuard(m_activationNoGuard);
                            rv.back().setUseScaleAB(m_useScaleAB);
                            if(m_useScaleAB == "Scalar")
                            {
                                rv.back().setScaleA(
                                    m_constantTypes[ContractionProblemGemm::CONST::ALPHA], 1);
                                rv.back().setScaleB(
                                    m_constantTypes[ContractionProblemGemm::CONST::ALPHA], 1);
                            }
                            else if(m_useScaleAB == "Vector")
                            {
                                rv.back().setScaleA(
                                    m_constantTypes[ContractionProblemGemm::CONST::ALPHA],
                                    rv.back().d().sizes()[0]);
                                rv.back().setScaleB(
                                    m_constantTypes[ContractionProblemGemm::CONST::ALPHA],
                                    rv.back().d().sizes()[1]);
                            }
                            rv.back().setUseScaleCD(m_useScaleCD);
                            if(m_useScaleCD)
                            {
                                rv.back().setScaleC(
                                    m_constantTypes[ContractionProblemGemm::CONST::BETA]);
                                rv.back().setScaleD(
                                    m_constantTypes[ContractionProblemGemm::CONST::BETA]);
                            }
                            rv.back().setUseScaleAlphaVec(m_useScaleAlphaVec);
                            rv.back().setScaleAlphaVec(
                                m_constantTypes[ContractionProblemGemm::CONST::ALPHA],
                                rv.back().d().sizes()[factorDim],
                                factorDim);
                            rv.back().setGroupedGemm(m_groupedGemm);
                            rv.back().setF32XdlMathOp(m_f32XdlMathOp);
                            rv.back().setActivationComputeType(m_activationComputeType);
                            rv.back().setUseDeviceUserArguments(m_useUserArgs);
                            if(m_mxBlockA)
                            {
                                rv.back().setMXScaleA(m_tensorTypes[ContractionProblemGemm::TENSOR::MXSA], m_mxBlockA, {}, m_padMXScaleTensorFreeDim);
                            }
                            if(m_mxBlockB)
                            {
                                rv.back().setMXScaleB(m_tensorTypes[ContractionProblemGemm::TENSOR::MXSB], m_mxBlockB, {}, m_padMXScaleTensorFreeDim);
                            }
                            // StreamK=5 hybrid-mode toggle. Accepts the full
                            // tri-state {0=OFF (static), 1=ON (dynamic per-XCD
                            // work-queue), 2=AUTO (heuristic)}. The reference
                            // path is unaffected by the choice, so all three
                            // values validate cleanly against the CPU ref.
                            // YAML sweep tests should prefer [0, 1] to
                            // guarantee both deterministic sub-paths are
                            // exercised; AUTO is most useful when overriding
                            // from the command line (e.g.
                            // `--streamk-hybrid-mode 2`) to exercise the
                            // runtime heuristic end-to-end on a real problem.
                            if(m < (int)m_streamKHybridMode.size())
                            {
                                rv.back().setParams().setStreamKTileSchedulingMode(
                                    m_streamKHybridMode[m]);
                            }
                            } // for i
                        } // for j
                    } // for g
                } // for k
            } // for l
            } // streamk-hybrid-mode outer loop (for m)
        }
    } // namespace Client
} // namespace TensileLite
