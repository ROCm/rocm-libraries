/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
            , m_factorDimArgs(std::vector<int>(1, 0))
            , m_activationType(ActivationType::None)
            , m_activationNoGuard(false)
            , m_activationEnumArg(std::vector<ActivationType>(1, ActivationType::None))
            , m_computeInputType(rocisa::DataType::Float)
            , m_f32XdlMathOp(rocisa::DataType::Float)
            , m_activationComputeType(rocisa::DataType::Float)
            , m_useUserArgs(false)
            , m_swizzleTensorA(false)
            , m_swizzleTensorB(false)
        {
            using std::static_pointer_cast;

            std::vector<bool> isComplex;
            if(args.count("problem-identifier"))
            {
                ContractionProblemGemm::IdentifierToIndices(
                    args["problem-identifier"].as<std::string>(),
                    m_freeIndices,
                    m_batchIndices,
                    m_boundIndices,
                    isComplex);

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
            if(args.count("use-bias"))
                m_useBias = args["use-bias"].as<int>();
            if(args.count("bias-source"))
                m_biasSrc = args["bias-source"].as<int>();
            if(args.count("use-scaleAB"))
                m_useScaleAB = args["use-scaleAB"].as<std::string>();
            if(args.count("use-scaleCD"))
                m_useScaleCD = args["use-scaleCD"].as<bool>();
            if(args.count("use-scaleAlphaVec"))
                m_useScaleAlphaVec = args["use-scaleAlphaVec"].as<int>();
            if(args.count("max-workspace-size"))
                m_maxWorkspaceSize = args["max-workspace-size"].as<size_t>();

            if(args.count("compute-input-type"))
            {
                //accept mix-types (i.g. Float8BFloat8); there no need to set m_computeInputTypeA and m_computeInputTypeB
                m_computeInputType = args["compute-input-type"].as<rocisa::DataType>();
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
            int activationSize = std::max(1, (int)m_activationEnumArg.size());
            int factorDimSize  = std::max(
                1, m_useScaleAlphaVec == 3 || m_useBias == 3 ? (int)m_factorDimArgs.size() : 1);
            rv.reserve(m_problemSizes.size() * activationSize * biasSize * factorDimSize);

            std::vector<size_t> aStrides, bStrides, cStrides, dStrides, eStrides, biasStrides;

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

            for(int l = 0; l < factorDimSize; l++)
            {
                for(int k = 0; k < biasSize; k++)
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
                                m_tensorTypes[ContractionProblemGemm::TENSOR::B],
                                bStrides,
                                m_tensorTypes[ContractionProblemGemm::TENSOR::C],
                                cStrides,
                                m_tensorTypes[ContractionProblemGemm::TENSOR::D],
                                dStrides,
                                m_constantValues[ContractionProblemGemm::CONST::BETA]));

                            rv.back().setComputeInputType(m_computeInputType);
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
                            rv.back().setUseE(m_useE);
                            rv.back().setOutputAmaxD(m_outputAmaxD);
                            rv.back().setKernelLanguage(m_kernelLanguage);
                            rv.back().setPerformanceMetric(m_performanceMetric);
                            rv.back().setDeterministicMode(m_deterministicMode);
                            rv.back().setSparse(m_sparse);
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
                        }
                    }
                }
            }
        }
    } // namespace Client
} // namespace TensileLite
