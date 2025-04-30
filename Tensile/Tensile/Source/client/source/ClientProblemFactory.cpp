/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

namespace Tensile
{
    namespace Client
    {
        ClientProblemFactory::ClientProblemFactory(po::variables_map const& args)
            : m_freeIndices(args["free"].as<ContractionProblem::FreeIndices>())
            , m_batchIndices(args["batch"].as<ContractionProblem::BatchIndices>())
            , m_boundIndices(args["bound"].as<ContractionProblem::BoundIndices>())
            , m_problemSizes(args["problem-size"].as<std::vector<std::vector<size_t>>>())
            , m_aZeroPads(args["a-zero-pads"].as<std::vector<std::vector<size_t>>>())
            , m_bZeroPads(args["b-zero-pads"].as<std::vector<std::vector<size_t>>>())
            , m_aType(DataType::Float)
            , m_bType(DataType::Float)
            , m_cType(DataType::Float)
            , m_dType(DataType::Float)
            , m_alphaType(DataType::Float)
            , m_betaType(DataType::Float)
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_highPrecisionAccumulate(args["high-precision-accumulate"].as<bool>())
            , m_kernelLanguage(args["kernel-language"].as<KernelLanguage>())
            , m_performanceMetric(args["performance-metric"].as<PerformanceMetric>())
            , m_deterministicMode(args["deterministic-mode"].as<bool>())
            , m_cEqualsD(args["c-equal-d"].as<bool>())
            , m_arithmeticUnit(args["arithmetic-unit"].as<ArithmeticUnit>())
            , m_aStrides(args["a-strides"].as<std::vector<std::vector<size_t>>>())
            , m_bStrides(args["b-strides"].as<std::vector<std::vector<size_t>>>())
            , m_cStrides(args["c-strides"].as<std::vector<std::vector<size_t>>>())
            , m_dStrides(args["d-strides"].as<std::vector<std::vector<size_t>>>())
            , m_aOps(args["a-ops"].as<TensorOps>())
            , m_bOps(args["b-ops"].as<TensorOps>())
            , m_cOps(args["c-ops"].as<TensorOps>())
            , m_dOps(args["d-ops"].as<TensorOps>())
            , m_aOffset(args["offset-a"].as<size_t>())
            , m_bOffset(args["offset-b"].as<size_t>())
            , m_cOffset(args["offset-c"].as<size_t>())
            , m_dOffset(args["offset-d"].as<size_t>())
            , m_f32XdlMathOp(DataType::Float)
            , m_stochasticRounding(args["stochastic-rounding"].as<bool>())

        {
            if(args.count("problem-identifier"))
                ContractionProblem::IdentifierToIndices(
                    args["problem-identifier"].as<std::string>(),
                    m_freeIndices,
                    m_batchIndices,
                    m_boundIndices,
                    m_aOps,
                    m_bOps,
                    m_cOps,
                    m_dOps);

            if(args.count("type"))
            {
                m_aType = m_bType = m_cType = m_dType = m_alphaType = m_betaType
                    = args["type"].as<DataType>();
            }

            if(args.count("a-type"))
                m_aType = args["a-type"].as<DataType>();
            if(args.count("b-type"))
                m_bType = args["b-type"].as<DataType>();
            if(args.count("c-type"))
                m_cType = args["c-type"].as<DataType>();
            if(args.count("d-type"))
                m_dType = args["d-type"].as<DataType>();
            if(args.count("alpha-type"))
                m_alphaType = args["alpha-type"].as<DataType>();
            if(args.count("beta-type"))
                m_betaType = args["beta-type"].as<DataType>();
            if(args.count("f32-xdl-math-op"))
                m_f32XdlMathOp = args["f32-xdl-math-op"].as<DataType>();

            m_beta  = DataInitialization::getValue<double>(args["init-beta"].as<InitMode>());
            m_alpha = DataInitialization::getValue<double>(args["init-alpha"].as<InitMode>());

            if(args["convolution-vs-contraction"].as<bool>())
                m_convProblemSizes
                    = args["convolution-problem"].as<std::vector<std::vector<size_t>>>();

            m_problems = createProblems();
        }

        ClientProblemFactory::~ClientProblemFactory() = default;

        std::vector<ContractionProblem> const& ClientProblemFactory::problems() const
        {
            return m_problems;
        }

        std::vector<ContractionProblem> ClientProblemFactory::createProblems()
        {
            std::vector<ContractionProblem> rv;
            rv.reserve(m_problemSizes.size());

            std::vector<size_t> aStrides, bStrides, cStrides, dStrides;

            if(m_aStrides.size() == 1)
                aStrides = m_aStrides[0];
            if(m_bStrides.size() == 1)
                bStrides = m_bStrides[0];
            if(m_cStrides.size() == 1)
                cStrides = m_cStrides[0];
            if(m_dStrides.size() == 1)
                dStrides = m_dStrides[0];

            for(int i = 0; i < m_problemSizes.size(); i++)
            {
                if(m_aStrides.size() == m_problemSizes.size())
                    aStrides = m_aStrides[i];
                if(m_bStrides.size() == m_problemSizes.size())
                    bStrides = m_bStrides[i];
                if(m_cStrides.size() == m_problemSizes.size())
                    cStrides = m_cStrides[i];
                if(m_dStrides.size() == m_problemSizes.size())
                    dStrides = m_dStrides[i];

                rv.push_back(ContractionProblem::FromIndexSizes(m_freeIndices,
                                                                m_batchIndices,
                                                                m_boundIndices,
                                                                m_problemSizes[i],
                                                                m_aType,
                                                                aStrides,
                                                                m_aOps,
                                                                m_aOffset,
                                                                m_bType,
                                                                bStrides,
                                                                m_bOps,
                                                                m_bOffset,
                                                                m_cType,
                                                                cStrides,
                                                                m_cOps,
                                                                m_cOffset,
                                                                m_dType,
                                                                dStrides,
                                                                m_dOps,
                                                                m_dOffset,
                                                                m_beta));

                rv.back().setAlphaRestriction(toScalarValueEnum(m_alpha));
                rv.back().setCEqualsD(m_cEqualsD);

                if(i < m_aZeroPads.size())
                {
                    const auto& zp = m_aZeroPads[i];
                    if(zp.size() % 4 != 0)
                        throw std::runtime_error("zero-pad must contain tuples of 4 values");
                    for(int zi = 0; zi < zp.size(); zi += 4)
                    {
                        rv.back().addAZeroPad(
                            ContractionProblem::ZeroPad({static_cast<int32_t>(zp[zi + 0]),
                                                         static_cast<int32_t>(zp[zi + 1]),
                                                         static_cast<int64_t>(zp[zi + 2]),
                                                         static_cast<int64_t>(zp[zi + 3])}));
                    }
                }
                if(i < m_bZeroPads.size())
                {
                    const auto& zp = m_bZeroPads[i];
                    if(zp.size() % 4 != 0)
                        throw std::runtime_error("zero-pad must contain tuples of 4 values");
                    for(int zi = 0; zi < zp.size(); zi += 4)
                    {
                        rv.back().addBZeroPad(
                            ContractionProblem::ZeroPad({static_cast<int32_t>(zp[zi + 0]),
                                                         static_cast<int32_t>(zp[zi + 1]),
                                                         static_cast<int64_t>(zp[zi + 2]),
                                                         static_cast<int64_t>(zp[zi + 3])}));
                    }
                }
                rv.back().setAlphaType(m_alphaType);
                rv.back().setBetaType(m_betaType);
                rv.back().setStridedBatched(m_stridedBatched);
                rv.back().setHighPrecisionAccumulate(m_highPrecisionAccumulate);
                rv.back().setKernelLanguage(m_kernelLanguage);
                rv.back().setPerformanceMetric(m_performanceMetric);
                rv.back().setDeterministicMode(m_deterministicMode);
                rv.back().setArithmeticUnit(m_arithmeticUnit);
                rv.back().setFp16AltImpl(m_fp16AltImpl);
                rv.back().setFp16AltImplRound(m_fp16AltImplRound);
                rv.back().setStochasticRounding(m_stochasticRounding);
                rv.back().setF32XdlMathOp(m_f32XdlMathOp);

                if(m_convProblemSizes.size())
                    rv.back().setConvProblemSizes(m_convProblemSizes[i]);
            }

            return rv;
        }
    } // namespace Client
} // namespace Tensile
