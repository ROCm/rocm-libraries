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

#include "DataInitialization.hpp"
#include "DataInitializationTyped.hpp"

#include <Tensile/Utils.hpp>

#include <hip/hip_runtime.h>

namespace Tensile
{
    namespace Client
    {
        std::string ToString(InitMode mode)
        {
            switch(mode)
            {
            case InitMode::Zero:
                return "Zero";
            case InitMode::One:
                return "One";
            case InitMode::Two:
                return "Two";
            case InitMode::Random:
                return "Random";
            case InitMode::NaN:
                return "NaN";
            case InitMode::Inf:
                return "Inf";
            case InitMode::BadInput:
                return "BadInput";
            case InitMode::BadOutput:
                return "BadOutput";
            case InitMode::SerialIdx:
                return "SerialIdx";
            case InitMode::SerialDim0:
                return "SerialDim0";
            case InitMode::SerialDim1:
                return "SerialDim1";
            case InitMode::Identity:
                return "Identity";
            case InitMode::TrigSin:
                return "TrigSin";
            case InitMode::TrigCos:
                return "TrigCos";
            case InitMode::TrigAbsSin:
                return "TrigAbsSin";
            case InitMode::TrigAbsCos:
                return "TrigAbsCos";
            case InitMode::RandomNarrow:
                return "RandomNarrow";
            case InitMode::NegOne:
                return "NegOne";
            case InitMode::Max:
                return "Max";
            case InitMode::DenormMin:
                return "DenormMin";
            case InitMode::DenormMax:
                return "DenormMax";

            case InitMode::Count:
                break;
            }

            throw std::runtime_error(
                concatenate("Invalid InitMode value: ", static_cast<int>(mode)));
        }

        std::ostream& operator<<(std::ostream& stream, InitMode const& mode)
        {
            return stream << ToString(mode);
        }

        std::istream& operator>>(std::istream& stream, InitMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == ToString(InitMode::Zero))
                mode = InitMode::Zero;
            else if(strValue == ToString(InitMode::One))
                mode = InitMode::One;
            else if(strValue == ToString(InitMode::Two))
                mode = InitMode::Two;
            else if(strValue == ToString(InitMode::Random))
                mode = InitMode::Random;
            else if(strValue == ToString(InitMode::NaN))
                mode = InitMode::NaN;
            else if(strValue == ToString(InitMode::Inf))
                mode = InitMode::Inf;
            else if(strValue == ToString(InitMode::BadInput))
                mode = InitMode::BadInput;
            else if(strValue == ToString(InitMode::BadOutput))
                mode = InitMode::BadOutput;
            else if(strValue == ToString(InitMode::SerialIdx))
                mode = InitMode::SerialIdx;
            else if(strValue == ToString(InitMode::SerialDim0))
                mode = InitMode::SerialDim0;
            else if(strValue == ToString(InitMode::SerialDim1))
                mode = InitMode::SerialDim1;
            else if(strValue == ToString(InitMode::Identity))
                mode = InitMode::Identity;
            else if(strValue == ToString(InitMode::TrigSin))
                mode = InitMode::TrigSin;
            else if(strValue == ToString(InitMode::TrigCos))
                mode = InitMode::TrigCos;
            else if(strValue == ToString(InitMode::TrigAbsSin))
                mode = InitMode::TrigAbsSin;
            else if(strValue == ToString(InitMode::TrigAbsCos))
                mode = InitMode::TrigAbsCos;
            else if(strValue == ToString(InitMode::RandomNarrow))
                mode = InitMode::RandomNarrow;
            else if(strValue == ToString(InitMode::NegOne))
                mode = InitMode::NegOne;
            else if(strValue == ToString(InitMode::Max))
                mode = InitMode::Max;
            else if(strValue == ToString(InitMode::DenormMin))
                mode = InitMode::DenormMin;
            else if(strValue == ToString(InitMode::DenormMax))
                mode = InitMode::DenormMax;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(InitMode::Count))
                    mode = static_cast<InitMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to InitMode."));
            }
            else
            {
                throw std::runtime_error(concatenate("Can't convert ", strValue, " to InitMode."));
            }

            return stream;
        }

        std::ostream& operator<<(std::ostream& stream, BoundsCheckMode const& mode)
        {
            std::string strValue;

            if(mode == BoundsCheckMode::Disable)
                strValue = "Disable";
            else if(mode == BoundsCheckMode::NaN)
                strValue = "NaN";
            else if(mode == BoundsCheckMode::GuardPageFront)
                strValue = "GuardPageFront";
            else if(mode == BoundsCheckMode::GuardPageBack)
                strValue = "GuardPageBack";
            else if(mode == BoundsCheckMode::GuardPageAll)
                strValue = "GuardPageAll";
            else
                throw std::runtime_error(
                    concatenate("Invalid BoundsCheckMode value: ", static_cast<int>(mode)));

            return stream << strValue;
        }

        std::istream& operator>>(std::istream& stream, BoundsCheckMode& mode)
        {
            std::string strValue;
            stream >> strValue;

            if(strValue == "Disable")
                mode = BoundsCheckMode::Disable;
            else if(strValue == "NaN")
                mode = BoundsCheckMode::NaN;
            else if(strValue == "GuardPageFront")
                mode = BoundsCheckMode::GuardPageFront;
            else if(strValue == "GuardPageBack")
                mode = BoundsCheckMode::GuardPageBack;
            else if(strValue == "GuardPageAll")
                mode = BoundsCheckMode::GuardPageAll;
            else if(std::all_of(strValue.begin(), strValue.end(), isdigit))
            {
                int value = atoi(strValue.c_str());
                if(value >= 0 && value < static_cast<int>(BoundsCheckMode::MaxMode))
                    mode = static_cast<BoundsCheckMode>(value);
                else
                    throw std::runtime_error(
                        concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
            }
            else
            {
                throw std::runtime_error(
                    concatenate("Can't convert ", strValue, " to BoundsCheckMode."));
            }

            return stream;
        }

        double DataInitialization::GetRepresentativeBetaValue(po::variables_map const& args)
        {
            auto argValue = args["init-beta"].as<int>();

            if(argValue == 0)
                return 0.0;

            if(argValue == 1)
                return 1.0;

            return 1.5;
        }

        template <typename TypedInputs>
        std::shared_ptr<TypedDataInitialization<TypedInputs>>
            DataInitialization::GetTyped(po::variables_map const&    args,
                                         ClientProblemFactory const& problemFactory,
                                         size_t                      maxWorkspaceSize)
        {
            auto* ptr
                = new TypedDataInitialization<TypedInputs>(args, problemFactory, maxWorkspaceSize);

            return std::shared_ptr<TypedDataInitialization<TypedInputs>>(ptr);
        }

        std::shared_ptr<DataInitialization>
            DataInitialization::Get(po::variables_map const&    args,
                                    ClientProblemFactory const& problemFactory,
                                    size_t                      maxWorkspaceSize)
        {
            auto aType     = args["a-type"].as<DataType>();
            auto bType     = args["b-type"].as<DataType>();
            auto cType     = args["c-type"].as<DataType>();
            auto dType     = args["d-type"].as<DataType>();
            auto alphaType = args["alpha-type"].as<DataType>();
            auto betaType  = args["beta-type"].as<DataType>();

            auto contractionInputsTypeId
                = ContractionInputs::TypeId(aType, bType, cType, dType, alphaType, betaType);

            switch(contractionInputsTypeId)
            {
            case ContractionInputs_S_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_S_S_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_D_D_D::TypeId():
            {
                return GetTyped<ContractionInputs_D_D_D>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_C_C_C::TypeId():
            {
                return GetTyped<ContractionInputs_C_C_C>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_Z_Z_Z::TypeId():
            {
                return GetTyped<ContractionInputs_Z_Z_Z>(args, problemFactory, maxWorkspaceSize);
            }
#ifdef TENSILE_USE_HALF
            case ContractionInputs_H_H_H::TypeId():
            {
                return GetTyped<ContractionInputs_H_H_H>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_H_H_S::TypeId():
            {
                return GetTyped<ContractionInputs_H_H_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_H_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_H_S_S>(args, problemFactory, maxWorkspaceSize);
            }
#endif // TENSILE_USE_HALF
            case ContractionInputs_I8x4_I32_I32::TypeId():
            {
                return GetTyped<ContractionInputs_I8x4_I32_I32>(
                    args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_I32_I32_I32::TypeId():
            {
                return GetTyped<ContractionInputs_I32_I32_I32>(
                    args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_I8_I32_I32::TypeId():
            {
                return GetTyped<ContractionInputs_I8_I32_I32>(
                    args, problemFactory, maxWorkspaceSize);
            }
#ifdef TENSILE_USE_BF16
            case ContractionInputs_B_B_S::TypeId():
            {
                return GetTyped<ContractionInputs_B_B_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_B_S_S>(args, problemFactory, maxWorkspaceSize);
            }
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
            case ContractionInputs_F8_F8_S::TypeId():
            {
                return GetTyped<ContractionInputs_F8_F8_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_F8_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_F8_S_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B8_B8_S::TypeId():
            {
                return GetTyped<ContractionInputs_B8_B8_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B8_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_B8_S_S>(args, problemFactory, maxWorkspaceSize);
            }
            // Hybrid cases: F8B8SS, B8SS
            case ContractionInputs_F8B8_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_F8B8_S_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B8F8_S_S::TypeId():
            {
                return GetTyped<ContractionInputs_B8F8_S_S>(args, problemFactory, maxWorkspaceSize);
            }
            // Hybrid cases with To = B8
            case ContractionInputs_F8B8_B8_S::TypeId():
            {
                return GetTyped<ContractionInputs_F8B8_B8_S>(
                    args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B8F8_B8_S::TypeId():
            {
                return GetTyped<ContractionInputs_B8F8_B8_S>(
                    args, problemFactory, maxWorkspaceSize);
            }
            // cases with To = half
            case ContractionInputs_F8_H_S::TypeId():
            {
                return GetTyped<ContractionInputs_F8_H_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B8_H_S::TypeId():
            {
                return GetTyped<ContractionInputs_B8_H_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_F8B8_H_S::TypeId():
            {
                return GetTyped<ContractionInputs_F8B8_H_S>(args, problemFactory, maxWorkspaceSize);
            }
            case ContractionInputs_B8F8_H_S::TypeId():
            {
                return GetTyped<ContractionInputs_B8F8_H_S>(args, problemFactory, maxWorkspaceSize);
            }
#endif // TENSILE_USE_FP8_BF8
            default:;
            }

            throw std::runtime_error(concatenate("Invalid combination of data types: ",
                                                 "a: ",
                                                 aType,
                                                 ", b: ",
                                                 bType,
                                                 ", c: ",
                                                 cType,
                                                 ", d: ",
                                                 dType,
                                                 ", alpha: ",
                                                 alphaType,
                                                 ", beta: ",
                                                 betaType));
        }

        DataInitialization::DataInitialization(po::variables_map const&    args,
                                               ClientProblemFactory const& problemFactory,
                                               size_t                      maxWorkspaceSize)
            : m_aInit(args["init-a"].as<InitMode>())
            , m_bInit(args["init-b"].as<InitMode>())
            , m_cInit(args["init-c"].as<InitMode>())
            , m_dInit(args["init-d"].as<InitMode>())
            , m_alphaInit(args["init-alpha"].as<InitMode>())
            , m_betaInit(args["init-beta"].as<InitMode>())
            , m_aMaxElements(0)
            , m_bMaxElements(0)
            , m_cMaxElements(0)
            , m_dMaxElements(0)
            , m_maxBatch(0)
            , m_stridedBatched(args["strided-batched"].as<bool>())
            , m_cEqualsD(args["c-equal-d"].as<bool>())
            , m_elementsToValidate(args["num-elements-to-validate"].as<int>())
            , m_keepPristineCopyOnGPU(args["pristine-on-gpu"].as<bool>())
            , m_workspaceSize(maxWorkspaceSize)
        {
            m_boundsCheck    = args["bounds-check"].as<BoundsCheckMode>();
            m_curBoundsCheck = m_boundsCheck;

            if(m_boundsCheck == BoundsCheckMode::GuardPageAll)
            {
                //GuardPageAll needs 2 runs per solution.
                //First run perform front side guard page checking.
                m_curBoundsCheck     = BoundsCheckMode::GuardPageFront;
                m_numRunsPerSolution = 2;
            }

            if(args.count("convolution-vs-contraction"))
                m_convolutionVsContraction = args["convolution-vs-contraction"].as<bool>();

            for(auto const& problem : problemFactory.problems())
            {
                m_aMaxElements = std::max(m_aMaxElements, problem.a().totalAllocatedElements());
                m_bMaxElements = std::max(m_bMaxElements, problem.b().totalAllocatedElements());
                m_cMaxElements = std::max(m_cMaxElements, problem.c().totalAllocatedElements());
                m_dMaxElements = std::max(m_dMaxElements, problem.d().totalAllocatedElements());

                size_t numOfBatch = 1;
                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                    numOfBatch *= problem.batchSize(i);
                m_maxBatch = std::max(m_maxBatch, numOfBatch);
            }

            if(m_curBoundsCheck == BoundsCheckMode::NaN)
            {
                m_aMaxElements += 1024;
                m_bMaxElements += 1024;
                m_cMaxElements += 1024;
                m_dMaxElements += 1024;
            }
            else if(m_curBoundsCheck == BoundsCheckMode::GuardPageFront
                    || m_curBoundsCheck == BoundsCheckMode::GuardPageBack)
            {
                unsigned int aRoundUpSize
                    = pageSize / DataTypeInfo::Get(args["a-type"].as<DataType>()).elementSize;
                unsigned int bRoundUpSize
                    = pageSize / DataTypeInfo::Get(args["b-type"].as<DataType>()).elementSize;
                unsigned int cRoundUpSize
                    = pageSize / DataTypeInfo::Get(args["c-type"].as<DataType>()).elementSize;
                unsigned int dRoundUpSize
                    = pageSize / DataTypeInfo::Get(args["d-type"].as<DataType>()).elementSize;

                m_aMaxElements = RoundUpToMultiple<unsigned int>(m_aMaxElements, aRoundUpSize);
                m_bMaxElements = RoundUpToMultiple<unsigned int>(m_bMaxElements, bRoundUpSize);
                m_cMaxElements = RoundUpToMultiple<unsigned int>(m_cMaxElements, cRoundUpSize);
                m_dMaxElements = RoundUpToMultiple<unsigned int>(m_dMaxElements, dRoundUpSize);
            }
            m_problemDependentData = IsProblemDependent(m_aInit) || IsProblemDependent(m_bInit)
                                     || IsProblemDependent(m_cInit) || IsProblemDependent(m_dInit);
        }

        DataInitialization::~DataInitialization() {}
    } // namespace Client
} // namespace Tensile
