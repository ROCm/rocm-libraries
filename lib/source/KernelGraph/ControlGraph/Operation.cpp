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

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller::KernelGraph::ControlGraph
{
    VariableType getVariableType(Operation const& op)
    {
        auto visitor = [](auto const& op) -> VariableType {
            using T = std::decay_t<decltype(op)>;
            if constexpr(CHasVarTypeMember<T>)
            {
                return op.varType;
            }

            Throw<FatalError>(ShowValue(op), " has no varType member.");
        };

        return std::visit(visitor, op);
    }

    void setVariableType(Operation& op, VariableType varType)
    {
        auto visitor = [&varType](auto& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr(CHasVarTypeMember<T>)
            {
                op.varType = varType;
            }
            else
            {
                Throw<FatalError>(ShowValue(op), " has no varType member.");
            }
        };

        std::visit(visitor, op);
    }

    SetCoordinate::SetCoordinate() = default;
    SetCoordinate::SetCoordinate(Expression::ExpressionPtr value)
        : value(value)
    {
    }

    std::string ForLoopOp::toString() const
    {
        return concatenate(name(), " ", loopName, ": ", condition);
    }

    std::string ConditionalOp::toString() const
    {
        return concatenate(name(), " ", conditionName, ": ", condition);
    }

    std::string AssertOp::toString() const
    {
        std::string nameString      = assertName.empty() ? concatenate(" ", assertName) : "";
        std::string conditionString = condition != nullptr ? concatenate(": ", condition) : "";
        return concatenate(name(), nameString, conditionString);
    }

    std::string DoWhileOp::toString() const
    {
        return concatenate(name(), " ", loopName, ": ", condition);
    }

    std::string UnrollOp::toString() const
    {
        return concatenate(name(), " ", size);
    }

    std::string Assign::toString() const
    {
        return concatenate(name(), " ", regType, " ", expression);
    }

    std::string SeedPRNG::toString() const
    {
        return concatenate(name(), " ", addTID);
    }

    ComputeIndex::ComputeIndex() = default;
    ComputeIndex::ComputeIndex(bool     forward,
                               DataType valueType,
                               DataType offsetType,
                               DataType strideType)
        : forward(forward)
        , valueType(valueType)
        , offsetType(offsetType)
        , strideType(strideType)
    {
    }

    LoadLinear::LoadLinear() = default;
    LoadLinear::LoadLinear(rocRoller::VariableType const varType)
        : varType(varType)
    {
    }

    LoadTiled::LoadTiled() = default;
    LoadTiled::LoadTiled(rocRoller::VariableType const varType, bool const isTransposedTile)
        : varType(varType)
        , isTransposedTile(isTransposedTile)
    {
    }

    LoadVGPR::LoadVGPR() = default;
    LoadVGPR::LoadVGPR(VariableType const varType, bool const scalar)
        : varType(varType)
        , scalar(scalar)
    {
    }

    LoadSGPR::LoadSGPR()
        : bufOpts{} {};
    LoadSGPR::LoadSGPR(VariableType const varType, BufferInstructionOptions const bio)
        : varType(varType)
        , bufOpts(bio)
    {
    }

    LoadLDSTile::LoadLDSTile() = default;
    LoadLDSTile::LoadLDSTile(VariableType const varType, bool const isTransposedTile)
        : varType(varType)
        , isTransposedTile(isTransposedTile)
    {
    }

    LoadTileDirect2LDS::LoadTileDirect2LDS() = default;
    LoadTileDirect2LDS::LoadTileDirect2LDS(rocRoller::VariableType const varType)
        : varType(varType)
    {
    }

    StoreTiled::StoreTiled() = default;
    StoreTiled::StoreTiled(VariableType const varType)
        : varType(varType)
    {
    }

    StoreSGPR::StoreSGPR()
        : bufOpts{} {};
    StoreSGPR::StoreSGPR(VariableType const varType, BufferInstructionOptions const bio)
        : varType(varType)
        , bufOpts(bio)
    {
    }

    StoreLDSTile::StoreLDSTile() = default;
    StoreLDSTile::StoreLDSTile(VariableType const varType)
        : varType(varType)
    {
    }

    SeedPRNG::SeedPRNG() = default;
    SeedPRNG::SeedPRNG(bool addTID)
        : addTID(addTID)
    {
    }

    TensorContraction::TensorContraction() = default;
    TensorContraction::TensorContraction(std::vector<int> const& aContractedDimensions,
                                         std::vector<int> const& bContractedDimensions)
        : aDims(aContractedDimensions)
        , bDims(bContractedDimensions)
    {
    }

    Multiply::Multiply()
        : scaleA(Operations::ScaleMode::None)
        , scaleB(Operations::ScaleMode::None)
    {
    }

    Multiply::Multiply(Operations::ScaleMode scaleA_, Operations::ScaleMode scaleB_)
        : scaleA(scaleA_)
        , scaleB(scaleB_)
    {
    }

    Exchange::Exchange() = default;
    Exchange::Exchange(rocRoller::VariableType const varType)
        : varType(varType)
    {
    }

    std::string LoadLDSTile::toString() const
    {
        return fmt::format("LoadLDSTile{{{}}}", rocRoller::toString(varType));
    }

    RR_CLASS_NAME_IMPL(SetCoordinate);
    RR_CLASS_NAME_IMPL(ConditionalOp);
    RR_CLASS_NAME_IMPL(AssertOp);
    RR_CLASS_NAME_IMPL(DoWhileOp);
    RR_CLASS_NAME_IMPL(Exchange);
    RR_CLASS_NAME_IMPL(ForLoopOp);
    RR_CLASS_NAME_IMPL(UnrollOp);
    RR_CLASS_NAME_IMPL(Assign);
    RR_CLASS_NAME_IMPL(ComputeIndex);
    RR_CLASS_NAME_IMPL(LoadLinear);
    RR_CLASS_NAME_IMPL(LoadTiled);
    RR_CLASS_NAME_IMPL(LoadVGPR);
    RR_CLASS_NAME_IMPL(LoadSGPR);
    RR_CLASS_NAME_IMPL(LoadLDSTile);
    RR_CLASS_NAME_IMPL(Multiply);
    RR_CLASS_NAME_IMPL(LoadTileDirect2LDS);
    RR_CLASS_NAME_IMPL(StoreTiled);
    RR_CLASS_NAME_IMPL(StoreSGPR);
    RR_CLASS_NAME_IMPL(StoreLDSTile);
    RR_CLASS_NAME_IMPL(SeedPRNG);
    RR_CLASS_NAME_IMPL(TensorContraction);

}
