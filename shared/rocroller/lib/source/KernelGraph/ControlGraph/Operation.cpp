// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Expression.hpp>

namespace rocRoller::KernelGraph::ControlGraph
{
    static_assert(!COperationWithBody<Assign>);
    static_assert(!COperationWithBody<Multiply>);
    static_assert(COperationWithBody<ForLoopOp>);
    static_assert(COperationWithBody<Kernel>);

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

    LoadLinear::LoadLinear() = default;
    LoadLinear::LoadLinear(rocRoller::VariableType const varType)
        : varType(varType)
    {
    }

    LoadTiled::LoadTiled() = default;
    LoadTiled::LoadTiled(rocRoller::VariableType const varType)
        : varType(varType)
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

    LoadTiledTDMToLDS::LoadTiledTDMToLDS() = default;
    LoadTiledTDMToLDS::LoadTiledTDMToLDS(rocRoller::VariableType const varType)
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
                                         std::vector<int> const& bContractedDimensions,
                                         VariableType const      accType)
        : aDims(aContractedDimensions)
        , bDims(bContractedDimensions)
        , accType(accType)
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

    std::string Deallocate::toString() const
    {
        std::ostringstream msg;
        msg << "Deallocate{";
        streamJoin(msg, arguments, ", ");
        msg << "}";
        return msg.str();
    }

    RR_CLASS_NAME_IMPL(AssertOp);
    RR_CLASS_NAME_IMPL(Assign);
    RR_CLASS_NAME_IMPL(ConditionalOp);
    RR_CLASS_NAME_IMPL(Deallocate);
    RR_CLASS_NAME_IMPL(DoWhileOp);
    RR_CLASS_NAME_IMPL(Exchange);
    RR_CLASS_NAME_IMPL(ForLoopOp);
    RR_CLASS_NAME_IMPL(LoadLDSTile);
    RR_CLASS_NAME_IMPL(LoadLinear);
    RR_CLASS_NAME_IMPL(LoadSGPR);
    RR_CLASS_NAME_IMPL(LoadTileDirect2LDS);
    RR_CLASS_NAME_IMPL(LoadTiledTDMToLDS);
    RR_CLASS_NAME_IMPL(LoadTiled);
    RR_CLASS_NAME_IMPL(LoadVGPR);
    RR_CLASS_NAME_IMPL(Multiply);
    RR_CLASS_NAME_IMPL(SeedPRNG);
    RR_CLASS_NAME_IMPL(SetCoordinate);
    RR_CLASS_NAME_IMPL(StoreLDSTile);
    RR_CLASS_NAME_IMPL(StoreSGPR);
    RR_CLASS_NAME_IMPL(StoreTiled);
    RR_CLASS_NAME_IMPL(TensorContraction);
    RR_CLASS_NAME_IMPL(UnrollOp);

}
