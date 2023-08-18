
#include <rocRoller/KernelGraph/ControlGraph/Operation.hpp>

namespace rocRoller::KernelGraph::ControlGraph
{
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

    LoadLDSTile::LoadLDSTile() = default;
    LoadLDSTile::LoadLDSTile(VariableType const varType)
        : varType(varType)
    {
    }

    StoreTiled::StoreTiled() = default;
    StoreTiled::StoreTiled(DataType const dtype)
        : dataType(dtype)
    {
    }

    StoreLDSTile::StoreLDSTile() = default;
    StoreLDSTile::StoreLDSTile(DataType const dtype)
        : dataType(dtype)
    {
    }

    TensorContraction::TensorContraction() = default;
    TensorContraction::TensorContraction(std::vector<int> const& aContractedDimensions,
                                         std::vector<int> const& bContractedDimensions)
        : aDims(aContractedDimensions)
        , bDims(bContractedDimensions)
    {
    }

    RR_CLASS_NAME_IMPL(SetCoordinate);
    RR_CLASS_NAME_IMPL(ConditionalOp);
    RR_CLASS_NAME_IMPL(DoWhileOp);
    RR_CLASS_NAME_IMPL(ForLoopOp);
    RR_CLASS_NAME_IMPL(UnrollOp);
    RR_CLASS_NAME_IMPL(Assign);
    RR_CLASS_NAME_IMPL(ComputeIndex);
    RR_CLASS_NAME_IMPL(LoadLinear);
    RR_CLASS_NAME_IMPL(LoadTiled);
    RR_CLASS_NAME_IMPL(LoadVGPR);
    RR_CLASS_NAME_IMPL(LoadLDSTile);
    RR_CLASS_NAME_IMPL(StoreTiled);
    RR_CLASS_NAME_IMPL(StoreLDSTile);
    RR_CLASS_NAME_IMPL(TensorContraction);

}
