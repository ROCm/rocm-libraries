/**
 * @copyright Copyright 2022 Advanced Micro Devices, Inc.
 */

#include <rocRoller/CodeGen/Arithmetic.hpp>

namespace rocRoller
{
    RegisterComponentBase(Arithmetic);

    std::shared_ptr<Arithmetic> Arithmetic::Get(Register::ValuePtr reg)
    {
        return Component::Get<Arithmetic>(reg->context(), reg->regType(), reg->variableType());
    }

    std::shared_ptr<Arithmetic> Arithmetic::Get(Register::ValuePtr dst, Register::ValuePtr src)
    {
        // So far this is only used for negate, which makes no sense for pointer types.
        if(dst->variableType().isPointer() || src->variableType().isPointer())
            Throw<FatalError>("Implement this function for pointer types!");

        auto context = dst->context();
        if(!context)
            context = src->context();

        auto regType = Register::PromoteType(dst->regType(), src->regType());
        auto varType = VariableType::Promote(dst->variableType(), src->variableType());

        return Component::Get<Arithmetic>(context, regType, varType);
    }

    std::shared_ptr<Arithmetic>
        Arithmetic::Get(Register::ValuePtr dst, Register::ValuePtr lhs, Register::ValuePtr rhs)
    {
        auto context = dst->context();
        if(!context)
            context = lhs->context();
        if(!context)
            context = rhs->context();

        auto regType = Register::PromoteType(lhs->regType(), rhs->regType());
        AssertFatal(regType == dst->regType(),
                    "Register type mismatch: ",
                    ShowValue(dst->regType()),
                    ShowValue(lhs->regType()),
                    ShowValue(rhs->regType()));

        auto varType = VariableType::Promote(lhs->variableType(), rhs->variableType());

        auto dstVarType = dst->variableType();
        if(varType != dstVarType)
        {
            auto const& varTypeInfo    = DataTypeInfo::Get(varType);
            auto const& dstVarTypeInfo = DataTypeInfo::Get(dstVarType);

            AssertFatal(varTypeInfo.elementSize <= dstVarTypeInfo.elementSize
                            && varTypeInfo.isIntegral == dstVarTypeInfo.isIntegral,
                        ShowValue(varType),
                        ShowValue(dstVarType));

            varType = dstVarType;
        }

        return Component::Get<Arithmetic>(context, regType, varType);
    }

    std::shared_ptr<Arithmetic> Arithmetic::GetComparison(Register::ValuePtr dst,
                                                          Register::ValuePtr lhs,
                                                          Register::ValuePtr rhs)
    {
        auto context = dst->context();
        if(!context)
            context = lhs->context();
        if(!context)
            context = rhs->context();

        auto regType = Register::PromoteType(lhs->regType(), rhs->regType());
        auto varType = VariableType::Promote(lhs->variableType(), rhs->variableType());

        // Pay no attention to dst

        return Component::Get<Arithmetic>(context, regType, varType);
    }

    std::shared_ptr<Arithmetic> Arithmetic::Get(Register::ValuePtr dst,
                                                Register::ValuePtr lhs,
                                                Register::ValuePtr r1hs,
                                                Register::ValuePtr r2hs)
    {
        auto context = dst->context();
        if(!context)
            context = lhs->context();
        if(!context)
            context = r1hs->context();
        if(!context)
            context = r2hs->context();

        auto regType = Register::PromoteType(lhs->regType(), r1hs->regType());
        regType      = Register::PromoteType(regType, r2hs->regType());
        AssertFatal(regType == dst->regType(),
                    "Register type mismatch: ",
                    ShowValue(dst->regType()),
                    ShowValue(lhs->regType()),
                    ShowValue(r1hs->regType()),
                    ShowValue(r2hs->regType()));

        auto varType = VariableType::Promote(lhs->variableType(), r1hs->variableType());
        varType      = VariableType::Promote(varType, r2hs->variableType());

        auto dstVarType = dst->variableType();
        if(varType != dstVarType)
        {
            auto const& varTypeInfo    = DataTypeInfo::Get(varType);
            auto const& dstVarTypeInfo = DataTypeInfo::Get(dstVarType);

            AssertFatal(varTypeInfo.elementSize <= dstVarTypeInfo.elementSize
                            && varTypeInfo.isIntegral == dstVarTypeInfo.isIntegral,
                        ShowValue(varType),
                        ShowValue(dstVarType));

            varType = dstVarType;
        }

        return Component::Get<Arithmetic>(context, regType, varType);
    }

    Arithmetic::Arithmetic(std::shared_ptr<Context> context)
        : m_context(context)
    {
    }

    Arithmetic::~Arithmetic() = default;
}
