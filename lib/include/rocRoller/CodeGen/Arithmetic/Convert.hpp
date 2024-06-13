#pragma once

#include "ArithmeticGenerator.hpp"

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
#define SpecializeGetGeneratorConvert(dtype)                                        \
    template <>                                                                     \
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::dtype>>> \
        GetGenerator<Expression::Convert<DataType::dtype>>(Register::ValuePtr dst,  \
                                                           Register::ValuePtr arg)

    SpecializeGetGeneratorConvert(Double);
    SpecializeGetGeneratorConvert(Float);
    SpecializeGetGeneratorConvert(Half);
    SpecializeGetGeneratorConvert(Halfx2);
    SpecializeGetGeneratorConvert(FP8x4);
    SpecializeGetGeneratorConvert(BF8x4);
    SpecializeGetGeneratorConvert(FP6x16);
    SpecializeGetGeneratorConvert(Int32);
    SpecializeGetGeneratorConvert(Int64);
    SpecializeGetGeneratorConvert(UInt32);
    SpecializeGetGeneratorConvert(UInt64);

#undef SpecializeGetGeneratorConvert

    /**
     * @brief Generates instructions to convert register value to a new datatype.
     *
     * @param dataType The new datatype
     * @param dest The destination register
     * @param arg The value to be converted
     * @return Generator<Instruction>
     */
    Generator<Instruction>
        generateConvertOp(DataType dataType, Register::ValuePtr dest, Register::ValuePtr arg);

    // Templated Generator class based on the return type.
    template <DataType DATATYPE>
    class ConvertGenerator : public UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>
    {
    public:
        ConvertGenerator(ContextPtr c)
            : UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool
            Match(typename UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>::Argument const&)
        {

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<
            typename UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>::Base>
            Build(typename UnaryArithmeticGenerator<Expression::Convert<DATATYPE>>::Argument const&
                      arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<ConvertGenerator<DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst, Register::ValuePtr arg) override;

        inline bool isIdentity(Register::ValuePtr arg) const override
        {
            return arg->variableType() == DATATYPE;
        }

        inline static const std::string Name;
    };
}
