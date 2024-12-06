#pragma once

#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>

namespace rocRoller
{

    // GetGenerator function will return the Generator to use based on the provided arguments.
#define SpecializeGetGeneratorConvert(dtype)                                        \
    template <>                                                                     \
    std::shared_ptr<UnaryArithmeticGenerator<Expression::Convert<DataType::dtype>>> \
        GetGenerator<Expression::Convert<DataType::dtype>>(                         \
            Register::ValuePtr dst,                                                 \
            Register::ValuePtr arg,                                                 \
            Expression::Convert<DataType::dtype> const&)

    SpecializeGetGeneratorConvert(Double);
    SpecializeGetGeneratorConvert(Float);
    SpecializeGetGeneratorConvert(Half);
    SpecializeGetGeneratorConvert(Halfx2);
    SpecializeGetGeneratorConvert(BFloat16);
    SpecializeGetGeneratorConvert(BFloat16x2);
    SpecializeGetGeneratorConvert(FP8);
    SpecializeGetGeneratorConvert(BF8);
    SpecializeGetGeneratorConvert(FP8x4);
    SpecializeGetGeneratorConvert(BF8x4);
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
        Generator<Instruction> generate(Register::ValuePtr dst,
                                        Register::ValuePtr arg,
                                        Expression::Convert<DATATYPE> const&) override;

        inline static const std::string Name;
    };

    /**
     *  Below are for Stochastic Rounding (SR) conversion.
     *  SR conversion takes two args: the first arg is the value to be converted, and
     *  the second arg is a seed for stochastic rounding.
     */
    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SRConvert<DataType::FP8>>>
        GetGenerator<Expression::SRConvert<DataType::FP8>>(
            Register::ValuePtr dst,
            Register::ValuePtr lhs,
            Register::ValuePtr rhs,
            Expression::SRConvert<DataType::FP8> const&);

    template <>
    std::shared_ptr<BinaryArithmeticGenerator<Expression::SRConvert<DataType::BF8>>>
        GetGenerator<Expression::SRConvert<DataType::BF8>>(
            Register::ValuePtr dst,
            Register::ValuePtr lhs,
            Register::ValuePtr rhs,
            Expression::SRConvert<DataType::BF8> const&);

    // Templated Generator class based on the return type.
    template <DataType DATATYPE>
    class SRConvertGenerator : public BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>
    {
    public:
        SRConvertGenerator(ContextPtr c)
            : BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>(c)
        {
        }

        // Match function required by Component system for selecting the correct
        // generator.
        static bool Match(
            typename BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>::Argument const&)
        {

            return true;
        }

        // Build function required by Component system to return the generator.
        static std::shared_ptr<
            typename BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>::Base>
            Build(
                typename BinaryArithmeticGenerator<Expression::SRConvert<DATATYPE>>::Argument const&
                    arg)
        {
            if(!Match(arg))
                return nullptr;

            return std::make_shared<SRConvertGenerator<DATATYPE>>(std::get<0>(arg));
        }

        // Method to generate instructions
        Generator<Instruction> generate(Register::ValuePtr dst,
                                        Register::ValuePtr lhs,
                                        Register::ValuePtr rhs,
                                        Expression::SRConvert<DATATYPE> const&) override;

        inline static const std::string Name;
    };

}
