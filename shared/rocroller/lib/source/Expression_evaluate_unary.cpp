// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>

#include <rocRoller/Expression.hpp>
#include <rocRoller/Expression_evaluate_detail.hpp>

#include <rocRoller/Utilities/RTTI.hpp>
#include <rocRoller/Utilities/Random.hpp>

#include <libdivide.h>

namespace rocRoller::Expression::EvaluateDetail
{
    template <CUnary UnaryExpr, DataType DESTTYPE = DataType::None>
    struct UnaryEvaluatorVisitor
    {
        UnaryExpr expr;

        using TheEvaluator = OperationEvaluatorVisitor<UnaryExpr, DESTTYPE>;

        template <CCommandArgumentValue ARG>
        requires CCanEvaluateUnary<TheEvaluator, ARG>
            CommandArgumentValue operator()(ARG const& arg) const
        {
            auto evaluator = static_cast<TheEvaluator const*>(this);
            return evaluator->evaluate(arg);
        }

        template <CCommandArgumentValue ARG>
        requires(!CCanEvaluateUnary<TheEvaluator, ARG>) CommandArgumentValue
            operator()(ARG const& arg) const
        {
            if constexpr(CHasTypeInfo<ARG>)
            {
                Throw<FatalError>("Incompatible type ",
                                  TypeInfo<ARG>::Name(),
                                  " for expression ",
                                  typeName<UnaryExpr>());
            }
            else
            {
                Throw<FatalError>("Incompatible type ",
                                  ShowValue(ARG()),
                                  typeName<ARG>(),
                                  " for expression ",
                                  typeName<UnaryExpr>());
            }
        }

        CommandArgumentValue call(CommandArgumentValue const& arg) const
        {
            return std::visit(*this, arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MagicMultiple> : public UnaryEvaluatorVisitor<MagicMultiple>
    {
        uint32_t evaluate(uint32_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return std::numeric_limits<uint32_t>::max() / 2;

            // When the divisor is 1, this constant has no use, so return 0.
            if(arg == 1)
                return 0;

            auto magic = libdivide::libdivide_u32_branchfree_gen(arg);

            return magic.magic;
        }

        int32_t evaluate(int32_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return std::numeric_limits<int32_t>::max() / 2;

            auto magic = libdivide::libdivide_s32_branchfree_gen(arg);

            return magic.magic;
        }

        int64_t evaluate(int64_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return std::numeric_limits<int64_t>::max() / 2;

            auto magic = libdivide::libdivide_s64_branchfree_gen(arg);

            return magic.magic;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Convert>
    {
        Convert exp;

        template <int Idx = 0>
        CommandArgumentValue call(CommandArgumentValue const& arg)
        {
            constexpr auto IdxType = static_cast<DataType>(Idx);

            if constexpr(IdxType == DataType::None)
            {
                Throw<FatalError>("Cannot convert to None type.");
                return 0;
            }
            else if constexpr(IdxType == DataType::Count)
            {
                Throw<FatalError>("Cannot convert to Count type.");
                return 0;
            }
            else
            {
                using ResultType = typename EnumTypeInfo<IdxType>::Type;

                if(exp.destinationType == IdxType)
                {
                    if constexpr(!CCommandArgumentValue<ResultType>)
                    {
                        Throw<FatalError>("Cannot convert to ", friendlyTypeName<ResultType>());
                        return 0;
                    }
                    else
                    {
                        auto theVisitor = [](auto value) -> ResultType {
                            using ExistingType = std::decay_t<decltype(value)>;
                            if constexpr(CCanStaticCastTo<ResultType, ExistingType>)
                            {
                                return static_cast<ResultType>(value);
                            }
                            else
                            {
                                Throw<FatalError>("Cannot cast ",
                                                  friendlyTypeName<ExistingType>(),
                                                  " to ",
                                                  friendlyTypeName<ResultType>());

                                return ResultType{};
                            }
                        };

                        return std::visit(theVisitor, arg);
                    }
                }
                else
                {
                    return call<Idx + 1>(arg);
                }
            }
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Reinterpret>
    {
        Reinterpret exp;

        template <CCommandArgumentValue FromType, int Idx = 0>
        CommandArgumentValue reinterpret(FromType const& value, DataType targetDataType)
        {
            constexpr auto IdxType = static_cast<DataType>(Idx);

            if constexpr(IdxType == DataType::None || IdxType == DataType::Count)
            {
                Throw<FatalError>("Unsupported reinterpret to type: ", toString(targetDataType));
                return 0;
            }
            else
            {
                using ToType = typename EnumTypeInfo<IdxType>::Type;

                if(targetDataType == IdxType)
                {
                    AssertFatal(std::is_trivially_copyable_v<FromType>,
                                "FromType must be trivially copyable");
                    AssertFatal(std::is_trivially_copyable_v<ToType>,
                                "ToType must be trivially copyable");

                    if constexpr(!CCommandArgumentValue<
                                     ToType> || sizeof(ToType) != sizeof(FromType))
                    {
                        Throw<FatalError>("Cannot reinterpret to ",
                                          friendlyTypeName<ToType>(),
                                          " from ",
                                          friendlyTypeName<FromType>());
                        return 0;
                    }
                    else
                    {
                        return std::bit_cast<ToType>(value);
                    }
                }
                else
                {
                    return reinterpret<FromType, Idx + 1>(value, targetDataType);
                }
            }
        }

        CommandArgumentValue call(CommandArgumentValue const& arg)
        {
            return std::visit(
                [this](auto const& val) -> CommandArgumentValue {
                    using FromType = std::decay_t<decltype(val)>;
                    return reinterpret<FromType, 0>(val, exp.destinationType);
                },
                arg);
        }
    };

    template <CCommandArgumentValue FromType, int Idx = 0>
    CommandArgumentValue reinterpretTruncateValue(FromType const& value,
                                                  DataType        targetDataType,
                                                  std::endian     endianness = std::endian::native)
    {
        constexpr auto IdxType = static_cast<DataType>(Idx);

        AssertFatal(
            endianness == std::endian::little || endianness == std::endian::big,
            "Unsupported or mixed endianness: only pure little- or big-endian are supported.");

        if constexpr(IdxType == DataType::None || IdxType == DataType::Count)
        {
            Throw<FatalError>("Unsupported reinterpretTruncate to type: ",
                              toString(targetDataType));
            return 0;
        }
        else
        {
            using ToType = typename EnumTypeInfo<IdxType>::Type;

            AssertFatal(std::is_trivially_copyable_v<FromType>,
                        "FromType must be trivially copyable");
            AssertFatal(std::is_trivially_copyable_v<ToType>, "ToType must be trivially copyable");

            if(targetDataType == IdxType)
            {
                if constexpr(!CCommandArgumentValue<ToType>)
                {
                    Throw<FatalError>("Cannot reinterpret to ", friendlyTypeName<ToType>());
                    return 0;
                }
                else
                {
                    if constexpr(sizeof(ToType) == sizeof(FromType))
                    {
                        return std::bit_cast<ToType>(value);
                    }
                    // Truncate
                    else if constexpr(sizeof(ToType) < sizeof(FromType))
                    {
                        constexpr std::size_t N = sizeof(ToType);

                        // Source and destination as bytes
                        const auto src_bytes
                            = std::bit_cast<std::array<std::byte, sizeof(FromType)>>(value);
                        std::array<std::byte, sizeof(ToType)> dst_bytes{};
                        dst_bytes.fill(std::byte{0});

                        if(endianness == std::endian::little)
                        {
                            // Keep least-significant bytes: at low addresses.
                            std::copy_n(src_bytes.data(), N, dst_bytes.data());
                        }
                        else
                        {
                            // Big-endian: least-significant bytes live at high addresses.
                            std::copy_n(
                                src_bytes.data() + (sizeof(FromType) - N), N, dst_bytes.data());
                        }

                        return std::bit_cast<ToType>(dst_bytes);
                    }
                    else
                    {
                        Throw<FatalError>("Cannot truncate to wider type ",
                                          friendlyTypeName<ToType>(),
                                          " from ",
                                          friendlyTypeName<FromType>());
                        return 0;
                    }
                }
            }
            else
            {
                return reinterpretTruncateValue<FromType, Idx + 1>(
                    value, targetDataType, endianness);
            }
        }
    }

    CommandArgumentValue reinterpretTruncateValue(CommandArgumentValue const& value,
                                                  DataType                    targetDataType,
                                                  std::endian                 endianness)
    {
        return std::visit(
            [targetDataType, endianness](auto const& val) -> CommandArgumentValue {
                using FromType = std::decay_t<decltype(val)>;
                return reinterpretTruncateValue<FromType>(val, targetDataType, endianness);
            },
            value);
    }

    template <>
    struct OperationEvaluatorVisitor<BitFieldExtract>
    {
        BitFieldExtract expr;

        template <CCommandArgumentValue ARG>
        requires(CIntegral<ARG>) CommandArgumentValue operator()(ARG const& arg) const
        {
            auto argBits = resultVariableType(arg).getElementSize() * 8u;
            AssertFatal(argBits >= expr.offset + expr.width,
                        "BitFieldExtract out of bounds: offset={} + width={} > argBits={}",
                        expr.offset,
                        expr.width,
                        argBits);

            using UnsignedARG       = std::make_unsigned_t<ARG>;
            UnsignedARG unsignedArg = static_cast<UnsignedARG>(arg);

            UnsignedARG mask = 0;
            if(argBits == this->expr.width)
            {
                mask = std::numeric_limits<UnsignedARG>::max();
            }
            else
            {
                mask = (static_cast<UnsignedARG>(1) << this->expr.width) - 1;
            }

            UnsignedARG result = (unsignedArg >> this->expr.offset) & mask;

            // Sign extend if needed
            if constexpr(std::is_signed_v<ARG>)
            {
                bool signBit = (result >> (this->expr.width - 1)) & 1;
                if(signBit)
                {
                    UnsignedARG signExtensionMask = ~mask;
                    result |= signExtensionMask;
                }
            }

            AssertFatal(expr.width <= DataTypeInfo::Get(expr.outputDataType).elementBits,
                        fmt::format("BitFieldExtract: width {} exceeds output type size {} bits",
                                    expr.width,
                                    DataTypeInfo::Get(expr.outputDataType).elementBits));

            return reinterpretTruncateValue(static_cast<ARG>(result), expr.outputDataType);
        }

        CommandArgumentValue operator()(Raw32 const& arg) const
        {
            return call(arg.value);
        }

        template <typename ARG>
        CommandArgumentValue operator()(ARG const&) const
        {
            Throw<FatalError>("BitFieldExtract: unsupported argument type ", ShowValue(expr));
        }

        CommandArgumentValue call(CommandArgumentValue const& arg) const
        {
            return std::visit(*this, arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MagicShifts> : public UnaryEvaluatorVisitor<MagicShifts>
    {
        int evaluate(uint32_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return 0;

            // When the divisor is 1, we set the MSB of MagicShift to 1 so we can detect this case
            // by checking the MSB.
            if(arg == 1)
                return 1 << 31;

            auto magic = libdivide::libdivide_u32_branchfree_gen(arg);

            return magic.more & libdivide::LIBDIVIDE_32_SHIFT_MASK;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<MagicShiftAndSign>
        : public UnaryEvaluatorVisitor<MagicShiftAndSign>
    {
        static_assert(libdivide::LIBDIVIDE_32_SHIFT_MASK == 31,
                      "magicNumberDivision assumes this is true.");
        static_assert(libdivide::LIBDIVIDE_64_SHIFT_MASK == 63,
                      "magicNumberDivision assumes this is true.");
        static_assert(libdivide::LIBDIVIDE_NEGATIVE_DIVISOR == 1 << 7,
                      "magicNumberDivision assumes this is true.");

        uint32_t evaluate(int32_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return 0;

            auto magic = libdivide::libdivide_s32_branchfree_gen(arg);

            return static_cast<uint32_t>(magic.more);
        }

        uint32_t evaluate(int64_t const& arg) const
        {
            assertNonNullPointer(arg);

            if(arg == 0)
                return 0;

            auto magic = libdivide::libdivide_s64_branchfree_gen(arg);

            return static_cast<uint32_t>(magic.more);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<BitwiseNegate> : public UnaryEvaluatorVisitor<BitwiseNegate>
    {
        template <typename T>
        requires(CIntegral<T> or std::is_same_v<T, Raw32>) constexpr T evaluate(T const& arg) const
        {
            return ~arg;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Exponential2> : public UnaryEvaluatorVisitor<Exponential2>
    {
        template <isFP32 T>
        constexpr T evaluate(T const& arg) const
        {
            return std::exp2(arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Exponential> : public UnaryEvaluatorVisitor<Exponential>
    {
        template <isFP32 T>
        constexpr T evaluate(T const& arg) const
        {
            return std::exp(arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<Negate> : public UnaryEvaluatorVisitor<Negate>
    {
        template <typename T>
        requires(std::floating_point<T> || std::signed_integral<T>) constexpr T
            evaluate(T const& arg) const
        {
            return -arg;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<LogicalNot> : public UnaryEvaluatorVisitor<LogicalNot>
    {
        template <typename T>
        constexpr bool evaluate(T const& arg) const
        {
            return !arg;
        }
    };

    template <>
    struct OperationEvaluatorVisitor<RandomNumber> : public UnaryEvaluatorVisitor<RandomNumber>
    {
        template <typename T>
        requires(!std::same_as<bool, T> && std::unsigned_integral<T>) constexpr T
            evaluate(T const& arg) const
        {
            return LFSRRandomNumberGenerator(arg);
        }
    };

    template <>
    struct OperationEvaluatorVisitor<ToScalar> : public UnaryEvaluatorVisitor<ToScalar>
    {
        template <typename T>
        constexpr T evaluate(T arg) const
        {
            return arg;
        }
    };

    struct ToBoolVisitor
    {
        template <typename T>
        constexpr bool operator()(T const& val)
        {
            Throw<FatalError>("Invalid bool type: ", typeName<T>());
            return 0;
        }

        template <std::integral T>
        constexpr bool operator()(T const& val)
        {
            return val != 0;
        }

        bool call(CommandArgumentValue const& val)
        {
            return std::visit(*this, val);
        }
    };

    template <CUnary T>
    __attribute__((noinline)) CommandArgumentValue evaluateOp(T const&                    op,
                                                              CommandArgumentValue const& arg)
    {
        OperationEvaluatorVisitor<T> visitor{op};
        return visitor.call(arg);
    }

#define INSTANTIATE_UNARY_OP(Op) \
    template CommandArgumentValue evaluateOp<Op>(Op const& op, CommandArgumentValue const& arg)

    INSTANTIATE_UNARY_OP(MagicMultiple);
    INSTANTIATE_UNARY_OP(MagicShifts);
    INSTANTIATE_UNARY_OP(MagicShiftAndSign);
    INSTANTIATE_UNARY_OP(Negate);
    INSTANTIATE_UNARY_OP(BitwiseNegate);
    INSTANTIATE_UNARY_OP(LogicalNot);
    INSTANTIATE_UNARY_OP(Exponential2);
    INSTANTIATE_UNARY_OP(Exponential);
    INSTANTIATE_UNARY_OP(RandomNumber);
    INSTANTIATE_UNARY_OP(ToScalar);
    INSTANTIATE_UNARY_OP(BitFieldExtract);
    INSTANTIATE_UNARY_OP(Convert);
    INSTANTIATE_UNARY_OP(Reinterpret);

}
