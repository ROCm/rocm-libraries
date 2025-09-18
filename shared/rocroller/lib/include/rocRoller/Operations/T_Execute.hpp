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

/**
 *
 */
#pragma once

#include <memory>
#include <unordered_set>
#include <variant>
#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command_fwd.hpp>
#include <rocRoller/Operations/Operation.hpp>
#include <rocRoller/Operations/T_Execute_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {

        struct E_Unary
        {
            explicit E_Unary(OperationTag a);
            E_Unary(const std::initializer_list<OperationTag>&);
            std::string                      toString() const;
            OperationTag                     getTag() const;
            void                             setTag(OperationTag tag);
            std::unordered_set<OperationTag> getOutputs() const;
            std::unordered_set<OperationTag> getInputs() const;
            virtual std::string              name() const
            {
                return "";
            }
            OperationTag dest;
            OperationTag a;

            auto operator<=>(E_Unary const&) const = default;
        };

// Macro for declaring a new Unary XOp
#define MAKE_UNARY_XOP(FUNC)                                  \
    struct FUNC : public E_Unary                              \
    {                                                         \
        explicit FUNC(OperationTag a)                         \
            : E_Unary(a)                                      \
        {                                                     \
        }                                                     \
        FUNC(const std::initializer_list<OperationTag>& args) \
            : E_Unary(args)                                   \
        {                                                     \
        }                                                     \
        std::string name() const override                     \
        {                                                     \
            return #FUNC;                                     \
        }                                                     \
    };

        MAKE_UNARY_XOP(E_Neg)
        MAKE_UNARY_XOP(E_Abs)
        MAKE_UNARY_XOP(E_Not)
        MAKE_UNARY_XOP(E_RandomNumber)

        struct E_Cvt : public E_Unary
        {
            E_Cvt(OperationTag a, rocRoller::DataType destType)
                : E_Unary(a)
                , destType(destType)
            {
            }
            E_Cvt(const std::initializer_list<OperationTag>& args, rocRoller::DataType destType)
                : E_Unary(args)
                , destType(destType)
            {
            }
            std::string name() const override
            {
                return "E_Cvt";
            }

            rocRoller::DataType destType;
        };

        struct E_Binary
        {
            E_Binary(OperationTag a, OperationTag b);
            E_Binary(const std::initializer_list<OperationTag>&);

            std::string                      toString() const;
            OperationTag                     getTag() const;
            void                             setTag(OperationTag tag);
            std::unordered_set<OperationTag> getOutputs() const;
            std::unordered_set<OperationTag> getInputs() const;
            virtual std::string              name() const
            {
                return "";
            }
            OperationTag dest;
            OperationTag a;
            OperationTag b;

            auto operator<=>(E_Binary const&) const = default;
        };

// Macro for defining a new binary XOp
#define MAKE_BINARY_XOP(FUNC)                                 \
    struct FUNC : public E_Binary                             \
    {                                                         \
        FUNC(OperationTag a, OperationTag b)                  \
            : E_Binary(a, b)                                  \
        {                                                     \
        }                                                     \
        FUNC(const std::initializer_list<OperationTag>& args) \
            : E_Binary(args)                                  \
        {                                                     \
        }                                                     \
        std::string name() const override                     \
        {                                                     \
            return #FUNC;                                     \
        }                                                     \
    };

        struct E_StochasticRoundingCvt : public E_Binary
        {
            E_StochasticRoundingCvt(OperationTag        data,
                                    OperationTag        seed,
                                    rocRoller::DataType destType)
                : E_Binary(data, seed)
                , destType(destType)
            {
            }
            E_StochasticRoundingCvt(const std::initializer_list<OperationTag>& args,
                                    rocRoller::DataType                        destType)
                : E_Binary(args)
                , destType(destType)
            {
            }
            std::string name() const override
            {
                return "E_StochasticRoundingCvt";
            }

            rocRoller::DataType destType;
        };

        MAKE_BINARY_XOP(E_Add)
        MAKE_BINARY_XOP(E_Sub)
        MAKE_BINARY_XOP(E_Mul)
        MAKE_BINARY_XOP(E_Div)
        MAKE_BINARY_XOP(E_And)
        MAKE_BINARY_XOP(E_Or)
        MAKE_BINARY_XOP(E_GreaterThan)

        struct E_Ternary
        {
            E_Ternary(OperationTag a, OperationTag b, OperationTag c);
            E_Ternary(const std::initializer_list<OperationTag>&);

            std::string                      toString() const;
            OperationTag                     getTag() const;
            void                             setTag(OperationTag tag);
            std::unordered_set<OperationTag> getOutputs() const;
            std::unordered_set<OperationTag> getInputs() const;
            virtual std::string              name() const
            {
                return "";
            }
            OperationTag dest;
            OperationTag a;
            OperationTag b;
            OperationTag c;

            auto operator<=>(E_Ternary const&) const = default;
        };

// Macro for defining a new ternary XOp
#define MAKE_TERNARY_XOP(FUNC)                                \
    struct FUNC : public E_Ternary                            \
    {                                                         \
        FUNC(OperationTag a, OperationTag b, OperationTag c)  \
            : E_Ternary(a, b, c)                              \
        {                                                     \
        }                                                     \
        FUNC(const std::initializer_list<OperationTag>& args) \
            : E_Ternary(args)                                 \
        {                                                     \
        }                                                     \
        std::string name() const override                     \
        {                                                     \
            return #FUNC;                                     \
        }                                                     \
    };
        MAKE_TERNARY_XOP(E_Conditional)

        class T_Execute : public BaseOperation
        {
        public:
            T_Execute() = delete;
            explicit T_Execute(OperationTag starting_tag);
            std::unordered_set<OperationTag> getInputs() const;
            std::unordered_set<OperationTag> getOutputs() const;
            OperationTag                     addXOp(std::shared_ptr<XOp>);
            template <CXOp T>
            OperationTag addXOp(T&& op);
            OperationTag getNextTag() const;
            std::string  toString() const;

            std::vector<std::shared_ptr<XOp>> getXOps() const
            {
                return m_xops;
            }

            bool operator==(T_Execute const&) const;

        private:
            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;

            std::vector<std::shared_ptr<XOp>> m_xops;

            std::unordered_set<OperationTag> m_inputs;
            std::unordered_set<OperationTag> m_outputs;

            OperationTag m_nextTag;
        };

        template <typename T>
        concept CUnaryXOp = requires()
        {
            requires std::derived_from<T, E_Unary>;
        };

        template <typename T>
        concept CBinaryXOp = requires()
        {
            requires std::derived_from<T, E_Binary>;
        };

        template <typename T>
        concept CTernaryXOp = requires()
        {
            requires std::derived_from<T, E_Ternary>;
        };

    }
}

#include <rocRoller/Operations/T_Execute_impl.hpp>
