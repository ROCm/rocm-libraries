/**
 *
 */
#pragma once

#include "Command_fwd.hpp"
#include "Operation.hpp"
#include "T_Execute_fwd.hpp"

#include "rocRoller/DataTypes/DataTypes.hpp"

#include <memory>
#include <unordered_set>
#include <variant>
#include <vector>

namespace rocRoller
{
    namespace Operations
    {

        struct E_Unary
        {
            E_Unary(OperationTag a);
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
        };

// Macro for declaring a new Unary XOp
#define MAKE_UNARY_XOP(FUNC)                                  \
    struct FUNC : public E_Unary                              \
    {                                                         \
        FUNC(OperationTag a)                                  \
            : E_Unary(a)                                      \
        {                                                     \
        }                                                     \
        FUNC(const std::initializer_list<OperationTag>& args) \
            : E_Unary(args)                                   \
        {                                                     \
        }                                                     \
        std::string name() const                              \
        {                                                     \
            return #FUNC;                                     \
        }                                                     \
    };

        MAKE_UNARY_XOP(E_Neg)
        MAKE_UNARY_XOP(E_Abs)
        MAKE_UNARY_XOP(E_Not)

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
            std::string name() const
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
        std::string name() const                              \
        {                                                     \
            return #FUNC;                                     \
        }                                                     \
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
        std::string name() const                              \
        {                                                     \
            return #FUNC;                                     \
        }                                                     \
    };
        MAKE_TERNARY_XOP(E_Conditional)

        class T_Execute : public BaseOperation
        {
        public:
            T_Execute() = delete;
            T_Execute(OperationTag starting_tag);
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

        private:
            std::vector<std::shared_ptr<XOp>> m_xops;

            std::unordered_set<OperationTag> m_inputs;
            std::unordered_set<OperationTag> m_outputs;

            OperationTag m_nextTag;
        };
    }
}

#include "T_Execute_impl.hpp"
