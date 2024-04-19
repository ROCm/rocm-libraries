/**
 *
 */
#pragma once

#include "Command_fwd.hpp"
#include "Operation.hpp"
#include "T_Execute_fwd.hpp"

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
            E_Unary(int a);
            E_Unary(const std::initializer_list<unsigned>&);
            std::string             toString() const;
            int                     getTag() const;
            void                    setTag(int tag);
            std::unordered_set<int> getOutputs() const;
            std::unordered_set<int> getInputs() const;
            virtual std::string     name() const
            {
                return "";
            }
            int dest = -1;
            int a;
        };

// Macro for declaring a new Unary XOp
#define MAKE_UNARY_XOP(FUNC)                              \
    struct FUNC : public E_Unary                          \
    {                                                     \
        FUNC(int a)                                       \
            : E_Unary(a)                                  \
        {                                                 \
        }                                                 \
        FUNC(const std::initializer_list<unsigned>& args) \
            : E_Unary(args)                               \
        {                                                 \
        }                                                 \
        std::string name() const                          \
        {                                                 \
            return #FUNC;                                 \
        }                                                 \
    };

        MAKE_UNARY_XOP(E_Neg)
        MAKE_UNARY_XOP(E_Abs)
        MAKE_UNARY_XOP(E_Not)

        struct E_Binary
        {
            E_Binary(int a, int b);
            E_Binary(const std::initializer_list<unsigned>&);

            std::string             toString() const;
            int                     getTag() const;
            void                    setTag(int tag);
            std::unordered_set<int> getOutputs() const;
            std::unordered_set<int> getInputs() const;
            virtual std::string     name() const
            {
                return "";
            }
            int dest = -1;
            int a;
            int b;
        };

// Macro for defining a new binary XOp
#define MAKE_BINARY_XOP(FUNC)                             \
    struct FUNC : public E_Binary                         \
    {                                                     \
        FUNC(int a, int b)                                \
            : E_Binary(a, b)                              \
        {                                                 \
        }                                                 \
        FUNC(const std::initializer_list<unsigned>& args) \
            : E_Binary(args)                              \
        {                                                 \
        }                                                 \
        std::string name() const                          \
        {                                                 \
            return #FUNC;                                 \
        }                                                 \
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
            E_Ternary(int a, int b, int c);
            E_Ternary(const std::initializer_list<unsigned>&);

            std::string             toString() const;
            int                     getTag() const;
            void                    setTag(int tag);
            std::unordered_set<int> getOutputs() const;
            std::unordered_set<int> getInputs() const;
            virtual std::string     name() const
            {
                return "";
            }
            int dest = -1;
            int a;
            int b;
            int c;
        };

// Macro for defining a new ternary XOp
#define MAKE_TERNARY_XOP(FUNC)                            \
    struct FUNC : public E_Ternary                        \
    {                                                     \
        FUNC(int a, int b, int c)                         \
            : E_Ternary(a, b, c)                          \
        {                                                 \
        }                                                 \
        FUNC(const std::initializer_list<unsigned>& args) \
            : E_Ternary(args)                             \
        {                                                 \
        }                                                 \
        std::string name() const                          \
        {                                                 \
            return #FUNC;                                 \
        }                                                 \
    };
        MAKE_TERNARY_XOP(E_Conditional)

        class T_Execute : public BaseOperation
        {
        public:
            T_Execute() = delete;
            T_Execute(int starting_tag);
            std::unordered_set<int> getInputs() const;
            std::unordered_set<int> getOutputs() const;
            int                     addXOp(std::shared_ptr<XOp>);
            template <CXOp T>
            int         addXOp(T&& op);
            int         getNextTag() const;
            std::string toString() const;

            std::vector<std::shared_ptr<XOp>> getXOps() const
            {
                return m_xops;
            }

        private:
            std::vector<std::shared_ptr<XOp>> m_xops;

            std::unordered_set<int> m_inputs;
            std::unordered_set<int> m_outputs;

            int m_nextTag;
        };
    }
}

#include "T_Execute_impl.hpp"
