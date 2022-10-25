/**
 *
 */
#pragma once

#include "Command_fwd.hpp"
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
            E_Unary(int dest, int a);
            E_Unary(const std::initializer_list<unsigned>&, const unsigned&);
            std::string             toString() const;
            int                     getTag() const;
            void                    setTag(int tag);
            std::unordered_set<int> getOutputs() const;
            std::unordered_set<int> getInputs() const;
            virtual std::string     name() const
            {
                return "";
            }
            int dest;
            int a;
        };

// Macro for declaring a new Unary XOp
#define MAKE_UNARY_XOP(FUNC)                                                                     \
    struct FUNC : public E_Unary                                                                 \
    {                                                                                            \
        FUNC(int dest, int a)                                                                    \
            : E_Unary(dest, a)                                                                   \
        {                                                                                        \
        }                                                                                        \
        FUNC(const std::initializer_list<unsigned>& args, const unsigned& dest, const unsigned&) \
            : E_Unary(args, dest)                                                                \
        {                                                                                        \
        }                                                                                        \
        std::string name() const                                                                 \
        {                                                                                        \
            return #FUNC;                                                                        \
        }                                                                                        \
    };

        MAKE_UNARY_XOP(E_Neg)
        MAKE_UNARY_XOP(E_Abs)
        MAKE_UNARY_XOP(E_Not)

        struct E_Binary
        {
            E_Binary(int dest, int a, int b);
            E_Binary(const std::initializer_list<unsigned>&, unsigned&);

            std::string             toString() const;
            int                     getTag() const;
            void                    setTag(int tag);
            std::unordered_set<int> getOutputs() const;
            std::unordered_set<int> getInputs() const;
            virtual std::string     name() const
            {
                return "";
            }
            int dest;
            int a;
            int b;
        };

// Macro for defining a new binary XOp
#define MAKE_BINARY_XOP(FUNC)                                                        \
    struct FUNC : public E_Binary                                                    \
    {                                                                                \
        FUNC(int dest, int a, int b)                                                 \
            : E_Binary(dest, a, b)                                                   \
        {                                                                            \
        }                                                                            \
        FUNC(const std::initializer_list<unsigned>& args, unsigned& dest, unsigned&) \
            : E_Binary(args, dest)                                                   \
        {                                                                            \
        }                                                                            \
        std::string name() const                                                     \
        {                                                                            \
            return #FUNC;                                                            \
        }                                                                            \
    };

        MAKE_BINARY_XOP(E_Add)
        MAKE_BINARY_XOP(E_Sub)
        MAKE_BINARY_XOP(E_Mul)
        MAKE_BINARY_XOP(E_Div)
        MAKE_BINARY_XOP(E_And)
        MAKE_BINARY_XOP(E_Or)

        template <typename T>
        concept CXOp = requires
        {
            std::constructible_from<XOp, T>;
        };

        class T_Execute
        {
        public:
            T_Execute();
            T_Execute(int starting_tag);
            void                    setCommand(std::shared_ptr<Command> command);
            std::unordered_set<int> getInputs() const;
            std::unordered_set<int> getOutputs() const;
            void                    addXOp(std::shared_ptr<XOp>);
            int                     getNextTag() const;
            std::string             toString() const;

            std::vector<std::shared_ptr<XOp>> getXOps() const
            {
                return m_xops;
            }

        private:
            std::weak_ptr<Command> m_command;

            std::vector<std::shared_ptr<XOp>> m_xops;

            std::unordered_set<int> m_inputs;
            std::unordered_set<int> m_outputs;

            int m_next_tag;
        };
    }
}

#include "T_Execute_impl.hpp"
