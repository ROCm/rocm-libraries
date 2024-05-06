#pragma once

#include "Command.hpp"

namespace rocRoller
{
    namespace Operations
    {
        // ------------------------
        // XOp methods
        // ------------------------

        inline E_Unary::E_Unary(OperationTag a)
            : a(a)
        {
        }

        inline E_Unary::E_Unary(const std::initializer_list<OperationTag>& args)
            : a(*args.begin())
        {
            AssertFatal(args.size() == 1, ShowValue(args.size()));
        }

        inline std::string E_Unary::toString() const
        {
            std::ostringstream msg;

            msg << name() << " " << dest << ", " << a;

            return msg.str();
        }

        inline OperationTag E_Unary::getTag() const
        {
            return dest;
        }

        inline void E_Unary::setTag(OperationTag tag)
        {
            dest = tag;
        }

        inline std::unordered_set<OperationTag> E_Unary::getOutputs() const
        {
            return {dest};
        }

        inline std::unordered_set<OperationTag> E_Unary::getInputs() const
        {
            return {a};
        }

        inline E_Binary::E_Binary(OperationTag a, OperationTag b)
            : a(a)
            , b(b)
        {
        }

        inline E_Binary::E_Binary(const std::initializer_list<OperationTag>& args)
            : a(*args.begin())
            , b(*(args.begin() + 1))
        {
            AssertFatal(args.size() == 2, ShowValue(args.size()));
        }

        inline std::string E_Binary::toString() const
        {
            std::ostringstream msg;

            msg << name() << " " << dest << ", " << a << ", " << b;

            return msg.str();
        }

        inline OperationTag E_Binary::getTag() const
        {
            return dest;
        }

        inline void E_Binary::setTag(OperationTag tag)
        {
            dest = tag;
        }

        inline std::unordered_set<OperationTag> E_Binary::getOutputs() const
        {
            return {dest};
        }

        inline std::unordered_set<OperationTag> E_Binary::getInputs() const
        {
            return {a, b};
        }

        inline E_Ternary::E_Ternary(OperationTag a, OperationTag b, OperationTag c)
            : a(a)
            , b(b)
            , c(c)
        {
        }

        inline E_Ternary::E_Ternary(const std::initializer_list<OperationTag>& args)
            : a(*args.begin())
            , b(*(args.begin() + 1))
            , c(*(args.begin() + 2))
        {
            AssertFatal(args.size() == 3, ShowValue(args.size()));
        }

        inline std::string E_Ternary::toString() const
        {
            std::ostringstream msg;

            msg << name() << " " << dest << ", " << a << ", " << b;

            return msg.str();
        }

        inline OperationTag E_Ternary::getTag() const
        {
            return dest;
        }

        inline void E_Ternary::setTag(OperationTag tag)
        {
            dest = tag;
        }

        inline std::unordered_set<OperationTag> E_Ternary::getOutputs() const
        {
            return {dest};
        }

        inline std::unordered_set<OperationTag> E_Ternary::getInputs() const
        {
            return {a, b, c};
        }

        // ------------------------
        // T_Execute methods
        // ------------------------

        inline T_Execute::T_Execute(OperationTag starting_tag)
            : BaseOperation()
            , m_nextTag(starting_tag)
        {
        }

        inline std::unordered_set<OperationTag> T_Execute::getInputs() const
        {
            return m_inputs;
        }

        inline std::unordered_set<OperationTag> T_Execute::getOutputs() const
        {
            return m_outputs;
        }

        inline OperationTag T_Execute::addXOp(std::shared_ptr<XOp> xop)
        {
            auto tag = m_nextTag++;
            // Determine the inputs and outputs of the xop
            auto inputs_func         = rocRoller::Operations::Inputs();
            auto assign_outputs_func = rocRoller::Operations::AssignOutputs();
            auto outputs             = assign_outputs_func.call(*xop, tag);
            auto inputs              = inputs_func.call(*xop);

            // Add the outputs of the xop to the outputs of the
            // T_Execute operation.
            for(const auto& output : outputs)
                m_outputs.insert(output);

            // Add all of the inputs that aren't outputs of the
            // T_Exectute operation as inputs to the T_Execute
            // operation.
            for(const auto& input : inputs)
            {
                if(m_outputs.find(input) == m_outputs.end())
                    m_inputs.insert(input);
            }

            m_xops.emplace_back(xop);
            // update the tag assigned to T_Execute
            setTag(m_nextTag);
            return tag;
        }

        template <CXOp T>
        inline OperationTag T_Execute::addXOp(T&& op)
        {
            return addXOp(std::make_shared<XOp>(std::forward<T>(op)));
        }

        inline OperationTag T_Execute::getNextTag() const
        {
            return m_nextTag;
        }

        inline std::string T_Execute::toString() const
        {
            std::ostringstream msg;

            msg << "T_EXECUTE";
            for(auto input : m_inputs)
                msg << " " << input;

            Operations::ToStringVisitor toStringVistor;

            for(auto const& xop : m_xops)
                msg << std::endl << "  " << toStringVistor.call(*xop);

            return msg.str();
        }
    }
}
