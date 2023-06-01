#pragma once

#include "Command.hpp"

namespace rocRoller
{
    namespace Operations
    {
        // ------------------------
        // XOp methods
        // ------------------------

        inline E_Unary::E_Unary(int dest, int a)
            : dest(dest)
            , a(a)
        {
        }

        inline E_Unary::E_Unary(const std::initializer_list<unsigned>& args, const unsigned& dest)
            : dest(dest)
            , a(*args.begin())
        {
            AssertFatal(args.size() == 1, ShowValue(args.size()));
        }

        inline std::string E_Unary::toString() const
        {
            std::ostringstream msg;

            msg << name() << " " << dest << ", " << a;

            return msg.str();
        }

        inline int E_Unary::getTag() const
        {
            return dest;
        }

        inline void E_Unary::setTag(int tag)
        {
            dest = tag;
        }

        inline std::unordered_set<int> E_Unary::getOutputs() const
        {
            return {dest};
        }

        inline std::unordered_set<int> E_Unary::getInputs() const
        {
            return {a};
        }

        inline E_Binary::E_Binary(int dest, int a, int b)
            : dest(dest)
            , a(a)
            , b(b)
        {
        }

        inline E_Binary::E_Binary(const std::initializer_list<unsigned>& args, unsigned& dest)
            : dest(dest)
            , a(*args.begin())
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

        inline int E_Binary::getTag() const
        {
            return dest;
        }

        inline void E_Binary::setTag(int tag)
        {
            dest = tag;
        }

        inline std::unordered_set<int> E_Binary::getOutputs() const
        {
            return {dest};
        }

        inline std::unordered_set<int> E_Binary::getInputs() const
        {
            return {a, b};
        }

        // ------------------------
        // T_Execute methods
        // ------------------------

        inline T_Execute::T_Execute()
            : m_nextTag(0)
        {
        }

        inline T_Execute::T_Execute(int starting_tag)
            : m_nextTag(starting_tag)
        {
        }

        inline void T_Execute::setCommand(std::shared_ptr<Command> command)
        {
            m_command = command;
        }

        inline std::unordered_set<int> T_Execute::getInputs() const
        {
            return m_inputs;
        }

        inline std::unordered_set<int> T_Execute::getOutputs() const
        {
            return m_outputs;
        }

        inline void T_Execute::addXOp(std::shared_ptr<XOp> xop)
        {
            // Determine the inputs and outputs of the xop
            auto inputs_func         = rocRoller::Operations::Inputs();
            auto assign_outputs_func = rocRoller::Operations::AssignOutputs();
            auto outputs             = assign_outputs_func.call(*xop, m_nextTag);
            m_nextTag++;
            auto inputs = inputs_func.call(*xop);

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
        }

        inline int T_Execute::getNextTag() const
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
