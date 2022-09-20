/**
 */

#pragma once

#include "CommandArgument.hpp"
#include "Command_fwd.hpp"

#include "../CodeGen/Buffer.hpp"
#include "../DataTypes/DataTypes.hpp"

#include <memory>
#include <variant>

namespace rocRoller
{
    namespace Operations
    {
        class Load_Store_Operation
        {
        public:
            Load_Store_Operation();
            Load_Store_Operation(DataType dataType, int dims, int dest);

            void setCommand(std::shared_ptr<Command>);
            int  getTag() const;
            void setTag(int tag);

            std::vector<std::shared_ptr<CommandArgument>> strides() const
            {
                return m_strides;
            }
            std::vector<std::shared_ptr<CommandArgument>> sizes() const
            {
                return m_sizes;
            }
            DataType dataType() const
            {
                return m_data_type;
            }
            std::shared_ptr<CommandArgument> data() const
            {
                return m_pointer;
            }

        protected:
            std::string getArgumentString(const unsigned char*) const;

            int m_tag;

            DataType m_data_type;
            int      m_dims;

            std::weak_ptr<Command> m_command;

            std::shared_ptr<CommandArgument> m_pointer;
            std::shared_ptr<CommandArgument> m_extent;

            std::vector<std::shared_ptr<CommandArgument>> m_sizes;
            std::vector<std::shared_ptr<CommandArgument>> m_strides;
        };

        class T_Load_Linear : public Load_Store_Operation
        {
        public:
            T_Load_Linear();
            T_Load_Linear(DataType dataType, int dims, int dest);
            std::string toString() const;
            std::string toString(const unsigned char*) const;
            void        allocateArguments();
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val);

        class T_Load_Scalar : public Load_Store_Operation
        {
        public:
            T_Load_Scalar();
            T_Load_Scalar(VariableType variableType, int dest);
            std::string  toString() const;
            std::string  toString(const unsigned char*) const;
            void         allocateArguments();
            VariableType variableType() const;

        private:
            VariableType m_variable_type;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val);

        class T_Load_Tiled : public Load_Store_Operation
        {
        public:
            T_Load_Tiled();
            T_Load_Tiled(DataType dataType, int dims, int dest);
            std::string  toString() const;
            std::string  toString(const unsigned char*) const;
            void         allocateArguments();
            VariableType variableType() const;

        private:
            VariableType m_variable_type;
            int          m_rank;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val);

        class T_Store_Linear : public Load_Store_Operation
        {
        public:
            T_Store_Linear();
            T_Store_Linear(int dims, int dest);
            T_Store_Linear(const std::initializer_list<unsigned>&,
                           void* const,
                           const unsigned,
                           const std::vector<uint64_t>&);
            std::string toString() const;
            std::string toString(const unsigned char*) const;
            void        allocateArguments();
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val);

        class T_Store_Tiled : public Load_Store_Operation
        {
        public:
            T_Store_Tiled();
            T_Store_Tiled(DataType dataType, int dims, int dest);
            T_Store_Tiled(const std::initializer_list<unsigned>&,
                          void* const,
                          const unsigned,
                          const std::vector<uint64_t>&);
            std::string toString() const;
            std::string toString(const unsigned char*) const;
            void        allocateArguments();
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val);
    }
}

#include "T_LoadStore_impl.hpp"
