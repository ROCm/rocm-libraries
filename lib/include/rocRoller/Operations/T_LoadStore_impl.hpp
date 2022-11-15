#pragma once

#include "Command.hpp"
#include "T_LoadStore.hpp"
#include <iterator>

namespace rocRoller
{
    namespace Operations
    {
        inline Load_Store_Operation::Load_Store_Operation()
            : m_tag(-1)
            , m_data_type(DataType::Float)
            , m_dims(1)
        {
        }

        inline Load_Store_Operation::Load_Store_Operation(DataType dataType, int dims, int dest)
            : m_data_type(dataType)
            , m_dims(dims)
            , m_tag(dest)
        {
        }

        inline void Load_Store_Operation::setCommand(std::shared_ptr<Command> command)
        {
            m_command = command;
        }

        inline int Load_Store_Operation::getTag() const
        {
            return m_tag;
        }

        inline void Load_Store_Operation::setTag(int tag)
        {
            m_tag = tag;
        }

        inline std::string
            Load_Store_Operation::getArgumentString(const unsigned char* runtime_args) const
        {
            std::ostringstream msg;

            if(m_data_type != DataType::None)
                msg << "." << m_data_type;
            msg << ".d" << m_dims << " " << m_tag << ", "
                << "(base=";
            if(m_pointer)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + m_pointer->offset()));
                else
                    msg << "&" << m_pointer->offset();
            }
            msg << ", lim=";
            if(m_extent)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + m_extent->offset()));
                else
                    msg << "&" << m_extent->offset();
            }
            msg << ", sizes={";
            for(auto i : m_sizes)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + i->offset())) << " ";
                else
                    msg << "&" << i->offset() << " ";
            }
            msg << "}, strides={";
            for(auto i : m_strides)
            {
                if(runtime_args)
                    msg << *(reinterpret_cast<const size_t*>(runtime_args + i->offset())) << " ";
                else
                    msg << "&" << i->offset() << " ";
            }
            msg << "})";

            return msg.str();
        }

        inline T_Load_Linear::T_Load_Linear()
            : Load_Store_Operation()
        {
        }

        inline T_Load_Linear::T_Load_Linear(DataType dataType, int dims, int dest)
            : Load_Store_Operation(dataType, dims, dest)
        {
        }

        inline void T_Load_Linear::allocateArguments()
        {
            if(auto ptr = m_command.lock())
            {
                std::string base = concatenate("Load_Linear_", m_tag);

                m_pointer = ptr->allocateArgument({m_data_type, PointerType::PointerGlobal},
                                                  DataDirection::ReadOnly,
                                                  base + "_pointer");

                m_extent = ptr->allocateArgument(
                    DataType::Int64, DataDirection::ReadOnly, base + "_extent");
                m_sizes = ptr->allocateArgumentVector(
                    DataType::Int64, m_dims, DataDirection::ReadOnly, base + "_size");
                m_strides = ptr->allocateArgumentVector(
                    DataType::Int64, m_dims, DataDirection::ReadOnly, base + "_stride");
            }
        }

        inline std::string T_Load_Linear::toString() const
        {
            return "T_LOAD_LINEAR" + Load_Store_Operation::getArgumentString(nullptr);
        }

        inline std::string T_Load_Linear::toString(const unsigned char* runtime_args) const
        {
            return "T_LOAD_LINEAR" + Load_Store_Operation::getArgumentString(runtime_args);
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val)
        {
            return stream << val.toString();
        }

        inline T_Load_Scalar::T_Load_Scalar()
            : Load_Store_Operation()
        {
        }

        inline T_Load_Scalar::T_Load_Scalar(VariableType variableType, int dest)
            : Load_Store_Operation(variableType.dataType, 0, dest)
            , m_variable_type(variableType)
        {
        }

        inline void T_Load_Scalar::allocateArguments()
        {
            if(auto ptr = m_command.lock())
            {
                m_pointer = ptr->allocateArgument(m_variable_type, DataDirection::ReadOnly);
            }
        }

        inline std::string T_Load_Scalar::toString() const
        {
            return "T_LOAD_SCALAR" + Load_Store_Operation::getArgumentString(nullptr);
        }

        inline std::string T_Load_Scalar::toString(const unsigned char* runtime_args) const
        {
            return "T_LOAD_SCALAR" + Load_Store_Operation::getArgumentString(runtime_args);
        }

        inline VariableType T_Load_Scalar::variableType() const
        {
            return m_variable_type;
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val)
        {
            return stream << val.toString();
        }

        inline T_Load_Tiled::T_Load_Tiled()
            : Load_Store_Operation()
        {
        }

        inline T_Load_Tiled::T_Load_Tiled(DataType dataType, int dims, int dest)
            : Load_Store_Operation(dataType, dims, dest)
            , m_variable_type(dataType)
        {
        }

        inline void T_Load_Tiled::allocateArguments()
        {

            if(auto ptr = m_command.lock())
            {
                rocRoller::Log::getLogger()->debug("T_Load_Tiled::allocateArguments");
                std::string base = concatenate("Load_Tiled_", m_tag);

                m_pointer = ptr->allocateArgument({m_data_type, PointerType::PointerGlobal},
                                                  DataDirection::ReadOnly,
                                                  base + "_pointer");

                m_extent = ptr->allocateArgument(
                    DataType::Int64, DataDirection::ReadOnly, base + "_extent");
                m_sizes = ptr->allocateArgumentVector(
                    DataType::Int64, m_dims, DataDirection::ReadOnly, base + "_size");
                m_strides = ptr->allocateArgumentVector(
                    DataType::Int64, m_dims, DataDirection::ReadOnly, base + "_stride");
                rocRoller::Log::getLogger()->debug("T_Load_Tiled::allocateArguments done");
            }
        }

        inline std::string T_Load_Tiled::toString() const
        {
            return "T_LOAD_TILED" + Load_Store_Operation::getArgumentString(nullptr);
        }

        inline std::string T_Load_Tiled::toString(const unsigned char* runtime_args) const
        {
            return "T_LOAD_TILED" + Load_Store_Operation::getArgumentString(runtime_args);
        }

        inline VariableType T_Load_Tiled::variableType() const
        {
            return m_variable_type;
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val)
        {
            return stream << val.toString();
        }

        inline T_Store_Linear::T_Store_Linear()
            : Load_Store_Operation()
        {
        }

        inline T_Store_Linear::T_Store_Linear(int dims, int dest)
            : Load_Store_Operation(DataType::None, dims, dest)
        {
        }

        inline T_Store_Linear::T_Store_Linear(const std::initializer_list<unsigned>& args,
                                              void* const,
                                              const unsigned dims,
                                              const std::vector<uint64_t>&)
            : Load_Store_Operation(DataType::None, dims, *args.begin())
        {
        }

        inline void T_Store_Linear::allocateArguments()
        {
            if(auto ptr = m_command.lock())
            {
                std::string base = concatenate("Store_Linear_", m_tag);

                m_pointer = ptr->allocateArgument({DataType::Int32, PointerType::PointerGlobal},
                                                  DataDirection::WriteOnly,
                                                  base + "_pointer");
                m_extent  = ptr->allocateArgument(
                    DataType::Int64, DataDirection::ReadOnly, base + "_extent");
                m_strides = ptr->allocateArgumentVector(
                    DataType::Int64, m_dims, DataDirection::ReadOnly, base + "_stride");
            }
        }

        inline std::string T_Store_Linear::toString() const
        {
            return "T_STORE_LINEAR" + Load_Store_Operation::getArgumentString(nullptr);
        }

        inline std::string T_Store_Linear::toString(const unsigned char* runtime_args) const
        {
            return "T_STORE_LINEAR" + Load_Store_Operation::getArgumentString(runtime_args);
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val)
        {
            return stream << val.toString();
        }

        inline T_Store_Tiled::T_Store_Tiled()
            : Load_Store_Operation()
        {
        }

        inline T_Store_Tiled::T_Store_Tiled(DataType dataType, int dims, int dest)
            : Load_Store_Operation(dataType, dims, dest)
        {
        }

        inline T_Store_Tiled::T_Store_Tiled(const std::initializer_list<unsigned>& args,
                                            void* const,
                                            const unsigned dims,
                                            const std::vector<uint64_t>&)
            : Load_Store_Operation(DataType::None, dims, *args.begin())
        {
        }

        inline void T_Store_Tiled::allocateArguments()
        {
            if(auto ptr = m_command.lock())
            {
                std::string base = concatenate("Store_Tiled_", m_tag);

                m_pointer = ptr->allocateArgument({DataType::Int32, PointerType::PointerGlobal},
                                                  DataDirection::WriteOnly,
                                                  base + "_pointer");
                m_extent  = ptr->allocateArgument(
                    DataType::Int64, DataDirection::ReadOnly, base + "_extent");
                m_strides = ptr->allocateArgumentVector(
                    DataType::Int64, m_dims, DataDirection::ReadOnly, base + "_stride");
            }
        }

        inline std::string T_Store_Tiled::toString() const
        {
            return "T_STORE_TILED" + Load_Store_Operation::getArgumentString(nullptr);
        }

        inline std::string T_Store_Tiled::toString(const unsigned char* runtime_args) const
        {
            return "T_STORE_TILED" + Load_Store_Operation::getArgumentString(runtime_args);
        }

        inline std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val)
        {
            return stream << val.toString();
        }
    }
}
