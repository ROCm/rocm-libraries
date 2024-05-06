#pragma once

#include "Operation.hpp"

namespace rocRoller
{
    namespace Operations
    {
        class T_Load_Linear : public BaseOperation
        {
        public:
            T_Load_Linear() = delete;
            T_Load_Linear(OperationTag tensor);

            OperationTag getTensorTag() const;
            std::string  toString() const;

        private:
            OperationTag m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val);

        class T_Load_Scalar : public BaseOperation
        {
        public:
            T_Load_Scalar() = delete;
            T_Load_Scalar(OperationTag scalar);

            OperationTag getScalarTag() const;
            std::string  toString() const;

        private:
            OperationTag m_scalarTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val);

        class T_Load_Tiled : public BaseOperation
        {
        public:
            T_Load_Tiled() = delete;
            T_Load_Tiled(OperationTag tensor);

            OperationTag getTensorTag() const;
            std::string  toString() const;

        private:
            OperationTag m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val);

        class T_Store_Linear : public BaseOperation
        {
        public:
            T_Store_Linear() = delete;
            T_Store_Linear(OperationTag source, OperationTag tensor);

            OperationTag getSrcTag() const;
            OperationTag getTensorTag() const;
            std::string  toString() const;

        private:
            OperationTag m_srcTag;
            OperationTag m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val);

        class T_Store_Tiled : public BaseOperation
        {
        public:
            T_Store_Tiled() = delete;
            T_Store_Tiled(OperationTag source, OperationTag tensor);

            OperationTag getSrcTag() const;
            OperationTag getTensorTag() const;
            std::string  toString() const;

        private:
            OperationTag m_srcTag;
            OperationTag m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val);
    }
}

#include "T_LoadStore_impl.hpp"
