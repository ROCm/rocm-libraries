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
            T_Load_Linear(int tensor);

            int         getTensorTag() const;
            std::string toString() const;

        private:
            int m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Linear const& val);

        class T_Load_Scalar : public BaseOperation
        {
        public:
            T_Load_Scalar() = delete;
            T_Load_Scalar(int scalar);

            int         getScalarTag() const;
            std::string toString() const;

        private:
            int m_scalarTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Scalar const& val);

        class T_Load_Tiled : public BaseOperation
        {
        public:
            T_Load_Tiled() = delete;
            T_Load_Tiled(int tensor);

            int         getTensorTag() const;
            std::string toString() const;

        private:
            int m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Load_Tiled const& val);

        class T_Store_Linear : public BaseOperation
        {
        public:
            T_Store_Linear() = delete;
            T_Store_Linear(int source, int tensor);

            int         getSrcTag() const;
            int         getTensorTag() const;
            std::string toString() const;

        private:
            int m_srcTag;
            int m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Linear const& val);

        class T_Store_Tiled : public BaseOperation
        {
        public:
            T_Store_Tiled() = delete;
            T_Store_Tiled(int source, int tensor);

            int         getSrcTag() const;
            int         getTensorTag() const;
            std::string toString() const;

        private:
            int m_srcTag;
            int m_tensorTag;
        };

        std::ostream& operator<<(std::ostream& stream, T_Store_Tiled const& val);
    }
}

#include "T_LoadStore_impl.hpp"
