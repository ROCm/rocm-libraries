#pragma once

namespace rocRoller
{
    namespace Operations
    {
        /**
         * @brief Prefix increment operator
         */
        inline OperationTag& OperationTag::operator++()
        {
            value++;
            return *this;
        }

        /**
         * @brief Postfix increment operator
         */
        inline OperationTag OperationTag::operator++(int)
        {
            auto rv = *this;
            ++(*this);
            return rv;
        }

        inline OperationTag operator+(OperationTag const& tag, int val)
        {
            rocRoller::Operations::OperationTag rv(static_cast<int32_t>(tag) + val);
            return rv;
        }

        inline bool OperationTag::uninitialized() const
        {
            return value == -1;
        }
    }
}

namespace std
{
    inline ostream& operator<<(ostream& stream, rocRoller::Operations::OperationTag const tag)
    {
        return stream << static_cast<int32_t>(tag);
    }

    template <>
    struct hash<rocRoller::Operations::OperationTag>
    {
        inline size_t operator()(rocRoller::Operations::OperationTag const& t) const noexcept
        {
            hash<int32_t> int_hash;
            return int_hash(static_cast<int32_t>(t));
        }
    };
}
