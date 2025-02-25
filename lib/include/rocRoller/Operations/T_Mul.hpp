/**
 * T_Mul (tensor/matrix multiply) command.
 */

#pragma once

#include <memory>
#include <unordered_set>

#include "Operation.hpp"

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    namespace Operations
    {
        class T_Mul : public BaseOperation
        {
        public:
            T_Mul() = delete;
            T_Mul(OperationTag a, OperationTag b);

            std::unordered_set<OperationTag> getInputs() const;
            std::string                      toString() const;

            OperationTag a, b;

            bool operator==(T_Mul const&) const;

            template <typename T1, typename T2, typename T3>
            friend struct rocRoller::Serialization::MappingTraits;
        };

    }
}

#include "T_Mul_impl.hpp"
