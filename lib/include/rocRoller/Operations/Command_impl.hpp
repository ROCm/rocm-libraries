/**
 */

#pragma once

#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/Operations.hpp>

namespace rocRoller
{
    template <Operations::CConcreteOperation T>
    Operations::OperationTag Command::addOperation(T&& op)
    {
        return addOperation(std::make_shared<Operations::Operation>(std::forward<T>(op)));
    }

    template <Operations::CConcreteOperation T>
    T Command::getOperation(Operations::OperationTag const& tag) const
    {
        return std::get<T>(*findTag(tag));
    }
}
