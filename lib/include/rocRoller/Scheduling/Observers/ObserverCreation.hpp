
#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>

#include "../../Context_fwd.hpp"
#include "../Scheduling.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * @brief Create a MetaObserver object
         *
         * This function builds a MetaObserver out of all observers that are required for the given context.
         *
         * @param ctx
         * @return std::shared_ptr<Scheduling::IObserver>
         */
        std::shared_ptr<Scheduling::IObserver> createObserver(ContextPtr const& ctx);
    };
}
