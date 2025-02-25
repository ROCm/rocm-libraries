/**
 * Block scale MX datatypes command.
 */

#pragma once

#include <string>

namespace rocRoller
{
    namespace Operations
    {
        /**
         * A block scale operation for MX datatypes
        */
        class BlockScale;

        enum class ScaleMode
        {
            None = 0, //< Tensor is not scaled.
            SingleScale, //< A single scalar value for the whole tensor
            Separate, //< Scale is separate from data
            Inline, //< Scale is inline with data
            Count
        };

        std::string   toString(ScaleMode const& mode);
        std::ostream& operator<<(std::ostream& stream, ScaleMode const& mode);

    }
}
