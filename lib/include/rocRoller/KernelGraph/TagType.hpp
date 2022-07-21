#pragma once

#include <string>
#include <vector>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * TagType - keys for mappings.
         *
         * Tags should be generated deterministically (and without
         * state).  See getTag().
         */
        struct TagType
        {
            // Command tag
            int ctag = -1;
            // Sub dimension (0 if not applicable)
            int dim = 0;
            // Destined for output
            bool output = false;
            // Variant index
            int index = -1;
            // Adhoc hash
            size_t unique = 0;

            auto operator<=>(const TagType&) const = default;

            std::string toString() const;
        };

        template <typename... Args>
        inline std::vector<TagType> tags(Args... args)
        {
            std::vector<TagType> r;
            for(auto const& dims : {args...})
                for(auto const& d : dims)
                    r.push_back(getTag(d));
            return r;
        }

        inline std::string toString(const TagType& t)
        {
            return t.toString();
        }

        std::ostream& operator<<(std::ostream& stream, TagType const& x);
    }
}
