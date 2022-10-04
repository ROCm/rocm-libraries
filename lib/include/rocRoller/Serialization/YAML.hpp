
#pragma once

#include <istream>
#include <string>

namespace rocRoller
{
    namespace Serialization
    {
        /**
         * Note that the functions declared here will assume that you have included the serialization headers for any
         * type(s) you want to serialize.
         */

        /**
         * Returns T converted to YAML as a string.
         */
        template <typename T>
        std::string toYAML(T obj);

        /**
         * Parses YAML as a string into a T.
         */
        template <typename T>
        T fromYAML(std::string const& yaml);

        /**
         * Writes T to stream as YAML
         */
        template <typename T>
        void writeYAML(std::ostream& stream, T obj);

        /**
         * Reads the file `filename` and returns a T parsed from the YAML data it contains.
         */
        template <typename T>
        T readYAMLFile(std::string const& filename);

    }
}

#include "YAML_impl.hpp"
