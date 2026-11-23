// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "DigitGroupingLocale.hpp"

#include <string>

namespace hipdnn_plugin_sdk::test
{
namespace
{

class GroupingPunct : public std::numpunct<char>
{
protected:
    char do_thousands_sep() const override
    {
        return ',';
    }

    std::string do_grouping() const override
    {
        return "\3";
    }
};

} // namespace

std::locale digitGroupingLocale()
{
    // The locale takes ownership of the facet and deletes it with its last copy.
    return {std::locale::classic(), new GroupingPunct};
}

} // namespace hipdnn_plugin_sdk::test
