#pragma once

#include "internal_api.h"

#include <string_view>
#include <utility>

namespace hipconv
{

// Splits a kernel description into family name and config string.
//
// Descriptions matching the following two regular expressions are accepted:
//
// ^([^\[\]]+)\[(.+)\]$ -> returns pair(\1, \2)
// ^([^\[\]]+)$         -> returns pair(\1, "")
//
// all other descriptions return pair("", "").
HIPCONV_INTERNAL_API auto
split_kernel_description(std::string_view desc) -> std::pair<std::string_view, std::string_view>;

} // namespace hipconv
