#include "direction_field.h"

#include <string_view>

namespace hipconv
{

namespace
{

// to_string(Direction) is capitalized; descriptors use lowercase.
const char* direction_tag(Direction d)
{
    switch(d)
    {
    case Direction::Dgrad:
        return "dgrad";
    case Direction::Wgrad:
        return "wgrad";
    case Direction::Fprop:
    default:
        return "fprop";
    }
}

bool parse_direction_tag(std::string_view s, Direction& out)
{
    if(s == "fprop")
        return out = Direction::Fprop, true;
    if(s == "dgrad")
        return out = Direction::Dgrad, true;
    if(s == "wgrad")
        return out = Direction::Wgrad, true;
    return false;
}

} // namespace

void direction_field(KVDescriptor& d, Direction value)
{
    d.custom_field("direction", value, parse_direction_tag, direction_tag);
}

void direction_field(KVDescriptor& d, Direction value, Direction default_value)
{
    d.custom_field("direction", value, parse_direction_tag, direction_tag, default_value);
}

} // namespace hipconv
