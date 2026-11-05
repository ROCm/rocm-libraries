#include "config_desc.h"
#include "data_type_field.h"
#include "direction_field.h"

namespace hipconv::cdna4::direct_flex
{

ConfigMatcher::ConfigMatcher(const Variant& v)
{
    int_field("x_tile16", v.tile.x_tile16);
    int_field("k_tile16", v.tile.k_tile16);
    int_field("waves_x", v.tile.waves_x);
    int_field("waves_k", v.tile.waves_k);
    int_field("csplit", v.tile.csplit, 1);
    data_type_field(*this, "type", v.type, DataType::fp16);
    direction_field(*this, v.direction, Direction::Fprop);
    bool_field("narrow_k", v.narrow_k, false);
    bool_field("c_tail", v.c_tail, false);
    int_field("sweep", v.sweep, 0);
    bool_field("one_ftile", v.one_ftile, false);
    bool_field("depth", v.depth, false);
}

} // namespace hipconv::cdna4::direct_flex
