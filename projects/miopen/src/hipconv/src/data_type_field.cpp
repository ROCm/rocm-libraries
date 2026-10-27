#include "data_type_field.h"

#include <array>

namespace hipconv
{

namespace
{

// Every DataType, not only the input tags parse_data_type accepts, so a field reads back
// whatever it renders.
bool parse_data_type_tag(std::string_view s, DataType& out)
{
    for(const auto dtype : std::array{DataType::fp16,
                                      DataType::bf16,
                                      DataType::fp32,
                                      DataType::fp8,
                                      DataType::bf8,
                                      DataType::tf32})
        if(s == to_string(dtype))
            return out = dtype, true;
    return false;
}

const char* data_type_tag(DataType dtype)
{
    return to_string(dtype);
}

} // namespace

void data_type_field(KVDescriptor& d, std::string_view key, DataType value, DataType default_value)
{
    d.custom_field(key, value, parse_data_type_tag, data_type_tag, default_value);
}

} // namespace hipconv
