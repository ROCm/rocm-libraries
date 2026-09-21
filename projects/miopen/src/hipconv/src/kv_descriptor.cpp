#include "kv_descriptor.h"

#include <charconv>

namespace hipconv
{

namespace
{

// Strip leading and trailing ASCII whitespace.
std::string_view trim(std::string_view s)
{
    const auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
    while(!s.empty() && ws(s.front()))
        s.remove_prefix(1);
    while(!s.empty() && ws(s.back()))
        s.remove_suffix(1);
    return s;
}

// Parse the whole string as a base-10 int.
//
// Returns false (leaving out unchanged) unless every character was consumed.
bool to_int(std::string_view s, int& out)
{
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    return ec == std::errc{} && ptr == s.data() + s.size();
}

// Parse "true"/"1" or "false"/"0". Returns false on anything else.
bool to_bool(std::string_view s, bool& out)
{
    if(s == "true" || s == "1")
        return out = true, true;
    if(s == "false" || s == "0")
        return out = false, true;
    return false;
}

// Split a "key=value" token: on the first '=', set key/val to the trimmed halves
// and return true.
//
// Return false if there is no '=' (a bare tag).
bool split_kv(std::string_view tok, std::string_view& key, std::string_view& val)
{
    auto e = tok.find('=');
    if(e == std::string_view::npos)
        return false;
    key = trim(tok.substr(0, e));
    val = trim(tok.substr(e + 1));
    return true;
}

} // namespace

namespace
{

std::string int_to_string(int v)
{
    return std::to_string(v);
}

const char* bool_to_string(bool b)
{
    return b ? "true" : "false";
}

} // namespace

void KVDescriptor::int_field(std::string_view key, int value)
{
    add_field<int>(key,
                   value,
                   /*has_default=*/false,
                   /*default_value=*/value,
                   "integer",
                   to_int,
                   int_to_string);
}

void KVDescriptor::int_field(std::string_view key, int value, int default_value)
{
    add_field<int>(
        key, value, /*has_default=*/true, default_value, "integer", to_int, int_to_string);
}

void KVDescriptor::bool_field(std::string_view key, bool value)
{
    add_field<bool>(key,
                    value,
                    /*has_default=*/false,
                    /*default_value=*/value,
                    "bool",
                    to_bool,
                    bool_to_string);
}

void KVDescriptor::bool_field(std::string_view key, bool value, bool default_value)
{
    add_field<bool>(
        key, value, /*has_default=*/true, default_value, "bool", to_bool, bool_to_string);
}

bool KVDescriptor::match(std::string_view spec)
{
    size_t pos = 0;
    while(pos <= spec.size())
    {
        size_t comma = spec.find(',', pos);
        std::string_view tok =
            (comma == std::string_view::npos) ? spec.substr(pos) : spec.substr(pos, comma - pos);
        tok = trim(tok);
        if(!tok.empty())
        {
            std::string_view key, val;
            if(!split_kv(tok, key, val))
                return fail("expected key=value, got '" + std::string(tok) + "'");
            const Field* field = nullptr;
            for(const auto& f : fields_)
                if(f.key == key)
                {
                    field = &f;
                    break;
                }
            if(!field)
                return fail("unknown key '" + std::string(key) + "'");
            if(!field->matches(val))
                return false;
        }
        if(comma == std::string_view::npos)
            break;
        pos = comma + 1;
    }
    return true;
}

std::string KVDescriptor::describe() const
{
    std::string out;
    for(const auto& f : fields_)
    {
        std::string rendered = f.render();
        if(rendered.empty())
            continue;
        if(!out.empty())
            out += ',';
        out += rendered;
    }
    return out;
}

bool KVDescriptor::fail(std::string msg)
{
    error_ = std::move(msg);
    return false;
}

} // namespace hipconv
