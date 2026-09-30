#pragma once

// A general key-value descriptor.
//
// A set of named typed fields matched against a "key=value,..." spec string and
// rendered back out to one. In hipconv it is used for kernel configuration
// descriptors (the --config selector and the config column), but nothing here is
// kernel-specific.
//
// A spec is a comma-separated list of `key=value` tokens (e.g.
// "waves_k=2,wave_k16=4,kh=3,direction=fprop"). Which keys exist, and how each
// maps to an owner's data, is the only caller-specific part; everything else
// (splitting the list, trimming, converting scalars, comparing, rendering) is
// mechanics every caller would otherwise repeat.
//
// KVDescriptor captures that: an owner derives from it and, in its constructor,
// registers one entry per field with int_field() / bool_field() /
// custom_field(). The base then drives both directions from that one table:
// match() checks a spec against the fields, and describe() renders them. A field
// is declared once and is automatically matchable and describable, so the two
// can never drift.
//
// Match contract: match() returns false (setting error()) on an unknown key or a
// value that does not parse; false (no error) on a known field whose value the
// spec does not satisfy; true only if every token matches. An empty spec matches.

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace hipconv
{

class KVDescriptor
{
public:
    virtual ~KVDescriptor() = default;

    // Match a comma-separated key=value spec against the registered fields.
    //
    // Each token is trimmed and empty tokens (from empty or trailing commas) are
    // skipped. See the contract in the file header.
    bool match(std::string_view spec);

    // Render the fields as a comma separated list of key-value pairs.
    //
    // Default values are omitted. The result is a valid match() spec.
    std::string describe() const;

    // Diagnostic set on an unknown key or unparseable value; empty otherwise.
    const std::string& error() const { return error_; }

protected:
    // Derived classes register their fields in their constructor.
    KVDescriptor() = default;

    // Register one field.
    //
    // `value` is the owner's current value for the field, captured by copy; `key`
    // is the spec key. Each field is both matchable (in match()) and describable
    // (in describe()).
    //
    // Two forms per type. The plain form always renders in describe() (use it for
    // identifying fields). The default_value form renders only when
    // value != default_value, so a field at its ordinary value stays out of the
    // output (use it for flags). Matching is identical either way.
    //
    // custom_field handles a non-scalar type via a pair of conversions: parse
    // (spec string -> T, false if malformed) and render (T -> string).
    void int_field(std::string_view key, int value);
    void int_field(std::string_view key, int value, int default_value);
    void bool_field(std::string_view key, bool value);
    void bool_field(std::string_view key, bool value, bool default_value);

    template <class T>
    void custom_field(std::string_view key,
                      T value,
                      bool (*parse)(std::string_view, T&),
                      const char* (*render)(T))
    {
        // An identifying field: no default, so it always renders. render (a
        // const char* function) converts to the std::string(T) core expects.
        add_field<T>(
            key, value, /*has_default=*/false, /*default_value=*/value, "value", parse, render);
    }

    // Record a malformed/unknown token: store `msg` as the error and return false.
    //
    // The single point that turns a diagnostic into the false return.
    bool fail(std::string msg);

private:
    // The one registrar every field type funnels through.
    //
    // Builds both closures once from a type's parse (string -> T) and render
    // (T -> string-ish, either std::string or const char*) so the int/bool/custom
    // entry points share all the match and describe logic. `type_label` names the
    // type in a parse-error diagnostic ("bad <type_label> for 'key'"). When
    // `has_default`, describe() omits the field at `default_value`; otherwise it
    // always renders.
    template <class T, class Render>
    void add_field(std::string_view key,
                   T value,
                   bool has_default,
                   T default_value,
                   const char* type_label,
                   bool (*parse)(std::string_view, T&),
                   Render render)
    {
        fields_.push_back(Field{
            key,
            [=, this](std::string_view val) {
            T parsed;
            if(!parse(val, parsed))
                return fail(std::string("bad ") + type_label + " for '" + std::string(key) + "'");
            return value == parsed;
        },
            [=]() -> std::string {
            if(has_default && value == default_value)
                return {};
            return std::string(key) + "=" + render(value);
        },
        });
    }

    struct Field
    {
        std::string_view key;
        std::function<bool(std::string_view)> matches; // parse val, compare to config's value
        std::function<std::string()> render;           // "key=value", or "" if at default
    };

    std::vector<Field> fields_;
    std::string error_;
};

} // namespace hipconv
