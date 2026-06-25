// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal, dependency-free JSON reader for ck_dsl manifests.
//
// The ck_dsl manifest (`ck.dsl.example.manifest/v1`) and the kernel-library
// index are machine-generated and well-formed, so we keep the runtime library
// dependency-light (HIP + comgr only) by parsing JSON ourselves instead of
// pulling nlohmann/json. Supports objects, arrays, strings (with the common
// escapes), numbers, booleans, and null. Not a general-purpose parser; it is
// exactly enough for the manifest schema.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ck_dsl {
namespace json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

enum class Kind { Null, Bool, Number, String, Array, Object };

class Value {
   public:
    Value() : kind_(Kind::Null) {}
    Value(bool b) : kind_(Kind::Bool), bool_(b) {}
    Value(double n) : kind_(Kind::Number), num_(n) {}
    Value(std::string s) : kind_(Kind::String), str_(std::move(s)) {}
    Value(Array a) : kind_(Kind::Array), arr_(std::make_shared<Array>(std::move(a))) {}
    Value(Object o) : kind_(Kind::Object), obj_(std::make_shared<Object>(std::move(o))) {}

    Kind kind() const {
        return kind_;
    }
    bool is_object() const {
        return kind_ == Kind::Object;
    }
    bool is_array() const {
        return kind_ == Kind::Array;
    }
    bool is_string() const {
        return kind_ == Kind::String;
    }
    bool is_number() const {
        return kind_ == Kind::Number;
    }
    bool is_null() const {
        return kind_ == Kind::Null;
    }

    // Accessors (throw std::runtime_error on type mismatch).
    const std::string& as_string() const {
        if (kind_ != Kind::String) throw std::runtime_error("json: not a string");
        return str_;
    }
    double as_number() const {
        if (kind_ != Kind::Number) throw std::runtime_error("json: not a number");
        return num_;
    }
    long as_int() const {
        return static_cast<long>(as_number());
    }
    bool as_bool() const {
        if (kind_ != Kind::Bool) throw std::runtime_error("json: not a bool");
        return bool_;
    }
    const Array& as_array() const {
        if (kind_ != Kind::Array) throw std::runtime_error("json: not an array");
        return *arr_;
    }
    const Object& as_object() const {
        if (kind_ != Kind::Object) throw std::runtime_error("json: not an object");
        return *obj_;
    }

    // Object helpers.
    bool has(const std::string& k) const {
        return kind_ == Kind::Object && obj_->find(k) != obj_->end();
    }
    const Value& at(const std::string& k) const {
        auto it = as_object().find(k);
        if (it == as_object().end()) throw std::runtime_error("json: missing key '" + k + "'");
        return it->second;
    }
    // Optional getters with defaults.
    std::string get_str(const std::string& k, const std::string& dflt = "") const {
        return has(k) && at(k).is_string() ? at(k).as_string() : dflt;
    }
    long get_int(const std::string& k, long dflt = 0) const {
        return has(k) && at(k).is_number() ? at(k).as_int() : dflt;
    }

   private:
    Kind kind_;
    bool bool_ = false;
    double num_ = 0.0;
    std::string str_;
    std::shared_ptr<Array> arr_;
    std::shared_ptr<Object> obj_;
};

namespace detail {

struct Parser {
    const char* p;
    const char* end;

    explicit Parser(const std::string& s) : p(s.data()), end(s.data() + s.size()) {}

    [[noreturn]] void fail(const char* msg) {
        throw std::runtime_error(std::string("json parse: ") + msg);
    }

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        return v;
    }

    Value parse_value() {
        skip_ws();
        if (p >= end) fail("unexpected end");
        char c = *p;
        switch (c) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"':
                return Value(parse_string());
            case 't':
            case 'f':
                return parse_bool();
            case 'n':
                return parse_null();
            default:
                return parse_number();
        }
    }

    Value parse_object() {
        Object o;
        ++p;  // '{'
        skip_ws();
        if (p < end && *p == '}') {
            ++p;
            return Value(std::move(o));
        }
        while (true) {
            skip_ws();
            if (p >= end || *p != '"') fail("expected key string");
            std::string key = parse_string();
            skip_ws();
            if (p >= end || *p != ':') fail("expected ':'");
            ++p;
            o[key] = parse_value();
            skip_ws();
            if (p >= end) fail("unterminated object");
            if (*p == ',') {
                ++p;
                continue;
            }
            if (*p == '}') {
                ++p;
                break;
            }
            fail("expected ',' or '}'");
        }
        return Value(std::move(o));
    }

    Value parse_array() {
        Array a;
        ++p;  // '['
        skip_ws();
        if (p < end && *p == ']') {
            ++p;
            return Value(std::move(a));
        }
        while (true) {
            a.push_back(parse_value());
            skip_ws();
            if (p >= end) fail("unterminated array");
            if (*p == ',') {
                ++p;
                continue;
            }
            if (*p == ']') {
                ++p;
                break;
            }
            fail("expected ',' or ']'");
        }
        return Value(std::move(a));
    }

    std::string parse_string() {
        ++p;  // opening quote
        std::string out;
        while (p < end && *p != '"') {
            char c = *p++;
            if (c == '\\') {
                if (p >= end) fail("bad escape");
                char e = *p++;
                switch (e) {
                    case '"':
                        out += '"';
                        break;
                    case '\\':
                        out += '\\';
                        break;
                    case '/':
                        out += '/';
                        break;
                    case 'n':
                        out += '\n';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    case 'b':
                        out += '\b';
                        break;
                    case 'f':
                        out += '\f';
                        break;
                    case 'u': {
                        if (end - p < 4) fail("bad \\u");
                        int cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = *p++;
                            cp <<= 4;
                            if (h >= '0' && h <= '9')
                                cp += h - '0';
                            else if (h >= 'a' && h <= 'f')
                                cp += h - 'a' + 10;
                            else if (h >= 'A' && h <= 'F')
                                cp += h - 'A' + 10;
                            else
                                fail("bad hex");
                        }
                        // Minimal UTF-8 encode (BMP only; enough for manifest notes).
                        if (cp < 0x80) {
                            out += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            out += static_cast<char>(0xC0 | (cp >> 6));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            out += static_cast<char>(0xE0 | (cp >> 12));
                            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            out += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default:
                        fail("unknown escape");
                }
            } else {
                out += c;
            }
        }
        if (p >= end) fail("unterminated string");
        ++p;  // closing quote
        return out;
    }

    Value parse_bool() {
        if (end - p >= 4 && std::string(p, p + 4) == "true") {
            p += 4;
            return Value(true);
        }
        if (end - p >= 5 && std::string(p, p + 5) == "false") {
            p += 5;
            return Value(false);
        }
        fail("bad bool");
    }

    Value parse_null() {
        if (end - p >= 4 && std::string(p, p + 4) == "null") {
            p += 4;
            return Value();
        }
        fail("bad null");
    }

    Value parse_number() {
        const char* start = p;
        if (p < end && (*p == '-' || *p == '+')) ++p;
        while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' || *p == 'E' ||
                           *p == '+' || *p == '-'))
            ++p;
        if (p == start) fail("bad number");
        return Value(std::strtod(std::string(start, p).c_str(), nullptr));
    }
};

}  // namespace detail

inline Value parse(const std::string& text) {
    return detail::Parser(text).parse();
}

}  // namespace json
}  // namespace ck_dsl
