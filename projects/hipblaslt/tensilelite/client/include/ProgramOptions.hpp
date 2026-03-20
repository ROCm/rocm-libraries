/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Drop-in replacement for boost::program_options used by tensilelite client.
 * Uses std::any and standard library only; no Boost dependency.
 *
 *******************************************************************************/

#pragma once

#include <any>
#include <cctype>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace TensileLite
{
    namespace Client
    {
        namespace po
        {

            inline std::vector<std::string> split_string(std::string const& s,
                                                         char const*        sep = ",;")
            {
                std::vector<std::string> out;
                std::string::size_type   pos = 0;
                for(;;)
                {
                    auto next = s.find_first_of(sep, pos);
                    if(next == std::string::npos)
                    {
                        if(pos < s.size())
                        {
                            std::string part = s.substr(pos);
                            if(!part.empty())
                                out.push_back(std::move(part));
                        }
                        break;
                    }
                    if(next > pos)
                        out.push_back(s.substr(pos, next - pos));
                    pos = next + 1;
                }
                return out;
            }

            inline bool parse_bool_string(std::string const& s)
            {
                std::string lower;
                lower.reserve(s.size());
                for(char c : s)
                    lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
                if(lower == "1" || lower == "true" || lower == "yes")
                    return true;
                if(lower == "0" || lower == "false" || lower == "no")
                    return false;
                throw std::runtime_error("Failed to parse bool: " + s);
            }

            inline std::string trim_opt(std::string s)
            {
                auto start = s.find_first_not_of(" \t");
                if(start == std::string::npos)
                    return {};
                auto end = s.find_last_not_of(" \t");
                return s.substr(start, end == std::string::npos ? end : end - start + 1);
            }

            inline std::string get_canonical_option_name(std::string const& opts)
            {
                auto pos = opts.find(',');
                if(pos == std::string::npos)
                    return trim_opt(opts);
                return trim_opt(opts.substr(0, pos));
            }

            class variable_value
            {
                std::any m_value;

            public:
                variable_value() = default;
                explicit variable_value(std::any v)
                    : m_value(std::move(v))
                {
                }

                bool empty() const
                {
                    return !m_value.has_value();
                }

                std::any& value()
                {
                    return m_value;
                }
                std::any const& value() const
                {
                    return m_value;
                }

                template <typename T>
                T const& as() const
                {
                    if(!m_value.has_value())
                    {
                        // Match Boost: unset option yields default-constructed T when .as<T>() is used
                        static T const defaultVal{};
                        return defaultVal;
                    }
                    try
                    {
                        return std::any_cast<T const&>(m_value);
                    }
                    catch(std::bad_any_cast const&)
                    {
                        // Allow bool option stored as string (e.g. "True"/"False" from config)
                        if constexpr(std::is_same_v<T, bool>)
                        {
                            if(m_value.type() == typeid(std::string))
                            {
                                static bool converted;
                                converted
                                    = parse_bool_string(std::any_cast<std::string const&>(m_value));
                                return converted;
                            }
                        }
                        throw std::runtime_error(
                            "program_options: bad_any_cast - stored option value has a different "
                            "type than the requested .as<T>()");
                    }
                }
            };

            class variables_map
            {
                std::map<std::string, variable_value> m_map;
                static variable_value const           s_empty;

            public:
                using key_type       = std::string;
                using mapped_type    = variable_value;
                using value_type     = std::pair<const std::string, variable_value>;
                using iterator       = std::map<std::string, variable_value>::iterator;
                using const_iterator = std::map<std::string, variable_value>::const_iterator;

                variable_value& operator[](key_type const& k)
                {
                    return m_map[k];
                }
                variable_value const& operator[](key_type const& k) const
                {
                    auto it = m_map.find(k);
                    if(it != m_map.end())
                        return it->second;
                    return s_empty;
                }

                variable_value& at(key_type const& k)
                {
                    return m_map.at(k);
                }
                variable_value const& at(key_type const& k) const
                {
                    return m_map.at(k);
                }

                size_t count(key_type const& k) const
                {
                    return m_map.count(k);
                }

                iterator begin()
                {
                    return m_map.begin();
                }
                iterator end()
                {
                    return m_map.end();
                }
                const_iterator begin() const
                {
                    return m_map.begin();
                }
                const_iterator end() const
                {
                    return m_map.end();
                }
                const_iterator find(key_type const& k) const
                {
                    return m_map.find(k);
                }
                iterator find(key_type const& k)
                {
                    return m_map.find(k);
                }

                std::pair<iterator, bool> insert(value_type const& v)
                {
                    return m_map.insert(v);
                }
                void clear()
                {
                    m_map.clear();
                }
            };

            struct option_value_base
            {
                virtual ~option_value_base()                                            = default;
                virtual std::any                                    get_default() const = 0;
                virtual std::function<std::any(std::string const&)> parser() const      = 0;
                /** Parse one line from config file. Default: parser()(value). Override for Boost-compatible
                 *  behaviour where each config line is one value (no comma split). */
                virtual std::any parse_config_line(std::string const& value) const
                {
                    return parser()(value);
                }
                virtual bool is_multivalued() const
                {
                    return false;
                }
                virtual bool is_flag() const
                {
                    return false;
                }
                virtual bool has_explicit_default() const
                {
                    return false;
                }
                virtual bool has_implicit_value() const
                {
                    return false;
                }
                virtual std::any get_implicit_value() const
                {
                    return std::any();
                }
                virtual void merge_into(std::any& current, std::any const& new_val) const
                {
                    current = new_val;
                }
            };

            template <typename T>
            struct typed_value_holder : option_value_base
            {
                T    m_default{};
                bool m_has_explicit_default = false;

                typed_value_holder* default_value(T v)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }
                typed_value_holder* default_value(T v, std::string const&)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }

                bool has_explicit_default() const override
                {
                    return m_has_explicit_default;
                }

                std::any get_default() const override
                {
                    return std::any(m_default);
                }

                std::function<std::any(std::string const&)> parser() const override
                {
                    return [](std::string const& s) {
                        T                  t;
                        std::istringstream ss(s);
                        ss >> t;
                        if(!ss && !ss.eof())
                            throw std::runtime_error("Failed to parse option value: " + s);
                        return std::any(t);
                    };
                }

                bool is_flag() const override
                {
                    return false;
                }
            };

            template <>
            struct typed_value_holder<bool> : option_value_base
            {
                bool m_default              = false;
                bool m_has_explicit_default = false;
                bool m_has_implicit         = false;
                bool m_implicit_value       = false;

                typed_value_holder* default_value(bool v)
                {
                    m_default              = v;
                    m_has_explicit_default = true;
                    return this;
                }
                typed_value_holder* default_value(bool v, std::string const&)
                {
                    m_default              = v;
                    m_has_explicit_default = true;
                    return this;
                }
                typed_value_holder* implicit_value(bool v)
                {
                    m_has_implicit   = true;
                    m_implicit_value = v;
                    return this;
                }

                bool has_explicit_default() const override
                {
                    return m_has_explicit_default;
                }
                bool has_implicit_value() const override
                {
                    return m_has_implicit;
                }
                std::any get_implicit_value() const override
                {
                    return std::any(m_implicit_value);
                }

                std::any get_default() const override
                {
                    return std::any(m_default);
                }

                std::function<std::any(std::string const&)> parser() const override
                {
                    return [](std::string const& s) { return std::any(parse_bool_string(s)); };
                }

                bool is_flag() const override
                {
                    return true;
                }
            };

            template <typename U>
            struct typed_value_holder<std::vector<U>> : option_value_base
            {
                std::vector<U> m_default;
                bool           m_has_explicit_default = false;

                typed_value_holder* default_value(std::vector<U> v)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }
                typed_value_holder* default_value(std::vector<U> v, std::string const&)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }

                bool has_explicit_default() const override
                {
                    return m_has_explicit_default;
                }

                std::any get_default() const override
                {
                    return std::any(m_default);
                }

                bool is_multivalued() const override
                {
                    return true;
                }

                std::function<std::any(std::string const&)> parser() const override
                {
                    return [](std::string const& s) {
                        auto           parts = split_string(s);
                        std::vector<U> out;
                        out.reserve(parts.size());
                        for(auto const& p : parts)
                        {
                            if(p.empty())
                                continue;
                            U                  u;
                            std::istringstream ss(p);
                            ss >> u;
                            if(!ss && !ss.eof())
                                throw std::runtime_error("Failed to parse: " + p);
                            out.push_back(std::move(u));
                        }
                        return std::any(out);
                    };
                }

                void merge_into(std::any& current, std::any const& new_val) const override
                {
                    auto&       c = std::any_cast<std::vector<U>&>(current);
                    auto const& n = std::any_cast<std::vector<U> const&>(new_val);
                    c.insert(c.end(), n.begin(), n.end());
                }
            };

            template <>
            struct typed_value_holder<std::vector<std::string>> : option_value_base
            {
                std::vector<std::string> m_default;
                bool                     m_has_explicit_default = false;

                typed_value_holder* default_value(std::vector<std::string> v)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }
                typed_value_holder* default_value(std::vector<std::string> v, std::string const&)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }

                bool has_explicit_default() const override
                {
                    return m_has_explicit_default;
                }

                std::any get_default() const override
                {
                    return std::any(m_default);
                }

                bool is_multivalued() const override
                {
                    return true;
                }

                std::function<std::any(std::string const&)> parser() const override
                {
                    return [](std::string const& s) { return std::any(split_string(s)); };
                }

                /** Boost config file behaviour: each line is one value (no comma split). */
                std::any parse_config_line(std::string const& value) const override
                {
                    std::string t = trim_opt(value);
                    if(t.empty())
                        return std::any(std::vector<std::string>{});
                    return std::any(std::vector<std::string>{std::move(t)});
                }

                void merge_into(std::any& current, std::any const& new_val) const override
                {
                    auto&       c = std::any_cast<std::vector<std::string>&>(current);
                    auto const& n = std::any_cast<std::vector<std::string> const&>(new_val);
                    c.insert(c.end(), n.begin(), n.end());
                }
            };

            template <>
            struct typed_value_holder<std::vector<bool>> : option_value_base
            {
                std::vector<bool> m_default;
                bool              m_has_explicit_default = false;

                typed_value_holder* default_value(std::vector<bool> v)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }
                typed_value_holder* default_value(std::vector<bool> v, std::string const&)
                {
                    m_default              = std::move(v);
                    m_has_explicit_default = true;
                    return this;
                }

                bool has_explicit_default() const override
                {
                    return m_has_explicit_default;
                }

                std::any get_default() const override
                {
                    return std::any(m_default);
                }

                bool is_multivalued() const override
                {
                    return true;
                }

                std::function<std::any(std::string const&)> parser() const override
                {
                    return [](std::string const& s) {
                        auto              parts = split_string(s);
                        std::vector<bool> out;
                        out.reserve(parts.size());
                        for(auto const& p : parts)
                        {
                            if(p.empty())
                                continue;
                            out.push_back(parse_bool_string(p));
                        }
                        return std::any(out);
                    };
                }

                void merge_into(std::any& current, std::any const& new_val) const override
                {
                    auto&       c = std::any_cast<std::vector<bool>&>(current);
                    auto const& n = std::any_cast<std::vector<bool> const&>(new_val);
                    c.insert(c.end(), n.begin(), n.end());
                }
            };

            template <typename T>
            using typed_value = typed_value_holder<T>;

            template <typename T>
            typed_value_holder<T>* value()
            {
                return new typed_value_holder<T>();
            }

            class options_description
            {
            public:
                struct option_entry
                {
                    std::string                        opts; // e.g. "help,h"
                    std::string                        description;
                    std::unique_ptr<option_value_base> value_semantic;
                };
                using option_list = std::vector<option_entry>;

                explicit options_description(std::string const&) {}

                class add_options_helper
                {
                    option_list& m_list;

                public:
                    explicit add_options_helper(option_list& list)
                        : m_list(list)
                    {
                    }

                    add_options_helper& operator()(std::string const& opts, std::string const& desc)
                    {
                        auto* e = new typed_value_holder<bool>();
                        e->default_value(false);
                        m_list.push_back({opts, desc, std::unique_ptr<option_value_base>(e)});
                        return *this;
                    }

                    template <typename T>
                    add_options_helper& operator()(std::string const&     opts,
                                                   typed_value_holder<T>* val,
                                                   std::string const&     desc)
                    {
                        m_list.push_back({opts, desc, std::unique_ptr<option_value_base>(val)});
                        return *this;
                    }
                };

                add_options_helper add_options()
                {
                    return add_options_helper(m_options);
                }

                option_list const& options() const
                {
                    return m_options;
                }

            private:
                option_list m_options;
            };

            variables_map parse_command_line(int                        argc,
                                             char const* const          argv[],
                                             options_description const& desc);

            variables_map parse_config_file(std::istream& is, options_description const& desc);

            inline void store(variables_map const& from, variables_map& to)
            {
                for(auto const& kv : from)
                    to[kv.first].value() = kv.second.value();
            }

            inline void notify(variables_map const&) {}

            std::ostream& operator<<(std::ostream& os, options_description const& desc);

        } // namespace po
    } // namespace Client
} // namespace TensileLite
