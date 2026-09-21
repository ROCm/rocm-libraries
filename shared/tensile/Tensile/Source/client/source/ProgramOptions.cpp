/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 *******************************************************************************/

#include "ProgramOptions.hpp"

#include <iomanip>
#include <iostream>

namespace Tensile
{
    namespace Client
    {
        namespace po
        {

            variable_value const variables_map::s_empty;

            variable_value::variable_value(variable_value const& other)
                : m_value(other.m_value)
            {
            }

            variable_value::variable_value(variable_value&& other) noexcept
                : m_value(std::move(other.m_value))
            {
            }

            variable_value& variable_value::operator=(variable_value const& other)
            {
                m_value = other.m_value;
                return *this;
            }

            variable_value& variable_value::operator=(variable_value&& other) noexcept
            {
                m_value = std::move(other.m_value);
                return *this;
            }

            static std::vector<std::string> get_option_names(std::string const& opts)
            {
                std::vector<std::string> names;
                std::string::size_type   pos = 0;
                for(;;)
                {
                    auto next = opts.find(',', pos);
                    if(next == std::string::npos)
                    {
                        names.push_back(trim_opt(opts.substr(pos)));
                        break;
                    }
                    names.push_back(trim_opt(opts.substr(pos, next - pos)));
                    pos = next + 1;
                }
                return names;
            }

            variables_map parse_command_line(int                        argc,
                                             char const* const          argv[],
                                             options_description const& desc)
            {
                variables_map m_vm;
                auto const&   options = desc.options();

                std::map<std::string, std::pair<size_t, std::string>> arg_to_option;
                for(size_t i = 0; i < options.size(); i++)
                {
                    std::string canonical = get_canonical_option_name(options[i].opts);
                    auto        names     = get_option_names(options[i].opts);
                    for(auto const& name : names)
                    {
                        if(name.empty())
                            continue;
                        std::string key    = (name.size() == 1 ? "-" : "--") + name;
                        arg_to_option[key] = {i, canonical};
                    }
                }

                for(int i = 1; i < argc; i++)
                {
                    std::string arg = argv[i];
                    auto        it  = arg_to_option.find(arg);
                    if(it == arg_to_option.end())
                        throw std::invalid_argument("Unknown option: " + arg);

                    size_t      idx       = it->second.first;
                    std::string canonical = it->second.second;
                    auto const& opt       = options[idx];

                    if(!opt.value_semantic)
                        continue;

                    if(opt.value_semantic->is_flag())
                    {
                        if(opt.value_semantic->has_implicit_value())
                        {
                            // Optional value: --opt means implicit value; --opt false means parse value
                            if(i + 1 < argc && argv[i + 1][0] != '-')
                            {
                                i++;
                                std::any parsed         = opt.value_semantic->parser()(argv[i]);
                                m_vm[canonical].value() = std::move(parsed);
                            }
                            else
                            {
                                m_vm[canonical].value() = opt.value_semantic->get_implicit_value();
                            }
                        }
                        else
                        {
                            m_vm[canonical].value() = std::any(true);
                        }
                        continue;
                    }

                    i++;
                    if(i >= argc)
                        throw std::invalid_argument("Missing value for option " + arg);
                    std::string value_str = argv[i];
                    std::any    parsed    = opt.value_semantic->parser()(value_str);

                    if(opt.value_semantic->is_multivalued() && m_vm.count(canonical)
                       && !m_vm[canonical].empty())
                    {
                        opt.value_semantic->merge_into(m_vm[canonical].value(), parsed);
                    }
                    else
                    {
                        m_vm[canonical].value() = std::move(parsed);
                    }
                }

                for(size_t i = 0; i < options.size(); i++)
                {
                    if(!options[i].value_semantic)
                        continue;
                    std::string canonical = get_canonical_option_name(options[i].opts);
                    if(canonical == "help")
                        continue;
                    // Only set default for options that explicitly have one; otherwise
                    // unspecified options (e.g. convolution-identifier) would get empty
                    // string and count() would be true with an incomplete value.
                    if((!m_vm.count(canonical) || m_vm[canonical].empty())
                       && options[i].value_semantic->has_explicit_default())
                    {
                        m_vm[canonical].value() = options[i].value_semantic->get_default();
                    }
                }
                return m_vm;
            }

            variables_map parse_config_file(std::istream& is, options_description const& desc)
            {
                variables_map                                         m_vm;
                auto const&                                           options = desc.options();
                std::map<std::string, std::pair<size_t, std::string>> name_to_option;
                for(size_t i = 0; i < options.size(); i++)
                {
                    std::string canonical = get_canonical_option_name(options[i].opts);
                    auto        names     = get_option_names(options[i].opts);
                    for(auto const& name : names)
                    {
                        if(name.empty())
                            continue;
                        name_to_option[name] = {i, canonical};
                    }
                }

                std::string line;
                while(std::getline(is, line))
                {
                    auto hash = line.find('#');
                    if(hash != std::string::npos)
                        line.resize(hash);
                    line = trim_opt(line);
                    if(line.empty())
                        continue;

                    auto eq = line.find('=');
                    if(eq == std::string::npos)
                        continue;

                    std::string key   = trim_opt(line.substr(0, eq));
                    std::string value = trim_opt(line.substr(eq + 1));
                    if(key.empty())
                        continue;

                    auto it = name_to_option.find(key);
                    if(it == name_to_option.end())
                        continue;

                    size_t      idx       = it->second.first;
                    std::string canonical = it->second.second;
                    auto const& opt       = options[idx];

                    if(!opt.value_semantic)
                        continue;

                    std::any parsed = opt.value_semantic->parse_config_line(value);
                    if(opt.value_semantic->is_multivalued() && m_vm.count(canonical)
                       && !m_vm[canonical].empty())
                    {
                        opt.value_semantic->merge_into(m_vm[canonical].value(), parsed);
                    }
                    else
                    {
                        m_vm[canonical].value() = std::move(parsed);
                    }
                }
                return m_vm;
            }

            std::ostream& operator<<(std::ostream& os, options_description const& desc)
            {
                for(auto const& opt : desc.options())
                {
                    auto               names = get_option_names(opt.opts);
                    std::ostringstream left;
                    bool               first = true;
                    for(auto const& name : names)
                    {
                        if(!first)
                            left << "|";
                        left << (name.size() == 1 ? "-" : "--") << name;
                        first = false;
                    }
                    bool print_value = true;
                    if(opt.value_semantic && opt.value_semantic->is_flag())
                        print_value = false;
                    if(print_value)
                        left << " <value>";
                    os << std::setw(34) << std::left << left.str() << " " << std::setw(82)
                       << std::left << opt.description << "\n";
                }
                return os << std::flush;
            }

        } // namespace po
    } // namespace Client
} // namespace Tensile
