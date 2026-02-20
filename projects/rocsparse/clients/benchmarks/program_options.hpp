/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
// This emulates the required functionality of boost::program_options

#pragma once

#include <cinttypes>
#include <cstdio>
#include <iomanip>
#include <ostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Regular expression for token delimiters (whitespace and commas)
static const std::regex program_options_regex{"[, \\f\\n\\r\\t\\v]+",
                                              std::regex_constants::optimize};

// variables_map is a set of seen options
using variables_map = std::set<std::string>;

// Polymorphic base class to use with dynamic_cast
class value_base
{
protected:
    bool m_has_default = false;

public:
    bool has_default() const
    {
        return m_has_default;
    }

    virtual ~value_base() = default;
};

// Value parameters
template <typename T>
class value : public value_base
{
    T* m_var; // Pointer to variable to be modified

public:
    // Constructor
    explicit value(T* var)
        : m_var(var)
    {
    }

    // Pointer to variable
    T* get_ptr() const
    {
        return m_var;
    }

    // Allows default_value()
    value* operator->()
    {
        return this;
    }

    // Set default value
    value& default_value(T val)
    {
        *m_var        = std::move(val);
        m_has_default = true;
        return *this;
    }
};

// bool_switch is a value<bool>, which is handled specially
using bool_switch = value<bool>;

class options_description
{
    // desc_option describes a particular option
    class desc_option
    {
        std::string m_opts;
        value_base* m_val;
        std::string m_desc;

    public:
        desc_option& operator=(const desc_option&) = delete;
        // Constructor with options, value and description
        template <typename T>
        desc_option(std::string opts, value<T> val, std::string desc)
            : m_opts(std::move(opts))
            , m_val(new auto(std::move(val)))
            , m_desc(std::move(desc))
        {
        }

        // Constructor with options and description
        desc_option(std::string opts, std::string desc)
            : m_opts(std::move(opts))
            , m_val(nullptr)
            , m_desc(std::move(desc))
        {
        }

        // Copy constructor is deleted
        desc_option(const desc_option&) = delete;

        // Move constructor
        desc_option(desc_option&& other)
            : m_opts(std::move(other.m_opts))
            , m_val(other.m_val)
            , m_desc(std::move(other.m_desc))
        {
            other.m_val = nullptr;
        }

        // Destructor
        ~desc_option()
        {
            delete m_val;
        }

        // Accessors
        const std::string& get_opts() const
        {
            return m_opts;
        }

        const value_base* get_val() const
        {
            return m_val;
        }

        const std::string& get_desc() const
        {
            return m_desc;
        }

        // Set a value
        void set_val(int& argc, char**& argv) const
        {
            // We test all supported types with dynamic_cast and parse accordingly
            bool match = false;
            if(dynamic_cast<value<int32_t>*>(m_val))
            {
                auto* val = dynamic_cast<value<int32_t>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, "%" SCNd32, val) == 1;
            }
            else if(dynamic_cast<value<uint32_t>*>(m_val))
            {
                auto* val = dynamic_cast<value<uint32_t>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, "%" SCNu32, val) == 1;
            }
            else if(dynamic_cast<value<int64_t>*>(m_val))
            {
                auto* val = dynamic_cast<value<int64_t>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, "%" SCNd64, val) == 1;
            }
            else if(dynamic_cast<value<uint64_t>*>(m_val))
            {
                auto* val = dynamic_cast<value<uint64_t>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, "%" SCNu64, val) == 1;
            }
            else if(dynamic_cast<value<float>*>(m_val))
            {
                auto* val = dynamic_cast<value<float>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, "%f", val) == 1;
            }
            else if(dynamic_cast<value<double>*>(m_val))
            {
                auto* val = dynamic_cast<value<double>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, "%lf", val) == 1;
            }
            else if(dynamic_cast<value<char>*>(m_val))
            {
                auto* val = dynamic_cast<value<char>*>(m_val)->get_ptr();
                match     = argc && sscanf(*argv, " %c", val) == 1;
            }
            else if(dynamic_cast<value<bool>*>(m_val))
            {
                // We handle bool specially, setting the value to true without argument
                auto* val = dynamic_cast<value<bool>*>(m_val)->get_ptr();
                *val      = true;
                return;
            }
            else if(dynamic_cast<value<std::string>*>(m_val))
            {
                if(argc)
                {
                    *dynamic_cast<value<std::string>*>(m_val)->get_ptr() = *argv;
                    match                                                = true;
                }
            }
            else
            {
                throw std::logic_error("Internal error: Unsupported data type");
            }

            if(!match)
                throw std::invalid_argument(argc ? *argv : "Missing required argument");

            // Skip past the argument's value
            ++argv;
            --argc;
        }
    };

    // Description and option list
    std::string              m_desc;
    std::vector<desc_option> m_optlist;

    // desc_optionlist allows chains of options to be parenthesized
    class desc_optionlist
    {
        std::vector<desc_option>& m_list;

    public:
        explicit desc_optionlist(std::vector<desc_option>& list)
            : m_list(list)
        {
        }

        template <typename... Ts>
        desc_optionlist operator()(Ts&&... arg)
        {
            m_list.push_back(desc_option(std::forward<Ts>(arg)...));
            return *this;
        }
    };

public:
    // Constructor
    explicit options_description(std::string desc)
        : m_desc(std::move(desc))
    {
    }

    // Start a desc_optionlist chain
    desc_optionlist add_options() &
    {
        return desc_optionlist(m_optlist);
    }

    // Parse an option at the current (argc, argv) position
    void parse_option(int& argc, char**& argv, variables_map& vm, bool ignoreUnknown = false) const
    {
        // Iterate across all options
        for(const auto& opt : m_optlist)
        {
            // Canonical name used for map
            std::string canonical_name;

            // Iterate across tokens in the opts
            for(std::sregex_token_iterator tok{
                    opt.get_opts().begin(), opt.get_opts().end(), program_options_regex, -1};
                tok != std::sregex_token_iterator();
                ++tok)
            {
                // The first option in a list of options is the canonical name
                if(!canonical_name.length())
                    canonical_name = tok->str();

                // If the length of the option is 1, it is single-dash; otherwise double-dash
                const char* prefix = tok->length() == 1 ? "-" : "--";

                // If option matches
                if(*argv == prefix + tok->str())
                {
                    ++argv;
                    --argc;

                    // If option has a value, set it; otherwise indicate option in set
                    if(opt.get_val())
                        opt.set_val(argc, argv);
                    else
                        vm.insert(canonical_name);
                    return; // Return successfully
                }
            }
        }

        // No options were matched
        if(ignoreUnknown)
        {
            ++argv;
            --argc;
        }
        else
        {
            throw std::invalid_argument(*argv);
        }
    }

    // Formatted output of command-line arguments description
    friend std::ostream& operator<<(std::ostream& os, const options_description& d)
    {
        // Iterate across all options
        for(const auto& opt : d.m_optlist)
        {
            bool               first = true;
            const char*        delim = "";
            std::ostringstream left;

            // Iterate across tokens in the opts
            for(std::sregex_token_iterator tok{opt.get_opts().begin(),
                                               opt.get_opts().end(),
                                               program_options_regex,
                                               -1};
                tok != std::sregex_token_iterator();
                ++tok, first = false, delim = " ")
            {
                // If the length of the option is 1, it is single-dash; otherwise double-dash
                const char* prefix = tok->length() == 1 ? "-" : "--";
                left << delim << (first ? "" : "[ ") << prefix << tok->str() << (first ? "" : " ]");
            }

            // Print the default value of the variable type if it exists
            // We do not print the default value for bool
            const value_base* val = opt.get_val();
            if(val && !dynamic_cast<const value<bool>*>(val))
            {
                left << " arg";
                if(val->has_default())
                {
                    // We test all supported types with dynamic_cast and print accordingly
                    left << " (=";
                    if(dynamic_cast<const value<int32_t>*>(val))
                        left << *dynamic_cast<const value<int32_t>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<uint32_t>*>(val))
                        left << *dynamic_cast<const value<uint32_t>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<int64_t>*>(val))
                        left << *dynamic_cast<const value<int64_t>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<uint64_t>*>(val))
                        left << *dynamic_cast<const value<uint64_t>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<float>*>(val))
                        left << *dynamic_cast<const value<float>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<double>*>(val))
                        left << *dynamic_cast<const value<double>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<char>*>(val))
                        left << *dynamic_cast<const value<char>*>(val)->get_ptr();
                    else if(dynamic_cast<const value<std::string>*>(val))
                        left << *dynamic_cast<const value<std::string>*>(val)->get_ptr();
                    else
                        throw std::logic_error("Internal error: Unsupported data type");
                    left << ")";
                }
            }
            os << std::setw(36) << std::left << left.str() << " " << opt.get_desc() << "\n\n";
        }
        return os << std::flush;
    }
};

// Class representing command line parser
class parse_command_line
{
    variables_map m_vm;

public:
    parse_command_line(int                        argc,
                       char**                     argv,
                       const options_description& desc,
                       bool                       ignoreUnknown = false)
    {
        ++argv; // Skip argv[0]
        --argc;
        while(argc)
            desc.parse_option(argc, argv, m_vm, ignoreUnknown);
    }

    // Copy the variables_map
    friend void store(const parse_command_line& p, variables_map& vm)
    {
        vm = p.m_vm;
    }

    // Move the variables_map
    friend void store(parse_command_line&& p, variables_map& vm)
    {
        vm = std::move(p.m_vm);
    }
};

// We can define the notify() function as a no-op for our purposes
inline void notify(const variables_map&) {}
