/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_clients_envariables.hpp"
#include "rocsparse-types.h"
#include <iostream>
#include <mutex>
constexpr rocsparse_clients_envariables::var_bool rocsparse_clients_envariables::s_var_bool_all[];
constexpr rocsparse_clients_envariables::var_string
    rocsparse_clients_envariables::s_var_string_all[];

template <std::size_t N, typename T>
static inline constexpr std::size_t countof(T (&)[N])
{
    return N;
}

static constexpr size_t s_var_bool_size = countof(rocsparse_clients_envariables::s_var_bool_all);
static constexpr size_t s_var_string_size
    = countof(rocsparse_clients_envariables::s_var_string_all);

static constexpr const char* s_var_bool_names[s_var_bool_size]
    = {"ROCSPARSE_CLIENTS_VERBOSE", "ROCSPARSE_CLIENTS_TEST_DEBUG_ARGUMENTS"};
static constexpr const char* s_var_string_names[s_var_string_size]
    = {"ROCSPARSE_CLIENTS_MATRICES_DIR", "ROCSPARSE_TEST_DATA"};
static constexpr const char* s_var_bool_descriptions[s_var_bool_size]
    = {"0: disabled, 1: enabled", "0: disabled, 1: enabled"};
static constexpr const char* s_var_string_descriptions[s_var_string_size]
    = {"Full path of the matrices directory", "The path where the test data file is located"};

///
/// @brief Grab an environment variable value.
/// @return true if the operation is successful, false otherwise.
///
template <typename T>
static bool rocsparse_getenv(const char* name, bool& defined, T& val);

template <>
bool rocsparse_getenv<bool>(const char* name, bool& defined, bool& val)
{
    val                    = false;
    const char* getenv_str = getenv(name);
    defined                = (getenv_str != nullptr);
    if(defined)
    {
        auto getenv_int = atoi(getenv_str);
        if((getenv_int != 0) && (getenv_int != 1))
        {
            std::cerr << "rocsparse error, invalid environment variable " << name
                      << " must be 0 or 1." << std::endl;
            val = false;
            return false;
        }
        else
        {
            val = (getenv_int == 1);
            return true;
        }
    }
    else
    {
        return true;
    }
}

template <>
bool rocsparse_getenv<std::string>(const char* name, bool& defined, std::string& val)
{
    const char* getenv_str = getenv(name);
    defined                = (getenv_str != nullptr);
    if(defined)
    {
        val = getenv_str;
    }
    return true;
}

struct rocsparse_clients_envariables_impl
{
    static std::mutex s_mutex;

public:
    //
    // \brief Return value of a Boolean variable.
    //
    inline bool get(rocsparse_clients_envariables::var_bool v) const
    {
        return this->m_var_bool[v];
    };

    //
    // \brief Return value of a string variable.
    //
    inline const char* get(rocsparse_clients_envariables::var_string v) const
    {
        return this->m_var_string[v].c_str();
    };

    //
    // \brief Set value of a Boolean variable.
    //
    inline void set(rocsparse_clients_envariables::var_bool v, bool value)
    {

        rocsparse_clients_envariables_impl::s_mutex.lock();
        this->m_var_bool[v] = value;
        rocsparse_clients_envariables_impl::s_mutex.unlock();
    };

    inline void set(rocsparse_clients_envariables::var_string v, const char* value)
    {
        rocsparse_clients_envariables_impl::s_mutex.lock();
        this->m_var_string[v] = value;
        rocsparse_clients_envariables_impl::s_mutex.unlock();
    };

    //
    // \brief Return value of a string variable.
    //
    //
    // \brief Is a Boolean variable defined ?
    //
    inline bool is_defined(rocsparse_clients_envariables::var_bool v) const
    {
        return this->m_var_bool_defined[v];
    };

    //
    // \brief Is a string variable defined ?
    //
    inline bool is_defined(rocsparse_clients_envariables::var_string v) const
    {
        return this->m_var_string_defined[v];
    };

    //
    // Return the unique instance.
    //
    static rocsparse_clients_envariables_impl& Instance();

private:
    ~rocsparse_clients_envariables_impl()                                         = default;
    rocsparse_clients_envariables_impl(const rocsparse_clients_envariables_impl&) = delete;
    rocsparse_clients_envariables_impl& operator=(const rocsparse_clients_envariables_impl&)
        = delete;

    bool m_var_bool[s_var_bool_size]{};
    bool m_var_bool_defined[s_var_bool_size]{};

    std::string m_var_string[s_var_string_size]{};
    bool        m_var_string_defined[s_var_string_size]{};

    rocsparse_clients_envariables_impl()
    {
        for(auto tag : rocsparse_clients_envariables::s_var_bool_all)
        {
            switch(tag)
            {
            case rocsparse_clients_envariables::VERBOSE:
            {
                const bool success = rocsparse_getenv(
                    s_var_bool_names[tag], this->m_var_bool_defined[tag], this->m_var_bool[tag]);
                if(!success)
                {
                    std::cerr << "rocsparse_getenv failed on fetching " << s_var_bool_names[tag]
                              << std::endl;
                    throw(rocsparse_status_invalid_value);
                }
                break;
            }
            case rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS:
            {
                const bool success = rocsparse_getenv(
                    s_var_bool_names[tag], this->m_var_bool_defined[tag], this->m_var_bool[tag]);
                if(!success)
                {
                    std::cerr << "rocsparse_getenv failed on fetching " << s_var_bool_names[tag]
                              << std::endl;
                    throw(rocsparse_status_invalid_value);
                }
                break;
            }
            }
        }

        for(auto tag : rocsparse_clients_envariables::s_var_string_all)
        {
            switch(tag)
            {
            case rocsparse_clients_envariables::MATRICES_DIR:
            {
                const bool success = rocsparse_getenv(s_var_string_names[tag],
                                                      this->m_var_string_defined[tag],
                                                      this->m_var_string[tag]);
                if(!success)
                {
                    std::cerr << "rocsparse_getenv failed on fetching " << s_var_string_names[tag]
                              << std::endl;
                    throw(rocsparse_status_invalid_value);
                }
                break;
            }
            case rocsparse_clients_envariables::TEST_DATA_DIR:
            {
                const bool success = rocsparse_getenv(s_var_string_names[tag],
                                                      this->m_var_string_defined[tag],
                                                      this->m_var_string[tag]);
                if(!success)
                {
                    std::cerr << "rocsparse_getenv failed on fetching " << s_var_string_names[tag]
                              << std::endl;
                    throw(rocsparse_status_invalid_value);
                }
                break;
            }
            }
        }

        if(this->m_var_bool[rocsparse_clients_envariables::VERBOSE])
        {
            for(auto tag : rocsparse_clients_envariables::s_var_bool_all)
            {
                switch(tag)
                {
                case rocsparse_clients_envariables::VERBOSE:
                {
                    const bool v = this->m_var_bool[tag];
                    std::cout << ""
                              << "env variable " << s_var_bool_names[tag] << " : "
                              << ((this->m_var_bool_defined[tag]) ? ((v) ? "enabled" : "disabled")
                                                                  : "<undefined>")
                              << std::endl;
                    break;
                }
                case rocsparse_clients_envariables::TEST_DEBUG_ARGUMENTS:
                {
                    const bool v = this->m_var_bool[tag];
                    std::cout << ""
                              << "env variable " << s_var_bool_names[tag] << " : "
                              << ((this->m_var_bool_defined[tag]) ? ((v) ? "enabled" : "disabled")
                                                                  : "<undefined>")
                              << std::endl;
                    break;
                }
                }
            }

            for(auto tag : rocsparse_clients_envariables::s_var_string_all)
            {
                switch(tag)
                {
                case rocsparse_clients_envariables::MATRICES_DIR:
                {
                    const std::string v = this->m_var_string[tag];
                    std::cout << ""
                              << "env variable " << s_var_string_names[tag] << " : "
                              << ((this->m_var_string_defined[tag]) ? this->m_var_string[tag]
                                                                    : "<undefined>")
                              << std::endl;
                    break;
                }
                case rocsparse_clients_envariables::TEST_DATA_DIR:
                {
                    const std::string v = this->m_var_string[tag];
                    std::cout << ""
                              << "env variable " << s_var_string_names[tag] << " : "
                              << ((this->m_var_string_defined[tag]) ? this->m_var_string[tag]
                                                                    : "<undefined>")
                              << std::endl;
                    break;
                }
                }
            }
        }
    }
};

std::mutex rocsparse_clients_envariables_impl::s_mutex;

rocsparse_clients_envariables_impl& rocsparse_clients_envariables_impl::Instance()
{
    static rocsparse_clients_envariables_impl instance;
    return instance;
}

bool rocsparse_clients_envariables::is_defined(rocsparse_clients_envariables::var_string v)
{
    return rocsparse_clients_envariables_impl::Instance().is_defined(v);
}

const char* rocsparse_clients_envariables::get(rocsparse_clients_envariables::var_string v)
{
    return rocsparse_clients_envariables_impl::Instance().get(v);
}

void rocsparse_clients_envariables::set(rocsparse_clients_envariables::var_string v,
                                        const char*                               value)
{
    rocsparse_clients_envariables_impl::Instance().set(v, value);
}

const char* rocsparse_clients_envariables::get_name(rocsparse_clients_envariables::var_string v)
{
    return s_var_string_names[v];
}

const char*
    rocsparse_clients_envariables::get_description(rocsparse_clients_envariables::var_string v)
{
    return s_var_string_descriptions[v];
}

bool rocsparse_clients_envariables::is_defined(rocsparse_clients_envariables::var_bool v)
{
    return rocsparse_clients_envariables_impl::Instance().is_defined(v);
}

bool rocsparse_clients_envariables::get(rocsparse_clients_envariables::var_bool v)
{
    return rocsparse_clients_envariables_impl::Instance().get(v);
}

void rocsparse_clients_envariables::set(rocsparse_clients_envariables::var_bool v, bool value)
{
    rocsparse_clients_envariables_impl::Instance().set(v, value);
}

const char* rocsparse_clients_envariables::get_name(rocsparse_clients_envariables::var_bool v)
{
    return s_var_bool_names[v];
}

const char*
    rocsparse_clients_envariables::get_description(rocsparse_clients_envariables::var_bool v)
{
    return s_var_bool_descriptions[v];
}
