/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include <fstream>
#include <list>
#include <map>
#include <string>
#include <vector>

struct rocsparse_reproducibility_t
{
private:
    rocsparse_reproducibility_t() = default;

public:
    static rocsparse_reproducibility_t& instance()
    {
        static rocsparse_reproducibility_t s_instance;
        return s_instance;
    }

    ///
    /// @brief Enumeration to describe the reproducibility status.
    ///
    enum class status_t
    {
        /// @brief Describes a non-tested routine for reproducibility.
        non_tested = 0,
        /// @brief Describes a non-executed configuration for reproducibility.
        non_executed,
        /// @brief Describes an irrelevant routine regarding the reproducibility.
        irrelevant,
        /// @brief Describes a reproducible routine.
        reproducible,
        /// @brief Describes a non reproducible routine.
        non_reproducible,
        /// @brief Describes an execution error.
        execution_failed,
        /// @brief Describes a testing error.
        test_failed,
        NVALUES
    };

    ///
    /// @brief Enable reproducibility.
    ///
    void enable();

    ///
    /// @brief Check if reproducibility is enabled.
    /// @return True if reproducibility is enabled, false otherwise.
    ///
    bool is_enabled() const;

    ///
    /// @brief Set the number of iterations of the reproducibility testing.
    /// @param[in] niter number of iterations.
    ///
    void set_num_iterations(uint32_t niter);

    ///
    /// @brief Get the number of iterations of the reproducibility testing.
    /// @return The number of iterations.
    ///
    uint32_t get_num_iterations() const;

    struct config_t
    {

    private:
        static constexpr uint32_t s_default_num_iterations = 10;

        bool        m_enabled{};
        uint32_t    m_num_iterations{s_default_num_iterations};
        uint32_t    m_info_level{};
        std::string m_command{};
        std::string m_date{__DATE__};
        std::string m_filename{"rocsparse_reproducibility.json"};
        std::string m_gpu_name{};

    public:
        config_t() {}

        bool is_enabled() const
        {
            return this->m_enabled;
        }

        void enable()
        {
            this->m_enabled = true;
        }

        void set_info_level(uint32_t level)
        {
            this->m_info_level = level;
        }
        void set_filename(const std::string& filename)
        {
            this->m_filename = filename;
        }
        void set_command(const std::string& command)
        {
            this->m_command = command;
        }
        void set_date(const std::string& date)
        {
            this->m_date = date;
        }
        void set_gpu_name(const std::string& gpu_name)
        {
            this->m_gpu_name = gpu_name;
        }
        const std::string& get_filename() const
        {
            return this->m_filename;
        }
        const std::string& get_command() const
        {
            return this->m_command;
        }
        const std::string& get_date() const
        {
            return this->m_date;
        }
        const std::string& get_gpu_name() const
        {
            return this->m_gpu_name;
        }
        uint32_t get_info_level() const
        {
            return this->m_info_level;
        }

        void set_num_iterations(uint32_t num_iterations)
        {
            this->m_num_iterations = num_iterations;
        }

        uint32_t get_num_iterations() const
        {

            return this->m_num_iterations;
        }
    };

    ///
    /// @brief Data container for a reproducibility tests.
    ///
    struct test_data_t
    {

    private:
        static constexpr uint64_t s_maxsize = 32;
        uint32_t                  m_num_objects{};
        uint64_t                  m_object_numbytes[s_maxsize]{};
        void*                     m_objects[s_maxsize]{};
        std::string               m_object_names[s_maxsize]{};

    public:
        test_data_t()  = default;
        ~test_data_t() = default;

        void reset();

        uint32_t    get_num_objects() const;
        const char* get_object_name(uint32_t object_index) const;
        const void* get_object(uint32_t object_index) const;
        uint64_t    get_object_numbytes(uint32_t object_index) const;

        ///
        ///
        ///
        void* add(const std::string& name, uint64_t numbytes);

        ///
        ///
        ///
        bool     is_same(const test_data_t*, std::ostream& descr) const;
        uint64_t compute_hash() const;
    };

    //
    //
    //
    struct test_t
    {
    private:
        test_data_t* m_datas[2]{nullptr, nullptr};
        bool         m_next_call{};

    public:
        test_t() = default;
        ~test_t()
        {
            reset();
        }

        const test_data_t* get_current_data() const;
        test_data_t*       get_current_data();
        const test_data_t* get_initial_data() const;
        test_data_t*       get_initial_data();
        void               reset();
        void               reset_next();
        void               set_next();
        bool               is_next() const;
    };

    struct results_test_input_t : std::vector<std::pair<std::string, std::string>>
    {
    public:
        results_test_input_t() = default;

        void add(const std::string& a, const std::string& b)
        {
            this->push_back(std::pair<std::string, std::string>(a, b));
        }

        std::string to_string() const
        {
            bool        next = false;
            std::string all;
            for(const auto& p : *this)
            {
                if(next)
                {
                    all += ",";
                }
                all += "\"" + p.first + "\" : \"" + p.second + "\"";
                next = true;
            }
            return all;
        }
    };

    struct results_test_t
    {
    private:
        std::string                           m_name{};
        rocsparse_reproducibility_t::status_t m_status{};
        uint64_t                              m_hash{};
        std::string                           m_description{};

    public:
        results_test_t()  = default;
        ~results_test_t() = default;
        const std::string& get_name() const
        {
            return this->m_name;
        }
        void set_name(const std::string& name)
        {
            this->m_name = name;
        }
        const std::string& get_description() const
        {
            return this->m_description;
        }
        void set_description(const std::string& description)
        {
            this->m_description = description;
        }
        uint64_t get_hash() const
        {
            return this->m_hash;
        }
        void set_hash(uint64_t h)
        {
            this->m_hash = h;
        }
        rocsparse_reproducibility_t::status_t get_status() const
        {
            return this->m_status;
        }
        void set_status(rocsparse_reproducibility_t::status_t status)
        {
            this->m_status = status;
        }
    };

    struct results_test_classification_t : std::list<results_test_t>
    {

    private:
        results_test_input_t                  m_input;
        rocsparse_reproducibility_t::status_t m_status;

    public:
        void set_status(rocsparse_reproducibility_t::status_t status)
        {
            this->m_status = status;
        }

        rocsparse_reproducibility_t::status_t get_status() const
        {
            if(this->size() == 0)
            {
                return this->m_status;
            }
            else
            {
                rocsparse_reproducibility_t::status_t s
                    = rocsparse_reproducibility_t::status_t::reproducible;
                for(const auto& t : *this)
                {
                    auto st = t.get_status();
                    if(st != rocsparse_reproducibility_t::status_t::reproducible)
                    {
                        s = st;
                        break;
                    }
                }

                return s;
            }
        }

        const results_test_input_t& get_input() const
        {
            return this->m_input;
        }

        results_test_input_t& get_input()
        {
            return this->m_input;
        }
        void set_input(const results_test_input_t& input)
        {
            this->m_input = input;
        }
        void set(const results_test_input_t& input, const results_test_t& a)
        {
            this->m_input = input;
            this->push_back(a);
        }

        std::string to_string() const
        {
            return this->m_input.to_string();
        }

        void add(const results_test_t& test_result)
        {
            (*this).push_back(test_result);
        }
    };

    struct results_test_routine_t : std::map<std::string, results_test_classification_t>
    {
    protected:
        std::string m_name{};
        double      m_coverage{};

    public:
        const std::string& get_name() const
        {
            return this->m_name;
        }
        void set_name(const std::string& name)
        {
            this->m_name = name;
        }

        double get_coverage() const
        {
            return this->m_coverage;
        }
        void set_coverage(double coverage)
        {
            this->m_coverage = coverage;
        }

        rocsparse_reproducibility_t::status_t get_status() const
        {
            rocsparse_reproducibility_t::status_t s
                = rocsparse_reproducibility_t::status_t::reproducible;
            for(const auto& p : *this)
            {
                auto st = p.second.get_status();
                if(st != rocsparse_reproducibility_t::status_t::reproducible)
                {
                    s = st;
                    break;
                }
            }
            return s;
        }

        void add(const results_test_input_t& input, const results_test_t& a)
        {
            const auto s  = input.to_string();
            auto       it = this->find(s);
            if(it == this->end())
            {
                (*this)[s].set(input, a);
            }
            else
            {
                (*this)[s].push_back(a);
            }
        }
    };

    struct results_t : std::map<std::string, results_test_routine_t>
    {
    public:
        results_t() = default;
        ~results_t();

        void add(const char*                 routine_name,
                 const results_test_input_t& test_input,
                 const results_test_t&       test_result);

        void info();
    };

    const test_t& test() const
    {
        return this->m_test;
    }
    test_t& test()
    {
        return this->m_test;
    }

    const results_t& results() const
    {
        return this->m_results;
    }
    results_t& results()
    {
        return this->m_results;
    }

    const config_t& config() const
    {
        return this->m_config;
    }
    config_t& config()
    {
        return this->m_config;
    }

private:
    test_t    m_test;
    results_t m_results;
    config_t  m_config;
};

///
/// @brief Write report.
///
void rocsparse_reproducibility_write_report(rocsparse_reproducibility_t& self);
