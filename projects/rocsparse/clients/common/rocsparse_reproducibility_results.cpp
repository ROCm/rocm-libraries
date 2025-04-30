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

#include "rocsparse_reproducibility.hpp"
#include <iostream>
#include <list>
#include <vector>

class json_file
{
protected:
    std::string    m_filename;
    std::ofstream* m_fout;
    mutable size_t current_line{0};
    mutable size_t newline{1};
    uint16_t       m_num_prefix = 0;

    void config()
    {
        os().precision(15);
        os().setf(std::ios::scientific);
    }

public:
    json_file(const std::string& filename)
    {
        this->open(filename);
        this->config();
    };

    void open(const std::string& filename)
    {
        this->m_filename = filename;
        this->m_fout     = new std::ofstream(filename.c_str());
        this->config();
        this->m_num_prefix = 0;
    }

    void close()
    {
        if(this->m_fout != nullptr)
        {
            this->m_fout->close();
            delete this->m_fout;
            this->m_fout = nullptr;
        }
    }

    json_file()
        : m_filename("")
    {
        this->config();
    };

    virtual ~json_file()
    {
        this->close();
    };

    std::ostream& os() const
    {
        return (this->m_fout) ? this->m_fout[0] : std::cout;
    }

protected:
    virtual void headline() const
    {
        for(uint16_t i = 0; i < this->m_num_prefix; ++i)
        {
            os() << "   ";
        }
    }

public:
    void increase_prefix()
    {
        ++this->m_num_prefix;
    }

    void decrease_prefix()
    {
        --this->m_num_prefix;
    }

    template <typename _type>
    json_file const& operator<<(const _type& type_) const
    {
        if(current_line < newline)
        {
            this->headline();
            ++current_line;
        }
        os() << type_;
        return *this;
    };

    json_file const& operator<<(std::ostream& (*F)(std::ostream&)) const
    {
        ++newline;
        os() << F;
        return *this;
    };

    static json_file& instance();
};

json_file& json_file::instance()
{
    static json_file s_instance;
    return s_instance;
}

#define rocsparse_json_exporter json_file::instance()

static std::ostream& operator<<(std::ostream& out, const rocsparse_reproducibility_t::status_t s)
{
    switch(s)
    {
    case rocsparse_reproducibility_t::status_t::non_tested:
    {
        out << "non_tested";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::non_executed:
    {
        out << "non_executed";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::irrelevant:
    {
        out << "irrelevant";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::reproducible:
    {
        out << "reproducible";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::non_reproducible:
    {
        out << "non_reproducible";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::execution_failed:
    {
        out << "execution_failed";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::test_failed:
    {
        out << "test_failed";
        return out;
    }
    case rocsparse_reproducibility_t::status_t::NVALUES:
    {
        break;
    }
    }
    out << "unknown";
    return out;
}

static json_file& operator<<(json_file&                                         out,
                             const rocsparse_reproducibility_t::results_test_t& test_result)
{
    out << "{" << std::endl;
    out.increase_prefix();
    out << " "
        << "\"name\": \"" << test_result.get_name() << "\"," << std::endl;
    out << " "
        << "\"hash\": \"" << test_result.get_hash() << "\"," << std::endl;
    out << " "
        << "\"status\": \"" << test_result.get_status() << "\"," << std::endl;
    out << " "
        << "\"description\": \"" << test_result.get_description() << "\"" << std::endl;
    out.decrease_prefix();
    out << "}";
    return out;
}

static json_file& operator<<(json_file& out, const rocsparse_reproducibility_t::config_t& config)
{
    out << "{" << std::endl;
    out.increase_prefix();
    out << "\"date\": \"" << config.get_date() << "\"," << std::endl;
    out << "\"command\": \"" << config.get_command() << "\"," << std::endl;
    out << "\"gpu\": \"" << config.get_gpu_name() << "\"," << std::endl;
    out << "\"num_iterations\": \"" << config.get_num_iterations() << "\"" << std::endl;
    out.decrease_prefix();
    out << "}";
    return out;
}

static json_file& operator<<(json_file&                                               out,
                             const rocsparse_reproducibility_t::results_test_input_t& test_input)
{
    out << "{" << std::endl;
    bool next = false;
    out.increase_prefix();
    for(auto& p : test_input)
    {
        if(next)
        {
            out << "," << std::endl;
        }
        out << "\"" << p.first << "\" : \"" << p.second << "\"";
        next = true;
    }
    out.decrease_prefix();
    out << std::endl << "}";
    return out;
}

static void
    compute_combinations(uint64_t n_size, const uint64_t* n, uint64_t N, uint64_t* p, uint64_t ld)
{
    uint64_t s  = N / n[0];
    uint64_t at = 0;
    while(at < N)
    {
        for(uint64_t i = 0; i < n[0]; ++i)
        {
            for(uint64_t j = 0; j < s; ++j)
            {
                p[0 + ld * at] = i;
                ++at;
            }
        }
    }

    for(uint64_t k = 1; k < n_size; ++k)
    {
        s  = s / n[k];
        at = 0;
        while(at < N)
        {
            for(uint64_t i = 0; i < n[k]; ++i)
            {
                for(uint64_t j = 0; j < s; ++j)
                {
                    p[k + ld * at] = i;
                    ++at;
                }
            }
        }
    }
}

const char* get_name(const char* enum_type, uint64_t index);
bool        get_size(uint64_t& v, const char* enum_type);

json_file& operator<<(
    json_file&                                                        out,
    const rocsparse_reproducibility_t::results_test_classification_t& test_classification)
{
    out.increase_prefix();
    out << "{" << std::endl;
    out << " \"input\" : " << std::endl;
    out.increase_prefix();
    out << test_classification.get_input();
    out.decrease_prefix();
    out << "," << std::endl;
    const auto status = test_classification.get_status();
    out << " \"status\" : \"" << status << "\"," << std::endl;
    out << " \"num_tests\" : \"" << test_classification.size() << "\"";
    out << "," << std::endl;
    out << " \"stats\" : [" << std::endl;
    out.increase_prefix();
    {
        bool next  = false;
        int* count = new int[(int)rocsparse_reproducibility_t::status_t::NVALUES];
        for(auto i = 0; i < (int)rocsparse_reproducibility_t::status_t::NVALUES; ++i)
            count[i] = 0;
        for(const auto& test_result : test_classification)
        {
            count[(int)test_result.get_status()] += 1;
        }
        for(auto i = 0; i < (int)rocsparse_reproducibility_t::status_t::NVALUES; ++i)
        {
            if(count[i] > 0)
            {
                if(next)
                {
                    out << "," << std::endl;
                }
                out << "\"" << ((rocsparse_reproducibility_t::status_t)i) << "\" : \"" << count[i]
                    << "\"";
                next = true;
            }
        }
        delete[] count;
    }
    out.decrease_prefix();
    out << std::endl << " ]";
    if(rocsparse_reproducibility_t::instance().config().get_info_level() < 2)
    {
        out << "," << std::endl;
        out << " \"tests\" : [" << std::endl;
        out.increase_prefix();
        bool next = false;
        for(const auto& test_result : test_classification)
        {
            if(next)
            {
                out << "," << std::endl;
            }
            out << test_result;
            next = true;
        }
        out.decrease_prefix();
        out << " ]" << std::endl;
    }
    else
    {
        out << std::endl;
    }
    out << "}";
    out.decrease_prefix();
    return out;
}

json_file& operator<<(json_file&                                                 out,
                      const rocsparse_reproducibility_t::results_test_routine_t& test_routine)
{
    out << "{" << std::endl;
    out << " \"name\" : \"" << test_routine.get_name() << "\"," << std::endl;
    out << " \"classification_size\" : \"" << test_routine.size() << "\"," << std::endl;
    if(rocsparse_reproducibility_t::instance().config().get_info_level() < 1)
    {
        out << " \"coverage\" : \"" << test_routine.get_coverage() << "\"," << std::endl;
    }
    out << " \"classification\" : [";
    out << std::endl;
    bool next = false;
    for(const auto& it : test_routine)
    {
        if(next)
        {
            out << "," << std::endl;
        }
        const auto& test_classification = it.second;
        out << test_classification;
        next = true;
    }
    out << std::endl;
    out << " ]" << std::endl;
    out << "}";
    return out;
}

json_file& operator<<(json_file& out, const rocsparse_reproducibility_t::results_t& results)
{
    out << "{" << std::endl;
    out.increase_prefix();
    out << "\"routines_size\" : \"" << results.size() << "\"," << std::endl;
    out << "\"routines\" : [" << std::endl;
    out.increase_prefix();
    bool next_results = false;
    for(auto& r : results)
    {
        const auto& test_routine = r.second;
        if(next_results)
        {
            out << "," << std::endl;
        }
        out << test_routine;
        next_results = true;
    }
    out << std::endl;
    out << "]" << std::endl;
    out.decrease_prefix();
    out << "}" << std::endl;
    out.decrease_prefix();
    return out;
}

void rocsparse_reproducibility_update(
    rocsparse_reproducibility_t::results_test_input_t&   test_input,
    rocsparse_reproducibility_t::results_test_routine_t& test_routine)
{
    const uint64_t        ninputs = test_input.size();
    std::vector<uint64_t> sizes(ninputs);
    uint64_t              num_combinations = 1;
    for(uint64_t i = 0; i < ninputs; ++i)
    {
        uint64_t s;
        bool     found = get_size(s, test_input[i].first.c_str());
        if(found == false)
        {
            std::cerr << "not found " << test_input[i].first;
            exit(1);
        }
        num_combinations *= s;
        sizes[i] = s;
    }

    uint64_t* combinations = new uint64_t[num_combinations * ninputs];
    compute_combinations(ninputs, sizes.data(), num_combinations, combinations, ninputs);

    rocsparse_reproducibility_t::results_test_input_t input;
    input.resize(ninputs);
    uint64_t count_p = 0;
    uint64_t count_n = 0;
    for(uint64_t i = 0; i < num_combinations; ++i)
    {
        for(uint64_t j = 0; j < ninputs; ++j)
        {
            input[j].first  = test_input[j].first;
            input[j].second = get_name(test_input[j].first.c_str(), combinations[ninputs * i + j]);
        }
        auto ss = input.to_string();
        if(test_routine.find(ss) != test_routine.end())
        {
            count_p++;
        }
        else
        {
            test_routine[ss].set_input(input);
            test_routine[ss].set_status(rocsparse_reproducibility_t::status_t::non_executed);
            count_n++;
        }
        //
        //
        //
    }
    delete[] combinations;

    test_routine.set_coverage((double(count_p) / double(num_combinations)));
#if NDEBUG
    std::cout << "// rocsparse_reproducibility covered     " << count_p << std::endl;
    std::cout << "// rocsparse_reproducibility non covered " << count_n << std::endl;
    std::cout << "// rocsparse_reproducibility coverage "
              << ((double(count_p) / double(num_combinations)) * 100) << std::endl;
#endif
}

void rocsparse_reproducibility_update(
    rocsparse_reproducibility_t::results_test_routine_t& test_routine)
{
    for(auto& it : test_routine)
    {
        auto& test_classification = it.second;
        rocsparse_reproducibility_update(test_classification.get_input(), test_routine);
        break;
    }
}

void rocsparse_reproducibility_update(rocsparse_reproducibility_t::results_t& results)
{
    for(auto& r : results)
    {
        auto& test_routine = r.second;
        rocsparse_reproducibility_update(test_routine);
    }
}

void rocsparse_reproducibility_update(rocsparse_reproducibility_t& reproducibility)
{
    rocsparse_reproducibility_update(reproducibility.results());
}

void rocsparse_reproducibility_write_report(rocsparse_reproducibility_t& reproducibility)
{
    if(rocsparse_reproducibility_t::instance().config().get_info_level() < 1)
    {
        rocsparse_reproducibility_update(reproducibility);
    }

    json_file out(reproducibility.config().get_filename());
    out << "{" << std::endl;
    out << " \"config\" : " << std::endl;
    out.increase_prefix();
    out << reproducibility.config();
    out.decrease_prefix();
    out << "," << std::endl;
    out << " \"results\" : " << std::endl;
    out.increase_prefix();
    out << reproducibility.results();
    out.decrease_prefix();
    out << "}" << std::endl;
    out.close();
}

rocsparse_reproducibility_t::results_t::~results_t() {}

void rocsparse_reproducibility_t::results_t::info() {}

void rocsparse_reproducibility_t::results_t::add(const char*                 routine_name,
                                                 const results_test_input_t& test_input,
                                                 const results_test_t&       test_result)
{
    auto it = this->find(routine_name);
    if(it == this->end())
    {
        (*this)[routine_name].set_name(routine_name);
    }
    (*this)[routine_name].add(test_input, test_result);
}
