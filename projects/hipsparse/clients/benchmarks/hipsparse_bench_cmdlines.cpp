#include "hipsparse_bench_cmdlines.hpp"

//
// @brief Get the output filename.
//
const char* hipsparse_bench_cmdlines::get_ofilename() const
{
    return this->m_cmd.get_ofilename();
}

//
// @brief Get the number of samples..
//
int hipsparse_bench_cmdlines::get_nsamples() const
{
    return this->m_cmd.get_nsamples();
};
size_t hipsparse_bench_cmdlines::get_option_index_x() const
{
    return this->m_cmd.get_option_index_x();
};

int hipsparse_bench_cmdlines::get_option_nargs(int i)
{
    return this->m_cmd.get_option_nargs(i);
}
const char* hipsparse_bench_cmdlines::get_option_arg(int i, int j)
{
    return this->m_cmd.get_option_arg(i, j);
}
const char* hipsparse_bench_cmdlines::get_option_name(int i)
{
    return this->m_cmd.get_option_name(i);
}
int hipsparse_bench_cmdlines::get_noptions_x() const
{
    return this->m_cmd.get_noptions_x();
};
int hipsparse_bench_cmdlines::get_noptions() const
{
    return this->m_cmd.get_noptions();
};
bool hipsparse_bench_cmdlines::is_stdout_disabled() const
{
    return this->m_cmd.is_stdout_disabled();
};
bool hipsparse_bench_cmdlines::no_rawdata() const
{
    return this->m_cmd.no_rawdata();
};

//
// @brief Get the number of runs per sample.
//
int hipsparse_bench_cmdlines::get_nruns() const
{
    return this->m_cmd.get_nruns();
};

//
// @brief Copy the command line arguments corresponding to a given sample.
//
void hipsparse_bench_cmdlines::get(int isample, int& argc, char** argv) const
{
    const auto& cmdsample = this->m_cmdset[isample];
    for(int j = 0; j < cmdsample.argc; ++j)
    {
        argv[j] = cmdsample.argv[j];
    }
    argc = cmdsample.argc;
}

void hipsparse_bench_cmdlines::get_argc(int isample, int& argc_) const
{
    argc_ = this->m_cmdset[isample].argc;
}

hipsparse_bench_cmdlines::~hipsparse_bench_cmdlines()
{
    if(this->m_cmdset != nullptr)
    {
        delete[] this->m_cmdset;
        this->m_cmdset = nullptr;
    }
}

//
// @brief Constructor.
//
hipsparse_bench_cmdlines::hipsparse_bench_cmdlines(int argc, char** argv)
    : m_cmd(argc, argv)
{
    //
    // Expand the command line .
    //
    this->m_cmdset = new val[this->m_cmd.get_nsamples()];
    this->m_cmd.expand(this->m_cmdset);
}

bool hipsparse_bench_cmdlines::applies(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        if(!strcmp(argv[i], "--bench-x"))
        {
            return true;
        }
    }
    return false;
}

void hipsparse_bench_cmdlines::info() const
{
    int nsamples = this->m_cmd.get_nsamples();
    for(int isample = 0; isample < nsamples; ++isample)
    {
        const auto& cmdsample = this->m_cmdset[isample];
        const auto  argc      = cmdsample.argc;
        const auto  argv      = cmdsample.argv;
        std::cout << "sample[" << isample << "/" << nsamples << "], argc = " << argc << std::endl;

        for(int jarg = 0; jarg < argc; ++jarg)
        {
            std::cout << " " << argv[jarg];
        }
        std::cout << std::endl;
    }
}
