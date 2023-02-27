
#include "include/Parser.hpp"

#include <regex>

#include <rocRoller/Utilities/Error.hpp>

ParseOptions::ParseOptions() {}

ParseOptions::ParseOptions(std::string helpMessage)
    : m_helpMessage(std::move(helpMessage))
{
}

void ParseOptions::print_help()
{
    std::cout << m_helpMessage << std::endl;

    for(auto arg : m_valid_args)
    {
        for(auto flag : arg.second.flags())
        {
            if(flag.size() == 1)
            {
                std::cout << "-" << flag << ",";
            }
            else
            {
                std::cout << "--" << flag << ",";
            }
        }

        std::cout << "\t" << arg.second.usage() << std::endl;
    }
}

void ParseOptions::parse_args(int argc, const char* argv[])
{
    // No command line args
    if(argc == 1)
    {
        return;
    }

    // regex for: "-f=val", "-f val", "--flag=val", and "--flag val"

    std::regex cmd_regex("(--?)(\\w+)[=\\s](\\S+)");

    // Iterate over command line args
    for(int i = 1; i < argc; ++i)
    {
        std::string arg(argv[i]);

        // Look for help flag
        if(arg == "-h" || arg == "--help")
        {
            print_help();
            m_parsed_args.clear();
            exit(EXIT_SUCCESS);
        }

        // if flag and val seperated by whitespace
        if(arg.find("=") == std::string::npos)
        {
            rocRoller::Throw<rocRoller::FatalError>(i + 1 < argc, "No matching argument for flag");
            arg += " " + std::string(argv[++i]);
        }

        std::smatch match;
        std::regex_match(arg, match, cmd_regex);

        // matches in order: entire string, -(-), f(lag), val
        if(match.size() != 4)
        {
            std::cout << "Command line args incorretly formatted" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string flag  = match[2];
        std::string value = match[3];

        m_parsed_args.insert({flag, value});
    }

    validateArgs();
}

void ParseOptions::validateArgs()
{
    std::set<std::string> validFlags;

    for(auto arg : m_valid_args)
    {
        for(auto flag : arg.second.flags())
        {
            validFlags.insert(flag);
        }
    }

    for(auto arg : m_parsed_args)
    {
        if(validFlags.find(arg.first) == validFlags.end())
        {
            std::cout << "Invalid command option: " << arg.first << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    return;
}

void ParseOptions::addArg(std::string name, Arg const& arg)
{
    m_valid_args.insert({std::move(name), arg});
}

template <>
std::string ParseOptions::get<std::string>(std::string const& name,
                                           std::string const& defaultVal) const
{
    auto flags = m_valid_args.at(name).flags();
    for(auto flag : flags)
    {
        if(m_parsed_args.find(flag) != m_parsed_args.end())
        {
            return m_parsed_args.at(flag);
        }
    }
    return defaultVal;
}

template <>
int ParseOptions::get<int>(std::string const& name, int const& defaultVal) const
{
    auto flags = m_valid_args.at(name).flags();
    for(auto flag : flags)
    {
        if(m_parsed_args.find(flag) != m_parsed_args.end())
        {
            return std::stoi(m_parsed_args.at(flag));
        }
    }
    return defaultVal;
}

template <>
float ParseOptions::get<float>(std::string const& name, float const& defaultVal) const
{
    auto flags = m_valid_args.at(name).flags();
    for(auto flag : flags)
    {
        if(m_parsed_args.find(flag) != m_parsed_args.end())
        {
            return std::stof(m_parsed_args.at(flag));
        }
    }
    return defaultVal;
}

template <>
bool ParseOptions::get<bool>(std::string const& name, bool const& defaultVal) const
{
    auto flags = m_valid_args.at(name).flags();
    for(auto flag : flags)
    {
        if(m_parsed_args.find(flag) != m_parsed_args.end())
        {
            return m_parsed_args.at(flag) == "1" || m_parsed_args.at(flag) == "True"
                   || m_parsed_args.at(flag) == "true";
        }
    }
    return defaultVal;
}

Arg::Arg(std::vector<std::string> flags, std::string usage)
    : m_flags(std::move(flags))
    , m_usage(std::move(usage))
{
}

std::vector<std::string> Arg::flags() const
{
    return m_flags;
}

std::string Arg::usage() const
{
    return m_usage;
}
