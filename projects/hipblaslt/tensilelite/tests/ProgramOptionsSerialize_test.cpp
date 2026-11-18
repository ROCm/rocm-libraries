// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "DataInitialization.hpp"
#include "BenchmarkDatabaseSetup.hpp"
#include "ProgramOptions.hpp"

#include <Tensile/DataTypes.hpp>

#include <algorithm>
#include <any>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

// One option of every kind of value the client's table holds.
static po::options_description makeOptions()
{
    po::options_description options("test");
    options.add_options()
        ("help,h", "Show help.")
        ("config-file", po::value<std::vector<std::string>>()->default_value({}), "Config.")
        ("name,n", po::value<std::string>(), "String without a default.")
        ("count", po::value<int>()->default_value(3), "Integer.")
        ("size", po::value<size_t>()->default_value(7), "Size.")
        ("seed", po::value<unsigned int>()->default_value(0), "Unsigned.")
        ("ratio", po::value<double>()->default_value(0.1), "Double.")
        ("fraction", po::value<float>()->default_value(0.3f), "Float.")
        ("flag", po::value<bool>()->default_value(false), "Bool.")
        ("mode", po::value<InitMode>()->default_value(InitMode::Random), "Enum.")
        ("type", po::value<rocisa::DataType>()->default_value(rocisa::DataType::Float), "Type.")
        ("sizes", po::value<std::vector<std::string>>()->default_value({}), "Strings.")
        ("ints", po::value<std::vector<int>>()->default_value({}), "Ints.")
        ("bools", po::value<std::vector<bool>>()->default_value(std::vector<bool>(1, true)),
         "Bools.")
        ("types", po::value<std::vector<rocisa::DataType>>()->default_value({}), "Types.");
    return options;
}

static po::variables_map parseArgs(po::options_description const& options,
                                   std::vector<std::string> const& arguments)
{
    std::vector<char const*> argv = {"client"};
    for(std::string const& argument : arguments)
        argv.push_back(argument.c_str());
    return po::parse_command_line(static_cast<int>(argv.size()), argv.data(), options);
}

static po::variables_map parseIni(po::options_description const& options, std::string const& ini)
{
    std::istringstream stream(ini);
    return po::parse_config_file(stream, options);
}

static std::set<std::string> excluded()
{
    return {"help", "config-file"};
}

TEST(ProgramOptionsSerialize, ValuesRoundTripThroughTheConfigParser)
{
    po::options_description options = makeOptions();
    po::variables_map       args    = parseArgs(options,
                                          {"--name", "result.csv",
                                           "--count", "-5",
                                           "--size", "123456789012",
                                           "--seed", "1234",
                                           "--ratio", "0.1234567890123456",
                                           "--fraction", "0.3",
                                           "--flag",
                                           "--mode", "Zero",
                                           "--type", "BFloat16",
                                           "--ints", "1,2,3",
                                           "--bools", "false",
                                           "--bools", "true",
                                           "--types", "Half"});

    // The command line splits a string at commas, a config file line is one value.
    args["sizes"].value() = std::any(std::vector<std::string>{"512,512,1,1024", "64,64,1,64"});

    std::string ini = po::serialize_config(args, options, excluded());

    po::variables_map back = parseIni(options, ini);
    EXPECT_EQ(back["name"].as<std::string>(), "result.csv");
    EXPECT_EQ(back["count"].as<int>(), -5);
    EXPECT_EQ(back["size"].as<size_t>(), 123456789012u);
    EXPECT_EQ(back["seed"].as<unsigned int>(), 1234u);
    EXPECT_EQ(back["ratio"].as<double>(), 0.1234567890123456);
    EXPECT_EQ(back["fraction"].as<float>(), 0.3f);
    EXPECT_TRUE(back["flag"].as<bool>());
    EXPECT_EQ(back["mode"].as<InitMode>(), InitMode::Zero);
    EXPECT_EQ(back["type"].as<rocisa::DataType>(), rocisa::DataType::BFloat16);
    EXPECT_EQ(back["sizes"].as<std::vector<std::string>>(),
              (std::vector<std::string>{"512,512,1,1024", "64,64,1,64"}));
    EXPECT_EQ(back["ints"].as<std::vector<int>>(), (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(back["bools"].as<std::vector<bool>>(), (std::vector<bool>{false, true}));
    EXPECT_EQ(back["types"].as<std::vector<rocisa::DataType>>(),
              (std::vector<rocisa::DataType>{rocisa::DataType::Half}));

    // The text is canonical: reading it back and writing it again gives the same text.
    EXPECT_EQ(po::serialize_config(back, options, excluded()), ini);
}

TEST(ProgramOptionsSerialize, DataTypeNoneReadsBackFromItsOwnText)
{
    std::ostringstream out;
    out << rocisa::DataType::None;

    std::istringstream  in(out.str());
    rocisa::DataType    type = rocisa::DataType::Float;
    in >> type;
    EXPECT_EQ(type, rocisa::DataType::None);
}

TEST(ProgramOptionsSerialize, DefaultsAreWritten)
{
    po::options_description options = makeOptions();
    std::string ini = po::serialize_config(parseArgs(options, {}), options, excluded());

    EXPECT_NE(ini.find("count=3\n"), std::string::npos) << ini;
    EXPECT_NE(ini.find("flag=false\n"), std::string::npos) << ini;
    EXPECT_NE(ini.find("bools=true\n"), std::string::npos) << ini;
    // An empty list is written as an empty value, which reads back as an empty list.
    EXPECT_NE(ini.find("sizes=\n"), std::string::npos) << ini;
    EXPECT_TRUE(parseIni(options, ini)["sizes"].as<std::vector<std::string>>().empty());
    EXPECT_TRUE(parseIni(options, ini)["ints"].as<std::vector<int>>().empty());
}

TEST(ProgramOptionsSerialize, OptionsWithoutValueAndExcludedOptionsAreOmitted)
{
    po::options_description options = makeOptions();
    std::string ini = po::serialize_config(parseArgs(options, {"--count", "4"}), options, excluded());

    EXPECT_EQ(ini.find("name="), std::string::npos) << ini;
    EXPECT_EQ(ini.find("help"), std::string::npos) << ini;
    EXPECT_EQ(ini.find("config-file"), std::string::npos) << ini;

    std::set<std::string> more = excluded();
    more.insert("count");
    EXPECT_EQ(po::serialize_config(parseArgs(options, {"--count", "4"}), options, more)
                  .find("count="),
              std::string::npos);
}

TEST(ProgramOptionsSerialize, LinesAreSortedByOptionName)
{
    po::options_description options = makeOptions();
    std::string ini = po::serialize_config(parseArgs(options, {}), options, excluded());

    std::vector<std::string> names;
    std::istringstream       lines(ini);
    for(std::string line; std::getline(lines, line);)
        names.push_back(line.substr(0, line.find('=')));
    EXPECT_TRUE(std::is_sorted(names.begin(), names.end()));
}

TEST(ProgramOptionsSerialize, UnrepresentableValuesFailLoudly)
{
    po::options_description options = makeOptions();
    for(std::string const& bad : {"two words", "with#hash"})
    {
        po::variables_map args = parseArgs(options, {});
        args["name"].value()   = std::any(bad);
        EXPECT_THROW(po::serialize_config(args, options, excluded()), std::runtime_error) << bad;
    }

    po::variables_map args = parseArgs(options, {});
    args["sizes"].value()  = std::any(std::vector<std::string>{"a\nb"});
    EXPECT_THROW(po::serialize_config(args, options, excluded()), std::runtime_error);

    // A value of another type than the option's, as left by the client's post-processing.
    args = parseArgs(options, {});
    args["count"].value() = std::any(std::string("x"));
    EXPECT_THROW(po::serialize_config(args, options, excluded()), std::runtime_error);
}

TEST(ProgramOptionsSerialize, SchedulingOptionsAreExcludedFromTheDatabaseConfig)
{
    po::options_description options("test");
    options.add_options()
        ("help,h", "Show help.")
        ("config-file", po::value<std::vector<std::string>>()->default_value({}), "Config.")
        ("benchmark-db", po::value<std::string>()->default_value(""), "Database.")
        ("indirect-failures", po::value<bool>()->default_value(true), "Crashes.")
        ("indirect-verification-failures", po::value<bool>()->default_value(true), "Verification.")
        ("count", po::value<int>()->default_value(3), "Integer.");

    std::set<std::string> excludedOptions = optionsExcludedFromDatabaseConfig();
    // Boolean options are flags on the command line, so values come from an ini text.
    auto serialized = [&](std::string const& ini) {
        return po::serialize_config(parseIni(options, ini), options, excludedOptions);
    };

    EXPECT_EQ(serialized("indirect-failures=true\n"), serialized("indirect-failures=false\n"));
    EXPECT_EQ(serialized("indirect-verification-failures=true\n"),
              serialized("indirect-verification-failures=false\n"));
    EXPECT_NE(serialized("count=3\n"), serialized("count=4\n"));
}
