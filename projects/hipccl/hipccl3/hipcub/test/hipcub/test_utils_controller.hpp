// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <algorithm>
#include <any>
#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
// This macro prevents windows.h from defining min/max functions
// that conflict with those in the standard library.
#define NOMINMAX
// This macro prevents windows.h from defining a byte type that
// conflicts with the one in the standard library.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "common_test_header.hpp"

// -- Compile-time checks for build types --

// Valgrind
// Checking if we're running Valgrind require two steps:
// - checking if the main header is included (at compile time, here), and
// - calling the RUNNING_ON_VALGRIND macro at runtime
//   (done in TestController::is_running_valgrind).
#if __has_include(<valgrind/valgrind.h>)
    #include <valgrind/valgrind.h>
    #define HAS_VALGRIND_H 1
#else
    #define HAS_VALGRIND_H 0
#endif

// ASAN
// Rely on defines/features for this detection.
#if defined(__SANITIZE_ADDRESS__)
    #define IS_ASAN_BUILD 1
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define IS_ASAN_BUILD 1
    #endif
#endif
#ifndef IS_ASAN_BUILD
    #define IS_ASAN_BUILD 0
#endif

// This anonymous namespace contains functions needed to print non-primitive types.
// These must be defined before the TestController class.
// If you use a transformer (see documentation for TestController::set_transformer),
// you'll need to ensure that the type you're transforming has a corresponding function
// - either in your own code before including this header, or here.
namespace
{

// pairs
template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p)
{
    return os << "(" << p.first << ", " << p.second << ")";
}

// tuples, 2-element
template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::tuple<T, U>& p)
{
    return os << "(" << std::get<0>(p) << ", " << std::get<1>(p) << ")";
}

// tuples, 3-element
template <typename T, typename U, typename V>
std::ostream& operator<<(std::ostream& os, const std::tuple<T, U, V>& p)
{
    return os << "(" << std::get<0>(p) << ", " << std::get<1>(p) << ", " << std::get<2>(p) << ")";
}
    
}

namespace test_controller
{

// This is the default size-type transformer.
// It returns the value it's given without performing any transformation.
struct IdentityTransformer
{
    using size_type = size_t;
    size_t operator()(const size_type& size) const
    {
        return size;
    }
};

// These structs are used by the file parsing classes, and should not be visible outside this file.
namespace
{
    // Convenience struct to encapsulate all of the information from a single line from the control file.
    struct ControlInfo
    {
        // This regex is applied to the fully-qualified test name (<TestSuite.TestName>).
        std::regex test_regex;
        // Matched against the gfx id for the current device (gfx<integer>).
        std::regex arch_regex;
        // Message to print when sizes (or the entire test) are skipped.
        std::string skip_msg;
        // Parsed from size filters. These are unary functions that accept a size and return true if that size should be skipped.
        std::vector<std::function<bool(size_t)>> size_test_fns;
        // Set to true if the user has specified a '*' to skip all test sizes.
        bool disable_all_sizes = false;
        // Parsed from build types. These are no-argument functions that return true if the tests should be skipped under the current build type.
        std::vector<std::function<bool()>> build_type_test_fns;
        // Record the test control file line number this information comes from, so we can output it when skipping tests.
        // This allows you to see exactly which line in the control file is causing a given test to be skipped,
        // which can be useful when when adding/removing rules.
        size_t line_num = 0;
    };
}

// These functions are used to query information about the environment the test is running in.
namespace env
{
    // Checks if the HIPCUB_EXTRA_TC_INFO exists and is defined and set to 1. If so,
    // individual skipped sizes and the control file line numbers that caused them to
    // be skipped will be appended to messages that are output by the filtering functions.
    // Information about the path to the control file will also be displayed.
    inline bool should_print_extra_info()
    {
        char* val_str = test_common_utils::__get_env("HIPCUB_EXTRA_TC_INFO");
        const bool result = (val_str && std::strlen(val_str) == 1 && val_str[0] == '1');
        test_common_utils::clean_env(val_str);
        return result;
    }

    inline bool is_running_valgrind()
    {
        bool result = false;
#if HAS_VALGRIND_H
        result = RUNNING_ON_VALGRIND;
#endif
        return result;
    }

    constexpr inline bool is_running_asan()
    {
        return IS_ASAN_BUILD;
    }

    constexpr inline bool is_windows()
    {
#ifdef _WIN32
        return true;
#else
        return false;
#endif
    }

    constexpr inline bool is_linux()
    {
        return !is_windows();
    }

    // Uses system calls to get the path to the currently running test binary.
    inline bool get_running_binary_path(std::filesystem::path& path)
    {
        char path_buf[256];
        const size_t max_len = sizeof(path_buf);
        
#ifdef _WIN32
        // In the case where path length > max_len, Windows will truncate the path
        // max_len (including space for '\0').
        const int bytes_read = GetModuleFileName(NULL, path_buf, max_len);
        if (bytes_read == 0)
        {
            std::cerr << "Error: Unable to determine path of running binary." << std::endl;
            return false;
        }
#else
        // Linux does not add a '\0' at the end of the path, so we must do that manually.
        // If truncate if the path exceeds max_len chars.
        const int bytes_read = readlink("/proc/self/exe", path_buf, max_len);
        
        if (bytes_read <= 0 || bytes_read >= max_len)
        {
            std::cerr << "Error: Unable to determine path of running binary." << std::endl;
            return false;
        }
        
        path_buf[bytes_read] = '\0';
#endif

        // At this point we have the path to the binary executable.
        // We want the path to the directory it's in.
        path = std::filesystem::path(path_buf).parent_path();

        // Make sure it exists.
        if (!std::filesystem::exists(path))
        {
            std::cerr << "Could not determine path to running binary." << std::endl;
            return false;
        }
        
        return true;
    }

    // Check if the control file path is give in the HIPCUB_TEST_CONTROL_FILE env var.
    inline bool get_control_file_path_from_env(std::filesystem::path& path)
    {
        char* env_str = test_common_utils::__get_env("HIPCUB_TEST_CONTROL_FILE");
        if (env_str)
        {
            path = std::filesystem::path(std::string(env_str));
            test_common_utils::clean_env(env_str);
            return true;
        }

        return false;
    }
    
    // Search for the control file in expected locations.
    // If we're running from the install location, it will be in a different place than if
    // we're running from the build directory.
    // So we locate it relative to the current test binary.
    // If multiple control files are found, the most recently modified one will be used.
    inline bool find_control_file_path(std::filesystem::path& path)
    {
        // Note: if you change either of these, you'll also need to update the CMake rule that copies
        // it to the build folder. See hipcub/test_CMakeLists.txt.
        const static std::filesystem::path control_file_name("control.txt");
        const static std::filesystem::path project_dir("hipcub");

        std::filesystem::path cur_bin_dir;
        if (!env::get_running_binary_path(cur_bin_dir))
        {
            std::cerr << "Error: Unable to determine path to test control file." << std::endl;
            return false;
        }

        // We will grab the path to the currently running binary, and then look for the test
        // control file in these locations relative to it.
        // When running a test binary that exists in the install directory (eg. /opt/rocm/bin/), we expect the control file to be at hipcub/control.txt
        // When running a test binary that lives in the build directory (eg. build/test/hipcub/), we expect the control file to be at ../control.txt.
        std::vector<std::filesystem::path> possible_paths = {
            cur_bin_dir / project_dir / control_file_name,
            cur_bin_dir.parent_path() / control_file_name
        };
        path.clear();

        // Filter out any paths that don't exist.
        std::vector<std::filesystem::path> existing_paths(possible_paths);
        auto it = std::remove_if(existing_paths.begin(), existing_paths.end(), [](const std::filesystem::path& path) {
            return !std::filesystem::exists(path);
        });
        existing_paths.erase(it, existing_paths.end());

        // If nothing was found, let the user know which paths were searched.
        // We'll return false below.
        if (existing_paths.empty())
        {
            std::cerr << "Error: unable to locate control file." << std::endl
                      << "Locations searched: " << std::endl;
            for (auto const& path : possible_paths)
                std::cerr << path << std::endl;
        }
        // Only one path exists - use it.
        else if (existing_paths.size() == 1)
        {
            path = existing_paths[0];
        }
        // Multiple paths exist - choose the one that was created most recently.
        else
        {
            if (env::should_print_extra_info())
            {
                std::cout << "Multiple matches, selecting path with last write time from these candidates:" << std::endl;
                for (auto p : existing_paths)
                    std::cout << p << std::endl;
                std::cout << std::endl;
            }
            // Note: the selected path will be printed by the caller.
            
            path = std::accumulate(existing_paths.begin() + 1, existing_paths.end(), existing_paths.front(), [](auto const& path1, auto const path2) {
                // last_write_time returns time size last epoch, so a higher value means more recent (closer to now).
                return (std::filesystem::last_write_time(path1) > std::filesystem::last_write_time(path2) ? path1 : path2);
            });
        }

        return !path.empty();
    }
};

// Extracts information from each line of the control file.
// The information is stored in ControlInfo objects.
// Also handles keywords.
class ControlFileParser
{
public:
    // If init is set to true, initializes the parser by reading from
    // the control file.
    // If it's false, then the parser remains uninitialized, and you
    // should call ControlFileParser::reset to provide control data
    // before using it.
    ControlFileParser(const bool init=true)
    {
        if (init)
            this->reset();
    }

    // Resets the internal state of the controller (i.e. the control data)
    // and reparses it from the (optional) string it's passed. If no string is passed,
    // re-reads from the control file.
    // This is useful in the TestController unit tests, where we need to set
    // the control file data for specific scenarios.
    inline void reset(const std::optional<std::string> text=std::nullopt)
    {
        this->control_info.clear();
        if (text)
        {
            std::istringstream ss(text.value());
            this->parse_control_info(ss);
        }
        else
        {
            this->parse_control_info();
        }
    }

    // Returns an interator to the start of the vector of ControlInfo objects
    // that were parsed.
    inline std::vector<ControlInfo>::const_iterator begin() const
    {
        return this->control_info.begin();
    }

    // As above, but returns an iterator to the end.
    inline std::vector<ControlInfo>::const_iterator end() const
    {
        return this->control_info.end();
    }
    
private:
    // Maps <keyword> => <regex string for all gfx ids that keyword represents>
    // Store the regexes as strings since they will be substituted into user-provided strings.
    // Note: these should be enclosed in parenthesis to ensure correctness.
    const inline static std::unordered_map<std::string, std::string> keywords = {
        {"all",           "(.+)"},
        {"amd",           "(gfx[0-9a-f]+)"},
        {"nvidia",        "(nvidia)"}, // see TestController::get_arch
        {"apus",          "(gfx1103|gfx1150|gfx1151|gfx1152)"},
        {"navi2x-family", "(gfx1030|gfx1031|gfx1032)"},
        {"navi3x-family", "(gfx1100|gfx1101|gfx1102)"},
        {"navi4x-family", "(gfx1200|gfx1201)"},
        {"mi100-family",  "(gfx908)"},
        {"mi200-family",  "(gfx90a)"},
        {"mi300-family",  "(gfx942|gfx950)"}
    };

    // Maps build types to functions that check if they match the type of the currently running build.
    const inline static std::unordered_map<std::string, std::function<bool()>> str_to_build_type = {
            {"*",        []() { return true; }},
            {"asan",     env::is_running_asan},
            {"valgrind", env::is_running_valgrind},
            {"windows",  env::is_windows},
            {"linux",    env::is_linux}
    };

    // Maps size units to the values they represent.
    const inline static std::unordered_map<std::string, std::function<size_t(size_t)>> str_to_units = {
        {"K",  [](size_t val) { return val * 1000;}},
        {"Ki", [](size_t val) { return val * (1ull << 10);}},
        {"M",  [](size_t val) { return val * 1000000;}},
        {"Mi", [](size_t val) { return val * (1ull << 20);}},
        {"G",  [](size_t val) { return val * 1000000000;}},
        {"Gi", [](size_t val) { return val * (1ull << 30);}}
    };

    // Each entry represents the parsed information from one line of the control file.
    std::vector<ControlInfo> control_info;
    
    // Parses the control data from the given stream, populating this->control data with the results.
    template<class T>
    inline void parse_control_info(std::basic_istream<T>& istream)
    {
        std::string line;
        size_t line_num = 1;
        while (std::getline(istream, line))
        {
            ControlInfo info;
            bool is_ignored; // set to true when we parse a comment or blank line
            
            if (this->parse_line(line, line_num, is_ignored, info))
            {
                if (!is_ignored)
                    control_info.push_back(info);
            }
            
            ++line_num;
        }
    }

    // As above, but uses the control file as the input stream.
    inline void parse_control_info()
    {
        // Locate the test control file.
        std::filesystem::path control_path;

        // If the env var is set, read the control file path from it.
        if (env::get_control_file_path_from_env(control_path) && env::should_print_extra_info())
        {
            std::cout << "HIPCUB_TEST_CONTROL_FILE is set." << std::endl;
        }
        // Otherwise, fall back on search for it in expected locations.
        else
        {
            if (env::should_print_extra_info())
                std::cout << "HIPCUB_TEST_CONTROL_FILE is not set, falling back to searching for control file." << std::endl;
            
            if (!env::find_control_file_path(control_path))
                return;
        }
        
        if (env::should_print_extra_info())
            std::cout << "Using test control file at: " << control_path << std::endl;
        
        std::ifstream control_file(control_path.c_str());
        if (!control_file)
        {
            std::cerr << "Error: Cannot open test control file at: " << control_path << std::endl;
            return;
        }

        this->parse_control_info(control_file);
        control_file.close();
    }

    // Given an input string that contains the regex inside the architecture part of a line (excluding the '/' chars),
    // goes through it and replaces any keywords with their regexp equivalents.
    // The final regexp is stored in the result output parameter.
    inline bool process_arch_keywords(std::regex& result, const std::string& input, const size_t line_num) const
    {
        // Anything between angle brackets is considered to be a keyword.
        std::regex kwd_pattern(R"(<([^<>]*)>)");
        // The last argument here represents:
        // -1: the portion of the string before the match
        //  0: the match itself
        //  1: the contents of the first capture group
        std::sregex_token_iterator it(input.begin(), input.end(), kwd_pattern, {-1, 0, 1});
        const std::sregex_token_iterator end;

        std::stringstream ss;
        while(it != end)
        {
            const std::string prefix = *it++;
            const std::string match  = (it == end ? std::string() : *it++);
            const std::string group  = (it == end ? std::string() : *it++);

            // Always append any portion of the string before the match. This may be empty.
            ss << prefix;

            if (!match.empty())
            {
                // Replace the keyword with its regex equivalent using the keywords map.
                std::string replacement;
                auto find_it = keywords.find(group);
                if(find_it == keywords.end())
                {
                    std::cerr << "Error: unrecognized arch keyword on input " << line_num << ": \"" << group << "\"." << std::endl;
                    return false;
                }
                else
                {
                    ss << find_it->second;
                }
            }
        }

        result = std::regex(ss.str());
        return true;
    }

    // This function accepts two strings from the control line:
    // - the portion representing the size limit, and
    // - any (optional) unit (eg. M, Mi, etc.).
    // It converts the inputs to a single numeric limit, which is appended to the result output argument.
    // If num_input provided is a "*" (indicating all sizes should be disabled), also sets the corresponding
    // member flag in the info output argument.
    inline bool process_size_limit(std::vector<std::function<bool(const size_t)>>& result,
                                   ControlInfo& info,
                                   const std::string& num_input,
                                   const std::string& unit_input,
                                   const size_t line_num) const
    {
        // num_input may be "*" (disable all sizes) or an integer.
        // If it's a star, create a size test function that will disable all sizes.
        if (num_input == "*")
        {
            if (unit_input.empty())
            {
                const auto test_fn = [](const size_t test_size) { return true; };
                result.push_back(test_fn);
                info.disable_all_sizes = true;
                return true;
            }
            // You can't use units (eg. M, Mi, etc) together with "*".
            else
            {
                std::cerr << "Error: cannot use size limit \"*\" together with units \"" << unit_input << "\" on line " << line_num << "." << std::endl;
                return false;
            }

        }
        // Otherwise it's an integer. Convert it to size_t, multiply in the units, and construct a size test function.
        else
        {
            const char* size_c_str = num_input.c_str();
            size_t      size_limit = 0;
            std::from_chars(size_c_str, size_c_str + num_input.size(), size_limit);

            // Check for units attached to the size.
            // If there are some, multiply them into the size limit.
            if(!unit_input.empty())
            {
                auto it = str_to_units.find(unit_input);
                if(it == str_to_units.end())
                {
                    std::cerr << "Error: unrecognized unit in size on line " << line_num << ": \""
                              << unit_input << "\"" << std::endl;
                    return false;
                }
                else
                {
                    const auto unit_fn = it->second;
                    size_limit         = unit_fn(size_limit);
                }
            }

            const auto test_fn = [size_limit](const size_t test_size) { return test_size >= size_limit; };
            result.push_back(test_fn);
        }
        
        return true;
    }

    // This function accepts a string containing the list of build types from the current control line.
    // It parses it to extract the build types, and grabs functions that check to see if those build conditions exist.
    // These functions are appended to the result output argument.
    inline bool process_build_types(std::vector<std::function<bool()>>& result, const std::string& input, const size_t line_num) const
    {
        // In the control file, the build types may be separated either with commas or spaces.
        // Convert commas to spaces so we're left with a single delimiter.
        std::string normalized = input;
        std::replace(normalized.begin(), normalized.end(), ',', ' ');

        std::istringstream              ss(normalized);
        std::string                     token;
        std::unordered_set<std::string> seen;

        while(ss >> token) // skips whitespace
        {
            if(str_to_build_type.find(token) == str_to_build_type.end())
            {
                std::cerr << "Error: unrecognized build type on line " << line_num << ": \"" + token + "\"" << std::endl;
                return false;
            }

            if(seen.insert(token).second)
                result.push_back(str_to_build_type.at(token));
        }

        return true;
    }

    // This function parses a single line of the control file.
    // Parameters:
    // - line - the control line to parse
    // - line_num - the line number this line is at in the control file
    // - is_ignored - output parameter set to true when a line is empty or contains a comment
    // - info - output parameter that's populated with the parsed info about the current line
    inline bool parse_line(const std::string& line, const size_t line_num, bool& is_ignored, ControlInfo& info) const
    {
        // Remember the line number so we can display it in messages later.
        info.line_num = line_num;
        std::smatch match;
        
        // Skip lines that are empty or start with '#' (comment)
        if (line.empty())
        {
            is_ignored = true;
            return true;
        }
        else
        {
            // Check if this is a comment line (starts with '#')
            const std::regex comment_regex(R"(^\s*#.*)");
            if (std::regex_match(line, match, comment_regex))
            {
                is_ignored = true;
                return true;
            }
        }
        is_ignored = false;

        const std::regex line_regex(
            R"(^\s*\/([^:]+)\/\s*:)"   // Test regex
            R"(\s*\/([^:]+)\/\s*:)"    // Arch regex
            R"(\s*(\d+|\*))"           // Size limit digits
            R"((?:\s*([^:\s]+|))\s*:)" // Size limit units (optional, captures empty string if not provided)
            R"(\s*([^:]+?)\s*:)"       // Build types
            R"(\s*(".+")\s*$)"         // Skip message
        );

        if (!std::regex_match(line, match, line_regex))
        {
            std::cerr << "Warning: unable to parse line " << line_num << " of the test control file. It is not in the expected format."
                      << " This line will not be used." << std::endl;
            return false;
        }

        std::array<std::string, 6> groups;
        // Note: The first item in match is the text of the entire match, which we can ignore - we just want the text from each of the groups.
        std::transform(match.begin() + 1, match.end(), groups.begin(), [](auto submatch) { return submatch.str(); });
        const auto [test_regex, arch, size_val, size_units, build_type, skip_msg] = groups;
        info.test_regex = std::regex(test_regex);
        info.skip_msg = skip_msg;
        
        return process_arch_keywords(info.arch_regex, arch, line_num) &&                 // Replace keywords with their regex equivalents
               process_size_limit(info.size_test_fns, info, size_val, size_units, line_num) && // Create size limit test function
               process_build_types(info.build_type_test_fns, build_type, line_num);      // Create build type test functions        
    }
};
    
// TestController is a singleton that can be used to check if a
// test case or test size is disabled on a given architecture.
// It can also filter a given vector of sizes down to just those
// sizes that are enabled.
//
// ** How to write a test that uses TestController: **
// 1. Create a test fixture class for your test. Make that test fixture inherit from ControlledTest.
//   - ControlledTest inherits from GTest's ::testing::Test, so it inherits all of the regular functionality
//   - of a normal test fixture.
//   - Inheriting from ControlledTest ensures that the main test disablement check is performed automatically before
//     each test in the suite. Tests will be skipped if they are completely (i.e. for all sizes) disabled.
//   - If using a type other than size_t for your sizes, define a "transformer" functor and pass it to ControlledTest
//     as a template argument (within your test fixture definition, eg. class MyTestFixture : public ControlledTest<MyTransformer>).
//     See the documentation for TestController::set_transformer for more information on the functor requirements and how it's used.
//
// 2. Call a maco to filter the input sizes your test uses.
//   - If your test uses a single input size, call:
//     CHECK_SIZE_ENABLEMENT(size);
//     This will cause the test to be skipped if the size matches any of the rules in the control file.
//   - If your test iterates through a vector of sizes, use CHECK_SIZE_FILTERS(sizes) to filter out any sizes
//     that have been disabled by rules in the control file.
//     This macro both modifies the sizes vector in place, and returns the filtered vector so that
//     it can be used directly in a loop. For example:
//     for (const auto size : CHECK_SIZE_FILTERS(sizes))
//        <do your test work using size>
class TestController
{
public:
    // This function should be used to retrieve the instance.
    // Since this is a singleton class, there is no public constructor.
    inline static TestController& get_instance()
    {
        return TestController::get_or_create_instance();
    }

    // Checks whether an entire test is enabled.
    // Returns true if it's enabled, false otherwise.
    // If the test is disabled, populates msg with a string indicating
    // which control file line caused that.
    inline bool check_test_enablement(std::string& msg) const
    {
        return !this->is_test_disabled(msg);
    }

    // As above, but if disabled, prints the message to stdout.
    inline bool check_test_enablement() const
    {
        std::string msg;
        const bool is_enabled = this->check_test_enablement(msg);
        if (!is_enabled)
            std::cout << msg << std::endl;
        
        return is_enabled;
    }

    // Checks whether an individual test size is enabled.
    // This is useful for tests that use a single, fixed input size.
    // Returns true if the given individual size can be used, and
    // false if it should be skipped. If it should be skipped,
    // populates msg with a message about the size that was disabled
    // and which control file line caused it.
    template<class T>
    inline bool check_size_enablement(const T size, std::string& msg) const
    {
        std::vector<T> sizes = {size};
        return !this->filter_sizes_inplace(sizes, msg);
    }

    // As above, but prints the message to stdout.
    template<class T>
    inline bool check_size_enablement(const T size) const
    {
        std::string msg;
        const bool is_enabled = this->check_size_enablement(size, msg);
        if (!is_enabled)
            std::cout << msg << std::endl;
        
        return is_enabled;
    }

    // Examines the sizes in the given vector and removes any
    // that should be skipped. This function both filters the given list in place
    // and returns the filtered list (so it can be used directly in for loops /
    // function chaining). If some sizes should be skipped,
    // msg is populated with the message specified on the corresponding line
    // in the control file.
    // If the environment variable HIPCUB_EXTRA_TC_INFO == 1,
    // extra text is appended to msg that describes the sizes that
    // were skipped and which control file line caused it.
    template<class T>
    inline std::vector<T> filter_sizes(std::vector<T>& sizes, std::string& msg) const
    {
        this->filter_sizes_inplace(sizes, msg);
        return sizes;
    }

    // As above, but prints the message to stdout.
    template<class T>
    inline std::vector<T> filter_sizes(std::vector<T>& sizes) const
    {
        std::string msg;
        this->filter_sizes(sizes, msg);
        if (!msg.empty())
            std::cout << msg << std::endl;  
        
        return sizes;
    }

    // As above, but accepts an rvalue, which is required in some tests.
    // In this case, the argument is not modified in place.
    template<class T>
    inline std::vector<T> filter_sizes(std::vector<T>&& sizes) const
    {
        std::vector<T> sizes_copy(sizes);
        this->filter_sizes(sizes_copy);
        return sizes_copy;
    }

    // Some tests specify sizes using types other than size_t. For example, the device merge
    // algorithm requires two input sizes - one for each of the chunks of data being merged.
    // Because of this, sizes in the device merge tests are stored in a std::tuple<size_t, size_t>.
    //
    // The test control file limits you to using scalar values (size_t) when specifying size limits.
    // However, when performing size filtering, you can pass more complex types to TestController's
    // filtering functions. If you do this, you must also provide a "transformer" functor that converts
    // your complex size type into a scalar size_t.
    // When performing size filtering, TestController will call your functor on each provided input size, and
    // compare the resulting scalar size_t against the rules in the control file.
    // For example, device merge tests can provide a functor that converts
    // tuple<size_t, size_t> to a single size_t by summing the two tuple values to obtain a single, total size.
    //
    // Transform functors must provide:
    // - a type called size_type (eg. defined with `using size_type = ...`) that indicates the (complex)
    //   size type that will be transformed.
    // - an overloaded operator() member function that accepts a parameter of type size_type, and returns a
    //   single size_t value.
    // An example transformer is provided below.
    //
    // Once you've defined your functor, call TestController::set_transformer (below) to set it.
    // This will cause all of the filtering functions to use it.
    // Since TestController is a singleton, once you're done filtering, you'll need to call
    // TestController::reset_transformer to remove the transformer so that the next test doesn't also use it.
    //
    // This pattern is automated by the ControlledTest class at the bottom of this file.
    // When defining your test fixture class, just inherit from ControlledTests and pass
    // your Functor type as a template argument.
    // For example:
    //
    // struct PairTransformer
    // {
    //   using size_type = std::tuple<size_t, size_t>;
    //   size_t operator()(const size_type& size) const
    //   {
    //      return std::get<0>(size) + std::get<1>(size);
    //   }
    // };
    //
    // class MyTestFixture : public ControlledTest<PairTransformer>
    // {
    //   ...
    // }
    //
    // For each test using MyTestFixture, the ControlledTest parent class ensures that
    // TestController::set_transformer(PairTransformer()) is called
    // beforehand, and TestController::reset_transformer() is called afterwards.
    template<class F>
    inline void set_size_transformer(F size_transformer)
    {
        this->size_transformer = TestController::package_transformer(size_transformer);
    }

    // Resets the size transformer to the identity functor - this is equivalent to not using a transformer.
    inline void reset_size_transformer()
    {
        this->size_transformer = TestController::package_transformer(IdentityTransformer());
    }

    // Disallow copy construction and copy assignment,
    // since this is a singleton.
    TestController(const TestController&) = delete;
    TestController& operator=(const TestController&) = delete;

    ~TestController() = default;
    
private:
    // These tests need to access private member functions.
    // See hipcub/test/hipcub/test_hipcub_test_controller.cpp.
    friend class HipcubTestControllerTests;
    FRIEND_TEST(HipcubTestControllerTests, GetArch);
    FRIEND_TEST(HipcubTestControllerTests, CheckTestEnablement);
    FRIEND_TEST(HipcubTestControllerTests, FilterSizes);
    FRIEND_TEST(HipcubTestControllerTests, FilterSizesWithTransformer);
    FRIEND_TEST(HipcubTestControllerTests, FilterSizesWithUnits);
    FRIEND_TEST(HipcubTestControllerTests, CheckSizeEnablement);
    FRIEND_TEST(HipcubTestControllerTests, test_filter);
    FRIEND_TEST(HipcubTestControllerTests, test_size);

    // Private constructor accepting a flag that indicates whether it
    // should read from the control file. If not, the object is left
    // uninitialized and will not report any tests or sizes as disabled.
    TestController(const bool init) : parser(false)
    {
        if (init)
            this->reset();
    }

    // Resets the parser's state and reinitializes it by parsing the provided (optional) text.
    // If std::nullopt is passed, reinitializes by reading from the control file.
    inline void reset(const std::optional<std::string> text=std::nullopt)
    {
        parser.reset(text);
    }

    // This private version of get_instance accepts a bool indicating whether
    // it should read from the control file.
    // If not (TestController unit tests may do this), then before calling member
    // functions, you should call reset(control_text), where control_text is a
    // string representation of the control file contents.
    inline static TestController& get_or_create_instance(const bool init=true)
    {
        // This is the static instance that exists for the duration of the application.
        // It is declared here rather than as a static class member in order
        // to prevent initialization until the first call to this function, which
        // allows us to call runtime functions to load the control data.
        static TestController instance(init);
        return instance;
    }

    // We need a way to store a user-provided custom transformer as a data member.
    // To do this, we'll "package" it using a lambda function that has a fixed signature,
    // then store that in a data member of type std::any.
    template<class F>
    static constexpr std::function<size_t(const typename F::size_type&)> package_transformer(F transformer)
    {
        return std::function<size_t(const typename F::size_type&)>(
            [transformer](const typename F::size_type& size) {return transformer(size);}
        );
    }

    // When we need to use the transformer, we need to extract it from the std::any type data member.
    // Use std::any_cast to cast it back to the fixed signature we set up in TestController::package_transformer, above.
    // Note that here we don't need the functor's type, only the size type it operates on.
    template<class SizeType>
    static std::function<size_t(const SizeType&)> unpackage_transformer(std::any transformer)
    {
        return std::any_cast<std::function<size_t(const SizeType&)>>(transformer);
    }

    // Returns the gfx id of the device that's currently in use.
    inline static std::string get_arch()
    {
        std::string arch;
#ifdef __HIP_PLATFORM_AMD__
        // Make sure we get the device ID from ctest, in case we're running tests in
        // parallel on multiple devices.
        const int device_id = test_common_utils::obtain_device_from_ctest();
        hipDeviceProp_t dev_prop;
        HIP_CHECK(hipGetDeviceProperties(&dev_prop, device_id));
        std::string gcn_arch_name(dev_prop.gcnArchName);

        // The name may contain extra bits we don't need - eg. the xnack portion of "gfx942:xnack+".
        std::regex arch_regex(R"(^([^:\0]+).*)");
        std::smatch match;
        if (std::regex_match(gcn_arch_name, match, arch_regex))
        {
            arch = match[1].str();
        }
        else
        {
            std::cerr << "Warning: Test controller was unable to parse architecture identifier " << "\"" << gcn_arch_name << "\"" << std::endl
                      << "Architecture-based test control file rules may not be applied correctly." << std::endl;
        }
#else
        arch = "nvidia";
#endif
        return arch;
    }

    // Filters out disabled sizes (in-place) using the data parsed from the control file.
    template<class T>
    inline bool filter_sizes_inplace(std::vector<T>& sizes, std::string& msg) const
    {
        // Gather data required for filtering
        const std::string gfx_id = TestController::get_arch();
        const std::string qualified_name = TestController::get_qualified_test_name();
        const auto size_transformer = TestController::unpackage_transformer<T>(this->size_transformer);
        
        // Maps control file line numbers to the sizes that they caused to be skipped.
        std::map<size_t, std::set<T>> skipped_sources;
        // Each time a size is skipped, we'll generate a message saying which control line
        // is responsible. It's possible for multiple sizes to be skipped by the same control file line.
        // Store the skip messages in a set to prevent duplicates (but preserve ordering).
        std::set<std::string> skip_msgs;
        // Remember this so we can figure out how many sizes were removed later.
        const size_t num_unfiltered_sizes = sizes.size();

        // Each line of the control file has a ControlInfo object (stored in this->control_info)
        // that contains information about how sizes should be filtered.
        // For each object, we need to go through all (remaining) input sizes and check which ones it filters out.
        // Note: we can short-circuit here if all sizes have been filtered out.
        for (auto it = this->parser.begin(); !sizes.empty() && it != this->parser.end(); it++)
        {
            // Check if the name of the currently running test and the current architecture match
            // the filter data.
            if (this->is_build_type_considered(*it) &&
                std::regex_match(qualified_name, it->test_regex) &&
                std::regex_match(gfx_id, it->arch_regex))
            {
                // If we have a * in the control file, all sizes are disabled for this test name/arch combination.
                if (it->disable_all_sizes)
                {
                    skipped_sources[it->line_num] = std::set<T>(sizes.begin(), sizes.end());
                    sizes.clear();
                    skip_msgs.clear();
                    skip_msgs.insert(it->skip_msg);
                }

                // Otherwise, we need to check if any individual sizes should be filtered out.
                else
                {
                    // Each ControlInfo (line of the control file) generates one or more "test functions"
                    // that accept a size as an argument and return true if that size should be disabled.
                    // This lambda function returns true if any of the info's individual test functions return true.
                    const auto is_skipped = [&it, &size_transformer](const T& size) {
                        return std::any_of(
                            it->size_test_fns.begin(),
                            it->size_test_fns.end(),
                            // Before calling the test function, transform the size from the user-provided type
                            // to size_t using the transformer that's been set.
                            [&size, &size_transformer](const auto fn) {
                                return fn(size_transformer(size));
                            });
                    };

                    // std::remove_if moves all sizes that satisfy the condition to the end, preserving the
                    // order of other sizes. It returns an iterator pointing past the end of the last non-removed size.
                    const auto new_end = std::remove_if(sizes.begin(), sizes.end(), is_skipped);
                    if (new_end != sizes.end())
                    {
                        auto sources_it = skipped_sources.find(it->line_num);
                        if (sources_it == skipped_sources.end())
                            skipped_sources[it->line_num] = std::set<T>();

                        skipped_sources[it->line_num].insert(new_end, sizes.end());
                        // Erase the sizes that were pushed past the new end
                        sizes.erase(new_end, sizes.end());
                        skip_msgs.insert(it->skip_msg);
                    }
                }
            }
        }

        // If we removed some sizes, record some information about it in msg.
        if (!skipped_sources.empty())
        {
            std::stringstream ss;
            for (const auto& cur_msg : skip_msgs)
            {
                ss << cur_msg << std::endl;
            }
            
            if (env::should_print_extra_info())
            {
                const size_t num_skipped_sizes = num_unfiltered_sizes - sizes.size();
                ss << "Skipping " << num_skipped_sizes << " size(s) based on matches on test control file lines described below." << std::endl;

                ss << "Line\tSkipped Sizes" << std::endl;
                for (const auto& entry : skipped_sources)
                {
                    ss << entry.first << "\t";
                    size_t i = 0;
                    for (auto it = entry.second.begin(); it != entry.second.end(); it++, i++)
                    {
                        ss << *it;
                        if (i < entry.second.size() - 1)
                            ss << ", ";
                    }
                    ss << std::endl;
                }
            }

            msg = ss.str();
        }

        return !skipped_sources.empty();
    }

    // Returns the fully qualified name of the currently running test,
    // in the format: "<suite name>.<test name>"
    inline static std::string get_qualified_test_name()
    {
        std::stringstream ss;
        ss << ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()
           << "."
           << ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::string qualified_name = ss.str();

        // Tests compiled in parallel have a prefix of "Id<digits>/".
        // Strip off this prefix if it's present.
        const std::regex par_prefix_regex(R"(^(?:Id\d+\/)(.+)$)");
        std::smatch match;
        if (std::regex_match(qualified_name, match, par_prefix_regex))
        {
            qualified_name = match[1].str();
        }
        
        return qualified_name;
    }

    // Each control line includes a part that allows you to specify the build types that this rule should be applied to.
    // This function checks if the current build is one of those types.
    bool is_build_type_considered(const ControlInfo& info) const
    {
        return std::any_of(info.build_type_test_fns.begin(), info.build_type_test_fns.end(),
                           [](const auto& func) {
                               return func();
                           });
    }

    // Checks if a test is completely disabled. If so, returns true.
    // A test is completely disabled if all sizes are disabled with a * character
    // in the control file. When a test is completely disabled, this function
    // also populates msg with some information about which control line did that.
    inline bool is_test_disabled(std::string& msg) const
    {
        const std::string gfx_id = TestController::get_arch();
        const std::string qualified_name = TestController::get_qualified_test_name();

        bool is_disabled = false;
        ControlInfo info;
        std::vector<ControlInfo>::const_iterator it = this->parser.begin();
        while (!is_disabled && it != this->parser.end())
        {
            // info.disable_all_sizes records whether the "*" is present in the size part of
            // the control line.
            is_disabled = (this->is_build_type_considered(*it) &&
                           std::regex_match(qualified_name, it->test_regex) &&
                           std::regex_match(gfx_id, it->arch_regex) &&
                           it->disable_all_sizes
            );

            if (!is_disabled)
                it++;
        }

        if (is_disabled)
        {
            std::stringstream ss;
            ss << it->skip_msg;
            if (env::should_print_extra_info())
            {
                ss << std::endl;
                ss << "Test is marked as disabled for all sizes on test control file line " << it->line_num << ".";
            }
            msg = ss.str();
        }

        return is_disabled;
    }

    ControlFileParser parser;
    // Note: when filtering, a transformer is always applied - when no user-provided transformer has been set, we set it to IdentityTransformer,
    // which just returns exactly what it's passed.
    std::any size_transformer = TestController::package_transformer(IdentityTransformer());
};

// -- Macros to use in unit tests --
// Use macros here, even though they're ugly, since it's the only way to call
// GTEST_SKIP() in such a way that we can guarantee that the test is skipped.
// If not using a macro (even if using an inline function), GTEST_SKIP() will
// only skip the function that is currently running, which may not be the top
// level test function.

// Checks if a test is enabled. If not, skips the test.
// This is called automatically if your test fixure class
// inherits from ControlledTest (below), so you shouldn't normally need to call it yourself.
#define CHECK_TEST_ENABLEMENT() \
{ \
  std::string msg; \
  if (!test_controller::TestController::get_instance().check_test_enablement(msg)) \
      GTEST_SKIP() << msg; \
}

// Checks if a single size is enabled. If not, skips the test.
// Use this in tests that set a single, fixed size. If looping through a vector
// of sizes, use CHECK_SIZE_FILTERS below.
#define CHECK_SIZE_ENABLEMENT(size) \
{ \
  std::string msg; \
  if (!test_controller::TestController::get_instance().check_size_enablement(size, msg)) \
      GTEST_SKIP() << msg; \
}

// As above, but issues a continue instead of a GTEST_SKIP, so that it can be used in loops.
#define CHECK_SIZE_ENABLEMENT_WITH_CONTINUE(size) \
{ \
  std::string msg; \
  if (!test_controller::TestController::get_instance().check_size_enablement(size, msg)) \
  { \
      std::cout << msg; \
      continue; \
  } \
}   

// Filters a vector of sizes down to those that are enabled.
// Prints a message indicating the number of sizes that were skipped.
// If env var HIPCUB_EXTRA_TC_INFO is defined and set to 1, also
// prints the size values that were skipped (useful for debugging when adding a new control file rule).
#define CHECK_SIZE_FILTERS(sizes) test_controller::TestController::get_instance().filter_sizes(sizes)

// Unit tests that you want to be able to enable/disable via the control file should
// use a test fixture that inherits from this class. This will automatically
// cause a check to be executed at the beginning of each test that looks to see if
// the test is disabled, and skips it if that's the case.
// If you'd like to use a size transformer, you can pass that as a template argument,
// and it will be set in the test controller before each test is run, and then removed
// after each test completes.
template<class SizeTransformer=test_controller::IdentityTransformer>
class ControlledTest : public ::testing::Test
{
protected:
    // Called before each individual test is run.
    void SetUp() override
    {
        TestController::get_instance().set_size_transformer(SizeTransformer());
        CHECK_TEST_ENABLEMENT();
    }

    // Called after each individual test completes.
    void TearDown() override
    {
        TestController::get_instance().reset_size_transformer();
    }
};

template<class Param, class SizeTransformer=test_controller::IdentityTransformer>
class ControlledTestWithParam : public ::testing::TestWithParam<Param>
{
protected:
    // Called before each individual test is run.
    void SetUp() override
    {
        TestController::get_instance().set_size_transformer(SizeTransformer());
        CHECK_TEST_ENABLEMENT();
    }

    // Called after each individual test completes.
    void TearDown() override
    {
        TestController::get_instance().reset_size_transformer();
    }
};
    
} // namespace test_controller
