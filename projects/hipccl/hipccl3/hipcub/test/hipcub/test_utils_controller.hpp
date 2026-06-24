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

#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <unordered_map>
#include <algorithm>
#include <charconv>
#include <vector>
#include <set>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stack>
#include <numeric>
#include <functional>
#include <any>
#include <tuple>
#include <iostream>

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

// -- Compile-time checks for built types --

// Valgrind
// Checking if we're running Valgrind require two step:
// - checking if the main header is included (done here), and
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
// - either in your own code, before including this header, or here.
namespace
{
    
template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& p)
{
    return os << "(" << p.first << ", " << p.second << ")";
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::tuple<T, U>& p)
{
    return os << "(" << std::get<0>(p) << ", " << std::get<1>(p) << ")";
}

template <typename T, typename U, typename V>
std::ostream& operator<<(std::ostream& os, const std::tuple<T, U, V>& p)
{
    return os << "(" << std::get<0>(p) << ", " << std::get<1>(p) << ", " << std::get<2>(p) << ")";
}
    
}

namespace test_controller
{

// This is the default size type transformer it just returns the same value
// it's given without performing any transformation.
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

    // Represents a single token (operator or numeric value) in an arithmetic size expression.
    // Eg. the expression "1 << 2" contains tokens: "1", "<<", "2".
    struct Token
    {
        // Note: avoid using a union here, since op needs to be a string (non-primitive type),
        // which needs special treatment in unions.
        std::string op;
        size_t val = 0;
        bool is_val = false;

        Token(size_t val) : op(""), val(val), is_val(true)
        {}

        Token(std::string op) : op(op), val(0), is_val(false)
        {}
    };
}

// These functions are used to query information about the environment the test is running in.
namespace env
{
    // Checks if the HIPCUB_EXTRA_TC_INFO exists and is set to 1. If so,
    // individual skipped sizes and the control file line numbers that caused them to
    // be skipped will be appended to msgs in the filtering functions.
    // Information about the path to the control file will also be displayed.
    // This can be useful for debugging when adding new size constraints.
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

    inline bool is_running_asan()
    {
        return static_cast<bool>(IS_ASAN_BUILD);
    }

    inline bool is_windows()
    {
#ifdef _WIN32
        return true;
#else
        return false;
#endif
    }

    inline bool is_linux()
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

    // Attempt to locate the control file.
    // If we're running from the install location, it will be in a different place than if
    // we're running from the build directory.
    // So we locate it relative to the current test binary.
    // If multiple control files are found, the most recently modified one will be used.
    inline bool get_control_file_path(std::filesystem::path& path)
    {
        // Note: if you change either of these, you'll also need to update the CMake rule that copies
        // it to the build folder. See hipcub/test_CMakeLists.txt.
        const static std::filesystem::path control_file_name("control.txt");
        const static std::filesystem::path project_dir("hipcub");

        std::filesystem::path cur_bin_dir;
        if (!env::get_running_binary_path(cur_bin_dir))
        {
            std::cout << "Error: Unable to determine path to test control file." << std::endl;
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

// Parses size expressions from the control file and reduces them down to a single value.
class TokenParser
{
public:
    // Parses and reduces the given size expression down to a single size_t value.
    static size_t parse_and_simplify(const std::string& size_expr)
    {
        std::vector<Token> postfix_tokens = TokenParser::parse_size_expr(size_expr);
        return TokenParser::reduce_postfix_tokens(postfix_tokens);
    }

private:
    // Converts the size expression string (the size part of the control line) from infix
    // to postfix notation. Postfix is useful because it does not require any parenthesis
    // (order of operations is fully specified), and is easy to evaluate/simplify down
    // into a single value.
    // In order to do this, the function breaks the input string down into a list of tokens.
    // A token is a single numeric value or operator.
    // Returns a vector of Tokens in postfix order.
    inline static std::vector<Token> parse_size_expr(const std::string& size_expr)
    {
        // Translate to postfix notation using Djikstra's "shunting yard" algorithm.
        // This will hold the final output, in postfix order.
        std::vector<Token> output;
        // The shunting yard algorithm uses a stack to use to juggle operators.
        std::stack<std::string> op_stack;

        // Use regexes to break the input string into tokens.
        // Use start anchors, but not end anchors, since we want to
        // match these against the front of the input string.
        const std::regex val_regex(R"(^(\d+))");
        const std::regex op_regex(R"(^(\*|\/|\+|\-|<<|>>))");
        const std::regex left_paren_regex(R"(^(\())");
        const std::regex right_paren_regex(R"(^(\)))");
        const std::regex space_regex(R"(^(\s+))");

        // Iterate through the string, matching the regexes against the front.
        // On each iteration, run the shunting yard algorithm steps, and then
        // finally update `remaining` with whatever portion of the input string is
        // left after the match.
        std::string remaining = size_expr;
        std::smatch match;
        while (!remaining.empty())
        {
            // Whitespace: skip over it and continue.
            if (std::regex_search(remaining, match, space_regex))
            {
                remaining = match.suffix();
            }

            // Numeric value: push directly to output. 
            else if (std::regex_search(remaining, match, val_regex))
            {
                std::string val_str = match[1].str();
                size_t val;
                std::from_chars(val_str.data(), val_str.data() + val_str.size(), val);
                output.emplace_back(val);
                remaining = match.suffix();
            }
            
            // Operator: need to check precendence.
            else if (std::regex_search(remaining, match, op_regex))
            {
                std::string cur_op = match[1].str();
                // Pop operators of higher precedence from the stack and put them in the output.
                // Ensure we don't hit a left paranethesis, since they're also added to the operator stack.
                while (!op_stack.empty() && op_stack.top() != "(" && TokenParser::op_precedence.at(op_stack.top()) >= TokenParser::op_precedence.at(cur_op))
                {
                    output.emplace_back(op_stack.top());
                    op_stack.pop();
                }
                // Finally, push the current op.
                op_stack.emplace(cur_op);
                remaining = match.suffix();
            }

            // Left parenthesis: just push directly onto the operator stack.
            else if (std::regex_search(remaining, match, left_paren_regex))
            {
                op_stack.emplace(match[1].str());
                remaining = match.suffix();
            }

            // Right parenthesis: need to search the stack for matching left parenthesis.
            else if (std::regex_search(remaining, match, right_paren_regex))
            {
                std::string cur_op = match[1].str();
                while (!op_stack.empty() && op_stack.top() != "(")
                {
                    output.emplace_back(op_stack.top());
                    op_stack.pop();
                }
                if (op_stack.empty())
                {
                    std::cerr << "Error: unmatched parenthesis found in size expression beginning at: \"" << remaining << "\"" << std::endl;
                    return {};
                }
                else
                {
                    // pop the "("
                    op_stack.pop();
                }
                remaining = match.suffix();
            }

            // Add an extra case to catch anything else (in the even that our regex is not sufficiently strict).
            else
            {
                std::cerr << "Error: unrecognized size expression syntax beginning at: \"" << remaining << "\"" << std::endl;
                return {};
            }
        }

        // Anything left on the stack now goes into the output.
        while (!op_stack.empty())
        {
            // At this point, no parenthesis should remain.
            if (op_stack.top() == "(" || op_stack.top() == ")")
            {
                std::cerr << "Error: unmatched parenthesis found in size expression: \"" << size_expr << "\"" << std::endl;
                return {};
            }
            else
            {
                output.emplace_back(op_stack.top());
                op_stack.pop();
            }
        }

        return output;
    }
    
    // Given a vector of tokens in postfix order, evaluate/simplify them down until we are left with a single value.
    // Return that value.
    // Note that this operates using intermediate (and result) type size_t. This means negative values are not supported.
    // Because of the way unsigned ints roll over, in some cases it is possible to have an intermediate value that is negative,
    // so long as the final result becomes positive again (eg. 2 - 3 + 1 => 0)
    inline static size_t reduce_postfix_tokens(const std::vector<Token>& tokens)
    {
        // Algorithm:
        // - maintain a stack of operands
        // - For each token:
        //   - If it's an operand, push it onto the stack
        //   - Otherwise, it's an operator. Pop two operands off the stack and apply the operator to them (all operator are
        //     binary in this setup). Push the result back onto the stack.
        // - At the end we should be left with one token on the stack. This is the final result.
        
        size_t result = 0;
        std::stack<size_t> operands;
        for (const auto cur_token : tokens)
        {
            if (cur_token.is_val)
            {
                operands.push(cur_token.val);
            }
            // Otherwise, it's an operator.
            else
            {
                const std::string& op = cur_token.op;
                
                // Grab the two operands for this operator.
                if (operands.size() < 2)
                {
                    std::cerr << "Error simplifying posfix tokens for size expression." << std::endl;
                    return 0;
                }
                size_t right = operands.top();
                operands.pop();
                size_t left = operands.top();
                operands.pop();

                // Apply the operator
                if (op == "*")
                {
                    operands.push(left * right);
                }
                
                else if (op == "/")
                {
                    operands.push(left / right);
                    // Error out on divide by zero.
                    if (right == 0)
                    {
                        std::cerr << "Encountered division by zero when simplifying size expression: " << left << " / " << right << std::endl;
                        return 0;
                    }
                }
                
                else if (op == "+")
                    operands.push(left + right);
                    
                else if (op == "-")
                    operands.push(left - right);
                    
                else if (op == "<<")
                    operands.push(left << right);

                else if (op == ">>")
                    operands.push(left >> right);

                else
                {
                    std::cerr << "Error: unexpected operator encountered in size expression: " << op << std::endl;
                    return 0;
                }
            }
        }

        // At this point, we should have just a single value left: the final result.
        if (operands.size() != 1)
        {
            std:: cerr << "Encountered error simplifying postfix tokens for size expression." << std::endl;
        }

        return operands.top();
    }

    // To enforce the correct order of operations, define operator precedences.
    // Here, a higher value means higher precedence.
    const inline static std::unordered_map<std::string, size_t> op_precedence = {
        {"*",  2},
        {"/",  2},
        {"+",  1},
        {"-",  1},
        {"<<", 0},
        {">>", 0}
    };
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
            {"linux",    env::is_linux},
    };

    // Maps operators to functions that implement them.
    // Note: Treat empty operator as ==
    const inline static std::unordered_map<std::string, std::function<bool(size_t, size_t)>> str_to_op_fn = {
        {"",   std::equal_to<size_t>()},
        {"<",  std::less<size_t>()},
        {">",  std::greater<size_t>()},
        {"<=", std::less_equal<size_t>()},
        {">=", std::greater_equal<size_t>()}
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
        // Attempt to locate the path to the control file.
        std::filesystem::path control_path;
        if (!env::get_control_file_path(control_path))
        {
            std::cerr << "Error: Unable to locate test control file." << std::endl;
            return;
        }

        if (env::should_print_extra_info())
            std::cout << "Using test control file at: " << control_path << std::endl;
        
        std::ifstream control_file(control_path.c_str());
        if (!control_file)
        {
            std::cerr << "Error: Cannot open test control file at: \"" << control_path << "\"" << std::endl;
            return;
        }

        this->parse_control_info(control_file);
        control_file.close();
    }

    // Splits a given line using a vector or regexes.
    // Expects consecutive matches for each of parts_regexes, optionally followed by a delimiter and then another set of matches, etc.
    // Returns a vector of vectors (of strings). Each inner vector represents on set of matches (with parts_regexes.size() elements).
    std::vector<std::vector<std::string>> regex_split(const std::string line,
                                                      const size_t line_num,
                                                      const std::vector<std::regex>& parts_regexes,
                                                      const std::optional<std::regex> delim_regex=std::nullopt)
    {
        const std::vector<std::string> part_descriptions = {"test name regex", "arch regex", "sizes", "build types", "skip message"};
        std::vector<std::vector<std::string>> results;

        const auto print_parse_error = [&part_descriptions, &line_num](const size_t part_index, const std::string& remaining) {
            std::cerr << "Error matching regex on control file line " << line_num << ", in " << part_descriptions[part_index] << " part of rule, beginning at: \"" << remaining << "\"" << std::endl;
        };
        
        std::string remaining = line;
        bool done = false;
        std::smatch match;
        while (!done)
        {
            // match the parts
            std::vector<std::string> groups;
            size_t i;
            bool no_match = false;
            for (i = 0; !no_match && i < parts_regexes.size(); i++)
            {
                if (std::regex_search(remaining, match, parts_regexes[i]))
                {
                    if (match.size() != 2)
                    {
                        print_parse_error(i, remaining);
                        return results;
                    }
                    groups.push_back(match[1].str());
                    remaining = match.suffix();
                }
                else
                {
                    // no match - this is an error. Exit the loop.
                    no_match = true;
                }
            }
            if (no_match)
            {
                print_parse_error(i, remaining);
                return results;
            }
            else
            {
                results.push_back(groups);
            }
            done = remaining.empty();

            if (!done && delim_regex)
            {
                // match the delimiter
                if (std::regex_search(remaining, match, delim_regex.value()))
                {
                    remaining = match.suffix();
                }
                else
                {
                    done = true;
                }
            }
        }

        return results;
    }

    // Parse a single line of the control file.
    // Parameters:
    // - line - the control line to parse
    // - line_num - the line number this line is at in the control file
    // - is_ignored - output parameter set to true when a line is empty or contains a comment
    // - info - output parameter that's populated with the parsed info about the current line
    inline bool parse_line(const std::string& line, const size_t line_num, bool& is_ignored, ControlInfo& info)
    {
        // Remember the line number so we can display it in messages later.
        info.line_num = line_num;
        std::smatch match_result;
        
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
            if (std::regex_match(line, match_result, comment_regex))
            {
                is_ignored = true;
                return true;
            }
        }
        is_ignored = false;

        const std::regex test_part_regex(R"(^\s*\/(.+?)\/\s*:)");
        const std::regex arch_part_regex(R"(^\s*\/(.+?)\/\s*:)");
        const std::regex size_part_regex(R"(^\s*(.+?)\s*:)");
        const std::regex build_type_part_regex(R"(^\s*(.+?)\s*:)");
        const std::regex skip_msg_part_regex(R"(^\s*\"(.*)\"\s*$)");

        const std::vector<std::regex> parts_regexes = {test_part_regex, arch_part_regex, size_part_regex, build_type_part_regex, skip_msg_part_regex};
        const std::vector<std::vector<std::string>> parts_vec = this->regex_split(line, line_num, parts_regexes);
        if (parts_vec.size() != 1)
            return false;
        const std::vector<std::string>& parts = parts_vec[0];
        
        constexpr size_t expected_num_parts = 5;
        if (parts.size() != expected_num_parts)
        {
            std::cerr << "Error parsing line " << line_num << " of test control file. It is not in the expected format." << std::endl;
            return false;
        }
        
        // We can grab the test name regex directly from the match, since no further processing is required.
        info.test_regex = std::regex(parts[0]);
        // The arch regex requires additional processing because it can contain keywords.
        std::string arch_part = parts[1];
        // The size part also needs more processing because it can contain operators (<, >, <=, >=, *).
        std::string size_part = parts[2];
        // The build type part needs more processing.
        std::string build_type_part = parts[3];
        // The mesage can be grabbed directly from the match as well - no further processing needed.
        info.skip_msg = parts[4];

        // -- Process the arch_part --
        
        // Search for any occurrances of keywords. They're surrounded by angle brackets.
        // Note that this means that you cannot use angle brackets for purposes other than
        // keywords in the arch regex. In practice this shouldn't really be a problem since
        // our test names don't contain these characters.

        // Loop through the arch_part, searching for keywords.
        // For each keyword encountered, grab it's equivalent regex (from the ControlFileParser::keywords map).
        // Accumulate these keyword regexes into a final, combined regex.
        
        // String should consist of angle-bracketed keyword(s), and optionally '|' characters, and/or whitespace
        const std::regex kwd_regex(R"(<\s*([^\s\|\<\>]+)\s*>)");
        std::stringstream ss;
        while (std::regex_search(arch_part, match_result, kwd_regex))
        {
            const std::string cur_kwd = match_result[1].str();
            auto kwd_it = ControlFileParser::keywords.find(cur_kwd);
            if (kwd_it == ControlFileParser::keywords.end())
            {
                std::cerr << "Error: unrecognized keyword in arch regex: \"" << cur_kwd << "\"" << std::endl;
                return false;
            }
            else
            {
                // First append whatever came before this keyword
                ss << match_result.prefix();
                // Then append the keyword's regex equivalent
                ss << kwd_it->second;
            }

            // Continue the search from the character after the last match.
            arch_part = match_result.suffix();
        }
        // append whatever portion is left after the last keyword.
        ss << arch_part;

        // Record the final arch regex, which now contains no keywords.
        info.arch_regex = std::regex(ss.str());

        // -- Process the size_part --

        // Search through size_part, looking for operators and size expressions.
        // Once we've tokenized these, use them to create a list of one or more size test functions.
        // A size test function accepts a numeric size as an argument and returns true if the size should
        // be filtered out.
        
        // size_part should consist of comma-delimited items in the format <operator><size expression> (eg. <500,>= (2 * 1000)),
        // maybe with space in between.
        const std::regex size_op_regex = std::regex(R"(^\s*(\<\=|\>\=|\<|\>)?)");
        // Note: Allow for spaces in the size_expr, but only if they are followed by one or more additional non-space chars.
        // We must do this because size_expr is not surrounded by start and end characters like the test and arch regexes (/ ... /).
        const std::regex size_expr_regex = std::regex(R"(^\s*((?:[\d\*\/\+\-\(\)]|<<|>>)+(?:\s*(?:[\d\*\/\+\-\(\)]|<<|>>)+)*))");
        const std::regex size_delim_regex = std::regex(R"(\s*,\s*)");

        const std::vector<std::regex> size_parts_regexes = {size_op_regex, size_expr_regex};
        const std::vector<std::vector<std::string>> size_parts = this->regex_split(size_part, line_num, size_parts_regexes, size_delim_regex);
        
        std::stringstream size_ss;
        for (std::vector<std::string> groups : size_parts)
        {
            if (groups.size() != 2)
                return false;
            
            const std::string op = groups[0];
            const std::string size_expr = groups[1];
            // A solitary * (meaning disable all sizes) will be captured in the second group, since * is also
            // a valid operator.
            info.disable_all_sizes = groups[1] == "*";
            
            // If it's a star, we don't need to add any size_test_fns. Instead, we just set
            // info.disable_all_sizes to true (already done above).
            if (info.disable_all_sizes)
            {
                // Clear any previously recorded size test functions - we don't want to waste
                // time testing them since we're going to skip all sizes anyway.
                // There's no need to continue processing more size constraints either, so break out of the loop.
                info.size_test_fns.clear();
                break;
            }

            else
            {
                // Parse and simplify the size expression.
                const size_t size = TokenParser::parse_and_simplify(size_expr);
                
                auto it = ControlFileParser::str_to_op_fn.find(op);
                if (it == ControlFileParser::str_to_op_fn.end())
                {
                    // This shouldn't be possible if the regex validates correctly.
                    std::cerr << "Error: unknown size operator: \"" << op << "\"" << std::endl;
                    return false;
                }
                else
                {
                    // Grab the comparison function from the map and create a lambda function that calls it to compare the runtime size with the size
                    // obtained from the control file expression.
                    const std::function<bool(size_t, size_t)> compare_fn = it->second;
                    const std::function<bool(size_t)> test_fn = [size, compare_fn](const size_t test_size) {return compare_fn(test_size, size);};
                    info.size_test_fns.push_back(test_fn);
                }
            }
        }

        // -- Process the build_type_part --

        const std::regex build_type_regex = std::regex(R"(^\s*(asan|valgrind|windows|linux|\*)\s*)");
        const std::regex build_type_delim_regex = std::regex(R"(\s*\,\s*)");

        const std::vector<std::regex> build_type_parts_regexes = {build_type_regex};
        const std::vector<std::vector<std::string>> build_type_parts = this->regex_split(build_type_part, line_num, build_type_parts_regexes, build_type_delim_regex);

        for (std::vector<std::string> groups : build_type_parts)
        {
            if (groups.size() != 1)
                return false;
            
            const std::string build_type_str = groups[0];
            auto it = ControlFileParser::str_to_build_type.find(build_type_str);

            if (it == ControlFileParser::str_to_build_type.end())
            {
                std::cerr << "Error: unknown build type: \"" << build_type_str << "\"" << std::endl;
                return false;
            }
            else
            {
                if (it->first == "*")
                {
                    info.build_type_test_fns.clear();
                    info.build_type_test_fns.push_back(it->second);
                    break;
                }
                else
                    info.build_type_test_fns.push_back(it->second);
            }
        }

        return true;
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
//     for (auto size : CHECK_SIZE_FILTERS(sizes))
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
    // Returns true if it's enabled, false if disabled.
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
    // Returns true is the given individual size can be used, and
    // false if it should be skipped. If it should be skipped,
    // populates msg with a message about the size that was disabled
    // and which control file line that caused it.
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
    // Transform functions functors must provide:
    // - a type called size_type (usually defined with `using size_type = ...`) that indicates the (complex)
    //   size type that will be transformed.
    // - an overloaded operator() member function that accepts a parameter of type size_type, and returns a
    //   single size_t value.
    // An example transformer is provided below.
    //
    // Once you've defined your functor, call TestController::set_transformer (below) to set it.
    // This will cause all of the filtering functions to use it.
    // Since TestController is a singleton, once you're done filtering, you'll need to call
    // TestController::reset_transformer to remove the transformer so that the next test doesn't use it.
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
    FRIEND_TEST(HipcubTestControllerTests, CheckSizeEnablement);
    FRIEND_TEST(HipcubTestControllerTests, test_filter);

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
            std::cerr << "Warning: unable to parse architecture identifier " << "\"" << gcn_arch_name << "\"" << std::endl
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

        // Each line of the control file has create a ControlInfo object (stored in this->control_info)
        // that contains information about how sizes should be filtered.
        // For each object (i.e. line of the control file), we need to go through all (remaining)
        // input sizes and check which ones it filters out.
        // Note: we can short-circuit here if all sizes have been filtered out.
        for (auto it = this->parser.begin(); !sizes.empty() && it != this->parser.end(); it++)
        {
            // Check if the name of the currently running test and the current architecture match
            // the filter data.
            std::smatch test_match;
            std::smatch arch_match;

            if (this->is_build_type_considered(*it) &&
                std::regex_match(qualified_name, test_match, it->test_regex) &&
                std::regex_match(gfx_id, arch_match, it->arch_regex))
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
        const std::regex par_prefix_regex(R"(^(Id\d+\/)(.+)$)");
        std::smatch match;
        if (std::regex_match(qualified_name, match, par_prefix_regex))
        {
            qualified_name = match[2].str();
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
        std::smatch test_match;
        std::smatch arch_match;
        std::vector<ControlInfo>::const_iterator it = this->parser.begin();
        while (!is_disabled && it != this->parser.end())
        {
            // info.disable_all_sizes records whether the "*" is present in the size part of
            // the control line.
            is_disabled = (this->is_build_type_considered(*it) &&
                           std::regex_match(qualified_name, test_match, it->test_regex) &&
                           std::regex_match(gfx_id, arch_match, it->arch_regex) &&
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
