/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

/// stinkytofu-check: LLVM FileCheck-style test runner for stinkytofu-opt.
///
/// Reads a .stir file containing:
///   # RUN: %stinkytofu-opt --arch gfx1250 %s --SomePass --print-output
///   # CHECK: expected_substring
///   # CHECK-NEXT: must_be_on_next_line
///   # CHECK-NOT: must_not_appear
///   # XFAIL: the command is expected to exit non-zero; CHECK patterns still verified
///
/// Executes the RUN command, captures stdout, and verifies CHECK directives.

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {
// Match a pattern against a line. Patterns can contain {{regex}} sections.
// Plain text is matched literally, {{...}} sections are matched as regex.
// Example: "ds_load_b128{{.*}}v[8:11]" matches "  v[8:11] = ds_load_b128(v24)"
bool matchPattern(const std::string& line, const std::string& pattern) {
    // If no {{ in pattern, use simple substring match.
    if (pattern.find("{{") == std::string::npos) return line.find(pattern) != std::string::npos;

    // Convert pattern to regex: literal parts are escaped, {{...}} becomes regex.
    std::string regexStr;
    size_t pos = 0;
    while (pos < pattern.size()) {
        size_t start = pattern.find("{{", pos);
        if (start == std::string::npos) {
            // Remaining is literal — escape regex special chars.
            for (size_t i = pos; i < pattern.size(); ++i) {
                if (std::string("[](){}.*+?^$\\|").find(pattern[i]) != std::string::npos)
                    regexStr += '\\';
                regexStr += pattern[i];
            }
            break;
        }
        // Literal part before {{
        for (size_t i = pos; i < start; ++i) {
            if (std::string("[](){}.*+?^$\\|").find(pattern[i]) != std::string::npos)
                regexStr += '\\';
            regexStr += pattern[i];
        }
        // Find closing }}
        size_t end = pattern.find("}}", start + 2);
        if (end == std::string::npos) break;  // Malformed — treat rest as literal
        // Extract regex between {{ and }}
        regexStr += pattern.substr(start + 2, end - start - 2);
        pos = end + 2;
    }

    std::regex re(regexStr, std::regex::nosubs);
    return std::regex_search(line, re);
}

struct CheckDirective {
    enum Kind { CHECK, CHECK_NEXT, CHECK_SAME, CHECK_NOT, CHECK_DAG, CHECK_LABEL };
    Kind kind;
    std::string pattern;
    int lineNum;
};

// Parse # RUN:, # XFAIL, and # CHECK: directives from the test file.
bool parseTestFile(const std::string& filename, std::string& runCmd,
                   std::vector<CheckDirective>& checks, bool& xfail) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return false;
    }

    xfail = false;
    std::string line;
    int lineNum = 0;
    while (std::getline(file, line)) {
        ++lineNum;
        // Skip leading whitespace
        size_t pos = line.find_first_not_of(" \t");
        if (pos == std::string::npos) continue;

        // Lines starting with # (comments in .stir)
        if (line[pos] != '#') continue;

        std::string rest = line.substr(pos + 1);
        pos = rest.find_first_not_of(" \t");
        if (pos == std::string::npos) continue;
        rest = rest.substr(pos);

        if (rest.rfind("XFAIL", 0) == 0) {
            xfail = true;
        } else if (rest.rfind("RUN:", 0) == 0) {
            runCmd = rest.substr(4);
            // Trim leading whitespace
            pos = runCmd.find_first_not_of(" \t");
            if (pos != std::string::npos) runCmd = runCmd.substr(pos);
        } else if (rest.rfind("CHECK-LABEL:", 0) == 0) {
            std::string pat = rest.substr(12);
            pos = pat.find_first_not_of(" \t");
            if (pos != std::string::npos) pat = pat.substr(pos);
            checks.push_back({CheckDirective::CHECK_LABEL, pat, lineNum});
        } else if (rest.rfind("CHECK-SAME:", 0) == 0) {
            std::string pat = rest.substr(11);
            pos = pat.find_first_not_of(" \t");
            if (pos != std::string::npos) pat = pat.substr(pos);
            checks.push_back({CheckDirective::CHECK_SAME, pat, lineNum});
        } else if (rest.rfind("CHECK-NEXT:", 0) == 0) {
            std::string pat = rest.substr(11);
            pos = pat.find_first_not_of(" \t");
            if (pos != std::string::npos) pat = pat.substr(pos);
            checks.push_back({CheckDirective::CHECK_NEXT, pat, lineNum});
        } else if (rest.rfind("CHECK-DAG:", 0) == 0) {
            std::string pat = rest.substr(10);
            pos = pat.find_first_not_of(" \t");
            if (pos != std::string::npos) pat = pat.substr(pos);
            checks.push_back({CheckDirective::CHECK_DAG, pat, lineNum});
        } else if (rest.rfind("CHECK-NOT:", 0) == 0) {
            std::string pat = rest.substr(10);
            pos = pat.find_first_not_of(" \t");
            if (pos != std::string::npos) pat = pat.substr(pos);
            checks.push_back({CheckDirective::CHECK_NOT, pat, lineNum});
        } else if (rest.rfind("CHECK:", 0) == 0) {
            std::string pat = rest.substr(6);
            pos = pat.find_first_not_of(" \t");
            if (pos != std::string::npos) pat = pat.substr(pos);
            checks.push_back({CheckDirective::CHECK, pat, lineNum});
        }
    }

    return !runCmd.empty();
}

// Execute a command and capture stdout. Returns the exit status via out-param.
std::string executeCommand(const std::string& cmd, int& exitStatus) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "Error: failed to execute: " << cmd << "\n";
        exitStatus = -1;
        return {};
    }

    std::string result;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe)) result += buffer;

    exitStatus = pclose(pipe);
    return result;
}

// Substitute %s and %stinkytofu-opt in the RUN command.
std::string substituteVars(const std::string& cmd, const std::string& testFile,
                           const std::string& optBinary) {
    std::string result = cmd;

    // Replace %stinkytofu-opt first (longer match before %s)
    size_t pos;
    while ((pos = result.find("%stinkytofu-opt")) != std::string::npos)
        result.replace(pos, 15, optBinary);
    while ((pos = result.find("%s")) != std::string::npos) result.replace(pos, 2, testFile);

    return result;
}

// Verify CHECK directives against output lines.
bool verifyChecks(const std::vector<std::string>& outputLines,
                  const std::vector<CheckDirective>& checks, const std::string& testFile) {
    size_t outputIdx = 0;
    bool ok = true;

    for (size_t i = 0; i < checks.size(); ++i) {
        const auto& chk = checks[i];

        if (chk.kind == CheckDirective::CHECK_LABEL) {
            // Reset scan position — find pattern from the beginning.
            bool found = false;
            for (size_t j = 0; j < outputLines.size(); ++j) {
                if (matchPattern(outputLines[j], chk.pattern)) {
                    outputIdx = j + 1;
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << testFile << ":" << chk.lineNum
                          << ": CHECK-LABEL failed: pattern not found\n";
                std::cerr << "  expected: " << chk.pattern << "\n";
                ok = false;
            }
            continue;
        }

        if (chk.kind == CheckDirective::CHECK_SAME) {
            // Match on the same line as the previous CHECK (outputIdx-1).
            if (outputIdx == 0) {
                std::cerr << testFile << ":" << chk.lineNum
                          << ": CHECK-SAME failed: no previous CHECK\n";
                ok = false;
                continue;
            }
            size_t prevIdx = outputIdx - 1;
            if (!matchPattern(outputLines[prevIdx], chk.pattern)) {
                std::cerr << testFile << ":" << chk.lineNum << ": CHECK-SAME failed at output line "
                          << (prevIdx + 1) << "\n";
                std::cerr << "  expected: " << chk.pattern << "\n";
                std::cerr << "  got:      " << outputLines[prevIdx] << "\n";
                ok = false;
            }
            continue;
        }

        if (chk.kind == CheckDirective::CHECK_NOT) {
            // Pattern must NOT appear before the next CHECK/CHECK-LABEL.
            // Find scan end: the output line matched by the next non-NOT directive.
            size_t scanEnd = outputLines.size();
            for (size_t j = i + 1; j < checks.size(); ++j) {
                if (checks[j].kind == CheckDirective::CHECK_NOT) continue;
                // For CHECK-LABEL, find its match position to bound the scan.
                if (checks[j].kind == CheckDirective::CHECK_LABEL) {
                    for (size_t k = 0; k < outputLines.size(); ++k) {
                        if (matchPattern(outputLines[k], checks[j].pattern)) {
                            scanEnd = k;
                            break;
                        }
                    }
                } else {
                    // For other CHECK directives, scan forward from current pos.
                    for (size_t k = outputIdx; k < outputLines.size(); ++k) {
                        if (matchPattern(outputLines[k], checks[j].pattern)) {
                            scanEnd = k;
                            break;
                        }
                    }
                }
                break;
            }
            for (size_t j = outputIdx; j < scanEnd; ++j) {
                if (matchPattern(outputLines[j], chk.pattern)) {
                    std::cerr << testFile << ":" << chk.lineNum << ": CHECK-NOT failed: found '"
                              << chk.pattern << "' at output line " << (j + 1) << "\n";
                    std::cerr << "  output: " << outputLines[j] << "\n";
                    ok = false;
                    break;
                }
            }
            continue;
        }

        if (chk.kind == CheckDirective::CHECK_NEXT) {
            if (outputIdx >= outputLines.size()) {
                std::cerr << testFile << ":" << chk.lineNum
                          << ": CHECK-NEXT failed: no more output\n";
                std::cerr << "  expected: " << chk.pattern << "\n";
                ok = false;
                continue;
            }
            if (!matchPattern(outputLines[outputIdx], chk.pattern)) {
                std::cerr << testFile << ":" << chk.lineNum << ": CHECK-NEXT failed at output line "
                          << (outputIdx + 1) << "\n";
                std::cerr << "  expected: " << chk.pattern << "\n";
                std::cerr << "  got:      " << outputLines[outputIdx] << "\n";
                ok = false;
            }
            ++outputIdx;
            continue;
        }

        if (chk.kind == CheckDirective::CHECK_DAG) {
            // Collect consecutive CHECK-DAG directives into a group.
            std::vector<const CheckDirective*> dagGroup;
            dagGroup.push_back(&chk);
            while (i + 1 < checks.size() && checks[i + 1].kind == CheckDirective::CHECK_DAG)
                dagGroup.push_back(&checks[++i]);

            // All patterns in the group must match within the output range
            // [outputIdx, outputIdx + scanRange), in any order.
            // Each pattern consumes one output line.
            std::vector<bool> matched(dagGroup.size(), false);
            size_t startIdx = outputIdx;
            size_t endIdx = outputLines.size();

            for (size_t g = 0; g < dagGroup.size(); ++g) {
                bool found = false;
                for (size_t j = startIdx; j < endIdx; ++j) {
                    if (matchPattern(outputLines[j], dagGroup[g]->pattern)) {
                        matched[g] = true;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << testFile << ":" << dagGroup[g]->lineNum
                              << ": CHECK-DAG failed: pattern not found\n";
                    std::cerr << "  expected: " << dagGroup[g]->pattern << "\n";
                    ok = false;
                }
            }

            // Advance outputIdx past all matched lines.
            // Find the furthest matched line.
            for (size_t j = startIdx; j < endIdx; ++j) {
                for (size_t g = 0; g < dagGroup.size(); ++g) {
                    if (matched[g] && matchPattern(outputLines[j], dagGroup[g]->pattern)) {
                        if (j + 1 > outputIdx) outputIdx = j + 1;
                    }
                }
            }
            continue;
        }

        // CHECK: scan forward for the pattern.
        bool found = false;
        for (; outputIdx < outputLines.size(); ++outputIdx) {
            if (matchPattern(outputLines[outputIdx], chk.pattern)) {
                found = true;
                ++outputIdx;
                break;
            }
        }
        if (!found) {
            std::cerr << testFile << ":" << chk.lineNum << ": CHECK failed: pattern not found\n";
            std::cerr << "  expected: " << chk.pattern << "\n";
            ok = false;
        }
    }

    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: stinkytofu-check <test.stir> [--stinkytofu-opt <path>]\n";
        return 1;
    }

    std::string testFile = argv[1];
    std::string optBinary = "stinkytofu-opt";  // default

    for (int i = 2; i < argc; ++i) {
        if (std::string(argv[i]) == "--stinkytofu-opt" && i + 1 < argc) optBinary = argv[++i];
    }

    // Parse test file
    std::string runCmd;
    std::vector<CheckDirective> checks;
    bool xfail = false;
    if (!parseTestFile(testFile, runCmd, checks, xfail)) {
        std::cerr << "Error: no RUN: directive found in " << testFile << "\n";
        return 1;
    }
    if (checks.empty()) {
        std::cerr << "Error: no CHECK directives found in " << testFile << "\n";
        return 1;
    }

    // Substitute variables and execute
    std::string cmd = substituteVars(runCmd, testFile, optBinary);
    int exitStatus = 0;
    std::string output = executeCommand(cmd, exitStatus);
    if (exitStatus != 0) {
        if (xfail) {
            std::cerr << testFile << ": note: command exited non-zero (expected by XFAIL)\n";
        } else {
            std::cerr << "Warning: command exited with status " << exitStatus << "\n";
        }
    }

    // Split output into lines
    std::vector<std::string> outputLines;
    {
        std::istringstream stream(output);
        std::string line;
        while (std::getline(stream, line)) outputLines.push_back(line);
    }

    // Verify
    if (verifyChecks(outputLines, checks, testFile)) {
        std::cerr << testFile << ": PASSED\n";
        return 0;
    } else {
        std::cerr << testFile << ": FAILED\n";
        return 1;
    }
}
