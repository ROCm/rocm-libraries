/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_REPORTER_H
#define RPP_TEST_REPORTER_H

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

// Replaces GTest's console output. The suite runs tens of thousands of cases, most of them
// either passing or on the skip list, so the default one-block-per-case format buries the few
// lines that matter. This prints one line per test suite while the run proceeds, the detail of
// every failure as it happens, and a summary of the failures at the end. Skipped cases are only
// ever counted -- the skip list itself is the record of what they are.
//
// Under CTest each process runs a single case (gtest_discover_tests), so a one-test run drops
// the per-suite and summary sections and prints a single verdict line instead. That line keeps
// GTest's "[  SKIPPED ]" spelling, which is what CTest's SKIP_REGULAR_EXPRESSION looks for.

namespace rpptest {

// At most this many failing case names are listed per suite in the final report; the rest are
// counted. Every failure is still printed in full when it happens.
inline constexpr std::size_t kMaxListedFailuresPerSuite = 5;
// Failure detail is clipped to this many lines per assertion. Sized to fit a whole comparator
// verdict (see kMaxReportedMismatches in compare.hpp): the assertion's own two lines, the sampled
// mismatches, and the "... (N more)" tail.
inline constexpr std::size_t kMaxFailureDetailLines = 14;

inline bool reporter_use_color() {
    static const bool enabled = [] {
        const std::string flag = GTEST_FLAG_GET(color);
        if (flag == "yes" || flag == "true" || flag == "1") return true;
        if (flag == "no" || flag == "false" || flag == "0") return false;
        const char* term = std::getenv("TERM");
        return isatty(fileno(stdout)) != 0 && term != nullptr && std::string(term) != "dumb";
    }();
    return enabled;
}

inline std::string colored(const char* code, const std::string& text) {
    return reporter_use_color() ? "\033[" + std::string(code) + "m" + text + "\033[0m" : text;
}

inline std::string green(const std::string& s) {
    return colored("0;32", s);
}
inline std::string red(const std::string& s) {
    return colored("0;31", s);
}
inline std::string yellow(const std::string& s) {
    return colored("0;33", s);
}
inline std::string dim(const std::string& s) {
    return colored("0;90", s);
}

inline std::string seconds(::testing::TimeInMillis ms) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2fs", static_cast<double>(ms) / 1000.0);
    return buf;
}

inline std::string first_line(const std::string& s) {
    const std::size_t nl = s.find('\n');
    return nl == std::string::npos ? s : s.substr(0, nl);
}

inline std::string plural(std::size_t n, const std::string& noun) {
    return std::to_string(n) + " " + noun + (n == 1 ? "" : "s");
}

// Status tokens, all the same width. The spelling is GTest's own: CTest's
// SKIP_REGULAR_EXPRESSION (set by gtest_discover_tests) matches "[  SKIPPED ]" literally, so a
// one-case run has to print exactly that for a skip to be reported as one.
inline constexpr const char* kPassToken = "[  PASSED  ]";
inline constexpr const char* kFailToken = "[  FAILED  ]";
inline constexpr const char* kSkipToken = "[  SKIPPED ]";

class ConciseReporter : public ::testing::EmptyTestEventListener {
   public:
    void OnTestProgramStart(const ::testing::UnitTest& unitTest) override {
        singleTest_ = unitTest.test_to_run_count() == 1;
        if (singleTest_) return;
        std::printf(
            "Running %s from %s\n",
            plural(static_cast<std::size_t>(unitTest.test_to_run_count()), "test").c_str(),
            plural(static_cast<std::size_t>(unitTest.test_suite_to_run_count()), "suite").c_str());
        std::fflush(stdout);
    }

    void OnTestSuiteStart(const ::testing::TestSuite&) override {
        suitePassed_ = suiteFailed_ = suiteSkipped_ = 0;
    }

    void OnTestEnd(const ::testing::TestInfo& info) override {
        const ::testing::TestResult* result = info.result();
        const std::string full = std::string(info.test_suite_name()) + "." + info.name();

        if (result->Failed()) {
            ++suiteFailed_;
            ++totalFailed_;
            failuresBySuite_[info.test_suite_name()].push_back(info.name());
            print_failure(full, *result);
        } else if (result->Skipped()) {
            ++suiteSkipped_;
            ++totalSkipped_;
            if (singleTest_)
                std::printf("%s %s\n             %s\n", yellow(kSkipToken).c_str(), full.c_str(),
                            dim(skip_reason(*result)).c_str());
        } else {
            ++suitePassed_;
            ++totalPassed_;
            if (singleTest_)
                std::printf("%s %s %s\n", green(kPassToken).c_str(), full.c_str(),
                            dim("(" + seconds(result->elapsed_time()) + ")").c_str());
        }
        std::fflush(stdout);
    }

    void OnTestSuiteEnd(const ::testing::TestSuite& suite) override {
        if (singleTest_) return;

        std::string status = green(kPassToken);
        if (suiteFailed_ > 0)
            status = red(kFailToken);
        else if (suitePassed_ == 0 && suiteSkipped_ > 0)
            status = yellow(kSkipToken);

        std::string counts = std::to_string(suitePassed_) + " OK";
        if (suiteFailed_ > 0) counts += ", " + std::to_string(suiteFailed_) + " FAILED";
        if (suiteSkipped_ > 0) counts += ", " + std::to_string(suiteSkipped_) + " SKIPPED";

        std::printf("%s %-58s %-30s %s\n", status.c_str(), suite.name(), counts.c_str(),
                    dim(seconds(suite.elapsed_time())).c_str());
        std::fflush(stdout);
    }

    void OnTestIterationEnd(const ::testing::UnitTest& unitTest, int) override {
        if (singleTest_) return;
        print_report(unitTest);
        reset();
    }

   private:
    void print_failure(const std::string& full, const ::testing::TestResult& result) {
        std::printf("\n%s %s\n", red(kFailToken).c_str(), full.c_str());
        for (int i = 0; i < result.total_part_count(); ++i) {
            const ::testing::TestPartResult& part = result.GetTestPartResult(i);
            if (!part.failed()) continue;
            if (part.file_name() != nullptr)
                std::printf("     %s\n", dim(std::string(part.file_name()) + ":" +
                                             std::to_string(part.line_number()))
                                             .c_str());
            print_clipped(part.summary());
        }
        std::printf("\n");
    }

    static void print_clipped(const char* message) {
        if (message == nullptr) return;
        std::string text(message);
        std::size_t pos = 0, printed = 0;
        while (pos < text.size() && printed < kMaxFailureDetailLines) {
            const std::size_t nl = text.find('\n', pos);
            const std::string line = text.substr(pos, nl - pos);
            if (!line.empty()) {
                std::printf("     %s\n", line.c_str());
                ++printed;
            }
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        if (pos < text.size() && printed == kMaxFailureDetailLines)
            std::printf("     %s\n", dim("... (truncated)").c_str());
    }

    // The bucket a skipped case is counted under: its GTEST_SKIP message, which for skip-list
    // entries is the pattern that matched (so one line stands for every case it covers).
    static std::string skip_reason(const ::testing::TestResult& result) {
        for (int i = 0; i < result.total_part_count(); ++i) {
            const ::testing::TestPartResult& part = result.GetTestPartResult(i);
            if (part.type() != ::testing::TestPartResult::kSkip) continue;
            std::string reason = first_line(part.message());
            const std::string prefix = "skip_list.hpp: ";
            if (reason.compare(0, prefix.size(), prefix) == 0) reason.erase(0, prefix.size());
            if (!reason.empty()) return reason;
        }
        return "(no reason given)";
    }

    void print_report(const ::testing::UnitTest& unitTest) const {
        std::printf("\n%s\n", std::string(78, '-').c_str());

        std::string totals = std::to_string(totalPassed_) + " OK";
        if (totalSkipped_ > 0) totals += ", " + std::to_string(totalSkipped_) + " SKIPPED";
        totals += ", " + std::to_string(totalFailed_) + " FAILED";
        std::printf(
            "  %s in %s: %s  %s\n",
            plural(static_cast<std::size_t>(unitTest.test_to_run_count()), "test").c_str(),
            plural(static_cast<std::size_t>(unitTest.test_suite_to_run_count()), "suite").c_str(),
            (totalFailed_ > 0 ? red(totals) : green(totals)).c_str(),
            dim(seconds(unitTest.elapsed_time())).c_str());

        print_failure_report();

        std::printf("\n%s\n", (totalFailed_ > 0 ? red(kFailToken) : green(kPassToken)).c_str());
        std::fflush(stdout);
    }

    void print_failure_report() const {
        if (failuresBySuite_.empty()) return;
        std::printf("\n  %s\n",
                    red("FAILED (" + plural(static_cast<std::size_t>(totalFailed_), "case") +
                        " in " + plural(failuresBySuite_.size(), "suite") + "):")
                        .c_str());
        for (const auto& [suite, cases] : failuresBySuite_) {
            std::printf("    %5zu  %s\n", cases.size(), suite.c_str());
            for (std::size_t i = 0; i < cases.size() && i < kMaxListedFailuresPerSuite; ++i)
                std::printf("           %s\n", dim(cases[i]).c_str());
            if (cases.size() > kMaxListedFailuresPerSuite)
                std::printf(
                    "           %s\n",
                    dim("+" + std::to_string(cases.size() - kMaxListedFailuresPerSuite) + " more")
                        .c_str());
        }
    }

    void reset() {
        totalPassed_ = totalFailed_ = totalSkipped_ = 0;
        failuresBySuite_.clear();
    }

    bool singleTest_ = false;
    int suitePassed_ = 0, suiteFailed_ = 0, suiteSkipped_ = 0;
    int totalPassed_ = 0, totalFailed_ = 0, totalSkipped_ = 0;
    std::map<std::string, std::vector<std::string>> failuresBySuite_;
};

// Swaps GTest's default console printer for the one above. The XML/JSON generators, if
// requested with --gtest_output, are left in place.
inline void install_concise_reporter() {
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new ConciseReporter);
}

}  // namespace rpptest

#endif  // RPP_TEST_REPORTER_H
