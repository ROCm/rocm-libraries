// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// One GEMM for each selected solution index of the built library, constructed
// from that solution's own TensileLite record (solution_index_case.hpp).
// Opt-in: set HIPBLASLT_SOLUTION_INDEX_RANGE=BEGIN:END (half-open) or
// HIPBLASLT_SOLUTION_INDEX_LIST=FILE (one index per line). The library comes
// from HIPBLASLT_TENSILE_LIBPATH, as for the runtime, and must match the device.
// HIPBLASLT_SOLUTION_INDEX_SECONDARY_DEVICE keeps only solutions that need this
// device's CU count; HIPBLASLT_SOLUTION_INDEX_{REDUCTION_BLOCKS,ALPHA,BETA}
// override SolutionCaseOptions.

#include "hipblaslt_arguments.hpp"
#include "solution_index_case.hpp"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

// Defined in testing_matmul.hpp, which only one translation unit may include.
void testing_matmul(const Arguments& arg);

namespace
{
    using ContractionLibrary
        = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    double envNumber(char const* name, double fallback)
    {
        char const* value = std::getenv(name);
        return value ? std::atof(value) : fallback;
    }

    // A case this device does not run, and why.
    struct Skip
    {
        std::string reason;
    };

    struct SolutionCase
    {
        int                           index = -1;
        std::string                   kernel;
        std::variant<Arguments, Skip> plan = Skip{};
    };

    void PrintTo(SolutionCase const& c, std::ostream* os)
    {
        *os << "solution index " << c.index;
        if(auto const* skip = std::get_if<Skip>(&c.plan))
            *os << " (" << skip->reason << ")";
    }

    std::shared_ptr<ContractionLibrary> loadLibrary(std::string& reason)
    {
        hipDeviceProp_t props{};
        int             device = 0;
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&props, device) != hipSuccess)
        {
            reason = "no device";
            return nullptr;
        }
        std::string const arch      = props.gcnArchName;
        std::string const processor = arch.substr(0, arch.find(':'));
        char const*       env       = std::getenv("HIPBLASLT_TENSILE_LIBPATH");
        if(!env)
        {
            reason = "HIPBLASLT_TENSILE_LIBPATH is unset";
            return nullptr;
        }
        std::string const name = "TensileLibrary_lazy_" + processor + ".dat";
        for(auto const& dir : {std::filesystem::path(env), std::filesystem::path(env) / processor})
        {
            auto const logical = dir / name;
            if(!std::filesystem::exists(logical)
               && !std::filesystem::exists(logical.string() + ".zlib"))
                continue;
            auto loaded = TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(
                logical.string());
            auto master = std::dynamic_pointer_cast<ContractionLibrary>(loaded);
            if(master && master->initLibraryMapping(logical.string()))
                return master;
        }
        reason = "no " + name + " under " + env;
        return nullptr;
    }

    std::vector<SolutionCase> buildCases()
    {
        std::vector<SolutionCase> cases;
        // Indices come from a half-open range or from a file with one per line.
        std::vector<int> indices;
        if(char const* range = std::getenv("HIPBLASLT_SOLUTION_INDEX_RANGE"))
        {
            int begin = 0, end = 0;
            if(std::sscanf(range, "%d:%d", &begin, &end) != 2 || begin < 0 || end < begin)
            {
                std::cerr << "Invalid HIPBLASLT_SOLUTION_INDEX_RANGE " << range << std::endl;
                return cases;
            }
            for(int index = begin; index < end; ++index)
                indices.push_back(index);
        }
        else if(char const* list = std::getenv("HIPBLASLT_SOLUTION_INDEX_LIST"))
        {
            std::ifstream file(list);
            for(int index; file >> index;)
                indices.push_back(index);
        }
        if(indices.empty())
            return cases;
        std::string reason;
        auto        library  = loadLibrary(reason);
        auto        hardware = TensileLite::hip::GetCurrentDevice();
        if(!library || !hardware)
        {
            std::cerr << "Solution-index cases not generated: "
                      << (library ? "no TensileLite hardware description" : reason) << std::endl;
            return cases;
        }
        hipDeviceProp_t     props{};
        int                 device = 0;
        SolutionCaseOptions options;
        options.reductionBlocks = static_cast<int64_t>(envNumber(
            "HIPBLASLT_SOLUTION_INDEX_REDUCTION_BLOCKS", double(options.reductionBlocks)));
        options.alpha           = envNumber("HIPBLASLT_SOLUTION_INDEX_ALPHA", options.alpha);
        options.beta            = envNumber("HIPBLASLT_SOLUTION_INDEX_BETA", options.beta);
        options.gfx1250         = hipGetDevice(&device) == hipSuccess
                          && hipGetDeviceProperties(&props, device) == hipSuccess
                          && std::string(props.gcnArchName).rfind("gfx1250", 0) == 0;
        bool const secondaryDevice = std::getenv("HIPBLASLT_SOLUTION_INDEX_SECONDARY_DEVICE");
        for(int index : indices)
        {
            // Metadata only: this reads the solution's shard, not its code object.
            auto solution = library->resolveSolutionByIndex(index);
            if(!solution)
                continue;
            // The solution's own hardware predicate decides whether this device,
            // emulated or physical, can run it.
            // A solution without a CU-count requirement runs only on the
            // primary device; secondary devices exist for the CU-specific ones.
            if(secondaryDevice && !requiresCUCount(*solution, *hardware))
                continue;
            SolutionCase c;
            c.index  = index;
            c.kernel = solution->kernelName;
            if(!(*solution->hardwarePredicate)(*hardware))
            {
                std::ostringstream why;
                solution->hardwarePredicate->debugEval(*hardware, why);
                c.plan = Skip{"hardware predicate rejects this device: " + why.str()};
            }
            else
            {
                auto made = solutionIndexCase(index, *solution, options);
                if(auto const* arg = std::get_if<Arguments>(&made))
                    c.plan = *arg;
                else
                    c.plan = Skip{std::get<UnsupportedSolution>(made).reason};
            }
            cases.push_back(std::move(c));
        }
        return cases;
    }

    // With rocjitsu's file sinks (RJ_SINKS=file, RJ_SINK_DIR), each plugin
    // appends to <dir>/<plugin>.log. Tests run one at a time, so the text a
    // plugin appends during a test belongs to that test.
    class PluginLog
    {
    public:
        explicit PluginLog(char const* plugin)
        {
            if(char const* dir = std::getenv("RJ_SINK_DIR"))
                path = std::filesystem::path(dir) / (std::string(plugin) + ".log");
            start = size();
        }

        bool enabled() const
        {
            return !path.empty();
        }

        std::string appended() const
        {
            std::ifstream file(path, std::ios::binary);
            if(!file)
                return {};
            file.seekg(static_cast<std::streamoff>(start));
            return std::string(std::istreambuf_iterator<char>(file), {});
        }

    private:
        std::uintmax_t size() const
        {
            std::error_code error;
            auto            bytes = path.empty() ? 0 : std::filesystem::file_size(path, error);
            return error ? 0 : bytes;
        }

        std::filesystem::path path;
        std::uintmax_t        start = 0;
    };

    // The race plugin's log names each dispatch and reports each race on a line
    // starting "RACE ".
    size_t countDispatches(std::string const& log, std::string const& kernel)
    {
        std::string const needle = "Kernel dispatch: \"" + kernel + "\"";
        size_t            count  = 0;
        for(size_t at = log.find(needle); at != std::string::npos; at = log.find(needle, at + 1))
            ++count;
        return count;
    }

    std::string raceReports(std::string const& log)
    {
        std::istringstream lines(log);
        std::string        line, reports;
        while(std::getline(lines, line))
            if(line.rfind("RACE ", 0) == 0)
                reports += line + "\n";
        return reports;
    }

    class SolutionIndexTest : public ::testing::TestWithParam<SolutionCase>
    {
    };

    TEST_P(SolutionIndexTest, matmul)
    {
        auto const& c = GetParam();
        if(auto const* skip = std::get_if<Skip>(&c.plan))
            GTEST_SKIP() << "index " << c.index << ": " << skip->reason;
        auto const& arg = std::get<Arguments>(c.plan);
        PluginLog   race("race");
        testing_matmul(arg);
        if(race.enabled())
        {
            std::string const log = race.appended();
            EXPECT_EQ(raceReports(log), "") << "rocjitsu reported races";
            EXPECT_EQ(countDispatches(log, c.kernel), 1u)
                << "the selected kernel must run exactly once";
        }
    }

    INSTANTIATE_TEST_SUITE_P(Library,
                             SolutionIndexTest,
                             ::testing::ValuesIn(buildCases()),
                             [](::testing::TestParamInfo<SolutionCase> const& info) {
                                 return "idx" + std::to_string(info.param.index);
                             });
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(SolutionIndexTest);
}
