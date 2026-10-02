// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-process.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace debug = hipblaslt_jit::debug;
namespace fs    = std::filesystem;
using Environment = std::vector<std::pair<std::string, std::string>>;

namespace
{
    const std::string prefix = "hipblaslt jit-debug ";

    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    std::string read(const fs::path& path)
    {
        std::ifstream in(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    }

    std::vector<std::string> split(const std::string& text)
    {
        std::vector<std::string> lines;
        std::istringstream       in(text);
        for(std::string line; std::getline(in, line);)
            lines.push_back(line);
        return lines;
    }

    // The debug lines of text, each checked for the common keys.
    std::vector<std::string> debugLines(const std::string& text)
    {
        std::vector<std::string> lines;
        for(const auto& line : split(text))
        {
            if(line.rfind(prefix, 0) != 0)
                continue;
            require(line.rfind(prefix + "{\"v\":1,\"cat\":\"", 0) == 0 && line.back() == '}'
                        && line.find(",\"ev\":\"") != std::string::npos
                        && line.find(",\"pid\":") != std::string::npos
                        && line.find(",\"tid\":") != std::string::npos
                        && line.find(",\"t_ms\":") != std::string::npos
                        && line.find(",\"q\":") != std::string::npos
                        && line.size() <= 4096,
                    "Malformed debug line: " + line);
            lines.push_back(line);
        }
        return lines;
    }

    std::vector<std::string> otherLines(const std::string& text)
    {
        std::vector<std::string> lines;
        for(const auto& line : split(text))
            if(line.rfind(prefix, 0) != 0)
                lines.push_back(line);
        return lines;
    }

    std::string field(const std::string& line, const std::string& key)
    {
        const auto at = line.find("\"" + key + "\":");
        if(at == std::string::npos)
            return {};
        auto first = at + key.size() + 3;
        if(line[first] == '"')
            return line.substr(first + 1, line.find('"', first + 1) - first - 1);
        auto last = line.find_first_of(",}", first);
        return line.substr(first, last - first);
    }

    std::vector<std::string> events(const std::vector<std::string>& lines)
    {
        std::vector<std::string> out;
        for(const auto& line : lines)
            out.push_back(field(line, "ev"));
        return out;
    }

    const std::string& find(const std::vector<std::string>& lines,
                            const std::string&              event,
                            const std::string&              key   = {},
                            const std::string&              value = {})
    {
        for(const auto& line : lines)
            if(field(line, "ev") == event && (key.empty() || field(line, key) == value))
                return line;
        throw std::runtime_error("No " + event + " line");
    }

    bool contains(const std::string& text, const std::string& fragment)
    {
        return text.find(fragment) != std::string::npos;
    }

    size_t count(const std::vector<std::string>& lines, const std::string& event)
    {
        return std::count_if(lines.begin(), lines.end(), [&](const std::string& line) {
            return field(line, "ev") == event;
        });
    }

    struct Child
    {
        hipblaslt_jit::process::Result result;
        std::string                    log;
    };

    fs::path self;
    fs::path root;

    Child child(const std::string& scenario, const std::string& name, Environment overlay)
    {
        hipblaslt_jit::process::Request request;
        request.argv        = {self.string(), "--child", scenario, root.string()};
        request.environment = {{"HIPBLASLT_JIT", "1"},
                               {"HIPBLASLT_JIT_DEBUG", "all"},
                               {"HIPBLASLT_JIT_DEBUG_FILE", ""}};
        request.environment.insert(request.environment.end(), overlay.begin(), overlay.end());
        request.workingDirectory = root;
        request.logPath          = root / (name + ".log");
        auto result              = hipblaslt_jit::process::run(request);
        require(result.succeeded(), name + ": child failed: " + read(request.logPath));
        return {result, read(request.logPath)};
    }

    // Scenarios run in a child process, whose categories the environment sets.

    int linesScenario()
    {
        const unsigned categories = debug::categories();
        std::cout << "categories=" << categories << std::endl;
        if(!categories)
            return 0;
        {
            debug::Query query("c", 4, [] { return size_t(3); });
            debug::lap("override");
            debug::note("from.equality", 2);
            debug::note("from.jit", 1);
            debug::note("jit.hits", 1);
            {
                debug::Phase phase("equality");
            }
            debug::problem("NN f32 m=8");
            {
                debug::Generation generation(2);
                generation.started(5);
                debug::note("generated", 2);
                {
                    debug::Scope scope(&generation.solution(0, "K0"));
                    debug::Phase build("build");
                    debug::set("compile_hip", "[{\"name\":\"a.cpp\",\"ns\":1}]");
                }
                generation.built(0, "built", "");
                generation.solution(1, "K1");
                generation.built(1, "build_failed", "boom");
                generation.failure("build", "boom");
                generation.indexed(0, 7);
                generation.publishing(1);
                debug::note("fresh", 1);
                debug::note("published", 1);
                generation.published("ok", 1);
            }
            debug::lap("get_best");
            {
                debug::Query inner("cpp", 1, nullptr, false);
            }
            debug::lap("get_all");
        }
        require(debug::Context::current().query.empty(), "Query context leaked");
        if(categories & debug::Progress)
        {
            debug::Line(debug::Progress, "long").add("text", std::string(600, 'x')).write();
            debug::Line(debug::Progress, "escape").add("text", "a\"b\\c\nd\te\x01").write();
            debug::Line huge(debug::Progress, "huge");
            for(int i = 0; i < 10; ++i)
                huge.add("k" + std::to_string(i), std::string(500, 'h'));
            huge.write();
        }
        {
            debug::Phase outside("outside");
            debug::lap("outside");
            debug::note("outside", 1);
        }
        return 0;
    }

    int burstScenario()
    {
        debug::categories();
        for(int i = 0; i < 200; ++i)
            debug::Line(debug::Progress, "burst").write(debug::Rate::Limited);
        debug::Line(debug::Progress, "after").write();
        for(int i = 0; i < 100; ++i)
            debug::Query query("c", 1, nullptr, false);
        for(int i = 0; i < 5; ++i)
        {
            debug::Query query("matmul", 1, nullptr, false);
            query.aggregateBy("NN f32 m=8|index=-7");
        }
        return 0;
    }

    int appendScenario()
    {
        debug::categories();
        for(int i = 0; i < 500; ++i)
            debug::Line(debug::Progress, "append").add("i", i).write();
        return 0;
    }

    // Checks run in the test process, whose HIPBLASLT_JIT is unset.

    void parsing()
    {
        struct Case
        {
            const char* value;
            unsigned    categories;
            const char* unknown;
        };
        const Case cases[] = {
            {"", 0, ""},
            {"0", 0, "0"},
            {"1", 0, "1"},
            {"timing", debug::Timing, ""},
            {"progress", debug::Progress, ""},
            {"timing,progress", debug::Timing | debug::Progress, ""},
            {" Timing , progress ,timing", debug::Timing | debug::Progress, ""},
            {"TIMING", debug::Timing, ""},
            {",,timing,,", debug::Timing, ""},
            {"all", debug::All, ""},
            {"all,timing", debug::All, ""},
            {"bogus", 0, "bogus"},
            {"timing,bogus", debug::Timing, "bogus"},
            {"0,timing", debug::Timing, "0"},
            {"2, x ,progress", debug::Progress, "2,x"},
        };
        for(const auto& c : cases)
        {
            std::string unknown;
            require(debug::parse(c.value, unknown) == c.categories && unknown == c.unknown,
                    std::string("Wrong parse of \"") + c.value + "\": unknown \"" + unknown + "\"");
        }
        std::cout << "PASS HIPBLASLT_JIT_DEBUG names categories or all; numbers are unknown\n";
    }

    void disabled()
    {
        require(debug::categories() == 0, "Categories are on without HIPBLASLT_JIT");
        require(debug::innermost() == nullptr, "A record without a scope");
        {
            debug::Scope scope(nullptr);
            debug::Phase phase("x");
            debug::lap("x");
            debug::note("x", 1);
            require(debug::innermost() == nullptr, "A null scope set a record");
        }
        debug::Record record;
        {
            debug::Scope scope(&record);
            require(debug::innermost() == &record, "Scope did not set the record");
            debug::Phase phase("x");
            phase.stop();
            debug::note("n", 2);
            debug::note("n", 3);
        }
        require(debug::innermost() == nullptr && record.nanoseconds("x") == 0
                    && record.counted("n") == 5,
                "A phase timed with timing off, or notes were lost");
        std::cout << "PASS disabled categories record no durations and keep no scope\n";
    }

    void quoting()
    {
        bool cut = false;
        require(debug::Line::quote("a\"b\\c\nd\te\x01", &cut) == "\"a\\\"b\\\\c\\nd\\te\\u0001\""
                    && !cut,
                "Wrong escaping");
        const auto quoted = debug::Line::quote(std::string(600, 'x'), &cut);
        require(cut && quoted.size() == 514, "A long string was not cut at 512 bytes");
        cut = false;
        // A two-byte character straddling the limit is not split.
        const auto utf8 = debug::Line::quote(std::string(511, 'x') + "\xC3\xA9" + "y", &cut);
        require(cut && utf8.size() == 513, "A UTF-8 character was split");
        require(debug::milliseconds(1234567) == "1.234", "Wrong milliseconds");
        std::cout << "PASS strings are escaped and cut at 512 bytes on a character boundary\n";
    }

    void lines()
    {
        {
            const auto all   = child("lines", "lines-all", {});
            const auto lines = debugLines(all.log);
            const std::vector<std::string> expected{"process",
                                                    "query.start",
                                                    "generation.start",
                                                    "build.start",
                                                    "build.end",
                                                    "build.start",
                                                    "build.end",
                                                    "failure",
                                                    "publish.start",
                                                    "publish.done",
                                                    "solution",
                                                    "solution",
                                                    "generation",
                                                    "generation.end",
                                                    "query",
                                                    "query",
                                                    "query.end",
                                                    "long",
                                                    "escape",
                                                    "huge"};
            require(events(lines) == expected, "Wrong events with all:\n" + all.log);
            const auto& process = lines[0];
            require(field(process, "cat") == "timing" && field(process, "q") == "null"
                        && field(process, "mode") == "1"
                        && field(process, "categories") == "timing,progress"
                        && field(process, "destination") == "stderr"
                        && contains(process, "\"wall\":\"") && contains(process, "\"comgr_cache\":"),
                    "Wrong process line: " + process);
            const auto query = find(lines, "query", "api", "c");
            const auto id    = field(query, "q");
            require(contains(id, ".") && field(query, "api") == "c"
                        && field(query, "requested") == "4" && field(query, "returned") == "3"
                        && field(query, "problem") == "NN f32 m=8"
                        && contains(query, "\"from\":{\"equality\":2,\"jit\":1}")
                        && contains(query, "\"jit\":{\"hits\":1}")
                        && contains(query, "\"ns\":{\"total\":")
                        && contains(query, "\"override\":") && contains(query, "\"equality\":")
                        && contains(query, "\"get_best\":") && contains(query, "\"get_all\":")
                        && !contains(query, "outside"),
                    "Wrong query line: " + query);
            const auto inner = find(lines, "query", "api", "cpp");
            require(field(inner, "q") != id && field(inner, "returned") == "0"
                        && !contains(inner, "\"problem\""),
                    "The inner query shared the outer one's context: " + inner);
            const auto generation = find(lines, "generation");
            const auto gen        = field(generation, "gen");
            require(field(generation, "q") == id && contains(gen, ".g")
                        && field(query, "gen") == gen && field(generation, "requested") == "2"
                        && field(generation, "generated") == "2"
                        && field(generation, "failures") == "1"
                        && field(generation, "candidates") == "5"
                        && field(generation, "fresh") == "1"
                        && contains(generation, "\"ns\":{\"total\":")
                        && contains(generation, "\"build\":") && contains(generation, "\"other\":"),
                    "Wrong generation line: " + generation);
            const auto solution = find(lines, "solution");
            require(field(solution, "kernel") == "K0" && field(solution, "index") == "7"
                        && field(solution, "outcome") == "built"
                        && contains(solution, "\"compile_hip\":[{\"name\":\"a.cpp\",\"ns\":1}]")
                        && contains(solution, "\"ns\":{\"build\":"),
                    "Wrong solution line: " + solution);
            require(contains(find(lines, "build.end"), "\"ns\":{\"build\":")
                        && field(find(lines, "generation.end"), "outcome") == "partial"
                        && field(find(lines, "publish.done"), "fresh") == "1"
                        && field(find(lines, "query.end"), "returned") == "3",
                    "Wrong progress lines:\n" + all.log);
            for(const auto& line : lines)
            {
                const auto event = field(line, "ev");
                if(event == "process" || event == "long" || event == "escape" || event == "huge"
                   || line == inner)
                    require(field(line, "q") != id, "A line inside the query: " + line);
                else
                    require(field(line, "q") == id, "A line outside its query: " + line);
            }
            require(contains(find(lines, "long"), "\"truncated\":true")
                        && contains(find(lines, "escape"), R"("a\"b\\c\nd\te\u0001")")
                        && contains(find(lines, "huge"), "\"truncated\":true,\"oversize\":"),
                    "Long strings or lines were not cut:\n" + all.log);
            require(otherLines(all.log) == std::vector<std::string>{"categories=4294967295"},
                    "Unexpected output with all:\n" + all.log);
        }
        {
            const auto timing = child("lines", "lines-timing", {{"HIPBLASLT_JIT_DEBUG", "timing"}});
            const auto lines  = debugLines(timing.log);
            require(events(lines)
                            == std::vector<std::string>{"process",
                                                        "solution",
                                                        "solution",
                                                        "generation",
                                                        "query",
                                                        "query"}
                        && field(lines[0], "categories") == "timing",
                    "Wrong lines with timing:\n" + timing.log);
            for(const auto& line : lines)
                require(field(line, "cat") == "timing", "A progress line with timing: " + line);
        }
        {
            const auto progress
                = child("lines", "lines-progress", {{"HIPBLASLT_JIT_DEBUG", "Progress"}});
            const auto lines = debugLines(progress.log);
            require(lines.size() == 15 && field(lines[0], "cat") == "progress"
                        && !contains(progress.log, "\"ns\":"),
                    "Wrong lines with progress:\n" + progress.log);
            for(const auto& line : lines)
                require(field(line, "cat") == "progress", "A timing line with progress: " + line);
        }
        {
            const auto forced = child("lines", "lines-forced", {{"HIPBLASLT_JIT", "2"}});
            require(field(debugLines(forced.log).at(0), "mode") == "2", "Wrong forced mode");
        }
        std::cout << "PASS lines carry their query, generation and category\n";
    }

    void off()
    {
        const std::vector<std::pair<std::string, Environment>> silent{
            {"jit-unset", {{"HIPBLASLT_JIT", ""}}},
            {"jit-0", {{"HIPBLASLT_JIT", "0"}}},
            {"jit-0-bogus", {{"HIPBLASLT_JIT", "0"}, {"HIPBLASLT_JIT_DEBUG", "0,bogus"}}},
            {"debug-empty", {{"HIPBLASLT_JIT_DEBUG", ""}}},
            {"debug-spaces", {{"HIPBLASLT_JIT_DEBUG", " , "}}},
        };
        for(const auto& [name, environment] : silent)
        {
            const auto run = child("lines", "off-" + name, environment);
            require(run.log == "categories=0\n", name + ": output while off:\n" + run.log);
        }
        for(const char* value : {"0", "1", "bogus"})
        {
            const auto run = child("lines", std::string("warn-") + value, {{"HIPBLASLT_JIT_DEBUG", value}});
            require(run.log
                        == std::string("hipblaslt warning: HIPBLASLT_JIT_DEBUG=") + value
                               + ": ignoring " + value
                               + "; the value is timing, progress or all, comma-separated\n"
                                 "categories=0\n",
                    std::string(value) + ": wrong warning:\n" + run.log);
        }
        const auto mixed = child("lines", "warn-mixed", {{"HIPBLASLT_JIT_DEBUG", "0,timing"}});
        const auto other = otherLines(mixed.log);
        require(other.size() == 2 && contains(other[0], "HIPBLASLT_JIT_DEBUG=0,timing: ignoring 0;")
                    && debugLines(mixed.log).size() == 6,
                "Recognized tokens beside an unknown one were not applied:\n" + mixed.log);
        std::cout << "PASS no output without JIT, unset or empty; one warning for unknown tokens\n";
    }

    void files()
    {
        const auto dir = root / "files";
        fs::create_directories(dir);
        const auto pattern = (dir / "debug-%i.jsonl").string();
        const auto run     = child("append", "file-pid", {{"HIPBLASLT_JIT_DEBUG_FILE", pattern}});
        require(run.log.empty(), "Output on stderr with a file:\n" + run.log);
        std::vector<fs::path> created;
        for(const auto& entry : fs::directory_iterator(dir))
            created.push_back(entry.path());
        require(created.size() == 1 && created[0].filename().string().rfind("debug-", 0) == 0
                    && created[0].filename().string().find("%i") == std::string::npos,
                "%i was not replaced by the process ID");
        const auto lines = debugLines(read(created[0]));
        require(lines.size() == 501
                    && field(lines[0], "destination") == created[0].string()
                    && created[0].filename().string() == "debug-" + field(lines[0], "pid") + ".jsonl",
                "Wrong file lines");
#ifndef _WIN32
        struct stat info{};
        require(stat(created[0].c_str(), &info) == 0 && (info.st_mode & 0777) == 0600,
                "The debug file is not 0600");
#endif
        const auto shared = (dir / "shared.jsonl").string();
        std::ofstream(shared) << "existing\n";
        std::vector<std::thread> threads;
        std::string              errors[2];
        for(int i = 0; i < 2; ++i)
            threads.emplace_back([&, i] {
                try
                {
                    child("append",
                          "file-shared-" + std::to_string(i),
                          {{"HIPBLASLT_JIT_DEBUG_FILE", shared}});
                }
                catch(const std::exception& error)
                {
                    errors[i] = error.what();
                }
            });
        for(auto& thread : threads)
            thread.join();
        require(errors[0].empty() && errors[1].empty(), errors[0] + errors[1]);
        const auto text = read(shared);
        require(text.rfind("existing\n", 0) == 0 && debugLines(text).size() == 1002
                    && otherLines(text).size() == 1,
                "Two processes did not append whole lines to one file");
        const auto fallback = child("append",
                                    "file-fallback",
                                    {{"HIPBLASLT_JIT_DEBUG_FILE", (dir / "absent" / "x").string()}});
        const auto warnings = otherLines(fallback.log);
        require(warnings.size() == 1 && contains(warnings[0], "HIPBLASLT_JIT_DEBUG_FILE=")
                    && contains(warnings[0], "cannot be opened")
                    && debugLines(fallback.log).size() == 501
                    && field(debugLines(fallback.log)[0], "destination") == (dir / "absent" / "x").string(),
                "No fallback to stderr:\n" + fallback.log);
        std::cout << "PASS HIPBLASLT_JIT_DEBUG_FILE appends whole lines, expands %i, is 0600\n";
    }

    void limits()
    {
        const auto run   = child("burst", "burst", {});
        const auto lines = debugLines(run.log);
        const auto burst = count(lines, "burst");
        require(burst >= 50 && burst < 60, "The burst was not limited: " + std::to_string(burst));
        size_t at = 0;
        while(field(lines[at], "ev") != "after")
            ++at;
        require(field(lines[at - 1], "ev") == "suppressed" && field(lines[at - 1], "cat") == "progress"
                    && std::stoul(field(lines[at - 1], "count")) == 200 - burst,
                "No suppressed count before the next line:\n" + run.log);
        size_t queries = count(lines, "query"), aggregated = 0, suppressed = 0, matmuls = 0;
        for(const auto& line : lines)
        {
            if(field(line, "ev") == "query.aggregate")
                aggregated += std::stoul(field(line, "calls"));
            if(field(line, "ev") == "matmul.aggregate")
                matmuls += std::stoul(field(line, "calls"));
            if(field(line, "ev") == "suppressed" && field(line, "cat") == "timing")
                suppressed += std::stoul(field(line, "count"));
        }
        require(queries + aggregated == 100 && suppressed == aggregated && aggregated > 0,
                "Dropped queries were not aggregated:\n" + run.log);
        require(count(lines, "matmul") == 1 && matmuls == 4,
                "Matmul calls were not aggregated after the first:\n" + run.log);
        std::cout << "PASS limited lines are counted, dropped queries and repeat matmuls aggregated\n";
    }
}

int main(int argc, char** argv)
{
    try
    {
        if(argc == 4 && std::string(argv[1]) == "--child")
        {
            const std::string scenario = argv[2];
            if(scenario == "lines")
                return linesScenario();
            if(scenario == "burst")
                return burstScenario();
            if(scenario == "append")
                return appendScenario();
            return 2;
        }
        if(argc != 2)
        {
            std::cerr << "Usage: " << argv[0] << " FRESH_OUTPUT_DIRECTORY\n";
            return 2;
        }
#ifdef _WIN32
        _putenv_s("HIPBLASLT_JIT", "");
        _putenv_s("HIPBLASLT_JIT_DEBUG", "");
        _putenv_s("HIPBLASLT_JIT_DEBUG_FILE", "");
#else
        unsetenv("HIPBLASLT_JIT");
        unsetenv("HIPBLASLT_JIT_DEBUG");
        unsetenv("HIPBLASLT_JIT_DEBUG_FILE");
#endif
#ifdef __linux__
        self = fs::read_symlink("/proc/self/exe");
#else
        self = fs::absolute(argv[0]);
#endif
        root = fs::absolute(argv[1]);
        require(fs::create_directories(root), "Output directory already exists");
        parsing();
        disabled();
        quoting();
        lines();
        off();
        files();
        limits();
        std::cout << "ALL JIT DEBUG CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
