// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-debug-child.hpp"
#include "hipblaslt-jit-process.hpp"

#include <algorithm>
#include <chrono>
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

namespace debug = hipblaslt_jit::debug;
namespace fs    = std::filesystem;
using namespace std::chrono_literals;
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

    // The scenario runs in a child process, whose categories the environment sets.

    int observerScenario(const fs::path& dir)
    {
        debug::categories();
        std::cout << "child-categories " << debug::childCategories() << std::endl;
        const auto report = [](const char* name, const debug::ChildObserver& observer) {
            std::cout << name << " q=" << debug::Context::current().query
                      << " events=" << observer.events() << " dropped=" << observer.dropped()
                      << std::endl;
        };
        {
            debug::Query  query("live", 0, nullptr, false);
            const auto    path = dir / "live.jsonl";
            std::ofstream out(path, std::ios::binary);
            debug::ChildObserver observer(path);
            out << R"({"v":1,"seq":1,"pid":42,"mono_ns":1,"kind":"request","module":"M"})"
                << '\n'
                << std::flush;
            std::this_thread::sleep_for(300ms);
            std::cout << "live-before-stop events=" << observer.events() << std::endl;
            out << R"({"v":1,"seq":2,"pid":42,"kind":"stage","stage":"select","phase":"start"})"
                << "\nnot json\n"
                << R"({"v":2,"seq":3,"kind":"stage"})" << '\n'
                << R"({"v":1,"seq":2,"kind":"stage"})" << '\n'
                << std::string(5000, 'z') << '\n'
                << R"({"v":1,"seq":3,"kind":"BAD"})" << '\n'
                << R"({"v":1,"seq":4,"pid":42,"kind":"stage","stage":"select","phase":"end","ns":5})"
                << '\n'
                << R"({"v":1,"seq":5,"kind":"done")" << std::flush;
            std::this_thread::sleep_for(150ms);
            observer.stop();
            observer.stop();
            report("live", observer);
        }
        {
            debug::Query         query("missing", 0, nullptr, false);
            debug::ChildObserver observer(dir / "absent.jsonl");
            std::this_thread::sleep_for(120ms);
            observer.stop();
            report("missing", observer);
        }
        {
            debug::Query query("candidates", 0, nullptr, false);
            const auto   path = dir / "candidates.jsonl";
            {
                std::ofstream out(path, std::ios::binary);
                const char*   outcomes[] = {"rejected", "rejected", "selected", "rejected"};
                for(int i = 0; i < 4; ++i)
                    out << "{\"v\":1,\"seq\":" << i + 1 << ",\"kind\":\"candidate\",\"outcome\":\""
                        << outcomes[i] << "\",\"reason\":\"tensile\"}\n";
            }
            debug::ChildObserver observer(path);
            observer.stop();
            report("candidates", observer);
        }
        {
            debug::Query query("cap", 0, nullptr, false);
            const auto   path = dir / "cap.jsonl";
            {
                std::ofstream out(path, std::ios::binary);
                for(int i = 1; i <= 10001; ++i)
                    out << "{\"v\":1,\"seq\":" << i
                        << ",\"kind\":\"candidate\",\"outcome\":\"rejected\"}\n";
            }
            debug::ChildObserver observer(path);
            observer.stop();
            report("cap", observer);
        }
#ifndef _WIN32
        {
            debug::Query         query("killed", 0, nullptr, false);
            const auto           path = dir / "killed.jsonl";
            debug::ChildObserver observer(path);
            hipblaslt_jit::process::Request request;
            request.argv = {"/bin/sh",
                            "-c",
                            "printf '%s\\n%s\\n%s' \"$2\" \"$3\" \"$4\" >> \"$1\"; kill -9 $$",
                            "sh",
                            path.string(),
                            R"({"v":1,"seq":1,"kind":"request"})",
                            R"({"v":1,"seq":2,"kind":"stage","stage":"select","phase":"start"})",
                            R"({"v":1,"seq":3,"kind":"cand)"};
            request.workingDirectory = dir;
            request.logPath          = dir / "killed.log";
            const auto result        = hipblaslt_jit::process::run(request);
            observer.stop();
            std::cout << "killed-signal " << result.terminationSignal << std::endl;
            report("killed", observer);
        }
#endif
        try
        {
            debug::ChildObserver observer(dir / "unwind.jsonl");
            throw std::runtime_error("unwind");
        }
        catch(const std::runtime_error&)
        {
            std::cout << "unwind ok" << std::endl;
        }
        {
            debug::Query  query("heartbeat", 0, nullptr, false);
            const auto    path = dir / "heartbeat.jsonl";
            std::ofstream out(path, std::ios::binary);
            out << R"({"v":1,"seq":1,"kind":"stage","stage":"select","phase":"start"})" << '\n'
                << std::flush;
            debug::ChildObserver observer(path);
            std::this_thread::sleep_for(10500ms);
            observer.stop();
            report("heartbeat", observer);
        }
        return 0;
    }

    // Checks run in the test process, whose HIPBLASLT_JIT is unset.

    void categories()
    {
        require(debug::childCategories().empty(), "Child categories without HIPBLASLT_JIT");
        std::cout << "PASS a generator child gets no --debug value without HIPBLASLT_JIT\n";
    }

    void childTiming()
    {
        std::string why;
        require(debug::childTiming(root / "no-timing.json", 1, why).empty() && why == "missing",
                "A missing timing.json was not missing");
        std::ofstream(root / "bad-timing.json") << "{\"v\":1,\"total_ns\":";
        require(debug::childTiming(root / "bad-timing.json", 1, why).empty() && why == "invalid",
                "A truncated timing.json was not invalid");
        std::ofstream(root / "v2-timing.json") << "{\"v\":2,\"total_ns\":5}";
        require(debug::childTiming(root / "v2-timing.json", 1, why).empty() && why == "invalid",
                "A version 2 timing.json was accepted");
        std::ofstream(root / "timing.json")
            << R"({"v":1,"producer":"Tensile.JitGemm","status":"ok","total_ns":100,)"
            << R"("cpu_ns":80,"children_cpu_ns":null,"totals":{"select":30,"imports":20},)"
            << R"("spans":[{"name":"select","ns":30}]})";
        const auto timing = debug::childTiming(root / "timing.json", 150, why);
        require(why.empty()
                    && timing
                           == R"({"total":100,"cpu":80,"select":30,"imports":20,)"
                              R"("status":"ok","unattributed":50})",
                "Wrong child timing: " + timing);
        std::cout << "PASS timing.json is read, or reported missing or invalid\n";
    }

    void observer()
    {
        const auto dir = root / "observer";
        fs::create_directories(dir);
        const auto file = (dir / "observer.jsonl").string();
        const auto run
            = child("observer", "observer", {{"HIPBLASLT_JIT_DEBUG", "progress"}, {"HIPBLASLT_JIT_DEBUG_FILE", file}});
        const auto lines = debugLines(read(file));
        const auto out   = split(run.log);
        const auto line  = [&](const std::string& name) {
            for(const auto& l : out)
                if(l.rfind(name + " ", 0) == 0)
                    return l;
            throw std::runtime_error("No " + name + " report:\n" + run.log);
        };
        const auto query = [&](const std::string& name) {
            const auto l = line(name);
            return l.substr(l.find("q=") + 2, l.find(' ', l.find("q=")) - l.find("q=") - 2);
        };
        const auto relayed = [&](const std::string& name) {
            std::vector<std::string> mine;
            for(const auto& l : lines)
                if(field(l, "q") == query(name) && field(l, "ev").rfind("child.", 0) == 0)
                    mine.push_back(l);
            return mine;
        };
        require(line("child-categories") == "child-categories progress",
                "A child was not asked for progress: " + line("child-categories"));
        require(line("live-before-stop") == "live-before-stop events=1",
                "Events were not relayed while the child ran:\n" + run.log);
        require(contains(line("live"), "events=3 dropped=6"), "Wrong live counts: " + line("live"));
        const auto live = relayed("live");
        require(events(live) == std::vector<std::string>{"child.request", "child.stage", "child.stage"}
                    && field(live[0], "child_pid") == "42" && field(live[0], "module") == "M"
                    && field(live[0], "seq") == "1" && contains(live[0], "\"child_t_ms\":")
                    && field(live[2], "phase") == "end" && field(live[2], "ns") == "5",
                "Wrong relayed lines:\n" + read(file));
        require(contains(line("missing"), "events=0 dropped=0") && relayed("missing").empty(),
                "A missing events file was not ignored");
        const auto candidates = relayed("candidates");
        require(contains(line("candidates"), "events=4 dropped=0") && candidates.size() == 2
                    && field(candidates[0], "outcome") == "rejected"
                    && field(candidates[0], "rejected_so_far") == "1"
                    && field(candidates[1], "outcome") == "selected"
                    && field(candidates[1], "rejected_so_far") == "2",
                "Rejected candidates were not coalesced:\n" + read(file));
        require(contains(line("cap"), "events=10000 dropped=1") && relayed("cap").size() < 5,
                "The per-child cap was not applied: " + line("cap"));
#ifndef _WIN32
        require(line("killed-signal") == "killed-signal 9"
                    && contains(line("killed"), "events=2 dropped=1")
                    && relayed("killed").size() == 2,
                "A killed child's partial line was not dropped:\n" + run.log);
#endif
        require(std::find(out.begin(), out.end(), "unwind ok") != out.end(), "Unwinding failed");
        const auto beat = relayed("heartbeat");
        require(contains(line("heartbeat"), "events=1 dropped=0")
                    && count(beat, "child.heartbeat") == 1
                    && field(find(beat, "child.heartbeat"), "stage") == "select"
                    && std::stoul(field(find(beat, "child.heartbeat"), "silent_s")) >= 10,
                "No heartbeat after 10 s of silence:\n" + read(file));
        std::cout << "PASS the observer relays live events, drops malformed and partial lines\n";
    }
}

int main(int argc, char** argv)
{
    try
    {
        if(argc == 4 && std::string(argv[1]) == "--child")
            return std::string(argv[2]) == "observer"
                       ? observerScenario(fs::path(argv[3]) / "observer")
                       : 2;
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
        categories();
        childTiming();
        observer();
        std::cout << "ALL JIT DEBUG CHILD CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
