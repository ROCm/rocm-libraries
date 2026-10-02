// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

// HIPBLASLT_JIT_DEBUG: timing and progress lines, "hipblaslt jit-debug {json}",
// on stderr or appended to HIPBLASLT_JIT_DEBUG_FILE. Categories are on only
// while HIPBLASLT_JIT selects mode 1 or 2; until then nothing here reads a
// clock, allocates or writes.
//
// Code compiled with and without JIT uses the macros, which expand to nothing
// without HIPBLASLT_ENABLE_JIT.
#ifdef HIPBLASLT_ENABLE_JIT
#define HIPBLASLT_JIT_DEBUG_LAP(name) ::hipblaslt_jit::debug::lap(name)
#define HIPBLASLT_JIT_DEBUG_NOTE(name, value) ::hipblaslt_jit::debug::note(name, value)
#else
#define HIPBLASLT_JIT_DEBUG_LAP(name) static_cast<void>(0)
#define HIPBLASLT_JIT_DEBUG_NOTE(name, value) static_cast<void>(0)
#endif

namespace hipblaslt_jit::debug
{
    enum Category : unsigned
    {
        Timing   = 1u << 0,
        Progress = 1u << 1,
        All      = ~0u, // includes categories added later
    };

    // Parses a HIPBLASLT_JIT_DEBUG value: category names or "all", separated by
    // commas, in any case and order, with spaces around each. Returns the
    // categories named. unknown receives every other non-empty token,
    // numbers included, separated by commas.
    unsigned parse(std::string_view value, std::string& unknown);

    // The categories of this process, read once from HIPBLASLT_JIT_DEBUG: zero
    // unless hipblaslt_jit::mode() is Fallback or Forced, and in privileged
    // processes. The first call warns once about unknown tokens and, when a
    // category is on, writes the process line.
    unsigned categories() noexcept;

    inline bool on(Category category) noexcept
    {
        return (categories() & category) != 0;
    }

    using Clock = std::chrono::steady_clock;

    inline uint64_t since(Clock::time_point start) noexcept
    {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
    }

    // The time from which lines count t_ms.
    Clock::time_point origin() noexcept;

    // The query and generation a line belongs to; empty outside one.
    struct Context
    {
        std::string query;
        std::string generation;
        unsigned    thread = 0; // 0: the writing thread

        static Context current();
    };

    enum class Rate
    {
        Always,
        Limited, // shares a process-wide budget of 50 lines, refilled at 10 per second
    };

    // One line. Strings longer than 512 bytes are cut and the line is marked
    // "truncated"; a line longer than 4 KiB keeps only its common keys.
    class Line
    {
    public:
        Line(Category category, std::string_view event);
        Line(Category category, std::string_view event, const Context& context);

        Line& add(std::string_view key, std::string_view value);
        Line& add(std::string_view key, const char* value)
        {
            return add(key, std::string_view(value));
        }
        Line& add(std::string_view key, const std::string& value)
        {
            return add(key, std::string_view(value));
        }
        Line& add(std::string_view key, bool value);
        template <class T,
                  std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
        Line& add(std::string_view key, T value)
        {
            return json(key, std::to_string(value));
        }
        // value is already JSON.
        Line& json(std::string_view key, std::string_view value);

        // The line as written, with its prefix and newline.
        std::string text() const;
        // False when the line was dropped by the Limited budget.
        bool write(Rate rate = Rate::Always) noexcept;

        // A JSON string for value, cut like a line's strings.
        static std::string quote(std::string_view value, bool* truncated = nullptr);

    private:
        Category    m_category;
        std::string m_text;
        size_t      m_common; // length of the common keys, kept for an oversize line
        bool        m_truncated = false;
    };

    // Milliseconds with microsecond digits, as JSON.
    std::string milliseconds(uint64_t nanoseconds);

    // Durations and fields that a query, generation or solution line reports.
    // A name "group.key" is written as "group":{"key":...}.
    class Record
    {
    public:
        void     add(const std::string& name, uint64_t nanoseconds); // accumulates into ns
        void     count(const std::string& name, int64_t value); // accumulates
        void     set(const std::string& name, std::string json); // replaces
        uint64_t nanoseconds(const std::string& name) const noexcept;
        int64_t  counted(const std::string& name) const noexcept;
        bool     has(const std::string& name) const noexcept;
        // Appends the fields, then "ns" with first, then every duration in first-use order.
        void write(Line& line, const std::pair<const char*, uint64_t>* first = nullptr) const;

        // Set by lap(): the end of the previous lap.
        bool              lapping = false;
        Clock::time_point lapped;

    private:
        struct Field
        {
            std::string name, json;
            int64_t     count   = 0;
            bool        counted = false;
        };
        std::vector<std::pair<std::string, uint64_t>> m_durations;
        std::vector<Field>                            m_fields;
    };

    // Makes record the innermost one on this thread while it lives. Null does nothing.
    class Scope
    {
    public:
        explicit Scope(Record* record) noexcept;
        ~Scope();
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;

    private:
        Record* m_outer  = nullptr;
        bool    m_active = false;
    };

    // The innermost record on this thread, or nullptr.
    Record* innermost() noexcept;

    // Adds the time until destruction or stop() to name in the innermost record
    // with timing on. Reads no clock otherwise.
    class Phase
    {
    public:
        explicit Phase(const char* name) noexcept;
        ~Phase();
        Phase(const Phase&)            = delete;
        Phase& operator=(const Phase&) = delete;
        void stop() noexcept;

    private:
        Record*           m_record = nullptr;
        const char*       m_name   = nullptr;
        Clock::time_point m_start;
    };

    // Adds the time since the innermost record's previous lap, or its start, to
    // name. Does nothing unless that record laps (queries do) and timing is on.
    void lap(const char* name) noexcept;
    // Adds value to the field name of the innermost record.
    void note(const char* name, int64_t value) noexcept;
    // Sets the field name of the innermost record to a JSON value.
    void set(const char* name, std::string json) noexcept;
    // Describes the problem of the innermost query for its lines and generations.
    void problem(const std::string& description) noexcept;

    // A heuristic query or hipblasLtMatmul call. Timing: a "query" line, or a
    // "matmul" line when api is "matmul". Progress: query.start and query.end.
    class Query
    {
    public:
        // returned reports the result count when the query ends.
        Query(const char* api, int requested, std::function<size_t()> returned, bool progress = true);
        ~Query();
        Query(const Query&)            = delete;
        Query& operator=(const Query&) = delete;

        Record& record() noexcept
        {
            return m_record;
        }
        // A matmul line is written for the first call with key in the process,
        // and for calls that generated or are notable; later calls fold into a
        // "matmul.aggregate" line per key, at most one per second.
        void aggregateBy(std::string key)
        {
            m_key = std::move(key);
        }
        void notable() noexcept
        {
            m_notable = true;
        }

    private:
        Record                  m_record;
        std::string             m_api;
        int                     m_requested;
        std::function<size_t()> m_returned;
        bool                    m_progress;
        std::string             m_key;
        bool                    m_notable = false;
        Clock::time_point       m_start;
        std::string             m_outerQuery, m_outerProblem;
        Scope                   m_scope;
    };

    // The ID, "<pid>.g<n>", of the last generation on this thread; empty before one.
    std::string lastGeneration();

    // One Jit::generate call. Timing: a "solution" line per generated solution
    // and a "generation" line. Progress: generation.start, build, publish, load,
    // failure and generation.end events.
    class Generation
    {
    public:
        explicit Generation(size_t requested);
        ~Generation();
        Generation(const Generation&)            = delete;
        Generation& operator=(const Generation&) = delete;

        const std::string& id() const noexcept
        {
            return m_id;
        }
        Record& record() noexcept
        {
            return m_record;
        }
        // Writes generation.start; candidates is 0 without a prediction.
        void started(size_t candidates);
        // The record of the solution at rank, for a Scope around its build and
        // support. Writes build.start.
        Record& solution(size_t rank, const std::string& kernel);
        // Sets the solution's outcome; built() also writes build.end.
        void built(size_t rank, const char* outcome, const std::string& message);
        void outcome(size_t rank, const char* outcome, const std::string& message = {});
        void indexed(size_t rank, int32_t index);
        void    failure(const char* stage, const std::string& message);
        // publish.start; publish.done with the lock wait and the fresh and
        // reused counts the store noted; load.done.
        void publishing(size_t solutions);
        void published(const char* status, size_t solutions);
        void loaded(size_t solutions);

    private:
        struct Solution
        {
            size_t      rank = 0;
            std::string kernel, status = "pending", message;
            int32_t     index = -1;
            Record      record;
        };
        Record                                 m_record;
        Record*                                m_outer;
        std::string                            m_id, m_outerGeneration, m_problem;
        size_t                                 m_requested;
        size_t                                 m_failures = 0;
        Clock::time_point                      m_start;
        std::vector<std::unique_ptr<Solution>> m_solutions;
        Scope                                  m_scope;
    };
}
