// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-debug.hpp"
#include "hipblaslt-jit-mode.hpp"
#include "rocblaslt_secure_env.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <share.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace hipblaslt_jit::debug
{
    namespace
    {
        namespace fs = std::filesystem;

        constexpr size_t           lineLimit   = 4096;
        constexpr size_t           stringLimit = 512;
        constexpr std::string_view prefix      = "hipblaslt jit-debug ";

        long processId() noexcept
        {
#ifdef _WIN32
            return _getpid();
#else
            return static_cast<long>(getpid());
#endif
        }

        // Trivially destructible, so lines written at exit can still read them.
        thread_local Record*  innermostRecord = nullptr;
        thread_local unsigned threadNumber    = 0;

        struct ThreadStrings
        {
            std::string query, generation, problem, last;
        };
        ThreadStrings& strings()
        {
            thread_local ThreadStrings value;
            return value;
        }

        std::atomic<unsigned> threadCount{0};
        std::atomic<uint64_t> queryCount{0};
        std::atomic<uint64_t> generationCount{0};

        unsigned thisThread() noexcept
        {
            if(!threadNumber)
                threadNumber = ++threadCount;
            return threadNumber;
        }

        struct Aggregate
        {
            std::string event, keyName, key;
            uint64_t    calls = 0, sum = 0, max = 0;
        };

        struct Sink
        {
            std::mutex                       mutex;
            Clock::time_point                origin = Clock::now();
            std::string                      path; // empty: stderr
            int                              descriptor    = -1;
            bool                             opened        = false;
            double                           tokens        = 50;
            Clock::time_point                refilled      = origin;
            uint64_t                         suppressed[2] = {0, 0}; // timing, progress
            Clock::time_point                suppressedSince[2];
            std::map<std::string, Aggregate> aggregates;
            std::set<std::string>            seen;
            Clock::time_point                flushed = origin;
        };

        Sink& sink()
        {
            // Never destroyed: threads can still write while the process exits.
            static auto* instance = new Sink;
            return *instance;
        }

        uint64_t sinceOrigin(const Sink& s, Clock::time_point when) noexcept
        {
            return when > s.origin
                       ? static_cast<uint64_t>(
                           std::chrono::duration_cast<std::chrono::nanoseconds>(when - s.origin)
                               .count())
                       : 0;
        }

        void writeAll(int descriptor, std::string_view text) noexcept
        {
            while(!text.empty())
            {
#ifdef _WIN32
                const auto written
                    = _write(descriptor, text.data(), static_cast<unsigned>(text.size()));
#else
                const auto written = ::write(descriptor, text.data(), text.size());
#endif
                if(written < 0)
                {
                    if(errno == EINTR)
                        continue;
                    return;
                }
                text.remove_prefix(static_cast<size_t>(written));
            }
        }

        // Callers hold s.mutex.
        void emit(Sink& s, const std::string& text) noexcept
        {
            if(!s.opened)
            {
                s.opened = true;
                if(!s.path.empty())
                {
#ifdef _WIN32
                    int       descriptor = -1;
                    const int error
                        = _wsopen_s(&descriptor,
                                    fs::u8path(s.path).c_str(),
                                    _O_WRONLY | _O_CREAT | _O_APPEND | _O_BINARY | _O_NOINHERIT,
                                    _SH_DENYNO,
                                    _S_IREAD | _S_IWRITE);
                    if(error)
                        descriptor = -1;
#else
                    const int descriptor
                        = ::open(s.path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
                    const int error = errno;
#endif
                    if(descriptor < 0)
                        std::cerr << "hipblaslt warning: HIPBLASLT_JIT_DEBUG_FILE=" << s.path
                                  << " cannot be opened (" << std::strerror(error)
                                  << "); JIT debug lines go to stderr" << std::endl;
                    s.descriptor = descriptor;
                }
            }
#ifdef _WIN32
            writeAll(s.descriptor >= 0 ? s.descriptor : _fileno(stderr), text);
#else
            writeAll(s.descriptor >= 0 ? s.descriptor : STDERR_FILENO, text);
#endif
        }

        // Callers hold s.mutex.
        bool take(Sink& s, Clock::time_point now) noexcept
        {
            const double elapsed = std::chrono::duration<double>(now - s.refilled).count();
            s.refilled           = now;
            s.tokens             = std::min(50.0, s.tokens + elapsed * 10.0);
            if(s.tokens < 1.0)
                return false;
            s.tokens -= 1.0;
            return true;
        }

        // Callers hold s.mutex.
        void flushSuppressed(Sink& s) noexcept
        {
            try
            {
                for(int i = 0; i < 2; ++i)
                {
                    if(!s.suppressed[i])
                        continue;
                    Line line(i == 0 ? Timing : Progress, "suppressed", Context{});
                    line.add("count", s.suppressed[i])
                        .json("since_ms", milliseconds(sinceOrigin(s, s.suppressedSince[i])));
                    s.suppressed[i] = 0;
                    emit(s, line.text());
                }
            }
            catch(...)
            {
            }
        }

        // Writes the aggregates when a second has passed since the last ones
        // or force is set. Callers hold s.mutex.
        void flushAggregates(Sink& s, Clock::time_point now, bool force) noexcept
        {
            try
            {
                if(s.aggregates.empty() || (!force && now - s.flushed < std::chrono::seconds(1)))
                    return;
                for(const auto& entry : s.aggregates)
                {
                    const auto& a = entry.second;
                    Line        line(Timing, a.event, Context{});
                    line.add(a.keyName, a.key)
                        .add("calls", a.calls)
                        .json("ns",
                              "{\"sum\":" + std::to_string(a.sum)
                                  + ",\"max\":" + std::to_string(a.max) + "}");
                    emit(s, line.text());
                }
                s.aggregates.clear();
                s.flushed = now;
            }
            catch(...)
            {
            }
        }

        struct Flusher
        {
            ~Flusher()
            {
                auto&                        s = sink();
                std::unique_lock<std::mutex> lock(s.mutex, std::try_to_lock);
                if(!lock)
                    return;
                flushSuppressed(s);
                flushAggregates(s, Clock::now(), true);
            }
        };

        void aggregate(const char*        event,
                       const char*        keyName,
                       const std::string& key,
                       uint64_t           nanoseconds) noexcept
        {
            try
            {
                auto&                       s = sink();
                std::lock_guard<std::mutex> lock(s.mutex);
                auto&                       a = s.aggregates[std::string(event) + '\n' + key];
                if(!a.calls)
                {
                    a.event   = event;
                    a.keyName = keyName;
                    a.key     = key;
                }
                ++a.calls;
                a.sum += nanoseconds;
                a.max = std::max(a.max, nanoseconds);
                flushAggregates(s, Clock::now(), false);
            }
            catch(...)
            {
            }
        }

        bool firstOf(const std::string& key)
        {
            auto&                       s = sink();
            std::lock_guard<std::mutex> lock(s.mutex);
            return s.seen.insert(key).second;
        }

        std::string names(unsigned categories)
        {
            std::string out;
            if(categories & Timing)
                out = "timing";
            if(categories & Progress)
                out += out.empty() ? "progress" : ",progress";
            return out;
        }

        int modeNumber() noexcept
        {
            return mode() == Mode::Forced ? 2 : 1;
        }

        std::string wallClock()
        {
            const auto now     = std::chrono::system_clock::now();
            const auto seconds = std::chrono::system_clock::to_time_t(now);
            const auto millis  = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    now.time_since_epoch())
                                    .count()
                                % 1000;
            std::tm utc{};
#ifdef _WIN32
            gmtime_s(&utc, &seconds);
#else
            gmtime_r(&seconds, &utc);
#endif
            char buffer[40];
            const size_t length = std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &utc);
            std::snprintf(buffer + length,
                          sizeof(buffer) - length,
                          ".%03dZ",
                          static_cast<int>(millis));
            return buffer;
        }

        std::string expandPid(std::string path)
        {
            const auto pid = std::to_string(processId());
            for(size_t at = path.find("%i"); at != std::string::npos; at = path.find("%i", at))
            {
                path.replace(at, 2, pid);
                at += pid.size();
            }
            return path;
        }

        void start(unsigned categories, Mode jitMode)
        {
            auto& s = sink();
            if(const char* file = rocblaslt_secure_getenv("HIPBLASLT_JIT_DEBUG_FILE"); file && *file)
                s.path = expandPid(file);
            static Flusher flusher;
            static_cast<void>(flusher);
            const char* cache = std::getenv("AMD_COMGR_CACHE");
            Line(categories & Timing ? Timing : Progress, "process", Context{})
                .add("mode", jitMode == Mode::Forced ? 2 : 1)
                .add("categories", names(categories))
                .add("destination", s.path.empty() ? std::string("stderr") : s.path)
                .add("wall", wallClock())
                .add("comgr_cache", cache ? cache : "unset")
                .write();
        }

        unsigned readCategories() noexcept
        {
            try
            {
                const auto jitMode = mode();
                if(jitMode == Mode::Off)
                    return 0;
                const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT_DEBUG");
                if(!value || !*value)
                    return 0;
                std::string    unknown;
                const unsigned result = parse(value, unknown);
                if(!unknown.empty())
                    std::cerr << "hipblaslt warning: HIPBLASLT_JIT_DEBUG=" << value
                              << ": ignoring " << unknown
                              << "; the value is timing, progress or all, comma-separated"
                              << std::endl;
                if(result)
                    start(result, jitMode);
                return result;
            }
            catch(...)
            {
                return 0;
            }
        }

        // A JSON value, as much of one as the observer and childTiming read.
        struct Value
        {
            enum class Kind
            {
                Null,
                Bool,
                Number,
                String,
                Array,
                Object,
            };
            Kind                     kind = Kind::Null;
            std::string              text; // a number's digits, a string, "true" or "false"
            std::vector<std::string> keys; // an object's, parallel to items
            std::vector<Value>       items;

            const Value* find(std::string_view key) const noexcept
            {
                if(kind != Kind::Object)
                    return nullptr;
                for(size_t i = 0; i < keys.size(); ++i)
                    if(keys[i] == key)
                        return &items[i];
                return nullptr;
            }
        };

        class Parser
        {
        public:
            explicit Parser(std::string_view text)
                : m_text(text)
            {
            }

            bool document(Value& out)
            {
                space();
                if(!value(out, 0))
                    return false;
                space();
                return m_at == m_text.size();
            }

        private:
            std::string_view m_text;
            size_t           m_at = 0;

            bool peek(char c) const noexcept
            {
                return m_at < m_text.size() && m_text[m_at] == c;
            }
            void space() noexcept
            {
                while(m_at < m_text.size()
                      && (m_text[m_at] == ' ' || m_text[m_at] == '\t' || m_text[m_at] == '\n'
                          || m_text[m_at] == '\r'))
                    ++m_at;
            }
            bool word(std::string_view w) noexcept
            {
                if(m_text.substr(m_at, w.size()) != w)
                    return false;
                m_at += w.size();
                return true;
            }
            bool digits() noexcept
            {
                const size_t first = m_at;
                while(m_at < m_text.size() && std::isdigit(static_cast<unsigned char>(m_text[m_at])))
                    ++m_at;
                return m_at > first;
            }
            bool value(Value& out, int depth)
            {
                if(depth > 32 || m_at >= m_text.size())
                    return false;
                switch(m_text[m_at])
                {
                case '{':
                    return object(out, depth);
                case '[':
                    return array(out, depth);
                case '"':
                    out.kind = Value::Kind::String;
                    return string(out.text);
                case 't':
                    out.kind = Value::Kind::Bool;
                    out.text = "true";
                    return word("true");
                case 'f':
                    out.kind = Value::Kind::Bool;
                    out.text = "false";
                    return word("false");
                case 'n':
                    out.kind = Value::Kind::Null;
                    return word("null");
                default:
                    out.kind = Value::Kind::Number;
                    return number(out.text);
                }
            }
            bool object(Value& out, int depth)
            {
                out.kind = Value::Kind::Object;
                ++m_at;
                space();
                if(peek('}'))
                    return ++m_at, true;
                for(;;)
                {
                    space();
                    std::string key;
                    if(!peek('"') || !string(key))
                        return false;
                    space();
                    if(!peek(':'))
                        return false;
                    ++m_at;
                    space();
                    Value item;
                    if(!value(item, depth + 1))
                        return false;
                    out.keys.push_back(std::move(key));
                    out.items.push_back(std::move(item));
                    space();
                    if(peek(','))
                    {
                        ++m_at;
                        continue;
                    }
                    if(peek('}'))
                        return ++m_at, true;
                    return false;
                }
            }
            bool array(Value& out, int depth)
            {
                out.kind = Value::Kind::Array;
                ++m_at;
                space();
                if(peek(']'))
                    return ++m_at, true;
                for(;;)
                {
                    space();
                    Value item;
                    if(!value(item, depth + 1))
                        return false;
                    out.items.push_back(std::move(item));
                    space();
                    if(peek(','))
                    {
                        ++m_at;
                        continue;
                    }
                    if(peek(']'))
                        return ++m_at, true;
                    return false;
                }
            }
            bool hex(unsigned& code) noexcept
            {
                if(m_at + 4 > m_text.size())
                    return false;
                code = 0;
                for(int i = 0; i < 4; ++i)
                {
                    const char c = m_text[m_at++];
                    code <<= 4;
                    if(c >= '0' && c <= '9')
                        code |= unsigned(c - '0');
                    else if(c >= 'a' && c <= 'f')
                        code |= unsigned(c - 'a' + 10);
                    else if(c >= 'A' && c <= 'F')
                        code |= unsigned(c - 'A' + 10);
                    else
                        return false;
                }
                return true;
            }
            static void utf8(unsigned code, std::string& out)
            {
                if(code < 0x80)
                    out += char(code);
                else if(code < 0x800)
                {
                    out += char(0xC0 | (code >> 6));
                    out += char(0x80 | (code & 0x3F));
                }
                else if(code < 0x10000)
                {
                    out += char(0xE0 | (code >> 12));
                    out += char(0x80 | ((code >> 6) & 0x3F));
                    out += char(0x80 | (code & 0x3F));
                }
                else
                {
                    out += char(0xF0 | (code >> 18));
                    out += char(0x80 | ((code >> 12) & 0x3F));
                    out += char(0x80 | ((code >> 6) & 0x3F));
                    out += char(0x80 | (code & 0x3F));
                }
            }
            bool string(std::string& out)
            {
                ++m_at;
                while(m_at < m_text.size())
                {
                    const char c = m_text[m_at++];
                    if(c == '"')
                        return true;
                    if(static_cast<unsigned char>(c) < 0x20)
                        return false;
                    if(c != '\\')
                    {
                        out += c;
                        continue;
                    }
                    if(m_at >= m_text.size())
                        return false;
                    switch(const char e = m_text[m_at++])
                    {
                    case '"':
                    case '\\':
                    case '/':
                        out += e;
                        break;
                    case 'b':
                        out += '\b';
                        break;
                    case 'f':
                        out += '\f';
                        break;
                    case 'n':
                        out += '\n';
                        break;
                    case 'r':
                        out += '\r';
                        break;
                    case 't':
                        out += '\t';
                        break;
                    case 'u':
                    {
                        unsigned code = 0;
                        if(!hex(code))
                            return false;
                        if(code >= 0xD800 && code < 0xDC00)
                        {
                            unsigned low = 0;
                            if(!word("\\u") || !hex(low) || low < 0xDC00 || low > 0xDFFF)
                                return false;
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        }
                        utf8(code, out);
                        break;
                    }
                    default:
                        return false;
                    }
                }
                return false;
            }
            bool number(std::string& out)
            {
                const size_t first = m_at;
                if(peek('-'))
                    ++m_at;
                if(!digits())
                    return false;
                if(peek('.'))
                {
                    ++m_at;
                    if(!digits())
                        return false;
                }
                if(peek('e') || peek('E'))
                {
                    ++m_at;
                    if(peek('+') || peek('-'))
                        ++m_at;
                    if(!digits())
                        return false;
                }
                out.assign(m_text.substr(first, m_at - first));
                return true;
            }
        };

        bool integer(const Value* value, int64_t& out) noexcept
        {
            if(!value || value->kind != Value::Kind::Number
               || value->text.find_first_of(".eE") != std::string::npos)
                return false;
            errno     = 0;
            char* end = nullptr;
            out       = std::strtoll(value->text.c_str(), &end, 10);
            return errno == 0 && end && *end == '\0';
        }

        bool count(const Value* value, uint64_t& out) noexcept
        {
            int64_t signedValue = 0;
            if(!integer(value, signedValue) || signedValue < 0)
                return false;
            out = static_cast<uint64_t>(signedValue);
            return true;
        }

        bool versionOne(const Value* value) noexcept
        {
            int64_t version = 0;
            return integer(value, version) && version == 1;
        }

        std::string serialize(const Value& value)
        {
            switch(value.kind)
            {
            case Value::Kind::Null:
                return "null";
            case Value::Kind::Bool:
            case Value::Kind::Number:
                return value.text;
            case Value::Kind::String:
                return Line::quote(value.text);
            case Value::Kind::Array:
            {
                std::string out = "[";
                for(size_t i = 0; i < value.items.size(); ++i)
                    out += (i ? "," : "") + serialize(value.items[i]);
                return out + "]";
            }
            case Value::Kind::Object:
            {
                std::string out = "{";
                for(size_t i = 0; i < value.items.size(); ++i)
                    out += (i ? "," : "") + Line::quote(value.keys[i]) + ":"
                           + serialize(value.items[i]);
                return out + "}";
            }
            }
            return "null";
        }
    }

    unsigned parse(std::string_view value, std::string& unknown)
    {
        unknown.clear();
        unsigned result = 0;
        for(size_t first = 0;;)
        {
            const size_t comma = value.find(',', first);
            auto         token = value.substr(first, comma == std::string_view::npos
                                                         ? std::string_view::npos
                                                         : comma - first);
            while(!token.empty() && std::isspace(static_cast<unsigned char>(token.front())))
                token.remove_prefix(1);
            while(!token.empty() && std::isspace(static_cast<unsigned char>(token.back())))
                token.remove_suffix(1);
            if(!token.empty())
            {
                std::string name(token);
                std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) {
                    return static_cast<char>(std::tolower(c));
                });
                if(name == "timing")
                    result |= Timing;
                else if(name == "progress")
                    result |= Progress;
                else if(name == "all")
                    result |= All;
                else
                    unknown += (unknown.empty() ? "" : ",") + std::string(token);
            }
            if(comma == std::string_view::npos)
                return result;
            first = comma + 1;
        }
    }

    unsigned categories() noexcept
    {
        static const unsigned value = readCategories();
        return value;
    }

    std::string childCategories()
    {
        return names(categories());
    }

    std::string milliseconds(uint64_t nanoseconds)
    {
        char buffer[32];
        std::snprintf(buffer,
                      sizeof(buffer),
                      "%llu.%03llu",
                      static_cast<unsigned long long>(nanoseconds / 1000000),
                      static_cast<unsigned long long>(nanoseconds / 1000 % 1000));
        return buffer;
    }

    Context Context::current()
    {
        const auto& t = strings();
        return {t.query, t.generation, thisThread()};
    }

    Line::Line(Category category, std::string_view event)
        : Line(category, event, Context::current())
    {
    }

    Line::Line(Category category, std::string_view event, const Context& context)
        : m_category(category)
    {
        m_text.reserve(256);
        m_text = "{\"v\":1,\"cat\":\"";
        m_text += category == Timing ? "timing" : "progress";
        m_text += "\",\"ev\":";
        m_text += quote(event);
        m_text += ",\"pid\":" + std::to_string(processId());
        m_text += ",\"tid\":" + std::to_string(context.thread ? context.thread : thisThread());
        m_text += ",\"t_ms\":" + milliseconds(sinceOrigin(sink(), Clock::now()));
        m_text += ",\"q\":";
        m_text += context.query.empty() ? std::string("null") : quote(context.query);
        if(!context.generation.empty())
            m_text += ",\"gen\":" + quote(context.generation);
        m_common = m_text.size();
    }

    Line& Line::add(std::string_view key, std::string_view value)
    {
        return json(key, quote(value, &m_truncated));
    }

    Line& Line::add(std::string_view key, bool value)
    {
        return json(key, value ? "true" : "false");
    }

    Line& Line::json(std::string_view key, std::string_view value)
    {
        m_text += ',';
        m_text += quote(key, &m_truncated);
        m_text += ':';
        m_text += value;
        return *this;
    }

    std::string Line::text() const
    {
        std::string body = m_text;
        if(m_truncated)
            body += ",\"truncated\":true";
        body += '}';
        if(prefix.size() + body.size() + 1 > lineLimit)
            body = m_text.substr(0, m_common) + ",\"truncated\":true,\"oversize\":"
                   + std::to_string(body.size()) + "}";
        std::string out;
        out.reserve(prefix.size() + body.size() + 1);
        out += prefix;
        out += body;
        out += '\n';
        return out;
    }

    bool Line::write(Rate rate) noexcept
    {
        try
        {
            const auto                  line = text();
            auto&                       s    = sink();
            std::lock_guard<std::mutex> lock(s.mutex);
            const auto                  now = Clock::now();
            if(rate == Rate::Limited && !take(s, now))
            {
                const int i = m_category == Timing ? 0 : 1;
                if(!s.suppressed[i]++)
                    s.suppressedSince[i] = now;
                return false;
            }
            flushSuppressed(s);
            flushAggregates(s, now, false);
            emit(s, line);
            return true;
        }
        catch(...)
        {
            return false;
        }
    }

    std::string Line::quote(std::string_view value, bool* truncated)
    {
        if(value.size() > stringLimit)
        {
            size_t end = stringLimit;
            while(end > 0 && (static_cast<unsigned char>(value[end]) & 0xC0) == 0x80)
                --end;
            value = value.substr(0, end);
            if(truncated)
                *truncated = true;
        }
        std::string out;
        out.reserve(value.size() + 2);
        out += '"';
        for(const char c : value)
        {
            switch(c)
            {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if(static_cast<unsigned char>(c) < 0x20 || c == 0x7F)
                {
                    char escaped[8];
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", unsigned(c) & 0xFF);
                    out += escaped;
                }
                else
                    out += c;
            }
        }
        out += '"';
        return out;
    }

    void Record::add(const std::string& name, uint64_t nanoseconds)
    {
        for(auto& duration : m_durations)
            if(duration.first == name)
            {
                duration.second += nanoseconds;
                return;
            }
        m_durations.emplace_back(name, nanoseconds);
    }

    void Record::count(const std::string& name, int64_t value)
    {
        for(auto& field : m_fields)
            if(field.name == name)
            {
                field.count   = (field.counted ? field.count : 0) + value;
                field.counted = true;
                return;
            }
        m_fields.push_back({name, {}, value, true});
    }

    void Record::set(const std::string& name, std::string json)
    {
        for(auto& field : m_fields)
            if(field.name == name)
            {
                field.json    = std::move(json);
                field.counted = false;
                return;
            }
        m_fields.push_back({name, std::move(json), 0, false});
    }

    uint64_t Record::nanoseconds(const std::string& name) const noexcept
    {
        for(const auto& duration : m_durations)
            if(duration.first == name)
                return duration.second;
        return 0;
    }

    int64_t Record::counted(const std::string& name) const noexcept
    {
        for(const auto& field : m_fields)
            if(field.name == name && field.counted)
                return field.count;
        return 0;
    }

    bool Record::has(const std::string& name) const noexcept
    {
        return std::any_of(m_fields.begin(), m_fields.end(), [&](const Field& field) {
            return field.name == name;
        });
    }

    void Record::write(Line& line, const std::pair<const char*, uint64_t>* first) const
    {
        const auto value = [](const Field& field) {
            return field.counted ? std::to_string(field.count) : field.json;
        };
        std::vector<bool> done(m_fields.size());
        for(size_t i = 0; i < m_fields.size(); ++i)
        {
            if(done[i])
                continue;
            const auto& name = m_fields[i].name;
            const auto  dot  = name.find('.');
            if(dot == std::string::npos)
            {
                line.json(name, value(m_fields[i]));
                continue;
            }
            const auto  group  = name.substr(0, dot + 1);
            std::string object = "{";
            for(size_t j = i; j < m_fields.size(); ++j)
                if(!done[j] && m_fields[j].name.compare(0, group.size(), group) == 0)
                {
                    done[j] = true;
                    object += (object.size() > 1 ? "," : "")
                              + Line::quote(m_fields[j].name.substr(group.size())) + ":"
                              + value(m_fields[j]);
                }
            line.json(name.substr(0, dot), object + "}");
        }
        if(!first && m_durations.empty())
            return;
        std::string ns = "{";
        if(first)
            ns += Line::quote(first->first) + ":" + std::to_string(first->second);
        for(const auto& duration : m_durations)
            ns += (ns.size() > 1 ? "," : "") + Line::quote(duration.first) + ":"
                  + std::to_string(duration.second);
        line.json("ns", ns + "}");
    }

    Scope::Scope(Record* record) noexcept
    {
        if(!record)
            return;
        m_outer         = innermostRecord;
        innermostRecord = record;
        m_active        = true;
    }

    Scope::~Scope()
    {
        if(m_active)
            innermostRecord = m_outer;
    }

    Record* innermost() noexcept
    {
        return innermostRecord;
    }

    Phase::Phase(const char* name) noexcept
    {
        Record* record = innermostRecord;
        if(!record || !(categories() & Timing))
            return;
        m_record = record;
        m_name   = name;
        m_start  = Clock::now();
    }

    Phase::~Phase()
    {
        stop();
    }

    void Phase::stop() noexcept
    {
        if(!m_record)
            return;
        try
        {
            m_record->add(m_name, since(m_start));
        }
        catch(...)
        {
        }
        m_record = nullptr;
    }

    void lap(const char* name) noexcept
    {
        Record* record = innermostRecord;
        if(!record || !record->lapping || !(categories() & Timing))
            return;
        try
        {
            const auto now = Clock::now();
            record->add(name,
                        static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(now - record->lapped)
                                .count()));
            record->lapped = now;
        }
        catch(...)
        {
        }
    }

    void note(const char* name, int64_t value) noexcept
    {
        if(Record* record = innermostRecord)
            try
            {
                record->count(name, value);
            }
            catch(...)
            {
            }
    }

    void set(const char* name, std::string json) noexcept
    {
        if(Record* record = innermostRecord)
            try
            {
                record->set(name, std::move(json));
            }
            catch(...)
            {
            }
    }

    void problem(const std::string& description) noexcept
    {
        try
        {
            auto& t = strings();
            if(!t.query.empty())
                t.problem = description;
        }
        catch(...)
        {
        }
    }

    std::string lastGeneration()
    {
        return strings().last;
    }

    Query::Query(const char*             api,
                 int                     requested,
                 std::function<size_t()> returned,
                 bool                    progress)
        : m_api(api)
        , m_requested(requested)
        , m_returned(std::move(returned))
        , m_progress(progress)
        , m_start(Clock::now())
        , m_scope(&m_record)
    {
        try
        {
            auto& t        = strings();
            m_outerQuery   = std::move(t.query);
            m_outerProblem = std::move(t.problem);
            t.query        = std::to_string(processId()) + "." + std::to_string(++queryCount);
            t.problem.clear();
            m_record.lapping = true;
            m_record.lapped  = m_start;
            if(m_progress && (categories() & Progress))
                Line(Progress, "query.start")
                    .add("api", m_api)
                    .add("requested", m_requested)
                    .write(Rate::Limited);
        }
        catch(...)
        {
        }
    }

    Query::~Query()
    {
        try
        {
            const auto   total    = since(m_start);
            const size_t returned = m_returned ? m_returned() : 0;
            const auto&  problem  = strings().problem;
            if(categories() & Timing)
            {
                const bool matmul = m_api == "matmul";
                Line       line(Timing, matmul ? "matmul" : "query");
                line.add("api", m_api).add("mode", modeNumber()).add("requested", m_requested);
                line.add("returned", returned);
                if(!problem.empty())
                    line.add("problem", problem);
                const std::pair<const char*, uint64_t> first{"total", total};
                m_record.write(line, &first);
                if(!m_key.empty())
                {
                    const bool first = firstOf("matmul\n" + m_key);
                    if(first || m_notable || m_record.has("gen"))
                        line.write();
                    else
                        aggregate("matmul.aggregate", "key", m_key, total);
                }
                else if(!line.write(Rate::Limited))
                    aggregate("query.aggregate", "api", m_api, total);
            }
            if(m_progress && (categories() & Progress))
            {
                Line line(Progress, "query.end");
                line.add("api", m_api).add("requested", m_requested).add("returned", returned);
                if(categories() & Timing)
                    line.json("ns", "{\"total\":" + std::to_string(total) + "}");
                line.write(Rate::Limited);
            }
        }
        catch(...)
        {
        }
        try
        {
            auto& t   = strings();
            t.query   = std::move(m_outerQuery);
            t.problem = std::move(m_outerProblem);
        }
        catch(...)
        {
        }
    }

    Generation::Generation(size_t requested)
        : m_outer(innermostRecord)
        , m_requested(requested)
        , m_start(Clock::now())
        , m_scope(&m_record)
    {
        try
        {
            auto& t = strings();
            m_id = std::to_string(processId()) + ".g" + std::to_string(++generationCount);
            m_outerGeneration = std::move(t.generation);
            t.generation      = m_id;
            m_problem         = t.problem;
            if(m_outer)
                m_outer->set("gen", Line::quote(m_id));
        }
        catch(...)
        {
        }
    }

    void Generation::started(size_t candidates)
    {
        if(candidates)
            m_record.count("candidates", static_cast<int64_t>(candidates));
        if(!(categories() & Progress))
            return;
        Line line(Progress, "generation.start");
        line.add("requested", m_requested);
        if(candidates)
            line.add("candidates", candidates);
        if(!m_problem.empty())
            line.add("problem", m_problem);
        line.write();
    }

    Record& Generation::solution(size_t rank, const std::string& kernel)
    {
        m_solutions.push_back(std::make_unique<Solution>());
        auto& s  = *m_solutions.back();
        s.rank   = rank;
        s.kernel = kernel;
        if(categories() & Progress)
            Line(Progress, "build.start").add("rank", rank).add("kernel", kernel).write();
        return s.record;
    }

    void Generation::outcome(size_t rank, const char* outcome, const std::string& message)
    {
        for(auto& s : m_solutions)
            if(s->rank == rank)
            {
                s->status  = outcome;
                s->message = message;
            }
    }

    void Generation::built(size_t rank, const char* outcome, const std::string& message)
    {
        this->outcome(rank, outcome, message);
        for(auto& s : m_solutions)
        {
            if(s->rank != rank)
                continue;
            if(!(categories() & Progress))
                return;
            Line line(Progress, "build.end");
            line.add("rank", rank).add("kernel", s->kernel).add("outcome", s->status);
            if(!message.empty())
                line.add("message", message);
            if(categories() & Timing)
                line.json("ns",
                          "{\"build\":" + std::to_string(s->record.nanoseconds("build")) + "}");
            line.write();
            return;
        }
    }

    void Generation::indexed(size_t rank, int32_t index)
    {
        for(auto& s : m_solutions)
            if(s->rank == rank)
                s->index = index;
    }

    void Generation::failure(const char* stage, const std::string& message)
    {
        ++m_failures;
        if(categories() & Progress)
            Line(Progress, "failure").add("stage", stage).add("message", message).write();
    }

    void Generation::publishing(size_t solutions)
    {
        if(categories() & Progress)
            Line(Progress, "publish.start").add("solutions", solutions).write();
    }

    void Generation::published(const char* status, size_t solutions)
    {
        if(!(categories() & Progress))
            return;
        Line line(Progress, "publish.done");
        line.add("status", status).add("solutions", solutions);
        for(const char* placement : {"fresh", "reused"})
            if(m_record.has(placement))
                line.add(placement, m_record.counted(placement));
        if(categories() & Timing)
            line.json("ns",
                      "{\"publish\":" + std::to_string(m_record.nanoseconds("publish"))
                          + ",\"lock_wait\":"
                          + std::to_string(m_record.nanoseconds("publish_lock_wait")) + "}");
        line.write();
    }

    void Generation::loaded(size_t solutions)
    {
        if(categories() & Progress)
            Line(Progress, "load.done").add("solutions", solutions).write();
    }

    Generation::~Generation()
    {
        try
        {
            const auto total     = since(m_start);
            const auto published = m_record.counted("published") + m_record.counted("loaded");
            if(categories() & Timing)
            {
                uint64_t build = 0, support = 0;
                for(const auto& s : m_solutions)
                {
                    build += s->record.nanoseconds("build");
                    support += s->record.nanoseconds("support");
                    Line line(Timing, "solution");
                    line.add("rank", s->rank).add("kernel", s->kernel).add("outcome", s->status);
                    if(s->index >= 0)
                        line.add("index", s->index);
                    if(!s->message.empty())
                        line.add("message", s->message);
                    s->record.write(line);
                    line.write();
                }
                if(build)
                    m_record.add("build", build);
                if(support)
                    m_record.add("support", support);
                uint64_t accounted = 0;
                for(const char* stage :
                    {"predict", "scratch", "backend", "build", "support", "publish", "load"})
                    accounted += m_record.nanoseconds(stage);
                m_record.add("other", total > accounted ? total - accounted : 0);
                Line line(Timing, "generation");
                line.add("requested", m_requested).add("failures", m_failures);
                if(!m_problem.empty())
                    line.add("problem", m_problem);
                const std::pair<const char*, uint64_t> first{"total", total};
                m_record.write(line, &first);
                line.write();
            }
            if(categories() & Progress)
            {
                Line line(Progress, "generation.end");
                line.add("outcome",
                         published > 0 ? (m_failures ? "partial" : "ok")
                                       : (m_failures ? "failed" : "empty"));
                line.add("generated", m_record.counted("generated"))
                    .add("published", m_record.counted("published"))
                    .add("loaded", m_record.counted("loaded"))
                    .add("failures", m_failures);
                if(categories() & Timing)
                    line.json("ns", "{\"total\":" + std::to_string(total) + "}");
                line.write();
            }
        }
        catch(...)
        {
        }
        try
        {
            auto& t      = strings();
            t.generation = std::move(m_outerGeneration);
            t.last       = m_id;
        }
        catch(...)
        {
        }
    }

    struct ChildObserver::State
    {
        fs::path                path;
        Context                 context;
        uint64_t                origin = 0; // the sink origin on the steady clock, in ns
        std::mutex              mutex;
        std::condition_variable wake;
        bool                    stopping = false;
        std::thread             thread;
        std::atomic<size_t>     accepted{0}, dropped{0};

        // The observer thread's own.
        std::FILE*        file = nullptr;
        std::string       partial;
        bool              skipping = false;
        bool              ordered  = false;
        int64_t           seq      = 0;
        Clock::time_point heard, beat;
        std::string       stage;
        bool              candidates = false;
        Clock::time_point candidate;
        size_t            rejected = 0;

        void run() noexcept
        {
            try
            {
                for(;;)
                {
                    bool stop = false;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        wake.wait_for(
                            lock, std::chrono::milliseconds(100), [this] { return stopping; });
                        stop = stopping;
                    }
                    drain();
                    if(stop)
                        break;
                    const auto now = Clock::now();
                    if(now - heard >= std::chrono::seconds(10)
                       && now - beat >= std::chrono::seconds(10))
                    {
                        beat = now;
                        Line line(Progress, "child.heartbeat", context);
                        line.add("silent_s",
                                 std::chrono::duration_cast<std::chrono::seconds>(now - heard)
                                     .count());
                        line.add("events", accepted.load());
                        if(!stage.empty())
                            line.add("stage", stage);
                        line.write();
                    }
                }
                if(!partial.empty() || skipping)
                    ++dropped;
            }
            catch(...)
            {
            }
            if(file)
                std::fclose(file);
            file = nullptr;
        }

        void drain()
        {
            if(!file)
            {
#ifdef _WIN32
                file = _wfopen(path.c_str(), L"rb");
#else
                file = std::fopen(path.c_str(), "rbe");
#endif
                if(!file)
                    return;
            }
            char buffer[16384];
            for(;;)
            {
                const size_t n = std::fread(buffer, 1, sizeof(buffer), file);
                feed(buffer, n);
                if(n < sizeof(buffer))
                    break;
            }
            std::clearerr(file);
        }

        void feed(const char* data, size_t size)
        {
            for(size_t i = 0; i < size; ++i)
            {
                const char c = data[i];
                if(skipping)
                {
                    skipping = c != '\n';
                    continue;
                }
                if(c == '\n')
                {
                    handle(partial);
                    partial.clear();
                }
                else if(partial.size() < lineLimit)
                    partial += c;
                else
                {
                    ++dropped;
                    partial.clear();
                    skipping = true;
                }
            }
        }

        static bool eventName(const std::string& name) noexcept
        {
            return !name.empty() && name.size() <= 32
                   && std::all_of(name.begin(), name.end(), [](char c) {
                          return (c >= 'a' && c <= 'z') || c == '_';
                      });
        }

        void handle(std::string_view text)
        {
            if(!text.empty() && text.back() == '\r')
                text.remove_suffix(1);
            if(text.empty())
                return;
            Value   event;
            int64_t number = 0;
            if(accepted >= 10000 || !Parser(text).document(event)
               || !versionOne(event.find("v")) || !integer(event.find("seq"), number)
               || (ordered && number <= seq))
            {
                ++dropped;
                return;
            }
            const Value* kind = event.find("kind");
            if(!kind || kind->kind != Value::Kind::String || !eventName(kind->text))
            {
                ++dropped;
                return;
            }
            ordered = true;
            seq     = number;
            heard   = Clock::now();
            ++accepted;
            relay(event, kind->text);
        }

        void relay(const Value& event, const std::string& kind)
        {
            const auto now = Clock::now();
            if(kind == "stage")
            {
                const Value* name  = event.find("stage");
                const Value* phase = event.find("phase");
                if(name && name->kind == Value::Kind::String && phase
                   && phase->kind == Value::Kind::String)
                    stage = phase->text == "start" ? name->text : std::string();
            }
            if(kind == "candidate")
            {
                const Value* outcome  = event.find("outcome");
                const bool   selected = outcome && outcome->kind == Value::Kind::String
                                      && outcome->text == "selected";
                if(!selected)
                {
                    ++rejected;
                    if(candidates && now - candidate < std::chrono::milliseconds(500))
                        return;
                }
                candidates = true;
                candidate  = now;
            }
            Line line(Progress, "child." + kind, context);
            if(const Value* pid = event.find("pid"); pid && pid->kind == Value::Kind::Number)
                line.json("child_pid", pid->text);
            line.add("seq", seq);
#ifndef _WIN32
            uint64_t mono = 0;
            if(count(event.find("mono_ns"), mono))
                line.json("child_t_ms", milliseconds(mono > origin ? mono - origin : 0));
#endif
            if(kind == "candidate")
                line.add("rejected_so_far", rejected);
            static const std::set<std::string_view> reserved{
                "v", "cat", "ev", "pid", "tid", "t_ms", "q", "gen", "seq", "mono_ns", "kind"};
            for(size_t i = 0; i < event.keys.size(); ++i)
                if(!reserved.count(event.keys[i]))
                    line.json(event.keys[i], serialize(event.items[i]));
            line.write();
        }
    };

    ChildObserver::ChildObserver(std::filesystem::path events)
        : m_state(std::make_shared<State>())
    {
        m_state->path    = std::move(events);
        m_state->context = Context::current();
        m_state->origin  = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(sink().origin.time_since_epoch())
                .count());
        m_state->heard = m_state->beat = Clock::now();
        m_state->thread                = std::thread([state = m_state] { state->run(); });
    }

    ChildObserver::~ChildObserver()
    {
        stop();
    }

    void ChildObserver::stop() noexcept
    {
        if(!m_state || !m_state->thread.joinable())
            return;
        try
        {
            {
                std::lock_guard<std::mutex> lock(m_state->mutex);
                m_state->stopping = true;
            }
            m_state->wake.notify_all();
            m_state->thread.join();
        }
        catch(...)
        {
        }
    }

    size_t ChildObserver::events() const noexcept
    {
        return m_state ? m_state->accepted.load() : 0;
    }

    size_t ChildObserver::dropped() const noexcept
    {
        return m_state ? m_state->dropped.load() : 0;
    }

    std::string childTiming(const std::filesystem::path& file,
                            uint64_t                     childNanoseconds,
                            std::string&                 why)
    {
        std::error_code error;
        const auto      size = fs::file_size(file, error);
        if(error)
        {
            why = "missing";
            return {};
        }
        why = "invalid";
        if(size > (1u << 20))
            return {};
        std::ifstream in(file, std::ios::binary);
        if(!in)
            return {};
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        Value             root;
        uint64_t          total = 0, n = 0;
        if(!Parser(text).document(root) || !versionOne(root.find("v"))
           || !count(root.find("total_ns"), total))
            return {};
        std::string out = "{\"total\":" + std::to_string(total);
        if(count(root.find("cpu_ns"), n))
            out += ",\"cpu\":" + std::to_string(n);
        if(count(root.find("children_cpu_ns"), n))
            out += ",\"children_cpu\":" + std::to_string(n);
        if(const Value* totals = root.find("totals"); totals && totals->kind == Value::Kind::Object)
            for(size_t i = 0; i < totals->items.size(); ++i)
                if(count(&totals->items[i], n))
                    out += "," + Line::quote(totals->keys[i]) + ":" + std::to_string(n);
        if(const Value* status = root.find("status"); status && status->kind == Value::Kind::String)
            out += ",\"status\":" + Line::quote(status->text);
        out += ",\"unattributed\":"
               + std::to_string(childNanoseconds > total ? childNanoseconds - total : 0) + "}";
        why.clear();
        return out;
    }
}
