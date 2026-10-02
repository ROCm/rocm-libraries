// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-debug-child.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <string_view>
#include <thread>
#include <vector>

namespace hipblaslt_jit::debug
{
    namespace
    {
        namespace fs = std::filesystem;

        constexpr size_t lineLimit = 4096;

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

    std::string childCategories()
    {
        const unsigned on = categories();
        std::string    out;
        if(on & Timing)
            out = "timing";
        if(on & Progress)
            out += out.empty() ? "progress" : ",progress";
        return out;
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
            std::chrono::duration_cast<std::chrono::nanoseconds>(origin().time_since_epoch())
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
