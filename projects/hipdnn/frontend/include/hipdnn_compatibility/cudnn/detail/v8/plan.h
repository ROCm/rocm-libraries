// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/v8/engine_views.h>
#include <hipdnn_compatibility/cudnn/detail/v8/error.h>
#include <hipdnn_compatibility/cudnn/detail/v8/records.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
{

// The plan JSON format is the shim's own:
//   {"version":1,
//    "engine":{"globalIndex":i,"id":"<decimal>","name":"...","behaviorNotes":[n,...]},
//    "knobs":[{"type":t,"value":v},...],
//    "payload":"<base64 of the hipDNN serialized plan>"}
// "id" is a string because hashed engine IDs exceed 2^53. "name" and "payload"
// are optional; a plan without a payload cannot be loaded.
namespace plan_json
{

inline constexpr std::int64_t VERSION = 1;
inline constexpr std::string_view BASE64_ALPHABET
    = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

struct Plan
{
    std::int64_t globalIndex = 0;
    std::int64_t id = 0;
    std::optional<std::string> name;
    std::vector<cudnnBackendBehaviorNote_t> behaviorNotes;
    std::vector<std::pair<cudnnBackendKnobType_t, std::int64_t>> knobs;
    std::vector<std::uint8_t> payload;
};

inline void appendBase64(std::string& out, const std::vector<std::uint8_t>& data)
{
    out.reserve(out.size() + (data.size() + 2) / 3 * 4);
    std::size_t i = 0;
    for(; i + 3 <= data.size(); i += 3)
    {
        const std::uint32_t group
            = (std::uint32_t{data[i]} << 16U) | (std::uint32_t{data[i + 1]} << 8U) | data[i + 2];
        out += BASE64_ALPHABET[(group >> 18U) & 0x3FU];
        out += BASE64_ALPHABET[(group >> 12U) & 0x3FU];
        out += BASE64_ALPHABET[(group >> 6U) & 0x3FU];
        out += BASE64_ALPHABET[group & 0x3FU];
    }
    const std::size_t rest = data.size() - i;
    if(rest == 0)
    {
        return;
    }
    std::uint32_t group = std::uint32_t{data[i]} << 16U;
    if(rest == 2)
    {
        group |= std::uint32_t{data[i + 1]} << 8U;
    }
    out += BASE64_ALPHABET[(group >> 18U) & 0x3FU];
    out += BASE64_ALPHABET[(group >> 12U) & 0x3FU];
    out += rest == 2 ? BASE64_ALPHABET[(group >> 6U) & 0x3FU] : '=';
    out += '=';
}

inline int base64Value(char c) noexcept
{
    if(c >= 'A' && c <= 'Z')
    {
        return c - 'A';
    }
    if(c >= 'a' && c <= 'z')
    {
        return c - 'a' + 26;
    }
    if(c >= '0' && c <= '9')
    {
        return c - '0' + 52;
    }
    if(c == '+')
    {
        return 62;
    }
    return c == '/' ? 63 : -1;
}

// Strict: padded, no whitespace. nullopt for anything malformed.
inline std::optional<std::vector<std::uint8_t>> decodeBase64(std::string_view text)
{
    if(text.size() % 4 != 0)
    {
        return std::nullopt;
    }
    std::vector<std::uint8_t> out;
    out.reserve(text.size() / 4 * 3);
    for(std::size_t i = 0; i < text.size(); i += 4)
    {
        const bool last = i + 4 == text.size();
        std::size_t padding = 0;
        if(last && text[i + 3] == '=')
        {
            padding = text[i + 2] == '=' ? 2 : 1;
        }
        std::uint32_t group = 0;
        for(std::size_t j = 0; j < 4 - padding; ++j)
        {
            const int value = base64Value(text[i + j]);
            if(value < 0)
            {
                return std::nullopt;
            }
            group |= static_cast<std::uint32_t>(value) << (18U - 6U * j);
        }
        out.push_back(static_cast<std::uint8_t>(group >> 16U));
        if(padding < 2)
        {
            out.push_back(static_cast<std::uint8_t>(group >> 8U));
        }
        if(padding < 1)
        {
            out.push_back(static_cast<std::uint8_t>(group));
        }
    }
    return out;
}

inline void appendString(std::string& out, std::string_view text)
{
    constexpr std::string_view HEX = "0123456789abcdef";
    out += '"';
    for(const char c : text)
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
            if(static_cast<unsigned char>(c) < 0x20U)
            {
                out += "\\u00";
                out += HEX[static_cast<unsigned char>(c) >> 4U];
                out += HEX[static_cast<unsigned char>(c) & 0xFU];
            }
            else
            {
                out += c;
            }
        }
    }
    out += '"';
}

inline std::string write(const Plan& plan, bool withPayload)
{
    std::string out = R"({"version":)" + std::to_string(VERSION) + R"(,"engine":{"globalIndex":)"
                      + std::to_string(plan.globalIndex) + R"(,"id":")" + std::to_string(plan.id)
                      + '"';
    if(plan.name)
    {
        out += R"(,"name":)";
        appendString(out, *plan.name);
    }
    out += R"(,"behaviorNotes":[)";
    for(std::size_t i = 0; i < plan.behaviorNotes.size(); ++i)
    {
        out += (i == 0 ? "" : ",") + std::to_string(static_cast<int>(plan.behaviorNotes[i]));
    }
    out += R"(]},"knobs":[)";
    for(std::size_t i = 0; i < plan.knobs.size(); ++i)
    {
        out += (i == 0 ? R"({"type":)" : R"(,{"type":)")
               + std::to_string(static_cast<int>(plan.knobs[i].first)) + R"(,"value":)"
               + std::to_string(plan.knobs[i].second) + "}";
    }
    out += ']';
    if(withPayload)
    {
        out += R"(,"payload":")";
        appendBase64(out, plan.payload);
        out += '"';
    }
    out += '}';
    return out;
}

// Reader for exactly the schema above: keys in any order, whitespace allowed,
// unknown keys rejected. Every failure is BAD_PARAM naming `label` and the
// offset.
class Reader
{
public:
    Reader(std::string_view text, std::string_view label)
        : _text(text)
        , _label(label)
    {
    }

    [[noreturn]] void fail(const std::string& what) const
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(_label) + ": " + what + " at offset " + std::to_string(_pos)
                            + " (expected the format written by the hipDNN cuDNN shim)");
    }

    bool consume(char c)
    {
        skipWhitespace();
        if(_pos < _text.size() && _text[_pos] == c)
        {
            ++_pos;
            return true;
        }
        return false;
    }

    void expect(char c)
    {
        if(!consume(c))
        {
            fail(std::string("expected '") + c + "'");
        }
    }

    void expectEnd()
    {
        skipWhitespace();
        if(_pos != _text.size())
        {
            fail("trailing characters");
        }
    }

    std::int64_t integer()
    {
        skipWhitespace();
        std::int64_t value = 0;
        const char* begin = _text.data() + _pos;
        const auto [end, error] = std::from_chars(begin, _text.data() + _text.size(), value);
        if(error != std::errc{})
        {
            fail("expected an int64");
        }
        _pos += static_cast<std::size_t>(end - begin);
        return value;
    }

    std::int64_t integerString()
    {
        const std::string text = string();
        std::int64_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if(error != std::errc{} || end != text.data() + text.size())
        {
            fail("expected a decimal int64 string");
        }
        return value;
    }

    std::string string()
    {
        expect('"');
        std::string out;
        while(true)
        {
            if(_pos >= _text.size())
            {
                fail("unterminated string");
            }
            const char c = _text[_pos++];
            if(c == '"')
            {
                return out;
            }
            if(static_cast<unsigned char>(c) < 0x20U)
            {
                fail("control character in string");
            }
            if(c != '\\')
            {
                out += c;
                continue;
            }
            if(_pos >= _text.size())
            {
                fail("unterminated escape");
            }
            const char escaped = _text[_pos++];
            switch(escaped)
            {
            case '"':
            case '\\':
            case '/':
                out += escaped;
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
                out += unicodeEscape();
                break;
            default:
                fail("unknown escape");
            }
        }
    }

private:
    void skipWhitespace()
    {
        while(_pos < _text.size()
              && (_text[_pos] == ' ' || _text[_pos] == '\n' || _text[_pos] == '\r'
                  || _text[_pos] == '\t'))
        {
            ++_pos;
        }
    }

    // Only \u0000-\u007f: the writer escapes nothing else.
    char unicodeEscape()
    {
        unsigned int code = 0;
        const char* begin = _text.data() + _pos;
        const char* last = _text.data() + std::min(_text.size(), _pos + 4);
        const auto [end, error] = std::from_chars(begin, last, code, 16);
        if(error != std::errc{} || end != begin + 4 || code > 0x7FU)
        {
            fail("unsupported \\u escape");
        }
        _pos += 4;
        return static_cast<char>(code);
    }

    std::string_view _text;
    std::string_view _label;
    std::size_t _pos = 0;
};

inline void readEngine(Reader& reader, Plan& plan)
{
    bool haveIndex = false;
    bool haveId = false;
    bool haveName = false;
    bool haveNotes = false;
    const auto once = [&reader](bool& seen, const std::string& key) {
        if(seen)
        {
            reader.fail("duplicate key \"" + key + "\"");
        }
        seen = true;
    };
    reader.expect('{');
    if(!reader.consume('}'))
    {
        do
        {
            const std::string key = reader.string();
            reader.expect(':');
            if(key == "globalIndex")
            {
                once(haveIndex, key);
                plan.globalIndex = reader.integer();
                if(plan.globalIndex < 0)
                {
                    reader.fail("negative globalIndex");
                }
            }
            else if(key == "id")
            {
                once(haveId, key);
                plan.id = reader.integerString();
            }
            else if(key == "name")
            {
                once(haveName, key);
                plan.name = reader.string();
            }
            else if(key == "behaviorNotes")
            {
                once(haveNotes, key);
                reader.expect('[');
                if(!reader.consume(']'))
                {
                    do
                    {
                        const std::int64_t note = reader.integer();
                        if(note < 0 || note >= CUDNN_BEHAVIOR_NOTE_TYPE_COUNT)
                        {
                            reader.fail("invalid behavior note");
                        }
                        plan.behaviorNotes.push_back(static_cast<cudnnBackendBehaviorNote_t>(note));
                    } while(reader.consume(','));
                    reader.expect(']');
                }
            }
            else
            {
                reader.fail("unknown engine key \"" + key + "\"");
            }
        } while(reader.consume(','));
        reader.expect('}');
    }
    if(!haveIndex || !haveId)
    {
        reader.fail(R"(engine needs "globalIndex" and "id")");
    }
}

inline void readKnob(Reader& reader, Plan& plan)
{
    std::optional<std::int64_t> type;
    std::optional<std::int64_t> value;
    reader.expect('{');
    do
    {
        const std::string key = reader.string();
        reader.expect(':');
        std::optional<std::int64_t>* slot = nullptr;
        if(key == "type")
        {
            slot = &type;
        }
        else if(key == "value")
        {
            slot = &value;
        }
        else
        {
            reader.fail("unknown knob key \"" + key + "\"");
        }
        if(slot->has_value())
        {
            reader.fail("duplicate key \"" + key + "\"");
        }
        *slot = reader.integer();
    } while(reader.consume(','));
    reader.expect('}');
    if(!type || !value)
    {
        reader.fail(R"(knob needs "type" and "value")");
    }
    if(*type < 0 || *type >= CUDNN_KNOB_TYPE_COUNTS)
    {
        reader.fail("invalid knob type");
    }
    plan.knobs.emplace_back(static_cast<cudnnBackendKnobType_t>(*type), *value);
}

// nullopt when the JSON is well formed but carries no payload.
inline std::optional<Plan> read(std::string_view text, std::string_view label)
{
    Reader reader(text, label);
    Plan plan;
    bool haveVersion = false;
    bool haveEngine = false;
    bool haveKnobs = false;
    bool havePayload = false;
    const auto once = [&reader](bool& seen, const std::string& key) {
        if(seen)
        {
            reader.fail("duplicate key \"" + key + "\"");
        }
        seen = true;
    };
    reader.expect('{');
    if(!reader.consume('}'))
    {
        do
        {
            const std::string key = reader.string();
            reader.expect(':');
            if(key == "version")
            {
                once(haveVersion, key);
                if(reader.integer() != VERSION)
                {
                    reader.fail("unsupported version");
                }
            }
            else if(key == "engine")
            {
                once(haveEngine, key);
                readEngine(reader, plan);
            }
            else if(key == "knobs")
            {
                once(haveKnobs, key);
                reader.expect('[');
                if(!reader.consume(']'))
                {
                    do
                    {
                        readKnob(reader, plan);
                    } while(reader.consume(','));
                    reader.expect(']');
                }
            }
            else if(key == "payload")
            {
                once(havePayload, key);
                std::optional<std::vector<std::uint8_t>> payload = decodeBase64(reader.string());
                if(!payload || payload->empty())
                {
                    reader.fail("payload is not non-empty base64");
                }
                plan.payload = std::move(*payload);
            }
            else
            {
                reader.fail("unknown key \"" + key + "\"");
            }
        } while(reader.consume(','));
        reader.expect('}');
    }
    reader.expectEnd();
    if(!haveVersion || !haveEngine)
    {
        reader.fail(R"("version" and "engine" are required)");
    }
    if(!havePayload)
    {
        return std::nullopt;
    }
    return plan;
}

} // namespace plan_json

// Names the caller's inputs in plan error messages.
struct PlanLabels
{
    std::string_view planHandle;
    std::string_view graphHandle;
    std::string_view json;
};

// A finalized hipDNN EXECUTION_PLAN and the config it runs. The JSON
// representation is written on first use.
class PlanRecord final : public RecordOf<CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR>
{
public:
    PlanRecord(std::shared_ptr<const ConfigRecord> config,
               ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor plan,
               cudnnHandle_t contextHandle,
               std::int64_t workspaceSize)
        : _config(std::move(config))
        , _tag(engineTag(*_config))
        , _plan(std::move(plan))
        , _contextHandle(contextHandle)
        , _workspaceSize(workspaceSize)
    {
    }

    const ConfigRecord& config() const noexcept
    {
        return *_config;
    }

    const std::shared_ptr<const ConfigRecord>& configRecord() const noexcept
    {
        return _config;
    }

    hipdnnBackendDescriptor_t hipdnnPlan() const noexcept
    {
        return _plan.get();
    }

    // The handle whose hipDNN resource manager holds the plan's execution
    // context: the graph's handle (built plans) or the load handle.
    cudnnHandle_t contextHandle() const noexcept
    {
        return _contextHandle;
    }

    std::int64_t workspaceSize() const noexcept
    {
        return _workspaceSize;
    }

    // The engine part of ExecutionPlan_v8's tag: "eng<index>", then
    // "_k<type>=<value>" per requested choice (the type as its integer). The
    // builder prefixes its own tag.
    const std::string& tag() const noexcept
    {
        return _tag;
    }

    const std::vector<cudnnBackendBehaviorNote_t>& behaviorNotes() const
    {
        return _config->engine().behaviorNotes();
    }

    std::vector<cudnnBackendNumericalNote_t> numericalNotes() const
    {
        return _config->engine().numericalNotes();
    }

    const std::string& json() const
    {
        std::call_once(_jsonOnce, [this] { _json = toJson(); });
        return _json;
    }

private:
    static std::string engineTag(const ConfigRecord& config)
    {
        std::string tag = "eng" + std::to_string(config.engineIndex());
        for(const auto& [type, value] : config.requestedChoices())
        {
            tag += "_k" + std::to_string(static_cast<int>(type)) + "=" + std::to_string(value);
        }
        return tag;
    }

    // An engine that cannot serialize still gets a valid JSON object, without
    // a payload: the backend's support note is not a reliable gate.
    std::optional<std::vector<std::uint8_t>> serialize() const
    {
        const std::shared_ptr<::hipdnn_frontend::detail::IHipdnnBackend> backend
            = ::hipdnn_frontend::detail::hipdnnBackend();
        std::size_t size = 0;
        hipdnnStatus_t status
            = backend->backendGetSerializedExecutionPlanExt(_plan.get(), 0, &size, nullptr);
        std::vector<std::uint8_t> bytes;
        if(status == HIPDNN_STATUS_SUCCESS && size > 0)
        {
            bytes.resize(size);
            status = backend->backendGetSerializedExecutionPlanExt(
                _plan.get(), bytes.size(), &size, bytes.data());
            bytes.resize(std::min(size, bytes.size()));
        }
        if(status != HIPDNN_STATUS_SUCCESS || bytes.empty())
        {
            const std::string message
                = status == HIPDNN_STATUS_SUCCESS ? "empty plan" : takeHipdnnMessage();
            HIPDNN_FE_LOG_INFO("[cudnn_frontend] Plan JSON for engine "
                               << _config->engineId()
                               << " has no payload; hipDNN could not serialize the plan: "
                               << message);
            return std::nullopt;
        }
        return bytes;
    }

    std::string toJson() const
    {
        const EngineRecord& engine = _config->engine();
        plan_json::Plan plan;
        plan.globalIndex = _config->engineIndex();
        plan.id = _config->engineId();
        plan.name = engine.name();
        plan.behaviorNotes = engine.behaviorNotes();
        plan.knobs = _config->knobChoices();
        std::optional<std::vector<std::uint8_t>> payload = serialize();
        if(payload)
        {
            plan.payload = std::move(*payload);
        }
        return plan_json::write(plan, payload.has_value());
    }

    // Declared before _plan so the hipDNN plan is released before the config
    // (and the graph it keeps alive).
    std::shared_ptr<const ConfigRecord> _config;
    std::string _tag;
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _plan;
    cudnnHandle_t _contextHandle;
    std::int64_t _workspaceSize;
    mutable std::once_flag _jsonOnce;
    mutable std::string _json;
};

inline std::int64_t readWorkspaceSize(hipdnnBackendDescriptor_t plan)
{
    std::int64_t workspaceSize = 0;
    checkHipdnn(::hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
                    plan,
                    HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                    HIPDNN_TYPE_INT64,
                    1,
                    nullptr,
                    &workspaceSize),
                "reading the workspace size of the hipDNN execution plan");
    return workspaceSize;
}

// hipDNN creates the execution context on the graph's handle, so `planHandle`
// is optional (upstream's builder never sets one) and, when set, must name that
// handle.
inline std::shared_ptr<const PlanRecord> buildPlan(std::shared_ptr<const ConfigRecord> config,
                                                   cudnnHandle_t planHandle,
                                                   const PlanLabels& labels)
{
    using ::hipdnn_frontend::detail::hipdnnBackend;

    if(planHandle != nullptr && planHandle != config->engine().graph().handle())
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.planHandle) + " differs from the "
                            + std::string(labels.graphHandle) + " of the engine config's graph");
    }
    auto* const hipdnnConfig = config->hipdnnEngineConfig();
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor plan = createHipdnnDescriptor(
        HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, "creating the hipDNN execution plan");
    checkHipdnn(hipdnnBackend()->backendSetAttribute(plan.get(),
                                                     HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                     HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                     1,
                                                     static_cast<const void*>(&hipdnnConfig)),
                "setting the engine config of the hipDNN execution plan");
    checkHipdnn(hipdnnBackend()->backendFinalize(plan.get()),
                "finalizing the hipDNN execution plan for engine "
                    + std::to_string(config->engineId()));
    const std::int64_t workspaceSize = readWorkspaceSize(plan.get());
    const cudnnHandle_t contextHandle = config->engine().graph().handle();
    return std::make_shared<const PlanRecord>(
        std::move(config), std::move(plan), contextHandle, workspaceSize);
}

// Restores a plan written by PlanRecord::json() on `handle`, which holds its
// execution context. The JSON "id" must match the engine hipDNN reports for the
// deserialized payload.
inline std::shared_ptr<const PlanRecord>
    loadPlan(cudnnHandle_t handle, std::string_view json, const PlanLabels& labels)
{
    using ::hipdnn_frontend::detail::hipdnnBackend;

    if(handle == nullptr)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.planHandle)
                            + " is required to load a serialized execution plan");
    }
    std::optional<plan_json::Plan> parsed = plan_json::read(json, labels.json);
    if(!parsed)
    {
        throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                        std::string(labels.json)
                            + " has no \"payload\": the engine could not serialize the plan, so "
                              "it cannot be loaded");
    }
    hipdnnBackendDescriptor_t raw = nullptr;
    checkHipdnn(hipdnnBackend()->backendCreateAndDeserializeExecutionPlanExt(
                    handle, &raw, parsed->payload.data(), parsed->payload.size()),
                "deserializing the hipDNN execution plan");
    ::hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor plan(raw);

    std::int64_t id = 0;
    checkHipdnn(
        hipdnnBackend()->backendGetAttribute(plan.get(),
                                             HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_GLOBAL_INDEX_EXT,
                                             HIPDNN_TYPE_INT64,
                                             1,
                                             nullptr,
                                             &id),
        "reading the engine ID of the deserialized hipDNN execution plan");
    if(id != parsed->id)
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        std::string(labels.json) + ": \"id\" " + std::to_string(parsed->id)
                            + " does not match engine " + std::to_string(id)
                            + " of the serialized plan");
    }
    std::shared_ptr<const ConfigRecord> config
        = makeLoadedConfigRecord(parsed->id,
                                 parsed->globalIndex,
                                 std::move(parsed->behaviorNotes),
                                 std::move(parsed->knobs),
                                 std::move(parsed->name));
    const std::int64_t workspaceSize = readWorkspaceSize(plan.get());
    return std::make_shared<const PlanRecord>(
        std::move(config), std::move(plan), handle, workspaceSize);
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::v8
