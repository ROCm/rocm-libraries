// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <hipdnn_compatibility/cudnn/detail/backend/attribute_table.h>
#include <hipdnn_compatibility/cudnn/detail/backend/c_types.h>
#include <hipdnn_compatibility/cudnn/detail/backend/descriptor.h>
#include <hipdnn_compatibility/cudnn/detail/backend/engine_descriptors.h>
#include <hipdnn_compatibility/cudnn/detail/backend/operation_graph_descriptor.h>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/detail/BackendWrapper.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
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
// unknown keys rejected. Every failure is BAD_PARAM naming the offset.
class Reader
{
public:
    explicit Reader(std::string_view text)
        : _text(text)
    {
    }

    [[noreturn]] void fail(const std::string& what) const
    {
        throw ShimError(CUDNN_STATUS_BAD_PARAM,
                        "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION: " + what + " at offset "
                            + std::to_string(_pos)
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
inline std::optional<Plan> read(std::string_view text)
{
    Reader reader(text);
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

// Reads and clears the hipDNN backend's last error message.
inline std::string takeHipdnnMessage()
{
    std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> message{};
    hipdnn_frontend::detail::hipdnnBackend()->getLastErrorString(message.data(), message.size());
    return message.data();
}

class ExecutionPlanDescriptor final : public Descriptor
{
public:
    ExecutionPlanDescriptor()
        : Descriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR)
    {
    }

    // The handle whose hipDNN resource manager holds the plan's execution
    // context: the graph's handle (built plans) or the plan HANDLE (loaded).
    cudnnHandle_t contextHandle() const noexcept
    {
        return _contextHandle;
    }

    hipdnnBackendDescriptor_t hipdnnPlan() const noexcept
    {
        return _plan.get();
    }

protected:
    void finalizeImpl() override
    {
        const std::optional<cudnnHandle_t> planHandle
            = scalarAttribute<cudnnHandle_t>(CUDNN_ATTR_EXECUTION_PLAN_HANDLE);
        const cudnnHandle_t handle = planHandle ? *planHandle : nullptr;
        const std::shared_ptr<Descriptor> config
            = descriptorAttribute(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG);
        const std::optional<std::string> json
            = stringAttribute(CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION);
        if((config != nullptr) == json.has_value())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "exactly one of CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG and "
                            "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION must be set");
        }

        std::shared_ptr<EngineConfigDescriptor> engineConfig;
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor plan;
        cudnnHandle_t contextHandle = nullptr;
        if(config != nullptr)
        {
            engineConfig = std::shared_ptr<EngineConfigDescriptor>(
                config,
                &descriptorCast<EngineConfigDescriptor>(*config,
                                                        CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
            plan = build(handle, *engineConfig);
            contextHandle = engineConfig->graph().handle();
        }
        else
        {
            if(handle == nullptr)
            {
                throw ShimError(CUDNN_STATUS_BAD_PARAM,
                                "CUDNN_ATTR_EXECUTION_PLAN_HANDLE is required to load a serialized "
                                "execution plan");
            }
            auto loaded = load(handle, *json);
            engineConfig = std::move(loaded.first);
            plan = std::move(loaded.second);
            contextHandle = handle;
        }

        std::int64_t workspaceSize = 0;
        checkHipdnn(hipdnn_frontend::detail::hipdnnBackend()->backendGetAttribute(
                        plan.get(),
                        HIPDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                        HIPDNN_TYPE_INT64,
                        1,
                        nullptr,
                        &workspaceSize),
                    "reading the workspace size of the hipDNN execution plan");

        storeComputed(
            CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &workspaceSize);
        if(config == nullptr)
        {
            storeComputedDescriptors(CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, {engineConfig});
        }
        _config = std::move(engineConfig);
        _plan = std::move(plan);
        _contextHandle = contextHandle;
    }

    std::optional<StoredAttribute> lazyAttribute(cudnnBackendAttributeName_t name,
                                                 bool /*countOnly*/) const override
    {
        if(name != CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION)
        {
            return std::nullopt;
        }
        std::call_once(_jsonOnce, [this] { _json = toJson(); });
        StoredAttribute value;
        value.type = CUDNN_TYPE_CHAR;
        value.count = static_cast<std::int64_t>(_json.size() + 1);
        value.bytes.resize(_json.size());
        std::memcpy(value.bytes.data(), _json.data(), _json.size());
        return value;
    }

private:
    static hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor
        build(cudnnHandle_t handle, const EngineConfigDescriptor& config)
    {
        using hipdnn_frontend::detail::hipdnnBackend;

        // hipDNN creates the execution context on the graph's handle, so a plan
        // HANDLE is optional (upstream's builder never sets one) and, when set,
        // must name that handle.
        if(handle != nullptr && handle != config.graph().handle())
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_EXECUTION_PLAN_HANDLE differs from the "
                            "CUDNN_ATTR_OPERATIONGRAPH_HANDLE of the engine config's graph");
        }
        auto* const hipdnnConfig = config.hipdnnEngineConfig();
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor plan = createHipdnnDescriptor(
            HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, "creating the hipDNN execution plan");
        checkHipdnn(hipdnnBackend()->backendSetAttribute(plan.get(),
                                                         HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                         1,
                                                         static_cast<const void*>(&hipdnnConfig)),
                    "setting the engine config of the hipDNN execution plan");
        checkHipdnn(hipdnnBackend()->backendFinalize(plan.get()),
                    "finalizing the hipDNN execution plan for engine "
                        + std::to_string(config.engineId()));
        return plan;
    }

    static std::pair<std::shared_ptr<EngineConfigDescriptor>,
                     hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor>
        load(cudnnHandle_t handle, const std::string& json)
    {
        using hipdnn_frontend::detail::hipdnnBackend;

        std::optional<plan_json::Plan> parsed = plan_json::read(json);
        if(!parsed)
        {
            throw ShimError(CUDNN_STATUS_NOT_SUPPORTED,
                            "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION has no \"payload\": the "
                            "engine could not serialize the plan, so it cannot be loaded");
        }
        hipdnnBackendDescriptor_t raw = nullptr;
        checkHipdnn(hipdnnBackend()->backendCreateAndDeserializeExecutionPlanExt(
                        handle, &raw, parsed->payload.data(), parsed->payload.size()),
                    "deserializing the hipDNN execution plan");
        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor plan(raw);

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
                            "CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION: \"id\" "
                                + std::to_string(parsed->id) + " does not match engine "
                                + std::to_string(id) + " of the serialized plan");
        }
        return {EngineConfigDescriptor::forLoadedPlan(parsed->id,
                                                      parsed->globalIndex,
                                                      std::move(parsed->behaviorNotes),
                                                      std::move(parsed->knobs),
                                                      std::move(parsed->name)),
                std::move(plan)};
    }

    // An engine that cannot serialize still gets a valid JSON object, without
    // a payload: the backend's support note is not a reliable gate.
    std::optional<std::vector<std::uint8_t>> serialize() const
    {
        const std::shared_ptr<hipdnn_frontend::detail::IHipdnnBackend> backend
            = hipdnn_frontend::detail::hipdnnBackend();
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
        const EngineDescriptor& engine = _config->engine();
        plan_json::Plan plan;
        plan.globalIndex = _config->engineIndex();
        plan.id = _config->engineId();
        plan.name = engine.engineName();
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
    std::shared_ptr<EngineConfigDescriptor> _config;
    hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _plan;
    cudnnHandle_t _contextHandle = nullptr;
    mutable std::once_flag _jsonOnce;
    mutable std::string _json;
};

class VariantPackDescriptor final : public Descriptor
{
public:
    VariantPackDescriptor()
        : Descriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR)
    {
    }

    hipdnnBackendDescriptor_t hipdnnVariantPack() const noexcept
    {
        return _variantPack.get();
    }

protected:
    // Built here rather than at execute so execute is a single backend call.
    // populateBaseVariantPackDescriptor is not reused: it takes a UID map,
    // which would silently merge duplicate UIDs and copy both arrays again.
    void finalizeImpl() override
    {
        using hipdnn_frontend::detail::hipdnnBackend;

        const StoredAttribute& pointers = requiredArray(CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS);
        const StoredAttribute& uids = requiredArray(CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS);
        if(pointers.count != uids.count)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            "CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS has "
                                + std::to_string(pointers.count)
                                + " elements but CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS has "
                                + std::to_string(uids.count));
        }
        void* const workspace
            = scalarAttribute<void*>(CUDNN_ATTR_VARIANT_PACK_WORKSPACE).value_or(nullptr);

        hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor variantPack = createHipdnnDescriptor(
            HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, "creating the hipDNN variant pack");
        checkHipdnn(hipdnnBackend()->backendSetAttribute(variantPack.get(),
                                                         HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                         HIPDNN_TYPE_VOID_PTR,
                                                         pointers.count,
                                                         pointers.bytes.data()),
                    "setting the hipDNN variant pack data pointers");
        checkHipdnn(hipdnnBackend()->backendSetAttribute(variantPack.get(),
                                                         HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                         HIPDNN_TYPE_INT64,
                                                         uids.count,
                                                         uids.bytes.data()),
                    "setting the hipDNN variant pack unique IDs");
        checkHipdnn(hipdnnBackend()->backendSetAttribute(variantPack.get(),
                                                         HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                         HIPDNN_TYPE_VOID_PTR,
                                                         1,
                                                         static_cast<const void*>(&workspace)),
                    "setting the hipDNN variant pack workspace");
        checkHipdnn(hipdnnBackend()->backendFinalize(variantPack.get()),
                    "finalizing the hipDNN variant pack");
        _variantPack = std::move(variantPack);
    }

private:
    // hipDNN rejects an empty array with a null-pointer error that does not
    // name the cuDNN attribute.
    const StoredAttribute& requiredArray(cudnnBackendAttributeName_t name) const
    {
        const StoredAttribute* value = findAttribute(name);
        if(value == nullptr || value->count == 0)
        {
            throw ShimError(CUDNN_STATUS_BAD_PARAM,
                            attributeName(name) + " is required (non-empty)");
        }
        return *value;
    }

    hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor _variantPack;
};

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail::backend
