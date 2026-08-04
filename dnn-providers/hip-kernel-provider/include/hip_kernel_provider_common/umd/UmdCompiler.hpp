// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// UmdCompiler.hpp - compiles a Universal Match Descriptor (RFC 0018) into a
// reusable CompiledUmd.
//
// All names below live in namespace hip_kernel_provider_common::umd.
//
// The compiler turns a `hipdnn.umd/v1` descriptor (a nlohmann::json document)
// into a CompiledUmd the UniversalGraphMatcher runs against graphs. It:
//   1. validates the descriptor against Appendix A.10 (schema/id, node object,
//      registry-resolved opcodes and names, `?`<->optionality, reserved roots,
//      single producer);
//   2. lowers the JSON criteria to pure JsonLogic: expands the `shape`
//      short-hand (A.5) to a rank check while recording each named dim on its
//      tensor, and expands layout aliases (A.8) to stride-order arrays;
//   3. statically type-checks the lowered criteria against the Phase 0
//      generated registry (A.10 §9) -- the payoff of the strongly-typed
//      codegen; a type mismatch is a compile error, not a runtime decline;
//   4. compiles the criteria to a jlogic::Expression<BindingContext> and
//      publishes the referenced bound-symbol set (RFC 0018 §4).
//
// A descriptor that violates any check throws UmdCompileError (RFC 0018 §10 /
// A.10: refuse at compile, never match by default).

#include "hip_kernel_provider_common/JsonLogic.hpp"
#include "hip_kernel_provider_common/umd/BindingContext.hpp"
#include "hip_kernel_provider_common/umd/UmdPathParse.hpp"

#include <hipdnn_flatbuffers_sdk/umd/op_schema_registry_generated.hpp>

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hip_kernel_provider_common::umd
{

namespace reg = hipdnn_flatbuffers_sdk::umd;

// Thrown when a descriptor fails any Appendix A.10 compile-time check.
class UmdCompileError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// One operand/result binding slot of a pattern node: the op-schema tensor name
// (`q`) and the pattern variable it binds (`q`), with the generated typed UID
// reader for that node's opcode.
struct EdgeSlot
{
    std::string schemaName; // op-schema registry tensor name, e.g. "q" / "attn_mask"
    std::string tvar; // pattern variable, e.g. "q"
    bool optional = false;
    bool isOutput = false; // operand vs result
    const reg::UidReader* reader = nullptr; // generated typed UID reader
};

// A pattern node: its id, opcode/op-schema, and the operand/result edges that
// bind pattern variables. A descriptor may declare several nodes; edges connect
// them when two nodes share a pattern variable (RFC 0018 §3 / A.3).
struct NodeSpec
{
    std::string id;
    std::string opcode;
    const reg::OpSchemaEntry* opSchema = nullptr;
    std::vector<EdgeSlot> edges;
};

// A pattern variable (`$q`) shared across the descriptor's nodes. `dimNames` is
// populated by the `shape` lowering and read by BindingContext. `optional` is
// true only when every binding slot referencing it is optional (its `.present`
// is then legal).
struct TensorVarSpec
{
    std::string tvar;
    bool optional = false;
    std::unordered_map<std::string, std::size_t> dimNames; // shape short-hand (A.5)
    std::size_t pinnedRank = 0; // 0 == not pinned by a shape op
};

// A compiled descriptor: heap-stable (owned via unique_ptr by the matcher) so
// BindingContext may reference its TensorVarSpec::dimNames maps for the match's
// lifetime.
struct CompiledUmd
{
    std::string id;
    std::string name;
    // Matcher format version and the hipDNN graph schema version this matcher
    // was authored against; empty when the descriptor declares neither
    // (RFC 0017 §4).
    std::string version;
    std::string sdkVersion;
    bool allowOverrideShape = false;
    std::vector<NodeSpec> nodes;
    std::vector<TensorVarSpec> tvars;
    jlogic::Expression<BindingContext> criteria;
    std::set<std::string> boundSymbols;

    // The `$kernel.<field>` names the criteria expression reads, without the
    // `kernel.` prefix (`$kernel.tile_m` contributes "tile_m"). Set once by
    // UmdCompiler::run() and immutable thereafter; empty for a graph-only
    // matcher, which is what routes between the metadata and non-metadata
    // match overloads. A KDP/KMD loader checks each name exists in the engine's
    // KMD and uses the set as the per-kernel memoization key (RFC 0017 §5).
    std::set<std::string> kernelFields;

    const TensorVarSpec* findTvar(const std::string& tvar) const
    {
        for(const auto& t : tvars)
        {
            if(t.tvar == tvar)
            {
                return &t;
            }
        }
        return nullptr;
    }

    TensorVarSpec* findTvarMut(const std::string& tvar)
    {
        for(auto& t : tvars)
        {
            if(t.tvar == tvar)
            {
                return &t;
            }
        }
        return nullptr;
    }

    const NodeSpec* findNode(const std::string& nodeId) const
    {
        for(const auto& n : nodes)
        {
            if(n.id == nodeId)
            {
                return &n;
            }
        }
        return nullptr;
    }
};

class UmdCompiler
{
public:
    static CompiledUmd compile(const nlohmann::json& descriptor)
    {
        UmdCompiler c;
        return c.run(descriptor);
    }

    // The matcher format version this compiler implements. Gating is a
    // CEILING, not a floor (RFC 0017 §4): a descriptor whose major differs, or
    // whose minor is newer than this runtime's, is refused, because it carries
    // features this runtime cannot understand. An older minor within the same
    // major always loads -- a file stamped "1.0" loads on a "1.1" runtime.
    static constexpr const char* K_RUNTIME_VERSION = "1.0";

    // The hipDNN graph schema (SDK) version this compiler understands. Unlike
    // the format version, `sdk_version` is gated against a floor the GRAPH
    // sets, not a constant: a graph reports the schema version its own
    // contents require, and a matcher declaring less is declined before it runs
    // (RFC 0017 §4). This constant is only the ceiling half -- the upper bound
    // on what a matcher may claim to understand. The per-graph floor is applied
    // by UniversalGraphMatcher via graphRequiredSdkVersion().
    static constexpr const char* K_RUNTIME_SDK_VERSION = "1.2";

    // The version an omitted `version` / `sdk_version` key means: the lowest
    // this format ever had, which is what a descriptor authored before either
    // key existed implies (RFC 0018 A.1).
    static constexpr const char* K_DEFAULT_VERSION = "1.0";

    // Parsed "<major>.<minor>". A version that does not parse is not ordered
    // against anything; callers treat that as a refusal (fail closed).
    struct SemVer
    {
        std::size_t major = 0;
        std::size_t minor = 0;

        bool operator<(const SemVer& o) const
        {
            return major != o.major ? major < o.major : minor < o.minor;
        }
        bool operator<=(const SemVer& o) const
        {
            return !(o < *this);
        }
    };

    // Parse "<major>.<minor>"; false when the string is not exactly two
    // digit-only components.
    static bool parseSemVer(const std::string& s, SemVer& out)
    {
        return parseVersion(s, out.major, out.minor);
    }

    // True when `version` is at or above `floor`. Compared numerically on
    // (major, minor), never lexicographically, so "1.10" ranks above "1.9". A
    // version that does not parse is below every floor (fail closed).
    static bool versionAtLeast(const std::string& version, const std::string& floor)
    {
        SemVer v;
        SemVer f;
        if(!parseSemVer(version, v) || !parseSemVer(floor, f))
        {
            return false;
        }
        return f <= v;
    }

    // True when a descriptor stamped `version` is loadable on a runtime
    // implementing `runtime`: same major, and a minor no newer than the
    // runtime's (RFC 0017 §4). An unparseable version loads nowhere.
    static bool versionLoadableOn(const std::string& version, const std::string& runtime)
    {
        SemVer v;
        SemVer r;
        if(!parseSemVer(version, v) || !parseSemVer(runtime, r))
        {
            return false;
        }
        return v.major == r.major && v.minor <= r.minor;
    }

private:
    // Static type domain for A.10 §9. Numeric unifies Int and Float; ARRAY
    // carries its element kind for `in` / IntArray equality. DYNAMIC is a
    // runtime-resolved scalar wildcard: `$kernel.<field>` values come from a
    // UKD's KMD, unavailable at UMD-compile time (RFC 0018 §4/§10), so they are
    // not statically type-checked here. A referenced kernel field is assumed
    // validated against the engine's KMD schema at load (a typo'd field is
    // caught there, not here); the metadata supplied to match() is fully
    // resolved, so every referenced field is present at match time.
    enum class Kind
    {
        INT,
        FLOAT,
        BOOL,
        DTYPE,
        ARRAY,
        TENSOR,
        JSON_NULL,
        DYNAMIC,
        UNKNOWN
    };
    struct TypeInfo
    {
        Kind kind = Kind::UNKNOWN;
        Kind elem = Kind::UNKNOWN; // element kind when kind == ARRAY
    };

    static bool isNumeric(Kind k)
    {
        return k == Kind::INT || k == Kind::FLOAT;
    }

    // Kind rendered for a diagnostic (A.10 §9).
    static std::string kindName(Kind k)
    {
        switch(k)
        {
        case Kind::INT:
            return "Int";
        case Kind::FLOAT:
            return "Float";
        case Kind::BOOL:
            return "Bool";
        case Kind::DTYPE:
            return "Dtype";
        case Kind::ARRAY:
            return "Array";
        case Kind::TENSOR:
            return "Tensor";
        case Kind::JSON_NULL:
            return "Null";
        case Kind::DYNAMIC:
            return "Dynamic";
        default:
            break;
        }
        return "Unknown";
    }

    CompiledUmd run(const nlohmann::json& d)
    {
        CompiledUmd out;
        validateTopLevel(d);
        out.id = d.at("id").get<std::string>();
        out.name = d.at("name").get<std::string>();
        // Both version keys are optional. An omitted key means "1.0", the
        // version every descriptor authored against this revision implies
        // (RFC 0018 A.1), so the published value is never empty and the
        // per-graph SDK floor has a concrete version to compare against.
        out.version = d.contains("version") ? d.at("version").get<std::string>()
                                            : std::string(K_DEFAULT_VERSION);
        out.sdkVersion = d.contains("sdk_version") ? d.at("sdk_version").get<std::string>()
                                                   : std::string(K_DEFAULT_VERSION);
        out.allowOverrideShape
            = d.contains("allow_override_shape") ? d.at("allow_override_shape").get<bool>() : false;

        parseNodes(d.at("nodes"), out);

        nlohmann::json const lowered = lowerCriteria(d.at("criteria"), out);
        const TypeInfo t = inferType(lowered, &out);
        if(t.kind != Kind::BOOL && t.kind != Kind::DYNAMIC)
        {
            throw UmdCompileError("criteria must have static type Bool (A.10 §10)");
        }

        out.criteria = jlogic::compile<BindingContext>(lowered);
        for(const std::string& sym : out.criteria.variables())
        {
            // Validate each reference resolves (A.10 §8); varTypeOf throws if not.
            varTypeOf(sym, out);
            out.boundSymbols.insert(sym);
            std::string root;
            std::string rest;
            path::splitRoot(sym, root, rest);
            if(root == "kernel")
            {
                out.kernelFields.insert(rest);
            }
        }
        return out;
    }

    // ---- A.1 / A.10 top level -------------------------------------------
    static void validateTopLevel(const nlohmann::json& d)
    {
        if(!d.is_object())
        {
            throw UmdCompileError("descriptor must be a JSON object");
        }
        static const std::set<std::string> s_allowed = {"schema",
                                                        "version",
                                                        "sdk_version",
                                                        "id",
                                                        "name",
                                                        "allow_override_shape",
                                                        "nodes",
                                                        "criteria"};
        for(const auto& [key, value] : d.items())
        {
            if(s_allowed.find(key) == s_allowed.end())
            {
                throw UmdCompileError("unknown top-level key: " + key);
            }
        }
        for(const char* required : {"schema", "id", "name", "nodes", "criteria"})
        {
            if(!d.contains(required))
            {
                throw UmdCompileError(std::string("missing required key: ") + required);
            }
        }
        if(!d.at("schema").is_string() || d.at("schema").get<std::string>() != "hipdnn.umd/v1")
        {
            throw UmdCompileError("schema must equal \"hipdnn.umd/v1\" (A.10 §1)");
        }
        if(!d.at("id").is_string() || !isUuid(d.at("id").get<std::string>()))
        {
            throw UmdCompileError("id must be a well-formed UUID (A.10 §1)");
        }
        if(!d.at("name").is_string())
        {
            throw UmdCompileError("name must be a string");
        }
        if(d.contains("allow_override_shape") && !d.at("allow_override_shape").is_boolean())
        {
            throw UmdCompileError("allow_override_shape must be a boolean");
        }
        validateVersionKey(d, "version", K_RUNTIME_VERSION);
        validateVersionKey(d, "sdk_version", K_RUNTIME_SDK_VERSION);
        if(!d.at("nodes").is_array() || d.at("nodes").empty())
        {
            throw UmdCompileError("nodes must be a non-empty array (A.1)");
        }
    }

    static bool isUuid(const std::string& s)
    {
        if(s.size() != 36)
        {
            return false;
        }
        for(std::size_t i = 0; i < s.size(); ++i)
        {
            if(i == 8 || i == 13 || i == 18 || i == 23)
            {
                if(s[i] != '-')
                {
                    return false;
                }
            }
            else if(std::isxdigit(static_cast<unsigned char>(s[i])) == 0)
            {
                return false;
            }
        }
        return true;
    }

    // Both version keys are optional; when present each must be a
    // "<major>.<minor>" string this runtime can honor. The gate is a ceiling
    // (RFC 0017 §4): same major, minor no newer than the runtime's. An older
    // minor always loads, so a descriptor stays loadable on the oldest runtime
    // that can serve it. The per-graph `sdk_version` FLOOR is a separate,
    // match-time check (UniversalGraphMatcher), because only a graph can say
    // which schema version its own contents require.
    static void validateVersionKey(const nlohmann::json& d, const char* key, const char* runtime)
    {
        if(!d.contains(key))
        {
            return;
        }
        if(!d.at(key).is_string())
        {
            throw UmdCompileError(std::string(key)
                                  + " must be a \"<major>.<minor>\" string (A.10 §1)");
        }
        const std::string declared = d.at(key).get<std::string>();
        SemVer parsed;
        if(!parseSemVer(declared, parsed))
        {
            throw UmdCompileError(std::string(key) + " \"" + declared
                                  + "\" is not a \"<major>.<minor>\" string (A.10 §1)");
        }
        if(!versionLoadableOn(declared, runtime))
        {
            throw UmdCompileError(std::string(key) + " " + declared
                                  + " is not supported by this runtime, which implements " + runtime
                                  + " (A.10 §1)");
        }
    }

    static bool parseVersion(const std::string& s, std::size_t& major, std::size_t& minor)
    {
        const std::size_t dot = s.find('.');
        if(dot == std::string::npos || s.find('.', dot + 1) != std::string::npos)
        {
            return false;
        }
        return parseVersionComponent(s.substr(0, dot), major)
               && parseVersionComponent(s.substr(dot + 1), minor);
    }

    // A version component is a run of decimal digits: no sign, no whitespace,
    // no exponent. The length cap keeps the accumulation below overflow.
    static bool parseVersionComponent(const std::string& s, std::size_t& out)
    {
        if(s.empty() || s.size() > 9)
        {
            return false;
        }
        std::size_t value = 0;
        for(const char c : s)
        {
            if(std::isdigit(static_cast<unsigned char>(c)) == 0)
            {
                return false;
            }
            value = (value * 10) + static_cast<std::size_t>(c - '0');
        }
        out = value;
        return true;
    }

    static bool isReservedRoot(const std::string& s)
    {
        return s == "graph" || s == "kernel" || s == "device";
    }

    // ---- A.2 / A.3 node object ------------------------------------------
    static void parseNodes(const nlohmann::json& nodes, CompiledUmd& out)
    {
        for(const auto& node : nodes)
        {
            parseNode(node, out);
        }

        // A.10 §5: node ids unique.
        for(std::size_t i = 0; i < out.nodes.size(); ++i)
        {
            for(std::size_t j = i + 1; j < out.nodes.size(); ++j)
            {
                if(out.nodes[i].id == out.nodes[j].id)
                {
                    throw UmdCompileError("duplicate node id (A.10 §5): " + out.nodes[i].id);
                }
            }
        }

        // A.10 §5: pattern variables disjoint from node ids and reserved roots.
        for(const TensorVarSpec& t : out.tvars)
        {
            if(out.findNode(t.tvar) != nullptr)
            {
                throw UmdCompileError("pattern variable collides with a node id (A.10 §5): "
                                      + t.tvar);
            }
            if(isReservedRoot(t.tvar))
            {
                throw UmdCompileError("pattern variable uses a reserved root (A.10 §5): " + t.tvar);
            }
        }
    }

    static void parseNode(const nlohmann::json& node, CompiledUmd& out)
    {
        if(!node.is_object())
        {
            throw UmdCompileError("node must be a JSON object");
        }
        static const std::set<std::string> s_allowed = {"kind", "id", "op", "operands", "results"};
        for(const auto& [key, value] : node.items())
        {
            if(s_allowed.find(key) == s_allowed.end())
            {
                throw UmdCompileError("unknown node key: " + key);
            }
        }
        if(!node.contains("kind") || node.at("kind").get<std::string>() != "op")
        {
            throw UmdCompileError("node kind must be \"op\" (A.2)");
        }
        if(!node.contains("id") || !node.at("id").is_string())
        {
            throw UmdCompileError("node must have a string id (A.2)");
        }
        NodeSpec spec;
        spec.id = node.at("id").get<std::string>();
        if(isReservedRoot(spec.id))
        {
            throw UmdCompileError("node id must not be a reserved root (A.10 §5): " + spec.id);
        }
        if(!node.contains("op") || !node.at("op").is_string())
        {
            // one_of / "any" op-selectors are deferred (opcode-string PoC scope).
            throw UmdCompileError("node op must be an opcode string (PoC scope)");
        }
        spec.opcode = node.at("op").get<std::string>();
        spec.opSchema = reg::lookupOpByName(spec.opcode);
        if(spec.opSchema == nullptr)
        {
            throw UmdCompileError("opcode does not resolve in the op-schema registry: "
                                  + spec.opcode);
        }

        if(node.contains("operands"))
        {
            parseNameMap(node.at("operands"), out, spec, /*isOutput=*/false);
        }
        if(node.contains("results"))
        {
            parseNameMap(node.at("results"), out, spec, /*isOutput=*/true);
        }
        out.nodes.push_back(std::move(spec));
    }

    static void
        parseNameMap(const nlohmann::json& map, CompiledUmd& out, NodeSpec& spec, bool isOutput)
    {
        if(!map.is_object())
        {
            throw UmdCompileError("operands/results must be an object (A.2)");
        }
        for(const auto& [name, bindJson] : map.items())
        {
            if(!bindJson.is_string())
            {
                throw UmdCompileError("name binding must be a \"$var\" string: " + name);
            }
            std::string bind = bindJson.get<std::string>();
            if(bind.empty() || bind.front() != '$')
            {
                throw UmdCompileError("name binding must start with '$': " + bind);
            }
            bool optional = false;
            if(bind.back() == '?')
            {
                optional = true;
                bind.pop_back();
            }
            const std::string tvar = bind.substr(1);
            if(tvar.empty())
            {
                throw UmdCompileError("empty pattern variable in binding");
            }

            // A.10 §3/§4: name resolves in registry with matching optionality.
            const reg::UidReader* reader = nullptr;
            bool registryOptional = false;
            if(!resolveTensorName(*spec.opSchema, name, isOutput, reader, registryOptional))
            {
                throw UmdCompileError("name does not resolve in the op-schema registry for "
                                      + spec.opcode + ": " + name);
            }
            if(registryOptional != optional)
            {
                throw UmdCompileError(
                    "`?` suffix does not match registry optionality for name (A.10 §4): " + name);
            }
            // A.3: single producer -- a variable is a `results` value at most
            // once across all nodes.
            if(isOutput)
            {
                for(const NodeSpec& n : out.nodes)
                {
                    for(const EdgeSlot& e : n.edges)
                    {
                        if(e.isOutput && e.tvar == tvar)
                        {
                            throw UmdCompileError("pattern variable produced more than once (A.3): "
                                                  + tvar);
                        }
                    }
                }
                for(const EdgeSlot& e : spec.edges)
                {
                    if(e.isOutput && e.tvar == tvar)
                    {
                        throw UmdCompileError("pattern variable produced more than once (A.3): "
                                              + tvar);
                    }
                }
            }

            spec.edges.push_back(EdgeSlot{name, tvar, optional, isOutput, reader});
            registerTvar(out, tvar, optional);
        }
    }

    // Register (or update) a pattern variable. A variable is optional only when
    // every binding slot referencing it is optional (RFC 0018 A.4: `.present`
    // is legal only for an optional binding).
    static void registerTvar(CompiledUmd& out, const std::string& tvar, bool optional)
    {
        TensorVarSpec* existing = out.findTvarMut(tvar);
        if(existing == nullptr)
        {
            TensorVarSpec spec;
            spec.tvar = tvar;
            spec.optional = optional;
            out.tvars.push_back(std::move(spec));
            return;
        }
        existing->optional = existing->optional && optional;
    }

    static bool resolveTensorName(const reg::OpSchemaEntry& e,
                                  const std::string& name,
                                  bool isOutput,
                                  const reg::UidReader*& reader,
                                  bool& optional)
    {
        if(isOutput)
        {
            for(std::size_t i = 0; i < e.outputTensorCount; ++i)
            {
                if(e.outputTensors[i].name == name)
                {
                    reader = &e.outputTensors[i].read;
                    optional = e.outputTensors[i].optional;
                    return true;
                }
            }
            return false;
        }
        for(std::size_t i = 0; i < e.inputTensorCount; ++i)
        {
            if(e.inputTensors[i].name == name)
            {
                reader = &e.inputTensors[i].read;
                optional = e.inputTensors[i].optional;
                return true;
            }
        }
        return false;
    }

    // ---- A.5 / A.8 lowering ---------------------------------------------
    nlohmann::json lowerCriteria(const nlohmann::json& node, CompiledUmd& out)
    {
        if(node.is_object())
        {
            if(node.size() == 1 && node.begin().key() == "shape")
            {
                return lowerShape(node.begin().value(), out);
            }
            nlohmann::json result = nlohmann::json::object();
            for(const auto& [key, value] : node.items())
            {
                nlohmann::json lowered = lowerCriteria(value, out);
                if((key == "==" || key == "!=") && lowered.is_array() && lowered.size() == 2)
                {
                    expandLayoutAlias(lowered, out);
                }
                result[key] = std::move(lowered);
            }
            return result;
        }
        if(node.is_array())
        {
            nlohmann::json result = nlohmann::json::array();
            for(const auto& e : node)
            {
                result.push_back(lowerCriteria(e, out));
            }
            return result;
        }
        return node;
    }

    static nlohmann::json lowerShape(const nlohmann::json& args, CompiledUmd& out)
    {
        if(!args.is_array() || args.size() != 2 || !args.at(0).is_string()
           || !args.at(1).is_array())
        {
            throw UmdCompileError("shape short-hand must be [\"$tensor\", [entries]] (A.5)");
        }
        const std::string ref = args.at(0).get<std::string>();
        if(ref.empty() || ref.front() != '$')
        {
            throw UmdCompileError("shape target must be a tensor reference: " + ref);
        }
        const std::string tvar = ref.substr(1);
        TensorVarSpec* spec = out.findTvarMut(tvar);
        if(spec == nullptr)
        {
            throw UmdCompileError("shape target is not a bound tensor: " + ref);
        }

        const nlohmann::json& entries = args.at(1);
        bool hasCapture = false;
        std::size_t idx = 0;
        for(const auto& entry : entries)
        {
            if(!entry.is_string())
            {
                throw UmdCompileError("shape entry must be a string (A.5)");
            }
            const std::string name = entry.get<std::string>();
            if(hasCapture)
            {
                throw UmdCompileError("shape capture must be the last entry (A.5/A.10 §7)");
            }
            if(name == "_")
            {
                ++idx; // anonymous positional
                continue;
            }
            if(name.front() == '$')
            {
                hasCapture = true; // trailing capture vector; rank pinned as >= count-1
                ++idx;
                continue;
            }
            if(isReservedTensorField(name))
            {
                throw UmdCompileError(
                    "shape dim-name collides with a reserved tensor-field (A.10 §7): " + name);
            }
            if(spec->dimNames.find(name) != spec->dimNames.end())
            {
                throw UmdCompileError("duplicate shape dim-name (A.5): " + name);
            }
            spec->dimNames[name] = idx;
            ++idx;
        }

        const std::string rankRef = "$" + tvar + ".rank";
        if(hasCapture)
        {
            spec->pinnedRank = 0; // variable rank
            return nlohmann::json{
                {">=",
                 nlohmann::json::array({rankRef, static_cast<std::int64_t>(entries.size() - 1)})}};
        }
        spec->pinnedRank = entries.size();
        return nlohmann::json{
            {"==", nlohmann::json::array({rankRef, static_cast<std::int64_t>(entries.size())})}};
    }

    static bool isReservedTensorField(const std::string& name)
    {
        static const std::set<std::string> s_reserved = {"uid",
                                                         "rank",
                                                         "dtype",
                                                         "stride_order",
                                                         "packed",
                                                         "virtual",
                                                         "present",
                                                         "dims",
                                                         "strides",
                                                         "is_runtime_pass_by_value",
                                                         "value_f32"};
        return s_reserved.find(name) != s_reserved.end();
    }

    static void expandLayoutAlias(nlohmann::json& args, CompiledUmd& out)
    {
        const int strideRefIdx = strideOrderRefIndex(args);
        if(strideRefIdx < 0)
        {
            return;
        }
        const std::size_t aliasIdx = strideRefIdx == 0 ? 1 : 0;
        if(!args.at(aliasIdx).is_string())
        {
            return;
        }
        const std::string alias = args.at(aliasIdx).get<std::string>();
        if(alias.empty() || alias.front() == '$')
        {
            return; // a tensor reference, not an alias
        }
        const std::string tvar = tvarOfStrideOrderRef(
            args.at(static_cast<std::size_t>(strideRefIdx)).get<std::string>());
        std::size_t rank = 0;
        const TensorVarSpec* spec = out.findTvar(tvar);
        if(spec != nullptr)
        {
            rank = spec->pinnedRank;
        }
        args[aliasIdx] = aliasToArray(alias, rank);
    }

    static int strideOrderRefIndex(const nlohmann::json& args)
    {
        for(int i = 0; i < 2; ++i)
        {
            if(args.at(static_cast<std::size_t>(i)).is_string())
            {
                const std::string s = args.at(static_cast<std::size_t>(i)).get<std::string>();
                if(!s.empty() && s.front() == '$' && endsWith(s, ".stride_order"))
                {
                    return i;
                }
            }
        }
        return -1;
    }

    static std::string tvarOfStrideOrderRef(const std::string& ref)
    {
        // "$q.stride_order" -> "q"
        const std::size_t dot = ref.find('.');
        return ref.substr(1, dot - 1);
    }

    // Expand a layout alias to a stride-order array, in the RFC 0017 §5 form
    // BindingContext publishes: logical dimension indices ordered outermost
    // (largest-stride) first, so a contiguous rank-4 tensor is [0,1,2,3] and
    // NHWC is [0,2,3,1] -- the array reads as the layout it names. This is the
    // inverse of the stride-rank vector extractStrideOrder returns;
    // BindingContext converts once when it binds `$q.stride_order`, so
    // `stride_order == alias` holds against a live graph's strides.
    // RFC 0018 A.8 tabulates these arrays.
    static nlohmann::json aliasToArray(const std::string& alias, std::size_t rank)
    {
        nlohmann::json arr;
        if(alias == "nchw" || alias == "bhsd")
        {
            arr = nlohmann::json::array({0, 1, 2, 3});
        }
        else if(alias == "nhwc")
        {
            arr = nlohmann::json::array({0, 2, 3, 1});
        }
        else if(alias == "ncdhw")
        {
            arr = nlohmann::json::array({0, 1, 2, 3, 4});
        }
        else if(alias == "ndhwc")
        {
            arr = nlohmann::json::array({0, 2, 3, 4, 1});
        }
        else if(alias == "contiguous")
        {
            if(rank == 0)
            {
                throw UmdCompileError("layout alias \"contiguous\" requires a rank pinned by a "
                                      "shape short-hand (A.8)");
            }
            arr = nlohmann::json::array();
            for(std::size_t i = 0; i < rank; ++i)
            {
                arr.push_back(static_cast<std::int64_t>(i));
            }
        }
        else
        {
            throw UmdCompileError("unknown layout alias (A.8): " + alias);
        }
        // A fixed-rank alias compared against a tensor whose shape short-hand
        // pinned a different rank can never hold; refuse it at compile rather
        // than silently always declining at match time (A.8).
        if(rank != 0 && arr.size() != rank)
        {
            throw UmdCompileError(
                "layout alias \"" + alias + "\" has rank " + std::to_string(arr.size())
                + " but the tensor is pinned to rank " + std::to_string(rank) + " (A.8)");
        }
        return arr;
    }

    static bool endsWith(const std::string& s, const std::string& suffix)
    {
        return s.size() >= suffix.size()
               && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    // ---- A.10 §9 static type-check --------------------------------------
    TypeInfo inferType(const nlohmann::json& node, const CompiledUmd* out = nullptr)
    {
        // `out` threads the compiled context through recursion; it is set once
        // by run() via a member and reused. Store it on first entry.
        if(out != nullptr)
        {
            _ctx = out;
        }
        return inferTypeImpl(node);
    }

    TypeInfo inferTypeImpl(const nlohmann::json& node)
    {
        if(node.is_object())
        {
            if(node.size() != 1)
            {
                throw UmdCompileError("expression object must have exactly one operator key (A.6)");
            }
            return inferOp(node.begin().key(), node.begin().value());
        }
        if(node.is_array())
        {
            TypeInfo info;
            info.kind = Kind::ARRAY;
            info.elem = Kind::UNKNOWN;
            for(const auto& e : node)
            {
                const TypeInfo et = inferTypeImpl(e);
                if(info.elem == Kind::UNKNOWN)
                {
                    info.elem = et.kind;
                }
                else if(!sameDomain(info.elem, et.kind))
                {
                    throw UmdCompileError("array literal has mixed element types (A.7 `in`)");
                }
            }
            return info;
        }
        if(node.is_boolean())
        {
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(node.is_number_float())
        {
            return {Kind::FLOAT, Kind::UNKNOWN};
        }
        if(node.is_number())
        {
            return {Kind::INT, Kind::UNKNOWN};
        }
        if(node.is_string())
        {
            const std::string s = node.get<std::string>();
            if(!s.empty() && s.front() == '$')
            {
                // A.4: any string beginning with `$` is a variable reference; no
                // escape form exists, so a malformed ref resolves (and throws)
                // through varTypeOf rather than being reinterpreted as a literal.
                return varTypeOf(s.substr(1), *_ctx);
            }
            return {Kind::DTYPE, Kind::UNKNOWN}; // enum name / opcode literal
        }
        return {Kind::JSON_NULL, Kind::UNKNOWN};
    }

    static bool sameDomain(Kind a, Kind b)
    {
        // DYNAMIC (a runtime-resolved $kernel field) is a scalar wildcard: it
        // unifies with any scalar domain, but not with ARRAY or TENSOR. Its
        // concrete scalar type is unknown until match time.
        if(a == Kind::DYNAMIC)
        {
            return b != Kind::ARRAY && b != Kind::TENSOR;
        }
        if(b == Kind::DYNAMIC)
        {
            return a != Kind::ARRAY && a != Kind::TENSOR;
        }
        if(isNumeric(a) && isNumeric(b))
        {
            return true;
        }
        return a == b;
    }

    TypeInfo inferOp(const std::string& op, const nlohmann::json& operand)
    {
        std::vector<nlohmann::json> args;
        if(operand.is_array())
        {
            for(const auto& a : operand)
            {
                args.push_back(a);
            }
        }
        else
        {
            args.push_back(operand); // unary sugar
        }

        const auto argType = [&](std::size_t i) { return inferTypeImpl(args.at(i)); };
        // Reject arity violations at compile with a UmdCompileError (A.7/A.10 §9)
        // rather than letting a fixed-index argType() throw std::out_of_range.
        // `max == 0` means unbounded (n-ary).
        const auto requireArity = [&](std::size_t minArgs, std::size_t maxArgs) {
            if(args.size() < minArgs || (maxArgs != 0 && args.size() > maxArgs))
            {
                throw UmdCompileError("operator '" + op + "' has wrong arity (A.7)");
            }
        };
        const auto requireBool = [&](const TypeInfo& t) {
            if(t.kind != Kind::BOOL && t.kind != Kind::DYNAMIC)
            {
                throw UmdCompileError("operator '" + op + "' requires Bool arguments (A.7)");
            }
        };
        const auto requireNumeric = [&](const TypeInfo& t) {
            if(!isNumeric(t.kind) && t.kind != Kind::DYNAMIC)
            {
                throw UmdCompileError("operator '" + op + "' requires numeric arguments (A.7)");
            }
        };

        if(op == "and" || op == "or")
        {
            requireArity(2, 0);
            for(std::size_t i = 0; i < args.size(); ++i)
            {
                requireBool(argType(i));
            }
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(op == "!" || op == "!!")
        {
            requireArity(1, 1);
            requireBool(argType(0));
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(op == "==" || op == "!=")
        {
            requireArity(2, 2);
            const TypeInfo a = argType(0);
            const TypeInfo b = argType(1);
            if(!sameDomain(a.kind, b.kind))
            {
                throw UmdCompileError("'" + op + "' compares values of different types (A.7)");
            }
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(op == "<" || op == "<=" || op == ">" || op == ">=")
        {
            requireArity(2, 2);
            requireNumeric(argType(0));
            requireNumeric(argType(1));
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(op == "in")
        {
            requireArity(2, 2);
            const TypeInfo needle = argType(0);
            const TypeInfo hay = argType(1);
            if(hay.kind != Kind::ARRAY)
            {
                throw UmdCompileError("'in' requires an array second argument (A.7)");
            }
            if(hay.elem != Kind::UNKNOWN && !sameDomain(needle.kind, hay.elem))
            {
                throw UmdCompileError("'in' element type mismatch (A.7)");
            }
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(op == "+" || op == "*" || op == "min" || op == "max")
        {
            requireArity(2, 0); // n-ary
            Kind k = Kind::INT;
            for(std::size_t i = 0; i < args.size(); ++i)
            {
                const TypeInfo t = argType(i);
                requireNumeric(t);
                if(t.kind == Kind::FLOAT)
                {
                    k = Kind::FLOAT;
                }
            }
            return {k, Kind::UNKNOWN};
        }
        if(op == "-")
        {
            requireArity(2, 2);
            const TypeInfo a = argType(0);
            const TypeInfo b = argType(1);
            requireNumeric(a);
            requireNumeric(b);
            return {(a.kind == Kind::FLOAT || b.kind == Kind::FLOAT) ? Kind::FLOAT : Kind::INT,
                    Kind::UNKNOWN};
        }
        if(op == "%" || op == "ceil_div")
        {
            requireArity(2, 2);
            requireNumeric(argType(0));
            requireNumeric(argType(1));
            return {Kind::INT, Kind::UNKNOWN};
        }
        if(op == "/" || op == "pow")
        {
            requireArity(2, 2);
            requireNumeric(argType(0));
            requireNumeric(argType(1));
            return {Kind::FLOAT, Kind::UNKNOWN};
        }
        if(op == "abs")
        {
            requireArity(1, 1);
            const TypeInfo t = argType(0);
            requireNumeric(t);
            return {t.kind, Kind::UNKNOWN};
        }
        if(op == "log2" || op == "rsqrt")
        {
            requireArity(1, 1);
            requireNumeric(argType(0));
            return {Kind::FLOAT, Kind::UNKNOWN};
        }
        if(op == "value_or_default")
        {
            // A.7: the fallback is any expression of the same type, not just a
            // literal, so both arms are checked and unified. DYNAMIC carries no
            // concrete type, so the other arm supplies the result kind.
            requireArity(2, 2);
            const TypeInfo a = argType(0);
            const TypeInfo b = argType(1);
            if(!sameDomain(a.kind, b.kind))
            {
                throw UmdCompileError("'value_or_default' arms have different types (A.7): "
                                      + kindName(a.kind) + " vs " + kindName(b.kind));
            }
            return a.kind == Kind::DYNAMIC ? b : a;
        }
        if(op == "present" || op == "not_present")
        {
            // A.7: n-ary resolution predicates. Each argument is a variable
            // reference resolved as itself -- never through the `.present`
            // field form -- so a required operand is legal here, and the result
            // is always a real Bool because the operator inspects resolution
            // rather than a value.
            requireArity(1, 0);
            for(std::size_t i = 0; i < args.size(); ++i)
            {
                const nlohmann::json& a = args.at(i);
                if(!a.is_string() || a.get_ref<const std::string&>().size() < 2
                   || a.get_ref<const std::string&>().front() != '$')
                {
                    throw UmdCompileError("operator '" + op
                                          + "' requires variable-reference arguments (A.7)");
                }
                argType(i); // the reference must resolve (A.10 §8)
            }
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(op == "if")
        {
            return inferIf(args);
        }
        if(op.find('.') != std::string::npos)
        {
            // Custom operations (native predicates) are out of scope for this
            // PoC: no predicate registry ships, so the name fails to resolve
            // (RFC 0018 §8 / A.9: fail closed).
            throw UmdCompileError("custom operation not supported in PoC: " + op);
        }
        throw UmdCompileError("unrecognized operator: " + op);
    }

    TypeInfo inferIf(const std::vector<nlohmann::json>& args)
    {
        if(args.size() < 3 || args.size() % 2 == 0)
        {
            throw UmdCompileError("'if' requires 3 or 2n+1 arguments (A.7)");
        }
        TypeInfo result;
        bool haveResult = false;
        for(std::size_t i = 0; i + 1 < args.size(); i += 2)
        {
            const Kind cond = inferTypeImpl(args.at(i)).kind;
            if(cond != Kind::BOOL && cond != Kind::DYNAMIC)
            {
                throw UmdCompileError("'if' condition must be Bool (A.7)");
            }
            const TypeInfo branch = inferTypeImpl(args.at(i + 1));
            result = unify(result, branch, haveResult);
            haveResult = true;
        }
        if(args.size() % 2 == 1)
        {
            result = unify(result, inferTypeImpl(args.back()), haveResult);
        }
        return result;
    }

    static TypeInfo unify(const TypeInfo& acc, const TypeInfo& next, bool haveAcc)
    {
        if(!haveAcc)
        {
            return next;
        }
        if(!sameDomain(acc.kind, next.kind))
        {
            throw UmdCompileError("'if' branch results must share a type (A.7)");
        }
        return acc;
    }

    // Resolve the static type of a `$`-stripped variable path against the
    // registry and shape maps; throws if the reference does not resolve
    // (A.10 §8).
    static TypeInfo varTypeOf(const std::string& varPath, const CompiledUmd& out)
    {
        std::string root;
        std::string rest;
        path::splitRoot(varPath, root, rest);

        if(root == "graph")
        {
            if(rest == "node_count")
            {
                return {Kind::INT, Kind::UNKNOWN};
            }
            if(rest == "is_override_shape_enabled")
            {
                return {Kind::BOOL, Kind::UNKNOWN};
            }
            throw UmdCompileError("unknown graph field: " + rest);
        }
        if(root == "device")
        {
            if(rest.empty())
            {
                throw UmdCompileError("device reference needs a field");
            }
            return {Kind::INT, Kind::UNKNOWN}; // device scalars are numeric
        }
        if(root == "kernel")
        {
            if(rest.empty())
            {
                throw UmdCompileError("kernel reference needs a field");
            }
            // Kernel field types come from a UKD's KMD, unavailable at
            // UMD-compile time; resolve their domain at match time (DYNAMIC).
            // Field existence is validated against the engine's KMD schema at
            // load, so an undeclared field is rejected before match runs.
            return {Kind::DYNAMIC, Kind::UNKNOWN};
        }
        const NodeSpec* node = out.findNode(root);
        if(node != nullptr)
        {
            return attrType(rest, *node);
        }
        const TensorVarSpec* spec = out.findTvar(root);
        if(spec != nullptr)
        {
            return tensorFieldType(rest, *spec);
        }
        throw UmdCompileError("unresolved reference (A.10 §8): $" + varPath);
    }
    static TypeInfo attrType(const std::string& rest, const NodeSpec& node)
    {
        if(rest.empty())
        {
            // A bare node id is not a value reference (A.4 attr-ref needs a field).
            throw UmdCompileError("node id $" + node.id + " is not a value reference (A.4)");
        }
        std::string attr = rest;
        const bool present = path::stripPresentSuffix(attr);
        for(std::size_t i = 0; i < node.opSchema->attributeCount; ++i)
        {
            if(node.opSchema->attributes[i].name == attr)
            {
                if(present)
                {
                    if(!node.opSchema->attributes[i].optional)
                    {
                        throw UmdCompileError(".present on a required attribute is refused (A.4): "
                                              + attr);
                    }
                    return {Kind::BOOL, Kind::UNKNOWN};
                }
                switch(node.opSchema->attributes[i].type)
                {
                case reg::AttrType::INT:
                    return {Kind::INT, Kind::UNKNOWN};
                case reg::AttrType::FLOAT:
                    return {Kind::FLOAT, Kind::UNKNOWN};
                case reg::AttrType::BOOL:
                    return {Kind::BOOL, Kind::UNKNOWN};
                case reg::AttrType::DTYPE:
                    return {Kind::DTYPE, Kind::UNKNOWN};
                default:
                    break;
                }
            }
        }
        throw UmdCompileError("unknown attribute for " + node.opcode + ": " + attr);
    }

    static TypeInfo tensorFieldType(const std::string& rest, const TensorVarSpec& spec)
    {
        if(rest == "present")
        {
            if(!spec.optional)
            {
                throw UmdCompileError(".present on a required operand is refused (A.4): "
                                      + spec.tvar);
            }
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(rest.empty() || rest == "uid")
        {
            return rest.empty() ? TypeInfo{Kind::TENSOR, Kind::UNKNOWN}
                                : TypeInfo{Kind::INT, Kind::UNKNOWN};
        }
        if(rest == "rank")
        {
            return {Kind::INT, Kind::UNKNOWN};
        }
        if(rest == "dtype")
        {
            return {Kind::DTYPE, Kind::UNKNOWN};
        }
        if(rest == "stride_order")
        {
            return {Kind::ARRAY, Kind::INT};
        }
        if(rest == "packed" || rest == "virtual" || rest == "is_runtime_pass_by_value")
        {
            return {Kind::BOOL, Kind::UNKNOWN};
        }
        if(rest == "value_f32")
        {
            // The schema layer coerces whichever arm of the tensor's `value`
            // union is set to f32 and publishes it as one typed token; it reads
            // null when the tensor carries no compile-time value (RFC 0017 §5).
            return {Kind::FLOAT, Kind::UNKNOWN};
        }
        for(const std::string_view prefix : {std::string_view("dims"), std::string_view("strides")})
        {
            if(path::isSubscriptOf(rest, prefix))
            {
                std::size_t idx = 0;
                if(!path::parseSubscript(rest, prefix, idx))
                {
                    throw UmdCompileError("malformed subscript $" + spec.tvar + "." + rest
                                          + " (A.4 requires a non-negative integer index)");
                }
                return {Kind::INT, Kind::UNKNOWN};
            }
        }
        if(spec.dimNames.find(rest) != spec.dimNames.end())
        {
            return {Kind::INT, Kind::UNKNOWN};
        }
        throw UmdCompileError("unknown tensor field $" + spec.tvar + "." + rest);
    }

    const CompiledUmd* _ctx = nullptr;
};

} // namespace hip_kernel_provider_common::umd
