// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>

#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace hipdnn::graph_matcher {

namespace {

using json = nlohmann::json;
namespace data = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* kSchemaV1 = "hipdnn.criteria/v1";

// Thrown internally to abort a compile with a message; caught at the top of
// fromJson and turned into a CompileResult error. Never escapes the compiler.
struct CompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& msg) {
    throw CompileError(msg);
}

// Maximum JSON nesting depth in `s`, ignoring brackets inside strings. Computed
// before handing bytes to nlohmann's recursive-descent parser so adversarial
// deep nesting is rejected up front rather than overflowing the parser stack.
uint32_t scanMaxDepth(std::string_view s) {
    uint32_t depth = 0;
    uint32_t maxDepth = 0;
    bool inString = false;
    bool escaped = false;
    for (const char ch : s) {
        if (inString) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                inString = false;
            }
            continue;
        }
        switch (ch) {
            case '"':
                inString = true;
                break;
            case '{':
            case '[':
                ++depth;
                maxDepth = std::max(maxDepth, depth);
                break;
            case '}':
            case ']':
                if (depth > 0) {
                    --depth;
                }
                break;
            default:
                break;
        }
    }
    return maxDepth;
}

int32_t dtypeFromName(std::string_view name) {
    const char* const* names = data::EnumNamesDataType();
    for (size_t i = 0; names[i] != nullptr; ++i) {
        if (name == names[i]) {
            return static_cast<int32_t>(i);
        }
    }
    fail("unknown dtype '" + std::string{name} + "'");
}

// Type-checked JSON accessors that fail with context instead of throwing raw.
const json& expectObject(const json& j, const char* what) {
    if (!j.is_object()) {
        fail(std::string{what} + " must be an object");
    }
    return j;
}

std::string expectString(const json& j, const char* what, const CompileLimits& limits) {
    if (!j.is_string()) {
        fail(std::string{what} + " must be a string");
    }
    auto s = j.get<std::string>();
    if (s.size() > limits.maxNameLen) {
        fail(std::string{what} + " exceeds max length");
    }
    return s;
}

// One operand/result value: "$x" or {"var":"$x","optional":true}.
PatternBuilder::EdgeSpec parseEdge(std::string_view role, const json& value,
                                   std::vector<std::string>& storage, const CompileLimits& limits) {
    std::string var;
    bool optional = false;
    if (value.is_string()) {
        var = value.get<std::string>();
    } else if (value.is_object()) {
        if (!value.contains("var")) {
            fail("operand/result object for role '" + std::string{role} + "' needs a \"var\"");
        }
        var = expectString(value.at("var"), "operand var", limits);
        if (value.contains("optional")) {
            if (!value.at("optional").is_boolean()) {
                fail("\"optional\" must be a boolean");
            }
            optional = value.at("optional").get<bool>();
        }
    } else {
        fail("operand/result for role '" + std::string{role} + "' must be a string or object");
    }
    if (var.size() > limits.maxNameLen) {
        fail("variable name exceeds max length");
    }
    // The EdgeSpec holds string_views; back them with stable storage.
    storage.emplace_back(std::move(var));
    return PatternBuilder::EdgeSpec{role, storage.back(), optional};
}

Cmp parseCmpKey(const std::string& key) {
    if (key == "equals" || key == "eq") {
        return Cmp::Eq;
    }
    if (key == "not_equals" || key == "ne") {
        return Cmp::NotEq;
    }
    if (key == "at_most") {
        return Cmp::AtMost;
    }
    if (key == "at_least") {
        return Cmp::AtLeast;
    }
    if (key == "one_of") {
        return Cmp::OneOf;
    }
    fail("unknown comparison '" + key + "'");
}

int64_t parseIntValue(const json& j, const char* what) {
    if (j.is_boolean()) {
        return j.get<bool>() ? 1 : 0;
    }
    if (j.is_number_integer()) {
        return j.get<int64_t>();
    }
    fail(std::string{what} + " must be an integer or boolean");
}

// A count spec: "exactly_once" | {"eq"|"at_most"|"at_least": N}.
void parseCount(const json& j, Cmp& cmp, int64_t& n) {
    if (j.is_string()) {
        const auto s = j.get<std::string>();
        if (s == "exactly_once") {
            cmp = Cmp::Eq;
            n = 1;
            return;
        }
        fail("unknown count keyword '" + s + "'");
    }
    if (j.is_object()) {
        if (j.size() != 1) {
            fail("count object must have exactly one comparison key");
        }
        const auto& item = j.items().begin();
        cmp = parseCmpKey(item.key());
        n = parseIntValue(item.value(), "count");
        return;
    }
    fail("count must be a keyword string or a comparison object");
}

std::vector<int32_t> parseDtypeSet(const json& j, bool& negated, const CompileLimits& limits) {
    std::vector<int32_t> out;
    negated = false;
    const auto add = [&](const json& arr) {
        if (!arr.is_array()) {
            fail("dtype set must be an array");
        }
        if (arr.size() > limits.maxSetSize) {
            fail("dtype set exceeds max size");
        }
        for (const auto& e : arr) {
            out.push_back(dtypeFromName(expectString(e, "dtype", limits)));
        }
    };
    if (j.is_string()) {
        out.push_back(dtypeFromName(j.get<std::string>()));
    } else if (j.is_object()) {
        if (j.contains("one_of")) {
            add(j.at("one_of"));
        } else if (j.contains("not_one_of")) {
            add(j.at("not_one_of"));
            negated = true;
        } else if (j.contains("exact")) {
            out.push_back(dtypeFromName(expectString(j.at("exact"), "dtype", limits)));
        } else {
            fail("dtype object needs one_of / not_one_of / exact");
        }
    } else {
        fail("dtype must be a string or object");
    }
    return out;
}

void parseShape(PatternBuilder& builder, const std::string& var, const json& arr,
                const CompileLimits& limits) {
    if (!arr.is_array()) {
        fail("shape must be an array");
    }
    if (arr.size() > limits.maxShapeDims) {
        fail("shape exceeds max dims");
    }
    std::vector<DimSpec> dims;
    dims.reserve(arr.size());
    for (const auto& e : arr) {
        if (e.is_number_integer()) {
            dims.push_back(DimSpec::lit(e.get<int64_t>()));
        } else if (e.is_string()) {
            const auto s = e.get<std::string>();
            if (s.size() > limits.maxNameLen) {
                fail("shape symbol exceeds max length");
            }
            dims.push_back(s == "?" ? DimSpec::any() : DimSpec::of(s));
        } else {
            fail("shape element must be an integer or a symbol string");
        }
    }
    builder.constrainShape(var, dims);
}

void parseLayout(PatternBuilder& builder, const std::string& var, const json& j) {
    if (j.is_string()) {
        if (j.get<std::string>() == "contiguous") {
            builder.constrainContiguous(var);
            return;
        }
        fail("unknown layout keyword '" + j.get<std::string>() + "'");
    }
    if (j.is_object() && j.contains("order")) {
        const auto& order = j.at("order");
        if (!order.is_array()) {
            fail("layout order must be an array");
        }
        std::vector<uint32_t> axes;
        for (const auto& e : order) {
            if (!e.is_number_unsigned()) {
                fail("layout order axes must be non-negative integers");
            }
            axes.push_back(e.get<uint32_t>());
        }
        builder.constrainLayout(var, axes);
        return;
    }
    fail("layout must be \"contiguous\" or {\"order\":[...]}");
}

// A var-targeted constraint object: dtype/rank/shape/layout/use/consumers/
// no_consumer_outside, each optional.
void parseVarConstraint(PatternBuilder& builder, const std::string& var, const json& obj,
                        const CompileLimits& limits) {
    static const char* const kKeys[] = {
        "dtype", "rank", "shape", "layout", "use", "consumers", "no_consumer_outside"};
    bool anyKey = false;
    for (const char* key : kKeys) {
        if (obj.contains(key)) {
            anyKey = true;
            break;
        }
    }
    if (!anyKey) {
        fail(
            "constraint on '" + var +
            "' has no recognized keys (dtype/rank/shape/layout/use/consumers/no_consumer_outside)");
    }
    if (obj.contains("dtype")) {
        bool negated = false;
        auto set = parseDtypeSet(obj.at("dtype"), negated, limits);
        builder.constrainDtype(var, std::move(set), negated);
    }
    if (obj.contains("rank")) {
        if (!obj.at("rank").is_number_unsigned()) {
            fail("rank must be a non-negative integer");
        }
        builder.constrainRank(var, obj.at("rank").get<uint32_t>());
    }
    if (obj.contains("shape")) {
        parseShape(builder, var, obj.at("shape"), limits);
    }
    if (obj.contains("layout")) {
        parseLayout(builder, var, obj.at("layout"));
    }
    if (obj.contains("use")) {
        Cmp cmp = Cmp::Eq;
        int64_t n = 0;
        parseCount(obj.at("use"), cmp, n);
        builder.constrainUseCount(var, cmp, n);
    }
    if (obj.contains("consumers")) {
        Cmp cmp = Cmp::Eq;
        int64_t n = 0;
        parseCount(obj.at("consumers"), cmp, n);
        builder.constrainConsumerCount(var, cmp, n);
    }
    if (obj.contains("no_consumer_outside")) {
        if (!obj.at("no_consumer_outside").is_boolean()) {
            fail("no_consumer_outside must be a boolean");
        }
        if (obj.at("no_consumer_outside").get<bool>()) {
            builder.constrainNoConsumerOutside(var);
        }
    }
}

// A node-targeted attribute constraint: {"attr":{"name":{"equals":v}|{"one_of":[...]}}}.
void parseAttrConstraint(PatternBuilder& builder, uint32_t nodeIndex, const json& attrObj,
                         const CompileLimits& limits) {
    for (const auto& item : attrObj.items()) {
        const std::string attrName = item.key();
        if (attrName.size() > limits.maxNameLen) {
            fail("attribute name exceeds max length");
        }
        const json& cmpObj = item.value();
        if (!cmpObj.is_object() || cmpObj.size() != 1) {
            fail("attribute '" + attrName + "' needs a single comparison object");
        }
        const auto entry = cmpObj.items().begin();
        const Cmp cmp = parseCmpKey(entry.key());
        const bool negated = false;
        std::vector<int64_t> values;
        if (cmp == Cmp::OneOf) {
            if (!entry.value().is_array()) {
                fail("attribute one_of needs an array");
            }
            if (entry.value().size() > limits.maxSetSize) {
                fail("attribute one_of exceeds max size");
            }
            for (const auto& v : entry.value()) {
                values.push_back(parseIntValue(v, "attribute value"));
            }
        } else {
            values.push_back(parseIntValue(entry.value(), "attribute value"));
        }
        builder.constrainAttr(nodeIndex, attrName, cmp, std::move(values), negated);
    }
}

void parseCrossTensor(PatternBuilder& builder, const json& c, const CompileLimits& limits) {
    const std::string kind = expectString(c.at("kind"), "constraint kind", limits);
    const bool negated =
        c.contains("negated") && c.at("negated").is_boolean() ? c.at("negated").get<bool>() : false;
    if (!c.contains("args") || !c.at("args").is_array()) {
        fail("constraint '" + kind + "' needs an \"args\" array");
    }
    const json& args = c.at("args");
    if (kind == "same_dtype") {
        if (args.size() != 2) {
            fail("same_dtype needs exactly 2 args");
        }
        builder.constrainSameDtype(expectString(args[0], "arg", limits),
                                   expectString(args[1], "arg", limits), negated);
    } else if (kind == "same_dim") {
        if (args.size() != 4 || !args[1].is_number_unsigned() || !args[3].is_number_unsigned()) {
            fail("same_dim needs args [varA, axisA, varB, axisB]");
        }
        builder.constrainSameDim(expectString(args[0], "arg", limits), args[1].get<uint32_t>(),
                                 expectString(args[2], "arg", limits), args[3].get<uint32_t>(),
                                 negated);
    } else if (kind == "native_predicate") {
        if (!c.contains("name")) {
            fail("native_predicate needs a \"name\"");
        }
        const std::string pname = expectString(c.at("name"), "predicate name", limits);
        if (args.size() > limits.maxSetSize) {
            fail("native_predicate exceeds max args");
        }
        std::vector<std::string> argStorage;  // back the string_views
        argStorage.reserve(args.size());
        std::vector<PatternBuilder::PredArgSpec> specs;
        for (const auto& a : args) {
            PatternBuilder::PredArgSpec spec{};
            if (a.is_string()) {
                argStorage.push_back(a.get<std::string>());
                const std::string& nm = argStorage.back();
                if (nm.size() > limits.maxNameLen) {
                    fail("predicate arg name exceeds max length");
                }
                spec.source = (!nm.empty() && nm.front() == '$') ? PredicateArg::Source::Var
                                                                 : PredicateArg::Source::Sym;
                spec.name = nm;
            } else if (a.is_number_integer()) {
                spec.source = PredicateArg::Source::Literal;
                spec.literal = a.get<int64_t>();
            } else {
                fail("native_predicate arg must be a $variable, a symbol string, or an integer");
            }
            specs.push_back(spec);
        }
        builder.addPredicate(pname, specs, negated);
    } else {
        fail("unknown constraint kind '" + kind + "'");
    }
}

CompiledPattern compile(std::string_view rawJson, const OpSchemaRegistry& registry,
                        const CompileLimits& limits, Provenance provenance,
                        const PredicateRegistry& predicates, std::string& outName,
                        int64_t& outPriority) {
    if (rawJson.size() > limits.maxInputBytes) {
        fail("input exceeds max size");
    }
    if (scanMaxDepth(rawJson) > limits.maxDepth) {
        fail("JSON nesting exceeds max depth");
    }

    json doc = json::parse(rawJson, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded()) {
        fail("malformed JSON");
    }
    expectObject(doc, "top-level");

    const std::string schema =
        doc.contains("schema") ? expectString(doc.at("schema"), "schema", limits) : std::string{};
    if (schema != kSchemaV1) {
        fail("unsupported schema '" + schema + "'; this runtime understands " + kSchemaV1);
    }

    if (doc.contains("name")) {
        outName = expectString(doc.at("name"), "name", limits);
    }
    if (doc.contains("priority")) {
        if (!doc.at("priority").is_number_integer()) {
            fail("\"priority\" must be an integer");
        }
        outPriority = doc.at("priority").get<int64_t>();
    }

    if (!doc.contains("nodes") || !doc.at("nodes").is_array()) {
        fail("\"nodes\" array is required");
    }
    const json& nodes = doc.at("nodes");
    if (nodes.empty()) {
        fail("pattern has no nodes");
    }
    if (nodes.size() > limits.maxNodes) {
        fail("pattern exceeds max nodes");
    }

    PatternBuilder builder(registry, provenance, predicates);
    // string_view-backed EdgeSpecs need their strings to outlive addNode.
    std::vector<std::string> edgeStorage;
    edgeStorage.reserve(nodes.size() * 4);
    std::unordered_map<std::string, uint32_t> idToIndex;
    int64_t anchorIndex = -1;

    for (const auto& node : nodes) {
        expectObject(node, "node");
        if (!node.contains("op")) {
            fail("node needs an \"op\"");
        }
        const std::string op = expectString(node.at("op"), "op", limits);

        const auto collect = [&](const char* key) {
            std::vector<PatternBuilder::EdgeSpec> edges;
            if (node.contains(key)) {
                const json& m = node.at(key);
                if (!m.is_object()) {
                    fail(std::string{key} + " must be an object");
                }
                if (m.size() > limits.maxEdgesPerNode) {
                    fail(std::string{key} + " exceeds max edges per node");
                }
                for (const auto& item : m.items()) {
                    edges.push_back(parseEdge(item.key(), item.value(), edgeStorage, limits));
                }
            }
            return edges;
        };

        const auto operands = collect("operands");
        const auto results = collect("results");
        const uint32_t index = builder.addNode(op, operands, results);

        if (node.contains("id")) {
            const std::string id = expectString(node.at("id"), "node id", limits);
            if (!idToIndex.emplace(id, index).second) {
                fail("duplicate node id '" + id + "'");
            }
        }
        if (node.contains("anchor") && node.at("anchor").is_boolean() &&
            node.at("anchor").get<bool>()) {
            if (anchorIndex >= 0) {
                fail("more than one node marked anchor");
            }
            anchorIndex = index;
        }
    }

    if (anchorIndex >= 0) {
        builder.setAnchor(static_cast<uint32_t>(anchorIndex));
    }

    if (doc.contains("constraints")) {
        const json& constraints = doc.at("constraints");
        if (!constraints.is_array()) {
            fail("\"constraints\" must be an array");
        }
        if (constraints.size() > limits.maxConstraints) {
            fail("pattern exceeds max constraints");
        }
        for (const auto& c : constraints) {
            expectObject(c, "constraint");
            if (c.contains("kind")) {
                parseCrossTensor(builder, c, limits);
                continue;
            }
            if (!c.contains("on")) {
                fail("constraint needs an \"on\" target or a \"kind\"");
            }
            const std::string on = expectString(c.at("on"), "constraint target", limits);
            if (!on.empty() && on.front() == '$') {
                parseVarConstraint(builder, on, c, limits);
            } else {
                // Node-targeted: attribute constraint referencing a node id.
                const auto it = idToIndex.find(on);
                if (it == idToIndex.end()) {
                    fail("constraint target '" + on + "' is neither a $variable nor a node id");
                }
                if (!c.contains("attr")) {
                    fail("node-targeted constraint on '" + on + "' needs an \"attr\" object");
                }
                expectObject(c.at("attr"), "attr");
                parseAttrConstraint(builder, it->second, c.at("attr"), limits);
            }
        }
    }

    return builder.build();
}

}  // namespace

CompileResult PatternCompiler::fromJson(std::string_view json, const OpSchemaRegistry& registry,
                                        const CompileLimits& limits, Provenance provenance,
                                        const PredicateRegistry& predicates) {
    CompileResult result;
    try {
        result.pattern =
            compile(json, registry, limits, provenance, predicates, result.name, result.priority);
        result.ok = true;
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = e.what();
    }
    return result;
}

}  // namespace hipdnn::graph_matcher
