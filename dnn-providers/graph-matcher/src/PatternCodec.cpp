// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstring>
#include <hipdnn_graph_matcher/PatternCodec.hpp>
#include <stdexcept>
#include <unordered_map>

namespace hipdnn::graph_matcher {

namespace {

constexpr uint8_t kMagic[4] = {'H', 'D', 'G', 'M'};
constexpr uint8_t kEndianByte = 0;  // 0 = little-endian; big-endian unsupported

// --- Writer ---------------------------------------------------------------

class Writer {
   public:
    void u8(uint8_t v) {
        _bytes.push_back(v);
    }
    void u16(uint16_t v) {
        u8(static_cast<uint8_t>(v));
        u8(static_cast<uint8_t>(v >> 8));
    }
    void u32(uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            u8(static_cast<uint8_t>(v >> (8 * i)));
        }
    }
    void i64(int64_t v) {
        u64(static_cast<uint64_t>(v));
    }
    void u64(uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            u8(static_cast<uint8_t>(v >> (8 * i)));
        }
    }
    void raw(const uint8_t* p, size_t n) {
        _bytes.insert(_bytes.end(), p, p + n);
    }
    std::vector<uint8_t> take() {
        return std::move(_bytes);
    }

   private:
    std::vector<uint8_t> _bytes;
};

// Deterministic string interner: indices assigned in first-encounter order.
class Interner {
   public:
    uint32_t intern(const std::string& s) {
        const auto it = _index.find(s);
        if (it != _index.end()) {
            return it->second;
        }
        const auto id = static_cast<uint32_t>(_pool.size());
        _pool.push_back(s);
        _index.emplace(s, id);
        return id;
    }
    const std::vector<std::string>& pool() const {
        return _pool;
    }

   private:
    std::vector<std::string> _pool;
    std::unordered_map<std::string, uint32_t> _index;
};

void writeEdges(Writer& w, const std::vector<PatternEdge>& edges) {
    w.u32(static_cast<uint32_t>(edges.size()));
    for (const auto& e : edges) {
        w.u32(e.roleIndex);
        w.u32(e.var);
        w.u8(e.optional ? 1 : 0);
    }
}

// --- Reader (bounded, fail-closed) ----------------------------------------

struct DecodeError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const std::string& msg) {
    throw DecodeError(msg);
}

class Reader {
   public:
    Reader(const uint8_t* data, size_t size) : _p(data), _end(data + size) {}

    size_t remaining() const {
        return static_cast<size_t>(_end - _p);
    }

    uint8_t u8() {
        if (remaining() < 1) {
            fail("truncated: expected u8");
        }
        return *_p++;
    }
    uint16_t u16() {
        const uint16_t lo = u8();
        const uint16_t hi = u8();
        return static_cast<uint16_t>(lo | (hi << 8));
    }
    uint32_t u32() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            v |= static_cast<uint32_t>(u8()) << (8 * i);
        }
        return v;
    }
    int64_t i64() {
        return static_cast<int64_t>(u64());
    }
    uint64_t u64() {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= static_cast<uint64_t>(u8()) << (8 * i);
        }
        return v;
    }

    // A count is valid only if it cannot exceed the bytes left (each element
    // consumes >= 1 byte), which caps allocations without a magic constant.
    uint32_t count(const char* what) {
        const uint32_t n = u32();
        if (n > remaining()) {
            fail(std::string{"implausible count for "} + what);
        }
        return n;
    }

    std::string str() {
        const uint32_t len = u32();
        if (len > remaining()) {
            fail("truncated: string");
        }
        std::string s(reinterpret_cast<const char*>(_p), len);
        _p += len;
        return s;
    }

   private:
    const uint8_t* _p;
    const uint8_t* _end;
};

// Resolves a pool index or fails.
const std::string& poolAt(const std::vector<std::string>& pool, uint32_t idx) {
    if (idx >= pool.size()) {
        fail("string index out of range");
    }
    return pool[idx];
}

void requireLt(uint32_t value, size_t bound, const char* what) {
    if (value >= bound) {
        fail(std::string{"id out of range: "} + what);
    }
}

std::vector<PatternEdge> readEdges(Reader& r, uint32_t varCount) {
    std::vector<PatternEdge> edges;
    const uint32_t n = r.count("edges");
    edges.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        PatternEdge e;
        e.roleIndex = r.u32();
        e.var = r.u32();
        requireLt(e.var, varCount, "edge var");
        e.optional = r.u8() != 0;
        edges.push_back(e);
    }
    return edges;
}

}  // namespace

std::vector<uint8_t> PatternCodec::serialize(const CompiledPattern& p) {
    // Pass 1: intern every string in a fixed order for deterministic output.
    Interner interner;
    for (const auto& n : p._varNames) {
        interner.intern(n);
    }
    for (const auto& n : p._symNames) {
        interner.intern(n);
    }
    for (const auto& node : p._nodes) {
        interner.intern(node.opcode);
    }
    for (const auto& c : p._constraints) {
        interner.intern(c.name);
    }
    for (const auto& pr : p._predicates) {
        interner.intern(pr.name);
    }

    Writer w;
    w.raw(kMagic, 4);
    w.u16(kPatternWireVersion);
    w.u8(kEndianByte);
    w.u32(p._anchor);

    // String pool.
    const auto& pool = interner.pool();
    w.u32(static_cast<uint32_t>(pool.size()));
    for (const auto& s : pool) {
        w.u32(static_cast<uint32_t>(s.size()));
        w.raw(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }

    // Vars / syms (names as pool indices).
    w.u32(static_cast<uint32_t>(p._varNames.size()));
    for (const auto& n : p._varNames) {
        w.u32(interner.intern(n));
    }
    w.u32(static_cast<uint32_t>(p._symNames.size()));
    for (const auto& n : p._symNames) {
        w.u32(interner.intern(n));
    }

    // Nodes.
    w.u32(static_cast<uint32_t>(p._nodes.size()));
    for (const auto& node : p._nodes) {
        w.u32(interner.intern(node.opcode));
        writeEdges(w, node.operandEdges);
        writeEdges(w, node.resultEdges);
    }

    // Dim bindings.
    w.u32(static_cast<uint32_t>(p._dimBindings.size()));
    for (const auto& db : p._dimBindings) {
        w.u32(db.var);
        w.u32(db.axis);
        w.u32(db.sym);
    }

    // Constraints.
    w.u32(static_cast<uint32_t>(p._constraints.size()));
    for (const auto& c : p._constraints) {
        w.u8(static_cast<uint8_t>(c.kind));
        w.u8(c.negated ? 1 : 0);
        w.u32(c.varA);
        w.u32(c.varB);
        w.u32(c.nodeIndex);
        w.u32(c.axisA);
        w.u32(c.axisB);
        w.u8(static_cast<uint8_t>(c.cmp));
        w.i64(c.ival);
        w.u8(static_cast<uint8_t>(c.layoutKind));
        w.u32(static_cast<uint32_t>(c.ints.size()));
        for (const int64_t v : c.ints) {
            w.i64(v);
        }
        w.u32(static_cast<uint32_t>(c.axisOrder.size()));
        for (const uint32_t v : c.axisOrder) {
            w.u32(v);
        }
        w.u32(interner.intern(c.name));
    }

    // Predicates.
    w.u32(static_cast<uint32_t>(p._predicates.size()));
    for (const auto& pr : p._predicates) {
        w.u32(interner.intern(pr.name));
        w.u8(pr.negated ? 1 : 0);
        w.u32(static_cast<uint32_t>(pr.args.size()));
        for (const auto& a : pr.args) {
            w.u8(static_cast<uint8_t>(a.source));
            w.u32(a.var);
            w.u32(a.sym);
            w.i64(a.literal);
        }
    }

    return w.take();
}

DeserializeResult PatternCodec::deserialize(const uint8_t* data, size_t size) {
    DeserializeResult result;
    if (data == nullptr) {
        result.error = "null buffer";
        return result;
    }
    try {
        Reader r(data, size);
        uint8_t magic[4];
        for (auto& b : magic) {
            b = r.u8();
        }
        if (std::memcmp(magic, kMagic, 4) != 0) {
            fail("bad magic");
        }
        const uint16_t version = r.u16();
        if (version != kPatternWireVersion) {
            fail("unsupported wire version " + std::to_string(version));
        }
        if (r.u8() != kEndianByte) {
            fail("unsupported byte order");
        }

        CompiledPattern p;
        const uint32_t anchor = r.u32();

        // String pool.
        const uint32_t poolCount = r.count("string pool");
        std::vector<std::string> pool;
        pool.reserve(poolCount);
        for (uint32_t i = 0; i < poolCount; ++i) {
            pool.push_back(r.str());
        }

        // Vars / syms.
        const uint32_t varCount = r.count("vars");
        p._varNames.reserve(varCount);
        for (uint32_t i = 0; i < varCount; ++i) {
            p._varNames.push_back(poolAt(pool, r.u32()));
        }
        const uint32_t symCount = r.count("syms");
        p._symNames.reserve(symCount);
        for (uint32_t i = 0; i < symCount; ++i) {
            p._symNames.push_back(poolAt(pool, r.u32()));
        }

        // Nodes.
        const uint32_t nodeCount = r.count("nodes");
        p._nodes.reserve(nodeCount);
        for (uint32_t i = 0; i < nodeCount; ++i) {
            PatternNode node;
            node.opcode = poolAt(pool, r.u32());
            node.operandEdges = readEdges(r, varCount);
            node.resultEdges = readEdges(r, varCount);
            p._nodes.push_back(std::move(node));
        }
        if (anchor >= nodeCount) {
            fail("anchor out of range");
        }
        p._anchor = anchor;

        // Dim bindings.
        const uint32_t dbCount = r.count("dim bindings");
        p._dimBindings.reserve(dbCount);
        for (uint32_t i = 0; i < dbCount; ++i) {
            DimBinding db;
            db.var = r.u32();
            db.axis = r.u32();
            db.sym = r.u32();
            requireLt(db.var, varCount, "dimBinding var");
            requireLt(db.sym, symCount, "dimBinding sym");
            p._dimBindings.push_back(db);
        }

        // Constraints.
        const uint32_t cCount = r.count("constraints");
        p._constraints.reserve(cCount);
        for (uint32_t i = 0; i < cCount; ++i) {
            Constraint c;
            const uint8_t kind = r.u8();
            if (kind > static_cast<uint8_t>(ConstraintKind::SameDim)) {
                fail("unknown constraint kind");
            }
            c.kind = static_cast<ConstraintKind>(kind);
            c.negated = r.u8() != 0;
            c.varA = r.u32();
            c.varB = r.u32();
            c.nodeIndex = r.u32();
            c.axisA = r.u32();
            c.axisB = r.u32();
            const uint8_t cmp = r.u8();
            if (cmp > static_cast<uint8_t>(Cmp::OneOf)) {
                fail("unknown cmp");
            }
            c.cmp = static_cast<Cmp>(cmp);
            c.ival = r.i64();
            const uint8_t layout = r.u8();
            if (layout > static_cast<uint8_t>(LayoutKind::PackedOrder)) {
                fail("unknown layout kind");
            }
            c.layoutKind = static_cast<LayoutKind>(layout);
            const uint32_t nInts = r.count("constraint ints");
            c.ints.reserve(nInts);
            for (uint32_t k = 0; k < nInts; ++k) {
                c.ints.push_back(r.i64());
            }
            const uint32_t nAxes = r.count("constraint axisOrder");
            c.axisOrder.reserve(nAxes);
            for (uint32_t k = 0; k < nAxes; ++k) {
                c.axisOrder.push_back(r.u32());
            }
            c.name = poolAt(pool, r.u32());
            // Reference sanity for the kinds that carry ids.
            if (c.kind != ConstraintKind::Attr) {
                requireLt(c.varA, varCount, "constraint varA");
            }
            if (c.kind == ConstraintKind::SameDtype || c.kind == ConstraintKind::SameDim) {
                requireLt(c.varB, varCount, "constraint varB");
            }
            if (c.kind == ConstraintKind::Attr) {
                requireLt(c.nodeIndex, nodeCount, "constraint nodeIndex");
            }
            p._constraints.push_back(std::move(c));
        }

        // Predicates.
        const uint32_t pCount = r.count("predicates");
        p._predicates.reserve(pCount);
        for (uint32_t i = 0; i < pCount; ++i) {
            PredicateRef pr;
            pr.name = poolAt(pool, r.u32());
            pr.negated = r.u8() != 0;
            const uint32_t na = r.count("predicate args");
            pr.args.reserve(na);
            for (uint32_t k = 0; k < na; ++k) {
                PredicateArg a;
                const uint8_t source = r.u8();
                if (source > static_cast<uint8_t>(PredicateArg::Source::Literal)) {
                    fail("unknown predicate arg source");
                }
                a.source = static_cast<PredicateArg::Source>(source);
                a.var = r.u32();
                a.sym = r.u32();
                a.literal = r.i64();
                if (a.source == PredicateArg::Source::Var) {
                    requireLt(a.var, varCount, "predicate var");
                }
                if (a.source == PredicateArg::Source::Sym) {
                    requireLt(a.sym, symCount, "predicate sym");
                }
                pr.args.push_back(a);
            }
            p._predicates.push_back(std::move(pr));
        }

        result.pattern = std::move(p);
        result.ok = true;
    } catch (const std::exception& e) {
        result.ok = false;
        result.error = e.what();
    }
    return result;
}

std::string PatternCodec::emitEmbeddedArray(std::string_view symbol,
                                            const std::vector<uint8_t>& bytes) {
    std::string out;
    out += "// Generated by graph_matcher PatternCodec; do not edit.\n";
    out += "static constexpr unsigned char ";
    out += symbol;
    out += "[] = {\n";
    char buf[8];
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i % 12 == 0) {
            out += "    ";
        }
        std::snprintf(buf, sizeof(buf), "0x%02x,", bytes[i]);
        out += buf;
        out += (i % 12 == 11) ? '\n' : ' ';
    }
    if (!bytes.empty() && bytes.size() % 12 != 0) {
        out += '\n';
    }
    out += "};\n";
    out += "static constexpr unsigned long ";
    out += symbol;
    out += "_size = ";
    out += std::to_string(bytes.size());
    out += ";\n";
    return out;
}

}  // namespace hipdnn::graph_matcher
