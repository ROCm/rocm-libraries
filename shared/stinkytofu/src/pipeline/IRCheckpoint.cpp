/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 * ************************************************************************ */
#include "stinkytofu/pipeline/IRCheckpoint.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/bindings/python/Module.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace fs = std::filesystem;

namespace stinkytofu {
namespace {

constexpr int kBundleFormatVersion = 1;
constexpr const char* kMetaFile = "meta.json";
// Leaves room for "__<sha16>" inside the 255-byte limit of one path component.
constexpr size_t kMaxKernelDirChars = 200;

[[noreturn]] void fail(const std::string& message) {
    report_fatal_error("IRCheckpoint: " + message);
}

void warn(const std::string& message) {
    std::cerr << "Warning: IRCheckpoint: " << message << "\n";
}

std::string trim(std::string_view text) {
    const size_t first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    const size_t last = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(first, last - first + 1));
}

/// Pass names carry spaces ("CFG Builder"); a file name should not.
std::string fileSafe(std::string_view text) {
    std::string out(text);
    for (char& c : out)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '+' && c != '-' &&
            c != '.')
            c = '_';
    return out;
}

std::string jsonString(std::string_view text) {
    std::string out = "\"";
    for (char c : text) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                out += c;
        }
    }
    return out + "\"";
}

/// The value of a top-level string field in the meta.json the dump wrote.
std::optional<std::string> readJsonString(const std::string& json, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    size_t at = json.find(quoted);
    if (at == std::string::npos) return std::nullopt;
    at = json.find(':', at + quoted.size());
    if (at == std::string::npos) return std::nullopt;
    const size_t open = json.find('"', at);
    if (open == std::string::npos) return std::nullopt;
    const size_t close = json.find('"', open + 1);
    if (close == std::string::npos) return std::nullopt;
    return json.substr(open + 1, close - open - 1);
}

std::string readFile(const fs::path& path) {
    std::ifstream in(path);
    if (!in) fail("cannot read " + path.string());
    std::stringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

void writeFile(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) fail("cannot write " + path.string());
    out << text;
}

std::string withCheckpointId(const std::string& line, size_t id) {
    const std::string tag = "mod.ckpt = { id = " + std::to_string(id) + " }";
    // An instruction ends in its attribute dict; a directive has none.
    if (line.size() >= 2 && line.compare(line.size() - 2, 2, " }") == 0)
        return line.substr(0, line.size() - 2) + ", " + tag + " }";
    return line + " { " + tag + " }";
}

bool isControlFlow(const IRBase* node) {
    const auto* inst = dyn_cast<StinkyInstruction>(node);
    if (!inst) return false;
    return inst->getUnifiedOpcode() == GFX::LABEL || isBranch(*inst) || isEndOfFunction(*inst);
}

/// One IR line of an edited bundle.
struct EditedLine {
    long id = -1;
    std::string text;
    unsigned lineNumber = 0;
};

struct EditedBlock {
    std::string label;
    std::vector<EditedLine> lines;
};

std::vector<EditedBlock> parseEditedBundle(const std::string& text, const std::string& file) {
    std::vector<EditedBlock> blocks;
    std::istringstream in(text);
    unsigned lineNumber = 0;
    for (std::string raw; std::getline(in, raw);) {
        ++lineNumber;
        std::string line = trim(raw);
        if (line.empty() || line.starts_with("//") || line.starts_with("#")) continue;
        if (line.starts_with("st.func") || line == "}" || line.starts_with("Successors:")) continue;
        if (line.front() == '^' && line.back() == ':') {
            blocks.push_back({line.substr(1, line.size() - 2), {}});
            continue;
        }
        if (blocks.empty())
            fail(file + ":" + std::to_string(lineNumber) + ": instruction before the first block");
        EditedLine edited;
        edited.id = takeCheckpointId(line);
        edited.text = trim(line);
        edited.lineNumber = lineNumber;
        blocks.back().lines.push_back(std::move(edited));
    }
    return blocks;
}

/// The IR of \p lines, parsed as one block, in order. Fatal when a line does not parse into
/// exactly one node.
std::vector<IRBase*> parseNewLines(const std::vector<const EditedLine*>& lines,
                                   const std::array<int, 3>& arch, const std::string& file,
                                   Function& scratch) {
    std::string source = "st.func @ckpt() {\n  ^entry:\n";
    for (const EditedLine* line : lines) source += "    " + line->text + "\n";
    source += "}\n";

    const auto parsed = parseSourceStringWithDiagnostics(source);
    if (parsed.hasErrors()) {
        std::string message = file + ": the edited lines do not parse:";
        for (const auto& diag : parsed.diagnostics)
            if (diag.getLevel() == Diagnostic::Level::Error) {
                // Line 3 of the scratch source is the first edited line.
                const unsigned index = diag.getLine() >= 3 ? diag.getLine() - 3 : 0;
                message += "\n  ";
                if (index < lines.size())
                    message += "line " + std::to_string(lines[index]->lineNumber) + ": ";
                message += diag.getMessage();
            }
        fail(message);
    }

    PassContext ctx;
    const GfxArchID archID = getGfxArchID(arch[0], arch[1], arch[2]);
    if (StinkyIRConverter::populateFunctionFromString(source, scratch, ctx, archID) !=
        StinkyErrorCode::SUCCESS)
        fail(file + ": the edited lines do not convert to IR");

    std::vector<IRBase*> nodes;
    for (BasicBlock& bb : scratch)
        for (IRBase& node : bb) nodes.push_back(&node);
    if (nodes.size() != lines.size())
        fail(file + ": " + std::to_string(lines.size()) + " edited lines became " +
             std::to_string(nodes.size()) +
             " IR nodes; an unknown mnemonic or a label line is the usual cause");
    for (size_t i = 0; i < nodes.size(); ++i)
        if (isControlFlow(nodes[i]))
            fail(file + ":" + std::to_string(lines[i]->lineNumber) +
                 ": adding a branch or label is not supported");
    return nodes;
}

/// Dumps or loads the IR of the Function it runs on.
class IRCheckpointPass final : public Pass {
   public:
    static char ID;

    /// \p config belongs to the session, which owns this pass.
    explicit IRCheckpointPass(const IRCheckpointConfig& config) : config(config) {}

    PassID getPassID() const override {
        return &ID;
    }

    const char* getName() const override {
        return "IRCheckpointPass";
    }

    PreservedAnalyses run(Function& function, PassContext& /*ctx*/,
                          AnalysisManager& /*AM*/) override {
        if (!config.dumpDir.empty()) {
            dump(function);
            return PreservedAnalyses::all();
        }
        load(function);
        return PreservedAnalyses::none();
    }

   private:
    std::string stirFileName() const {
        return fileSafe(config.spec.label) + "." + fileSafe(config.spec.passName) + "#" +
               std::to_string(config.spec.occurrence) + ".stir";
    }

    std::string kernelDirPrefix() const {
        std::string kernel = fileSafe(config.kernelName.empty() ? "kernel" : config.kernelName);
        if (kernel.size() > kMaxKernelDirChars) kernel.resize(kMaxKernelDirChars);
        return kernel + "__";
    }

    void dump(Function& function) {
        const IRCheckpointRendering base = renderCheckpointRegion(function, /*withIds=*/false);
        const std::string fingerprint = sha256Hex(base.text);
        const fs::path dir =
            fs::path(config.dumpDir) / (kernelDirPrefix() + fingerprint.substr(0, 16));
        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) fail("cannot create " + dir.string() + ": " + ec.message());

        writeFile(dir / stirFileName(), renderCheckpointRegion(function, /*withIds=*/true).text);

        std::ostringstream meta;
        meta << "{\n"
             << "  \"format_version\": " << kBundleFormatVersion << ",\n"
             << "  \"kernel\": " << jsonString(config.kernelName) << ",\n"
             << "  \"arch\": [" << config.arch[0] << ", " << config.arch[1] << ", "
             << config.arch[2] << "],\n"
             << "  \"label\": " << jsonString(config.spec.label) << ",\n"
             << "  \"pass\": " << jsonString(config.spec.passName) << ",\n"
             << "  \"occurrence\": " << config.spec.occurrence << ",\n"
             << "  \"spec\": " << jsonString(config.spec.str()) << ",\n"
             << "  \"stir\": " << jsonString(stirFileName()) << ",\n"
             << "  \"instructions\": " << base.nodes.size() << ",\n"
             << "  \"fingerprint\": " << jsonString(fingerprint) << "\n"
             << "}\n";
        writeFile(dir / kMetaFile, meta.str());
        std::cerr << "IRCheckpoint: dumped " << (dir / stirFileName()).string() << "\n";
    }

    /// The bundle for this kernel in the load directory: the one whose fingerprint matches, else
    /// the only one there (a mismatch, reported by the caller), else none.
    std::optional<fs::path> findBundle(const std::string& fingerprint) const {
        const fs::path exact =
            fs::path(config.loadDir) / (kernelDirPrefix() + fingerprint.substr(0, 16));
        if (fs::exists(exact / stirFileName())) return exact;

        std::vector<fs::path> candidates;
        std::error_code ec;
        for (const auto& entry : fs::directory_iterator(config.loadDir, ec)) {
            const std::string name = entry.path().filename().string();
            if (entry.is_directory() && name.starts_with(kernelDirPrefix()) &&
                fs::exists(entry.path() / stirFileName()))
                candidates.push_back(entry.path());
        }
        if (ec) fail("cannot read the load directory " + config.loadDir + ": " + ec.message());
        if (candidates.size() > 1)
            fail("several bundles for kernel " + config.kernelName + " in " + config.loadDir +
                 " and none matches the current IR");
        if (candidates.empty()) return std::nullopt;
        return candidates.front();
    }

    void load(Function& function) {
        const IRCheckpointRendering base = renderCheckpointRegion(function, /*withIds=*/false);
        const std::string fingerprint = sha256Hex(base.text);
        const auto bundle = findBundle(fingerprint);
        if (!bundle) {
            warn("no bundle for kernel " + config.kernelName + " (" + stirFileName() + ") in " +
                 config.loadDir + "; the kernel is built unedited");
            return;
        }
        const fs::path stirPath = *bundle / stirFileName();
        const std::string file = stirPath.string();

        const auto metaFingerprint = readJsonString(readFile(*bundle / kMetaFile), "fingerprint");
        if (!metaFingerprint) fail((*bundle / kMetaFile).string() + " has no fingerprint");
        if (*metaFingerprint != fingerprint)
            fail("the bundle " + file + " was dumped from different IR.\n  bundle fingerprint " +
                 *metaFingerprint + "\n  current fingerprint " + fingerprint +
                 "\n  Dump again from this build, or restore the build options the bundle was "
                 "dumped with.");

        const std::vector<EditedBlock> edited = parseEditedBundle(readFile(stirPath), file);
        if (edited.size() != base.blockLabels.size())
            fail(file + ": " + std::to_string(edited.size()) + " blocks, the IR has " +
                 std::to_string(base.blockLabels.size()) + "; blocks cannot be added or removed");
        for (size_t b = 0; b < edited.size(); ++b)
            if (edited[b].label != base.blockLabels[b])
                fail(file + ": block " + std::to_string(b) + " is ^" + edited[b].label +
                     ", the IR has ^" + base.blockLabels[b]);

        // Which base node each line keeps, and which nodes go.
        const size_t numNodes = base.nodes.size();
        std::vector<bool> kept(numNodes, false);
        std::vector<bool> seen(numNodes, false);
        std::vector<const EditedLine*> newLines;
        size_t moved = 0;
        {
            long previous = -1;
            for (size_t b = 0; b < edited.size(); ++b)
                for (const EditedLine& line : edited[b].lines) {
                    if (line.id < 0) {
                        newLines.push_back(&line);
                        continue;
                    }
                    const auto id = static_cast<size_t>(line.id);
                    const std::string where = file + ":" + std::to_string(line.lineNumber);
                    if (id >= numNodes) fail(where + ": unknown id " + std::to_string(id));
                    if (seen[id]) fail(where + ": id " + std::to_string(id) + " appears twice");
                    seen[id] = true;
                    if (base.blockOf[id] != b)
                        fail(where + ": id " + std::to_string(id) + " moved from ^" +
                             base.blockLabels[base.blockOf[id]] + " to ^" + edited[b].label +
                             "; moves across blocks are not supported");
                    if (line.text == base.nodeText[id]) {
                        kept[id] = true;
                        if (line.id < previous) ++moved;
                        previous = line.id;
                    } else {
                        newLines.push_back(&line);
                    }
                }
        }

        for (size_t id = 0; id < numNodes; ++id) {
            if (kept[id]) continue;
            if (isControlFlow(base.nodes[id]))
                fail(file + ": id " + std::to_string(id) + " (" + base.nodeText[id] +
                     ") is a branch or label; it cannot be changed or removed");
            if (config.module && config.module->isGroupEndpoint(base.nodes[id]))
                fail(file + ": id " + std::to_string(id) +
                     " bounds an instruction group; it cannot be changed or removed");
        }

        Function scratch("ckpt");
        std::unordered_map<const EditedLine*, IRBase*> created;
        if (!newLines.empty()) {
            const auto nodes = parseNewLines(newLines, config.arch, file, scratch);
            for (size_t i = 0; i < nodes.size(); ++i) {
                nodes[i]->remove();
                created.emplace(newLines[i], nodes[i]);
            }
        }

        std::unordered_map<const IRBase*, size_t> idOf;
        for (size_t id = 0; id < numNodes; ++id) idOf.emplace(base.nodes[id], id);

        size_t deleted = 0;
        size_t blockIndex = 0;
        for (BasicBlock& bb : function) {
            // An unprinted node (a LABEL) travels with the printed node before it, or stays at
            // the top of the block.
            std::unordered_map<const IRBase*, std::vector<IRBase*>> anchored;
            std::vector<IRBase*> dying;
            const IRBase* anchor = nullptr;
            for (IRBase& node : bb) {
                auto id = idOf.find(&node);
                if (id == idOf.end()) {
                    anchored[anchor].push_back(&node);
                } else if (kept[id->second]) {
                    anchor = &node;
                } else {
                    dying.push_back(&node);
                }
            }
            for (IRBase* node : dying) bb.eraseIR(BasicBlock::iterator(node));
            deleted += dying.size();

            std::vector<IRBase*> order;
            const auto appendWithAnchored = [&](IRBase* node) {
                if (node) order.push_back(node);
                auto it = anchored.find(node);
                if (it != anchored.end())
                    order.insert(order.end(), it->second.begin(), it->second.end());
            };
            appendWithAnchored(nullptr);
            for (const EditedLine& line : edited[blockIndex].lines) {
                if (line.id >= 0 && kept[static_cast<size_t>(line.id)])
                    appendWithAnchored(base.nodes[static_cast<size_t>(line.id)]);
                else
                    order.push_back(created.at(&line));
            }

            for (IRBase* node : order)
                if (node->getParent() == &bb) bb.removeIR(node);
            for (IRBase* node : order) bb.appendIR(node);
            ++blockIndex;
        }

        AsmVerifierConfig verify;
        verify.abortOnError = false;
        if (const std::string problem = validateStinkyIR(function, verify); !problem.empty())
            fail(file + ": the edited IR does not verify: " + problem);

        std::cerr << "IRCheckpoint: loaded " << file << " (" << moved << " moved, "
                  << newLines.size() << " new or changed, " << deleted << " removed)\n";
    }

    const IRCheckpointConfig& config;
};

char IRCheckpointPass::ID = 0;

}  // namespace

std::string IRCheckpointSpec::str() const {
    return label + ":" + passName + "#" + std::to_string(occurrence);
}

std::optional<IRCheckpointSpec> parseIRCheckpointSpec(std::string_view text, std::string* error) {
    const auto bad = [&](const std::string& why) -> std::optional<IRCheckpointSpec> {
        if (error) *error = "bad checkpoint spec '" + std::string(text) + "': " + why;
        return std::nullopt;
    };
    const std::string spec = trim(text);
    const size_t colon = spec.find(':');
    if (colon == std::string::npos) return bad("expected <pmLabel>:<PassName>[#n]");
    IRCheckpointSpec out;
    out.label = trim(spec.substr(0, colon));
    std::string rest = trim(spec.substr(colon + 1));
    if (const size_t hash = rest.rfind('#'); hash != std::string::npos) {
        const std::string count = trim(rest.substr(hash + 1));
        unsigned n = 0;
        const auto [ptr, ec] = std::from_chars(count.data(), count.data() + count.size(), n);
        if (ec != std::errc{} || ptr != count.data() + count.size() || n == 0)
            return bad("the occurrence after '#' must be a positive integer");
        out.occurrence = n;
        rest = trim(rest.substr(0, hash));
    }
    out.passName = rest;
    if (out.label.empty()) return bad("empty pass manager label");
    if (out.passName.empty()) return bad("empty pass name");
    return out;
}

IRCheckpointRendering renderCheckpointRegion(const Function& function, bool withIds) {
    IRCheckpointRendering out;
    std::ostringstream os;
    AsmPrinterOptions nodeOptions;
    nodeOptions.indent = 0;

    os << "st.func @" << function.getName() << "() {\n";
    size_t blockIndex = 0;
    for (const BasicBlock& bb : function) {
        std::string blockId =
            bb.getLabel().empty() ? ("bb" + std::to_string(blockIndex)) : bb.getLabel();
        os << "  ^" << blockId << ":\n";
        out.blockLabels.push_back(blockId);
        for (const IRBase& node : bb) {
            std::ostringstream lineStream;
            AsmPrinter printer(lineStream, nodeOptions);
            printer.printIR(node);
            std::string line = lineStream.str();
            while (!line.empty() && line.back() == '\n') line.pop_back();
            if (line.empty()) continue;
            const size_t id = out.nodes.size();
            out.nodes.push_back(const_cast<IRBase*>(&node));
            out.blockOf.push_back(blockIndex);
            out.nodeText.push_back(trim(line));
            os << "    " << (withIds ? withCheckpointId(line, id) : line) << "\n";
        }
        const auto& successors = bb.getSuccessors();
        if (!successors.empty()) {
            os << "    Successors: ";
            for (size_t i = 0; i < successors.size(); ++i) {
                if (i > 0) os << ", ";
                os << "^" << successors[i]->getLabel();
            }
            os << "\n";
        }
        ++blockIndex;
    }
    os << "}\n";
    out.text = os.str();
    return out;
}

long takeCheckpointId(std::string& line) {
    // The directive form owns its whole attribute dict; the instruction form is one entry of it.
    static const std::regex ownDict(R"(\s*\{\s*mod\.ckpt\s*=\s*\{\s*id\s*=\s*(\d+)\s*\}\s*\})");
    static const std::regex lastEntry(R"(\s*,\s*mod\.ckpt\s*=\s*\{\s*id\s*=\s*(\d+)\s*\})");
    static const std::regex firstEntry(R"(mod\.ckpt\s*=\s*\{\s*id\s*=\s*(\d+)\s*\}\s*,\s*)");
    for (const std::regex* pattern : {&ownDict, &lastEntry, &firstEntry}) {
        std::smatch match;
        if (!std::regex_search(line, match, *pattern)) continue;
        const long id = std::stol(match[1].str());
        line = match.prefix().str() + match.suffix().str();
        return id;
    }
    return -1;
}

std::string stripCheckpointIds(std::string_view text) {
    std::string out;
    std::istringstream in{std::string(text)};
    for (std::string line; std::getline(in, line);) {
        if (line.find("mod.ckpt") != std::string::npos) takeCheckpointId(line);
        out += line;
        out += "\n";
    }
    return out;
}

std::string sha256Hex(std::string_view data) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::string message(data);
    const uint64_t bitLength = static_cast<uint64_t>(data.size()) * 8;
    message.push_back(static_cast<char>(0x80));
    while (message.size() % 64 != 56) message.push_back('\0');
    for (int i = 7; i >= 0; --i)
        message.push_back(static_cast<char>((bitLength >> (i * 8)) & 0xff));

    const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (size_t offset = 0; offset < message.size(); offset += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            const auto byte = [&](int j) {
                return static_cast<uint32_t>(static_cast<uint8_t>(message[offset + 4 * i + j]));
            };
            w[i] = (byte(0) << 24) | (byte(1) << 16) | (byte(2) << 8) | byte(3);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    static constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    for (uint32_t word : h)
        for (int shift = 28; shift >= 0; shift -= 4) hex.push_back(digits[(word >> shift) & 0xf]);
    return hex;
}

std::shared_ptr<IRCheckpointSession> IRCheckpointSession::create(const std::string& spec,
                                                                 const std::string& dumpDir,
                                                                 const std::string& loadDir,
                                                                 const std::string& kernelName,
                                                                 const std::array<int, 3>& arch) {
    if (spec.empty()) {
        if (!dumpDir.empty() || !loadDir.empty())
            fail("a dump or load directory needs IRCheckpointAfterPass (--checkpoint-after)");
        return nullptr;
    }
    if (!dumpDir.empty() && !loadDir.empty())
        fail("IRCheckpointDumpDir and IRCheckpointLoadDir are mutually exclusive");
    if (dumpDir.empty() && loadDir.empty()) {
        warn("IRCheckpointAfterPass '" + spec + "' has neither a dump nor a load directory");
        return nullptr;
    }
    std::string problem;
    auto parsed = parseIRCheckpointSpec(spec, &problem);
    if (!parsed) fail(problem);

    IRCheckpointConfig config;
    config.spec = *parsed;
    config.dumpDir = dumpDir;
    config.loadDir = loadDir;
    config.kernelName = kernelName;
    config.arch = arch;
    return std::make_shared<IRCheckpointSession>(std::move(config));
}

std::shared_ptr<IRCheckpointSession> IRCheckpointSession::create(const StinkyAsmModule& module) {
    const auto& options = module.getModuleOptions();
    const std::string kernel =
        module.getOutputName().empty() ? module.getName() : module.getOutputName();
    auto session = create(options.IRCheckpointAfterPass, options.IRCheckpointDumpDir,
                          options.IRCheckpointLoadDir, kernel, module.getArch());
    if (session) session->config.module = &module;
    return session;
}

IRCheckpointSession::IRCheckpointSession(IRCheckpointConfig config)
    : config(std::move(config)), checkpointPass(std::make_unique<IRCheckpointPass>(this->config)) {}

IRCheckpointSession::~IRCheckpointSession() {
    const bool loading = !config.loadDir.empty();
    const auto report = [&](const std::string& message) {
        if (loading) fail(message);
        warn(message);
    };
    const std::string spec = config.spec.str();
    if (!installed) {
        std::string labels;
        for (const std::string& label : labelsOffered) labels += " " + label;
        report("no pass manager is labelled '" + config.spec.label + "' (checkpoint " + spec +
               ", kernel " + config.kernelName + "); the labels are:" + labels);
        return;
    }
    if (!ran) {
        warn("the pass manager '" + config.spec.label + "' did not run for kernel " +
             config.kernelName + " (no such region); checkpoint " + spec + " skipped");
        return;
    }
    if (!fired) {
        std::string names;
        for (const std::string& name : passesSeen) names += "\n    " + name;
        report("pass '" + config.spec.passName + "' ran " + std::to_string(matches) +
               " time(s) in '" + config.spec.label + "', fewer than #" +
               std::to_string(config.spec.occurrence) + " (kernel " + config.kernelName +
               "). The passes that ran:" + names);
    }
}

void IRCheckpointSession::install(PassManager& pm, const std::string& label) {
    labelsOffered.push_back(label);
    if (label != config.spec.label) return;
    installed = true;
    pm.addInsertionPoint(shared_from_this());
}

Pass* IRCheckpointSession::passAfter(const std::string& passName, Function& /*function*/) {
    ran = true;
    if (std::find(passesSeen.begin(), passesSeen.end(), passName) == passesSeen.end())
        passesSeen.push_back(passName);
    if (fired || passName != config.spec.passName) return nullptr;
    if (++matches != config.spec.occurrence) return nullptr;
    fired = true;
    return checkpointPass.get();
}

}  // namespace stinkytofu
