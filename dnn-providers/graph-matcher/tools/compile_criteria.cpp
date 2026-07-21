// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Build-time (AOT) criteria compiler. Reads a JSON criteria file, compiles it
// with the same PatternCompiler the runtime uses, serializes the CompiledPattern
// to the flat wire form, and writes a C++ source file embedding the bytes as a
// static array. Embedding the compiled bytes (not the JSON) means startup does no
// JSON parse and no re-run of the semantic compiler -- it only deserializes.
//
//   compile_criteria <in.json> <out.inc> <symbol>
//
// The generated <out.inc> defines `static constexpr unsigned char <symbol>[]`
// and `<symbol>_size`; adopt at startup via PatternCodec::deserialize.

#include <cstdio>
#include <fstream>
#include <hipdnn_graph_matcher/PatternCodec.hpp>
#include <hipdnn_graph_matcher/PatternCompiler.hpp>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s <in.json> <out.inc> <symbol>\n", argv[0]);
        return 2;
    }
    const std::string inPath{argv[1]};
    const std::string outPath{argv[2]};
    const std::string symbol{argv[3]};

    std::ifstream in(inPath, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "error: cannot open %s\n", inPath.c_str());
        return 1;
    }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string json = ss.str();

    const auto result = hipdnn::graph_matcher::PatternCompiler::fromJson(json);
    if (!result.ok) {
        std::fprintf(stderr, "error: %s: %s\n", inPath.c_str(), result.error.c_str());
        return 1;
    }

    const auto bytes = hipdnn::graph_matcher::PatternCodec::serialize(result.pattern);
    const auto code = hipdnn::graph_matcher::PatternCodec::emitEmbeddedArray(symbol, bytes);

    std::ofstream out(outPath, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "error: cannot write %s\n", outPath.c_str());
        return 1;
    }
    out << code;
    return 0;
}
