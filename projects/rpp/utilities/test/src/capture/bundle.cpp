/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "capture/bundle.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace rpptest::capture {

namespace fs = std::filesystem;

namespace {

std::vector<Comparison>& pending() {
    static std::vector<Comparison> comparisons;
    return comparisons;
}

const char* capture_dir() {
    return std::getenv("RPP_TEST_CAPTURE_DIR");
}

// NPY format v1.0: magic, header length, a Python-literal dict header padded to 64 bytes.
void write_npy(const fs::path& path, const Array& a) {
    std::ostringstream shape;
    shape << "(";
    for (std::size_t e : a.shape) shape << e << ",";
    shape << ")";
    std::string header =
        "{'descr': '" + a.descr + "', 'fortran_order': False, 'shape': " + shape.str() + ", }";
    const std::size_t preamble = 10;
    header.append(63 - (preamble + header.size()) % 64, ' ');
    header += '\n';

    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    const auto len = static_cast<std::uint16_t>(header.size());
    const unsigned char lenBytes[2] = {static_cast<unsigned char>(len & 0xff),
                                       static_cast<unsigned char>(len >> 8)};
    out.write(reinterpret_cast<const char*>(lenBytes), 2);
    out << header;
    out.write(reinterpret_cast<const char*>(a.bytes.data()),
              static_cast<std::streamsize>(a.bytes.size()));
}

std::string json_string(const std::string& s) {
    std::ostringstream out;
    out << '"';
    for (const char ch : s) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", ch);
                    out << buf;
                } else {
                    out << ch;
                }
        }
    }
    out << '"';
    return out.str();
}

std::string failure_text(const ::testing::TestResult& result) {
    std::string text;
    for (int i = 0; i < result.total_part_count(); ++i) {
        const ::testing::TestPartResult& part = result.GetTestPartResult(i);
        if (part.failed()) text += part.summary() + std::string("\n");
    }
    return text;
}

void write_bundle(const fs::path& dir, const std::string& name, const ::testing::TestResult& result,
                  const std::vector<Comparison>& comparisons) {
    fs::remove_all(dir);  // a rerun may record fewer comparisons than the last one
    fs::create_directories(dir);

    std::ofstream manifest(dir / "manifest.json");
    manifest << "{\n  \"format\": \"rpp-test-capture\",\n  \"version\": 1,\n"
             << "  \"case\": " << json_string(name) << ",\n"
             << "  \"passed\": " << (result.Failed() ? "false" : "true") << ",\n"
             << "  \"failure\": " << json_string(failure_text(result)) << ",\n"
             << "  \"comparisons\": [";

    for (std::size_t k = 0; k < comparisons.size(); ++k) {
        const Comparison& c = comparisons[k];
        const std::string prefix = std::to_string(k) + "_";
        const std::pair<const char*, const Array*> arrays[] = {{"reference", &c.reference},
                                                               {"actual", &c.actual},
                                                               {"diff", &c.diff},
                                                               {"status", &c.status}};

        manifest << (k ? ",\n" : "\n") << "    {\"kind\": " << json_string(c.kind)
                 << ", \"axes\": [";
        for (std::size_t a = 0; a < c.axes.size(); ++a)
            manifest << (a ? ", " : "") << json_string(c.axes[a]);
        manifest << "], \"compared\": " << c.compared << ", \"failed\": " << c.failed
                 << ", \"arrays\": {";
        for (std::size_t a = 0; a < std::size(arrays); ++a) {
            const std::string file = prefix + arrays[a].first + ".npy";
            write_npy(dir / file, *arrays[a].second);
            manifest << (a ? ", " : "") << json_string(arrays[a].first) << ": "
                     << json_string(file);
        }
        manifest << "}}";
    }
    manifest << "\n  ]\n}\n";
}

class CaptureListener : public ::testing::EmptyTestEventListener {
   public:
    CaptureListener(fs::path root, bool all) : root_(std::move(root)), all_(all) {}

    void OnTestStart(const ::testing::TestInfo&) override {
        pending().clear();
    }

    void OnTestEnd(const ::testing::TestInfo& info) override {
        const ::testing::TestResult& result = *info.result();
        if (!pending().empty() && (all_ || result.Failed())) {
            // "Suite/Param.Test/Case" -> Suite/Param.Test/Case/ (the '/'s become directories).
            const std::string name = std::string(info.test_suite_name()) + "." + info.name();
            write_bundle(root_ / name, name, result, pending());
        }
        pending().clear();
    }

   private:
    fs::path root_;
    bool all_;
};

}  // namespace

bool enabled() {
    return capture_dir() != nullptr;
}

void submit(Comparison&& comparison) {
    pending().push_back(std::move(comparison));
}

void install() {
    if (!enabled()) {
        std::fprintf(stderr,
                     "note: RPP_TEST_CAPTURE_DIR is not set; rpp_tests_capture will not capture\n");
        return;
    }
    const bool all = std::getenv("RPP_TEST_CAPTURE_ALL") != nullptr;
    ::testing::UnitTest::GetInstance()->listeners().Append(
        new CaptureListener(fs::absolute(capture_dir()), all));
    std::fprintf(stderr, "note: capturing %s cases to %s\n", all ? "all" : "failing",
                 capture_dir());
}

}  // namespace rpptest::capture
