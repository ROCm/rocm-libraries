// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/*
 * Native lean path, no Python and no LLVM:
 *
 *   C++ instance spec -> C++ KernelDef -> C++ ck.dsl.ir/v1 serializer
 *     -> Rust direct compiler (code_object_ffi.h) -> HSACO
 *
 * for gfx1151 WMMA forward attention (head 64/128, unmasked/causal),
 * implicit-GEMM forward convolution and WMMA GEMM. Attention and convolution
 * go through rocke_codeobj_compile_rocke_ir_v1; the GEMM goes through
 * rocke_codeobj_compile_gfx1151_wmma_gemm_v1, and the generic entry point's
 * verdict on the same IR is recorded alongside.
 *
 * Each case runs once untimed (checked and optionally dumped), then
 * --warmup - 1 more untimed runs, then --reps timed runs; the median wall time
 * of every stage is reported, with the compiler's own phase split taken from
 * its diagnostics. A case the compiler rejects still reports its build and
 * serialize medians.
 *
 * Checks (non-zero exit on failure): every spec is valid, every compile not
 * named by --expect-reject succeeds, every image is an ELF holding the kernel
 * symbol, and repeated compiles of a case give identical IR and images. A case
 * named by --expect-reject must be rejected; if it compiles, that is reported
 * as a failure so the expectation gets removed.
 *
 * usage: rocke_native_direct_code_object [--reps N] [--warmup N] [--only SUBSTR]
 *            [--expect-reject SUBSTR[,SUBSTR...]] [--dump DIR] [--json FILE]
 *            [--cpp-parse] [--modes plain,session,stream] [--ir-case NAME=FILE]
 *            [--session-threads N] [--feed-bytes N]
 *   --dump writes <case>.cpp.ir (always) and <case>.cpp.hsaco into DIR.
 *   --cpp-parse also times the C++ parser (rocke_ir_parse) on the same text,
 *   after the timed stages, as a reference point for the cost of a text handoff.
 *   --modes runs each case through these paths, interleaved rep by rep:
 *     plain    rocke_ir_serialize, then rocke_codeobj_compile_rocke_ir_v1;
 *     session  rocke_ir_serialize, then rocke_codeobj_session_compile_v1;
 *     stream   a streaming serializer that feeds the session every
 *              --feed-bytes bytes (default 16384) while it writes, then
 *              rocke_codeobj_session_finish_v1.
 *   Every mode must give the plain mode's image, and the streamed pieces must
 *   join to rocke_ir_serialize's text. The session has --session-threads
 *   workers (0, the default, lets the library choose).
 *   --ir-case adds a case built by parsing FILE with rocke_ir_parse once,
 *   outside the timing (kernels without a C++ builder, such as attention
 *   backward); its build stage is 0.
 */

#include "rocke/code_object_ffi.h"
#include "rocke/helper_rocke.instances.common.conv_implicit_gemm.h"
#include "rocke/instance_conv_implicit_gemm.h"
#include "rocke/instance_gfx1151_wmma_fmha_fwd.h"
#include "rocke/instance_gfx1151_wmma_gemm.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;

constexpr const char* kArch = "gfx1151";

double ms_between(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double median(std::vector<double> values)
{
    if(values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if(values.size() % 2 == 0)
        return (values[middle - 1] + values[middle]) / 2.0;
    return values[middle];
}

/* Builds one kernel into an uninitialised builder; returns the kernel or NULL
 * with the builder's sticky error set. The builder is always initialised on
 * return, so the caller frees it either way. */
using BuildFn = std::function<rocke_kernel_def_t*(rocke_ir_builder_t*)>;

enum class Entry
{
    RockeIr,
    WmmaGemm,
};

struct Case
{
    std::string name;
    std::string family;
    Entry entry = Entry::RockeIr;
    BuildFn build;
    /* For --ir-case: the kernel, parsed once into `parsed_builder`. */
    std::shared_ptr<rocke_ir_builder_t> parsed_builder;
    rocke_kernel_def_t* parsed = nullptr;
};

enum class Mode
{
    Plain,
    Session,
    Stream,
};

const char* mode_name(Mode m)
{
    return m == Mode::Plain ? "plain" : m == Mode::Session ? "session" : "stream";
}

/* ------------------------------------------------------------------------
 * Streaming serializer: rocke_ir_serialize's output (core/ir/serialize.cpp,
 * same algorithms, line for line), handed to a sink every `flush_bytes` bytes
 * at a line end while it is being written. The harness checks that the pieces
 * join to rocke_ir_serialize's text for every case.
 * ------------------------------------------------------------------------ */
class StreamSerializer
{
public:
    StreamSerializer(size_t flush_bytes, std::function<void(const char*, size_t)> sink)
        : flush_bytes_(flush_bytes)
        , sink_(std::move(sink))
    {
        out_.reserve(flush_bytes * 2 + 4096);
    }

    void serialize(const rocke_kernel_def_t* k)
    {
        out_ += "ckdsl.ir v1\n";
        out_ += "kernel @";
        out_ += k->name ? k->name : "";
        out_ += " {\n";
        if(k->attrs.count > 0)
        {
            indent(1);
            out_ += "attrs ";
            attr_map(&k->attrs);
            out_ += '\n';
        }
        indent(1);
        out_ += "params {\n";
        for(int i = 0; i < k->num_params; ++i)
        {
            const rocke_param_t* p = k->params[i];
            indent(2);
            out_ += '%';
            out_ += (p && p->name) ? p->name : "";
            out_ += " : ";
            const rocke_type_t* t = p ? p->type : nullptr;
            out_ += (t && t->name) ? t->name : "";
            if(p && p->attrs.count > 0)
            {
                out_ += ' ';
                attr_map(&p->attrs);
            }
            out_ += '\n';
        }
        indent(1);
        out_ += "}\n";
        region(k->body, 1);
        out_ += "}\n";
        flush();
    }

private:
    void maybe_flush()
    {
        if(out_.size() >= flush_bytes_)
            flush();
    }

    void flush()
    {
        if(!out_.empty())
            sink_(out_.data(), out_.size());
        out_.clear();
    }

    void indent(int depth)
    {
        for(int i = 0; i < depth; ++i)
            out_ += "  ";
    }

    void escape(const char* s)
    {
        if(!s)
            return;
        for(const char* p = s; *p; ++p)
        {
            const char ch = *p;
            if(ch == '\\')
                out_ += "\\\\";
            else if(ch == '"')
                out_ += "\\\"";
            else if(ch == '\n')
                out_ += "\\n";
            else if(ch == '\t')
                out_ += "\\t";
            else
                out_ += ch;
        }
    }

    void float_repr(double f)
    {
        if(f != f)
        {
            out_ += "nan";
            return;
        }
        char sci[64];
        int p;
        for(p = 0; p <= 17; ++p)
        {
            std::snprintf(sci, sizeof(sci), "%.*e", p, f);
            if(std::strtod(sci, nullptr) == f)
                break;
        }
        if(p > 17)
        {
            p = 17;
            std::snprintf(sci, sizeof(sci), "%.*e", p, f);
        }
        const char* s = sci;
        bool negative = false;
        if(*s == '-')
        {
            negative = true;
            ++s;
        }
        if(s[0] == 'i' || s[0] == 'I')
        {
            if(negative)
                out_ += '-';
            out_ += "inf";
            return;
        }
        char digits[40];
        int nd = 0;
        digits[nd++] = *s++;
        if(*s == '.')
        {
            ++s;
            while(*s && *s != 'e' && *s != 'E')
                digits[nd++] = *s++;
        }
        digits[nd] = '\0';
        int exp = 0;
        if(*s == 'e' || *s == 'E')
        {
            ++s;
            exp = std::atoi(s);
        }
        while(nd > 1 && digits[nd - 1] == '0')
            digits[--nd] = '\0';
        const int decpt = exp + 1;
        if(negative)
            out_ += '-';
        if(decpt <= -4 || decpt > 16)
        {
            out_ += digits[0];
            if(nd > 1)
            {
                out_ += '.';
                out_ += digits + 1;
            }
            const int e = decpt - 1;
            out_ += 'e';
            out_ += e < 0 ? '-' : '+';
            int ae = e < 0 ? -e : e;
            char eb[8];
            int en = 0;
            do
            {
                eb[en++] = static_cast<char>('0' + ae % 10);
                ae /= 10;
            } while(ae);
            while(en < 2)
                eb[en++] = '0';
            while(en > 0)
                out_ += eb[--en];
        }
        else if(decpt <= 0)
        {
            out_ += "0.";
            for(int i = 0; i < -decpt; ++i)
                out_ += '0';
            out_ += digits;
        }
        else if(decpt >= nd)
        {
            out_ += digits;
            for(int i = 0; i < decpt - nd; ++i)
                out_ += '0';
            out_ += ".0";
        }
        else
        {
            out_.append(digits, static_cast<size_t>(decpt));
            out_ += '.';
            out_ += digits + decpt;
        }
    }

    void attr_value(const rocke_attr_value_t* v)
    {
        char num[32];
        switch(v->kind)
        {
        case ROCKE_ATTR_BOOL:
            out_ += v->u.b ? "b:true" : "b:false";
            break;
        case ROCKE_ATTR_INT:
            std::snprintf(num, sizeof(num), "i:%lld", static_cast<long long>(v->u.i));
            out_ += num;
            break;
        case ROCKE_ATTR_FLOAT:
            out_ += "f:";
            float_repr(v->u.f);
            break;
        case ROCKE_ATTR_STR:
            out_ += "s:\"";
            escape(v->u.s);
            out_ += '"';
            break;
        case ROCKE_ATTR_LIST:
            out_ += "l:[ ";
            for(int i = 0; i < v->u.list.count; ++i)
            {
                if(i > 0)
                    out_ += ", ";
                attr_map(v->u.list.items[i]);
            }
            out_ += " ]";
            break;
        case ROCKE_ATTR_INT_LIST:
            out_ += "l:[ ";
            for(int i = 0; i < v->u.ilist.count; ++i)
            {
                if(i > 0)
                    out_ += ", ";
                std::snprintf(
                    num, sizeof(num), "i:%lld", static_cast<long long>(v->u.ilist.ints[i]));
                out_ += num;
            }
            out_ += " ]";
            break;
        default:
            break;
        }
    }

    void attr_map(const rocke_attr_map_t* m)
    {
        const int count = m ? m->count : 0;
        if(count <= 0)
        {
            out_ += "{ }";
            return;
        }
        /* A fresh array per map, as nested maps (iter_args) sort their own. */
        std::vector<const rocke_attr_entry_t*> order;
        order.reserve(static_cast<size_t>(count));
        for(int i = 0; i < count; ++i)
            order.push_back(&m->entries[i]);
        std::qsort(
            order.data(), order.size(), sizeof(order[0]), [](const void* pa, const void* pb) {
                const auto* a = *static_cast<const rocke_attr_entry_t* const*>(pa);
                const auto* b = *static_cast<const rocke_attr_entry_t* const*>(pb);
                return std::strcmp(a->key ? a->key : "", b->key ? b->key : "");
            });
        out_ += "{ ";
        for(int i = 0; i < count; ++i)
        {
            if(i > 0)
                out_ += ", ";
            out_ += order[i]->key ? order[i]->key : "";
            out_ += " = ";
            attr_value(&order[i]->value);
        }
        out_ += " }";
    }

    void op(const rocke_op_t* op, int depth)
    {
        indent(depth);
        if(op->num_results > 0)
        {
            for(int i = 0; i < op->num_results; ++i)
            {
                if(i > 0)
                    out_ += ", ";
                const rocke_value_t* r = op->results[i];
                out_ += (r && r->name) ? r->name : "";
                out_ += " : ";
                const rocke_type_t* t = r ? r->type : nullptr;
                out_ += (t && t->name) ? t->name : "";
            }
            out_ += " = ";
        }
        out_ += op->name ? op->name : "";
        out_ += " ( ";
        for(int i = 0; i < op->num_operands; ++i)
        {
            if(i > 0)
                out_ += ", ";
            const rocke_value_t* o = op->operands[i];
            out_ += (o && o->name) ? o->name : "";
        }
        out_ += " )";
        if(op->attrs.count > 0)
        {
            out_ += ' ';
            attr_map(&op->attrs);
        }
        if(op->loc != nullptr)
        {
            out_ += " @loc \"";
            escape(op->loc);
            out_ += '"';
        }
        out_ += '\n';
        maybe_flush();
        for(int r = 0; r < op->num_regions; ++r)
            region(op->regions[r], depth + 1);
    }

    void region(const rocke_region_t* region, int depth)
    {
        indent(depth);
        out_ += "region @";
        out_ += (region && region->label) ? region->label : "";
        out_ += " {\n";
        if(region)
        {
            for(int i = 0; i < region->num_ops; ++i)
                op(region->ops[i], depth + 1);
        }
        indent(depth);
        out_ += "}\n";
        maybe_flush();
    }

    size_t flush_bytes_;
    std::function<void(const char*, size_t)> sink_;
    std::string out_;
};

/* Settings shared by every case. */
struct Settings
{
    std::vector<Mode> modes = {Mode::Plain};
    size_t feed_bytes = 16384;
    rocke_codeobj_session_v1* session = nullptr;
};

Settings g_settings;

Case fwd_case(int head_size, bool causal)
{
    rocke_wmma_fmha_fwd_spec_t spec = rocke_wmma_fmha_fwd_spec_default();
    spec.head_size = head_size;
    spec.mask_mode = causal ? ROCKE_FMHA_MASK_CAUSAL : ROCKE_FMHA_MASK_NONE;
    char reason[512] = {};
    if(!rocke_wmma_fmha_fwd_is_valid_spec(&spec, kArch, reason, sizeof(reason)))
        throw std::runtime_error(std::string("invalid fwd attention spec: ") + reason);
    Case c;
    c.name = "fwd_d" + std::to_string(head_size) + (causal ? "_causal" : "_none");
    c.family = "fwd";
    c.build = [spec](rocke_ir_builder_t* b) {
        char symbol[512] = {};
        rocke_wmma_fmha_fwd_kernel_name(&spec, symbol, sizeof(symbol));
        rocke_ir_builder_init(b, symbol);
        return rocke_build_wmma_fmha_fwd(b, &spec, kArch);
    };
    return c;
}

struct ConvShape
{
    int n, hi, wi, c, k, y, x, sh, sw, ph, pw;
};

struct ConvTile
{
    int tile_m, tile_n, tile_k, warp_m, warp_n;
};

Case conv_case(const std::string& name, const ConvShape& p, const ConvTile& t)
{
    rocke_implicit_gemm_conv_spec_t spec = rocke_implicit_gemm_conv_spec_default();
    spec.problem = rocke_conv_problem_make(
        p.n, p.hi, p.wi, p.c, p.k, p.y, p.x, p.sh, p.sw, p.ph, p.pw, 1, 1);
    spec.tile_m = t.tile_m;
    spec.tile_n = t.tile_n;
    spec.tile_k = t.tile_k;
    spec.warp_m = t.warp_m;
    spec.warp_n = t.warp_n;
    spec.warp_tile_m = 16;
    spec.warp_tile_n = 16;
    spec.warp_tile_k = 16;
    spec.wave_size = 32;
    spec.pipeline = "mem";
    spec.epilogue = "cshuffle";
    spec.dtype_a = "fp16";
    spec.dtype_b = "fp16";
    spec.dtype_d = "fp16";
    char reason[512] = {};
    if(!rocke_implicit_gemm_conv_is_valid_spec(&spec, kArch, reason, sizeof(reason)))
        throw std::runtime_error("invalid conv spec " + name + ": " + reason);
    Case c;
    c.name = name;
    c.family = "conv";
    c.build = [spec](rocke_ir_builder_t* b) {
        return rocke_build_implicit_gemm_conv_new(b, &spec, kArch, nullptr);
    };
    return c;
}

Case gemm_case()
{
    rocke_wmma_gemm_gfx1151_spec_t spec = rocke_wmma_gemm_gfx1151_spec_default();
    spec.name = "rocke_direct_wmma_gemm";
    spec.block_x_is_m = true;
    char reason[512] = {};
    if(!rocke_wmma_gemm_gfx1151_is_valid_spec(&spec, kArch, reason, sizeof(reason)))
        throw std::runtime_error(std::string("invalid WMMA GEMM spec: ") + reason);
    Case c;
    c.name = "gemm_f16_rcr_xm";
    c.family = "gemm";
    c.entry = Entry::WmmaGemm;
    c.build = [spec](rocke_ir_builder_t* b) {
        return rocke_build_wmma_gemm_gfx1151_new(b, &spec, kArch);
    };
    return c;
}

/* The case list mirrors the Python benchmarks so both paths time the same
 * kernels: forward attention as in compile_times/bench_ab/diff_fwd, the
 * compile_times/bench_ab conv shapes at both tiles, the diff_conv problems at
 * tile 32, and the WMMA GEMM of test_native_gemm_code_object. */
std::vector<Case> all_cases()
{
    std::vector<Case> cases;
    for(int d : {64, 128})
        for(bool causal : {false, true})
            cases.push_back(fwd_case(d, causal));
    const ConvShape bench[] = {{8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1},
                               {8, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1},
                               {8, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1},
                               {8, 56, 56, 64, 256, 1, 1, 1, 1, 0, 0}};
    const ConvTile tiles[] = {{32, 32, 32, 1, 1}, {64, 64, 32, 2, 2}};
    for(const ConvShape& p : bench)
        for(const ConvTile& t : tiles)
        {
            std::ostringstream name;
            name << "conv_" << p.hi << "x" << p.wi << "_c" << p.c << "_k" << p.k << "_" << p.y
                 << "x" << p.x << "_t" << t.tile_m;
            cases.push_back(conv_case(name.str(), p, t));
        }
    const ConvShape diff[] = {{2, 14, 14, 32, 32, 3, 3, 1, 1, 1, 1},
                              {1, 16, 16, 64, 64, 3, 3, 1, 1, 1, 1},
                              {2, 9, 7, 32, 64, 1, 1, 1, 1, 0, 0},
                              {1, 15, 15, 32, 32, 3, 3, 2, 2, 1, 1},
                              {3, 10, 12, 96, 48, 3, 3, 1, 1, 1, 1}};
    for(const ConvShape& p : diff)
    {
        std::ostringstream name;
        name << "conv_n" << p.n << "_" << p.hi << "x" << p.wi << "_c" << p.c << "_k" << p.k << "_"
             << p.y << "x" << p.x << "_s" << p.sh << "_t32";
        cases.push_back(conv_case(name.str(), p, tiles[0]));
    }
    cases.push_back(gemm_case());
    return cases;
}

struct Sample
{
    double build_ms = 0.0;
    double serialize_ms = 0.0;
    double compile_ms = 0.0; /* wall time around the FFI call */
    double compiler_ms = 0.0; /* the library's own executable_seconds */
    double teardown_ms = 0.0; /* dispose + free(text) + builder free */
    double scan_ms = 0.0; /* one read pass over the IR text */
    double cpp_parse_ms = 0.0; /* --cpp-parse: rocke_ir_parse of the same text */
    std::map<std::string, double> phases;
};

bool g_cpp_parse = false;

struct Outcome
{
    std::string symbol;
    std::string ir;
    std::vector<uint8_t> hsaco;
    std::string stats;
    std::string compile_error; /* empty when the compile succeeded */
    Sample sample;
};

std::string bytes_to_string(const uint8_t* data, size_t size)
{
    return data && size ? std::string(reinterpret_cast<const char*>(data), size) : std::string();
}

/* "k=v k=v phases_ms=a:1.0,b:2.0" -> phases, and the summary without phases. */
std::string split_diagnostics(const std::string& text, std::map<std::string, double>& phases)
{
    const std::string key = "phases_ms=";
    const size_t at = text.find(key);
    if(at == std::string::npos)
        return text;
    std::stringstream list(text.substr(at + key.size()));
    std::string item;
    while(std::getline(list, item, ','))
    {
        const size_t colon = item.find(':');
        if(colon != std::string::npos)
            phases[item.substr(0, colon)] = std::strtod(item.c_str() + colon + 1, nullptr);
    }
    std::string summary = text.substr(0, at);
    while(!summary.empty() && summary.back() == ' ')
        summary.pop_back();
    return summary;
}

size_t count_newlines(const char* text, size_t size)
{
    size_t lines = 0;
    for(const char* p = text;
        (p = static_cast<const char*>(std::memchr(p, '\n', size - (p - text))));
        ++p)
        ++lines;
    return lines;
}

int32_t compile_ir(Entry entry, const std::string& ir, rocke_codeobj_compile_result_v1* result)
{
    rocke_codeobj_compile_request_v1 request = {};
    request.struct_size = sizeof(request);
    request.serialized_ir = reinterpret_cast<const uint8_t*>(ir.data());
    request.serialized_ir_size = ir.size();
    *result = {};
    result->struct_size = sizeof(*result);
    return entry == Entry::WmmaGemm ? rocke_codeobj_compile_gfx1151_wmma_gemm_v1(&request, result)
                                    : rocke_codeobj_compile_rocke_ir_v1(&request, result);
}

/* One pass of case `c` through `mode`. With `keep_ir`, the serialized text is
 * kept in the outcome (the streamed pieces are joined outside the timing of
 * the producer only then). */
Outcome run_once(const Case& c, Mode mode = Mode::Plain, bool keep_ir = true)
{
    Outcome out;
    rocke_ir_builder_t builder;
    const bool prebuilt = c.parsed != nullptr;
    const auto t0 = Clock::now();
    rocke_kernel_def_t* kernel = prebuilt ? c.parsed : c.build(&builder);
    const auto t1 = Clock::now();
    if(!prebuilt && (!kernel || !rocke_ir_builder_ok(&builder)))
    {
        const char* message = rocke_ir_builder_error(&builder);
        const std::string detail = message ? message : "unknown builder failure";
        rocke_ir_builder_free(&builder);
        throw std::runtime_error(c.name + ": C++ builder failed: " + detail);
    }
    char* text = nullptr;
    size_t text_size = 0;
    std::string streamed;
    rocke_codeobj_compile_result_v1 result = {};
    result.struct_size = sizeof(result);
    Clock::time_point t2, t3, t4;
    if(mode == Mode::Stream)
    {
        bool fed = true;
        StreamSerializer producer(g_settings.feed_bytes, [&](const char* data, size_t size) {
            fed = fed
                  && rocke_codeobj_session_feed_v1(
                         g_settings.session, reinterpret_cast<const uint8_t*>(data), size)
                         == 0;
            if(keep_ir)
                streamed.append(data, size);
        });
        producer.serialize(kernel);
        t2 = Clock::now();
        if(!fed)
            throw std::runtime_error(c.name + ": rocke_codeobj_session_feed_v1 refused a piece");
        t3 = Clock::now();
        rocke_codeobj_session_finish_v1(g_settings.session, &result);
        t4 = Clock::now();
        if(keep_ir)
        {
            char* reference = nullptr;
            if(rocke_ir_serialize(kernel, &reference) != ROCKE_OK || !reference
               || streamed != reference)
            {
                std::free(reference);
                throw std::runtime_error(c.name
                                         + ": the streamed pieces differ from rocke_ir_serialize");
            }
            std::free(reference);
        }
    }
    else
    {
        const rocke_status_t serialized = rocke_ir_serialize(kernel, &text);
        t2 = Clock::now();
        if(serialized != ROCKE_OK || !text)
        {
            std::free(text);
            if(!prebuilt)
                rocke_ir_builder_free(&builder);
            throw std::runtime_error(c.name + ": C++ IR serialization failed");
        }
        text_size = std::strlen(text);
        rocke_codeobj_compile_request_v1 request = {};
        request.struct_size = sizeof(request);
        request.serialized_ir = reinterpret_cast<const uint8_t*>(text);
        request.serialized_ir_size = text_size;
        t3 = Clock::now();
        const int32_t status
            = c.entry == Entry::WmmaGemm
                  ? rocke_codeobj_compile_gfx1151_wmma_gemm_v1(&request, &result)
              : mode == Mode::Session
                  ? rocke_codeobj_session_compile_v1(g_settings.session, &request, &result)
                  : rocke_codeobj_compile_rocke_ir_v1(&request, &result);
        t4 = Clock::now();
        if(status != 0 && result.status == 0)
            result.status = status;
    }
    const std::string diagnostics = bytes_to_string(result.diagnostics, result.diagnostics_size);
    out.symbol = kernel->name ? kernel->name : "";
    if(mode == Mode::Stream)
        out.ir = std::move(streamed);
    else if(keep_ir)
        out.ir.assign(text, text_size);
    if(result.status != 0)
        out.compile_error = diagnostics.empty() ? "rejected without a diagnostic" : diagnostics;
    else
    {
        out.hsaco.assign(result.hsaco, result.hsaco + result.hsaco_size);
        out.sample.compiler_ms = result.executable_seconds * 1000.0;
        out.stats = split_diagnostics(diagnostics, out.sample.phases);
    }

    const auto t5 = Clock::now();
    rocke_codeobj_compile_result_dispose_v1(&result);
    std::free(text);
    if(!prebuilt)
        rocke_ir_builder_free(&builder);
    const auto t6 = Clock::now();

    const auto s0 = Clock::now();
    volatile size_t lines = count_newlines(out.ir.data(), out.ir.size());
    (void)lines;
    const auto s1 = Clock::now();

    if(g_cpp_parse)
    {
        rocke_ir_builder_t parser;
        rocke_ir_builder_init(&parser, "parse");
        rocke_kernel_def_t* parsed = nullptr;
        const auto p0 = Clock::now();
        const rocke_status_t ok = rocke_ir_parse(out.ir.c_str(), &parser, &parsed);
        out.sample.cpp_parse_ms = ms_between(p0, Clock::now());
        rocke_ir_builder_free(&parser);
        if(ok != ROCKE_OK || !parsed)
            throw std::runtime_error(c.name + ": the C++ parser rejected the serialized IR");
    }

    out.sample.build_ms = ms_between(t0, t1);
    out.sample.serialize_ms = ms_between(t1, t2);
    out.sample.compile_ms = ms_between(t3, t4);
    out.sample.teardown_ms = ms_between(t5, t6);
    out.sample.scan_ms = ms_between(s0, s1);
    return out;
}

void check_image(const Case& c, const Outcome& o)
{
    if(o.hsaco.size() < 4
       || std::memcmp(o.hsaco.data(),
                      "\x7f"
                      "ELF",
                      4)
              != 0)
        throw std::runtime_error(c.name + ": direct compiler returned a non-ELF image");
    const auto* begin = reinterpret_cast<const uint8_t*>(o.symbol.data());
    if(o.symbol.empty()
       || std::search(o.hsaco.begin(), o.hsaco.end(), begin, begin + o.symbol.size())
              == o.hsaco.end())
        throw std::runtime_error(c.name + ": image does not contain kernel symbol " + o.symbol);
}

void write_file(const std::string& path, const void* data, size_t size)
{
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if(!file)
        throw std::runtime_error("cannot write " + path);
}

int parse_count(const char* text, int lo, int hi, const char* what)
{
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if(!end || *end != '\0' || value < lo || value > hi)
        throw std::runtime_error(std::string(what) + " must be an integer in " + std::to_string(lo)
                                 + ".." + std::to_string(hi));
    return static_cast<int>(value);
}

std::string json_escape(const std::string& s)
{
    std::string out;
    for(char ch : s)
    {
        if(ch == '"' || ch == '\\')
            out.push_back('\\');
        out.push_back(ch);
    }
    return out;
}

bool matches_any(const std::string& name, const std::vector<std::string>& patterns)
{
    return std::any_of(patterns.begin(), patterns.end(), [&](const std::string& p) {
        return !p.empty() && name.find(p) != std::string::npos;
    });
}

struct Report
{
    std::string json;
    bool rejected = false;
};

Report report_mode(const Case& c,
                   Mode mode,
                   const std::vector<Sample>& samples,
                   const Outcome& first,
                   double first_ms,
                   bool rejected);

/* The modes a case runs in: the GEMM entry point has no session variant. */
std::vector<Mode> modes_for(const Case& c)
{
    if(c.entry == Entry::WmmaGemm)
        return {Mode::Plain};
    return g_settings.modes;
}

/* Runs one case (see the file comment) in every mode, prints a table row per
 * mode and returns the JSON objects. Throws on build or serialize failure, on
 * a bad image, on nondeterminism and when modes disagree; a compile rejection
 * is returned, not thrown. */
Report measure_case(const Case& c, int reps, int warmup, const std::string& dump_dir)
{
    const std::vector<Mode> modes = modes_for(c);
    const auto first_start = Clock::now();
    const Outcome first = run_once(c, modes.front());
    const double first_ms = ms_between(first_start, Clock::now());
    const bool rejected = !first.compile_error.empty();
    if(!dump_dir.empty())
        write_file(dump_dir + "/" + c.name + ".cpp.ir", first.ir.data(), first.ir.size());
    if(!rejected)
    {
        check_image(c, first);
        if(!dump_dir.empty())
            write_file(
                dump_dir + "/" + c.name + ".cpp.hsaco", first.hsaco.data(), first.hsaco.size());
    }
    for(size_t m = 1; m < modes.size(); ++m)
    {
        const Outcome other = run_once(c, modes[m]);
        if(other.ir != first.ir || other.hsaco != first.hsaco
           || other.compile_error != first.compile_error)
            throw std::runtime_error(c.name + ": mode " + mode_name(modes[m]) + " disagrees with "
                                     + mode_name(modes.front()));
    }
    for(int w = 1; w < warmup; ++w)
        for(Mode mode : modes)
            run_once(c, mode, false);
    std::vector<std::vector<Sample>> by_mode(modes.size());
    for(int r = 0; r < reps; ++r)
    {
        for(size_t k = 0; k < modes.size(); ++k)
        {
            const size_t m = (k + static_cast<size_t>(r)) % modes.size();
            Outcome o = run_once(c, modes[m], false);
            if(o.hsaco != first.hsaco || o.compile_error != first.compile_error)
                throw std::runtime_error(c.name + ": repeated compiles are not deterministic");
            by_mode[m].push_back(std::move(o.sample));
        }
    }
    std::string json_rows;
    for(size_t m = 0; m < modes.size(); ++m)
    {
        const Report row = report_mode(c, modes[m], by_mode[m], first, first_ms, rejected);
        json_rows += (m ? "," : "") + row.json;
    }
    return {json_rows, rejected};
}

/* The table row and JSON object of one case in one mode. */
Report report_mode(const Case& c,
                   Mode mode,
                   const std::vector<Sample>& samples,
                   const Outcome& first,
                   double first_ms,
                   bool rejected)
{
    auto med = [&](auto field) {
        std::vector<double> v;
        for(const Sample& s : samples)
            v.push_back(field(s));
        return median(v);
    };
    const double build = med([](const Sample& s) { return s.build_ms; });
    const double serialize = med([](const Sample& s) { return s.serialize_ms; });
    const double compile = med([](const Sample& s) { return s.compile_ms; });
    const double lean
        = med([](const Sample& s) { return s.build_ms + s.serialize_ms + s.compile_ms; });
    std::map<std::string, double> phase_medians;
    for(const auto& kv : samples.front().phases)
    {
        const std::string phase = kv.first;
        phase_medians[phase] = med([&](const Sample& s) {
            const auto it = s.phases.find(phase);
            return it == s.phases.end() ? 0.0 : it->second;
        });
    }
    const double parse = phase_medians.count("parse") ? phase_medians["parse"] : 0.0;
    const std::string label
        = c.name + (g_settings.modes.size() > 1 ? std::string(" [") + mode_name(mode) + "]" : "");
    if(rejected)
        std::printf("%-40s %9.4f %9.4f  REJECTED: %s\n",
                    label.c_str(),
                    build,
                    serialize,
                    first.compile_error.c_str());
    else
        std::printf("%-40s %9.4f %9.4f %9.4f %9.4f %9.4f %9.3f %8.1f %8zu\n",
                    label.c_str(),
                    build,
                    serialize,
                    compile,
                    parse,
                    lean,
                    first_ms,
                    first.ir.size() / 1024.0,
                    first.hsaco.size());

    /* The GEMM graph also goes to the generic entry point once, for the record. */
    std::string generic_verdict;
    if(c.entry == Entry::WmmaGemm && mode == Mode::Plain)
    {
        rocke_codeobj_compile_result_v1 result;
        const int32_t status = compile_ir(Entry::RockeIr, first.ir, &result);
        generic_verdict = status == 0 && result.status == 0
                              ? std::string("compiled")
                              : bytes_to_string(result.diagnostics, result.diagnostics_size);
        rocke_codeobj_compile_result_dispose_v1(&result);
        std::printf("%-32s rocke_codeobj_compile_rocke_ir_v1 on the same IR: %s\n",
                    "",
                    generic_verdict.c_str());
    }
    std::fflush(stdout);

    std::ostringstream json;
    json << "{\"case\":\"" << c.name << "\",\"family\":\"" << c.family << "\",\"mode\":\""
         << mode_name(mode) << "\",\"entry\":\""
         << (c.entry == Entry::WmmaGemm ? "rocke_codeobj_compile_gfx1151_wmma_gemm_v1"
             : mode == Mode::Plain      ? "rocke_codeobj_compile_rocke_ir_v1"
             : mode == Mode::Session
                 ? "rocke_codeobj_session_compile_v1"
                 : "rocke_codeobj_session_feed_v1+rocke_codeobj_session_finish_v1")
         << "\",\"symbol\":\"" << json_escape(first.symbol) << "\",\"ir_bytes\":" << first.ir.size()
         << ",\"ir_lines\":" << count_newlines(first.ir.data(), first.ir.size())
         << ",\"hsaco_bytes\":" << first.hsaco.size() << ",\"stats\":\"" << json_escape(first.stats)
         << "\",\"compile_error\":\"" << json_escape(first.compile_error) << "\"";
    if(!generic_verdict.empty())
        json << ",\"generic_entry\":\"" << json_escape(generic_verdict) << "\"";
    json << ",\"first_ms\":" << first_ms << ",\"median_ms\":{\"build\":" << build
         << ",\"serialize\":" << serialize << ",\"compile\":" << compile
         << ",\"compiler_internal\":" << med([](const Sample& s) { return s.compiler_ms; })
         << ",\"teardown\":" << med([](const Sample& s) { return s.teardown_ms; })
         << ",\"scan\":" << med([](const Sample& s) { return s.scan_ms; })
         << ",\"cpp_parse\":" << med([](const Sample& s) { return s.cpp_parse_ms; })
         << ",\"lean_total\":" << lean << ",\"phases\":{";
    bool comma = false;
    for(const auto& kv : phase_medians)
    {
        json << (comma ? "," : "") << "\"" << kv.first << "\":" << kv.second;
        comma = true;
    }
    json << "}}}";
    return {json.str(), rejected};
}
} // namespace

int main(int argc, char** argv)
try
{
    int reps = 3;
    int warmup = 1;
    int session_threads = 0;
    std::string only, dump_dir, json_path;
    std::vector<std::string> expect_reject;
    std::vector<std::pair<std::string, std::string>> ir_cases;
    for(int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if(arg == "--cpp-parse")
        {
            g_cpp_parse = true;
            continue;
        }
        if(i + 1 >= argc)
            throw std::runtime_error("missing value after " + arg);
        if(arg == "--reps")
            reps = parse_count(argv[++i], 1, 100000, "--reps");
        else if(arg == "--warmup")
            warmup = parse_count(argv[++i], 1, 100000, "--warmup");
        else if(arg == "--only")
            only = argv[++i];
        else if(arg == "--expect-reject")
        {
            std::stringstream list(argv[++i]);
            std::string item;
            while(std::getline(list, item, ','))
                expect_reject.push_back(item);
        }
        else if(arg == "--dump")
            dump_dir = argv[++i];
        else if(arg == "--json")
            json_path = argv[++i];
        else if(arg == "--modes")
        {
            g_settings.modes.clear();
            std::stringstream list(argv[++i]);
            std::string item;
            while(std::getline(list, item, ','))
            {
                if(item == "plain")
                    g_settings.modes.push_back(Mode::Plain);
                else if(item == "session")
                    g_settings.modes.push_back(Mode::Session);
                else if(item == "stream")
                    g_settings.modes.push_back(Mode::Stream);
                else
                    throw std::runtime_error("unknown mode " + item);
            }
            if(g_settings.modes.empty())
                throw std::runtime_error("--modes needs at least one mode");
        }
        else if(arg == "--ir-case")
        {
            const std::string spec = argv[++i];
            const size_t eq = spec.find('=');
            if(eq == std::string::npos || eq == 0)
                throw std::runtime_error("--ir-case takes NAME=FILE");
            ir_cases.emplace_back(spec.substr(0, eq), spec.substr(eq + 1));
        }
        else if(arg == "--session-threads")
            session_threads = parse_count(argv[++i], 0, 256, "--session-threads");
        else if(arg == "--feed-bytes")
            g_settings.feed_bytes
                = static_cast<size_t>(parse_count(argv[++i], 1, 1 << 30, "--feed-bytes"));
        else
            throw std::runtime_error("unknown argument " + arg);
    }
    if(rocke_codeobj_ffi_abi_version() != ROCKE_CODEOBJ_FFI_ABI_VERSION)
        throw std::runtime_error("Rust code-object library has an incompatible ABI");
    const bool needs_session = std::any_of(
        g_settings.modes.begin(), g_settings.modes.end(), [](Mode m) { return m != Mode::Plain; });
    if(needs_session)
    {
        rocke_codeobj_session_options_v1 options = {};
        options.struct_size = sizeof(options);
        options.worker_threads = static_cast<uint32_t>(session_threads);
        if(rocke_codeobj_session_create_v1(&options, &g_settings.session) != 0)
            throw std::runtime_error("rocke_codeobj_session_create_v1 failed");
    }
    struct SessionGuard
    {
        ~SessionGuard()
        {
            rocke_codeobj_session_free_v1(g_settings.session);
            g_settings.session = nullptr;
        }
    } session_guard;

    std::vector<Case> cases = all_cases();
    for(const auto& [name, path] : ir_cases)
    {
        std::ifstream file(path, std::ios::binary);
        if(!file)
            throw std::runtime_error("cannot read " + path);
        const std::string text((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        Case c;
        c.name = name;
        c.family = "ir";
        c.parsed_builder = std::shared_ptr<rocke_ir_builder_t>(new rocke_ir_builder_t,
                                                               [](rocke_ir_builder_t* b) {
                                                                   rocke_ir_builder_free(b);
                                                                   delete b;
                                                               });
        rocke_ir_builder_init(c.parsed_builder.get(), "parse");
        if(rocke_ir_parse(text.c_str(), c.parsed_builder.get(), &c.parsed) != ROCKE_OK || !c.parsed)
        {
            const char* message = rocke_ir_builder_error(c.parsed_builder.get());
            throw std::runtime_error(name
                                     + ": rocke_ir_parse failed: " + (message ? message : "?"));
        }
        cases.push_back(std::move(c));
    }

    std::ostringstream json;
    json << "{\"arch\":\"" << kArch << "\",\"reps\":" << reps << ",\"warmup\":" << warmup
         << ",\"cases\":[";
    std::printf("%-40s %9s %9s %9s %9s %9s %9s %8s %8s\n",
                "case",
                "build",
                "serialize",
                "compile",
                "parse",
                "lean",
                "first",
                "ir_kb",
                "hsaco");
    int ran = 0;
    std::vector<std::string> failures;
    for(const Case& c : cases)
    {
        if(!only.empty() && c.name.find(only) == std::string::npos)
            continue;
        json << (ran ? "," : "") << "\n ";
        ++ran;
        try
        {
            const Report report = measure_case(c, reps, warmup, dump_dir);
            json << report.json;
            const bool expected = matches_any(c.name, expect_reject);
            if(report.rejected && !expected)
                failures.push_back(c.name + " (rejected)");
            else if(!report.rejected && expected)
                failures.push_back(c.name + " (compiles now; drop it from --expect-reject)");
        }
        catch(const std::exception& error)
        {
            std::printf("%-40s FAILED: %s\n", c.name.c_str(), error.what());
            std::fflush(stdout);
            json << "{\"case\":\"" << c.name << "\",\"family\":\"" << c.family << "\",\"error\":\""
                 << json_escape(error.what()) << "\"}";
            failures.push_back(c.name);
        }
    }
    json << "\n]}\n";
    if(ran == 0)
        throw std::runtime_error("no case matched --only " + only);
    if(!json_path.empty())
    {
        const std::string text = json.str();
        write_file(json_path, text.data(), text.size());
    }
    std::printf("cases=%d failed=%zu reps=%d external_compiler=false cache=false\n",
                ran,
                failures.size(),
                reps);
    for(const std::string& name : failures)
        std::fprintf(stderr, "failed: %s\n", name.c_str());
    return failures.empty() ? 0 : 1;
}
catch(const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
