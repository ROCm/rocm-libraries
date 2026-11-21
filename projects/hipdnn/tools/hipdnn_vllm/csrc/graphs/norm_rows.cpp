// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// rms_norm_rows / fused_add_rms_norm_rows: the vLLM IR providers' ops.
//
// Dynamic rows. No shipped norm engine takes a runtime row count, and
// a plan is built for one exact shape. Rows along dim 0 (num_tokens in vLLM)
// are independent, so a call runs as:
//   * one launch of the exact plan when the row count is one of the exact
//     rows (vLLM's cudagraph capture sizes) whose plans were pre-built the
//     first time this signature ran eagerly;
//   * otherwise one launch per set bit of the row count, each a 2^k-row chunk
//     with its own plan, so at most log2(max rows) plans per signature.
// The route depends only on the row count, never on data or capture state, so
// a capture replays the launches its eager warm-up built. Under capture no
// plan is built (runtime.h): a missing plan is a CAPTURE_INVARIANT fallback.
// Each launch describes its chunk as the [R, N] view norm.cpp takes.
//
// Fallback. vLLM resolves the IR provider when it traces the graph, so a
// decline at run time cannot go back to vLLM's priority list. The op instead
// computes vLLM's native formula with ATen (route=fallback, counted with the
// reason), or raises when `strict`.
//
// fused_add_rms_norm. If an engine takes the graph pointwise(ADD) -> rmsnorm
// with the sum as a second output, the op runs that graph. That is probed once
// per signature on first use. Otherwise it decomposes into an ATen add plus
// the hipDNN rmsnorm, counted as (fused_add_rms_norm, hipdnn, PARTIAL).

#include "graphs.h"

#include "../graph_api.h"
#include "../stats.h"

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = hipdnn_frontend;
namespace py = pybind11;

namespace hipdnn_vllm
{
namespace
{

constexpr const char* OP_RMS = "rms_norm";
constexpr const char* OP_FUSED = "fused_add_rms_norm";
constexpr const char* REASON_PARTIAL = "PARTIAL";

constexpr int64_t UID_FX = 1;
constexpr int64_t UID_FR = 2;
constexpr int64_t UID_FW = 3;
constexpr int64_t UID_FEPS = 4;
constexpr int64_t UID_FSUM = 5;
constexpr int64_t UID_FY = 6;

constexpr int MAX_ROW_BIT = 62;

uint64_t nowNs()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

bool capturing(const at::Tensor& anchor)
{
    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA(anchor.device().index()).stream();
    hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
    return hipStreamIsCapturing(stream, &status) == hipSuccess && status == hipStreamCaptureStatusActive;
}

// ---------------------------------------------------------------------------
// Process state

struct RowSig
{
    std::vector<int64_t> exact; // sorted; rows whose exact plan built
};

struct RowState
{
    // Exact-row configuration.
    std::mutex configMutex;
    bool configured = false;
    std::vector<int64_t> exactRows;

    // Per-signature state.
    std::mutex sigMutex;
    std::unordered_map<std::string, std::unique_ptr<RowSig>> sigs;
    std::unordered_map<std::string, bool> fused; // fused_add signature -> engine took the fused graph

    std::atomic<uint64_t> prebuildSigs{0};
    std::atomic<uint64_t> prebuildPlans{0};
    std::atomic<uint64_t> prebuildDeclines{0};
    std::atomic<uint64_t> prebuildNs{0};
};

RowState& rowState()
{
    // Never destroyed, like the plan cache: no teardown ordering at exit.
    static auto* s_state = new RowState();
    return *s_state;
}

void setExactRowsLocked(RowState& state, std::vector<int64_t> rows)
{
    std::erase_if(rows, [](int64_t r) { return r < 1; });
    std::ranges::sort(rows);
    const auto duplicates = std::ranges::unique(rows);
    rows.erase(duplicates.begin(), duplicates.end());
    state.exactRows = std::move(rows);
    state.configured = true;
}

// The configured exact rows. When nothing configured them yet, ask
// hipdnn_vllm.ops.norm_ir (it reads vLLM's current compilation config);
// None from Python means "not known in this context", so ask again later.
std::vector<int64_t> exactRows()
{
    RowState& state = rowState();
    {
        const std::lock_guard<std::mutex> lock(state.configMutex);
        if(state.configured)
        {
            return state.exactRows;
        }
    }
    std::optional<std::vector<int64_t>> rows;
    {
        const py::gil_scoped_acquire gil;
        try
        {
            const py::object result
                = py::module_::import("hipdnn_vllm.ops.norm_ir").attr("exact_rows_for_process")();
            if(!result.is_none())
            {
                rows = result.cast<std::vector<int64_t>>();
            }
        }
        catch(py::error_already_set& e)
        {
            TORCH_WARN("hipdnn_vllm: could not read the exact norm row counts: ", e.what());
        }
    }
    const std::lock_guard<std::mutex> lock(state.configMutex);
    if(!state.configured && rows)
    {
        setExactRowsLocked(state, std::move(*rows));
    }
    return state.configured ? state.exactRows : std::vector<int64_t>{};
}

// Everything but the row count.
std::string sigKey(const char* op, const at::Tensor& x, const at::Tensor& weight, double eps)
{
    PlanKey key(op);
    key.add(eps).add(static_cast<int64_t>(x.device().index())).add(weight);
    key.add(static_cast<int64_t>(x.scalar_type()));
    for(int64_t i = 1; i < x.dim(); ++i)
    {
        key.add(x.size(i));
    }
    for(const int64_t s : x.strides())
    {
        key.add(s);
    }
    return key.str();
}

// Find the signature's state. On first sight outside capture, pre-build the
// exact-row plans with `prepare(rows)`. A signature first seen while capturing
// gets an empty, unrecorded state (pre-building would build inside the
// capture), so its next eager call still pre-builds.
template <typename Prepare>
const RowSig& rowSig(const std::string& key, bool capture, const Prepare& prepare)
{
    RowState& state = rowState();
    {
        const std::lock_guard<std::mutex> lock(state.sigMutex);
        auto it = state.sigs.find(key);
        if(it != state.sigs.end())
        {
            return *it->second;
        }
    }
    if(capture)
    {
        static const RowSig s_unprepared;
        return s_unprepared;
    }
    auto sig = std::make_unique<RowSig>();
    const std::vector<int64_t> rows = exactRows();
    const uint64_t start = nowNs();
    for(const int64_t r : rows)
    {
        if(prepare(r))
        {
            state.prebuildDeclines.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            sig->exact.push_back(r);
            state.prebuildPlans.fetch_add(1, std::memory_order_relaxed);
        }
    }
    state.prebuildNs.fetch_add(nowNs() - start, std::memory_order_relaxed);
    state.prebuildSigs.fetch_add(1, std::memory_order_relaxed);
    const std::lock_guard<std::mutex> lock(state.sigMutex);
    auto it = state.sigs.emplace(key, std::move(sig)).first;
    return *it->second;
}

// The fused-graph decision for a signature, probed (`probe()`) on its first
// eager call. Inside a capture an undecided signature decomposes (no build
// while capturing) and is decided on its next eager call.
template <typename Probe>
bool fusedDecision(const std::string& key, bool capture, const Probe& probe)
{
    RowState& state = rowState();
    {
        const std::lock_guard<std::mutex> lock(state.sigMutex);
        auto it = state.fused.find(key);
        if(it != state.fused.end())
        {
            return it->second;
        }
    }
    if(capture)
    {
        return false;
    }
    const bool ok = probe();
    const std::lock_guard<std::mutex> lock(state.sigMutex);
    return state.fused.emplace(key, ok).first->second;
}

// Run `launch(row0, rows)` per the row policy. Returns the first decline.
template <typename Launch>
std::optional<Decline> runRows(int64_t m, const RowSig& sig, const Launch& launch)
{
    if(std::ranges::binary_search(sig.exact, m))
    {
        auto decline = launch(0, m);
        if(!decline)
        {
            return decline;
        }
    }
    int64_t row0 = 0;
    for(int bit = MAX_ROW_BIT; bit >= 0; --bit)
    {
        const int64_t chunk = int64_t{1} << bit;
        if((m & chunk) == 0)
        {
            continue;
        }
        if(auto decline = launch(row0, chunk))
        {
            return decline;
        }
        row0 += chunk;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Tensor views

at::Tensor asRank2Plus(const at::Tensor& x)
{
    return x.dim() == 1 ? x.unsqueeze(0) : x;
}

// The [R, N] view of a tensor whose full view is known to collapse (any dim-0
// slice of a collapsible tensor collapses too).
at::Tensor rows2d(const at::Tensor& t)
{
    std::optional<at::Tensor> rows = rmsNormAsRows(t);
    TORCH_INTERNAL_ASSERT(rows.has_value(), "hipdnn_vllm: row view of a collapsible tensor failed");
    return *std::move(rows);
}

// `t` with `rows` rows, as a meta tensor (same strides and dtype).
at::Tensor metaRows(const at::Tensor& t, int64_t rows)
{
    std::vector<int64_t> sizes = t.sizes().vec();
    sizes[0] = rows;
    return at::empty_strided(sizes, t.strides(), t.options().device(at::kMeta));
}

at::Tensor metaContiguousRows(const at::Tensor& t, int64_t rows)
{
    std::vector<int64_t> sizes = t.sizes().vec();
    sizes[0] = rows;
    return at::empty(sizes, t.options().device(at::kMeta));
}

// ---------------------------------------------------------------------------
// Native formulas (vllm/ir/ops/layernorm.py), used on a decline. Outputs are
// contiguous, as the fake impls state.

at::Tensor nativeRms(const at::Tensor& xf, const at::Tensor& weight, at::ScalarType outType, double eps)
{
    const at::Tensor variance = xf.pow(2).mean(-1, /*keepdim=*/true);
    at::Tensor y = xf * at::rsqrt(variance + eps);
    y = y.to(weight.scalar_type()) * weight;
    return y.to(outType).contiguous();
}

[[noreturn]] void raiseDecline(const char* op, const Decline& decline)
{
    TORCH_CHECK(false, declineErrorPrefix(op), decline.reason(), " :: ", decline.message);
}

struct Cells
{
    std::atomic<uint64_t>* eager;
    std::atomic<uint64_t>* capture;
};

const Cells& partialCells()
{
    static const Cells s_cells{.eager = Stats::instance().cell(OP_FUSED, ROUTE_HIPDNN, REASON_PARTIAL, false),
                               .capture = Stats::instance().cell(OP_FUSED, ROUTE_HIPDNN, REASON_PARTIAL, true)};
    return s_cells;
}

// rms over x (>= 2-D, collapsible) into a fresh contiguous y of x's shape; the
// decline when hipDNN did not run (already counted by the graph API).
std::optional<Decline> rmsRowsInto(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps)
{
    const bool capture = capturing(x);
    const RowSig& sig = rowSig(sigKey(OP_RMS, x, weight, eps), capture, [&](int64_t rows) {
        return rmsNormPrepare(
            rows2d(metaRows(x, rows)), weight, rows2d(metaContiguousRows(y, rows)), eps, x);
    });
    return runRows(x.size(0), sig, [&](int64_t row0, int64_t rows) {
        return rmsNormInto(rows2d(x.narrow(0, row0, rows)), weight, rows2d(y.narrow(0, row0, rows)), eps);
    });
}

// ---------------------------------------------------------------------------
// fused graph: sum = x + residual (output), y = rmsnorm(sum, weight); all
// tensors are [R, N] views described as [R, N, 1, 1].

GraphBuilder fusedBuilder(const at::Tensor& x,
                          const at::Tensor& residual,
                          const at::Tensor& weight,
                          const at::Tensor& sum,
                          const at::Tensor& y,
                          double eps)
{
    return [&x, &residual, &weight, &sum, &y, eps]() -> BuildResult {
        const fe::DataType io = *toHipdnnType(x.scalar_type());
        auto graph = std::make_shared<fe::graph::Graph>();
        graph->set_name(OP_FUSED).set_io_data_type(io).set_compute_data_type(fe::DataType::FLOAT);

        auto described = [](const at::Tensor& t, int64_t uid, const char* name) {
            std::vector<int64_t> dims;
            std::vector<int64_t> strides;
            rmsNormRowLayout(t, dims, strides);
            return fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                                .set_name(name)
                                                .set_uid(uid)
                                                .set_dim(dims)
                                                .set_stride(strides)
                                                .set_data_type(*toHipdnnType(t.scalar_type())));
        };
        auto xT = described(x, UID_FX, "x");
        auto rT = described(residual, UID_FR, "residual");

        std::vector<int64_t> dims;
        std::vector<int64_t> strides;
        rmsNormRowLayout(x, dims, strides);
        std::vector<int64_t> wDims(dims.size(), 1);
        std::vector<int64_t> wStrides(dims.size(), 0);
        wDims[1] = weight.size(0);
        wStrides[1] = weight.stride(0);
        auto wT = fe::graph::Graph::tensor(fe::graph::TensorAttributes()
                                               .set_name("weight")
                                               .set_uid(UID_FW)
                                               .set_dim(wDims)
                                               .set_stride(wStrides)
                                               .set_data_type(*toHipdnnType(weight.scalar_type())));
        auto epsT = fe::graph::Graph::tensor(
            fe::graph::TensorAttributes(static_cast<float>(eps)).set_name("eps").set_uid(UID_FEPS));

        auto sumT = graph->pointwise(
            xT, rT, fe::graph::PointwiseAttributes().set_name("add").set_mode(fe::PointwiseMode::ADD));
        rmsNormRowLayout(sum, dims, strides);
        sumT->set_uid(UID_FSUM).set_dim(dims).set_stride(strides).set_data_type(io).set_output(true);

        fe::graph::RMSNormAttributes attrs;
        attrs.set_forward_phase(fe::NormFwdPhase::INFERENCE).set_epsilon(epsT);
        auto outputs = graph->rmsnorm(sumT, wT, attrs);
        rmsNormRowLayout(y, dims, strides);
        outputs[0]->set_uid(UID_FY).set_dim(dims).set_stride(strides).set_data_type(io).set_output(true);
        return GraphDef{.graph = graph, .slotUids = {UID_FX, UID_FR, UID_FW, UID_FSUM, UID_FY}};
    };
}

PlanKey fusedKey(const at::Tensor& x,
                 const at::Tensor& residual,
                 const at::Tensor& weight,
                 const at::Tensor& sum,
                 const at::Tensor& y,
                 double eps)
{
    PlanKey key(OP_FUSED);
    key.add(eps).add(x).add(residual).add(weight).add(sum).add(y);
    return key;
}

std::optional<Decline> fusedInto(const at::Tensor& x,
                                 const at::Tensor& residual,
                                 const at::Tensor& weight,
                                 const at::Tensor& sum,
                                 const at::Tensor& y,
                                 double eps)
{
    return runGraphOp(OP_FUSED,
                      fusedKey(x, residual, weight, sum, y, eps),
                      x,
                      fusedBuilder(x, residual, weight, sum, y, eps),
                      {x.data_ptr(), residual.data_ptr(), weight.data_ptr(), sum.data_ptr(), y.data_ptr()});
}

std::string fusedSigKey(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double eps)
{
    std::string key = sigKey(OP_FUSED, x, weight, eps);
    key += sigKey("r", residual, weight, eps);
    return key;
}

std::optional<Decline> collapsible(const at::Tensor& t)
{
    if(!rmsNormAsRows(t))
    {
        return Decline::feature("noncollapsible_rows");
    }
    return std::nullopt;
}

} // namespace

at::Tensor rmsNormRows(const at::Tensor& xIn, const at::Tensor& weight, double eps, bool strict)
{
    std::optional<Decline> decline = rmsNormPrefilter(xIn, weight);
    if(!decline)
    {
        decline = collapsible(xIn);
    }
    if(decline)
    {
        if(strict)
        {
            raiseDecline(OP_RMS, *decline);
        }
        countDecline(OP_RMS, *decline, xIn);
        return nativeRms(xIn.to(at::kFloat), weight, xIn.scalar_type(), eps);
    }
    const at::Tensor x = asRank2Plus(xIn);
    const at::Tensor y = at::empty(x.sizes(), x.options().memory_format(at::MemoryFormat::Contiguous));
    if(auto failed = rmsRowsInto(x, weight, y, eps))
    {
        if(strict)
        {
            raiseDecline(OP_RMS, *failed);
        }
        return nativeRms(xIn.to(at::kFloat), weight, xIn.scalar_type(), eps);
    }
    return y.view(xIn.sizes());
}

std::tuple<at::Tensor, at::Tensor> fusedAddRmsNormRows(
    const at::Tensor& xIn, const at::Tensor& residualIn, const at::Tensor& weight, double eps, bool strict)
{
    TORCH_CHECK(xIn.sizes() == residualIn.sizes(),
                "fused_add_rms_norm: x ",
                xIn.sizes(),
                " and residual ",
                residualIn.sizes(),
                " must have the same shape");
    auto fallback = [&](const Decline& decline, bool count) {
        if(strict)
        {
            raiseDecline(OP_FUSED, decline);
        }
        if(count)
        {
            countDecline(OP_FUSED, decline, xIn);
        }
        const at::Tensor sumF = xIn.to(at::kFloat) + residualIn.to(at::kFloat);
        return std::make_tuple(nativeRms(sumF, weight, xIn.scalar_type(), eps),
                               sumF.to(xIn.scalar_type()).contiguous());
    };
    std::optional<Decline> decline = rmsNormPrefilter(xIn, weight);
    if(!decline && residualIn.scalar_type() != xIn.scalar_type())
    {
        decline = Decline::feature("residual_dtype");
    }
    if(!decline && residualIn.stride(-1) != 1)
    {
        decline = Decline::feature("strided_rows");
    }
    if(!decline)
    {
        decline = collapsible(xIn);
    }
    if(!decline)
    {
        decline = collapsible(residualIn);
    }
    if(decline)
    {
        return fallback(*decline, true);
    }

    const at::Tensor x = asRank2Plus(xIn);
    const at::Tensor residual = asRank2Plus(residualIn);
    const auto contiguous = x.options().memory_format(at::MemoryFormat::Contiguous);
    const bool capture = capturing(x);
    at::Tensor sum = at::empty(x.sizes(), contiguous);
    const at::Tensor y = at::empty(x.sizes(), contiguous);

    const std::string key = fusedSigKey(x, residual, weight, eps);
    auto prepare = [&](int64_t rows) {
        const at::Tensor xm = rows2d(metaRows(x, rows));
        const at::Tensor rm = rows2d(metaRows(residual, rows));
        const at::Tensor om = rows2d(metaContiguousRows(x, rows));
        return probeGraphOp(OP_FUSED, fusedKey(xm, rm, weight, om, om, eps), x, fusedBuilder(xm, rm, weight, om, om, eps));
    };
    if(fusedDecision(key, capture, [&] { return !prepare(1).has_value(); }))
    {
        const RowSig& sig = rowSig(key, capture, prepare);
        auto failed = runRows(x.size(0), sig, [&](int64_t row0, int64_t rows) {
            return fusedInto(rows2d(x.narrow(0, row0, rows)),
                             rows2d(residual.narrow(0, row0, rows)),
                             weight,
                             rows2d(sum.narrow(0, row0, rows)),
                             rows2d(y.narrow(0, row0, rows)),
                             eps);
        });
        if(failed)
        {
            // Already counted as a fused_add_rms_norm fallback by the graph API.
            return fallback(*failed, false);
        }
    }
    else
    {
        at::add_out(sum, x, residual);
        if(auto failed = rmsRowsInto(sum, weight, y, eps))
        {
            return fallback(*failed, true);
        }
        const Cells& cells = partialCells();
        (capture ? cells.capture : cells.eager)->fetch_add(1, std::memory_order_relaxed);
    }
    return std::make_tuple(y.view(xIn.sizes()), sum.view(xIn.sizes()));
}

void normSetExactRows(std::vector<int64_t> rows)
{
    RowState& state = rowState();
    const std::lock_guard<std::mutex> lock(state.configMutex);
    setExactRowsLocked(state, std::move(rows));
}

std::optional<std::vector<int64_t>> normExactRows()
{
    RowState& state = rowState();
    const std::lock_guard<std::mutex> lock(state.configMutex);
    if(!state.configured)
    {
        return std::nullopt;
    }
    return state.exactRows;
}

std::string fusedAddRoute(const at::Tensor& xIn, const at::Tensor& residualIn, const at::Tensor& weight, double eps)
{
    const std::string key = fusedSigKey(asRank2Plus(xIn), asRank2Plus(residualIn), weight, eps);
    RowState& state = rowState();
    const std::lock_guard<std::mutex> lock(state.sigMutex);
    auto it = state.fused.find(key);
    if(it == state.fused.end())
    {
        return {};
    }
    return it->second ? "fused" : "decomposed";
}

NormPrebuildTotals normPrebuildTotals()
{
    const RowState& state = rowState();
    return NormPrebuildTotals{.signatures = state.prebuildSigs.load(),
                              .plans = state.prebuildPlans.load(),
                              .declines = state.prebuildDeclines.load(),
                              .ns = state.prebuildNs.load()};
}

void normRowsReset()
{
    RowState& state = rowState();
    {
        const std::lock_guard<std::mutex> lock(state.sigMutex);
        state.sigs.clear();
        state.fused.clear();
    }
    const std::lock_guard<std::mutex> lock(state.configMutex);
    state.configured = false;
    state.exactRows.clear();
    state.prebuildSigs = 0;
    state.prebuildPlans = 0;
    state.prebuildDeclines = 0;
    state.prebuildNs = 0;
}

} // namespace hipdnn_vllm
