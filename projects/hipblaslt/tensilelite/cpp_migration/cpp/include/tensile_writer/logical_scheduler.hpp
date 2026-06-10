// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Pure C++ port of the *data / config* primitives of the subtile
// LogicalScheduler (Tensile/Components/Subtile/LogicalScheduler.py).
//
// This header is intentionally free of any nanobind / Python / rocisa
// dependency: it ports only the value/config types and the pure static
// helpers (no scheduling passes, no register allocation, no emission). The
// names below mirror the Python module so the two implementations can be
// reasoned about side by side.
//
// SCOPE: the pure data/config + value-type layer —
//   Pass                         -> Pass
//   fmt_mt                       -> fmt_mt
//   MFMATileRange                -> MFMATileRange
//   ReadGranularity              -> ReadGranularity
//   SchedulerConfig              -> SchedulerConfig (incl. partition
//                                   normalization + candidate generation)
//   MFMAPlacement / LRPlacement / GRPlacement / WaitGRCounts and the
//   before-chain op value types (their identity fields + kind + str()),
//   plus the remaining value types: the pass-populated list fields on
//   placements (deps / preOps / postOps / vgpr_tile_map[s]), Dep,
//   SubIterKSlot, EmittedModule, and InlineModuleOp.
//
// NOT ported here (remain Python): the scheduling passes themselves
//   (place_LRs / place_GRs / annotate_deps / remove_* / build /
//   populate_instructions), InstructionEmitter dispatch, rocisa Module
//   emission, the EmittedModule `instructions` list (rocisa objects), and the
//   InlineModuleOp `build` Callable. The pass-populated fields above default
//   to empty: they are ported as structurally faithful value types, not filled
//   by any C++ pass.

#pragma once

#include <algorithm>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tw::subtile::lsched {

// Floor division matching Python's `//` for the non-negative operands used
// throughout the config math.
inline long floordiv(long a, long b) {
  long q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

// Scheduler passes in dependency order. The numeric value defines topological
// order. Mirrors Python `class Pass(IntEnum)`.
enum class Pass : int {
  LR = 0,
  VGPR_TILES = 1,
  GR = 2,
  DEPS = 3,
  REMOVE_GR_DEPS = 4,
  REMOVE_LR_DEPS = 5,
  REMOVE_DEPS = 6,
  GR_INC = 7,
  GROUP_LR_GR = 8,
  REMOVE_WAIT_LR_SYNC = 9,
  EMIT = 10,
  BUILD = 11,
  POPULATE = 12,
};

// Format an MT iteration integer as a display string: 0 -> "n", 1 -> "n+1".
inline std::string fmt_mt(int mt) {
  return mt == 0 ? std::string("n") : ("n+" + std::to_string(mt));
}

// ── Core primitives ─────────────────────────────────────────

// A rectangular range of MFMA tile coordinates for one read.
struct MFMATileRange {
  int subIterK_start = 0;
  int subIterK_end = 0;  // exclusive
  int tileId_start = 0;
  int tileId_end = 0;    // exclusive

  MFMATileRange() = default;
  MFMATileRange(int sk_start, int sk_end, int t_start, int t_end)
      : subIterK_start(sk_start),
        subIterK_end(sk_end),
        tileId_start(t_start),
        tileId_end(t_end) {}

  std::vector<int> subIterK_list() const {
    std::vector<int> out;
    for (int i = subIterK_start; i < subIterK_end; ++i) out.push_back(i);
    return out;
  }

  std::vector<int> tileId_list() const {
    std::vector<int> out;
    for (int i = tileId_start; i < tileId_end; ++i) out.push_back(i);
    return out;
  }

  std::string fmt_k() const {
    std::vector<int> ids = subIterK_list();
    if (ids.size() == 1) return "[" + std::to_string(ids[0]) + "]";
    return "[" + std::to_string(ids.front()) + "," + std::to_string(ids.back()) +
           "]";
  }

  std::string fmt_tiles() const {
    return "[" + std::to_string(tileId_start) + "-" +
           std::to_string(tileId_end - 1) + "]";
  }
};

// ── Config ──────────────────────────────────────────────────

// Load granularity for one operation on one tensor, measured in MFMA tiles.
//   mn: MFMA tiles in the M (A/SA) or N (B/SB) dimension.
//   k:  subIterK steps one read covers.
struct ReadGranularity {
  int mn = 0;
  int k = 0;

  ReadGranularity() = default;
  ReadGranularity(int mn_, int k_) : mn(mn_), k(k_) {}

  // Snap subIterK and tile indices to this granularity.
  MFMATileRange tile_range(int k_, int t_start, int t_end) const {
    long ks = floordiv(k_, k) * k;
    long ts = floordiv(t_start, mn) * mn;
    long te = floordiv(t_end + mn - 1, mn) * mn;
    return MFMATileRange(static_cast<int>(ks), static_cast<int>(ks + k),
                         static_cast<int>(ts), static_cast<int>(te));
  }
};

// Partition spec: either a single tile size (0 == full dim) or an explicit
// per-partition list. Mirrors Python's Union[int, List[int]].
using PartitionSpec = std::variant<int, std::vector<int>>;

// Configuration for the MFMATile-based scheduler.
struct SchedulerConfig {
  // Inputs (mirror the Python dataclass fields).
  int numMFMATilesM = 0;
  int numMFMATilesN = 0;
  int numSubIterK = 0;
  ReadGranularity lrA;
  ReadGranularity lrB;
  ReadGranularity grA;
  ReadGranularity grB;
  std::optional<ReadGranularity> lrSA;
  std::optional<ReadGranularity> lrSB;
  std::optional<ReadGranularity> grSA;
  std::optional<ReadGranularity> grSB;
  PartitionSpec partitionSizeM = 0;
  PartitionSpec partitionSizeN = 0;
  int pgr = 2;

  // Derived (post-init).
  std::vector<int> _partitionSizesM;
  std::vector<int> _partitionSizesN;
  std::vector<int> _prefixM;
  std::vector<int> _prefixN;
  int plr = 0;
  int offsetPartition = 0;

  // Resolve a partition spec into per-partition sizes along one dimension.
  // Mirrors Python SchedulerConfig._normalize_partition_sizes.
  static std::vector<int> normalize_partition_sizes(const std::vector<int>& spec,
                                                    int total,
                                                    const std::string& dim,
                                                    int mn = 1) {
    int s = 0;
    for (int v : spec) s += v;
    if (s != total) {
      throw std::invalid_argument("partition sizes for " + dim +
                                  " must sum to " + std::to_string(total) +
                                  ", got " + std::to_string(s));
    }
    for (int v : spec) {
      if (v < 1) {
        throw std::invalid_argument("all partition sizes for " + dim +
                                    " must be >= 1");
      }
    }
    for (int v : spec) {
      if (v % mn != 0) {
        throw std::invalid_argument("partition sizes for " + dim +
                                    " must be multiples of mn=" +
                                    std::to_string(mn));
      }
    }
    return spec;
  }

  static std::vector<int> normalize_partition_sizes(int spec, int total,
                                                    const std::string& dim,
                                                    int mn = 1) {
    int s = spec != 0 ? spec : total;
    if (!(1 <= s && s <= total)) {
      throw std::invalid_argument("partition size for " + dim +
                                  " must be in [1, " + std::to_string(total) +
                                  "], got " + std::to_string(s));
    }
    if (total % mn != 0) return {total};
    s = std::max(mn, (s / mn) * mn);
    if (s > total) return {total};
    int num_full = total / s;
    int remainder = total - num_full * s;
    if (remainder == 0) return std::vector<int>(num_full, s);
    if (num_full == 1) return {s, remainder};
    int mid = num_full / 2;
    std::vector<int> out(mid, s);
    out.push_back(remainder);
    out.insert(out.end(), num_full - mid, s);
    return out;
  }

  static std::vector<int> normalize_partition_sizes(const PartitionSpec& spec,
                                                    int total,
                                                    const std::string& dim,
                                                    int mn = 1) {
    if (std::holds_alternative<int>(spec)) {
      return normalize_partition_sizes(std::get<int>(spec), total, dim, mn);
    }
    return normalize_partition_sizes(std::get<std::vector<int>>(spec), total,
                                     dim, mn);
  }

  static std::vector<int> build_prefix(const std::vector<int>& sizes) {
    std::vector<int> prefix = {0};
    for (int s : sizes) prefix.push_back(prefix.back() + s);
    return prefix;
  }

  // Mirrors Python __post_init__.
  void post_init() {
    if (!(pgr == 0 || pgr == 1 || pgr == 2)) {
      throw std::invalid_argument("pgr must be 0, 1, or 2, got " +
                                  std::to_string(pgr));
    }
    int mn_M = 1;
    if (lrA.mn > mn_M) mn_M = lrA.mn;
    if (lrSA.has_value() && lrSA->mn > mn_M) mn_M = lrSA->mn;
    int mn_N = 1;
    if (lrB.mn > mn_N) mn_N = lrB.mn;
    if (lrSB.has_value() && lrSB->mn > mn_N) mn_N = lrSB->mn;
    _partitionSizesM =
        normalize_partition_sizes(partitionSizeM, numMFMATilesM, "M", mn_M);
    _partitionSizesN =
        normalize_partition_sizes(partitionSizeN, numMFMATilesN, "N", mn_N);
    _prefixM = build_prefix(_partitionSizesM);
    _prefixN = build_prefix(_partitionSizesN);
    plr = (pgr == 0) ? 0 : 1;
    offsetPartition = (pgr >= 2) ? 1 : 0;
    if (pgr == 0 && numPartitions() != 1) {
      throw std::invalid_argument("pgr=0 requires numPartitions=1");
    }
  }

  const std::vector<int>& partitionSizesM() const { return _partitionSizesM; }
  const std::vector<int>& partitionSizesN() const { return _partitionSizesN; }
  bool hasScale() const { return lrSA.has_value() && lrSB.has_value(); }
  int numPartitionsM() const {
    return static_cast<int>(_partitionSizesM.size());
  }
  int numPartitionsN() const {
    return static_cast<int>(_partitionSizesN.size());
  }
  int numPartitions() const { return numPartitionsM() * numPartitionsN(); }

  // Return partition candidates as [(partitionSizeM, partitionSizeN), ...].
  // Mirrors Python SchedulerConfig.get_partition_candidates, taking the two
  // localMMATileGrid[0] values (M, N) directly so the C++ stays Python-free.
  static std::vector<std::pair<int, int>> get_partition_candidates(int M,
                                                                   int N) {
    auto divUp = [](int n, int d) { return (n + d - 1) / d; };
    auto partitionSizes = [&](int dim) {
      std::vector<int> out = {dim};
      for (int s = divUp(dim, 2); s > 0; --s) out.push_back(s);
      return out;
    };
    std::vector<std::pair<int, int>> candidates;
    if (N >= M) {
      for (int s : partitionSizes(N)) candidates.emplace_back(M, s);
    } else {
      for (int s : partitionSizes(M)) candidates.emplace_back(s, N);
    }
    return candidates;
  }
};

// ── Schedule operation / before-chain op value types ─────────
//
// The op value types are defined before the placement types so the placements
// can hold their pass-populated before-chains (preOps / postOps) by value.
// Only the *identity* fields plus `kind` and `str()` are modeled.

inline std::string ljust2(const std::string& s) {
  return s.size() >= 2 ? s : s + std::string(2 - s.size(), ' ');
}

// Per-tensor inflight load counts for the wait_gr preOp.
struct WaitGRCounts {
  int A = 0;
  int B = 0;
  int SA = 0;
  int SB = 0;

  WaitGRCounts() = default;
  WaitGRCounts(int a, int b, int sa, int sb) : A(a), B(b), SA(sa), SB(sb) {}

  std::string str() const {
    std::vector<std::pair<const char*, int>> fields = {
        {"A", A}, {"B", B}, {"SA", SA}, {"SB", SB}};
    std::string out;
    for (auto& f : fields) {
      if (f.second) {
        if (!out.empty()) out += ",";
        out += std::string(f.first) + "=" + std::to_string(f.second);
      }
    }
    return out.empty() ? "0" : out;
  }
};

// Wait for global reads to complete. Optionally includes a sync barrier.
struct WaitGROp {
  std::optional<WaitGRCounts> wait_gr_counts;
  bool has_sync = false;
  bool adjustVmcnt = true;
  std::string kind = "wait_gr";

  WaitGROp() = default;
  WaitGROp(std::optional<WaitGRCounts> c, bool sync, bool adjust)
      : wait_gr_counts(std::move(c)), has_sync(sync), adjustVmcnt(adjust) {}

  std::string str() const {
    if (wait_gr_counts.has_value()) {
      return kind + "(" + wait_gr_counts->str() + ")";
    }
    return kind;
  }
};

// Wait for local reads to complete. Optionally includes a sync barrier.
struct WaitLROp {
  bool has_sync = false;
  std::string kind = "wait_lr";

  WaitLROp() = default;
  explicit WaitLROp(bool sync) : has_sync(sync) {}

  std::string str() const { return has_sync ? "wait_lr_sync" : "wait_lr"; }
};

// Standalone sync barrier.
struct SyncOp {
  std::string kind = "sync";
  std::string str() const { return kind; }
};

// Zero A/B vgprs whose K-index >= remaining tail K for one subIterK group.
struct MaskKOp {
  int subIterK = 0;
  // {tileId: vgprTileId}; populated by the (Python) emit pass.
  std::map<int, int> vgpr_tile_map;
  std::string kind = "mask_k";

  MaskKOp() = default;
  explicit MaskKOp(int sk) : subIterK(sk) {}

  std::string str() const { return "mask_k(k=" + std::to_string(subIterK) + ")"; }
};

// LDS buffer swap for local reads on a specific tensor.
struct LRIncOp {
  std::string tensor;
  std::string kind = "lr_inc";

  LRIncOp() = default;
  explicit LRIncOp(std::string t) : tensor(std::move(t)) {}

  std::string str() const { return "lr_inc(" + tensor + ")"; }
};

// Pointer update + LDS swap for global reads on a specific tensor.
struct GRIncOp {
  std::string tensor;
  std::string kind = "gr_inc";

  GRIncOp() = default;
  explicit GRIncOp(std::string t) : tensor(std::move(t)) {}

  std::string str() const { return "gr_inc(" + tensor + ")"; }
};

// Skip guard: compare LoopCounter and branch.
struct SkipOp {
  std::string compare;
  int value = 0;
  std::string target;
  bool rawLabel = false;
  std::string branchComment;
  std::string kind = "skip";

  SkipOp() = default;
  SkipOp(std::string cmp, int val, std::string tgt, bool raw,
         std::string comment)
      : compare(std::move(cmp)),
        value(val),
        target(std::move(tgt)),
        rawLabel(raw),
        branchComment(std::move(comment)) {}

  std::string tensor() const {
    return compare + ":" + std::to_string(value) + ":" + target;
  }

  std::string str() const { return "skip(" + tensor() + ")"; }
};

// Inline a writer-built Module at this point in the schedule.
//
// The Python dataclass also carries a `build` Callable that the emit pass
// invokes to produce a rocisa Module. That callback is Python/rocisa state and
// cannot live in this Python-free header, so only the identity fields used for
// `kind` / `str()` are ported here.
struct InlineModuleOp {
  std::string label = "inline";
  std::string kind = "inline";

  InlineModuleOp() = default;
  explicit InlineModuleOp(std::string l) : label(std::move(l)) {}

  std::string str() const { return "inline(" + label + ")"; }
};

// A typed op that can appear in a placement's before-chain (preOps / postOps).
// Mirrors the Python `BaseOp` subclass hierarchy.
using BeforeOp = std::variant<WaitGROp, WaitLROp, SyncOp, MaskKOp, LRIncOp,
                              GRIncOp, SkipOp, InlineModuleOp>;

// ── Placement value types ───────────────────────────────────
//
// The identity fields mirror the Python dataclasses. The pass-populated list
// fields (deps / preOps / postOps / vgpr_tile_map[s]) default to empty: the
// scheduling passes that fill them remain in Python, but the fields are ported
// so the value types are structurally faithful and round-trip through bindings.

struct Dep;  // defined below (references an LR/GR placement)

// MFMA operation consuming data for one subIterK.
struct MFMAPlacement {
  int subIterK = 0;
  MFMATileRange tileA;
  MFMATileRange tileB;
  std::vector<Dep> deps;          // populated by annotate_deps()
  std::vector<BeforeOp> preOps;   // populated by remove_cross_deps()
  std::vector<BeforeOp> postOps;  // populated by insert_gr_lr_inc()
  // {tensor: [{groupIdx: vgprTileId}]} per unroll iter.
  std::map<std::string, std::vector<std::map<int, int>>> vgpr_tile_maps;
  std::string kind = "mfma";

  MFMAPlacement() = default;
  MFMAPlacement(int sk, MFMATileRange a, MFMATileRange b)
      : subIterK(sk), tileA(std::move(a)), tileB(std::move(b)) {}

  std::string str() const {
    return "MFMAs (MT n, subIterK " + std::to_string(subIterK) + "  ) A : " +
           tileA.fmt_tiles() + " , B : " + tileB.fmt_tiles();
  }
};

// Local Read placement for one tensor in one subIterK slot.
struct LRPlacement {
  std::string tensor;     // 'A', 'B', 'SA', 'SB'
  int mtIteration = 0;    // 0 = current MT, 1 = next MT
  MFMATileRange tiles;
  int subIterK_slot = 0;
  int partition = 0;
  std::vector<Dep> deps;          // populated by annotate_deps()
  std::vector<BeforeOp> preOps;   // populated by remove_cross_deps()
  std::vector<BeforeOp> postOps;  // populated by insert_gr_lr_inc()
  // [{tileId: vgprTileId}] per unroll iter.
  std::vector<std::map<int, int>> vgpr_tile_map;
  std::string kind = "lr";

  LRPlacement() = default;
  LRPlacement(std::string t, int mt, MFMATileRange tr, int slot, int part = 0)
      : tensor(std::move(t)),
        mtIteration(mt),
        tiles(std::move(tr)),
        subIterK_slot(slot),
        partition(part) {}

  std::string str() const {
    return "LR " + ljust2(tensor) + " (MT " + fmt_mt(mtIteration) +
           ", subIterK " + tiles.fmt_k() + ") " + tiles.fmt_tiles();
  }
};

// Global Read placement for one tensor in one subIterK slot.
struct GRPlacement {
  std::string tensor;     // 'A', 'B', 'SA', 'SB'
  int mtIteration = 0;    // 0/1/2 MTs ahead
  MFMATileRange tiles;
  int subIterK_slot = 0;
  int partition = 0;
  std::vector<Dep> deps;          // populated by annotate_deps()
  std::vector<BeforeOp> preOps;   // populated by remove_cross_deps()
  std::vector<BeforeOp> postOps;  // populated by insert_gr_lr_inc()
  std::string kind = "gr";

  GRPlacement() = default;
  GRPlacement(std::string t, int mt, MFMATileRange tr, int slot, int part = 0)
      : tensor(std::move(t)),
        mtIteration(mt),
        tiles(std::move(tr)),
        subIterK_slot(slot),
        partition(part) {}

  std::string str() const {
    return "GR " + tensor + " (MT " + fmt_mt(mtIteration) + ", subIterK " +
           tiles.fmt_k() + ") ids " + tiles.fmt_tiles();
  }
};

// ── Dependency / slot / emitted-module value types ──────────

// Dependency on another placement (annotate_deps output). `ref` is a value
// copy of the referenced LR/GR placement; Python uses an identity reference,
// the C++ value layer mirrors the data rather than Python object identity.
struct Dep {
  std::variant<LRPlacement, GRPlacement> ref;
  int mt_offset = 0;  // 0 = same MT, -1 = prev MT, -2 = two MTs back, ...

  Dep() = default;
  Dep(std::variant<LRPlacement, GRPlacement> r, int off = 0)
      : ref(std::move(r)), mt_offset(off) {}
};

// All operations placed in one subIterK step.
struct SubIterKSlot {
  int subIterK = 0;
  std::optional<MFMAPlacement> mfma;
  std::vector<LRPlacement> lrs;
  std::vector<GRPlacement> grs;

  SubIterKSlot() = default;
  explicit SubIterKSlot(int sk) : subIterK(sk) {}
};

// Anything that can be the `source` of an EmittedModule (Python `Emittable`):
// a placement or a before-chain op.
using Emittable =
    std::variant<MFMAPlacement, LRPlacement, GRPlacement, WaitGROp, WaitLROp,
                 SyncOp, MaskKOp, LRIncOp, GRIncOp, SkipOp, InlineModuleOp>;

inline std::string emittable_kind(const Emittable& e) {
  return std::visit([](const auto& v) { return v.kind; }, e);
}

inline std::string emittable_str(const Emittable& e) {
  return std::visit([](const auto& v) { return v.str(); }, e);
}

// One emitted module with before-link for instruction scheduling.
//
// `instructions` are rocisa objects filled during (Python) emission; they are
// not modeled in this Python-free header. Only the structural fields and the
// `opType` accessor are ported. Overlaps insched::ModuleRef by design.
struct EmittedModule {
  int moduleId = -1;
  std::optional<int> before;  // moduleId that must complete before this module
  std::optional<Emittable> source;

  EmittedModule() = default;
  EmittedModule(int id, std::optional<int> before_,
                std::optional<Emittable> src)
      : moduleId(id), before(std::move(before_)), source(std::move(src)) {}

  std::string opType() const {
    return source.has_value() ? emittable_kind(*source) : std::string();
  }
};

}  // namespace tw::subtile::lsched
