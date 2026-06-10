// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Pure C++ port of the writer-free *pass pipeline* of the subtile
// LogicalScheduler (Tensile/Components/Subtile/LogicalScheduler.py).
//
// This header builds on logical_scheduler.hpp (the value/config layer) and
// adds the scheduling passes that operate purely on the in-memory data model:
//
//   place_LRs / assign_vgpr_tiles / place_GRs / annotate_deps /
//   remove_unnecessary_gr_deps / remove_unnecessary_lr_deps /
//   remove_cross_deps / insert_gr_lr_inc / group_lr_gr /
//   remove_unnecessary_wait_lr_sync / emit / build
//
// It is intentionally free of any nanobind / Python / rocisa dependency: it
// ports only the data-only logical schedule (placements, deps, before-chain
// ops, and the EmittedModule before-link graph). It does NOT populate rocisa
// instructions, allocate writer VGPR pools, or emit Kernel.mainLoop control
// flow — those remain in Python (populate_instructions, emitMainAndExitLoops,
// emitTailLoop, allocVgprTiles, InstructionEmitter).
//
// The pass logic mirrors the Python module line-for-line so the two
// implementations can be reasoned about side by side, and the print_* methods
// produce byte-identical output to the Python print_* helpers for pass-by-pass
// parity testing.

#pragma once

#include <algorithm>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "tensile_writer/logical_scheduler.hpp"

// The pass pipeline lives in a nested namespace so its identity-based working
// types (Dep, Placement, Slot, EmittedModule) do not collide with the
// value-copy types of the same name in tw::subtile::lsched. Names from the
// enclosing namespace (MFMATileRange, ReadGranularity, SchedulerConfig,
// WaitGRCounts, Pass, fmt_mt, ljust2) resolve via enclosing-namespace lookup.
namespace tw::subtile::lsched::passes {

// ── Small formatting helpers ────────────────────────────────

inline std::string ljust(const std::string& s, std::size_t w) {
  return s.size() >= w ? s : s + std::string(w - s.size(), ' ');
}

inline std::string rjust(const std::string& s, std::size_t w) {
  return s.size() >= w ? s : std::string(w - s.size(), ' ') + s;
}

// Python dict repr for an ordered {int: int} map (ascending keys == the
// insertion order used by assign_vgpr_tiles).
inline std::string dict_repr(const std::map<int, int>& m) {
  std::string out = "{";
  bool first = true;
  for (const auto& kv : m) {
    if (!first) out += ", ";
    first = false;
    out += std::to_string(kv.first) + ": " + std::to_string(kv.second);
  }
  out += "}";
  return out;
}

inline const char* tensor_side(const std::string& t) {
  return (t == "A" || t == "SA") ? "A" : "B";
}

// ── Before-chain op (data-only BaseOp union) ────────────────
//
// One value type carrying all BaseOp variants used by the passes. `kind`
// matches the Python op kinds and drives str().
struct Op {
  std::string kind;
  // wait_gr
  bool hasCounts = false;
  WaitGRCounts counts;
  bool adjustVmcnt = true;
  // wait_gr / wait_lr
  bool has_sync = false;
  // mask_k
  int subIterK = 0;
  // lr_inc / gr_inc
  std::string tensor;

  static Op waitGR(std::optional<WaitGRCounts> c, bool sync, bool adjust) {
    Op o;
    o.kind = "wait_gr";
    o.hasCounts = c.has_value();
    if (c.has_value()) o.counts = *c;
    o.has_sync = sync;
    o.adjustVmcnt = adjust;
    return o;
  }
  static Op waitLR(bool sync) {
    Op o;
    o.kind = "wait_lr";
    o.has_sync = sync;
    return o;
  }
  static Op sync() {
    Op o;
    o.kind = "sync";
    return o;
  }
  static Op maskK(int k) {
    Op o;
    o.kind = "mask_k";
    o.subIterK = k;
    return o;
  }
  static Op lrInc(std::string t) {
    Op o;
    o.kind = "lr_inc";
    o.tensor = std::move(t);
    return o;
  }
  static Op grInc(std::string t) {
    Op o;
    o.kind = "gr_inc";
    o.tensor = std::move(t);
    return o;
  }

  std::string str() const {
    if (kind == "wait_gr") {
      return hasCounts ? ("wait_gr(" + counts.str() + ")") : std::string("wait_gr");
    }
    if (kind == "wait_lr") return has_sync ? "wait_lr_sync" : "wait_lr";
    if (kind == "sync") return "sync";
    if (kind == "mask_k") return "mask_k(k=" + std::to_string(subIterK) + ")";
    if (kind == "lr_inc") return "lr_inc(" + tensor + ")";
    if (kind == "gr_inc") return "gr_inc(" + tensor + ")";
    return kind;
  }
};

// ── Unified placement (MFMA / LR / GR) ──────────────────────

enum class PKind { MFMA, LR, GR };

struct Placement;

// Dependency on another placement (annotate_deps output).
struct Dep {
  Placement* ref = nullptr;
  int mt_offset = 0;  // 0 = same MT, -1 = prev MT, ...
};

struct Placement {
  PKind kind = PKind::MFMA;

  // common / identity
  int partition = 0;
  int subIterK_slot = 0;

  // MFMA-specific
  int subIterK = 0;
  MFMATileRange tileA;
  MFMATileRange tileB;

  // LR / GR-specific
  std::string tensor;
  int mtIteration = 0;
  MFMATileRange tiles;

  // pass-populated
  std::vector<Dep> deps;
  std::vector<Op> preOps;
  std::vector<Op> postOps;

  // vgpr maps
  std::map<std::string, std::vector<std::map<int, int>>> vgpr_tile_maps;  // MFMA
  std::vector<std::map<int, int>> vgpr_tile_map;                          // LR

  std::string kindStr() const {
    switch (kind) {
      case PKind::MFMA: return "mfma";
      case PKind::LR: return "lr";
      default: return "gr";
    }
  }

  std::string str() const {
    if (kind == PKind::MFMA) {
      return "MFMAs (MT n, subIterK " + std::to_string(subIterK) + "  ) A : " +
             tileA.fmt_tiles() + " , B : " + tileB.fmt_tiles();
    }
    if (kind == PKind::LR) {
      return "LR " + ljust2(tensor) + " (MT " + fmt_mt(mtIteration) +
             ", subIterK " + tiles.fmt_k() + ") " + tiles.fmt_tiles();
    }
    return "GR " + tensor + " (MT " + fmt_mt(mtIteration) + ", subIterK " +
           tiles.fmt_k() + ") ids " + tiles.fmt_tiles();
  }
};

// One subIterK step container.
struct CSlot {
  int subIterK = 0;
  Placement* mfma = nullptr;
  std::vector<Placement*> lrs;
  std::vector<Placement*> grs;
};

// ── The scheduler ───────────────────────────────────────────

class LogicalScheduler {
 public:
  explicit LogicalScheduler(SchedulerConfig cfg) : config(std::move(cfg)) {
    tensors = {"A", "B"};
    if (config.hasScale()) {
      tensors.push_back("SA");
      tensors.push_back("SB");
    }
  }

  // ── Pass driver ───────────────────────────────────────────
  void ensure(Pass p) {
    if (!completed.count(static_cast<int>(p))) runPass(p);
  }

  void runPass(Pass p) {
    switch (p) {
      case Pass::LR: place_LRs(); break;
      case Pass::VGPR_TILES: assign_vgpr_tiles(); break;
      case Pass::GR: place_GRs(); break;
      case Pass::DEPS: annotate_deps(); break;
      case Pass::REMOVE_GR_DEPS: remove_unnecessary_gr_deps(); break;
      case Pass::REMOVE_LR_DEPS: remove_unnecessary_lr_deps(); break;
      case Pass::REMOVE_DEPS: remove_cross_deps(); break;
      case Pass::GR_INC: insert_gr_lr_inc(); break;
      case Pass::GROUP_LR_GR: group_lr_gr(); break;
      case Pass::REMOVE_WAIT_LR_SYNC: remove_unnecessary_wait_lr_sync(); break;
      case Pass::EMIT: emit(); break;
      case Pass::BUILD: build(); break;
      default: break;
    }
  }

  // ── Helpers ──────────────────────────────────────────────
  struct PartRange {
    std::pair<int, int> A;
    std::pair<int, int> B;
    const std::pair<int, int>& by_side(const char* s) const {
      return s[0] == 'A' ? A : B;
    }
  };

  PartRange partition_tile_range(int pi) const {
    int piM = pi % config.numPartitionsM();
    int piN = pi / config.numPartitionsM();
    PartRange r;
    r.A = {config._prefixM[piM], config._prefixM[piM + 1]};
    r.B = {config._prefixN[piN], config._prefixN[piN + 1]};
    return r;
  }

  Placement* alloc(Placement p) {
    pool.push_back(std::move(p));
    return &pool.back();
  }

  std::vector<std::pair<std::string, ReadGranularity>> lr_tensors() const {
    std::vector<std::pair<std::string, ReadGranularity>> v;
    v.emplace_back("A", config.lrA);
    v.emplace_back("B", config.lrB);
    if (config.hasScale()) {
      v.emplace_back("SA", *config.lrSA);
      v.emplace_back("SB", *config.lrSB);
    }
    return v;
  }

  ReadGranularity gr_granularity(const std::string& t) const {
    if (t == "A") return config.grA;
    if (t == "B") return config.grB;
    if (t == "SA") return *config.grSA;
    return *config.grSB;
  }

  // ── Pass: place_LRs ──────────────────────────────────────
  void place_LRs();
  std::vector<CSlot> create_partition_slots(const PartRange& cur);
  void place_LRs_PLR0();
  std::vector<CSlot> place_LRs_for_partition(const PartRange& cur,
                                             const PartRange& nxt, bool is_last,
                                             const std::map<std::string, bool>& load,
                                             std::set<std::tuple<std::string, int, int, int, int>>& placed);

  // ── Pass: assign_vgpr_tiles ──────────────────────────────
  void assign_vgpr_tiles();

  // ── Pass: place_GRs ──────────────────────────────────────
  struct GREntry {
    std::string tensor;
    int mt;
    int ts, te, ks, ke;
    ReadGranularity gran;
  };
  std::vector<GREntry> build_gr_list(const std::vector<PartRange>& part_ranges,
                                     int offsetMT, int offsetPartition);
  void build_gr_slot_bounds(
      std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>>& lower,
      std::map<std::pair<std::string, int>, int>& upper);
  static bool has_lr_conflict(
      const std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>>& lower,
      const std::string& tensor, int mt_val, int pi, int subIterK, int gr_k_start,
      int gr_k_end);
  void distribute_grs(const std::vector<GREntry>& gr_list,
                      const std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>>& lower,
                      const std::map<std::pair<std::string, int>, int>& upper);
  void place_GRs();

  // ── Pass: annotate_deps ──────────────────────────────────
  void annotate_deps();
  void annotate_deps_partition(int pi, std::vector<CSlot>& slots,
                               std::map<std::string, std::vector<Placement*>>& gr_by_tensor,
                               std::map<std::string, std::vector<Placement*>>& lr_by_tensor);

  // ── Pass: remove_* dep passes ────────────────────────────
  void remove_unnecessary_gr_deps();
  void remove_unnecessary_lr_deps();
  int count_gr_atoms(const Placement& gr) const;
  WaitGRCounts compute_inflight_loads(int consumer_pi, int consumer_slot,
                                      const std::string& tensor, const Dep& dep_ref);
  void remove_cross_deps();

  // ── Pass: insert_gr_lr_inc ───────────────────────────────
  void insert_gr_lr_inc();

  // ── Pass: group_lr_gr ────────────────────────────────────
  std::vector<Op> merge_preops(const std::vector<std::vector<Op>*>& all_preops) const;
  void group_lr_gr();
  void remove_unnecessary_wait_lr_sync();

  // ── Pass: emit / build ───────────────────────────────────
  void emit();
  void build();

  // ── Loop-variant builders (NGLL / NLL / preloop / tail-PGR0) ─────────
  //
  // These port the Python LogicalScheduler.build_ngll / build_nll /
  // build_preloop / build_tailloop_pgr0 schedule rewrites. NGLL/NLL are pure
  // transforms over the mainloop `emitted` before-link graph (coordinate-only
  // sources, like emit()); the Python wrapper rebuilds them via the same
  // placement registry it uses for emit(). preloop / tail synthesize fresh
  // placements (and their vgpr tile maps), so their value sources carry full
  // placement data the Python wrapper rebuilds directly.
  using ModuleGrid =
      std::vector<std::vector<std::vector<tw::subtile::lsched::EmittedModule>>>;
  // {tensor: [{groupKey: flat_tile_id}]} per partition — the flat tail tile
  // layout computed by the (Python) _compute_flat_tail_tile_state and passed in.
  using FlatTileMaps =
      std::vector<std::map<std::string, std::vector<std::map<int, int>>>>;

  static std::vector<tw::subtile::lsched::EmittedModule> rewire_before(
      std::vector<tw::subtile::lsched::EmittedModule> mods,
      const std::set<int>& removed);

  void build_ngll();
  void build_nll();
  void build_preloop();
  void build_tailloop_pgr0(const FlatTileMaps& tile_maps, bool bf16, int miK);

  // Loop-variant placement builders (mirror the Python _make_* helpers).
  std::vector<tw::subtile::lsched::Emittable> make_gr_all_tensors(
      int mt, const MFMATileRange& tilesA, const MFMATileRange& tilesB) const;
  std::vector<tw::subtile::lsched::Emittable> make_lr_all_tensors(
      const std::map<std::string, MFMATileRange>& lr_tiles) const;
  std::vector<tw::subtile::lsched::Emittable> make_preloop_mt1_grs();

  const ModuleGrid& value_ngll() const { return ngll_emitted; }
  const ModuleGrid& value_nll() const { return nll_emitted; }
  const ModuleGrid& value_preloop() const { return preloop_emitted; }
  const ModuleGrid& value_tailloop() const { return tailloop_emitted; }

  // Export the emitted before-link graph as bound value EmittedModules
  // (list[partition][subIterK][EmittedModule]) so the Python writer can rebuild
  // its EmittedModule dataclasses. Each module's `source` is a coordinate-only
  // value Emittable: the Python converter re-uses its persistent placement
  // dataclasses (by coordinate match) for placement sources and constructs
  // fresh op dataclasses for before-chain ops, preserving placement identity so
  // assign_vgpr_tiles results flow through to emission.
  std::vector<std::vector<std::vector<tw::subtile::lsched::EmittedModule>>>
  value_emitted() const {
    return emitted;
  }

  // ── Value-object export (for the Python writer integration) ──
  //
  // Materialize the internal pointer-based `partitions` model as the
  // nanobind-bound value types of the enclosing lsched namespace
  // (SubIterKSlot / MFMAPlacement / LRPlacement / GRPlacement carrying
  // value Dep / before-chain ops / vgpr tile maps). Deps are emitted as
  // value Deps whose `ref` is a coordinate-only copy of the referenced
  // LR/GR placement: the Python converter re-establishes object identity by
  // matching those coordinates against its persistent dataclass placements.
  std::vector<std::vector<tw::subtile::lsched::SubIterKSlot>> value_partitions()
      const;

  // ── Dep / sort helpers ───────────────────────────────────
  static int tensor_order(const std::string& t) {
    if (t == "A") return 0;
    if (t == "B") return 1;
    if (t == "SA") return 2;
    return 3;
  }
  // _gr_sort_key
  static std::tuple<int, int, int, int> gr_sort_key(const Placement* gr) {
    return {gr->mtIteration, gr->tiles.subIterK_start, tensor_order(gr->tensor),
            gr->tiles.tileId_start};
  }
  std::pair<std::vector<Dep>, std::vector<Dep>> split_deps(
      const std::vector<Dep>& deps, int consumer_pi, int consumer_slot) const {
    std::vector<Dep> same, cross;
    for (const auto& dep : deps) {
      if (dep.mt_offset == 0 && dep.ref->partition == consumer_pi &&
          dep.ref->subIterK_slot == consumer_slot) {
        same.push_back(dep);
      } else {
        cross.push_back(dep);
      }
    }
    return {same, cross};
  }

  // ── Print helpers (byte-identical to Python) ─────────────
  std::string print_lr() const;
  std::string print_vgpr() const;
  std::string print_gr() const;
  std::string print_deps() const;
  std::string print_remove_deps() const;
  std::string print_group_lr_gr() const;
  std::string print_emit() const;

 private:
  void print_placement_with_deps(std::ostringstream& buf, const Placement* p) const;
  void print_placement_with_preops(std::ostringstream& buf, const Placement* p) const;
  std::string format_dep_ref(const Dep& dep) const;

 public:
  SchedulerConfig config;
  std::vector<std::string> tensors;
  std::set<int> completed;
  std::deque<Placement> pool;
  std::vector<std::vector<CSlot>> partitions;

  // vgpr results
  std::map<std::string, int> tile_peaks;
  int unroll_factor = 1;
  bool needs_unrolling = false;
  bool vgpr_done = false;

  // emit results (value EmittedModules carrying coordinate-only sources)
  std::vector<std::vector<std::vector<tw::subtile::lsched::EmittedModule>>>
      emitted;

  // Loop-variant results ([partition][subIterK/group][EmittedModule]).
  ModuleGrid ngll_emitted;
  ModuleGrid nll_emitted;
  ModuleGrid preloop_emitted;
  ModuleGrid tailloop_emitted;
};

// ════════════════════════════════════════════════════════════
// place_LRs
// ════════════════════════════════════════════════════════════

inline std::vector<CSlot> LogicalScheduler::create_partition_slots(
    const PartRange& cur) {
  int numK = config.numSubIterK;
  std::vector<CSlot> slots;
  slots.reserve(numK);
  for (int k = 0; k < numK; ++k) {
    CSlot s;
    s.subIterK = k;
    Placement m;
    m.kind = PKind::MFMA;
    m.subIterK = k;
    m.subIterK_slot = k;
    m.tileA = MFMATileRange(k, k + 1, cur.A.first, cur.A.second);
    m.tileB = MFMATileRange(k, k + 1, cur.B.first, cur.B.second);
    s.mfma = alloc(std::move(m));
    slots.push_back(std::move(s));
  }
  return slots;
}

inline void LogicalScheduler::place_LRs_PLR0() {
  int numK = config.numSubIterK;
  PartRange cur = partition_tile_range(0);
  std::vector<CSlot> slots = create_partition_slots(cur);

  for (auto& [tensor, gran] : lr_tensors()) {
    const char* side_key = tensor_side(tensor);
    auto [ts, te] = cur.by_side(side_key);
    int k_gran = gran.k;
    int num_chunks = numK / k_gran;
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      int lr_k_start = chunk_idx * k_gran;
      int lr_k_end = lr_k_start + k_gran;
      int slot_k = lr_k_start;
      Placement lr;
      lr.kind = PKind::LR;
      lr.tensor = tensor;
      lr.mtIteration = 0;
      lr.tiles = MFMATileRange(lr_k_start, lr_k_end, ts, te);
      lr.subIterK_slot = slot_k;
      slots[slot_k].lrs.push_back(alloc(std::move(lr)));
    }
  }

  partitions.clear();
  partitions.push_back(std::move(slots));
  completed.insert(static_cast<int>(Pass::LR));
}

inline std::vector<CSlot> LogicalScheduler::place_LRs_for_partition(
    const PartRange& cur, const PartRange& nxt, bool is_last,
    const std::map<std::string, bool>& load,
    std::set<std::tuple<std::string, int, int, int, int>>& placed) {
  int numK = config.numSubIterK;
  bool multi_part = config.numPartitions() > 1;

  std::vector<CSlot> slots = create_partition_slots(cur);
  std::map<int, int> slot_mt;  // slot_k -> lr_mt

  auto all_tensors = lr_tensors();

  // unique k grans ascending
  std::set<int> kset;
  for (auto& [t, g] : all_tensors) kset.insert(g.k);

  for (int k_gran : kset) {
    std::vector<std::pair<std::string, ReadGranularity>> group_all;
    for (auto& [t, g] : all_tensors)
      if (g.k == k_gran) group_all.push_back({t, g});
    int num_chunks = numK / k_gran;
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      int next_chunk = (chunk_idx + 1) % num_chunks;
      bool is_wrap = (next_chunk == 0);
      int lr_mt = (is_last && is_wrap) ? 1 : 0;
      int lr_k_start = next_chunk * k_gran;
      int lr_k_end = lr_k_start + k_gran;
      int base_slot = chunk_idx * k_gran;

      std::vector<std::pair<std::string, ReadGranularity>> group;
      if (is_wrap && multi_part) {
        for (auto& [t, g] : group_all) {
          bool keep = (t == "A" || t == "B") ||
                      load.at(tensor_side(t));
          if (keep) group.push_back({t, g});
        }
      } else {
        group = group_all;
      }

      // group by side (A/SA, B/SB)
      std::vector<std::vector<std::pair<std::string, ReadGranularity>>> sides;
      std::vector<std::pair<std::string, ReadGranularity>> sideA, sideB;
      for (auto& [t, g] : group) {
        if (t == "A" || t == "SA") sideA.push_back({t, g});
        else sideB.push_back({t, g});
      }
      if (!sideA.empty()) sides.push_back(sideA);
      if (!sideB.empty()) sides.push_back(sideB);

      for (int side_idx = 0; side_idx < (int)sides.size(); ++side_idx) {
        auto& side = sides[side_idx];
        int slot_k = base_slot + (side_idx % k_gran);
        auto it = slot_mt.find(slot_k);
        if (it != slot_mt.end() && it->second != lr_mt) {
          slot_k = numK - 1;
        }

        for (auto& [tensor, gran] : side) {
          const PartRange& tile_range = (is_wrap || !multi_part) ? nxt : cur;
          const char* side_key = tensor_side(tensor);
          auto [ts, te] = tile_range.by_side(side_key);

          if (is_wrap && multi_part) {
            if (!load.at(side_key)) continue;
          } else {
            auto lr_key = std::make_tuple(tensor, lr_k_start, lr_k_end, ts, te);
            if (placed.count(lr_key)) continue;
            placed.insert(lr_key);
          }

          Placement lr;
          lr.kind = PKind::LR;
          lr.tensor = tensor;
          lr.mtIteration = lr_mt;
          lr.tiles = MFMATileRange(lr_k_start, lr_k_end, ts, te);
          lr.subIterK_slot = slot_k;
          slots[slot_k].lrs.push_back(alloc(std::move(lr)));
          slot_mt[slot_k] = lr_mt;
        }
      }
    }
  }

  return slots;
}

inline void LogicalScheduler::place_LRs() {
  if (config.plr == 0) {
    place_LRs_PLR0();
    return;
  }

  int numP = config.numPartitions();
  std::vector<PartRange> part_ranges;
  for (int pi = 0; pi < numP; ++pi) part_ranges.push_back(partition_tile_range(pi));

  std::map<std::string, std::set<std::pair<int, int>>> loaded_ranges;
  loaded_ranges["A"].insert(part_ranges[0].A);
  loaded_ranges["B"].insert(part_ranges[0].B);

  std::set<std::tuple<std::string, int, int, int, int>> placed;

  partitions.clear();
  for (int pi = 0; pi < numP; ++pi) {
    const PartRange& cur = part_ranges[pi];
    const PartRange& nxt = part_ranges[(pi + 1) % numP];
    bool is_last = (pi == numP - 1);

    std::map<std::string, bool> load;
    for (const char* side : {"A", "B"}) {
      auto rng = (side[0] == 'A') ? nxt.A : nxt.B;
      load[side] = is_last || !loaded_ranges[side].count(rng);
    }

    std::vector<CSlot> slots = place_LRs_for_partition(cur, nxt, is_last, load, placed);
    for (auto& slot : slots)
      for (auto* lr : slot.lrs) lr->partition = pi;
    partitions.push_back(std::move(slots));

    for (const char* side : {"A", "B"}) {
      if (load[side]) {
        auto curr = (side[0] == 'A') ? cur.A : cur.B;
        auto nxtr = (side[0] == 'A') ? nxt.A : nxt.B;
        loaded_ranges[side] = {curr, nxtr};
      }
    }
  }

  completed.insert(static_cast<int>(Pass::LR));
}

// ════════════════════════════════════════════════════════════
// assign_vgpr_tiles
// ════════════════════════════════════════════════════════════

inline void LogicalScheduler::assign_vgpr_tiles() {
  ensure(Pass::LR);

  int numK = config.numSubIterK;
  int numP = config.numPartitions();

  std::map<std::string, ReadGranularity> lr_grans;
  lr_grans["A"] = config.lrA;
  lr_grans["B"] = config.lrB;
  if (config.hasScale()) {
    lr_grans["SA"] = *config.lrSA;
    lr_grans["SB"] = *config.lrSB;
  }

  std::vector<PartRange> part_ranges;
  for (int pi = 0; pi < numP; ++pi) part_ranges.push_back(partition_tile_range(pi));

  bool any_single_k_chunk = false;
  for (auto& t : tensors)
    if (numK / lr_grans[t].k == 1) any_single_k_chunk = true;
  bool use_global_pos = (numP > 1) && any_single_k_chunk;

  std::map<std::string, std::map<int, int>> group_to_pos;
  std::map<std::string, int> max_groups;
  for (auto& t : tensors) max_groups[t] = 0;

  if (use_global_pos) {
    for (int pi = 0; pi < numP; ++pi) {
      for (auto& tensor : tensors) {
        const char* side = tensor_side(tensor);
        auto [start, end] = part_ranges[pi].by_side(side);
        ReadGranularity gran = lr_grans[tensor];
        std::set<int> groups;
        for (int t = start; t < end; ++t) groups.insert((t / gran.mn) * gran.mn);
        for (int g : groups) {
          if (!group_to_pos[tensor].count(g)) {
            group_to_pos[tensor][g] = max_groups[tensor];
            max_groups[tensor] += 1;
          }
        }
      }
    }
  } else {
    for (int pi = 0; pi < numP; ++pi) {
      for (auto& tensor : tensors) {
        const char* side = tensor_side(tensor);
        auto [start, end] = part_ranges[pi].by_side(side);
        ReadGranularity gran = lr_grans[tensor];
        std::set<int> groups;
        for (int t = start; t < end; ++t) groups.insert((t / gran.mn) * gran.mn);
        int local_pos = 0;
        for (int g : groups) {
          if (!group_to_pos[tensor].count(g)) group_to_pos[tensor][g] = local_pos;
          local_pos += 1;
        }
        max_groups[tensor] = std::max(max_groups[tensor], local_pos);
      }
    }
  }

  std::map<std::string, int> num_k_groups;
  for (auto& tensor : tensors) num_k_groups[tensor] = numK / lr_grans[tensor].k;

  int uf = 1;
  for (auto& tensor : tensors)
    if (num_k_groups[tensor] % 2 != 0) {
      uf = 2;
      break;
    }
  bool pgr0 = config.pgr == 0;
  if (pgr0) uf = 1;

  for (int unroll_iter = 0; unroll_iter < uf; ++unroll_iter) {
    for (int pi = 0; pi < (int)partitions.size(); ++pi) {
      for (auto& slot : partitions[pi]) {
        int k = slot.subIterK;
        if (slot.mfma) {
          for (auto& tensor : tensors) {
            ReadGranularity gran = lr_grans[tensor];
            int nkg = num_k_groups[tensor];
            int set_idx = pgr0 ? 0 : ((unroll_iter * nkg + k / gran.k) % 2);
            const char* side = tensor_side(tensor);
            const MFMATileRange& tileRange =
                (side[0] == 'A') ? slot.mfma->tileA : slot.mfma->tileB;
            std::map<int, int> tile_map;
            for (int t : tileRange.tileId_list()) {
              int group = (t / gran.mn) * gran.mn;
              int pos = group_to_pos[tensor][group];
              tile_map[group] = set_idx * max_groups[tensor] + pos;
            }
            slot.mfma->vgpr_tile_maps[tensor].push_back(tile_map);
          }
        }
        for (auto* lr : slot.lrs) {
          const std::string& tensor = lr->tensor;
          ReadGranularity gran = lr_grans[tensor];
          int nkg = num_k_groups[tensor];
          int target_mt = unroll_iter + lr->mtIteration;
          int target_k = lr->tiles.subIterK_start;
          int set_idx = pgr0 ? 0 : ((target_mt * nkg + target_k / gran.k) % 2);
          std::map<int, int> tile_map;
          for (int t : lr->tiles.tileId_list()) {
            int group = (t / gran.mn) * gran.mn;
            if (tile_map.count(group)) continue;
            int pos = group_to_pos[tensor][group];
            tile_map[group] = set_idx * max_groups[tensor] + pos;
          }
          lr->vgpr_tile_map.push_back(tile_map);
        }
      }
    }
  }

  int num_sets = pgr0 ? 1 : 2;
  tile_peaks.clear();
  for (auto& t : tensors) tile_peaks[t] = num_sets * max_groups[t];
  unroll_factor = uf;
  needs_unrolling = uf > 1;
  vgpr_done = true;

  completed.insert(static_cast<int>(Pass::VGPR_TILES));
}

// ════════════════════════════════════════════════════════════
// place_GRs
// ════════════════════════════════════════════════════════════

inline std::vector<LogicalScheduler::GREntry> LogicalScheduler::build_gr_list(
    const std::vector<PartRange>& part_ranges, int offsetMT, int offsetPartition) {
  int numP = config.numPartitions();
  std::set<std::tuple<std::string, int, int, int, int, int>> seen;
  std::vector<GREntry> gr_list;

  for (int pi = 0; pi < numP; ++pi) {
    auto& partition_slots = partitions[pi];
    int target_pi = (pi + offsetPartition) % numP;
    bool wraps = (pi + offsetPartition) >= numP;
    int mt_val = offsetMT + (wraps ? 1 : 0);
    const PartRange& target_range = part_ranges[target_pi];

    for (auto& slot : partition_slots) {
      int k = slot.mfma->subIterK;
      struct Item {
        std::string tensor;
        std::pair<int, int> tr;
        ReadGranularity gran;
      };
      std::vector<Item> items;
      items.push_back({"A", target_range.A, config.grA});
      items.push_back({"B", target_range.B, config.grB});
      if (config.hasScale()) {
        items.push_back({"SA", target_range.A, *config.grSA});
        items.push_back({"SB", target_range.B, *config.grSB});
      }
      for (auto& it : items) {
        MFMATileRange tr = it.gran.tile_range(k, it.tr.first, it.tr.second);
        auto key = std::make_tuple(it.tensor, mt_val, tr.tileId_start,
                                   tr.tileId_end, tr.subIterK_start,
                                   tr.subIterK_end);
        if (seen.count(key)) continue;
        seen.insert(key);
        gr_list.push_back({it.tensor, mt_val, tr.tileId_start, tr.tileId_end,
                           tr.subIterK_start, tr.subIterK_end, it.gran});
      }
    }
  }

  int base_mt = offsetMT;
  std::set<std::tuple<std::string, int, int, int, int>> n2_keys;
  for (auto& e : gr_list)
    if (e.mt != base_mt)
      n2_keys.insert({e.tensor, e.ts, e.te, e.ks, e.ke});
  std::vector<GREntry> filtered;
  for (auto& e : gr_list) {
    if (e.mt != base_mt ||
        !n2_keys.count({e.tensor, e.ts, e.te, e.ks, e.ke})) {
      filtered.push_back(e);
    }
  }
  return filtered;
}

inline void LogicalScheduler::build_gr_slot_bounds(
    std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>>& lower,
    std::map<std::pair<std::string, int>, int>& upper) {
  int numK = config.numSubIterK;
  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    for (auto& slot : partitions[pi]) {
      int flat = pi * numK + slot.subIterK;
      for (auto* lr : slot.lrs) {
        if (lr->mtIteration == 0) {
          lower[{pi, lr->tensor}].push_back(
              {slot.subIterK, lr->tiles.subIterK_start, lr->tiles.subIterK_end});
        }
        auto key = std::make_pair(lr->tensor, lr->mtIteration);
        auto it = upper.find(key);
        if (it == upper.end() || flat < it->second) upper[key] = flat;
      }
    }
  }
}

inline bool LogicalScheduler::has_lr_conflict(
    const std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>>& lower,
    const std::string& tensor, int mt_val, int pi, int subIterK, int gr_k_start,
    int gr_k_end) {
  if (mt_val != 2) return false;
  auto it = lower.find({pi, tensor});
  if (it == lower.end()) return false;
  for (auto& [lr_slot, lr_ks, lr_ke] : it->second) {
    if (lr_slot > subIterK && gr_k_start < lr_ke && lr_ks < gr_k_end) return true;
  }
  return false;
}

inline void LogicalScheduler::distribute_grs(
    const std::vector<GREntry>& gr_list,
    const std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>>& lower,
    const std::map<std::pair<std::string, int>, int>& upper) {
  int numK = config.numSubIterK;
  int numP = config.numPartitions();
  int numSlots = numP * numK;

  // 2a. explode into atoms
  struct Atom {
    std::string tensor;
    int mt, ts, te, ks, ke, last;
  };
  std::vector<Atom> atoms;
  for (auto& e : gr_list) {
    int mn = e.gran.mn;
    int up = numSlots;
    auto it = upper.find({e.tensor, e.mt});
    if (it != upper.end()) up = it->second;
    int last = std::max(0, std::min(up - 1, numSlots - 1));
    for (int pos = e.ts; pos < e.te; pos += mn) {
      atoms.push_back({e.tensor, e.mt, pos, pos + mn, e.ks, e.ke, last});
    }
  }

  int nAtoms = (int)atoms.size();
  std::vector<std::vector<Atom>> buckets(numSlots);

  std::vector<int> mfma_per_partition;
  for (int pi = 0; pi < numP; ++pi) {
    int piM = pi % config.numPartitionsM();
    int piN = pi / config.numPartitionsM();
    mfma_per_partition.push_back(config.partitionSizesM()[piM] *
                                 config.partitionSizesN()[piN]);
  }

  std::vector<long long> weight_prefix = {0};
  for (int s = 0; s < numSlots; ++s)
    weight_prefix.push_back(weight_prefix.back() + mfma_per_partition[s / numK]);
  long long total_weight = weight_prefix[numSlots];
  std::vector<long long> slot_boundaries;
  for (int i = 1; i <= numSlots; ++i)
    slot_boundaries.push_back(weight_prefix[i] * (long long)nAtoms);

  for (int i = 0; i < nAtoms; ++i) {
    Atom& a = atoms[i];
    int slot;
    if (nAtoms) {
      long long x = (long long)i * total_weight + 1;
      // bisect_left
      int idx = (int)(std::lower_bound(slot_boundaries.begin(),
                                       slot_boundaries.end(), x) -
                      slot_boundaries.begin());
      slot = std::min(idx, a.last);
    } else {
      slot = 0;
    }
    while (slot < a.last &&
           has_lr_conflict(lower, a.tensor, a.mt, slot / numK, slot % numK,
                           a.ks, a.ke)) {
      slot += 1;
    }
    buckets[slot].push_back(a);
  }

  // 2c. remerge consecutive atoms
  for (int flat = 0; flat < numSlots; ++flat) {
    int pi = flat / numK;
    int si = flat % numK;
    CSlot& target_slot = partitions[pi][si];
    for (auto& a : buckets[flat]) {
      if (!target_slot.grs.empty()) {
        Placement* prev = target_slot.grs.back();
        if (prev->tensor == a.tensor && prev->mtIteration == a.mt &&
            prev->tiles.subIterK_start == a.ks &&
            prev->tiles.subIterK_end == a.ke && prev->tiles.tileId_end == a.ts) {
          prev->tiles = MFMATileRange(a.ks, a.ke, prev->tiles.tileId_start, a.te);
          continue;
        }
      }
      Placement gr;
      gr.kind = PKind::GR;
      gr.tensor = a.tensor;
      gr.mtIteration = a.mt;
      gr.tiles = MFMATileRange(a.ks, a.ke, a.ts, a.te);
      gr.subIterK_slot = si;
      gr.partition = pi;
      target_slot.grs.push_back(alloc(std::move(gr)));
    }
  }
}

inline void LogicalScheduler::place_GRs() {
  ensure(Pass::LR);

  std::vector<PartRange> part_ranges;
  for (int pi = 0; pi < config.numPartitions(); ++pi)
    part_ranges.push_back(partition_tile_range(pi));

  int pgr = config.pgr;
  int offsetMT = (pgr == 0) ? 0 : 1;
  auto gr_list = build_gr_list(part_ranges, offsetMT, config.offsetPartition);

  std::map<std::pair<int, std::string>, std::vector<std::tuple<int, int, int>>> lower;
  std::map<std::pair<std::string, int>, int> upper;
  build_gr_slot_bounds(lower, upper);
  distribute_grs(gr_list, lower, upper);

  completed.insert(static_cast<int>(Pass::GR));
}

// ════════════════════════════════════════════════════════════
// annotate_deps
// ════════════════════════════════════════════════════════════

inline void LogicalScheduler::annotate_deps() {
  ensure(Pass::GR);

  std::map<std::string, std::vector<Placement*>> gr_by_tensor;
  std::map<std::string, std::vector<Placement*>> lr_by_tensor;
  for (auto& slots : partitions) {
    for (auto& slot : slots) {
      for (auto* lr : slot.lrs) lr_by_tensor[lr->tensor].push_back(lr);
      for (auto* gr : slot.grs) gr_by_tensor[gr->tensor].push_back(gr);
    }
  }

  for (int pi = 0; pi < (int)partitions.size(); ++pi)
    annotate_deps_partition(pi, partitions[pi], gr_by_tensor, lr_by_tensor);

  completed.insert(static_cast<int>(Pass::DEPS));
}

inline void LogicalScheduler::annotate_deps_partition(
    int pi, std::vector<CSlot>& slots,
    std::map<std::string, std::vector<Placement*>>& gr_by_tensor,
    std::map<std::string, std::vector<Placement*>>& lr_by_tensor) {
  // clear annotations
  for (auto& slot : slots) {
    if (slot.mfma) slot.mfma->deps.clear();
    for (auto* lr : slot.lrs) lr->deps.clear();
    for (auto* gr : slot.grs) gr->deps.clear();
  }

  auto order_of = [](PKind k) { return k == PKind::MFMA ? 0 : (k == PKind::LR ? 1 : 2); };

  auto slot_offset = [&](int consumer_partition, int consumer_slot,
                         int consumer_type_order, Placement* producer) -> int {
    int prod_partition = producer->partition;
    if (prod_partition < consumer_partition) return 0;
    if (prod_partition > consumer_partition) return -1;
    int prod_slot = producer->subIterK_slot;
    if (prod_slot < consumer_slot) return 0;
    if (prod_slot > consumer_slot) return -1;
    int prod_type = order_of(producer->kind);
    return prod_type >= consumer_type_order ? -1 : 0;
  };

  // consumer_type: 0=MFMA,1=LR,2=GR
  auto mt_offset = [&](int consumer_partition, int consumer_slot,
                       int consumer_type, Placement* producer,
                       Placement* consumer) -> int {
    if (consumer_type == 0 && producer->kind == PKind::LR)
      return -producer->mtIteration;
    if (consumer_type == 1 && producer->kind == PKind::GR && consumer)
      return consumer->mtIteration - producer->mtIteration;
    return slot_offset(consumer_partition, consumer_slot, consumer_type, producer);
  };

  auto tiles_overlap = [](Placement* mfma, const std::string& lr_tensor,
                          const MFMATileRange& lr_tiles) -> bool {
    const MFMATileRange& r =
        (lr_tensor == "A" || lr_tensor == "SA") ? mfma->tileA : mfma->tileB;
    return lr_tiles.tileId_start < r.tileId_end &&
           lr_tiles.tileId_end > r.tileId_start &&
           lr_tiles.subIterK_start < r.subIterK_end &&
           lr_tiles.subIterK_end > r.subIterK_start;
  };

  auto range_overlaps = [](const MFMATileRange& a, const MFMATileRange& b) -> bool {
    return a.tileId_start < b.tileId_end && a.tileId_end > b.tileId_start &&
           a.subIterK_start < b.subIterK_end && a.subIterK_end > b.subIterK_start;
  };

  auto exec_order = [](const Dep& d) {
    return std::make_tuple(d.mt_offset, d.ref->partition, d.ref->subIterK_slot);
  };
  auto dedup_deps = [&](std::vector<Dep> deps) -> std::vector<Dep> {
    if (deps.size() <= 1) return deps;
    Dep best = deps[0];
    auto best_eo = exec_order(best);
    for (size_t i = 1; i < deps.size(); ++i) {
      auto eo = exec_order(deps[i]);
      if (eo > best_eo) {
        best = deps[i];
        best_eo = eo;
      }
    }
    return {best};
  };

  for (int k = 0; k < (int)slots.size(); ++k) {
    CSlot& slot = slots[k];
    if (slot.mfma) {
      for (auto& t : tensors) {
        std::vector<Dep> deps_for_t;
        auto it = lr_by_tensor.find(t);
        if (it != lr_by_tensor.end()) {
          for (auto* lr : it->second) {
            if (tiles_overlap(slot.mfma, t, lr->tiles)) {
              Dep d;
              d.ref = lr;
              d.mt_offset = mt_offset(pi, k, 0, lr, nullptr);
              deps_for_t.push_back(d);
            }
          }
        }
        auto dd = dedup_deps(deps_for_t);
        for (auto& d : dd) slot.mfma->deps.push_back(d);
      }
    }

    for (auto* lr : slot.lrs) {
      auto it = gr_by_tensor.find(lr->tensor);
      if (it != gr_by_tensor.end()) {
        for (auto* gr : it->second) {
          if (range_overlaps(lr->tiles, gr->tiles)) {
            Dep d;
            d.ref = gr;
            d.mt_offset = mt_offset(pi, k, 1, gr, lr);
            lr->deps.push_back(d);
          }
        }
      }
    }

    for (auto* gr : slot.grs) {
      int target_data = gr->mtIteration - 2;
      auto it = lr_by_tensor.find(gr->tensor);
      if (it != lr_by_tensor.end()) {
        for (auto* lr : it->second) {
          if (range_overlaps(lr->tiles, gr->tiles)) {
            Dep d;
            d.ref = lr;
            d.mt_offset = target_data - lr->mtIteration;
            gr->deps.push_back(d);
          }
        }
      }
      if (gr->deps.empty()) {
        throw std::runtime_error("GR " + gr->tensor + " mt=" +
                                 fmt_mt(gr->mtIteration) + " at slot " +
                                 std::to_string(k) +
                                 " has no overlapping LR(n) dependency");
      }
    }
  }

  for (auto& slot : slots) {
    for (auto* lr : slot.lrs) lr->deps = dedup_deps(lr->deps);
    for (auto* gr : slot.grs) gr->deps = dedup_deps(gr->deps);
  }
}

// ════════════════════════════════════════════════════════════
// remove_unnecessary_gr_deps
// ════════════════════════════════════════════════════════════

inline void LogicalScheduler::remove_unnecessary_gr_deps() {
  ensure(Pass::DEPS);

  for (auto& tensor : tensors) {
    // build intra-slot rank map
    std::map<std::tuple<int, int, int>, std::vector<Placement*>> slot_members;
    for (auto& slots : partitions)
      for (auto& slot : slots)
        for (auto* gr : slot.grs)
          if (gr->tensor == tensor)
            slot_members[{gr->mtIteration, gr->partition, gr->subIterK_slot}]
                .push_back(gr);
    std::map<Placement*, int> gr_intra_rank;
    for (auto& [key, grs] : slot_members) {
      std::vector<Placement*> sorted_grs = grs;
      std::sort(sorted_grs.begin(), sorted_grs.end(),
                [](Placement* a, Placement* b) {
                  return gr_sort_key(a) < gr_sort_key(b);
                });
      for (int rank = 0; rank < (int)sorted_grs.size(); ++rank)
        gr_intra_rank[sorted_grs[rank]] = rank;
    }
    auto dep_exec_order = [&](const Dep& dep) {
      Placement* gr = dep.ref;
      return std::make_tuple(dep.mt_offset, gr->partition, gr->subIterK_slot,
                             gr_intra_rank[gr]);
    };

    std::vector<std::pair<Placement*, Dep>> lr_with_gr_deps;
    for (auto& slots : partitions)
      for (auto& slot : slots)
        for (auto* lr : slot.lrs)
          if (lr->tensor == tensor && !lr->deps.empty()) {
            Dep dep = lr->deps[0];
            if (dep.ref->kind == PKind::GR) lr_with_gr_deps.push_back({lr, dep});
          }

    if (lr_with_gr_deps.size() <= 1) continue;

    std::tuple<int, int, int, int> max_eo = dep_exec_order(lr_with_gr_deps[0].second);
    for (auto& [lr, dep] : lr_with_gr_deps) {
      auto eo = dep_exec_order(dep);
      if (eo > max_eo) max_eo = eo;
    }
    std::tuple<int, int, int, int> max_guaranteed = {
        std::get<0>(max_eo) - 1, std::get<1>(max_eo), std::get<2>(max_eo), 0};

    for (auto& [lr, dep] : lr_with_gr_deps) {
      auto eo = dep_exec_order(dep);
      if (eo <= max_guaranteed) {
        lr->deps.clear();
      } else {
        max_guaranteed = eo;
      }
    }
  }

  completed.insert(static_cast<int>(Pass::REMOVE_GR_DEPS));
}

// ════════════════════════════════════════════════════════════
// remove_unnecessary_lr_deps
// ════════════════════════════════════════════════════════════

inline void LogicalScheduler::remove_unnecessary_lr_deps() {
  ensure(Pass::REMOVE_GR_DEPS);

  using EO = std::tuple<int, int, int>;
  auto dep_exec_order = [](const Dep& dep) {
    return std::make_tuple(dep.mt_offset, dep.ref->partition,
                           dep.ref->subIterK_slot);
  };

  struct SyncEntry {
    std::pair<int, int> pos;  // (pi, subIterK)
    std::map<std::string, EO> last_lr;
    std::vector<Placement*> grs_with_lr;
  };
  std::vector<SyncEntry> sync_slots;

  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    for (auto& slot : partitions[pi]) {
      std::vector<Placement*> grs_with_lr;
      for (auto* gr : slot.grs)
        if (!gr->deps.empty() && gr->deps[0].ref->kind == PKind::LR)
          grs_with_lr.push_back(gr);
      bool lr_with_gr_exists = false;
      for (auto* lr : slot.lrs)
        if (!lr->deps.empty() && lr->deps[0].ref->kind == PKind::GR)
          lr_with_gr_exists = true;
      if (grs_with_lr.empty() && !lr_with_gr_exists) continue;

      std::map<std::string, EO> last_lr;
      if (slot.mfma) {
        for (auto& d : slot.mfma->deps) {
          if (d.ref->kind == PKind::LR) {
            std::string t = d.ref->tensor;
            EO eo = dep_exec_order(d);
            auto it = last_lr.find(t);
            if (it == last_lr.end() || eo > it->second) last_lr[t] = eo;
          }
        }
      }
      for (auto* gr : grs_with_lr) {
        Dep dep = gr->deps[0];
        std::string t = dep.ref->tensor;
        EO eo = dep_exec_order(dep);
        auto it = last_lr.find(t);
        if (it == last_lr.end() || eo > it->second) last_lr[t] = eo;
      }
      sync_slots.push_back({{pi, slot.subIterK}, last_lr, grs_with_lr});
    }
  }

  if (sync_slots.empty()) {
    completed.insert(static_cast<int>(Pass::REMOVE_LR_DEPS));
    return;
  }

  std::sort(sync_slots.begin(), sync_slots.end(),
            [](const SyncEntry& a, const SyncEntry& b) { return a.pos < b.pos; });
  int n = (int)sync_slots.size();

  for (int i = 0; i < n; ++i) {
    auto grs_to_check = sync_slots[i].grs_with_lr;
    for (auto* gr : grs_to_check) {
      if (gr->deps.empty()) continue;
      Dep dep = gr->deps[0];
      std::string tensor = dep.ref->tensor;
      EO cur_eo = dep_exec_order(dep);

      std::optional<EO> prev_eo;
      auto cur_pos = sync_slots[i].pos;
      for (int j = 1; j <= n; ++j) {
        int idx = ((i - j) % n + n) % n;
        bool wrapped = j > i;
        if (!wrapped && sync_slots[idx].pos == cur_pos) continue;
        auto& prev_last_lr = sync_slots[idx].last_lr;
        auto it = prev_last_lr.find(tensor);
        if (it != prev_last_lr.end()) {
          EO eo = it->second;
          if (wrapped)
            eo = {std::get<0>(eo) - 1, std::get<1>(eo), std::get<2>(eo)};
          prev_eo = eo;
          break;
        }
      }

      if (prev_eo.has_value() && *prev_eo >= cur_eo) gr->deps.clear();
    }
  }

  completed.insert(static_cast<int>(Pass::REMOVE_LR_DEPS));
}

// ════════════════════════════════════════════════════════════
// remove_cross_deps
// ════════════════════════════════════════════════════════════

inline int LogicalScheduler::count_gr_atoms(const Placement& gr) const {
  ReadGranularity g = gr_granularity(gr.tensor);
  int n_tile = (gr.tiles.tileId_end - gr.tiles.tileId_start) / g.mn;
  int n_k = (gr.tiles.subIterK_end - gr.tiles.subIterK_start) / g.k;
  return n_tile * n_k;
}

inline WaitGRCounts LogicalScheduler::compute_inflight_loads(
    int consumer_pi, int consumer_slot, const std::string& tensor,
    const Dep& dep_ref) {
  int numP = (int)partitions.size();
  int numK = (int)partitions[0].size();
  int flat_len = numP * numK;

  int consumer_flat = consumer_pi * numK + consumer_slot;
  int wraps_needed = std::abs(dep_ref.mt_offset);

  int dep_flat = -1;
  for (int p_idx = 0; p_idx < numP && dep_flat < 0; ++p_idx) {
    for (int k_idx = 0; k_idx < (int)partitions[p_idx].size(); ++k_idx) {
      bool found = false;
      for (auto* gr : partitions[p_idx][k_idx].grs)
        if (gr == dep_ref.ref) found = true;
      if (found) {
        dep_flat = p_idx * numK + k_idx;
        break;
      }
    }
  }
  if (dep_flat < 0) return WaitGRCounts();

  int total_steps = wraps_needed * flat_len + consumer_flat - dep_flat;
  if (total_steps <= 0) return WaitGRCounts();

  auto add_count = [](WaitGRCounts& c, const std::string& t, int v) {
    if (t == "A") c.A += v;
    else if (t == "B") c.B += v;
    else if (t == "SA") c.SA += v;
    else c.SB += v;
  };

  WaitGRCounts counts;
  int pos = consumer_flat;
  for (int step = 0; step < total_steps; ++step) {
    pos = ((pos - 1) % flat_len + flat_len) % flat_len;
    int pi = pos / numK;
    int slot_k = pos % numK;
    CSlot& slot = partitions[pi][slot_k];
    bool is_final = (step == total_steps - 1);

    std::vector<Placement*> sorted_grs = slot.grs;
    std::sort(sorted_grs.begin(), sorted_grs.end(),
              [](Placement* a, Placement* b) {
                return gr_sort_key(a) > gr_sort_key(b);  // reverse
              });
    for (auto* gr : sorted_grs) {
      if (is_final && gr->tensor == tensor && gr == dep_ref.ref) return counts;
      int atoms = count_gr_atoms(*gr);
      add_count(counts, gr->tensor, atoms);
    }
  }
  return counts;
}

inline void LogicalScheduler::remove_cross_deps() {
  ensure(Pass::REMOVE_LR_DEPS);

  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    for (auto& slot : partitions[pi]) {
      if (slot.mfma) {
        auto [same, cross] = split_deps(slot.mfma->deps, pi, slot.subIterK);
        bool has_lr_dep = false;
        for (auto& d : same)
          if (d.ref->kind == PKind::LR) has_lr_dep = true;
        for (auto& d : cross)
          if (d.ref->kind == PKind::LR) has_lr_dep = true;
        slot.mfma->deps = same;
        slot.mfma->preOps.clear();
        if (has_lr_dep) slot.mfma->preOps.push_back(Op::waitLR(false));
      }

      for (auto* lr : slot.lrs) {
        std::vector<Dep> gr_deps;
        for (auto& d : lr->deps)
          if (d.ref->kind == PKind::GR) gr_deps.push_back(d);
        auto [same, cross] = split_deps(lr->deps, pi, lr->subIterK_slot);
        lr->deps = same;
        lr->preOps.clear();
        if (!gr_deps.empty()) {
          Dep dep = gr_deps[0];
          (void)cross;
          // is_cross: the dep is cross-subIterK (not same partition+slot at
          // mt_offset 0). Mirrors `id(dep) in cross_set` in Python.
          bool dep_is_same = (dep.mt_offset == 0 &&
                              dep.ref->partition == pi &&
                              dep.ref->subIterK_slot == lr->subIterK_slot);
          bool is_cross = !dep_is_same;
          WaitGRCounts counts =
              compute_inflight_loads(pi, lr->subIterK_slot, dep.ref->tensor, dep);
          lr->preOps.push_back(Op::waitGR(counts, true, is_cross));
        }
      }

      for (auto* gr : slot.grs) {
        auto [same, cross] = split_deps(gr->deps, pi, gr->subIterK_slot);
        gr->deps = same;
        bool has_lr_dep = false;
        for (auto& d : same)
          if (d.ref->kind == PKind::LR) has_lr_dep = true;
        for (auto& d : cross)
          if (d.ref->kind == PKind::LR) has_lr_dep = true;
        gr->preOps.clear();
        if (has_lr_dep) gr->preOps.push_back(Op::waitLR(true));
      }
    }
  }

  completed.insert(static_cast<int>(Pass::REMOVE_DEPS));
}

// ════════════════════════════════════════════════════════════
// insert_gr_lr_inc
// ════════════════════════════════════════════════════════════

inline void LogicalScheduler::insert_gr_lr_inc() {
  ensure(Pass::REMOVE_DEPS);

  std::map<std::string, int> last_lr_mt;
  std::map<std::string, int> last_gr_mt;
  std::map<std::string, Placement*> first_lr;
  std::map<std::string, Placement*> last_lr;
  std::set<std::string> lr_inc_tensors;
  std::vector<std::string> first_lr_order;

  for (auto& slots : partitions) {
    for (auto& slot : slots) {
      for (auto* lr : slot.lrs) {
        std::string tensor = lr->tensor;
        int mt = lr->mtIteration;
        if (!first_lr.count(tensor)) {
          first_lr[tensor] = lr;
          first_lr_order.push_back(tensor);
        }
        auto it = last_lr_mt.find(tensor);
        if (it != last_lr_mt.end() && it->second != mt) {
          lr->preOps.push_back(Op::lrInc(tensor));
          lr_inc_tensors.insert(tensor);
        }
        last_lr[tensor] = lr;
        last_lr_mt[tensor] = mt;
      }
      for (auto* gr : slot.grs) {
        std::string tensor = gr->tensor;
        int mt = gr->mtIteration;
        int prev_mt = last_gr_mt.count(tensor) ? last_gr_mt[tensor] : 0;
        if (prev_mt != mt) {
          if (gr->tiles.tileId_start == 0) gr->preOps.push_back(Op::grInc(tensor));
        }
        last_gr_mt[tensor] = mt;
      }
    }
  }

  static const std::vector<std::string> LR_GR_ORDER = {"A", "B", "SA", "SB"};
  if (config.pgr == 0) {
    std::map<std::string, Placement*> last_gr_per_tensor;
    for (auto& slots : partitions)
      for (auto& slot : slots)
        for (auto* gr : slot.grs) last_gr_per_tensor[gr->tensor] = gr;
    for (auto& tensor : LR_GR_ORDER) {
      if (last_lr.count(tensor) && last_lr_mt.count(tensor))
        last_lr[tensor]->postOps.push_back(Op::lrInc(tensor));
      if (last_gr_per_tensor.count(tensor) && last_gr_mt.count(tensor))
        last_gr_per_tensor[tensor]->postOps.push_back(Op::grInc(tensor));
    }
  } else {
    for (auto& tensor : first_lr_order) {
      if (!lr_inc_tensors.count(tensor))
        first_lr[tensor]->preOps.push_back(Op::lrInc(tensor));
    }
  }

  completed.insert(static_cast<int>(Pass::GR_INC));
}

// ════════════════════════════════════════════════════════════
// group_lr_gr
// ════════════════════════════════════════════════════════════

inline std::vector<Op> LogicalScheduler::merge_preops(
    const std::vector<std::vector<Op>*>& all_preops) const {
  std::vector<const Op*> wait_gr_ops_full;
  bool has_wait_gr_sync = false;
  bool seen_wait_lr = false;
  std::vector<Op> others;
  for (auto* preops : all_preops) {
    for (auto& op : *preops) {
      if (op.kind == "wait_gr" && op.hasCounts) {
        if (op.has_sync) has_wait_gr_sync = true;
        wait_gr_ops_full.push_back(&op);
      } else if (op.kind == "wait_lr") {
        if (!seen_wait_lr) {
          seen_wait_lr = true;
          others.push_back(op);
        }
      } else {
        others.push_back(op);
      }
    }
  }
  std::vector<Op> result;
  if (!wait_gr_ops_full.empty()) {
    WaitGRCounts merged;
    auto minf = [&](int WaitGRCounts::*field) {
      int m = wait_gr_ops_full[0]->counts.*field;
      for (auto* op : wait_gr_ops_full) m = std::min(m, op->counts.*field);
      return m;
    };
    merged.A = minf(&WaitGRCounts::A);
    merged.B = minf(&WaitGRCounts::B);
    merged.SA = minf(&WaitGRCounts::SA);
    merged.SB = minf(&WaitGRCounts::SB);
    bool adjust = true;
    for (auto* op : wait_gr_ops_full)
      if (!op->adjustVmcnt) adjust = false;
    result.push_back(Op::waitGR(merged, has_wait_gr_sync, adjust));
  }
  for (auto& o : others) result.push_back(o);
  return result;
}

inline void LogicalScheduler::group_lr_gr() {
  ensure(Pass::GR_INC);

  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    for (auto& slot : partitions[pi]) {
      // Phase 1: LR chain
      std::vector<Placement*> ordered_lrs = slot.lrs;
      std::stable_sort(ordered_lrs.begin(), ordered_lrs.end(),
                       [](Placement* a, Placement* b) {
                         return tensor_order(a->tensor) < tensor_order(b->tensor);
                       });

      if (ordered_lrs.size() > 1) {
        std::vector<std::vector<Op>*> preop_lists;
        for (auto* lr : ordered_lrs) preop_lists.push_back(&lr->preOps);
        std::vector<Op> merged = merge_preops(preop_lists);
        ordered_lrs[0]->preOps = merged;
        for (size_t i = 1; i < ordered_lrs.size(); ++i)
          ordered_lrs[i]->preOps.clear();
        for (size_t i = 1; i < ordered_lrs.size(); ++i)
          ordered_lrs[i]->deps = {Dep{ordered_lrs[i - 1], 0}};
      }

      Placement* last_lr = ordered_lrs.empty() ? nullptr : ordered_lrs.back();

      // Phase 2: GR chain
      std::vector<Placement*> ordered_grs = slot.grs;
      std::stable_sort(ordered_grs.begin(), ordered_grs.end(),
                       [](Placement* a, Placement* b) {
                         return gr_sort_key(a) < gr_sort_key(b);
                       });

      if (ordered_grs.size() > 1) {
        bool any_deps = false;
        for (auto* gr : ordered_grs)
          if (!gr->deps.empty()) any_deps = true;

        bool seen_wait_lr_sync = false;
        for (auto* gr : ordered_grs) {
          if (seen_wait_lr_sync) {
            std::vector<Op> kept;
            for (auto& op : gr->preOps)
              if (!(op.kind == "wait_lr" && op.has_sync)) kept.push_back(op);
            gr->preOps = kept;
          } else {
            for (auto& op : gr->preOps)
              if (op.kind == "wait_lr" && op.has_sync) {
                seen_wait_lr_sync = true;
                break;
              }
          }
        }

        if (any_deps && last_lr != nullptr)
          ordered_grs[0]->deps = {Dep{last_lr, 0}};
        else
          ordered_grs[0]->deps.clear();

        for (size_t i = 1; i < ordered_grs.size(); ++i)
          ordered_grs[i]->deps = {Dep{ordered_grs[i - 1], 0}};
      } else if (ordered_grs.size() == 1) {
        if (!ordered_grs[0]->deps.empty() && last_lr != nullptr)
          ordered_grs[0]->deps = {Dep{last_lr, 0}};
      }

      // Phase 3: cross-group merge
      if (!ordered_grs.empty() && !ordered_lrs.empty()) {
        std::set<Placement*> slot_gr_set(ordered_grs.begin(), ordered_grs.end());
        bool lr_has_gr_dep = false;
        for (auto* lr : ordered_lrs)
          for (auto& d : lr->deps)
            if (slot_gr_set.count(d.ref)) lr_has_gr_dep = true;
        if (lr_has_gr_dep) {
          Placement* last_gr = ordered_grs.back();
          for (auto* lr : ordered_lrs) {
            std::vector<Dep> kept;
            for (auto& d : lr->deps)
              if (!slot_gr_set.count(d.ref)) kept.push_back(d);
            lr->deps = kept;
          }
          ordered_lrs[0]->deps = {Dep{last_gr, 0}};
        }
      }

      // Phase 4: consolidate MFMA deps
      if (slot.mfma && last_lr != nullptr) {
        std::set<Placement*> slot_lr_set(ordered_lrs.begin(), ordered_lrs.end());
        std::vector<Dep> lr_deps;
        for (auto& d : slot.mfma->deps)
          if (slot_lr_set.count(d.ref)) lr_deps.push_back(d);
        if (lr_deps.size() > 1) {
          std::vector<Dep> other_deps;
          for (auto& d : slot.mfma->deps)
            if (!slot_lr_set.count(d.ref)) other_deps.push_back(d);
          other_deps.push_back(Dep{last_lr, lr_deps[0].mt_offset});
          slot.mfma->deps = other_deps;
        }
      }
    }
  }

  completed.insert(static_cast<int>(Pass::GROUP_LR_GR));
}

// ════════════════════════════════════════════════════════════
// remove_unnecessary_wait_lr_sync
// ════════════════════════════════════════════════════════════

inline void LogicalScheduler::remove_unnecessary_wait_lr_sync() {
  ensure(Pass::GROUP_LR_GR);

  auto has_wait_lr_sync = [](Placement* p) {
    for (auto& op : p->preOps)
      if (op.kind == "wait_lr" && op.has_sync) return true;
    return false;
  };

  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    auto& slots = partitions[pi];
    for (int si = 0; si < (int)slots.size(); ++si) {
      CSlot& slot = slots[si];
      if (slot.grs.empty()) continue;
      Placement* first_gr = slot.grs[0];
      if (!has_wait_lr_sync(first_gr)) continue;
      if (!first_gr->deps.empty()) continue;
      if (si == 0) continue;
      CSlot& prev_slot = slots[si - 1];
      if (prev_slot.grs.empty()) continue;
      Placement* prev_first_gr = prev_slot.grs[0];
      bool prev_has = has_wait_lr_sync(prev_first_gr);
      bool prev_deps_on_lrs = !prev_first_gr->deps.empty();
      if (prev_has && prev_deps_on_lrs) {
        std::vector<Op> kept;
        for (auto& op : first_gr->preOps)
          if (!(op.kind == "wait_lr" && op.has_sync)) kept.push_back(op);
        first_gr->preOps = kept;
      }
    }
  }

  for (auto& slots : partitions) {
    for (auto& slot : slots) {
      for (auto* gr : slot.grs) {
        if (!has_wait_lr_sync(gr)) continue;
        bool has_lr_dep = false;
        Placement* node = gr;
        while (node && !node->deps.empty()) {
          Placement* ref = node->deps[0].ref;
          if (ref->kind == PKind::LR) {
            has_lr_dep = true;
            break;
          }
          node = ref;
        }
        if (has_lr_dep) continue;
        for (auto& op : gr->preOps)
          if (op.kind == "wait_lr" && op.has_sync) op = Op::sync();
      }
    }
  }

  completed.insert(static_cast<int>(Pass::REMOVE_WAIT_LR_SYNC));
}

// ════════════════════════════════════════════════════════════
// emit / build
// ════════════════════════════════════════════════════════════

// Convert a pass-pipeline Placement (identity pointer) into a coordinate-only
// value Emittable source for an EmittedModule. Only the identity/coordinate
// fields are copied — the Python converter uses them to look up its persistent
// placement dataclass; str()/kind (the only fields print_emit consumes) match
// the placement.
inline tw::subtile::lsched::Emittable placement_to_emittable(
    const Placement* p) {
  using namespace tw::subtile::lsched;
  if (p->kind == PKind::MFMA) {
    return MFMAPlacement(p->subIterK, p->tileA, p->tileB);
  }
  if (p->kind == PKind::LR) {
    return LRPlacement(p->tensor, p->mtIteration, p->tiles, p->subIterK_slot,
                       p->partition);
  }
  return GRPlacement(p->tensor, p->mtIteration, p->tiles, p->subIterK_slot,
                     p->partition);
}

// Convert a pass-pipeline before-chain Op into a value Emittable source. emit()
// only ever produces wait_gr / wait_lr / sync / lr_inc / gr_inc ops.
inline tw::subtile::lsched::Emittable op_to_emittable(const Op& o) {
  using namespace tw::subtile::lsched;
  if (o.kind == "wait_gr") {
    std::optional<WaitGRCounts> c;
    if (o.hasCounts) c = o.counts;
    return WaitGROp(std::move(c), o.has_sync, o.adjustVmcnt);
  }
  if (o.kind == "wait_lr") return WaitLROp(o.has_sync);
  if (o.kind == "sync") return SyncOp();
  if (o.kind == "mask_k") return MaskKOp(o.subIterK);
  if (o.kind == "lr_inc") return LRIncOp(o.tensor);
  if (o.kind == "gr_inc") return GRIncOp(o.tensor);
  throw std::runtime_error("unexpected before-chain op kind: '" + o.kind + "'");
}

inline void LogicalScheduler::emit() {
  ensure(Pass::REMOVE_WAIT_LR_SYNC);
  using tw::subtile::lsched::EmittedModule;

  std::vector<std::vector<std::vector<EmittedModule>>> all_partitions;
  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    std::vector<std::vector<EmittedModule>> partition_emitted;
    for (auto& slot : partitions[pi]) {
      std::vector<EmittedModule> mods;
      std::map<Placement*, int> placement_to_id;
      std::map<Placement*, int> placement_tail_id;

      auto addPlacement = [&](Placement* p) -> int {
        int mid = (int)mods.size();
        EmittedModule em;
        em.moduleId = mid;
        em.source = placement_to_emittable(p);
        mods.push_back(std::move(em));
        return mid;
      };
      auto addOp = [&](const Op& op) -> int {
        int mid = (int)mods.size();
        EmittedModule em;
        em.moduleId = mid;
        em.source = op_to_emittable(op);
        mods.push_back(std::move(em));
        return mid;
      };
      auto setBefore = [&](int moduleId, std::optional<int> beforeId) {
        if (!beforeId.has_value() || *beforeId == moduleId) return;
        auto& cur = mods[moduleId].before;
        if (!cur.has_value()) {
          cur = beforeId;
          return;
        }
        // (Python asserts equality; mirror by ignoring duplicates.)
      };

      // Step 1: primary modules
      std::vector<Placement*> placements;
      if (slot.mfma) placements.push_back(slot.mfma);
      for (auto* lr : slot.lrs) placements.push_back(lr);
      for (auto* gr : slot.grs) placements.push_back(gr);

      for (auto* placement : placements) {
        int mid = addPlacement(placement);
        placement_to_id[placement] = mid;
        placement_tail_id[placement] = mid;
      }

      // Step 1b: postOps
      for (auto* placement : placements) {
        if (placement->postOps.empty()) continue;
        int postPrevId = placement_to_id[placement];
        for (auto& postOp : placement->postOps) {
          int postId = addOp(postOp);
          setBefore(postId, postPrevId);
          postPrevId = postId;
        }
        placement_tail_id[placement] = postPrevId;
      }

      // Step 2: before-chains from preOps + deps
      for (auto* placement : placements) {
        int curId = placement_to_id[placement];
        std::optional<int> prevId;
        std::optional<int> lastDepId;
        std::optional<int> firstPreOpId;

        for (auto& preOp : placement->preOps) {
          if (preOp.kind == "wait_gr") {
            int depId = addOp(preOp);
            prevId = depId;
            if (!firstPreOpId.has_value()) firstPreOpId = depId;
            if (preOp.has_sync) {
              int sId = addOp(Op::sync());
              setBefore(sId, prevId);
              prevId = sId;
              lastDepId = sId;
            }
            continue;
          } else if (preOp.kind == "wait_lr" && preOp.has_sync) {
            int depId = addOp(Op::waitLR(false));
            setBefore(depId, prevId);
            prevId = depId;
            lastDepId = depId;
            if (!firstPreOpId.has_value()) firstPreOpId = depId;
            int sId = addOp(Op::sync());
            setBefore(sId, prevId);
            prevId = sId;
            lastDepId = sId;
            continue;
          } else {
            int depId = addOp(preOp);
            setBefore(depId, prevId);
            prevId = depId;
            lastDepId = depId;
            if (!firstPreOpId.has_value()) firstPreOpId = depId;
          }
        }

        for (auto& dep : placement->deps) {
          auto it = placement_tail_id.find(dep.ref);
          if (it != placement_tail_id.end()) {
            int ref_id = it->second;
            if (firstPreOpId.has_value()) {
              setBefore(*firstPreOpId, ref_id);
            } else {
              prevId = ref_id;
            }
          }
        }

        if (lastDepId.has_value()) {
          setBefore(curId, lastDepId);
        } else if (prevId.has_value()) {
          setBefore(curId, prevId);
        }
      }

      partition_emitted.push_back(std::move(mods));
    }
    all_partitions.push_back(std::move(partition_emitted));
  }

  emitted = std::move(all_partitions);
  completed.insert(static_cast<int>(Pass::EMIT));
}

inline void LogicalScheduler::build() {
  emit();
  completed.insert(static_cast<int>(Pass::BUILD));
}

// ════════════════════════════════════════════════════════════
// Loop-variant builders (NGLL / NLL / preloop / tail-PGR0)
// ════════════════════════════════════════════════════════════

// Wrap value Emittables (placements / before-chain ops) into EmittedModules,
// moduleId = list index, before = none. Mirrors Python _to_emitted.
inline std::vector<tw::subtile::lsched::EmittedModule> to_emitted_value(
    const std::vector<tw::subtile::lsched::Emittable>& ops) {
  std::vector<tw::subtile::lsched::EmittedModule> out;
  out.reserve(ops.size());
  for (int i = 0; i < (int)ops.size(); ++i) {
    tw::subtile::lsched::EmittedModule em;
    em.moduleId = i;
    em.source = ops[i];
    out.push_back(std::move(em));
  }
  return out;
}

// A single empty-variant grid ([[[]]]): one partition, one subIterK, no
// modules — the shape emit() produces for the degenerate PGR cases.
inline LogicalScheduler::ModuleGrid empty_variant_grid() {
  return LogicalScheduler::ModuleGrid{
      {std::vector<tw::subtile::lsched::EmittedModule>{}}};
}

inline std::vector<tw::subtile::lsched::EmittedModule>
LogicalScheduler::rewire_before(
    std::vector<tw::subtile::lsched::EmittedModule> mods,
    const std::set<int>& removed) {
  std::map<int, const tw::subtile::lsched::EmittedModule*> id_to_em;
  for (auto& em : mods) id_to_em[em.moduleId] = &em;
  for (auto& em : mods) {
    if (removed.count(em.moduleId)) continue;
    std::optional<int> b = em.before;
    while (b.has_value() && removed.count(*b)) b = id_to_em[*b]->before;
    em.before = b;
  }
  std::vector<tw::subtile::lsched::EmittedModule> out;
  for (auto& em : mods)
    if (!removed.count(em.moduleId)) out.push_back(std::move(em));
  return out;
}

inline void LogicalScheduler::build_ngll() {
  ensure(Pass::EMIT);
  ngll_emitted.clear();
  if (config.pgr == 0 || config.pgr == 1) {
    ngll_emitted = empty_variant_grid();
    return;
  }
  using tw::subtile::lsched::EmittedModule;
  using tw::subtile::lsched::GRPlacement;
  using tw::subtile::lsched::WaitGROp;
  for (auto& partition_emitted : emitted) {
    std::vector<std::vector<EmittedModule>> part_ngll;
    for (auto& em_list : partition_emitted) {
      std::vector<EmittedModule> mods = em_list;  // copy
      std::set<int> removed;
      for (auto& em : mods) {
        std::string ot = em.opType();
        if (ot == "gr") {
          if (std::get<GRPlacement>(*em.source).mtIteration == 2)
            removed.insert(em.moduleId);
        } else if (ot == "wait_gr") {
          auto& wg = std::get<WaitGROp>(*em.source);
          if (wg.wait_gr_counts.has_value())
            wg.wait_gr_counts = tw::subtile::lsched::WaitGRCounts();
        }
      }
      part_ngll.push_back(rewire_before(std::move(mods), removed));
    }
    ngll_emitted.push_back(std::move(part_ngll));
  }
}

inline void LogicalScheduler::build_nll() {
  ensure(Pass::EMIT);
  nll_emitted.clear();
  if (config.pgr == 0) {
    nll_emitted = empty_variant_grid();
    return;
  }
  using tw::subtile::lsched::EmittedModule;
  using tw::subtile::lsched::LRPlacement;
  using tw::subtile::lsched::WaitGROp;
  for (auto& partition_emitted : emitted) {
    std::vector<std::vector<EmittedModule>> part_nll;
    for (auto& em_list : partition_emitted) {
      std::vector<EmittedModule> mods = em_list;  // copy
      std::set<int> removed;

      for (auto& em : mods) {
        std::string ot = em.opType();
        if (ot == "gr") {
          removed.insert(em.moduleId);
        } else if (ot == "lr") {
          if (std::get<LRPlacement>(*em.source).mtIteration == 1)
            removed.insert(em.moduleId);
        } else if (ot == "gr_inc" && config.pgr == 2) {
          removed.insert(em.moduleId);
        }
      }

      // Zero inflight counts on remaining WaitGR.
      for (auto& em : mods) {
        if (em.opType() == "wait_gr" && !removed.count(em.moduleId))
          std::get<WaitGROp>(*em.source).wait_gr_counts =
              tw::subtile::lsched::WaitGRCounts();
      }

      // Sync modules paired with a removed wait_gr.
      for (auto& em : mods) {
        if (em.opType() == "sync" && em.before.has_value() &&
            removed.count(*em.before))
          removed.insert(em.moduleId);
      }

      // Remove WaitLR if no LR remains, but keep ones a kept module depends on.
      bool has_lr = false;
      for (auto& em : mods)
        if (em.opType() == "lr" && !removed.count(em.moduleId)) has_lr = true;
      if (!has_lr) {
        std::set<int> depended_on;
        for (auto& em : mods)
          if (!removed.count(em.moduleId) && em.before.has_value())
            depended_on.insert(*em.before);
        for (auto& em : mods)
          if (em.opType() == "wait_lr" && !depended_on.count(em.moduleId))
            removed.insert(em.moduleId);
      }

      part_nll.push_back(rewire_before(std::move(mods), removed));
    }
    nll_emitted.push_back(std::move(part_nll));
  }
}

inline std::vector<tw::subtile::lsched::Emittable>
LogicalScheduler::make_gr_all_tensors(int mt, const MFMATileRange& tilesA,
                                      const MFMATileRange& tilesB) const {
  std::vector<tw::subtile::lsched::Emittable> out;
  for (const auto& tensor : tensors) {
    const MFMATileRange& tr =
        (tensor_side(tensor)[0] == 'A') ? tilesA : tilesB;
    out.push_back(tw::subtile::lsched::GRPlacement(tensor, mt, tr, 0));
  }
  return out;
}

inline std::vector<tw::subtile::lsched::Emittable>
LogicalScheduler::make_lr_all_tensors(
    const std::map<std::string, MFMATileRange>& lr_tiles) const {
  std::vector<tw::subtile::lsched::Emittable> out;
  const Placement* first_mfma = partitions[0][0].mfma;
  for (const auto& tensor : tensors) {
    tw::subtile::lsched::LRPlacement lr(tensor, 0, lr_tiles.at(tensor), 0, 0);
    auto it = first_mfma->vgpr_tile_maps.find(tensor);
    if (it != first_mfma->vgpr_tile_maps.end()) lr.vgpr_tile_map = it->second;
    out.push_back(std::move(lr));
  }
  return out;
}

inline std::vector<tw::subtile::lsched::Emittable>
LogicalScheduler::make_preloop_mt1_grs() {
  ensure(Pass::LR);
  std::vector<tw::subtile::lsched::Emittable> out;
  std::set<std::tuple<std::string, int, int, int, int>> seen;
  for (int pi = 0; pi < config.offsetPartition; ++pi) {
    PartRange target = partition_tile_range(pi);
    for (auto& slot : partitions[0]) {
      int k = slot.mfma->subIterK;
      struct Item {
        std::string tensor;
        std::pair<int, int> tr;
        ReadGranularity gran;
      };
      std::vector<Item> items;
      items.push_back({"A", target.A, config.grA});
      items.push_back({"B", target.B, config.grB});
      if (config.hasScale()) {
        items.push_back({"SA", target.A, *config.grSA});
        items.push_back({"SB", target.B, *config.grSB});
      }
      for (auto& it : items) {
        MFMATileRange tr = it.gran.tile_range(k, it.tr.first, it.tr.second);
        auto key = std::make_tuple(it.tensor, tr.tileId_start, tr.tileId_end,
                                   tr.subIterK_start, tr.subIterK_end);
        if (seen.count(key)) continue;
        seen.insert(key);
        out.push_back(tw::subtile::lsched::GRPlacement(it.tensor, 1, tr, k, pi));
      }
    }
  }
  return out;
}

inline void LogicalScheduler::build_preloop() {
  using namespace tw::subtile::lsched;
  // assign_vgpr_tiles populates the first MFMA's vgpr_tile_maps copied into the
  // entry LRs by make_lr_all_tensors.
  ensure(Pass::VGPR_TILES);
  preloop_emitted.clear();
  if (config.pgr == 0) {
    preloop_emitted = empty_variant_grid();
    return;
  }

  int numK = config.numSubIterK;
  PartRange part0 = partition_tile_range(0);
  MFMATileRange all_A(0, numK, 0, config.numMFMATilesM);
  MFMATileRange all_B(0, numK, 0, config.numMFMATilesN);

  std::map<std::string, MFMATileRange> lr_tiles;
  lr_tiles["A"] = MFMATileRange(0, config.lrA.k, part0.A.first, part0.A.second);
  lr_tiles["B"] = MFMATileRange(0, config.lrB.k, part0.B.first, part0.B.second);
  if (config.hasScale()) {
    lr_tiles["SA"] =
        MFMATileRange(0, config.lrSA->k, part0.A.first, part0.A.second);
    lr_tiles["SB"] =
        MFMATileRange(0, config.lrSB->k, part0.B.first, part0.B.second);
  }

  std::vector<Emittable> ops;
  auto append = [&](std::vector<Emittable> v) {
    for (auto& e : v) ops.push_back(std::move(e));
  };

  append(make_gr_all_tensors(0, all_A, all_B));
  if (config.pgr == 1) {
    ops.push_back(WaitGROp(WaitGRCounts(), false, true));
    ops.push_back(SyncOp());
    append(make_lr_all_tensors(lr_tiles));
    ops.push_back(SkipOp("LE", 1, "NLL", false, ""));
  } else {
    for (const auto& tensor : tensors) ops.push_back(GRIncOp(tensor));
    ops.push_back(WaitGROp(WaitGRCounts(), false, true));
    ops.push_back(SyncOp());
    append(make_lr_all_tensors(lr_tiles));
    ops.push_back(SkipOp("LE", 1, "NLL", false, ""));
    append(make_preloop_mt1_grs());
    ops.push_back(SkipOp("LE", 2, "NGLL", false, ""));
  }

  preloop_emitted = ModuleGrid{{to_emitted_value(ops)}};
}

inline void LogicalScheduler::build_tailloop_pgr0(const FlatTileMaps& tile_maps,
                                                  bool bf16, int miK) {
  using namespace tw::subtile::lsched;
  int numK = config.numSubIterK;
  int numP = config.numPartitions();

  // Preamble: GR all tensors, optional BF16 boundary fixup, wait_gr, sync.
  std::vector<Emittable> preamble;
  MFMATileRange all_A(0, numK, 0, config.numMFMATilesM);
  MFMATileRange all_B(0, numK, 0, config.numMFMATilesN);
  for (auto& e : make_gr_all_tensors(0, all_A, all_B)) preamble.push_back(e);
  if (bf16) {
    preamble.push_back(SyncOp());
    preamble.push_back(InlineModuleOp("tail_boundary_ab"));
  }
  preamble.push_back(WaitGROp(WaitGRCounts(), false, true));
  preamble.push_back(SyncOp());

  std::vector<std::vector<EmittedModule>> groups;
  groups.push_back(to_emitted_value(preamble));

  // Merged tile map across partitions for the K-mask op.
  std::map<std::string, std::vector<std::map<int, int>>> merged_tile_map;
  for (int pi = 0; pi < numP; ++pi) {
    for (const char* tname : {"A", "B", "SA", "SB"}) {
      std::string tensor = tname;
      auto it = tile_maps[pi].find(tensor);
      if (it == tile_maps[pi].end() || it->second.empty()) continue;
      const auto& src = it->second;
      auto& dst = merged_tile_map[tensor];
      if (dst.empty()) dst.emplace_back();
      while (dst.size() < src.size()) dst.emplace_back();
      for (size_t ui = 0; ui < src.size(); ++ui)
        for (const auto& kv : src[ui]) dst[ui][kv.first] = kv.second;
    }
  }

  auto lrt = lr_tensors();
  for (int k = 0; k < numK; ++k) {
    std::vector<Emittable> ops;
    // Dedup LRs across partitions by (tensor, tile range).
    std::set<std::tuple<std::string, int, int, int, int>> seen_lr;
    for (int pi = 0; pi < numP; ++pi) {
      PartRange cur = partition_tile_range(pi);
      for (auto& [tensor, gran] : lrt) {
        if (k % gran.k != 0) continue;
        const char* side_key = tensor_side(tensor);
        auto [ts, te] = cur.by_side(side_key);
        MFMATileRange tiles = gran.tile_range(k, ts, te);
        auto lr_key = std::make_tuple(tensor, tiles.tileId_start,
                                      tiles.tileId_end, tiles.subIterK_start,
                                      tiles.subIterK_end);
        if (seen_lr.count(lr_key)) continue;
        seen_lr.insert(lr_key);
        LRPlacement lr(tensor, 0, tiles, k, pi);
        auto it = tile_maps[pi].find(tensor);
        if (it != tile_maps[pi].end()) lr.vgpr_tile_map = it->second;
        ops.push_back(std::move(lr));
      }
    }
    ops.push_back(WaitLROp(false));
    MaskKOp mask(k);
    mask.vgpr_tile_map = merged_tile_map;
    ops.push_back(mask);
    // All partitions' MFMAs for this k, back-to-back.
    for (int pi = 0; pi < numP; ++pi) {
      PartRange cur = partition_tile_range(pi);
      MFMAPlacement mfma(k, MFMATileRange(k, k + 1, cur.A.first, cur.A.second),
                         MFMATileRange(k, k + 1, cur.B.first, cur.B.second));
      mfma.vgpr_tile_maps = tile_maps[pi];
      ops.push_back(std::move(mfma));
    }
    if (k != numK - 1) {
      ops.push_back(SkipOp("LE", miK * (k + 1), "SkipTailLoopL", true,
                           "early-exit tail after subIterK=" +
                               std::to_string(k) + " (no valid K left)"));
    }
    groups.push_back(to_emitted_value(ops));
  }

  tailloop_emitted = ModuleGrid{std::move(groups)};
}

// ════════════════════════════════════════════════════════════
// value_partitions — export the pass model as bound value types
// ════════════════════════════════════════════════════════════

// Convert a pass-pipeline Dep (identity pointer to a Placement) into the value
// Dep of the enclosing namespace, copying only the referenced placement's
// coordinate/identity fields. Deps only ever reference LR or GR placements.
inline tw::subtile::lsched::Dep to_value_dep(const Dep& d) {
  const Placement* r = d.ref;
  if (r->kind == PKind::LR) {
    tw::subtile::lsched::LRPlacement lr(r->tensor, r->mtIteration, r->tiles,
                                        r->subIterK_slot, r->partition);
    return tw::subtile::lsched::Dep(std::move(lr), d.mt_offset);
  }
  tw::subtile::lsched::GRPlacement gr(r->tensor, r->mtIteration, r->tiles,
                                      r->subIterK_slot, r->partition);
  return tw::subtile::lsched::Dep(std::move(gr), d.mt_offset);
}

// Convert a pass-pipeline before-chain Op into the value BeforeOp variant.
inline tw::subtile::lsched::BeforeOp to_value_op(const Op& o) {
  using namespace tw::subtile::lsched;
  if (o.kind == "wait_gr") {
    std::optional<WaitGRCounts> c;
    if (o.hasCounts) c = o.counts;
    return WaitGROp(std::move(c), o.has_sync, o.adjustVmcnt);
  }
  if (o.kind == "wait_lr") return WaitLROp(o.has_sync);
  if (o.kind == "sync") return SyncOp();
  if (o.kind == "mask_k") return MaskKOp(o.subIterK);
  if (o.kind == "lr_inc") return LRIncOp(o.tensor);
  if (o.kind == "gr_inc") return GRIncOp(o.tensor);
  throw std::runtime_error("unexpected C++ before-chain op kind: '" + o.kind +
                           "'");
}

inline std::vector<std::vector<tw::subtile::lsched::SubIterKSlot>>
LogicalScheduler::value_partitions() const {
  using namespace tw::subtile::lsched;
  std::vector<std::vector<SubIterKSlot>> out;
  out.reserve(partitions.size());
  for (const auto& slots : partitions) {
    std::vector<SubIterKSlot> vslots;
    vslots.reserve(slots.size());
    for (const auto& cs : slots) {
      SubIterKSlot vs(cs.subIterK);
      if (cs.mfma) {
        MFMAPlacement m(cs.mfma->subIterK, cs.mfma->tileA, cs.mfma->tileB);
        for (const auto& d : cs.mfma->deps) m.deps.push_back(to_value_dep(d));
        for (const auto& o : cs.mfma->preOps) m.preOps.push_back(to_value_op(o));
        for (const auto& o : cs.mfma->postOps)
          m.postOps.push_back(to_value_op(o));
        m.vgpr_tile_maps = cs.mfma->vgpr_tile_maps;
        vs.mfma = std::move(m);
      }
      for (const auto* lr : cs.lrs) {
        LRPlacement v(lr->tensor, lr->mtIteration, lr->tiles, lr->subIterK_slot,
                      lr->partition);
        for (const auto& d : lr->deps) v.deps.push_back(to_value_dep(d));
        for (const auto& o : lr->preOps) v.preOps.push_back(to_value_op(o));
        for (const auto& o : lr->postOps) v.postOps.push_back(to_value_op(o));
        v.vgpr_tile_map = lr->vgpr_tile_map;
        vs.lrs.push_back(std::move(v));
      }
      for (const auto* gr : cs.grs) {
        GRPlacement v(gr->tensor, gr->mtIteration, gr->tiles, gr->subIterK_slot,
                      gr->partition);
        for (const auto& d : gr->deps) v.deps.push_back(to_value_dep(d));
        for (const auto& o : gr->preOps) v.preOps.push_back(to_value_op(o));
        for (const auto& o : gr->postOps) v.postOps.push_back(to_value_op(o));
        vs.grs.push_back(std::move(v));
      }
      vslots.push_back(std::move(vs));
    }
    out.push_back(std::move(vslots));
  }
  return out;
}

// ════════════════════════════════════════════════════════════
// print_* helpers (byte-identical to Python)
// ════════════════════════════════════════════════════════════

inline std::string LogicalScheduler::print_lr() const {
  std::ostringstream buf;
  buf << "MAINLOOP:\n";
  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    buf << "  Partition " << pi << ":\n";
    for (auto& slot : partitions[pi]) {
      buf << "    subIterK=" << slot.subIterK << ":\n";
      if (slot.mfma) {
        auto* m = slot.mfma;
        buf << "      MFMAs (MT n, subIterK " << m->subIterK << "  ) A : "
            << m->tileA.fmt_tiles() << " , B : " << m->tileB.fmt_tiles() << "\n";
      }
      for (auto* lr : slot.lrs) {
        buf << "      LR " << ljust2(lr->tensor) << " (MT "
            << fmt_mt(lr->mtIteration) << ", subIterK " << lr->tiles.fmt_k()
            << ") " << lr->tiles.fmt_tiles() << "\n";
      }
    }
  }
  return buf.str();
}

inline std::string LogicalScheduler::print_vgpr() const {
  std::ostringstream buf;
  buf << "needsUnrolling: " << (needs_unrolling ? "True" : "False")
      << ", unrollFactor: " << unroll_factor << "\n";
  std::string peaks_str;
  bool first = true;
  for (auto& [t, cnt] : tile_peaks) {  // std::map already sorted by key
    if (!first) peaks_str += ", ";
    first = false;
    peaks_str += t + ": " + std::to_string(cnt);
  }
  buf << "vgprTiles: " << peaks_str << "\n";
  for (int ui = 0; ui < unroll_factor; ++ui) {
    if (unroll_factor > 1)
      buf << "MAINLOOP (unroll " << ui << "):\n";
    else
      buf << "MAINLOOP:\n";
    for (int pi = 0; pi < (int)partitions.size(); ++pi) {
      buf << "  Partition " << pi << ":\n";
      for (auto& slot : partitions[pi]) {
        buf << "    subIterK=" << slot.subIterK << ":\n";
        if (slot.mfma) {
          auto* m = slot.mfma;
          std::string tiles_str;
          std::vector<std::string> parts;
          for (auto& tensor : tensors) {
            auto it = m->vgpr_tile_maps.find(tensor);
            if (it != m->vgpr_tile_maps.end() && !it->second.empty()) {
              parts.push_back(tensor + ":" + dict_repr(it->second[ui]));
            }
          }
          if (!parts.empty()) {
            tiles_str = " ";
            for (size_t i = 0; i < parts.size(); ++i) {
              if (i) tiles_str += ", ";
              tiles_str += parts[i];
            }
          }
          buf << "      MFMAs (MT n, subIterK " << m->subIterK << "  ) A : "
              << m->tileA.fmt_tiles() << " , B : " << m->tileB.fmt_tiles()
              << tiles_str << "\n";
        }
        for (auto* lr : slot.lrs) {
          std::string tile_str;
          if (!lr->vgpr_tile_map.empty())
            tile_str = " tiles:" + dict_repr(lr->vgpr_tile_map[ui]);
          buf << "      LR " << ljust2(lr->tensor) << " (MT "
              << fmt_mt(lr->mtIteration) << ", subIterK " << lr->tiles.fmt_k()
              << ") " << lr->tiles.fmt_tiles() << tile_str << "\n";
        }
      }
    }
  }
  return buf.str();
}

inline std::string LogicalScheduler::print_gr() const {
  std::ostringstream buf;
  buf << "MAINLOOP:\n";
  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    buf << "  Partition " << pi << ":\n";
    for (auto& slot : partitions[pi]) {
      buf << "    subIterK=" << slot.subIterK << ":\n";
      if (slot.mfma) {
        auto* m = slot.mfma;
        buf << "      MFMAs (MT n, subIterK " << m->subIterK << "  ) A : "
            << m->tileA.fmt_tiles() << " , B : " << m->tileB.fmt_tiles() << "\n";
      }
      for (auto* lr : slot.lrs) {
        buf << "      LR " << ljust2(lr->tensor) << " (MT "
            << fmt_mt(lr->mtIteration) << ", subIterK " << lr->tiles.fmt_k()
            << ") " << lr->tiles.fmt_tiles() << "\n";
      }
      for (auto* gr : slot.grs) {
        buf << "      GR " << gr->tensor << " (MT " << fmt_mt(gr->mtIteration)
            << ", subIterK " << gr->tiles.fmt_k() << ") ids "
            << gr->tiles.fmt_tiles() << "\n";
      }
    }
  }
  return buf.str();
}

inline std::string LogicalScheduler::format_dep_ref(const Dep& dep) const {
  Placement* p = dep.ref;
  std::string kind = (p->kind == PKind::LR) ? "LR" : "GR";
  std::string mt =
      dep.mt_offset != 0 ? (" (MT" + std::to_string(dep.mt_offset) + ")") : "";
  return kind + " " + p->tensor + " @P" + std::to_string(p->partition) +
         ":subIterK=" + std::to_string(p->subIterK_slot) + mt;
}

inline void LogicalScheduler::print_placement_with_deps(std::ostringstream& buf,
                                                        const Placement* p) const {
  buf << "      " << p->str() << "\n";
  if (!p->deps.empty()) {
    buf << "        deps:\n";
    for (auto& dep : p->deps) buf << "            - " << format_dep_ref(dep) << "\n";
  }
}

inline void LogicalScheduler::print_placement_with_preops(
    std::ostringstream& buf, const Placement* p) const {
  buf << "      " << p->str() << "\n";
  if (!p->preOps.empty()) {
    buf << "        preOps:\n";
    for (auto& op : p->preOps) buf << "            - " << op.str() << "\n";
  }
  if (!p->deps.empty()) {
    buf << "        deps:\n";
    for (auto& dep : p->deps) buf << "            - " << format_dep_ref(dep) << "\n";
  }
  if (!p->postOps.empty()) {
    buf << "        postOps:\n";
    for (auto& op : p->postOps) buf << "            - " << op.str() << "\n";
  }
}

inline std::string LogicalScheduler::print_deps() const {
  std::ostringstream buf;
  buf << "MAINLOOP:\n";
  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    buf << "  Partition " << pi << ":\n";
    for (auto& slot : partitions[pi]) {
      buf << "    subIterK=" << slot.subIterK << ":\n";
      if (slot.mfma) print_placement_with_deps(buf, slot.mfma);
      for (auto* lr : slot.lrs) print_placement_with_deps(buf, lr);
      for (auto* gr : slot.grs) print_placement_with_deps(buf, gr);
    }
  }
  return buf.str();
}

inline std::string LogicalScheduler::print_remove_deps() const {
  std::ostringstream buf;
  buf << "MAINLOOP:\n";
  for (int pi = 0; pi < (int)partitions.size(); ++pi) {
    buf << "  Partition " << pi << ":\n";
    for (auto& slot : partitions[pi]) {
      buf << "    subIterK=" << slot.subIterK << ":\n";
      if (slot.mfma) print_placement_with_preops(buf, slot.mfma);
      for (auto* lr : slot.lrs) print_placement_with_preops(buf, lr);
      for (auto* gr : slot.grs) print_placement_with_preops(buf, gr);
    }
  }
  return buf.str();
}

inline std::string LogicalScheduler::print_group_lr_gr() const {
  return print_remove_deps();  // identical body in Python
}

inline std::string LogicalScheduler::print_emit() const {
  using tw::subtile::lsched::emittable_str;
  std::ostringstream buf;
  buf << "MAINLOOP:\n";
  for (int pi = 0; pi < (int)emitted.size(); ++pi) {
    buf << "  Partition " << pi << ":\n";
    for (int k = 0; k < (int)emitted[pi].size(); ++k) {
      buf << "    subIterK=" << k << ":\n";
      for (auto& em : emitted[pi][k]) {
        std::string before_str =
            em.before.has_value() ? (" <- [" + std::to_string(*em.before) + "]")
                                  : "";
        std::string src_str =
            em.source.has_value() ? emittable_str(*em.source) : std::string();
        buf << "      [" << rjust(std::to_string(em.moduleId), 2) << "] "
            << ljust(em.opType(), 10) << " " << src_str << before_str << "\n";
      }
    }
  }
  return buf.str();
}

}  // namespace tw::subtile::lsched::passes
