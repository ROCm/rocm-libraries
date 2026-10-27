// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Global memory alias facts from the IR's kernel-argument attributes, as
//! LLVM defines them: an access through a pointer based on a `noalias`
//! argument does not alias an access through a pointer not based on it, and
//! a `noalias readonly` argument's memory is not written while the kernel
//! runs.
//!
//! Isel tags each VMEM instruction (`MInst::mem`) with the argument its
//! address is based on, or `ANY`. The load sinker and the scheduler keep two
//! global accesses in order when one of them writes (an atomic writes) and
//! their tags may alias; fences and barriers order every access as before.
//! More loads in flight cost registers, so `compile` keeps a schedule the
//! facts allow only when it beats the one without them
//! (`schedule_with_facts` in `mod.rs`). `ROCKE_DIRECT_DEBUG=noalias` tags
//! everything `ANY`.

use super::ast::{Kernel, Op, OpKind, Type};
use super::mir::{MFunc, F_VMEM_LOAD, F_VMEM_STORE};

/// The tag of an access that may touch any global memory: its address is
/// not traced to exactly one `noalias` argument.
pub const ANY: u16 = 0;

/// Whether accesses tagged `a` and `b` may touch the same memory.
#[inline]
pub fn may_alias(a: u16, b: u16) -> bool {
    a == ANY || b == ANY || a == b
}

const UNTRACED: u32 = u32::MAX;

/// Ops that write global memory through their operand 0.
fn writes_through_operand0(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::GStore
            | OpKind::GAtomicAddF32
            | OpKind::GAtomicAdd
            | OpKind::BufferStoreVecF16
            | OpKind::BufferStoreScalar
            | OpKind::GlobalStore
            | OpKind::GlobalStoreVec
            | OpKind::GlobalAtomicAddF32
            | OpKind::GlobalAtomicAdd
    )
}

/// Per value, the tag of what a pointer or buffer resource addresses:
/// `1 + p` when it is based on argument `p` alone and `p` is `noalias`, else
/// `ANY`. A pointer is based on an argument through `tile.global_ptr_add` of
/// it and a resource through `tile.buffer_rsrc` of it; any other way to get
/// a pointer (a select, a loop-carried or merged value, a load, inline
/// assembly) is untraced. A write traced to a `readonly` argument
/// contradicts the attributes: then every value is `ANY`. Empty (every
/// value `ANY`) when no pointer argument is `noalias`.
pub fn value_tags(kernel: &Kernel) -> Vec<u16> {
    let pointer = |p: usize| kernel.params.get(p).is_some_and(|param| matches!(param.ty, Type::Ptr(_)));
    if !(0..kernel.params.len()).any(|p| pointer(p) && kernel.params[p].noalias) {
        return Vec::new();
    }
    let mut base = vec![UNTRACED; kernel.values.len()];
    for (p, param) in kernel.params.iter().enumerate() {
        if pointer(p) {
            base[param.value.0 as usize] = p as u32;
        }
    }
    let mut contradicted = false;
    super::analysis::preorder(&kernel.body, |op| {
        let operand0 = |base: &[u32]| op.operands.first().and_then(|v| base.get(v.0 as usize)).copied().unwrap_or(UNTRACED);
        let traced = match op.kind {
            OpKind::KernArg => match op.int_attr("param") {
                Some(p) if p >= 0 && pointer(p as usize) => p as u32,
                _ => UNTRACED,
            },
            OpKind::GlobalPtrAdd | OpKind::BufferRsrc => operand0(&base),
            _ => UNTRACED,
        };
        if let [result] = op.results[..] {
            if let Some(slot) = base.get_mut(result.0 as usize) {
                *slot = traced;
            }
        }
        if writes_through_operand0(op.kind) {
            let p = operand0(&base);
            contradicted |= p != UNTRACED && kernel.params[p as usize].readonly;
        }
    });
    let tag = |p: u32| match kernel.params.get(p as usize) {
        Some(param) if param.noalias && !contradicted && p < u16::MAX as u32 => p as u16 + 1,
        _ => ANY,
    };
    base.into_iter().map(tag).collect()
}

/// The tag of the global memory `op` accesses: that of its pointer or
/// buffer resource, operand 0 of every global and buffer access; `ANY` for
/// any other op.
pub fn op_tag(tags: &[u16], op: &Op) -> u16 {
    let access = matches!(
        op.kind,
        OpKind::GLoad | OpKind::GlobalLoad | OpKind::GlobalLoadVec | OpKind::BufferLoadVec | OpKind::BufferLoadScalar
    ) || writes_through_operand0(op.kind);
    match op.operands.first() {
        Some(v) if access => tags.get(v.0 as usize).copied().unwrap_or(ANY),
        _ => ANY,
    }
}

/// Whether the tags of `f` leave a pair of one block's global accesses
/// unordered that `ANY` tags order: a block with a tagged write and accesses
/// of two different tags. Otherwise the facts change no order.
pub fn relaxes(f: &MFunc) -> bool {
    f.blocks.iter().any(|block| {
        let (mut first, mut two, mut written) = (ANY, false, false);
        for inst in &block.insts {
            let flags = inst.op.flags();
            if flags & (F_VMEM_LOAD | F_VMEM_STORE) == 0 || inst.mem == ANY {
                continue;
            }
            written |= flags & F_VMEM_STORE != 0;
            if first == ANY {
                first = inst.mem;
            } else {
                two |= inst.mem != first;
            }
        }
        two && written
    })
}

/// `f` with every global access `ANY`: in the order it had without the facts.
pub fn without_facts(f: &MFunc) -> MFunc {
    let mut g = f.clone();
    for inst in g.blocks.iter_mut().flat_map(|b| b.insts.iter_mut()) {
        inst.mem = ANY;
    }
    g
}

/// Order of one tag's accesses since the last `ANY` write.
#[derive(Clone, Default)]
struct TagOrder {
    write: Option<u32>,
    reads: Vec<u32>,
    /// How many of the `ANY` reads (from the first) precede `write` already.
    ordered: usize,
}

/// What a block's dependence graph keeps of its earlier global accesses:
/// the last `ANY` write and the `ANY` reads since, and per tag its last
/// write and its reads since that `ANY` write, which follows everything
/// before it. With only `ANY` accesses it is the single class it replaces.
#[derive(Default)]
pub struct GlobalOrder {
    write: Option<u32>,
    reads: Vec<u32>,
    tags: Vec<TagOrder>,
    /// Tags with an access since the last `ANY` write.
    active: Vec<u16>,
}

impl GlobalOrder {
    /// Records access `i`, tagged `tag`, that reads (or writes: stores and
    /// atomics), and calls `edge` with each earlier access it must follow
    /// directly: every write and, for a write, every read it may alias,
    /// less those already ordered before it by earlier edges.
    #[inline]
    pub fn access(&mut self, i: u32, tag: u16, read: bool, mut edge: impl FnMut(u32)) {
        if let Some(w) = self.write {
            edge(w);
        }
        if tag == ANY {
            for &t in &self.active {
                let order = &self.tags[t as usize];
                if let Some(w) = order.write {
                    edge(w);
                }
                if !read {
                    order.reads.iter().for_each(|&r| edge(r));
                }
            }
            if read {
                self.reads.push(i);
                return;
            }
            self.reads.drain(..).for_each(&mut edge);
            for t in self.active.drain(..) {
                let order = &mut self.tags[t as usize];
                order.write = None;
                order.reads.clear();
                order.ordered = 0;
            }
            self.write = Some(i);
            return;
        }
        let t = tag as usize;
        if self.tags.len() <= t {
            self.tags.resize_with(t + 1, TagOrder::default);
        }
        let order = &mut self.tags[t];
        if order.write.is_none() && order.reads.is_empty() {
            self.active.push(tag);
        }
        if let Some(w) = order.write {
            edge(w);
        }
        if read {
            order.reads.push(i);
            return;
        }
        self.reads[order.ordered..].iter().for_each(|&r| edge(r));
        order.reads.drain(..).for_each(&mut edge);
        order.write = Some(i);
        order.ordered = self.reads.len();
    }
}

#[cfg(test)]
mod tests {
    use super::super::mir::*;
    use super::*;

    fn v(reg: u32, n: u8) -> Opnd {
        Opnd::R(VRef { reg, off: 0, n })
    }

    fn load(dst: u32, tag: u16) -> MInst {
        MInst::new(Opc::GLoadB128, [v(dst, 4)], [v(100, 2)]).with_imm(16 * dst as i64).with_mem(tag)
    }

    fn store(src: u32, tag: u16) -> MInst {
        MInst::new(Opc::GStoreB128, [], [v(101, 2), v(src, 4)]).with_imm(16 * src as i64).with_mem(tag)
    }

    fn func(insts: Vec<MInst>) -> MFunc {
        let mut vregs = vec![VRegInfo { class: Class::V, size: 4, align: 1, fixed: None }; 102];
        vregs[100].size = 2;
        vregs[101].size = 2;
        MFunc { blocks: vec![MBlock { insts, loop_depth: 0 }], vregs, ..Default::default() }
    }

    /// The block's (opcode, tag) after scheduling with a window of 16.
    fn scheduled(insts: Vec<MInst>) -> Vec<(Opc, u16)> {
        let mut f = func(insts);
        super::super::post::schedule_window(&mut f, 16);
        f.blocks[0].insts.iter().map(|i| (i.op, i.mem)).collect()
    }

    /// The block's (opcode, tag) after load-use sinking.
    fn sunk(insts: Vec<MInst>) -> Vec<(Opc, u16)> {
        let mut f = func(insts);
        super::super::post::sink_load_uses(&mut f, 32);
        f.blocks[0].insts.iter().map(|i| (i.op, i.mem)).collect()
    }

    fn ops(order: &[(Opc, u16)]) -> Vec<Opc> {
        order.iter().map(|x| x.0).collect()
    }

    /// Three pairs, each loading 16 bytes through tag `from` and storing
    /// them through `to`, at offsets 0, 16 and 32 of each.
    fn copies(from: u16, to: u16) -> Vec<MInst> {
        vec![load(1, from), store(1, to), load(2, from), store(2, to), load(3, from), store(3, to)]
    }

    const L: Opc = Opc::GLoadB128;
    const S: Opc = Opc::GStoreB128;

    #[test]
    fn loads_pass_stores_to_other_noalias_arguments() {
        assert_eq!(ops(&scheduled(copies(1, 2))), [L, L, L, S, S, S]);
        assert_eq!(ops(&sunk(copies(1, 2))), [L, L, L, S, S, S]);
        // Stores of one argument keep their order.
        let stores: Vec<i64> = {
            let mut f = func(copies(1, 2));
            super::super::post::schedule_window(&mut f, 16);
            f.blocks[0].insts.iter().filter(|i| i.op == S).map(|i| i.imm).collect()
        };
        assert_eq!(stores, [16, 32, 48]);
    }

    #[test]
    fn one_argument_stays_in_order_at_any_offsets() {
        // The same buffer through two offsets, an untraced load, an untraced
        // store, both untraced: the old order.
        for insts in [copies(1, 1), copies(ANY, 2), copies(1, ANY), copies(ANY, ANY)] {
            assert_eq!(ops(&scheduled(insts.clone())), [L, S, L, S, L, S]);
            assert_eq!(ops(&sunk(insts)), [L, S, L, S, L, S]);
        }
    }

    #[test]
    fn an_untraced_store_orders_every_later_load() {
        // Without the untraced store, the second load would issue first.
        let insts = vec![load(1, 1), store(1, ANY), load(2, 1), store(2, 2)];
        assert_eq!(scheduled(insts.clone()), [(L, 1), (S, ANY), (L, 1), (S, 2)]);
        assert_eq!(sunk(insts), [(L, 1), (S, ANY), (L, 1), (S, 2)]);
        // An untraced load after a tagged store follows it.
        let insts = vec![load(1, 1), store(1, 2), load(2, ANY), store(2, 3)];
        assert_eq!(scheduled(insts), [(L, 1), (S, 2), (L, ANY), (S, 3)]);
    }

    #[test]
    fn atomics_stay_ordered_within_their_argument_and_against_untraced() {
        let plain = |tag: u16| MInst::new(Opc::GAtomicAddF32, [], [v(101, 2), v(9, 1)]).with_mem(tag);
        let returning = |tag: u16| MInst::new(Opc::GAtomicAddF32Rtn, [v(7, 1)], [v(101, 2), v(9, 1)]).with_mem(tag);
        let load32 = |tag: u16| MInst::new(Opc::GLoadB32, [v(5, 1)], [v(100, 2)]).with_mem(tag);
        let use_it = MInst::new(Opc::VAddNcU32, [v(6, 1)], [v(5, 1), Opnd::Imm(1)]);
        for atomic in [plain, returning] {
            for (a, l, passes) in [(1, 1, false), (1, ANY, false), (ANY, 2, false), (1, 2, true)] {
                // The load heads the longest path, so it goes first when it may.
                let got = scheduled(vec![atomic(a), load32(l), use_it.clone()]);
                assert_eq!(got[0].0 == Opc::GLoadB32, passes, "{:?} {a}, load {l}", atomic(a).op);
            }
        }
    }

    #[test]
    fn barriers_and_fences_keep_global_order() {
        let fences = [
            MInst::new(Opc::SBarrier, [], []),
            MInst::new(Opc::SWaitcntVscnt, [Opnd::P(Phys::Null, 1)], []),
            MInst::new(Opc::BufferGl0Inv, [], []),
            MInst::new(Opc::SchedBarrier, [], []),
        ];
        for fence in fences {
            let insts = vec![load(1, 1), store(1, 2), fence, load(2, 1), store(2, 2)];
            let program: Vec<Opc> = insts.iter().map(|i| i.op).collect();
            assert_eq!(ops(&scheduled(insts.clone())), program);
            assert_eq!(ops(&sunk(insts)), program);
        }
    }

    #[test]
    fn lds_keeps_its_own_order() {
        // The global load heads the longest path and passes the LDS store;
        // the LDS load does not.
        let ds_store = MInst::new(Opc::DsStoreB32, [], [v(12, 1), v(13, 1)]);
        let ds_load = MInst::new(Opc::DsLoadB32, [v(14, 1)], [v(15, 1)]);
        let got = scheduled(vec![ds_store, ds_load, load(1, 1)]);
        assert_eq!(ops(&got), [L, Opc::DsStoreB32, Opc::DsLoadB32]);
    }

    #[test]
    fn facts_relax_only_blocks_writing_one_tag_beside_another() {
        let relaxes = |blocks: Vec<Vec<MInst>>| {
            let mut f = func(Vec::new());
            f.blocks = blocks.into_iter().map(|insts| MBlock { insts, loop_depth: 0 }).collect();
            super::relaxes(&f)
        };
        assert!(relaxes(vec![copies(1, 2)]));
        assert!(relaxes(vec![vec![load(1, ANY), load(2, 1), store(3, 2)]]));
        // One tag, untraced accesses, only reads, or the other tag in another block.
        assert!(!relaxes(vec![copies(1, 1)]));
        assert!(!relaxes(vec![copies(ANY, 2), copies(1, ANY)]));
        assert!(!relaxes(vec![vec![load(1, 1), load(2, 2), load(3, 3)]]));
        assert!(!relaxes(vec![vec![load(1, 1)], vec![store(1, 2)]]));
        let mut f = func(copies(1, 2));
        f = super::without_facts(&f);
        assert!(f.blocks[0].insts.iter().all(|i| i.mem == ANY) && !super::relaxes(&f));
    }

    /// Random access sequences: every pair that may alias, one of them a
    /// write, is ordered through the edges, and no edge joins any other pair.
    #[test]
    fn global_order_orders_exactly_the_conflicting_pairs() {
        let mut seed = 0x51f1_5eedu64;
        let mut next = |n: u64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        for _ in 0..400 {
            let n = 1 + next(48) as usize;
            let accesses: Vec<(u16, bool)> = (0..n).map(|_| (next(4) as u16, next(3) != 0)).collect();
            let conflict = |i: usize, j: usize| {
                let ((ti, ri), (tj, rj)) = (accesses[i], accesses[j]);
                !(ri && rj) && may_alias(ti, tj)
            };
            let mut order = GlobalOrder::default();
            let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
            for (i, &(tag, read)) in accesses.iter().enumerate() {
                order.access(i as u32, tag, read, |from| preds[i].push(from as usize));
            }
            // reach[j][i]: a path of edges leads from i to j.
            let mut reach = vec![vec![false; n]; n];
            for j in 0..n {
                for &p in &preds[j] {
                    assert!(p < j && conflict(p, j), "{accesses:?}: edge {p} -> {j}");
                    reach[j][p] = true;
                    for k in 0..p {
                        reach[j][k] |= reach[p][k];
                    }
                }
                let mut sorted = preds[j].clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), preds[j].len(), "{accesses:?}: repeated edge into {j}");
            }
            for j in 0..n {
                for i in 0..j {
                    assert!(!conflict(i, j) || reach[j][i], "{accesses:?}: {i} -> {j} unordered");
                }
            }
        }
    }

    fn kernel(params: &str, body: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n{params}  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    /// (opcode, tag) of every VMEM instruction isel selects for the kernel, sorted.
    fn vmem_tags(text: &str) -> Vec<(String, u16)> {
        use super::super::{analysis, isel, lds_swizzle, opt, parse};
        let mut kernel = parse::parse(text).unwrap_or_else(|e| panic!("{e}"));
        let (mut lds, lds_size) = analysis::lds_layout(&kernel);
        let facts = opt::optimize(&mut kernel, &mut lds).unwrap_or_else(|e| panic!("{e}"));
        let (facts, lds_size) = lds_swizzle::run(&mut kernel, facts, &mut lds, lds_size);
        let uniform = analysis::classify(&kernel);
        let sel = isel::select(&kernel, &facts, uniform, &lds, lds_size).unwrap_or_else(|e| panic!("{e}"));
        let mut out: Vec<(String, u16)> = sel
            .func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| i.op.flags() & (F_VMEM_LOAD | F_VMEM_STORE) != 0)
            .map(|i| (i.op.name(), i.mem))
            .collect();
        out.sort();
        out
    }

    const PARAMS: &str = "    %A : ptr<f32,global> { align = i:4, noalias = b:true, readonly = b:true }
    %B : ptr<f32,global> { align = i:4, noalias = b:true }
    %C : ptr<f32,global> { align = i:4 }
    %D : ptr<f32,global> { align = i:4, noalias = b:false, readonly = b:true }
    %n : i32
";

    #[test]
    fn isel_tags_each_access_with_its_argument() {
        let body = r#"    %a : f32 = memref.global_load_typed ( %A, %tid ) { align = i:4, elem_type = s:"f32" }
    %c : f32 = memref.global_load_typed ( %C, %tid ) { align = i:4, elem_type = s:"f32" }
    %d : f32 = memref.global_load_typed ( %D, %tid ) { align = i:4, elem_type = s:"f32" }
    %lt : i1 = arith.cmp ( %tid, %n ) { pred = s:"lt" }
    %p : ptr<f32,global> = arith.select ( %lt, %A, %B )
    %e : f32 = memref.global_load_typed ( %p, %tid ) { align = i:4, elem_type = s:"f32" }
    %c4 : i32 = arith.constant (  ) { ity = s:"i32", value = i:4 }
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %bo : i32 = arith.mul ( %tid, %c4 )
    %rs : vec<i32x4> = tile.buffer_rsrc ( %A, %n )
    %f : f32 = tile.buffer_load ( %rs, %bo, %c0 ) { elem_type = s:"f32" }
    %s1 : f32 = arith.fadd ( %a, %c )
    %s2 : f32 = arith.fadd ( %d, %e )
    %s3 : f32 = arith.fadd ( %s1, %s2 )
    %s4 : f32 = arith.fadd ( %s3, %f )
    memref.global_store_typed ( %B, %tid, %s4 ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %C, %tid, %s4 ) { align = i:4, elem_type = s:"f32" }
    memref.global_atomic_add_f32 ( %B, %n, %s4 ) { align = i:4 }
"#;
        let got = vmem_tags(&kernel(PARAMS, body));
        let want = |name: &str, tag: u16| (name.to_string(), tag);
        assert_eq!(
            got,
            [
                // %A's buffer resource; %B's atomic.
                want("buffer_load_b32", 1),
                want("global_atomic_add_f32", 2),
                // %A; %C (not noalias), %D (noalias false), the select (untraced).
                want("global_load_b32", ANY),
                want("global_load_b32", ANY),
                want("global_load_b32", ANY),
                want("global_load_b32", 1),
                want("global_store_b32", ANY),
                want("global_store_b32", 2),
            ]
        );
    }

    #[test]
    fn a_write_through_a_readonly_argument_drops_every_fact() {
        let body = r#"    %a : f32 = memref.global_load_typed ( %A, %tid ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %B, %tid, %a ) { align = i:4, elem_type = s:"f32" }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %t1 : i32 = arith.add ( %tid, %c1 )
    memref.global_store_typed ( %A, %t1, %a ) { align = i:4, elem_type = s:"f32" }
"#;
        let got = vmem_tags(&kernel(PARAMS, body));
        assert!(got.iter().all(|(_, tag)| *tag == ANY), "{got:?}");
        assert_eq!(got.len(), 3);
    }

    #[test]
    fn parameter_attributes_parse_leniently() {
        let facts = |attrs: &str| {
            let params = format!("    %X : ptr<f32,global> {attrs}\n");
            let k = super::super::parse::parse(&kernel(&params, "")).unwrap_or_else(|e| panic!("{e}"));
            (k.params[0].noalias, k.params[0].readonly)
        };
        assert_eq!(facts("{ align = i:4, noalias = b:true, readonly = b:true }"), (true, true));
        assert_eq!(facts("{ noalias = b:true }"), (true, false));
        assert_eq!(facts("{ readonly = b:true, noalias = b:false }"), (false, true));
        assert_eq!(facts("{ align = i:4 }"), (false, false));
        assert_eq!(facts(""), (false, false));
        // A repeated key holds only if every occurrence is true; a map that
        // does not parse states nothing (and is still ignored, as before).
        assert_eq!(facts("{ noalias = b:true, noalias = b:false }"), (false, false));
        assert_eq!(facts("{ noalias = b:true, readonly = oops }"), (false, false));
        assert_eq!(facts("{ noalias = i:1 }"), (false, false));
    }

    /// A copy between two buffers in eight 16-byte pieces, as `fmha_appendkv`.
    fn copy_kernel(attrs: &str) -> String {
        let params = format!("    %X : ptr<f16,global> {{ align = i:16{attrs} }}\n    %Y : ptr<f16,global> {{ align = i:16{attrs} }}\n");
        let mut body = String::from("    %c64 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:64 }\n    %row : i32 = arith.mul ( %tid, %c64 )\n");
        for k in 0..8 {
            body += &format!(
                "    %o{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    %i{k} : i32 = arith.add ( %row, %o{k} )\n    \
                 %v{k} : vec<f16x8> = memref.global_load_vN ( %X, %i{k} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n    \
                 memref.global_store_vN ( %Y, %i{k}, %v{k} ) {{ align = i:16, elem_type = s:\"f16\", vec = i:8 }}\n",
                8 * k
            );
        }
        kernel(&params, &body)
    }

    /// Global loads issued before the first global store, and the VMEM
    /// instructions in issue order.
    fn loads_ahead(text: &str) -> (usize, Vec<String>) {
        let compiled = super::super::compile(text).unwrap_or_else(|e| panic!("{e}"));
        let vmem: Vec<String> = compiled
            .func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| i.op.flags() & (F_VMEM_LOAD | F_VMEM_STORE) != 0)
            .map(|i| format!("{} {}", i.op.name(), i.imm))
            .collect();
        (vmem.iter().take_while(|i| i.starts_with("global_load")).count(), vmem)
    }

    #[test]
    fn a_schedule_the_facts_do_not_speed_up_is_the_one_without_them() {
        // One load and one store: the facts relax the pair, but no order
        // issues sooner, so the code is that of plain arguments.
        let params = |attrs: &str| format!("    %X : ptr<f32,global> {{ align = i:4{attrs} }}\n    %Y : ptr<f32,global> {{ align = i:4{attrs} }}\n");
        let body = r#"    %x : f32 = memref.global_load_typed ( %X, %tid ) { align = i:4, elem_type = s:"f32" }
    memref.global_store_typed ( %Y, %tid, %x ) { align = i:4, elem_type = s:"f32" }
"#;
        let compile = |attrs: &str| super::super::compile(&kernel(&params(attrs), body)).unwrap_or_else(|e| panic!("{e}")).hsaco;
        assert_eq!(compile(", noalias = b:true"), compile(""));
    }

    #[test]
    fn copies_between_noalias_buffers_keep_loads_in_flight() {
        let (plain, plain_order) = loads_ahead(&copy_kernel(""));
        let (noalias, noalias_order) = loads_ahead(&copy_kernel(", noalias = b:true"));
        assert_eq!(plain, 1, "{plain_order:?}");
        assert!(noalias >= 4, "{noalias_order:?}");
        // The same accesses, in another order.
        let (mut a, mut b) = (plain_order, noalias_order);
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
}
