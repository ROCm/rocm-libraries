// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! LDS DMA on gfx950 (design section 9): LDS addresses (`tile.smem_addr_of`,
//! `tile.smem_ptr_add`) and the buffer loads into LDS
//! (`tile.async_buffer_load_lds_addr`, `tile.async_buffer_load_lds`), as
//! rocKE's LLVM lowering defines them: `ptrtoint` of the allocation, an i64
//! add, and `llvm.amdgcn.raw.ptr.buffer.load.lds` with an instruction
//! offset of 0.
//!
//! An LDS address is an i64, but LDS pointers are 32 bits: only the low
//! dword addresses LDS. Where every use of an address reads no more than its
//! low dword (an offset sum, a DMA load, a truncation), selection computes
//! only that dword, as LLVM's demanded-bits narrowing does (`Facts::low`).
//!
//! A DMA load is `s_mov_b32 m0, <the first lane's address>` and
//! `buffer_load_dword{,x4} vaddr, srsrc, soffset offen lds`: lane L's data
//! goes to LDS at M0 + 4 * dwords * L. A divergent address is read from the
//! first active lane (`v_readfirstlane_b32`), as LLVM copies it to M0. M0 is
//! implicit in the load, whose rows are scheduling fences, and the move
//! names it, so neither is reordered with anything; H12 gives the move's
//! wait state (`hazards::fact`'s `m0_reader`).
//!
//! LDS tags (`MInst::mem` of the instructions that touch LDS): the KiB
//! range of LDS an instruction may touch, from the allocation its address is
//! based on (`Facts::provenance`). The `waitcnt` pass keys each DMA range on
//! vmcnt, and an LDS access waits only for the DMA loads whose range overlaps
//! its own (`overlaps`). `ANY` is all of LDS: an untraced DMA address, an
//! LDS spill slot.

use super::gen_isa::{self as t, Gfx9Op};
use crate::error::CompileError;
use crate::rir::alias::ANY;
use crate::rir::ast::{Kernel, OpKind, Scalar, Type, ValueId};
use crate::rir::fx::{FxHashMap, FxHashSet};
use crate::rir::mir::*;

/// The tag of LDS bytes `[start, start + bytes)`: `1 + (lo << 8 | hi)` for
/// the KiB range `[lo, hi)` that covers them; `ANY` past 255 KiB.
pub fn tag(start: u32, bytes: u32) -> u16 {
    let lo = start / 1024;
    let hi = (start as u64 + bytes as u64).div_ceil(1024).max(lo as u64 + 1);
    if hi > 255 {
        return ANY;
    }
    1 + (lo << 8 | hi as u32) as u16
}

/// Whether accesses tagged `a` and `b` may touch the same LDS bytes.
pub fn overlaps(a: u16, b: u16) -> bool {
    if a == ANY || b == ANY {
        return true;
    }
    let range = |x: u16| ((x - 1) >> 8, (x - 1) & 0xff);
    let ((alo, ahi), (blo, bhi)) = (range(a), range(b));
    alo < bhi && blo < ahi
}

/// The LDS bytes of an allocation of type `ty`.
pub fn alloc_bytes(ty: &Type) -> u32 {
    match ty {
        Type::Smem(elem, shape) => shape.iter().fold(elem.bytes(), |n, &d| n.saturating_mul(d)),
        _ => 0,
    }
}

/// Whether `op` is an LDS DMA load (the `lds` form).
pub fn is_dma(op: Opc) -> bool {
    t::info(op).is_some_and(|i| i.form == t::Form::Lds)
}

/// Whether an LDS instruction reads or writes LDS memory: all but the
/// cross-lane ones, which use the LDS hardware and no memory.
pub fn accesses_lds_memory(op: Opc) -> bool {
    op.flags() & F_LDS != 0 && !matches!(op, Opc::DsSwizzleB32 | Opc::DsBpermuteB32)
}

/// What selection reads of a kernel's LDS addresses.
#[derive(Default)]
pub struct Facts {
    /// The allocation each address is based on: `smem_addr_of`'s operand,
    /// carried through `smem_ptr_add` from its one based-on operand.
    pub provenance: FxHashMap<ValueId, ValueId>,
    /// The addresses whose every use reads no more than their low dword:
    /// an `smem_ptr_add` operand whose result is one, the LDS address of a
    /// DMA load, an `arith.trunc` to i32.
    pub low: FxHashSet<ValueId>,
}

impl Facts {
    pub fn of(kernel: &Kernel) -> Facts {
        let mut provenance = FxHashMap::default();
        let mut addresses: Vec<ValueId> = Vec::new();
        crate::rir::analysis::preorder(&kernel.body, |op| match op.kind {
            OpKind::SmemAddrOf => {
                provenance.insert(op.results[0], op.operands[0]);
                addresses.push(op.results[0]);
            }
            OpKind::SmemPtrAdd => {
                let based: Vec<ValueId> = op.operands.iter().filter_map(|v| provenance.get(v).copied()).collect();
                if let [alloc] = based[..] {
                    provenance.insert(op.results[0], alloc);
                }
                addresses.push(op.results[0]);
            }
            _ => {}
        });
        if addresses.is_empty() {
            return Facts::default();
        }
        // Uses of each address: (the using op's kind, the operand position,
        // its result). The greatest fixpoint drops an address with a use
        // that reads more, and then the operands of sums it is the result of.
        let address_set: FxHashSet<ValueId> = addresses.iter().copied().collect();
        let mut uses: FxHashMap<ValueId, Vec<(OpKind, usize, Option<ValueId>)>> = FxHashMap::default();
        crate::rir::analysis::preorder(&kernel.body, |op| {
            for (k, v) in op.operands.iter().enumerate() {
                if address_set.contains(v) {
                    let narrow = op.kind == OpKind::Trunc && op.results.first().is_some_and(|&r| *kernel.ty(r) == Type::Scalar(Scalar::I32));
                    let kind = if narrow { OpKind::Trunc } else { op.kind };
                    uses.entry(*v).or_default().push((kind, k, op.results.first().copied()));
                }
            }
        });
        let mut low: FxHashSet<ValueId> = address_set;
        loop {
            let dropped: Vec<ValueId> = low
                .iter()
                .copied()
                .filter(|v| {
                    uses.get(v).is_some_and(|list| {
                        list.iter().any(|&(kind, k, result)| match kind {
                            OpKind::BufferLoadLds => k != 1,
                            OpKind::GlobalLoadLds => k != 2,
                            OpKind::Trunc => false,
                            OpKind::SmemPtrAdd => !result.is_some_and(|r| low.contains(&r)),
                            _ => true,
                        })
                    })
                })
                .collect();
            if dropped.is_empty() {
                break;
            }
            for v in dropped {
                low.remove(&v);
            }
        }
        Facts { provenance, low }
    }
}

/// Where an LDS DMA load reads.
pub enum Source {
    /// A buffer: its resource (an SGPR quad), the lane's offset (a VGPR)
    /// and the wave's (an SGPR or an inline constant).
    Buffer { rsrc: VRef, voffset: Opnd, soffset: Opnd },
    /// Global memory: the lane's 64-bit address (a VGPR pair).
    Global { addr: VRef },
}

/// One LDS DMA load, its operands selected.
pub struct Dma {
    pub source: Source,
    /// The LDS address's low dword: an SGPR, a VGPR or a constant.
    pub lds: Opnd,
    /// Dwords per lane: 1 or 4.
    pub dwords: u8,
    /// The IR's `aux`: sc0 (1), nt (2), sc1 (16).
    pub aux: i64,
    /// The LDS tag of the bytes it may write.
    pub tag: u16,
}

/// Appends `dma` to `block` of `f`: M0 from the address, then the load.
pub fn select(f: &mut MFunc, block: usize, dma: Dma) -> Result<(), CompileError> {
    let (op, uses): (Gfx9Op, Uses) = match (&dma.source, dma.dwords) {
        (Source::Buffer { rsrc, voffset, soffset }, 1 | 4) => {
            let op = if dma.dwords == 1 { Gfx9Op::BufLoadLdsB32 } else { Gfx9Op::BufLoadLdsB128 };
            (op, [*voffset, Opnd::R(*rsrc), *soffset].into())
        }
        (Source::Global { addr }, 1 | 4) => {
            let op = if dma.dwords == 1 { Gfx9Op::GLoadLdsB32 } else { Gfx9Op::GLoadLdsB128 };
            (op, [Opnd::R(*addr)].into())
        }
        (_, n) => return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", format!("an LDS DMA load of {n} dwords per lane"))),
    };
    if dma.aux & !(1 | 2 | 16) != 0 {
        return Err(CompileError::new("ROCKE_DIRECT_UNSUPPORTED", format!("LDS DMA cache policy {:#x}", dma.aux)));
    }
    let vgpr = |o: Opnd| match o {
        Opnd::R(v) => f.vregs[v.reg as usize].class == Class::V,
        Opnd::P(Phys::V(_), _) => true,
        _ => false,
    };
    let m0_source = if vgpr(dma.lds) {
        let reg = f.vregs.len() as u32;
        f.vregs.push(VRegInfo { class: Class::S, size: 1, align: 1, fixed: None });
        let first = Opnd::R(VRef { reg, off: 0, n: 1 });
        f.blocks[block].insts.push(MInst::new(Opc::VReadfirstlaneB32, [first], [dma.lds]));
        first
    } else {
        dma.lds
    };
    let insts = &mut f.blocks[block].insts;
    insts.push(MInst::new(Opc::SMovB32, [Opnd::P(Phys::M0, 1)], [m0_source]));
    let cpol = (dma.aux & 1) | (dma.aux >> 1 & 1) << 1 | (dma.aux >> 4 & 1) << 2;
    insts.push(MInst::new(Opc::Gfx9(op), [], uses).with_imm(cpol << 12).with_mem(dma.tag));
    Ok(())
}

/// The LDS read a `tile.inline_asm` op names, `ds_read_b{32,64,128} $0, $1
/// [offset:N]` with constraints `=v,v`: its dwords and offset. rocKE emits it
/// for one GEMM; as an instruction of ours it is an LDS load like any other
/// (waited for, and waiting for the DMA loads into LDS).
pub fn asm_lds_read(op: &crate::rir::ast::Op) -> Option<(u8, i64)> {
    if op.kind != OpKind::InlineAsmIdentity || op.str_attr("constraints") != Some("=v,v") {
        return None;
    }
    let rest = op.str_attr("template")?.strip_prefix("ds_read_b")?;
    let (bits, rest) = rest.split_once(' ')?;
    let dwords = match bits {
        "32" => 1,
        "64" => 2,
        "128" => 4,
        _ => return None,
    };
    let offset = match rest.strip_prefix("$0, $1")? {
        "" => 0,
        tail => tail.strip_prefix(" offset:")?.parse::<i64>().ok().filter(|o| (0..65536).contains(o))?,
    };
    Some((dwords, offset))
}

/// Gives the LDS instructions of `block` from position `from` on (those
/// the selection of one LDS op appended) the tag `tag`.
pub fn tag_from(f: &mut MFunc, block: usize, from: usize, tag: u16) {
    for inst in f.blocks[block].insts.iter_mut().skip(from) {
        if accesses_lds_memory(inst.op) && !is_dma(inst.op) {
            inst.mem = tag;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rir;
    use crate::target::Target;

    #[test]
    fn tags_cover_their_kib_and_overlap_by_range() {
        let a = tag(0, 65536);
        let b = tag(65536, 65536);
        let c = tag(65536 + 100, 8);
        assert!(!overlaps(a, b) && overlaps(b, c) && !overlaps(a, c));
        // Ranges round out to KiB: neighbours within one KiB overlap.
        assert!(overlaps(tag(0, 64), tag(64, 64)));
        assert!(!overlaps(tag(0, 1024), tag(1024, 1)));
        assert!(overlaps(ANY, a) && overlaps(c, ANY));
        assert_eq!(tag(0, 0), tag(0, 1));
        assert_eq!(tag(250 * 1024, 8 * 1024), ANY);
        assert_ne!(tag(0, 1), ANY);
    }

    /// A kernel whose waves each DMA a 1 KiB slab of `%A` (four dwords per
    /// lane) and of `%B` (one dword), then, `sync` ("tile.sync" or a plain
    /// read), read back `%A` (`read_b` reads `%B` instead) and store it.
    fn dma_kernel(sync: &str, read_b: bool, aux: u32) -> String {
        let read = if read_b { "%B_smem" } else { "%A_smem" };
        format!(
            "ckdsl.ir v1\nkernel @dma {{\n  attrs {{ max_workgroup_size = i:256 }}\n  params {{\n    \
             %X : ptr<f32,global> {{ align = i:16, noalias = b:true, readonly = b:true }}\n    \
             %Y : ptr<f32,global> {{ align = i:16, noalias = b:true }}\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c64 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:64 }}\n    \
             %c16 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:16 }}\n    \
             %c4 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:4 }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1024 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1024 }}\n    \
             %c256 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:256 }}\n    \
             %size : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:65536 }}\n    \
             %A_smem : smem<f32, [1024]> = tile.smem_alloc (  )\n    \
             %B_smem : smem<f32, [1024]> = tile.smem_alloc (  )\n    \
             %rsrc : vec<i32x4> = tile.buffer_rsrc ( %X, %size )\n    \
             %a0 : i64 = tile.smem_addr_of ( %A_smem )\n    \
             %b0 : i64 = tile.smem_addr_of ( %B_smem )\n    \
             %wave : i32 = arith.div ( %tid, %c64 )\n    \
             %slab : i32 = arith.mul ( %wave, %c1024 )\n    \
             %zslab : i64 = arith.zext ( %slab )\n    \
             %a1 : i64 = tile.smem_ptr_add ( %a0, %zslab )\n    \
             %bslab : i32 = arith.mul ( %wave, %c256 )\n    \
             %zbslab : i64 = arith.zext ( %bslab )\n    \
             %b1 : i64 = tile.smem_ptr_add ( %b0, %zbslab )\n    \
             %voff : i32 = arith.mul ( %tid, %c16 )\n    \
             tile.async_buffer_load_lds_addr ( %rsrc, %a1, %voff, %c0 ) {{ aux = i:{aux}, dwords = i:4 }}\n    \
             %voff1 : i32 = arith.mul ( %tid, %c4 )\n    \
             tile.async_buffer_load_lds ( %rsrc, %b1, %voff1, %c0 ) {{ aux = i:0, dwords = i:1 }}\n\
             {sync}    \
             %v : f32 = tile.smem_load_vN ( {read}, %tid ) {{ elem_type = s:\"f32\", rank = i:1, vec = i:1 }}\n    \
             memref.global_store_typed ( %Y, %tid, %v ) {{ align = i:4, elem_type = s:\"f32\" }}\n    \
             cf.return (  )\n  }}\n}}\n"
        )
    }

    fn listing(text: &str) -> String {
        rir::compile_any_target(text, Target::Gfx950).unwrap_or_else(|e| panic!("{e}")).listing()
    }

    fn line_index(listing: &str, needle: &str) -> usize {
        listing.lines().position(|l| l.contains(needle)).unwrap_or_else(|| panic!("{needle} missing:\n{listing}"))
    }

    /// The encoded words of the first listing line containing `needle`.
    fn words(listing: &str, needle: &str) -> Vec<u32> {
        let line = listing.lines().find(|x| x.contains(needle)).unwrap_or_else(|| panic!("{needle} missing:\n{listing}"));
        let hex = line.rsplit(": ").next().unwrap_or("");
        hex.split_whitespace().map(|w| u32::from_str_radix(w, 16).expect("a hex word")).collect()
    }

    /// (vmcnt, lgkmcnt) of the `s_waitcnt` right before the first line
    /// containing `needle`, or `None` when that line follows something else.
    fn wait_before(listing: &str, needle: &str) -> Option<(u32, u32)> {
        let at = line_index(listing, needle);
        let line = listing.lines().nth(at - 1)?;
        if !line.trim_start().starts_with("s_waitcnt") {
            return None;
        }
        let imm = u32::from_str_radix(line.rsplit(": ").next()?.trim().get(4..)?, 16).ok()?;
        Some((imm & 0xf | (imm >> 14 & 3) << 4, imm >> 8 & 0xf))
    }

    #[test]
    fn dma_loads_set_m0_and_encode_as_llvm_mc_does() {
        let l = listing(&dma_kernel("    tile.sync (  )\n", false, 2));
        // M0 from the first lane of the per-wave address; a wait state (H12)
        // between each M0 write and its load: an s_nop where nothing else is.
        let lines: Vec<&str> = l.lines().collect();
        let movs: Vec<usize> = (0..lines.len()).filter(|&i| lines[i].contains("s_mov_b32 m0")).collect();
        assert_eq!(movs.len(), 2, "{l}");
        assert!(lines[..movs[0]].iter().any(|p| p.contains("v_readfirstlane_b32")), "{l}");
        for &m in &movs {
            let next = lines[m + 1].trim_start();
            assert!(next.starts_with("s_nop") || next.starts_with("s_waitcnt"), "H12:\n{l}");
            assert!(lines[m + 1..].iter().find(|x| !x.trim_start().starts_with('s')).is_some_and(|x| x.contains("buffer_load")), "{l}");
        }
        assert!(lines[movs[1] + 1].contains("s_nop"), "{l}");
        // As llvm-mc -mcpu=gfx950 encodes "buffer_load_dwordx4 vA, s[R:R+3], 0 offen nt lds"
        // (E05F1000, soffset 0 as code 0x80) and "buffer_load_dword vA, s[R:R+3], 0 offen lds" (E0511000).
        let (x4, x1) = (words(&l, "buffer_load_dwordx4"), words(&l, "buffer_load_dword "));
        assert_eq!((x4[0], x4[1] >> 24, x1[0], x1[1] >> 24), (0xE05F1000, 0x80, 0xE0511000, 0x80), "{l}");
        assert_eq!(x4[1] & 0xff00, 0, "no data register:\n{l}");
        // The barrier drains the DMA loads before other waves read LDS.
        assert_eq!(wait_before(&l, "s_barrier").map(|w| w.0), Some(0), "{l}");
    }

    #[test]
    fn an_lds_read_waits_for_the_dma_loads_of_its_allocation_only() {
        // No barrier: a read of %A waits for the DMA into %A (vmcnt(1): the
        // %B load issued after it may stay in flight); one of %B for that one.
        assert_eq!(wait_before(&listing(&dma_kernel("", false, 0)), "ds_read_b32"), Some((1, 15)));
        assert_eq!(wait_before(&listing(&dma_kernel("", true, 0)), "ds_read_b32"), Some((0, 15)));
        // An explicit wait for both loads already covers the read; the bare
        // barrier adds nothing; `sync_lds_only` drains LDS but not the DMA.
        let explicit = "    tile.s_waitcnt (  ) { expcnt = i:-1, lgkmcnt = i:-1, vmcnt = i:0 }\n    tile.s_barrier_bare (  )\n";
        let l = listing(&dma_kernel(explicit, false, 0));
        assert_eq!(wait_before(&l, "s_barrier"), Some((0, 15)), "{l}");
        assert_eq!(wait_before(&l, "ds_read_b32"), None, "{l}");
        let l = listing(&dma_kernel("    tile.sync_lds_only (  )\n", false, 0));
        assert_eq!(wait_before(&l, "ds_read_b32"), Some((1, 15)), "{l}");
    }

    #[test]
    fn addresses_used_as_lds_pointers_compute_their_low_dword_only() {
        let kernel = rir::parse::parse(&dma_kernel("", false, 0)).unwrap_or_else(|e| panic!("{e}"));
        let facts = Facts::of(&kernel);
        let named = |name: &str| (0..kernel.values.len() as u32).map(ValueId).find(|&v| kernel.name_of(v) == name).expect(name);
        for v in ["%a0", "%a1", "%b0", "%b1"] {
            assert!(facts.low.contains(&named(v)), "{v}");
        }
        assert_eq!(facts.provenance.get(&named("%a1")), Some(&named("%A_smem")));
        assert_eq!(facts.provenance.get(&named("%b1")), Some(&named("%B_smem")));
        // An address compared as an i64 is computed whole.
        let text = dma_kernel("", false, 0).replace(
            "    %voff : i32",
            "    %big : i1 = arith.cmp ( %a1, %zslab ) { pred = s:\"ult\" }\n    %voff : i32",
        );
        let kernel = rir::parse::parse(&text).unwrap_or_else(|e| panic!("{e}"));
        let facts = Facts::of(&kernel);
        let named = |name: &str| (0..kernel.values.len() as u32).map(ValueId).find(|&v| kernel.name_of(v) == name).expect(name);
        assert!(!facts.low.contains(&named("%a1")) && !facts.low.contains(&named("%a0")) && facts.low.contains(&named("%b1")));
    }

    #[test]
    fn dma_loads_are_m0_readers_for_h12() {
        for op in [Gfx9Op::BufLoadLdsB32, Gfx9Op::BufLoadLdsB128] {
            let inst = MInst::new(Opc::Gfx9(op), [], [Opnd::P(Phys::V(1), 1), Opnd::P(Phys::S(4), 4), Opnd::Imm(0)]);
            let fact = super::super::hazards::fact(&inst);
            assert!(fact.m0_reader && fact.kind == super::super::hazards::Kind::Vmem && fact.defs.is_empty(), "{op:?}");
            assert!(is_dma(Opc::Gfx9(op)) && op.flags() & (F_FENCE | F_LDS | F_VMEM_LOAD) == F_FENCE | F_LDS | F_VMEM_LOAD);
        }
        assert!(!is_dma(Opc::BufLoadB128) && accesses_lds_memory(Opc::DsLoadB32) && !accesses_lds_memory(Opc::DsSwizzleB32));
    }

    #[test]
    fn inline_asm_lds_reads_are_recognized_exactly() {
        let op = |template: &str, constraints: &str| {
            let mut op = crate::rir::ast::Op {
                kind: OpKind::InlineAsmIdentity,
                results: Default::default(),
                operands: Default::default(),
                attrs: Vec::new(),
                regions: Vec::new(),
                block_args: Vec::new(),
            };
            op.set_attr("template", crate::rir::ast::Attr::Str(template.to_string().into()));
            op.set_attr("constraints", crate::rir::ast::Attr::Str(constraints.to_string().into()));
            asm_lds_read(&op)
        };
        assert_eq!(op("ds_read_b128 $0, $1 offset:2048", "=v,v"), Some((4, 2048)));
        assert_eq!(op("ds_read_b64 $0, $1", "=v,v"), Some((2, 0)));
        assert_eq!(op("ds_read_b32 $0, $1 offset:65535", "=v,v"), Some((1, 65535)));
        for (t, c) in [("ds_read_b128 $0, $1 offset:65536", "=v,v"), ("ds_read_b128 $0, $1", "=v,0"), ("ds_read_b96 $0, $1", "=v,v"), ("", "=v,0"), ("ds_write_b32 $0, $1", "=v,v")] {
            assert_eq!(op(t, c), None, "{t} {c}");
        }
    }

    #[test]
    fn gfx1151_refuses_lds_dma() {
        let error = rir::compile(&dma_kernel("", false, 0)).err().expect("refused");
        assert!(error.to_string().contains("on gfx1151"), "{error}");
    }
}
