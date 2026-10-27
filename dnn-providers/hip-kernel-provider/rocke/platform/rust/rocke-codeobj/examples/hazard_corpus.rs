// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Offline corpus test of the gfx950 hazard recognizer (design section 5.3):
//! `cargo run --release --example hazard_corpus -- <report.tsv> <disassembly>...`.
//!
//! Each input is `llvm-objdump -d --mcpu=gfx950` text of a COMGR code object.
//! A test-only reader turns it into `hazards::Fact`s and blocks (split at
//! branches and their targets), every `s_nop` is deleted, and the recognizer
//! puts its own back as `insert_nops` does. At each site where COMGR waited,
//! ours must wait at least as long; a site where it waits less is a failure
//! (it names the instruction and its predecessors), one where it waits more
//! is listed for review. Exits 1 on any failure or unread instruction.

use rocke_codeobj_ffi::target::gfx9::hazards::{demand, Fact, Kind, Regs, EXEC, M0, VCC};

/// One register operand of the text, or `None` for constants and `off`.
fn reg(token: &str) -> Option<Regs> {
    let t = token.trim_start_matches('-').trim_matches('|');
    let t = t.split(['.', ' ']).next().unwrap_or(t);
    let (base, kind) = match t.chars().next()? {
        'v' if t != "vcc" && !t.starts_with("vcc_") => (0u16, &t[1..]),
        's' if t.len() > 1 && (t.as_bytes()[1].is_ascii_digit() || t.as_bytes()[1] == b'[') => (256, &t[1..]),
        'a' if t.len() > 1 && (t.as_bytes()[1].is_ascii_digit() || t.as_bytes()[1] == b'[') => (384, &t[1..]),
        _ => {
            return match t {
                "vcc" => Some(VCC),
                "vcc_lo" => Some(Regs { start: VCC.start, n: 1 }),
                "vcc_hi" => Some(Regs { start: VCC.start + 1, n: 1 }),
                "exec" => Some(EXEC),
                "exec_lo" => Some(Regs { start: EXEC.start, n: 1 }),
                "exec_hi" => Some(Regs { start: EXEC.start + 1, n: 1 }),
                "m0" => Some(M0),
                "flat_scratch" => Some(Regs { start: 256 + 102, n: 2 }),
                "xnack_mask" => Some(Regs { start: 256 + 104, n: 2 }),
                _ => None,
            }
        }
    };
    if let Some(range) = kind.strip_prefix('[').and_then(|k| k.strip_suffix(']')) {
        let (lo, hi) = range.split_once(':')?;
        let (lo, hi): (u16, u16) = (lo.parse().ok()?, hi.parse().ok()?);
        return Some(Regs { start: base + lo, n: hi - lo + 1 });
    }
    kind.parse::<u16>().ok().map(|r| Regs { start: base + r, n: 1 })
}

/// Operands split at top-level commas; the last one keeps only its first
/// word (the rest are modifiers).
fn operands(text: &str) -> (Vec<String>, String) {
    let mut out = Vec::new();
    let (mut depth, mut cur) = (0, String::new());
    for c in text.chars() {
        match c {
            '[' | '(' => depth += 1,
            ']' | ')' => depth -= 1,
            ',' if depth == 0 => {
                out.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    let last = cur.trim().to_string();
    let mut mods = String::new();
    if !last.is_empty() {
        // `v2 dst_sel:WORD_1 ...`: the operand, then the modifiers.
        let mut words = last.splitn(2, ' ');
        out.push(words.next().unwrap_or("").to_string());
        mods = words.next().unwrap_or("").to_string();
    }
    (out, mods)
}

const TRANS: [&str; 20] = [
    "v_exp_f32", "v_log_f32", "v_rcp_f32", "v_rcp_iflag_f32", "v_rsq_f32", "v_sqrt_f32", "v_sin_f32", "v_cos_f32",
    "v_exp_f16", "v_log_f16", "v_rcp_f16", "v_rsq_f16", "v_sqrt_f16", "v_sin_f16", "v_cos_f16", "v_rcp_f64",
    "v_rsq_f64", "v_sqrt_f64", "v_exp_legacy_f32", "v_log_legacy_f32",
];

/// Two destinations: carries, div_scale's flag, mad_u64's carry.
fn two_defs(base: &str) -> bool {
    base.contains("_co_u32") || base.starts_with("v_div_scale") || base.starts_with("v_mad_u64_u32") || base.starts_with("v_mad_i64_i32")
}

/// The `Fact` of one disassembled instruction, or `None` if unread.
fn read(mnemonic: &str, rest: &str) -> Option<Fact> {
    let (ops, mods) = operands(rest);
    let base = ["_e32", "_e64", "_sdwa", "_dpp"].iter().fold(mnemonic, |m, s| m.strip_suffix(s).unwrap_or(m));
    let regs_of = |list: &[String]| list.iter().filter_map(|o| reg(o)).collect::<Vec<_>>();
    let mut f = Fact { wait_states: 1, ..Fact::default() };
    if mnemonic == "s_nop" {
        let n: u32 = rest.trim().trim_start_matches("0x").parse().ok().or_else(|| u32::from_str_radix(rest.trim().trim_start_matches("0x"), 16).ok())?;
        f.nop = true;
        f.wait_states = n + 1;
        return Some(f);
    }
    if mnemonic.starts_with("v_") {
        f.kind = Kind::Valu;
        f.trans = TRANS.contains(&base);
        f.dpp = mnemonic.ends_with("_dpp") || mods.contains("quad_perm") || mods.contains("row_");
        let sdwa_dst = mnemonic.ends_with("_sdwa") && mods.contains("dst_sel:") && !mods.contains("dst_sel:DWORD");
        let mix = base.contains("_mix");
        let vop3p = base.starts_with("v_pk_") || mix || base.starts_with("v_dot");
        // The mix ops writing one half of their destination; a VOP3P with
        // src0's op_sel_hi set (the default of packed ops, explicit for mix
        // ops); a VOP3 op_sel whose last (destination) bit is set.
        let half_mix = base.contains("_mixlo") || base.contains("_mixhi");
        let src0_hi = match mods.split("op_sel_hi:[").nth(1) {
            Some(list) => list.starts_with('1'),
            None => !mix,
        };
        let opsel_dst = !vop3p && mods.contains("op_sel:[") && mods.contains(",1]");
        f.forwards_dst = sdwa_dst || half_mix || (vop3p && src0_hi) || opsel_dst;
        f.div_fmas = base.starts_with("v_div_fmas");
        f.reads_vcc = f.div_fmas;
        f.lane_read = matches!(base, "v_readlane_b32" | "v_readfirstlane_b32");
        f.writelane = base == "v_writelane_b32";
        f.lane_select = matches!(base, "v_readlane_b32" | "v_writelane_b32");
        if ops.is_empty() || base == "v_nop" {
            return Some(f);
        }
        let ndefs = if two_defs(base) { 2 } else { 1 };
        f.defs = regs_of(&ops[..ndefs.min(ops.len())]);
        if base.starts_with("v_cmpx") {
            f.defs.push(EXEC);
        }
        let srcs = &ops[ndefs.min(ops.len())..];
        f.uses = regs_of(srcs);
        if f.writelane {
            f.uses.extend(f.defs.clone());
            // LLVM lowers an SGPR-tuple spill to v_writelanes, the first of
            // which implicitly defines the whole tuple (`spillSGPR`), so its
            // recognizer sees a VALU write of the pair: the text cannot show
            // that operand, so the reader models it.
            if let Some(src) = srcs.first().and_then(|o| reg(o)).filter(|r| r.is_sgpr() && r.n == 1 && r.start % 2 == 0) {
                f.defs.push(Regs { start: src.start, n: 2 });
            }
        }
        f.src0 = srcs.first().and_then(|o| reg(o));
        f.src1 = srcs.get(1).and_then(|o| reg(o));
        return Some(f);
    }
    for (prefix, kind) in [("global_", Kind::Vmem), ("buffer_", Kind::Vmem), ("scratch_", Kind::Vmem), ("flat_", Kind::Vmem), ("ds_", Kind::Lds)] {
        if !mnemonic.starts_with(prefix) {
            continue;
        }
        f.kind = kind;
        let returns = mods.contains("sc0") || mods.contains("glc");
        let atomic = mnemonic.contains("atomic");
        f.store = kind == Kind::Vmem && (mnemonic.contains("store") || atomic);
        let has_def = if kind == Kind::Lds {
            mnemonic.contains("read") || mnemonic.contains("load") || mnemonic.contains("bpermute") || mnemonic.contains("swizzle") || mnemonic.contains("_rtn")
        } else {
            !f.store || (atomic && returns)
        };
        let ndefs = usize::from(has_def);
        f.defs = regs_of(&ops[..ndefs.min(ops.len())]);
        f.uses = regs_of(&ops[ndefs.min(ops.len())..]);
        if kind == Kind::Vmem && f.store && !has_def {
            // global/scratch: vaddr, vdata, saddr; buffer: vdata, vaddr, srsrc, soffset.
            let data = if prefix == "buffer_" { ops.first() } else { ops.get(1) };
            f.wide_data = data.and_then(|o| reg(o)).filter(|r| r.n > 2);
            f.buffer_sgpr_soffset = prefix == "buffer_" && ops.get(3).and_then(|o| reg(o)).is_some_and(|r| r.is_sgpr());
        }
        return Some(f);
    }
    if mnemonic.starts_with("s_load") || mnemonic.starts_with("s_buffer_load") || mnemonic.starts_with("s_store") || mnemonic.starts_with("s_dcache") {
        f.kind = Kind::Smem;
        f.store = mnemonic.starts_with("s_store");
        let ndefs = usize::from(!f.store && !mnemonic.starts_with("s_dcache"));
        f.defs = regs_of(&ops[..ndefs.min(ops.len())]);
        f.uses = regs_of(&ops[ndefs.min(ops.len())..]);
        return Some(f);
    }
    if mnemonic.starts_with("s_") {
        let sopp = ["s_waitcnt", "s_barrier", "s_branch", "s_cbranch", "s_endpgm", "s_setprio", "s_sleep", "s_sendmsg", "s_code_end", "s_trap"];
        if sopp.iter().any(|p| mnemonic.starts_with(p)) {
            return Some(f);
        }
        f.kind = Kind::Salu;
        let ndefs = usize::from(!(mnemonic.starts_with("s_cmp") || mnemonic.starts_with("s_bitcmp")));
        f.defs = regs_of(&ops[..ndefs.min(ops.len())]);
        f.uses = regs_of(&ops[ndefs.min(ops.len())..]);
        if mnemonic.contains("saveexec") {
            f.defs.push(EXEC);
            f.uses.push(EXEC);
        }
        return Some(f);
    }
    None
}

struct Inst {
    addr: u64,
    text: String,
    fact: Fact,
    branch: Option<(u64, bool)>,
    end: bool,
}

fn parse(text: &str) -> (Vec<Inst>, Vec<String>) {
    let (mut insts, mut unread) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let Some(body) = line.strip_prefix('\t') else { continue };
        let Some((asm, comment)) = body.split_once("//") else { continue };
        let asm = asm.trim();
        let Some((addr, words)) = comment.trim().split_once(':') else { continue };
        let Ok(addr) = u64::from_str_radix(addr.trim(), 16) else { continue };
        let word0 = words.split_whitespace().next().and_then(|w| u32::from_str_radix(w, 16).ok()).unwrap_or(0);
        let (mnemonic, rest) = asm.split_once(' ').unwrap_or((asm, ""));
        let Some(fact) = read(mnemonic, rest) else {
            unread.push(asm.to_string());
            continue;
        };
        let branch = if mnemonic == "s_branch" || mnemonic.starts_with("s_cbranch") {
            let delta = (word0 & 0xffff) as u16 as i16 as i64;
            Some(((addr as i64 + 4 + delta * 4) as u64, mnemonic == "s_branch"))
        } else {
            None
        };
        insts.push(Inst { addr, text: asm.to_string(), fact, branch, end: mnemonic == "s_endpgm" });
    }
    (insts, unread)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hazard_corpus <report.tsv> <disassembly>...");
        std::process::exit(2);
    }
    let mut report = String::from("kernel\taddress\tinstruction\tcomgr\tours\tverdict\tbefore\n");
    let (mut sites, mut under, mut over, mut unread_total, mut kernels) = (0, 0, 0, 0, 0);
    for path in &args[2..] {
        let text = std::fs::read_to_string(path).expect("read the disassembly");
        let name = std::path::Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let (insts, unread) = parse(&text);
        for u in &unread {
            report.push_str(&format!("{name}\t-\t{u}\t-\t-\tunread\t-\n"));
        }
        unread_total += unread.len();
        // COMGR's wait states before each non-nop instruction.
        let mut comgr = Vec::new();
        let mut code: Vec<&Inst> = Vec::new();
        let mut pending = 0;
        for inst in &insts {
            if inst.fact.nop {
                pending += inst.fact.wait_states;
            } else {
                comgr.push(pending);
                pending = 0;
                code.push(inst);
            }
        }
        // Stop at the first s_endpgm after which only padding follows.
        let last = code.iter().rposition(|i| i.end).map_or(code.len(), |p| p + 1);
        let code = &code[..last];
        let comgr = &comgr[..last];
        kernels += 1;
        // Blocks: split before branch targets and after branches and ends.
        let targets: std::collections::BTreeSet<u64> = code.iter().filter_map(|i| i.branch.map(|b| b.0)).collect();
        let mut starts = vec![0usize];
        for (k, inst) in code.iter().enumerate() {
            if k > 0 && targets.contains(&inst.addr) && *starts.last().unwrap() != k {
                starts.push(k);
            }
            if (inst.branch.is_some() || inst.end) && k + 1 < code.len() {
                starts.push(k + 1);
            }
        }
        starts.dedup();
        let block_of_addr = |a: u64| starts.iter().position(|&s| code[s].addr == a);
        let nblocks = starts.len();
        let mut blocks: Vec<Vec<Fact>> = Vec::with_capacity(nblocks);
        let mut preds = vec![Vec::new(); nblocks];
        for b in 0..nblocks {
            let (lo, hi) = (starts[b], if b + 1 < nblocks { starts[b + 1] } else { code.len() });
            blocks.push(code[lo..hi].iter().map(|i| i.fact.clone()).collect());
            let last = code[hi - 1];
            if let Some((target, always)) = last.branch {
                if let Some(t) = block_of_addr(target) {
                    preds[t].push(b);
                }
                if !always && b + 1 < nblocks {
                    preds[b + 1].push(b);
                }
            } else if !last.end && b + 1 < nblocks {
                preds[b + 1].push(b);
            }
        }
        // Ours: insert as the pass does, block by block, and compare.
        let mut index = 0;
        for b in 0..nblocks {
            let mut i = 0;
            while i < blocks[b].len() {
                let need = demand(&blocks, &preds, b, i);
                let (c, inst) = (comgr[index], code[index]);
                if c > 0 || need > 0 {
                    let verdict = if need < c {
                        under += 1;
                        "UNDER"
                    } else if need > c {
                        over += 1;
                        "over"
                    } else {
                        "same"
                    };
                    if c > 0 {
                        sites += 1;
                    }
                    let lo = index.saturating_sub(3);
                    let before: Vec<&str> = code[lo..index].iter().map(|i| i.text.as_str()).collect();
                    report.push_str(&format!("{name}\t{:#x}\t{}\t{c}\t{need}\t{verdict}\t{}\n", inst.addr, inst.text, before.join(" | ")));
                }
                if need > 0 {
                    let mut left = need;
                    while left > 0 {
                        let n = left.min(8);
                        blocks[b].insert(i, Fact { nop: true, wait_states: n, ..Fact::default() });
                        left -= n;
                        i += 1;
                    }
                }
                i += 1;
                index += 1;
            }
        }
    }
    std::fs::write(&args[1], &report).expect("write the report");
    println!("{kernels} kernels: {sites} COMGR s_nop sites; ours waits less at {under}, more at {over}; {unread_total} instructions unread");
    if under > 0 || unread_total > 0 {
        std::process::exit(1);
    }
}
