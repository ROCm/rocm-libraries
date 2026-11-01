// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Parse-time hash-consing of constants.
//!
//! A constant line whose text from the result type onwards repeats an
//! earlier constant that is still in scope is not built: its name is bound to
//! the earlier constant's value instead. The optimizer's first sweep would
//! have replaced both by the same canonical constant anyway, and the first
//! sweep is followed by a dense renumbering, so the kernel the optimizer
//! continues with is the same, value numbers included:
//!
//! - Equal text after the name means an equal type and equal attributes, so
//!   the sweep makes the same canonical constant of both, by its own rules.
//! - The earlier constant is in an enclosing (or the same) region, so the
//!   sweep visits it before the repeat and before every use of the repeat's
//!   name, and it never visits the repeat without it. The canonical constant
//!   is created at the first visit of its value, which is never a repeat.
//!   Visiting the repeat creates no value and changes no fact.
//! - The repeat's value would have been replaced, so it never survives the
//!   sweep; the values that do keep their order, and so their numbers after
//!   the renumbering.
//! - Nothing in the repeat's text can fail once the earlier one parsed.
//!
//! The parsed kernel is also validated and laid out before the sweep:
//!
//! - Validation checks a constant by itself, and reaches the earlier one
//!   first (same region, or an enclosing one, whose ops come first), so an
//!   error is the earlier one's either way. It also looks at the last op of
//!   regions (a loop body's yield), so a constant that ends its region is
//!   always built.
//! - The LDS layout compares positions of allocations, their uses and the
//!   last op of each loop, all of which are built, in the same order.
//!
//! Scoping follows the names: an entry is visible while the scope it was made
//! in is open. Because a constant is entered only when no visible entry
//! exists, a key needs one entry at a time.

use super::ast::ValueId;
use super::parse::name_hash;

/// `(name, key)` for a constant line in the DSL's printed shape,
/// `%name : type = arith.constant (  )...`: the key is the text after the
/// name. For any line accepted here, the parser's result type, attributes and
/// errors depend on the key alone.
pub(super) fn const_key(line: &str) -> Option<(&str, &str)> {
    let b = line.as_bytes();
    if b.first() != Some(&b'%') {
        return None;
    }
    let mut i = 1;
    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || b[i] == b'.') {
        i += 1;
    }
    if i == 1 || !b[i..].starts_with(b" : ") {
        return None;
    }
    let mut j = i + 3;
    while j < b.len() && b[j] != b' ' {
        j += 1;
    }
    if j == i + 3 || !b[j..].starts_with(b" = arith.constant (  )") {
        return None;
    }
    Some((&line[..i], &line[i..]))
}

#[derive(Clone, Copy)]
struct Entry<'a> {
    key: &'a str,
    hash: u32,
    value: ValueId,
    scope: u32,
}

/// Open-addressing table from constant keys to the value and scope of their
/// latest entry; at most half full.
pub(super) struct ConstTable<'a> {
    slots: Vec<u32>,
    entries: Vec<Entry<'a>>,
}

impl<'a> ConstTable<'a> {
    pub(super) fn new() -> Self {
        ConstTable { slots: vec![0; 64], entries: Vec::new() }
    }

    fn slot(&self, key: &str, hash: u32) -> Result<usize, usize> {
        let mask = self.slots.len() - 1;
        let mut i = hash as usize & mask;
        loop {
            let e = self.slots[i];
            if e == 0 {
                return Err(i);
            }
            let entry = &self.entries[e as usize - 1];
            if entry.hash == hash && entry.key == key {
                return Ok(i);
            }
            i = (i + 1) & mask;
        }
    }

    /// The value of the visible entry for `key`, if any.
    pub(super) fn lookup(&self, key: &str, hash: u32, open: impl Fn(u32) -> bool) -> Option<ValueId> {
        let i = self.slot(key, hash).ok()?;
        let entry = self.entries[self.slots[i] as usize - 1];
        open(entry.scope).then_some(entry.value)
    }

    /// Enters `key` with `value`, made in `scope`; replaces an entry that is
    /// no longer visible.
    pub(super) fn insert(&mut self, key: &'a str, hash: u32, value: ValueId, scope: u32) {
        let entry = Entry { key, hash, value, scope };
        match self.slot(key, hash) {
            Ok(i) => self.entries[self.slots[i] as usize - 1] = entry,
            Err(i) => {
                self.entries.push(entry);
                self.slots[i] = self.entries.len() as u32;
                if 2 * self.entries.len() > self.slots.len() {
                    self.grow();
                }
            }
        }
    }

    fn grow(&mut self) {
        let mut slots = vec![0u32; self.slots.len() * 2];
        let mask = slots.len() - 1;
        for (index, entry) in self.entries.iter().enumerate() {
            let mut i = entry.hash as usize & mask;
            while slots[i] != 0 {
                i = (i + 1) & mask;
            }
            slots[i] = index as u32 + 1;
        }
        self.slots = slots;
    }
}

/// The hash `ConstTable` keys by.
pub(super) fn key_hash(key: &str) -> u32 {
    name_hash(key)
}

#[cfg(test)]
mod tests {
    use super::super::{analysis, opt, parse, print, validate};
    use super::*;

    /// What the compiler makes of a parse before instruction selection:
    /// the error, or the LDS layout and the optimized kernel, value numbers
    /// included.
    fn front(parsed: Result<super::super::ast::Kernel, crate::CompileError>) -> String {
        let mut kernel = match parsed {
            Ok(k) => k,
            Err(e) => return format!("{}: {}", e.kind, e.message),
        };
        if let Err(e) = validate::kernel(&kernel) {
            return format!("{}: {}", e.kind, e.message);
        }
        let (mut lds, size) = analysis::lds_layout(&kernel);
        let mut layout: Vec<(u32, u32)> = lds.iter().map(|(v, &o)| (v.0, o)).collect();
        layout.sort_unstable();
        match opt::optimize(&mut kernel, &mut lds) {
            Ok(_) => format!("lds {size} {layout:?}\n{}", print::dump(&kernel)),
            Err(e) => format!("{}: {}", e.kind, e.message),
        }
    }

    /// Repeated constants in the same region, in nested and sibling regions,
    /// in a branch the optimizer drops, after a loop's yield, under shadowed
    /// names, and of the scalar types the DSL prints; LDS in a loop.
    fn kernel(crlf: bool) -> String {
        let c = |name: &str, ty: &str, value: &str| format!("    %{name} : {ty} = arith.constant (  ) {{ ity = s:\"{ty}\", value = {value} }}\n");
        let mut t = String::from("ckdsl.ir v1\nkernel @k {\n  attrs { max_workgroup_size = i:64 }\n  params {\n    %n : i32\n    %Y : ptr<f32,global>\n  }\n  region @entry {\n");
        t += "    %tid : i32 = gpu.thread_id (  ) { axis = s:\"x\" }\n";
        t += "    %s : smem<f32, [256]> = tile.smem_alloc (  )\n";
        t += &c("c0", "i32", "i:0");
        t += &c("c1", "i32", "i:1");
        t += &c("c0b", "i32", "i:0");
        t += &c("d", "f64", "f:2.0");
        t += &c("b8", "i8", "i:7");
        t += &c("w16", "i16", "i:-3");
        t += &c("d2", "f64", "f:2.0");
        t += &c("b8b", "i8", "i:7");
        t += "    %f : i1 = arith.cmp ( %c0, %c1 ) { pred = s:\"gt\" }\n";
        t += "    scf.if ( %f )\n      region @then {\n";
        t += &c("c7", "i32", "i:7");
        t += &c("c7b", "i32", "i:7");
        t += "        %u : i32 = arith.add ( %c7b, %tid )\n        memref.global_store_typed ( %Y, %u, %tid ) { align = i:4 }\n      }\n";
        t += &c("c7c", "i32", "i:7");
        t += "    %r : i32 = scf.for ( %c0b, %n, %c1, %c0 ) { iter_args = l:[ { name = s:\"%x\", type = s:\"i32\" } ], iv = s:\"%i\", iv_type = s:\"i32\", num_iter_args = i:1 }\n      region @body {\n";
        t += &c("c1b", "i32", "i:1");
        t += "        %y : i32 = arith.add ( %x, %c1b )\n        %ft : f32 = arith.bitcast ( %y )\n";
        t += "        tile.smem_store ( %s, %tid, %ft ) { align = i:4 }\n        tile.sync (  )\n";
        t += &c("c0", "i32", "i:0");
        t += "        %z : i32 = arith.add ( %y, %c0 )\n        scf.yield ( %z )\n      }\n";
        t += "    %g : vec<f32x1> = tile.smem_load_vN ( %s, %tid ) { vec = i:1 }\n    %e : f32 = vector.extract ( %g ) { index = i:0 }\n";
        t += &c("c1c", "i32", "i:1");
        t += "    %q : i32 = arith.add ( %r, %c1c )\n    memref.global_store_typed ( %Y, %q, %e ) { align = i:4 }\n";
        t += &c("c1d", "i32", "i:1");
        t += "    cf.return (  )\n  }\n}\n";
        if crlf {
            t = t.replace('\n', "\r\n");
        }
        t
    }

    fn agree(text: &str) {
        for (threads, chunk) in [(1, 0), (2, 0), (3, 64), (2, 1), (5, 300)] {
            let shared = front(parse::parse_chunked(text, threads, chunk));
            let plain = front(parse::parse_plain(text, threads, chunk));
            assert_eq!(shared, plain, "threads {threads} chunk {chunk}:\n{text}");
        }
    }

    #[test]
    fn repeats_change_nothing_the_optimizer_produces() {
        for crlf in [false, true] {
            let text = kernel(crlf);
            assert!(!front(parse::parse_plain(&text, 1, 0)).starts_with("ROCKE"), "{}", front(parse::parse_plain(&text, 1, 0)));
            agree(&text);
            // The repeats are shared: fewer values than written.
            let shared = parse::parse_chunked(&text, 1, 0).expect("parses").values.len();
            let plain = parse::parse_plain(&text, 1, 0).expect("parses").values.len();
            assert!(shared < plain, "{shared} {plain}");
        }
        // A constant the optimizer refuses fails the same way when repeated.
        let bf16 = "    %h : bf16 = arith.constant (  ) { ity = s:\"bf16\", value = f:1.5 }\n";
        let text = kernel(false).replace("    %f : i1", &format!("{bf16}{}    %f : i1", bf16.replace("%h ", "%h2 ")));
        assert!(front(parse::parse_plain(&text, 1, 0)).starts_with("ROCKE_DIRECT_UNSUPPORTED"));
        agree(&text);
    }

    /// A constant after a loop's yield ends the body: validation rejects the
    /// loop whether or not the constant repeats an earlier one.
    #[test]
    fn a_constant_ending_its_region_is_always_built() {
        let text = kernel(false).replace(
            "        scf.yield ( %z )\n      }\n",
            "        scf.yield ( %z )\n        %c1e : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:1 }\n      }\n",
        );
        let plain = front(parse::parse_plain(&text, 1, 0));
        assert!(plain.starts_with("ROCKE_DIRECT_MALFORMED"), "{plain}");
        agree(&text);
    }

    /// Every one-line deletion, duplication and swap: the same result with
    /// and without hash-consing.
    #[test]
    fn line_edits_agree() {
        let text = kernel(false);
        let lines: Vec<&str> = text.lines().collect();
        for i in 0..lines.len() {
            let mut deleted = lines.clone();
            deleted.remove(i);
            agree(&(deleted.join("\n") + "\n"));
            let mut duplicated = lines.clone();
            duplicated.insert(i, lines[i]);
            agree(&(duplicated.join("\n") + "\n"));
            if i + 1 < lines.len() {
                let mut swapped = lines.clone();
                swapped.swap(i, i + 1);
                agree(&(swapped.join("\n") + "\n"));
            }
        }
    }

    #[test]
    fn const_key_takes_only_the_printed_constant_shape() {
        let line = "%c12 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16 }";
        assert_eq!(const_key(line), Some(("%c12", &line[4..])));
        for other in [
            "%c : i32 = arith.add ( %a, %b )",
            "%c : i32 = arith.constant ( %a )",
            "c : i32 = arith.constant (  ) { value = i:1 }",
            "% : i32 = arith.constant (  ) { value = i:1 }",
            "%c :  = arith.constant (  )",
            "%c: i32 = arith.constant (  )",
            "%c : i32 =  arith.constant (  )",
            "%c : i32 = arith.constant_vec (  ) { fill = f:0.0 }",
        ] {
            assert_eq!(const_key(other), None, "{other:?}");
        }
    }

    #[test]
    fn entries_are_visible_while_their_scope_is_open() {
        let mut t = ConstTable::new();
        let open = [true, false, true];
        for k in 0..200u32 {
            let key: &'static str = Box::leak(format!(" : i32 = arith.constant (  ) {{ value = i:{k} }}").into_boxed_str());
            assert_eq!(t.lookup(key, key_hash(key), |s| open[s as usize]), None);
            t.insert(key, key_hash(key), ValueId(k), k % 3);
        }
        let key = " : i32 = arith.constant (  ) { value = i:7 }";
        assert_eq!(t.lookup(key, key_hash(key), |s| open[s as usize]), None, "scope 1 is closed");
        let key = " : i32 = arith.constant (  ) { value = i:9 }";
        assert_eq!(t.lookup(key, key_hash(key), |s| open[s as usize]), Some(ValueId(9)));
        t.insert(" : i32 = arith.constant (  ) { value = i:7 }", key_hash(" : i32 = arith.constant (  ) { value = i:7 }"), ValueId(700), 2);
        let key = " : i32 = arith.constant (  ) { value = i:7 }";
        assert_eq!(t.lookup(key, key_hash(key), |s| open[s as usize]), Some(ValueId(700)));
    }
}
