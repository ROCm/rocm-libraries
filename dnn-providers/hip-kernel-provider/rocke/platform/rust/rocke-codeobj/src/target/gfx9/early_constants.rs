// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Constants read where they are not in scope (gfx950). A builder can define an
//! `arith.constant` after the region that reads it (dsa_indexer), or inside a
//! nested region (an `scf.for` body) and reuse it after that region closes
//! (attention fwd with LSE/ext masks). The LLVM lowering writes a constant's
//! value into each use instead of an instruction, so COMGR never sees either;
//! the parser here resolves names in text order and by open region, and
//! refuses both. When it does, `reorder` moves each such constant's line to
//! the start of the kernel's body, which dominates every use, and the text
//! parses again. A constant has no operands, so the move changes no value.

use std::collections::HashMap;

/// `text` with each `arith.constant` line that has a use outside its scope
/// (before it in text order, or after the region that defines it closed)
/// moved to the start of the kernel's body; `None` when there is none.
pub fn reorder(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let entry = lines.iter().position(|l| l.trim() == "region @entry {")?;
    // Region ids of the open regions at each line; constants: name -> (line, region id), None if defined twice.
    let mut open = vec![0usize];
    let mut next = 1usize;
    let mut scopes: Vec<Vec<usize>> = Vec::with_capacity(lines.len() - entry);
    let mut defs: HashMap<&str, Option<(usize, usize)>> = HashMap::new();
    for (i, line) in lines.iter().enumerate().skip(entry + 1) {
        let t = line.trim();
        scopes.push(open.clone());
        if t == "}" {
            open.pop();
        } else if t.starts_with("region @") && t.ends_with('{') {
            open.push(next);
            next += 1;
        } else if let Some(name) = constant_name(t) {
            let at = (i, *open.last().unwrap_or(&0));
            defs.entry(name).and_modify(|d| *d = None).or_insert(Some(at));
        }
    }
    let mut moved = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(entry + 1) {
        for name in names(line) {
            if let Some(&Some((def, region))) = defs.get(name) {
                if (i < def || (i > def && !scopes[i - entry - 1].contains(&region))) && !moved.contains(&def) {
                    moved.push(def);
                }
            }
        }
    }
    if moved.is_empty() {
        return None;
    }
    moved.sort_unstable();
    let mut out = String::with_capacity(text.len());
    for (i, line) in lines.iter().enumerate() {
        if moved.binary_search(&i).is_ok() {
            continue;
        }
        out.push_str(line);
        if i == entry {
            for &m in &moved {
                out.push_str("    ");
                out.push_str(lines[m].trim_start());
            }
        }
    }
    Some(out)
}

/// The value a trimmed `%v : ty = arith.constant (` line defines.
fn constant_name(t: &str) -> Option<&str> {
    let (name, rest) = t.split_once(" : ")?;
    (name.starts_with('%') && rest.split_once(" = ").is_some_and(|(_, op)| op.starts_with("arith.constant ("))).then_some(name)
}

/// The `%name` tokens of `line`.
fn names(line: &str) -> impl Iterator<Item = &str> {
    let token = |c: char| c.is_alphanumeric() || matches!(c, '_' | '.' | '$' | '-');
    line.match_indices('%').map(move |(at, _)| {
        let rest = &line[at + 1..];
        &line[at..at + 1 + rest.find(|c| !token(c)).unwrap_or(rest.len())]
    })
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// `%c` read in a guarded region; defined first, or (`late`) after it.
    fn kernel(late: bool) -> String {
        let c = "    %c : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:7 }\n";
        format!(
            "ckdsl.ir v1\nkernel @early {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n    %n : i32\n  }}\n  \
             region @entry {{\n{}    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %lt : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    scf.if ( %lt )\n      region @then {{\n        \
             %s : i32 = arith.select ( %lt, %tid, %c )\n        \
             memref.global_store_typed ( %Y, %tid, %s ) {{ align = i:4, elem_type = s:\"i32\" }}\n      }}\n{}    \
             %c70 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:70 }}\n    cf.return (  )\n  }}\n}}\n",
            if late { "" } else { c },
            if late { c } else { "" }
        )
    }

    #[test]
    fn constants_used_before_their_definition_move_to_the_start() {
        assert_eq!(super::reorder(&kernel(true)).as_deref(), Some(kernel(false).as_str()));
        assert_eq!(super::reorder(&kernel(false)), None);
        let first = rir::compile_any_target(&kernel(false), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let late = rir::compile_any_target(&kernel(true), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(late.hsaco == first.hsaco);
        let e = rir::compile_any_target(&kernel(true), Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("used before definition"), "{e}");
    }

    /// `%c` defined in a guarded region and read after it (`inside`), or defined first.
    fn escaping(inside: bool) -> String {
        let c = "%c : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:7 }\n";
        let pad = |s: &str, n: usize| format!("{}{s}", " ".repeat(n));
        format!(
            "ckdsl.ir v1\nkernel @escape {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n    %n : i32\n  }}\n  \
             region @entry {{\n{}    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %lt : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    scf.if ( %lt )\n      region @then {{\n{}        \
             memref.global_store_typed ( %Y, %tid, %tid ) {{ align = i:4, elem_type = s:\"i32\" }}\n      }}\n    \
             %s : i32 = arith.select ( %lt, %tid, %c )\n    \
             memref.global_store_typed ( %Y, %tid, %s ) {{ align = i:4, elem_type = s:\"i32\" }}\n    cf.return (  )\n  }}\n}}\n",
            if inside { String::new() } else { pad(c, 4) },
            if inside { pad(c, 8) } else { String::new() },
        )
    }

    #[test]
    fn constants_read_after_their_region_closed_move_to_the_start() {
        assert_eq!(super::reorder(&escaping(true)).as_deref(), Some(escaping(false).as_str()));
        assert_eq!(super::reorder(&escaping(false)), None);
        let first = rir::compile_any_target(&escaping(false), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let inner = rir::compile_any_target(&escaping(true), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(inner.hsaco == first.hsaco);
    }

    #[test]
    fn constants_read_inside_their_region_stay() {
        let kept = escaping(true).replace("%s : i32 = arith.select ( %lt, %tid, %c )", "%s : i32 = arith.select ( %lt, %tid, %tid )");
        assert_eq!(super::reorder(&kept), None);
    }
}
