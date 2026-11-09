// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The parser's line scans driven by `scan::Masks`: each makes exactly the
//! decisions of its byte-by-byte twin in `parse` or `hashcons`, jumping over
//! runs of a byte class with bit operations.

use super::parse::{canonical, quick_type, Canonical};
use super::scan::{self, Level, Masks};

/// The masks of a line the scans should use, if any: at a vector level and
/// for a line `scan::classify` takes.
#[inline]
pub(super) fn masks(level: Level, line: &str) -> Option<Masks> {
    if level == Level::Scalar {
        return None;
    }
    scan::classify(level, line.as_bytes())
}

/// `parse::canonical`, with the line's masks when there are some.
pub(super) fn canonical_with<'a>(line: &'a str, masks: Option<&Masks>) -> Option<Canonical<'a>> {
    match masks {
        Some(m) => canonical_masked(line, m),
        None => canonical(line),
    }
}

/// `hashcons::const_key`, with the line's masks when there are some.
pub(super) fn const_key_with<'a>(line: &'a str, masks: Option<&Masks>) -> Option<(&'a str, &'a str)> {
    match masks {
        Some(m) => const_key_masked(line, m),
        None => super::hashcons::const_key(line),
    }
}

/// `parse::canonical` from the masks of `line`.
fn canonical_masked<'a>(line: &'a str, m: &Masks) -> Option<Canonical<'a>> {
    let b = line.as_bytes();
    let n = b.len();
    let mut i = 0;
    let mut result = None;
    if b.first() == Some(&b'%') {
        i = Masks::skip(m.name, 1, n);
        if i == 1 || !b[i..].starts_with(b" : ") {
            return None;
        }
        let ty_start = i + 3;
        let j = Masks::find(m.space, ty_start, n);
        if !b[j..].starts_with(b" = ") {
            return None;
        }
        result = Some((&line[..i], quick_type(&b[ty_start..j])?));
        i = j + 3;
    }
    let op_start = i;
    i = Masks::skip(m.name, i, n);
    if i == op_start || !b[i..].starts_with(b" ( ") {
        return None;
    }
    let opname = &line[op_start..i];
    i += 3;
    let operands_start = i;
    let close = if b[i..].starts_with(b" )") {
        i + 1
    } else {
        loop {
            if b.get(i) != Some(&b'%') {
                return None;
            }
            i += 1;
            let name_start = i;
            i = Masks::skip(m.name, i, n);
            if i == name_start {
                return None;
            }
            if b[i..].starts_with(b", ") {
                i += 2;
            } else if b[i..].starts_with(b" )") {
                break i + 1;
            } else {
                return None;
            }
        }
    };
    Some(Canonical {
        result,
        opname,
        operands: line.get(operands_start..close.saturating_sub(1).max(operands_start)).unwrap_or(""),
        tail: &line[close + 1..],
    })
}

/// `hashcons::const_key` from the masks of `line`.
fn const_key_masked<'a>(line: &'a str, m: &Masks) -> Option<(&'a str, &'a str)> {
    let b = line.as_bytes();
    if b.first() != Some(&b'%') {
        return None;
    }
    let n = b.len();
    let i = Masks::skip(m.name, 1, n);
    if i == 1 || !b[i..].starts_with(b" : ") {
        return None;
    }
    let j = Masks::find(m.space, i + 3, n);
    if j == i + 3 || !b[j..].starts_with(b" = arith.constant (  )") {
        return None;
    }
    Some((&line[..i], &line[i..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n.max(1) as u64) as usize
        }
    }

    const LINES: &[&str] = &[
        "%a : i32 = arith.add ( %b, %c )",
        "%c12 : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:16 }",
        "%v.x_1 : vec<f16x8> = tile.smem_load_vN ( %s, %i ) { elem_type = s:\"f16\", rank = i:1, vec = i:8 }",
        "%p : ptr<f16,global> = tile.global_ptr_add ( %K, %o )",
        "tile.sync (  ) { fence = s:\"compiler\" }",
        "scf.yield ( %x ) @loc(\"a.py\":3)",
        "%r : i32, %q : f32 = scf.for ( %a, %b, %c, %d, %e ) { iter_args = l:[ ] }",
        "%h : bf16 = arith.constant (  ) { ity = s:\"bf16\", value = f:1.5 }",
        "cf.return (  )",
    ];
    const PIECES: &[&str] = &[" ", ",", ", ", " ( ", " )", " = ", " : ", "%", "é", "\t", "_", ".", "x9", "{", "}", "(", ")", "\"", ""];

    /// Masked and byte-by-byte scans agree on printed lines and on every
    /// kind of damage to them, at every vector level the CPU has.
    #[test]
    fn masked_scans_match_the_byte_scans() {
        let mut rng = Rng(3);
        let mut checked = 0;
        for round in 0..60_000 {
            let mut line = LINES[rng.below(LINES.len())].to_string();
            for _ in 0..rng.below(4) {
                let cuts: Vec<usize> = (0..=line.len()).filter(|&c| line.is_char_boundary(c)).collect();
                let at = cuts[rng.below(cuts.len())];
                match rng.below(3) {
                    0 => line.insert_str(at, PIECES[rng.below(PIECES.len())]),
                    1 => line.truncate(at),
                    _ => {
                        let end = cuts.iter().copied().filter(|&c| c >= at).nth(rng.below(3)).unwrap_or(at);
                        line.replace_range(at..end, PIECES[rng.below(PIECES.len())]);
                    }
                }
            }
            if round % 7 == 0 {
                while line.len() < scan::MAX_LINE - 2 {
                    line.push_str(" x");
                }
            }
            for level in Level::supported().into_iter().filter(|&l| l != Level::Scalar) {
                let Some(m) = masks(level, &line) else { continue };
                checked += 1;
                let fast = canonical_masked(&line, &m).map(|c| (c.result, c.opname, c.operands, c.tail));
                let slow = canonical(&line).map(|c| (c.result, c.opname, c.operands, c.tail));
                assert_eq!(fast, slow, "{level:?} {line:?}");
                assert_eq!(const_key_masked(&line, &m), super::super::hashcons::const_key(&line), "{level:?} {line:?}");
            }
        }
        assert!(checked > 50_000 || Level::best() == Level::Scalar, "{checked}");
    }
}
