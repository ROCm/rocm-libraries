// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! SIMD byte classification for the parser: a structural index of one line
//! at a time, and newline search.
//!
//! `classify` turns a line of at most `MAX_LINE` bytes into bitmasks, one
//! bit per byte, of the classes the parser's scans stop at: name bytes
//! (`[A-Za-z0-9_.]`), spaces, value terminators (`,`, space, `}`, `]`, tab),
//! string stops (`"` and `\`), and `%` sigils. The scans then jump with bit
//! operations instead of testing byte by byte. `find_newline` finds line ends.
//!
//! Three implementations, chosen at run time by `Level::detect`: AVX-512
//! (BW), AVX2, and a scalar one that gives the same masks; the parser uses
//! its plain byte loops at `Level::Scalar`. All `unsafe` code of the parser
//! is in this module: the `std::arch` intrinsics, each behind a safe function
//! that checks the CPU feature first and only reads within its slice.

/// Longest line `classify` takes; longer lines are scanned byte by byte.
pub(super) const MAX_LINE: usize = 128;

/// The vector instructions the scans use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Level {
    Scalar,
    Avx2,
    Avx512,
}

impl Level {
    /// The best level this CPU supports. `ROCKE_DIRECT_SCAN=scalar|avx2|avx512`
    /// selects a lower one (a higher one than the CPU has is ignored).
    pub(super) fn detect() -> Level {
        let best = Level::best();
        match std::env::var("ROCKE_DIRECT_SCAN").ok().as_deref().map(str::trim) {
            Some("scalar") => Level::Scalar,
            Some("avx2") if best != Level::Scalar => Level::Avx2,
            _ => best,
        }
    }

    /// The best level this CPU supports.
    pub(super) fn best() -> Level {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
                return Level::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Level::Avx2;
            }
        }
        Level::Scalar
    }

    /// Every level this CPU supports, scalar first.
    #[cfg(test)]
    pub(super) fn supported() -> Vec<Level> {
        let best = Level::best();
        [Level::Scalar, Level::Avx2, Level::Avx512].into_iter().filter(|&l| l as u8 <= best as u8).collect()
    }
}

/// Byte classes of a line: bit `i` describes byte `i`; bits past the line
/// are clear.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Masks {
    /// `[A-Za-z0-9_.]`.
    pub name: u128,
    /// ` `.
    pub space: u128,
    /// `,`, ` `, `}`, `]` or tab: where an attribute value token ends.
    pub stop: u128,
    /// `"` or `\`: where a string's plain run ends.
    pub quote: u128,
    /// `%`.
    pub sigil: u128,
}

impl Masks {
    /// The first index at or after `i` (at most `len`) whose byte is not in
    /// `class`.
    #[inline]
    pub(super) fn skip(class: u128, i: usize, len: usize) -> usize {
        if i >= len {
            return len;
        }
        (i + (!(class >> i)).trailing_zeros() as usize).min(len)
    }

    /// The first index at or after `i` whose byte is in `class`, or `len`.
    #[inline]
    pub(super) fn find(class: u128, i: usize, len: usize) -> usize {
        if i >= len {
            return len;
        }
        (i + (class >> i).trailing_zeros() as usize).min(len)
    }
}

fn is_name(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'.'
}

/// The masks of `line`, or `None` if it is longer than `MAX_LINE`.
pub(super) fn classify(level: Level, line: &[u8]) -> Option<Masks> {
    if line.len() > MAX_LINE {
        return None;
    }
    let keep = if line.len() == MAX_LINE { u128::MAX } else { (1u128 << line.len()) - 1 };
    #[cfg(target_arch = "x86_64")]
    if level == Level::Avx512 && is_x86_feature_detected!("avx512bw") {
        // SAFETY: the CPU has AVX-512 BW (checked), and the function's masked
        // loads touch only the bytes of `line`.
        let m = unsafe { x86::classify_avx512(line) };
        return Some(m);
    }
    let mut block = [0u8; MAX_LINE];
    block[..line.len()].copy_from_slice(line);
    let masks = match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Avx512 if is_x86_feature_detected!("avx2") => {
            // SAFETY: the CPU has AVX2 (checked), and the function reads
            // exactly the 128 bytes of `block`.
            unsafe { x86::classify_avx2(&block) }
        }
        _ => classify_scalar(&block),
    };
    Some(Masks {
        name: masks.name & keep,
        space: masks.space & keep,
        stop: masks.stop & keep,
        quote: masks.quote & keep,
        sigil: masks.sigil & keep,
    })
}

/// `classify` byte by byte.
pub(super) fn classify_scalar(block: &[u8; MAX_LINE]) -> Masks {
    let mut m = Masks::default();
    for (i, &b) in block.iter().enumerate() {
        let bit = 1u128 << i;
        if is_name(b) {
            m.name |= bit;
        }
        if b == b' ' {
            m.space |= bit;
        }
        if matches!(b, b',' | b' ' | b'}' | b']' | b'\t') {
            m.stop |= bit;
        }
        if matches!(b, b'"' | b'\\') {
            m.quote |= bit;
        }
        if b == b'%' {
            m.sigil |= bit;
        }
    }
    m
}

/// The offset of the first `\n` in `text[from..]`, if any.
pub(super) fn find_newline(level: Level, text: &[u8], from: usize) -> Option<usize> {
    let rest = text.get(from..)?;
    let found = match level {
        #[cfg(target_arch = "x86_64")]
        Level::Avx512 if is_x86_feature_detected!("avx512bw") => {
            // SAFETY: the CPU has AVX-512 BW (checked); the function reads
            // only within `rest`.
            unsafe { x86::find_avx512(rest, b'\n') }
        }
        #[cfg(target_arch = "x86_64")]
        Level::Avx2 | Level::Avx512 if is_x86_feature_detected!("avx2") => {
            // SAFETY: the CPU has AVX2 (checked); the function reads only
            // within `rest`.
            unsafe { x86::find_avx2(rest, b'\n') }
        }
        _ => rest.iter().position(|&b| b == b'\n'),
    };
    found.map(|i| from + i)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{Masks, MAX_LINE};
    use std::arch::x86_64::*;

    /// # Safety
    /// The CPU must support AVX-512 F and BW.
    /// The masks of `line` (at most `MAX_LINE` bytes); bits past it are clear.
    ///
    /// # Safety
    /// The CPU must support AVX-512 F and BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn classify_avx512(line: &[u8]) -> Masks {
        debug_assert!(line.len() <= MAX_LINE);
        let mut m = Masks::default();
        for half in 0..line.len().div_ceil(64) {
            let n = (line.len() - 64 * half).min(64);
            let keep: u64 = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
            // SAFETY: the masked load touches only the `n` bytes of `line`
            // at `64 * half`; the others read as zero.
            let v = unsafe { _mm512_maskz_loadu_epi8(keep, line.as_ptr().add(64 * half) as *const i8) };
            macro_rules! eq {
                ($c:expr) => {
                    _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8($c as i8))
                };
            }
            let lower = _mm512_or_si512(v, _mm512_set1_epi8(0x20));
            let letter = _mm512_cmple_epu8_mask(_mm512_sub_epi8(lower, _mm512_set1_epi8(b'a' as i8)), _mm512_set1_epi8(25));
            let digit = _mm512_cmple_epu8_mask(_mm512_sub_epi8(v, _mm512_set1_epi8(b'0' as i8)), _mm512_set1_epi8(9));
            let name = (letter | digit | eq!(b'_') | eq!(b'.')) & keep;
            let space = eq!(b' ') & keep;
            let stop = (space | eq!(b',') | eq!(b'}') | eq!(b']') | eq!(b'\t')) & keep;
            let quote = (eq!(b'"') | eq!(b'\\')) & keep;
            let sigil = eq!(b'%') & keep;
            let shift = 64 * half;
            m.name |= (name as u128) << shift;
            m.space |= (space as u128) << shift;
            m.stop |= (stop as u128) << shift;
            m.quote |= (quote as u128) << shift;
            m.sigil |= (sigil as u128) << shift;
        }
        m
    }

    /// # Safety
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn classify_avx2(block: &[u8; MAX_LINE]) -> Masks {
        let mut m = Masks::default();
        for quarter in 0..4 {
            // SAFETY: `block` has 128 bytes; this reads bytes 32*quarter..32*quarter+32.
            let v = unsafe { _mm256_loadu_si256(block.as_ptr().add(32 * quarter) as *const __m256i) };
            macro_rules! bits {
                ($x:expr) => {
                    _mm256_movemask_epi8($x) as u32 as u128
                };
            }
            // Unsigned `x <= hi` as `max(x, hi) == hi`.
            macro_rules! le {
                ($x:expr, $hi:expr) => {
                    _mm256_cmpeq_epi8(_mm256_max_epu8($x, _mm256_set1_epi8($hi as i8)), _mm256_set1_epi8($hi as i8))
                };
            }
            macro_rules! eq {
                ($c:expr) => {
                    _mm256_cmpeq_epi8(v, _mm256_set1_epi8($c as i8))
                };
            }
            let lower = _mm256_or_si256(v, _mm256_set1_epi8(0x20));
            let letter = le!(_mm256_sub_epi8(lower, _mm256_set1_epi8(b'a' as i8)), 25u8);
            let digit = le!(_mm256_sub_epi8(v, _mm256_set1_epi8(b'0' as i8)), 9u8);
            let name = _mm256_or_si256(_mm256_or_si256(letter, digit), _mm256_or_si256(eq!(b'_'), eq!(b'.')));
            let space = eq!(b' ');
            let stop = _mm256_or_si256(
                _mm256_or_si256(space, eq!(b',')),
                _mm256_or_si256(_mm256_or_si256(eq!(b'}'), eq!(b']')), eq!(b'\t')),
            );
            let quote = _mm256_or_si256(eq!(b'"'), eq!(b'\\'));
            let sigil = eq!(b'%');
            let shift = 32 * quarter;
            m.name |= bits!(name) << shift;
            m.space |= bits!(space) << shift;
            m.stop |= bits!(stop) << shift;
            m.quote |= bits!(quote) << shift;
            m.sigil |= bits!(sigil) << shift;
        }
        m
    }

    /// The index of the first `byte` in `rest`.
    ///
    /// # Safety
    /// The CPU must support AVX-512 F and BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn find_avx512(rest: &[u8], byte: u8) -> Option<usize> {
        let needle = _mm512_set1_epi8(byte as i8);
        let mut i = 0;
        while i + 64 <= rest.len() {
            // SAFETY: bytes `i..i + 64` are inside `rest`.
            let v = unsafe { _mm512_loadu_si512(rest.as_ptr().add(i) as *const _) };
            let hit = _mm512_cmpeq_epi8_mask(v, needle);
            if hit != 0 {
                return Some(i + hit.trailing_zeros() as usize);
            }
            i += 64;
        }
        let n = rest.len() - i;
        if n == 0 {
            return None;
        }
        let keep: u64 = (1u64 << n) - 1;
        // SAFETY: the masked load touches only the `n < 64` bytes at `i`,
        // which are inside `rest`.
        let v = unsafe { _mm512_maskz_loadu_epi8(keep, rest.as_ptr().add(i) as *const i8) };
        let hit = _mm512_cmpeq_epi8_mask(v, needle) & keep;
        (hit != 0).then(|| i + hit.trailing_zeros() as usize)
    }

    /// The index of the first `byte` in `rest`.
    ///
    /// # Safety
    /// The CPU must support AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn find_avx2(rest: &[u8], byte: u8) -> Option<usize> {
        let needle = _mm256_set1_epi8(byte as i8);
        let mut i = 0;
        while i + 32 <= rest.len() {
            // SAFETY: bytes `i..i + 32` are inside `rest`.
            let v = unsafe { _mm256_loadu_si256(rest.as_ptr().add(i) as *const __m256i) };
            let hit = _mm256_movemask_epi8(_mm256_cmpeq_epi8(v, needle)) as u32;
            if hit != 0 {
                return Some(i + hit.trailing_zeros() as usize);
            }
            i += 32;
        }
        rest[i..].iter().position(|&b| b == byte).map(|j| i + j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator for byte strings rich in the
    /// classes' boundary values.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    const EDGES: &[u8] = b"\0\t\n\r !\"#%,-./09:;@AZ[\\]^_`az{|}~\x7f\x80\xa0\xc3\xe9\xff";

    #[test]
    fn every_level_classifies_as_the_scalar_one() {
        let mut rng = Rng(1);
        let levels = Level::supported();
        for round in 0..20_000 {
            let len = (rng.next() % (MAX_LINE as u64 + 1)) as usize;
            let line: Vec<u8> = (0..len)
                .map(|_| if rng.next() % 3 == 0 { (rng.next() % 256) as u8 } else { EDGES[(rng.next() % EDGES.len() as u64) as usize] })
                .collect();
            let reference = classify(Level::Scalar, &line).expect("short");
            for &level in &levels {
                assert_eq!(classify(level, &line), Some(reference), "round {round} {level:?} {line:?}");
            }
        }
        for b in 0..=255u8 {
            let line = [b; MAX_LINE];
            for &level in &levels {
                assert_eq!(classify(level, &line), classify(Level::Scalar, &line), "byte {b}");
            }
        }
        assert_eq!(classify(Level::Scalar, &[b'a'; MAX_LINE + 1]), None);
    }

    #[test]
    fn every_level_finds_newlines_as_the_scalar_one() {
        let mut rng = Rng(2);
        for _ in 0..5_000 {
            let len = (rng.next() % 300) as usize;
            let text: Vec<u8> = (0..len).map(|_| if rng.next() % 40 == 0 { b'\n' } else { EDGES[(rng.next() % EDGES.len() as u64) as usize] }).collect();
            for from in [0, 1, len / 2, len.saturating_sub(1), len, len + 1] {
                let reference = text.get(from..).and_then(|r| r.iter().position(|&b| b == b'\n')).map(|i| from + i);
                for level in Level::supported() {
                    assert_eq!(find_newline(level, &text, from), reference, "{level:?} from {from}");
                }
            }
        }
    }

    #[test]
    fn skip_and_find_stop_at_the_line_end() {
        let m = classify(Level::best(), b"%ab_c.9 : i32").expect("short");
        assert_eq!(Masks::skip(m.name, 1, 13), 7);
        assert_eq!(Masks::find(m.space, 0, 13), 7);
        assert_eq!(Masks::skip(m.name, 10, 13), 13);
        assert_eq!(Masks::find(m.sigil, 1, 13), 13);
        assert_eq!(Masks::skip(m.name, 20, 13), 13);
    }
}
