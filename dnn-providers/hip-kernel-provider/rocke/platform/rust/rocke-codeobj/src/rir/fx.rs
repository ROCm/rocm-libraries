// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Fx-style multiply-add hasher for the compiler's integer-keyed tables, and
//! a scoped hash table keyed by `u32` words.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0xf135_7aea_2e62_a9c5;

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.add(u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut last = [0u8; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(value as u64);
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.add(value as u64);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(value as u64);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuild>;
pub type FxHashSet<K> = HashSet<K, FxBuild>;

pub fn hash_words(words: &[u32]) -> u64 {
    let mut h = FxHasher::default();
    for &w in words {
        h.add(w as u64);
    }
    h.hash
}

/// The hash `ScopedTable` files a key under: two multiply-xor lanes over
/// pairs of words, so that a key's multiplications overlap (`hash_words`
/// chains one per word). Only where entries go depends on it.
fn key_hash(words: &[u32]) -> u64 {
    const K0: u64 = 0x9e37_79b9_7f4a_7c15;
    const K1: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let pair = |w: &[u32]| u64::from(w[0]) | u64::from(w[1]) << 32;
    let (mut a, mut b) = (words.len() as u64, SEED);
    let mut quads = words.chunks_exact(4);
    for q in &mut quads {
        a = (a ^ pair(&q[..2])).wrapping_mul(K0);
        b = (b ^ pair(&q[2..])).wrapping_mul(K1);
    }
    let mut rest = [0u32; 4];
    let tail = quads.remainder();
    if !tail.is_empty() {
        rest[..tail.len()].copy_from_slice(tail);
        a = (a ^ pair(&rest[..2])).wrapping_mul(K0);
        b = (b ^ pair(&rest[2..])).wrapping_mul(K1);
    }
    let h = (a ^ b.rotate_left(31)).wrapping_mul(K0);
    h ^ h >> 29
}

/// Map from `u32`-word keys to values whose entries are removed in reverse
/// insertion order (`truncate` to an earlier `mark`), as scoped CSE needs.
pub struct ScopedTable<V> {
    bits: u32,
    heads: Vec<u32>,
    entries: Vec<Entry<V>>,
    words: Vec<u32>,
}

struct Entry<V> {
    hash: u64,
    start: u32,
    len: u32,
    next: u32,
    value: V,
}

impl<V: Copy> Default for ScopedTable<V> {
    fn default() -> Self {
        Self { bits: 0, heads: Vec::new(), entries: Vec::new(), words: Vec::new() }
    }
}

impl<V: Copy> ScopedTable<V> {
    fn bucket(&self, hash: u64) -> usize {
        (hash >> (64 - self.bits)) as usize
    }

    pub fn get(&self, key: &[u32]) -> Option<V> {
        if self.heads.is_empty() {
            return None;
        }
        let hash = key_hash(key);
        let mut i = self.heads[self.bucket(hash)];
        while i != 0 {
            let e = &self.entries[i as usize - 1];
            if e.hash == hash && &self.words[e.start as usize..(e.start + e.len) as usize] == key {
                return Some(e.value);
            }
            i = e.next;
        }
        None
    }

    /// `get(key)`, adding `key` with `value` when it is absent: one hash
    /// and one probe for both.
    pub fn get_or_insert(&mut self, key: &[u32], value: V) -> Option<V> {
        let hash = key_hash(key);
        if !self.heads.is_empty() {
            let mut i = self.heads[self.bucket(hash)];
            while i != 0 {
                let e = &self.entries[i as usize - 1];
                if e.hash == hash && &self.words[e.start as usize..(e.start + e.len) as usize] == key {
                    return Some(e.value);
                }
                i = e.next;
            }
        }
        self.insert_hashed(key, hash, value);
        None
    }

    /// Adds `key`; while present, it shadows an equal key added earlier.
    pub fn insert(&mut self, key: &[u32], value: V) {
        self.insert_hashed(key, key_hash(key), value);
    }

    fn insert_hashed(&mut self, key: &[u32], hash: u64, value: V) {
        if self.entries.len() >= self.heads.len() {
            self.bits = (self.bits + 1).max(8);
            self.heads = vec![0; 1 << self.bits];
            for i in 0..self.entries.len() {
                let b = self.bucket(self.entries[i].hash);
                self.entries[i].next = self.heads[b];
                self.heads[b] = i as u32 + 1;
            }
        }
        let b = self.bucket(hash);
        let start = self.words.len() as u32;
        self.words.extend_from_slice(key);
        self.entries.push(Entry { hash, start, len: key.len() as u32, next: self.heads[b], value });
        self.heads[b] = self.entries.len() as u32;
    }

    pub fn mark(&self) -> usize {
        self.entries.len()
    }

    pub fn truncate(&mut self, mark: usize) {
        while self.entries.len() > mark {
            let e = self.entries.pop().expect("non-empty");
            let b = self.bucket(e.hash);
            self.heads[b] = e.next;
            self.words.truncate(e.start as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_table_restores_outer_scope() {
        let mut t = ScopedTable::default();
        for k in 0..1000u32 {
            t.insert(&[k, k * 7], k);
        }
        let mark = t.mark();
        for k in 1000..3000u32 {
            t.insert(&[k, k * 7], k);
        }
        t.insert(&[5, 1, 2], 99);
        assert_eq!(t.get(&[2500, 2500 * 7]), Some(2500));
        assert_eq!(t.get(&[5, 1, 2]), Some(99));
        t.truncate(mark);
        assert_eq!(t.get(&[2500, 2500 * 7]), None);
        assert_eq!(t.get(&[5, 1, 2]), None);
        assert_eq!(t.get(&[999, 999 * 7]), Some(999));
        assert_eq!(t.get(&[999, 999]), None);
        t.insert(&[2500, 2500 * 7], 7);
        assert_eq!(t.get(&[2500, 2500 * 7]), Some(7));
    }

    #[test]
    fn get_or_insert_is_get_then_insert_when_absent() {
        let (mut a, mut b) = (ScopedTable::default(), ScopedTable::default());
        let mut seed = 0x9e37_79b9u64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        let mut marks = Vec::new();
        for i in 0..20000u32 {
            let key: Vec<u32> = (0..1 + next(4)).map(|_| next(40)).collect();
            match next(50) {
                0 => marks.push((a.mark(), b.mark())),
                1 => {
                    if let Some((ma, mb)) = marks.pop() {
                        a.truncate(ma);
                        b.truncate(mb);
                    }
                }
                _ => {
                    let want = a.get(&key);
                    if want.is_none() {
                        a.insert(&key, i);
                    }
                    assert_eq!(b.get_or_insert(&key, i), want);
                }
            }
            assert_eq!(a.get(&key), b.get(&key));
        }
    }

    /// The table against a stack of entries searched newest first, with
    /// keys of every length up to eleven words sharing prefixes and hashes'
    /// buckets: where `key_hash` files entries changes no answer.
    #[test]
    fn scoped_table_answers_as_a_stack_of_entries() {
        let mut table = ScopedTable::default();
        let mut stack: Vec<(Vec<u32>, u32)> = Vec::new();
        let mut marks = Vec::new();
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32 % n
        };
        for i in 0..60000u32 {
            let key: Vec<u32> = (0..next(12)).map(|_| next(6)).collect();
            let want = stack.iter().rev().find(|(k, _)| *k == key).map(|&(_, v)| v);
            match next(40) {
                0 => marks.push((table.mark(), stack.len())),
                1 => {
                    if let Some((m, n)) = marks.pop() {
                        table.truncate(m);
                        stack.truncate(n);
                    }
                }
                2..=9 => {
                    table.insert(&key, i);
                    stack.push((key.clone(), i));
                }
                10..=19 => assert_eq!(table.get(&key), want),
                _ => {
                    assert_eq!(table.get_or_insert(&key, i), want);
                    if want.is_none() {
                        stack.push((key.clone(), i));
                    }
                }
            }
            assert_eq!(table.mark(), stack.len());
        }
    }
}
