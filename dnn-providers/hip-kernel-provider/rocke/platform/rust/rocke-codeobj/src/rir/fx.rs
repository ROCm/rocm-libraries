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
        let hash = hash_words(key);
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

    /// Adds `key`; while present, it shadows an equal key added earlier.
    pub fn insert(&mut self, key: &[u32], value: V) {
        if self.entries.len() >= self.heads.len() {
            self.bits = (self.bits + 1).max(8);
            self.heads = vec![0; 1 << self.bits];
            for i in 0..self.entries.len() {
                let b = self.bucket(self.entries[i].hash);
                self.entries[i].next = self.heads[b];
                self.heads[b] = i as u32 + 1;
            }
        }
        let hash = hash_words(key);
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
}
