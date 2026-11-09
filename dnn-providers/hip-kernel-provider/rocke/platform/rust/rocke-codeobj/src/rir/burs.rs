// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Prototype BURS instruction selector for 32-bit integer arithmetic, the
//! arithmetic that forms buffer and global offsets. Off unless
//! `ROCKE_DIRECT_DEBUG=burs`. The grammar and its costs come from the
//! instruction description (`[burs]` and `issue=` in `isa/gfx11.isa`);
//! `gen_burs.rs`, generated from them, labels nodes and binds patterns.
//!
//! Before selection, `plan` cuts each region's integer arithmetic into trees
//! (an operation whose only use is another such operation of the same region,
//! not uniform under a divergent one, is that operation's child), labels each
//! tree bottom-up with the cheapest rule per nonterminal, and covers it from
//! its root. An operation inside a chosen multi-operation pattern is folded:
//! nothing is emitted for it. Every other operation is emitted at its own
//! place by its rule, through isel's own `int_bin` for a single-operation
//! rule, so code the patterns do not change stays as isel selects it.

use super::gen_burs::{self as g, Action, Nt, Term, NTS};
use super::{Isel, Val};
use crate::error::CompileError;
use crate::rir::ast::{Op, Region, Scalar, Type, ValueId};
use crate::rir::fx::{FxHashMap, FxHashSet};
use crate::rir::mir::{Opc, Opnd};

/// No rule: a cost three of which still add without overflow.
pub const INF: u32 = u32::MAX / 4;

/// The cheapest rule found for one nonterminal at a node.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub cost: u32,
    pub rule: u16,
}

impl Cell {
    pub const NONE: Cell = Cell { cost: INF, rule: u16::MAX };

    #[inline]
    pub fn offer(&mut self, cost: u32, rule: u16) {
        if cost < self.cost {
            *self = Cell { cost, rule };
        }
    }
}

struct Node {
    term: Term,
    value: ValueId,
    kids: [u32; 2],
    uniform: bool,
    /// Known to be below 2^24.
    small: bool,
    konst: Option<u32>,
}

/// A tree of integer operations, children before parents, and its labels.
pub struct Tree {
    nodes: Vec<Node>,
    cells: Vec<[Cell; NTS]>,
}

impl Tree {
    #[inline]
    pub fn term(&self, n: usize) -> Term {
        self.nodes[n].term
    }

    #[inline]
    pub fn kid(&self, n: usize, i: usize) -> usize {
        self.nodes[n].kids[i] as usize
    }

    #[inline]
    pub fn cost(&self, n: usize, nt: Nt) -> u32 {
        self.cells[n][nt as usize].cost
    }

    /// Predicate `u24`: both values below 2^24, so 24-bit multiplies are exact.
    pub fn u24(&self, a: usize, b: usize) -> bool {
        self.nodes[a].small && self.nodes[b].small
    }

    /// Predicate `shift1to4`: a constant shift amount of 1 to 4.
    pub fn shift1to4(&self, n: usize) -> bool {
        matches!(self.nodes[n].konst, Some(1..=4))
    }

    /// Predicate `fits3`: three VOP3 sources within the scalar limit isel's
    /// fusion keeps (two SGPRs or literals, at most one literal).
    pub fn fits3(&self, a: usize, b: usize, c: usize) -> bool {
        let (mut scalars, mut literals) = (Vec::with_capacity(3), Vec::with_capacity(3));
        for n in [a, b, c] {
            let node = &self.nodes[n];
            match node.konst {
                Some(k) if !(super::is_inline_int(k) || super::is_inline_f32(k)) => {
                    if !literals.contains(&k) {
                        literals.push(k);
                    }
                }
                Some(_) => {}
                None if node.uniform => {
                    if !scalars.contains(&node.value) {
                        scalars.push(node.value);
                    }
                }
                None => {}
            }
        }
        literals.len() <= 1 && scalars.len() + literals.len() <= 2
    }
}

/// Which operations the selector emits and with which rule, and which it folds.
pub struct Plan {
    trees: Vec<Tree>,
    emit: FxHashMap<ValueId, (u32, u32, u16)>,
    folded: FxHashSet<ValueId>,
}

/// The selector's plan for a kernel (before isel emits anything).
pub fn plan(isel: &Isel) -> Plan {
    let mut plan = Plan { trees: Vec::new(), emit: FxHashMap::default(), folded: FxHashSet::default() };
    plan_region(isel, &isel.k.body, &mut plan);
    plan
}

/// The terminal of an operation the selector takes: a 32-bit integer binary
/// operation that no other isel pattern claims.
fn terminal(isel: &Isel, op: &Op) -> Option<Term> {
    let term = g::term_of(op.kind)?;
    let &[v] = &op.results[..] else { return None };
    let ok = op.operands.len() == 2
        && *isel.k.ty(v) == Type::Scalar(Scalar::I32)
        && !isel.chains.skip.contains(&v)
        && !isel.chains.packs.contains_key(&v)
        && !isel.packs.contains_key(&v);
    ok.then_some(term)
}

fn plan_region(isel: &Isel, region: &Region, plan: &mut Plan) {
    for op in &region.ops {
        for inner in &op.regions {
            plan_region(isel, inner, plan);
        }
    }
    let mut at: FxHashMap<ValueId, usize> = FxHashMap::default();
    for (i, op) in region.ops.iter().enumerate() {
        if terminal(isel, op).is_some() {
            at.insert(op.results[0], i);
        }
    }
    let mut child: FxHashSet<ValueId> = FxHashSet::default();
    for op in region.ops.iter().filter(|op| op.results.first().is_some_and(|v| at.contains_key(v))) {
        let user = op.results[0];
        for &x in &op.operands {
            if at.contains_key(&x) && isel.uses[x.0 as usize] == 1 && (isel.uni(x) || !isel.uni(user)) {
                child.insert(x);
            }
        }
    }
    for op in &region.ops {
        let Some(&root) = op.results.first() else { continue };
        if !at.contains_key(&root) || child.contains(&root) {
            continue;
        }
        let mut tree = Tree { nodes: Vec::new(), cells: Vec::new() };
        build(isel, region, &at, &child, root, true, &mut tree.nodes);
        tree.cells = vec![[Cell::NONE; NTS]; tree.nodes.len()];
        for n in 0..tree.nodes.len() {
            tree.cells[n] = g::label(&tree, n, tree.nodes[n].uniform);
        }
        let top = tree.nodes.len() - 1;
        let goal = if tree.nodes[top].uniform { Nt::S } else { Nt::V };
        if tree.cost(top, goal) >= INF {
            continue;
        }
        plan.trees.push(tree);
        cover(plan, plan.trees.len() - 1, top, goal);
    }
}

/// Appends the subtree of value `v` (its kids first) and returns its index.
fn build(
    isel: &Isel,
    region: &Region,
    at: &FxHashMap<ValueId, usize>,
    child: &FxHashSet<ValueId>,
    v: ValueId,
    root: bool,
    nodes: &mut Vec<Node>,
) -> u32 {
    let (uniform, small) = (isel.uni(v), isel.range_lt(v, 1 << 24));
    let node = if root || child.contains(&v) {
        let op = &region.ops[at[&v]];
        let term = g::term_of(op.kind).expect("a terminal operation");
        let a = build(isel, region, at, child, op.operands[0], false, nodes);
        let b = build(isel, region, at, child, op.operands[1], false, nodes);
        Node { term, value: v, kids: [a, b], uniform, small, konst: None }
    } else if let Some(k) = isel.facts.int(v) {
        Node { term: g::KONST, value: v, kids: [0, 0], uniform, small, konst: Some(k as u32) }
    } else {
        let term = if uniform { g::SREG } else { g::VREG };
        Node { term, value: v, kids: [0, 0], uniform, small, konst: None }
    };
    nodes.push(node);
    (nodes.len() - 1) as u32
}

/// Records the rules covering node `n` as nonterminal `nt`.
fn cover(plan: &mut Plan, t: usize, n: usize, nt: Nt) {
    let tree = &plan.trees[t];
    let rule = tree.cells[n][nt as usize].rule as usize;
    match g::RULES[rule].action {
        Action::Chain(sub) => cover(plan, t, n, sub),
        Action::Leaf => {}
        _ => {
            let value = tree.nodes[n].value;
            let mut inner = [0usize; 2];
            let ni = g::interior(tree, n, rule, &mut inner);
            let folded: Vec<ValueId> = inner[..ni].iter().map(|&m| tree.nodes[m].value).collect();
            let mut vars = [(0usize, Nt::S); 3];
            let nv = g::vars(tree, n, rule, &mut vars);
            plan.emit.insert(value, (t as u32, n as u32, rule as u16));
            plan.folded.extend(folded);
            for &(m, ntm) in &vars[..nv] {
                cover(plan, t, m, ntm);
            }
        }
    }
}

/// Selects `op` if the plan covers it: emits its rule, or nothing for an
/// operation folded into its user. Returns whether it did.
pub fn select(isel: &mut Isel, op: &Op) -> Result<bool, CompileError> {
    let Some(&v) = op.results.first() else { return Ok(false) };
    let plan = isel.burs.take().expect("the BURS plan");
    let done = if plan.folded.contains(&v) {
        Ok(true)
    } else if let Some(&(t, n, rule)) = plan.emit.get(&v) {
        emit(isel, &plan.trees[t as usize], n as usize, rule as usize).map(|out| {
            isel.set(v, Val::W(out));
            true
        })
    } else {
        Ok(false)
    };
    isel.burs = Some(plan);
    done
}

fn emit(isel: &mut Isel, tree: &Tree, n: usize, rule: usize) -> Result<Opnd, CompileError> {
    let mut vars = [(0usize, Nt::S); 3];
    let nv = g::vars(tree, n, rule, &mut vars);
    let mut ops = [Opnd::Imm(0); 3];
    for (slot, &(m, _)) in ops.iter_mut().zip(&vars[..nv]) {
        *slot = isel.w(tree.nodes[m].value)?;
    }
    Ok(match g::RULES[rule].action {
        Action::Hand(kind) => {
            let small = tree.u24(tree.kid(n, 0), tree.kid(n, 1));
            isel.int_bin(kind, ops[0], ops[1], tree.nodes[n].uniform, small)?
        }
        Action::V3 { op, vars: [a, b, c], commute } => {
            let (mut x, mut y) = (ops[a as usize], ops[b as usize]);
            if commute && !isel.is_v(y) && isel.is_v(x) {
                std::mem::swap(&mut x, &mut y);
            }
            isel.v3(op, x, y, ops[c as usize], 0)
        }
        Action::LshlAdd { vars: [a, k, b] } => {
            let opc = match tree.nodes[vars[k as usize].0].konst {
                Some(1) => Opc::SLshl1AddU32,
                Some(2) => Opc::SLshl2AddU32,
                Some(3) => Opc::SLshl3AddU32,
                _ => Opc::SLshl4AddU32,
            };
            isel.s_bin(opc, ops[a as usize], ops[b as usize])
        }
        Action::Leaf | Action::Chain(_) => unreachable!("leaves and chains emit nothing"),
    })
}
