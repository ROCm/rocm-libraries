# PROPOSAL: <short capability name>

- **Status:** proposed
- **Raised while:** <which kernel/algorithm design surfaced this>
- **Extends SOT:** <tiling_api_surface.md / tiling_interleaving_design.md / lds_banks.md / mma_is_machinery.md — which doc + section this would update>

## Friction (what's awkward today)
<The boilerplate / manual work-around / thing the tiling model should express but doesn't. Include the awkward
code as it must be written now.>

```python
# today (awkward)
```

## Proposed addition
<The new verb / knob / signature / type, and where it lives (`emit.py`, `mma/`, `fragments.py`, ...).>

```python
# proposed API
```

## Example — before → after
```python
# before:  <current>
# after:   <with the proposal>
```

## Soundness / perf caveats
<Any correctness gate (e.g. must preserve vector contiguity — lds_banks.md §4), bit-exactness, perf
tradeoff, or arch dependence. Note what MUST be validated before/after implementing.>

## Notes
<Alternatives considered; relation to existing verbs; open questions.>
