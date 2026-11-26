# Lane: kernel edge cases (`KEDGE`)

**Trigger:** always, after `KMAP`. **Agent:** `scout`. **Input:** `R/lanes/KMAP.md`. **Output:** `R/lanes/KEDGE.md`, ideas.

For each kernel in the map, find the shapes its code handles specially and could get wrong.

## Look for

- Tile and tail handling: sizes one below, at and one above each tile multiple; sizes below one tile; sequence length 1 and 2.
- Masks with dead rows (no live key), single-key rows, windows wider than the sequence, bottom-right alignment with `sq > skv`.
- Head dims at ladder edges and padded head dims; asymmetric Q/K and V/O head dims.
- GQA group sizes outside powers of two; one KV head; many heads with small batch.
- Precision paths: accumulator type, low-precision intermediates, split-K style reductions and their order, atomic accumulation
  (nondeterministic objects).
- 32-bit index arithmetic: per-head or per-batch byte offsets past 2^31; grid dimensions near their limits.
- LDS and register limits per arch; vector widths against alignment.

Each idea names the kernel from the map it targets, in `Why:`.
