# Worked queries

## “GEMM on gfx950 is memory stalled”

```bash
python3 scripts/query.py --operator gemm --architecture gfx950 --symptom memory-bound
python3 scripts/get_page.py family-gemm
python3 scripts/get_page.py technique-async-copy
```

Family table says `compv4` is legal on gfx950; RDNA column would have said no.

## “Reduce is wrong on gfx1151”

```bash
python3 scripts/get_page.py family-small-ops
python3 scripts/get_page.py technique-wave32
```

Wave64 XOR tree on wave32 hardware.

## “What do hipBLASLt and TensileLite do for grouped GEMM?”

```bash
python3 scripts/query.py --tag hipblaslt --type project
python3 scripts/get_page.py project-tensilelite --follow-sources
python3 scripts/get_page.py family-moe
```

## “gfx1250 attention — can I reuse gfx950 ds_read_tr?”

```bash
python3 scripts/get_page.py hw-gfx1250
python3 scripts/get_page.py technique-ds-read-tr
python3 scripts/get_page.py technique-gfx1250-ds-load-tr
python3 scripts/get_page.py migration-ds-read-tr-to-ds-load-tr
```

No. Different opcodes; `has_ds_read_tr` is false on gfx1250 on purpose.

## “Port a gfx950 GEMM to gfx1250 (Blackwell-shaped programming model)”

```bash
python3 scripts/get_page.py hw-gfx1250
python3 scripts/query.py --type migration --architecture gfx1250
python3 scripts/get_page.py migration-gfx950-to-gfx1250
python3 scripts/get_page.py hw-wmma-gfx1250
python3 scripts/get_page.py hw-async-global-lds
```

Redesign, not an opcode swap. No TMEM, no AGPR, no CLC, no TMA descriptor.

## “Convolution — is there a hipconv repo?”

```bash
python3 scripts/get_page.py project-hipdnn
python3 scripts/get_page.py family-convolution
```

No `projects/hipconv` on develop. hipDNN descriptors + MIOpen + CK + rocke.

## “GEMM levers are exhausted, ISA histogram did not move”

```bash
python3 scripts/query.py --symptom catalog-exhausted
python3 scripts/get_page.py process-escape-hatch
python3 scripts/get_page.py technique-algorithm-break
python3 scripts/query.py --operator gemm --type project
```

Stall test first (Step 0 + ≥3 one-lever ISA diffs, same limiter). Then steal a
mapping hipBLASLt / TensileLite / CK expose that this spec cannot name — not
another `tile_k`.
