# Rocke optimization wiki

Agent entry: [`SKILL.md`](SKILL.md). Query from this directory:

```bash
python3 scripts/query.py --operator gemm --family cdna
python3 scripts/query.py --symptom lds-stall --architecture gfx950
python3 scripts/get_page.py family-overview
python3 scripts/query.py --symptom catalog-exhausted
python3 scripts/get_page.py hw-gfx1250
python3 scripts/query.py --type migration --architecture gfx1250
python3 scripts/validate.py
python3 scripts/generate-indices.py
```

## Layout

| Path | Role |
|---|---|
| `wiki/families/` | **Routing tables** — operator family × architecture family |
| `wiki/techniques/common/` | Tech that applies across gfx |
| `wiki/techniques/{cdna,rdna,gfx1250}/` | Arch-specific tech |
| `wiki/patterns/` | Symptom → techniques |
| `wiki/hardware/` | gfx / MFMA / WMMA / LDS / **gfx1250 feature pages** |
| `wiki/migration/` | gfx950 / gfx1201 → gfx1250 ports (KernelWiki analog) |
| `wiki/kernels/` | Rocke instance case pages |
| `wiki/process/` | Step 0, probes, routing, **escape hatch** |
| `sources/projects/` | hipBLASLt, Tensile, TensileLite, stinkytofu, rocRoller, Origami, MIOpen, hipDNN, CK, rocke |
| `queries/` | Auto-generated indices |
| `optimization_runbook.md` | Long-form appendix — do not read linearly |
| `utilities/` | Probes, ATT capture, WaveScope |

There is no `projects/hipconv` on `develop`; convolution sources are MIOpen +
hipDNN + CK Tile + rocke (`sources/projects/hipdnn.md`).
