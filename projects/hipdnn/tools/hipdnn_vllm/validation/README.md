# hipdnn_vllm validation harness

End-to-end model matrix, coverage tables and relative perf for the plugin, run
through Slurm. Op-level GPU tests stay in `tests/gpu`.

| File | Role |
|---|---|
| `site.env.example` | Template for the site settings: shared paths, partition, node excludes, and per arch the wheel family, GPU request and extra sbatch flags. Copy it to `site.env` beside it (git ignores that name) or elsewhere and set `VALIDATION_CONFIG`; the scripts stop when the file is missing. |
| `models.json` | The model matrix: path, kind (`generate`, `vlm`, `embed`), memory needs, extra `LLM` arguments. |
| `download_models.py` | Fetches the models.json weights at each entry's `revision` (default `main`) and records the resolved commit sha in `<model>/hf_revision.json` (network only; runs on a submit host when compute nodes have no internet). |
| `submit.sh` | Submit-host entry point: one job per arch. Relative `VALIDATION_CONFIG` and `HV_WHEEL` paths are made absolute before the job starts. |
| `run_matrix.sbatch` | The job: installs the arch's wheel in a throwaway overlay venv on top of the pinned shared venv, runs `doctor`, every model, then `aggregate.py`. Exits non-zero when the doctor check, any model or the aggregation fails, or no model ran. |
| `e2e_matrix.py` | One model: plugin-off base and control, vLLM-default perf reference, and one run per candidate op set, each in its own process. |
| `aggregate.py` | Turns any number of result directories into `coverage.{csv,md}`, `accuracy.md` (with a Failures section), `perf.md` and `rows.json`. Exits non-zero when any failure is listed. |

## Usage

```bash
# submit host (a Slurm cluster login node)
cp site.env.example site.env                                 # once; fill in the site values
python download_models.py --root "$HV_ROOT"                  # once
./submit.sh gfx942,gfx1100                                   # whole matrix, default candidates
./submit.sh gfx942 qwen2.5-0.5b rms_norm,linear,rms_norm+linear bisect
# bundled-backend wheels, or one specific wheel file
HV_WHEEL_ROOT=$HV_ROOT/wheels/bundled ./submit.sh gfx942,gfx1100 matrix vllm_c,default,default+linear bundled
HV_WHEEL=/path/to/hipdnn_vllm.whl ./submit.sh gfx950 matrix
# any node
python aggregate.py "$HV_RESULTS"/<tag>/* --out tables/
python e2e_matrix.py pairwise "$HV_RESULTS"/<tag>/<arch>-<job>/<model> --ref base --ref cand-vllm_c
```

A candidate is plugin ops joined with `+`: `default` is the plugin's default op
set, `attention` selects `--attention-backend HIPDNN`, and `rms_norm` and
`fused_add_rms_norm` put `hipdnn` first in the IR priority. `vllm_c` is a
plugin-off run with vLLM's own `vllm_c` norm kernels instead of the default
`native` ones. It is a legitimate kernel change of the same ops, so it shows
how much a non-hipDNN numerics change moves the metric.

## Accuracy criterion

This is the accuracy metric for every plugin op.
1. Base is plugin off, using TRITON_ATTN, the implementation HIPDNN falls back to. It greedy-generates a fixed prompt set.
2. Every other run re-scores base's prompt and completion with `prompt_logprobs` (teacher forcing). All runs therefore see identical inputs, and a single early token flip does not cascade into the rest of the sequence.
3. At each completion position, the run's top-1 token is compared with base's own re-scored top-1.

A candidate passes when all of these hold:
- It routed through hipDNN. Every op family it enables that the model exercised
  (decoder attention always, `encoder_attention` for `vlm` models, and any family
  with counter rows) has at least one call with route `hipdnn`. A model lists the
  families the plugin is known to leave on fallback in its models.json
  `fallback_ops`. A candidate with no hipDNN route at all always fails.
- No log-probability is NaN.
- Its agreement is at least `min(0.99, max(control, 0.97))`.

The control is a second plugin-off run, so it measures run-to-run
self-agreement. It can relax the 0.99 target down to the floor of 0.97, never
further. A control below the floor, with NaN, or that errors fails the model,
because the noise is then too large to judge any candidate. The control's
verdict is `-`. Every plugin-off run (base, control, default, `vllm_c`) must
record no plugin counters.

The tables also report:
- the mean |Δ log-probability| of the forced token;
- how many disagreements are near ties, where base's top-1/top-2 margin is under 0.1 nats.

Embedding models use the minimum cosine similarity to base's embeddings instead, with a threshold of `min(0.999, max(control, 0.995))`.

Positional comparison of free-running greedy outputs is not used. After the
first flipped token, the two runs decode different contexts, so that number
measures how early the first near tie happens, not how accurate the op is.

## Model substitutions

- **Qwen3-8B instead of Llama-3.1-8B.** Either serves the matrix, and Llama is gated on Hugging Face.
- **unsloth/gemma-3-4b-it**: an ungated copy of the gated google/gemma-3-4b-it, run text-only.
- **gpt-oss-20b** is skipped on devices under 40 GB. Its decoder attention carries the attention sinks as the graph's `sink_token`, so it must route to hipDNN like any other decoder model.
- **qwen2.5-0.5b** is not in the `matrix` set. It is the small model for smoke and bisect runs.
- **gemma-3-4b-strict-bigkv** is not in the `matrix` set. It runs gemma-3-4b in strict mode with a KV pool past 4 GiB, on devices with at least 30 GB; run it with the `attention` candidate (`./submit.sh gfx1100 gemma-3-4b-strict-bigkv attention bigkv`).

## Outputs

`$HV_RESULTS/<tag>/<arch>-<job>/<model>/` contains:
- `summary.json`;
- per run: `run.json` and `run.log`;
- the plugin's `report-*.json` files under `<run>/reports/`.

`tables/` sits beside them. Perf is reported only as ratios against base and against vLLM's default backend. Stall times are in seconds: plan builds, norm exact-row pre-build, and hipDNN handle creation.
