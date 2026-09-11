# `geko.config_generator`

Generates Tensile tuning configs (and related artifacts) from a YAML GEMM specification and a local [hipBLASLt](https://github.com/ROCm/hipBLASLt) tree.

## How to run

### CLI

From the **GEKO package root** ([`scripts/config_generator.py`](../../scripts/config_generator.py)):

```bash
python3 scripts/config_generator.py [--hipblaslt PATH] [--config PATH] [--outputPath DIR] \
  [--verbose 0|1] [--arch ARCH] [--gemm-log-path PATH] [--no-shell-scripts]
```

- **`--hipblaslt` PATH:** hipBLASLt checkout root (must contain `tensilelite/`). Optional — auto-detected from the script location (and `$GEKO_HIPBLASLT_PATH`) when omitted.
- **`--config` / `-c`:** tuning YAML. Pass it, or omit it and pass **`--arch`** with **`--gemm-log-path`** (hipBLASLt-style workload / GEMM list YAML; parse-only, no benchmark run inside `load_prepared_config_from_yaml`).
- **`--arch` / `--gemm-log-path`:** workload-only mode or overrides when a tuning YAML is present (CLI sets `SIZE_OPTION` to 2 when `--gemm-log-path` is passed).
- **`--outputPath` / `-o`:** output directory (default `./`).
- **`--verbose` / `-v`:** `0` = WARNING, `1` = INFO (default `1`).
- **`--no-shell-scripts`:** emit YAML and config log only (skip per-entity `.sh` and `run_*_all.sh`).

The driver prepends `hipblaslt_path/tensilelite` to `sys.path`. When shell scripts are enabled (default), it may build the Tensile client via `geko.utils.build_tensilelite_client` (optional YAML key `BUILD_DIR` overrides the build directory). YAML-only runs (`write_shell_scripts=False`) skip the client build.

### Python API

`run` takes an already-prepared config **dict** (not a path). Build it with
`load_prepared_config_from_yaml`, then call `run`:

```python
from geko.config_generator.config_generator import run
from geko.config_generator.load_input_config import load_prepared_config_from_yaml

config = load_prepared_config_from_yaml(config_path)  # or arch=..., gemm_log_path=...
run(config, hipblaslt_path, output_path)              # write_shell_scripts=True by default
```

### Tests

See [`tests/config_generator/README.md`](../../tests/config_generator/README.md).

### Extending hardware / fork parameters

See [`fork_params/README.md`](fork_params/README.md).

---

## Input YAML

[`config.yaml`](config.yaml) in this package is a commented template (valid type codes, example `ARCH` values, and optional keys).

### Required fields

| Key | Notes |
|-----|--------|
| `TRANSA`, `TRANSB` | `'T'` or `'N'` |
| `DataType`, `DestDataType`, `ComputeDataType` | See comments in `config.yaml` |
| `ARCH` | Must be a key of `HARDWARE_MAP` in [`constants.py`](constants.py). Tensile `LibraryLogic` defaults (`ScheduleName`, `ArchitectureName`, `DeviceNames`) are stored per `ARCH` in `_ARCH_SPECS` and exposed as `HARDWARE_MAP[ARCH]['LibraryLogic']`. |

### Sizes and `SIZE_OPTION`

| `SIZE_OPTION` | Meaning |
|---------------|---------|
| `0` (default) | `Sizes` required: non-empty list of `[M, N, B, K]` |
| `1` | Grid-generated sizes; optional `GRID_DENSITY` (default `4`) |
| `2` | `GEMM_LOG_PATH` required: hipBLASLt-style workload YAML; `GemmProblems` built via [`parse`](../../geko/bench/log.py) in [`load_input_config`](load_input_config.py) (no `hipblaslt_path` on that API). For benchmark + `keep_thr` filtering, use [`scripts/configure.py`](../../scripts/configure.py) with `summarize`. |

### Log / workload YAML (`SIZE_OPTION` 2)

Set `GEMM_LOG_PATH` to a readable YAML (same shape as hipBLASLt log / workload entries) and `SIZE_OPTION: 2`. Only `ARCH` and `GEMM_LOG_PATH` are required in that mode (no `TRANSA` / dtype keys on the tuning dict). Optional tuning YAML can supply fork/CMS defaults; CLI `--gemm-log-path` forces `SIZE_OPTION` 2 and overrides the path.

### Hardware defaults (`ARCH`)

If the YAML omits them, these are filled from `HARDWARE_MAP[ARCH]` in [`load_input_config._apply_arch_hardware_defaults`](load_input_config.py): `CUs`, `XCC`, `DTYPE_MIs` (from the map’s `ONLY_INCLUDE_MIs` entry for `DataType`), and `WGMUnit` (defaults to `XCC`). You can override any of these in YAML.

**Supported generation today:** `HARDWARE_MAP` lists gfx-style arches (e.g. `gfx942_80cu`). **Fork-parameter profiles** map all listed `ARCH` values to their corresponding codepaths (e.g. `gfx950_128cu` to `gfx950`) in [`fork_params/__init__.py`](fork_params/__init__.py). Unknown `ARCH` raises during validation or registry lookup (see [`fork_params/README.md`](fork_params/README.md)).

### Optional fields (defaults from `CONFIG_DEFAULTS_BY_ARCH`)

- **Per-ARCH defaults:** [`constants.py`](constants.py) defines `CONFIG_DEFAULTS_BY_ARCH`: a full optional-field dict for **each** supported `ARCH` (same keys as `HARDWARE_MAP`). Notably, **`CMS` / `CMS_PRIORITY`** default to **on** for `gfx950` and **off** for gfx942-family ids (gfx942-class CMS not supported in the fork pipeline). To change defaults for an architecture, edit that dict in `constants.py`. Any key present in YAML overrides these via `setdefault` in [`_prepare_config`](load_input_config.py) after required fields and `ARCH` validation.

- **Programmatic use:** [`merged_config_defaults(arch)`](load_input_config.py) returns `dict(CONFIG_DEFAULTS_BY_ARCH[arch])` (used by `geko.optim.config` so CLI/program defaults match YAML prep).

| Key | Role |
|-----|------|
| `StreamK` | `True` = StreamK, `False` = DataParallel |
| `backend` | `"ductile"` or `"tensile"` (default: `"ductile"`; CLI `--backend` overrides). |
| `search_space` | `"heuristic"`, `"generic"`, or `"subtile"` (default: auto from `backend` — `generic` for ductile, `heuristic` for tensile). For full fork ranges with `backend: tensile`, set `search_space: generic`. |
| `MACROTILE_OPT` | Origami macro-tile tuning. Works with the Ductile backend, or with `backend: tensile` when `SIZE_OPTION: 0` (explicit `Sizes:` list); the tensile path emits `Backend: Tensile` and Tensile enumerates the pinned fork space exhaustively. |
| `MT_DU` | Fixed `[MT0, MT1, DU]` when `MACROTILE_OPT` |
| `USE_HEURISTICS` | Refined heuristic param lists per size |
| `ONE_SIZE_PER_CONFIG` | One size per output config file |
| `CMS` / `CMS_PRIORITY` | CMS optimization / tile priority (defaults **on** for `gfx950`, **off** for gfx942 family) |
| `MI_FILTER` | MI filtering: `0` none, `1` moderate, `2` aggressive |
| `EPILOGUES` | Include epilogues |
| `CLUSTER` | `0` = all-in-one, `1` = cluster by top MI |

### Environment variable override

Selected config values can be overridden from the shell environment at runtime.
The allowlist is defined by `ENV_UPDATABLE_KEYS` in [`constants.py`](constants.py),
and each environment variable name must match the config key exactly.

Current keys (from `ENV_UPDATABLE_KEYS`):

- `StreamK`
- `MI_FILTER`
- `DUCTILE_VALIDATION_PROFILE`

Example (force `MI_FILTER` to `0` for one run):

```bash
MI_FILTER=0 python3 scripts/config_generator.py --hipblaslt /path/to/hipBLASLt --config geko/config_generator/config.yaml
```

### Other behavior

- **`MAX_NUM_KERNELS_PER_CONFIG`:** With heuristic search space, caps merged kernels per file (default in `constants.py`; overridable in YAML). With generic search space it is set to effectively unlimited (`sys.maxsize`).
- **`load_prepared_config`** / **`_prepare_config`** ([`load_input_config.py`](load_input_config.py)): if `MACROTILE_OPT` is false, `MT_DU` is cleared to `None`. `MACROTILE_OPT` with a non-Ductile backend is supported for `SIZE_OPTION: 0` only (explicit `Sizes:` list); attempting it with `SIZE_OPTION: 1` raises `NotImplementedError`. The post-processor's `_apply_mt_du` runs regardless of backend.
- **`get_sizes`** ([`sizes.py`](sizes.py)): `SIZE_OPTION=1` (grid) with `MACROTILE_OPT` and the Ductile backend raises `NotImplementedError`.
- **`BUILD_DIR`:** Optional; passed to `geko.utils.build_tensilelite_client` from [`scripts/config_generator.py`](../../scripts/config_generator.py) when generating shell scripts.

---

## Pipeline (high level)

Order matches [`config_generator.run`](config_generator.py):

1. **`load_prepared_config`** — load YAML, `_prepare_config` (defaults, ARCH overrides, validate, arch hardware defaults).
2. **`get_sizes`** — resolve `[M, N, B, K]` list (deduplicated).
3. **`MIDesign`**, **`get_optimization_params`**, **`get_post_processor`** — per-run instances from config.
4. **Per size:** **`generate_fork_params`** — MI groups from `MIDesign`, non-MI params/groups from optimization profile, optional post-processing, then `Groups` assembled ([`fork_param_generator.py`](fork_param_generator.py)); build **`ConfigEntry`**.
5. **`do_cluster`** / **`do_merge`** — group sizes and merge fork params; respect kernel cap for heuristic search space.
6. **`geko.utils.build_tensilelite_client`** — when shell scripts are enabled, ensure prebuilt Tensile client path for generated `.sh` files.
7. **`ConfigSectionGenerator.build_config`** + **`EntityOutputWriter.write_entity_files_only`** / **`append_aggregate_metadata`** — one output bundle per merged entry.

```mermaid
flowchart LR
  yaml[YAML] --> prep[_prepare_config]
  prep --> sizes[get_sizes]
  sizes --> fork[generate_fork_params]
  fork --> cluster[cluster_merge]
  cluster --> write[write_outputs]
```

---

## Design

### Module responsibilities

| Area | Role |
|------|------|
| [`load_input_config.py`](load_input_config.py) | Single place for loaded + normalized config dict |
| [`sizes.py`](sizes.py) | Size list vs grid sources |
| [`mi_designer.py`](mi_designer.py) | MI / macro-tile exploration for each size |
| [`fork_params/`](fork_params/) | Per-`ARCH` (and generic vs heuristic) parameter and post-process implementations |
| [`fork_param_generator.py`](fork_param_generator.py) | Combines MI design, optimization params, post-processor |
| [`cluster_sizes.py`](cluster_sizes.py), [`config_merger.py`](config_merger.py) | Clustering and merging fork params under kernel limits |
| [`config_sections_generator.py`](config_sections_generator.py), [`output_writer.py`](output_writer.py) | Final YAML / scripts / logs |

### Core types ([`shared_utils.py`](shared_utils.py))

- **`ForkParameter`** — Named fork axis: value list, optional comment, `active` flag.
- **`GroupDimension`** — List of MI/param bundles that vary together in Tensile `Groups`.
- **`ConfigEntry`** — One logical output config: sizes, merged `fork_params`, kernel count, MI counts per size.

### Registries

[`fork_params/__init__.py`](fork_params/__init__.py) picks heuristic vs generic classes from `config['ARCH']` and `config['search_space']`. There is no generic fallback: an unregistered `ARCH` fails at optimization-param lookup (post-processor registry can return `None` only when the key is missing—generation still requires a registered optimization profile).