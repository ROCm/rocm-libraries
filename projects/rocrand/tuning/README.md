# rocRAND Tuning

This document is for rocRAND developers and contributors.

The pseudo-random number generators (RNG engines) in rocRAND use tuned launch
configurations (block size and grid size) to improve performance on different GPU
architectures. These are located in `library/src/rng/config/<engine>_config.hpp`.
Manually finding optimal configurations is impractical, so the set of tools in this
folder is provided to automate the process. The tooling is built on top of
[Kernel Tuner](https://kerneltuner.github.io/kernel_tuner/stable/contents.html).

## Supported engines

The following engines have tuners (`tuner/tuning_<engine>.py`):

* `lfsr113`
* `mrg31k3p`
* `mrg32k3a`
* `mt19937`
* `mtgp32`
* `philox4_32_10`
* `threefry2_32_20`
* `threefry2_64_20`
* `threefry4_32_20`
* `threefry4_64_20`
* `xorwow`

Run `./run_tuning.py --list` to print the list that is actually discovered.

## Provided tools

The `tuning` folder contains mostly tools and templates.

### Tuner

Provides tuning scripts to tune individual engines. A tool to run multiple tuners is
provided as `run_tuning.py`; the individual tuners live in the `tuner` folder.

Tuners are implemented by deriving from `tuner/base_tuner.py`, which contains reasonable
defaults for Kernel Tuner. The base class already tunes `block_size_x` and `grid_size`
over the ranges defined in `base_tuner.py`, so most engines only need a thin subclass
that sets the engine name (see `tuner/tuning_xorwow.py` for the simplest example). An
engine that needs extra constraints overrides `_get_restrictions(...)` to reject invalid
parameter combinations (see `tuner/tuning_mt19937.py` and `tuner/tuning_mtgp32.py`).

Each engine also needs a benchmark wrapper template in `tuner/templates/<engine>.cpp.jinja2`,
which generates the C++ code that Kernel Tuner compiles and benchmarks.

### Confgen

Provides scripts and utilities to generate configuration headers.
`confgen/generate.py` reads the tuner output (JSON files, by default under `./output`)
and generates a config header per engine. It can optionally merge with the existing
config headers in `library/src/rng/config/` to keep previously tuned configurations for
other architectures.

The configuration header is generated from templates. Most of the shared template
content is defined in `confgen/templates/common/base.h.jinja2`. This base template is
then inherited by `confgen/templates/<engine>.h.jinja2`, which provides the
engine-specific details.

## Dependencies

The tooling depends on the following Python libraries:

* `numpy`
* `jinja2`
* `kernel-tuner`
* `hip-python`

## Example: tuning xorwow

As an example, the following commands can be used to tune the `xorwow` engine:

```sh
# Change directory
cd projects/rocrand/tuning

# Activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -i https://test.pypi.org/simple hip-python
pip3 install jinja2 kernel-tuner numpy

# Run tuning (--algo-regex accepts a regex matched against engine names)
./run_tuning.py --help
./run_tuning.py --algo-regex xorwow

# Generate configurations, merging with the existing headers.
# --target-arch/-t tells confgen which GPU architecture these results are for;
# this information is not present in the tuner output and must be supplied explicitly.
./confgen/generate.py --help
./confgen/generate.py \
    --input "./output/*.json" \
    --existing ../library/src/rng/config/ \
    --output ./tmp \
    --target-arch gfx90a
```

The generated headers are written to the `--output` directory (`./tmp` above). Once you
are satisfied with the results, copy them over the headers in
`library/src/rng/config/`.

## Simulation mode and strategy comparison

The tuner can run in simulation mode (`--simulation-mode`, requires `--arch-name`) to
replay a previously collected full search space cache under a different search strategy,
without touching the GPU. `utils/compare_simulated_strategies.py` compares the results of
multiple simulated strategies against the best-known times and renders a heatmap, which
is useful for choosing a tuning strategy:

```sh
python3 utils/compare_simulated_strategies.py --arch gfx90a --algo xorwow
```

## Adding a new engine

To implement tuning and configuration generation for a new engine, the following files
need to be created:

* `tuner/tuning_<engine>.py`
  * A subclass of `BaseTuner` that returns the engine name from `_get_default_args(...)`.
  * Override `_get_restrictions(...)` if the engine has constraints on valid
    `block_size_x` / `grid_size` combinations.
* `tuner/templates/<engine>.cpp.jinja2`
  * The template used to generate the benchmark that Kernel Tuner runs.
* `confgen/templates/<engine>.h.jinja2`
  * The template that specifies how to generate the config header.

After creating these files, `run_tuning.py` will automatically discover the new engine
and `confgen/generate.py` can pick up and merge its configurations.
