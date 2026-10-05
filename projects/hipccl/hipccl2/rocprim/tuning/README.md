# rocPRIM Tuning

This document is for rocPRIM developers and contributors.

Device-wide algorithms may use tuned configurations to improve performance.
These are located in `rocprim/include/rocprim/device/detail/config`.
Manually finding optimal configurations is impractical, so a set of tools in this folder are provided to automate this process.

## Provided tools

The `rocprim/tuning` folder contains mostly tools and templates.  

### Tuner

Provides various tuning scripts to tune single algorithms.
A tool to run multiple tuners is provided as `run_tuning.py`.
Individual tuners are located in the `tuner` folder.

Tuners are implemented by deriving from `tuner/base_tuner.py` which contains reasonable defaults for Kernel Tuner.
An algorithm tuner needs to provide the tuning params `_get_tune_params(...)` and the restrictions `_get_restrictions(...)`, see `tuner/tuning_device_merge.py` for a simple implementation.
Additionally a template on how Kernel Tuner should build the benchmark is provided in `tuner/templates`.

### Confgen

Provides various scripts and utilities to generate configurations.
This tool reads the output from the tuner (often located in `./output`) and generates a config header.
It can optionally combine with pre-existing config headers if `TARGET:` and `CONFIG:` annotation comments are available.

The configuration header is generated from templates. Most of the shared template content is defined in `confgen/templates/common/base.h.jinja2`. This base template is then inherited by `confgen/templates/{alg}.h.jinja2`, which provides the algorithm-specific details.

## Dependencies

Kernel Tuner depends on the following libraries:

* `numpy`
* `jinja2`
* `kernel-tuner`
* `hip-python`

## Example: Tuning device merge

As an example, the following commands can be used to tune the device merge algorithm:

```sh
# Change directory 
cd projects/rocprim/tuning

# Activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip3 install -i https://test.pypi.org/simple hip-python
pip3 install jinja2 kernel-tuner numpy

# Run tuning
./run_tuning.py --help
./run_tuning.py --algo-regex device_merge

# Generate configurations
./confgen/generate.py --help
./confgen/generate.py --input "./output/*.json" --existing ../rocprim/include/rocprim/device/detail/config/ --output ./tmp --map-gfx gfx90a mi210
```

Note that we have to explicitly tell it which specific GPU architecture we're generating the configs for, using `--map-gfx`.
This is because this information is missing from the output of the tuning scripts.

## Migrating from old tuning infrastructure

First ensure the already existing configs have config annotations. For example:

```cpp
// TARGET: {'gen': 'rdna2', 'arch': 'gfx1030', 'rep': 'amdgcn', 'gpu': 'rx6900'}
template<class Target, class key_type, class value_type>
constexpr auto merge_config_picker() -> std::enable_if_t<
    std::is_same_v<Target, comp_target<gen::rdna2, target_arch::gfx1030, gpu::rx6900, rep::amdgcn>>,
    merge_config_params>
{
    // CONFIG: {'key_type': 'double', 'value_type': 'int64_t', 'block_size_x': 1024, 'ipt': 2}
    if constexpr((rocprim::is_floating_point<key_type>::value) && (sizeof(key_type) > 4)
                 && (sizeof(key_type) <= 8) && (sizeof(value_type) > 4)
                 && (sizeof(value_type) <= 8))
    {
        // ...
    }
}
```

If they are not available, existing configurations can be easily migrated via the migration tool.

```sh
./confgen/migrate.py --help
# Configs are constructed as
#   return merge_config_params{{1024, 8}};
# where the first variable is 'block_size_x', and second variable is 'ipt'.
# The variable names should be equivalent to the Kernel Tuner parameter names.
#
# The following command will modify the config file.
./confgen/migrate.py --config ../rocprim/include/rocprim/device/detail/config/device_merge.hpp -v block_size_x -v ipt
```

To implement tuning and configuration generation the following files need to be created:

* `tuner/tuning_device_{alg}.py`
  * `_get_tune_params(...)` should return the parameters that Kernel Tuner should tune and the range of these parameters.
  * `_get_restrictions(...)` should return whether a set of parameters is valid or not. An invalid kernel will be skipped by Kernel Tuner.
* `tuner/templates/device_{alg}.cpp.jinja2`
  * Specifies the template to generate the benchmark that Kernel Tuner runs.
* `confgen/templates/device_{alg}.h.jinja2`
  * Specifies how to generate the config header.

After creating these files, the tuning and configuration header generation scripts can pick up the existing configurations.
