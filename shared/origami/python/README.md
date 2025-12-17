# Origami Python Bindings

This directory provides Python bindings for the C++ Origami project.
The bindings allow you to use Origami's functionality directly from Python.

## Installation

First, install `nanobind` using pip:

```bash
pip install nanobind
```

Make sure `ROCM_PATH` is set (e.g. `export ROCM_PATH=/opt/rocm`). Then, build the bindings using the following command:

```bash
python setup.py build_ext --inplace
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Testing

After building the bindings, you can test them using the provided test file:

```bash
python ./origami_test.py
```

## Usage

Once the bindings are built and tested, you can import and use them in your Python scripts:

```python
import origami

hardware = origami.getHardwareForDevice(args.device)

result = origami.select_best_macro_tile_size(
            args.m, args.n, args.k, args.transA, args.transB, hardware, tile_list, args.element_size, args.miDataType, 0.8, args.debug, args.print, args.wgm
        )
```

## Modifying `origami_module.cpp`

To expose more C++ functionality to Python, you can modify the `origami_module.cpp` file.
This involves adding new functions or classes to the module and ensuring they are properly wrapped for Python.
After making your changes, rebuild the bindings using the installation command above.
