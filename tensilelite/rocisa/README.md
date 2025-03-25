# rocIsa

This ia a Python module wrapped with Nanobind. Need to install ``nanobind`` before compiling this module.

## How to install

```
pip3 install nanobind
```

## How to build the module independently

Simple version.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ ..
```

If you want to specify the current python version,

```
mkdir build
cd build
cmake -DPYTHON_VERSION=$(python3 --version 2>&1 | cut -d ' ' -f 2) -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ ..
make -j8
```

## How to use the module without installing

```
export PYTHONPATH=<path-to-build-folder>/lib
```

For more information, please check the doc (on-going).
