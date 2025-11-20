# Fusilli Plugin

Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation.

:construction: **This project is under active development, many things don't work yet** :construction:

The plugin builds as a shared library (`fusilli_plugin.so`) providing a `hipDNN` [kernel engine plugin](https://github.com/ROCm/hipDNN/blob/develop/docs/PluginDevelopment.md#creating-a-kernel-engine-plugin) [API](https://github.com/ROCm/hipDNN/blob/839cf6c4bc6fe403d0ef72cb5d7df004e2004743/sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h).

## Developer Guide

### Setup

For the time being, `fusilli-plugin` setup relies on / builds on [Fusilli setup](../sharkfuser/README.md#setup).
Keeping the projects in sync prevents "works on my machine" style bugs.
Requirements that are unique to `fusilli-plugin`, `hipDNN` and `googletest` for
example, are fetched configured and built as part of `fusilli-plugin` build.

After following steps in [Fusilli Setup](../sharkfuser/README.md#setup), build and test
`fusilli-plugin` as follows:
```shell
$ cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++
$ cmake --build build --target all
$ ctest --test-dir build
```
