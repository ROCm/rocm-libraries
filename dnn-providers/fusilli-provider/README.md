# Fusilli Plugin

Fusilli-Plugin: A Fusilli/IREE powered hipDNN plugin for graph JIT compilation.

:construction: **This project is under active development, many things don't work yet** :construction:

The plugin builds as a shared library (`fusilli_plugin.so`) providing a `hipDNN` [kernel engine plugin](https://github.com/ROCm/hipDNN/blob/develop/docs/PluginDevelopment.md#creating-a-kernel-engine-plugin) [API](https://github.com/ROCm/hipDNN/blob/839cf6c4bc6fe403d0ef72cb5d7df004e2004743/sdk/include/hipdnn_sdk/plugin/EnginePluginApi.h).
