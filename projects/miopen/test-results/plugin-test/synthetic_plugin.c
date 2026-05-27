// synthetic_plugin.c — stand-in for the hipDNN miopen-provider plugin .so
// for the purpose of dynamic-linker verification.
//
// Mirrors how the real plugin is wired in CMakeLists.txt:
//   if(TARGET MIOpen_private)
//     set(_MIOPEN_PROVIDER_LINK_TARGET MIOpen_private)
//   else()
//     set(_MIOPEN_PROVIDER_LINK_TARGET MIOpen)
//   endif()
//
// This .so calls a handful of MIOpen public entry points; the build script
// links one variant against libMIOpen.so.1 (legacy wrapper-mode), another
// against libMIOpen_private.so.1 (Phase-4 short-circuit). Comparing the two
// shows the loader behavior the real plugin will exhibit when built against
// each kind of MIOpen install.

#include <miopen/miopen.h>
#include <stdio.h>

extern "C" __attribute__((visibility("default")))
int plugin_run(void)
{
    size_t maj = 0, min = 0, pat = 0;
    miopenHandle_t h = NULL;

    miopenGetVersion(&maj, &min, &pat);
    miopenCreate(&h);
    miopenDestroy(h);

    return (int)(maj + min + pat);
}
