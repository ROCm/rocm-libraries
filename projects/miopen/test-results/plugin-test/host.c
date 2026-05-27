// host.c — dlopens a synthetic plugin and invokes its entry point. Mimics
// the way hipDNN loads engine plugins at runtime.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

typedef int (*plugin_run_fn)(void);

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path/to/plugin.so>\n", argv[0]);
        return 2;
    }

    void* h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    plugin_run_fn fn = (plugin_run_fn)dlsym(h, "plugin_run");
    if (!fn) {
        fprintf(stderr, "dlsym(plugin_run) failed: %s\n", dlerror());
        dlclose(h);
        return 1;
    }

    int r = fn();
    printf("plugin_run returned %d\n", r);

    dlclose(h);
    return 0;
}
