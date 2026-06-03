/* Arch-A integration harness (WIP): embed MicroPython, import the frozen
 * codegen module, run it. Currently validates the embed + freeze pipeline with a
 * trivial frozen module. Frozen modules are searched via the ".frozen" sys.path
 * entry (present by default). KNOWN ISSUE: the manifest currently registers the
 * module name with a ".py" suffix, so `import frozen_test` does not match yet —
 * manifest-naming fix pending (see plan). */
#include "port/micropython_embed.h"

static char heap[256 * 1024];

int main(void) {
    int stack_top;
    mp_embed_init(&heap[0], sizeof(heap), &stack_top);
    mp_embed_exec_str(
        "import frozen_test\n"
        "print(frozen_test.greet())\n");
    mp_embed_deinit();
    return 0;
}
