#include "port/micropython_embed.h"
static char heap[512 * 1024];
int main(void) {
    int stack_top;
    mp_embed_init(&heap[0], sizeof(heap), &stack_top);
    mp_embed_exec_str(
        "import re\n"
        "m = re.match(r'(\\d+)-(\\w+)', '42-abc')\n"
        "print('re match:', m.group(1), m.group(2))\n"
        "print('re sub:', re.sub('a', 'X', 'banana'))\n"
        "import frozen_test\n"
        "print(frozen_test.greet())\n");
    mp_embed_deinit();
    return 0;
}
