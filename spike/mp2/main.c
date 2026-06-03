#include <stdio.h>
#include <stdlib.h>

#include "port/micropython_embed.h"

int main(void) {
    int stack_top;
    size_t heap_size = (size_t)256 * 1024 * 1024;
    char* heap = malloc(heap_size);
    if (!heap) {
        fprintf(stderr, "heap alloc failed\n");
        return 1;
    }
    mp_embed_init(heap, heap_size, &stack_top);
    mp_embed_exec_str(
        "from ck_dsl.instances.common.conv_implicit_gemm import "
        "ImplicitGemmConvSpec, ConvProblem, build_implicit_gemm_conv\n"
        "from ck_dsl.core.lower_llvm import lower_kernel_to_llvm\n"
        "spec = ImplicitGemmConvSpec(problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, "
        "K=64, R=3, S=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1), tile_m=64, tile_n=64, "
        "tile_k=64, warp_m=2, warp_n=2, warp_tile_m=32, warp_tile_n=32, warp_tile_k=16)\n"
        "kd = build_implicit_gemm_conv(spec, arch='gfx950')\n"
        "llvm = lower_kernel_to_llvm(kd, arch='gfx950')\n"
        "print('EMBED CONV LLVM_LEN', len(llvm))\n"
        "print('EMBED CONV LLVM_SUM', sum(bytearray(llvm.encode())))\n");
    mp_embed_deinit();
    free(heap);
    return 0;
}
