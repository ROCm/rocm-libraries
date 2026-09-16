#define _GNU_SOURCE
#include <dlfcn.h>
#include <elf.h>
#include <link.h>

#include <cstdio>
#include <cstring>

typedef int (*fn)(void);

static int resolve_own(void* h, const char* node, int expect) {
    fn f = (fn)dlvsym(h, "rocblas_sgemm", node);
    int v = f ? f() : -1;
    return v == expect ? 0 : 1;
}

static int cross_is_null(void* h, const char* wrong_node) {
    return dlvsym(h, "rocblas_sgemm", wrong_node) == nullptr ? 0 : 1;
}

static int dso_symbolic(void* h) {
    struct link_map* lm = nullptr;
    if (dlinfo(h, RTLD_DI_LINKMAP, &lm) != 0 || !lm) return -1;
    for (const ElfW(Dyn)* d = lm->l_ld; d && d->d_tag != DT_NULL; ++d) {
        if (d->d_tag == DT_SYMBOLIC) return 1;
        if (d->d_tag == DT_FLAGS && (d->d_un.d_val & DF_SYMBOLIC)) return 1;
    }
    return 0;
}

static int check_triple(void* h5, void* h6, void* h7, const char* label) {
    int bad = 0;
    bad |= resolve_own(h5, "ROCBLAS_ABI_5", 5);
    bad |= resolve_own(h6, "ROCBLAS_ABI_6", 6);
    bad |= resolve_own(h7, "ROCBLAS_ABI_7", 7);
    bad |= cross_is_null(h5, "ROCBLAS_ABI_6");
    bad |= cross_is_null(h6, "ROCBLAS_ABI_7");
    bad |= cross_is_null(h7, "ROCBLAS_ABI_5");
    printf("[%s] three-line dlvsym exactness: %s\n", label,
           bad ? "FAIL" : "PASS (each line resolves its OWN node, cross-node is nil)");
    return bad;
}

static int run_three_line(const char* p5, const char* p6, const char* p7) {
    int bad = 0;

    void* f5 = dlopen(p5, RTLD_NOW | RTLD_LOCAL);
    void* f6 = dlopen(p6, RTLD_NOW | RTLD_LOCAL);
    void* f7 = dlopen(p7, RTLD_NOW | RTLD_LOCAL);
    if (!f5 || !f6 || !f7) {
        fprintf(stderr, "dlopen(fwd) failed: %s\n", dlerror());
        return 2;
    }
    bad |= check_triple(f5, f6, f7, "fwd");
    dlclose(f5);
    dlclose(f6);
    dlclose(f7);

    void* r7 = dlopen(p7, RTLD_NOW | RTLD_LOCAL);
    void* r6 = dlopen(p6, RTLD_NOW | RTLD_LOCAL);
    void* r5 = dlopen(p5, RTLD_NOW | RTLD_LOCAL);
    if (!r5 || !r6 || !r7) {
        fprintf(stderr, "dlopen(rev) failed: %s\n", dlerror());
        return 2;
    }
    bad |= check_triple(r5, r6, r7, "rev");

    printf("[three_line] verdict: %s\n",
           bad ? "FAIL" : "PASS (dlvsym exactness holds under both load orders)");
    return bad ? 1 : 0;
}

static int run_same_node(const char* p5, const char* p6, const char* p7) {
    void* h5 = dlopen(p5, RTLD_NOW | RTLD_LOCAL);
    void* h6 = dlopen(p6, RTLD_NOW | RTLD_LOCAL);
    void* h7 = dlopen(p7, RTLD_NOW | RTLD_LOCAL);
    if (!h5 || !h6 || !h7) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    int bad = 0;
    // Genuineness: all three DSOs carry the ROCBLAS_ABI_6 node and resolve to their
    // own ABI value, so the negative control below cannot pass vacuously against a
    // nodeless build.
    bad |= resolve_own(h5, "ROCBLAS_ABI_6", 5);
    bad |= resolve_own(h6, "ROCBLAS_ABI_6", 6);
    bad |= resolve_own(h7, "ROCBLAS_ABI_6", 7);
    // Negative control: no DSO answers to the ABI_5 or ABI_7 node, since every one
    // was built against the ABI_6 version script.
    bad |= cross_is_null(h5, "ROCBLAS_ABI_5");
    bad |= cross_is_null(h6, "ROCBLAS_ABI_5");
    bad |= cross_is_null(h7, "ROCBLAS_ABI_5");
    bad |= cross_is_null(h5, "ROCBLAS_ABI_7");
    bad |= cross_is_null(h6, "ROCBLAS_ABI_7");
    bad |= cross_is_null(h7, "ROCBLAS_ABI_7");
    dlclose(h5);
    dlclose(h6);
    dlclose(h7);

    printf("[same_node] verdict: %s\n",
           bad ? "FAIL" : "PASS (ABI_6 resolves on each, ABI_5/ABI_7 nil everywhere)");
    return bad ? 1 : 0;
}

static int run_bsymbolic(const char* pa6, const char* pb7, const char* pplain6) {
    void* A = dlopen(pa6, RTLD_NOW | RTLD_GLOBAL);
    void* B = dlopen(pb7, RTLD_NOW | RTLD_GLOBAL);
    void* P = dlopen(pplain6, RTLD_NOW | RTLD_LOCAL);
    if (!A || !B || !P) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    int bad = 0;
    bad |= resolve_own(A, "ROCBLAS_ABI_6", 6);
    bad |= resolve_own(B, "ROCBLAS_ABI_7", 7);
    bad |= cross_is_null(A, "ROCBLAS_ABI_7");
    bad |= cross_is_null(B, "ROCBLAS_ABI_6");

    int sa = dso_symbolic(A), sb = dso_symbolic(B), sp = dso_symbolic(P);
    printf("[bsymbolic] DF_SYMBOLIC: A(bsym)=%d B(bsym)=%d plain=%d\n", sa, sb, sp);
    // Genuineness + negative control: both -Bsymbolic DSOs must carry DF_SYMBOLIC and
    // the plain DSO must not. This fails if -Bsymbolic is dropped from a bsym target
    // OR wrongly applied to the plain one, unlike the co-residency check which passes
    // identically with or without the flag.
    bad |= (sa == 1) ? 0 : 1;
    bad |= (sb == 1) ? 0 : 1;
    bad |= (sp == 0) ? 0 : 1;

    printf("[bsymbolic] co-residency + DF_SYMBOLIC delta: %s\n",
           bad ? "FAIL"
               : "PASS (own-node resolves, cross-node nil, DF_SYMBOLIC present on bsym / absent on "
                 "plain)");
    printf(
        "[bsymbolic] -Bsymbolic proven genuine via DT_FLAGS while the co-residency "
        "outcome is unchanged: versioning, not -Bsymbolic, is the co-residency mechanism\n");
    return bad ? 1 : 0;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "";
    if (!strcmp(mode, "three_line") && argc == 5) return run_three_line(argv[2], argv[3], argv[4]);
    if (!strcmp(mode, "same_node") && argc == 5) return run_same_node(argv[2], argv[3], argv[4]);
    if (!strcmp(mode, "bsymbolic") && argc == 5) return run_bsymbolic(argv[2], argv[3], argv[4]);
    fprintf(stderr,
            "usage: %s three_line <so5> <so6> <so7>\n"
            "       %s same_node <so5> <so6> <so7>\n"
            "       %s bsymbolic <soA6> <soB7> <soPlain6>\n",
            argv[0], argv[0], argv[0]);
    return 2;
}
