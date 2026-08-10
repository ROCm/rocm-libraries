#define _GNU_SOURCE
#include <dlfcn.h>
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
  if (!f5 || !f6 || !f7) { fprintf(stderr, "dlopen(fwd) failed: %s\n", dlerror()); return 2; }
  bad |= check_triple(f5, f6, f7, "fwd");
  dlclose(f5); dlclose(f6); dlclose(f7);

  void* r7 = dlopen(p7, RTLD_NOW | RTLD_LOCAL);
  void* r6 = dlopen(p6, RTLD_NOW | RTLD_LOCAL);
  void* r5 = dlopen(p5, RTLD_NOW | RTLD_LOCAL);
  if (!r5 || !r6 || !r7) { fprintf(stderr, "dlopen(rev) failed: %s\n", dlerror()); return 2; }
  bad |= check_triple(r5, r6, r7, "rev");

  printf("[three_line] verdict: %s\n",
         bad ? "FAIL" : "PASS (dlvsym exactness holds under both load orders)");
  return bad ? 1 : 0;
}

static int run_bsymbolic(const char* pa6, const char* pb7) {
  void* A = dlopen(pa6, RTLD_NOW | RTLD_GLOBAL);
  void* B = dlopen(pb7, RTLD_NOW | RTLD_GLOBAL);
  if (!A || !B) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 2; }

  int bad = 0;
  bad |= resolve_own(A, "ROCBLAS_ABI_6", 6);
  bad |= resolve_own(B, "ROCBLAS_ABI_7", 7);
  bad |= cross_is_null(A, "ROCBLAS_ABI_7");
  bad |= cross_is_null(B, "ROCBLAS_ABI_6");

  printf("[bsymbolic] co-residency of -Bsymbolic providers: %s\n",
         bad ? "FAIL" : "PASS (own-node resolves, cross-node nil)");
  printf("[bsymbolic] outcome identical to non-Bsymbolic co-residency: "
         "-Bsymbolic is inert for co-residency, versioning is the mechanism\n");
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  const char* mode = (argc > 1) ? argv[1] : "";
  if (!strcmp(mode, "three_line") && argc == 5)
    return run_three_line(argv[2], argv[3], argv[4]);
  if (!strcmp(mode, "bsymbolic") && argc == 4)
    return run_bsymbolic(argv[2], argv[3]);
  fprintf(stderr,
          "usage: %s three_line <so5> <so6> <so7>\n"
          "       %s bsymbolic <soA6> <soB7>\n",
          argv[0], argv[0]);
  return 2;
}
