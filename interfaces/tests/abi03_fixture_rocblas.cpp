#ifndef ABI_VER
#define ABI_VER 0
#endif
extern "C" int rocblas_create_handle(void) {
    return ABI_VER;
}
extern "C" int rocblas_destroy_handle(void) {
    return ABI_VER;
}
extern "C" int rocblas_get_pointer_mode(void) {
    return ABI_VER;
}
extern "C" int rocblas_get_stream(void) {
    return ABI_VER;
}
extern "C" int rocblas_saxpy(void) {
    return ABI_VER;
}
extern "C" int rocblas_sdot(void) {
    return ABI_VER;
}
extern "C" int rocblas_set_pointer_mode(void) {
    return ABI_VER;
}
extern "C" int rocblas_set_stream(void) {
    return ABI_VER;
}
extern "C" int rocblas_sgemm(void) {
    return ABI_VER;
}
extern "C" int rocblas_sgemm_64(void) {
    return ABI_VER;
}
extern "C" int rocblas_sgemm_strided_batched(void) {
    return ABI_VER;
}
