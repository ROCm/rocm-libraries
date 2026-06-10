// Line 2276 (first):
        rocblas_int blocksReset = calculate_nblocks(batch_count, BS1);
// Line 2330 (second):
        rocblas_int const blocksReset = calculate_nblocks(batch_count + 1, BS1);