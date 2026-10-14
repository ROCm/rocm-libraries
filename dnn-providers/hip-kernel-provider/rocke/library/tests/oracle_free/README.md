# Oracle-free tests

This suite is a starting point for oracle-free kernel tests. Its single no-op
test checks no kernel behavior; replace it as real tests are added.
The placeholder is marked `gpu` and requires a HIP-visible gfx942 GPU. It skips
when no GPU is available or the visible device has another architecture.

Add `test_*.py` files here. The installed `rocke_oracle_free_pytest` CTest entry
collects the whole directory and participates in the provider's `quick` and
`ffm-quick` categories, including their larger-tier labels.

From the `rocke` directory, run the source suite with:

```bash
python -m pytest library/tests/oracle_free -v -rs
```

From the installed `bin/hip_kernel_provider` directory, run:

```bash
ctest -R '^rocke_oracle_free_pytest$' --output-on-failure --no-tests=error
```
