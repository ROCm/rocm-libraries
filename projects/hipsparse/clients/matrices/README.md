# hipSPARSE test matrices

The `.dvc` files in this directory point to the test matrices that `cmake/ClientMatrices.cmake`
converts for the hipSPARSE tests. Run `dvc pull -R projects/hipsparse/clients/matrices` from the
root of the rocm-libraries repository to download them. Reading the DVC remote (`sparse-matrices`)
doesn't require credentials.

DVC reads its configuration from the `.dvc` directory at the repository root. A sparse checkout of
`projects/hipsparse` omits that directory, and `dvc pull` then fails with
`you are not inside of a DVC repository`. Add it to the checkout first:

```shell
git sparse-checkout add .dvc
```

## Source and license

The matrices are from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu), formerly the
University of Florida Sparse Matrix Collection. They are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Each `.tar.gz` file is the unmodified Matrix Market archive from `https://sparse.tamu.edu/MM/<group>/<name>.tar.gz`.
Its MD5 hash is in the `.dvc` file and in `cmake/ClientMatrices.cmake`. The header of each `.mtx` file
lists the authors and editors of that matrix.

At build time, the hipSPARSE build converts each matrix to a hipSPARSE-specific binary CSR file
(`<name>.bin`) with `deps/convert.cpp`. The conversion changes only the storage format.

| Matrix | Group |
|---|---|
| amazon0312 | SNAP |
| Chebyshev4 | Muite |
| sme3Dc | FEMLAB |
| webbase-1M | Williams |
| rma10 | Bova |
| bibd_22_8 | JGD_BIBD |
| mac_econ_fwd500 | Williams |
| mc2depi | Williams |
| scircuit | Hamm |
| ASIC_320k | Sandia |
| bmwcra_1 | GHS_psdef |
| nos1 to nos7 | HB |
| shipsec1 | DNVS |

## Citation

- Timothy A. Davis and Yifan Hu. 2011. The University of Florida Sparse Matrix Collection.
  ACM Transactions on Mathematical Software 38, 1, Article 1 (December 2011), 25 pages.
  [doi:10.1145/2049662.2049663](https://doi.org/10.1145/2049662.2049663)
- Scott P. Kolodziej, Mohsen Aznaveh, Matthew Bullock, Jarrett David, Timothy A. Davis,
  Matthew Henderson, Yifan Hu, and Read Sandstrom. 2019. The SuiteSparse Matrix Collection
  Website Interface. Journal of Open Source Software 4, 35 (March 2019), 1244.
  [doi:10.21105/joss.01244](https://doi.org/10.21105/joss.01244)
