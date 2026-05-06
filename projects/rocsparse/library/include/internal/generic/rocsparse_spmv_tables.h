/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the Software), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED AS IS, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************
 *
 * This header holds reusable doxygen documentation fragments for the
 * rocsparse_spmv / rocsparse_v2_spmv algorithm tables. Public-API doxygen
 * comments reference these fragments via \\copydetails so the table content
 * lives in a single place rather than being duplicated in each header.
 *
 * The fragments are defined as doxygen \\page entities, which are parsed
 * by doxygen but never instantiate any C/C++ code. The pages themselves
 * are excluded from the generated reference output via EXCLUDE_SYMBOLS
 * in the Doxyfile so they do not appear in the navigation; only the
 * documentation text attached to them is reused.
 *
 * ************************************************************************ */

#ifndef ROCSPARSE_SPMV_TABLES_H
#define ROCSPARSE_SPMV_TABLES_H

/*! \page rocsparse_internal_doc_spmv_algorithm_tables Internal SpMV algorithm tables fragment
 *
 *  <table>
 *  <caption>CSR/CSC Algorithms</caption>
 *  <tr><th>Algorithm                            <th>Deterministic  <th>Preprocessing  <th>Notes
 *  <tr><td>rocsparse_spmv_alg_csr_rowsplit</td> <td>Yes</td>       <td>No</td>        <td>Is best suited for matrices with all rows having a similar number of non-zeros. Can outperform adaptive and LRB algorithms in certain sparsity patterns. Will perform very poorly if some rows have few non-zeros and some rows have many non-zeros.</td>
 *  <tr><td>rocsparse_spmv_alg_csr_stream</td>   <td>Yes</td>       <td>No</td>        <td>[Deprecated] The old name for rocsparse_spmv_alg_csr_rowsplit.</td>
 *  <tr><td>rocsparse_spmv_alg_csr_adaptive</td> <td>No</td>        <td>Yes</td>       <td>Generally the fastest algorithm across all matrix sparsity patterns. This includes matrices that have some rows with many non-zeros and some rows with few non-zeros. Requires lengthy preprocessing that needs to be amortized over many subsequent sparse vector products.</td>
 *  <tr><td>rocsparse_spmv_alg_csr_lrb</td>      <td>No</td>        <td>Yes</td>       <td>Like the adaptive algorithm, it generally performs well across all matrix sparsity patterns. Generally not as fast as the adaptive algorithm, however, it uses a much faster pre-processing step. Good for when only a small number of sparse vector products will be performed.</td>
 *  <tr><td>rocsparse_spmv_alg_csr_nnzsplit</td> <td>No</td>        <td>Yes</td>       <td>Like the adaptive algorithm, it generally performs well across all matrix sparsity patterns. Generally not as fast as the adaptive algorithm but faster than the LRB algorithm. It uses a much faster preprocessing step than LRB. Good when the number of sparse vector products that will be performed is less than one hundred. If more products need to be computed, the adaptive algorithm is probably faster.</td>
 *  </table>
 *
 *  <table>
 *  <caption>COO Algorithms</caption>
 *  <tr><th>COO Algorithms                     <th>Deterministic   <th>Preprocessing <th>Notes
 *  <tr><td>rocsparse_spmv_alg_coo</td>        <td>Yes</td>        <td>Yes</td>      <td>Generally not as fast as the atomic algorithm but is deterministic.</td>
 *  <tr><td>rocsparse_spmv_alg_coo_atomic</td> <td>No</td>         <td>No</td>       <td>Generally the fastest COO algorithm.</td>
 *  </table>
 *
 *  <table>
 *  <caption>ELL Algorithms</caption>
 *  <tr><th>ELL Algorithms                <th>Deterministic   <th>Preprocessing <th>Notes
 *  <tr><td>rocsparse_spmv_alg_ell</td>   <td>Yes</td>        <td>No</td>       <td></td>
 *  </table>
 *
 *  <table>
 *  <caption>BSR Algorithms</caption>
 *  <tr><th>BSR Algorithm                 <th>Deterministic   <th>Preprocessing <th>Notes
 *  <tr><td>rocsparse_spmv_alg_bsr</td>   <td>Yes</td>        <td>No</td>       <td></td>
 *  </table>
 */

#endif /* ROCSPARSE_SPMV_TABLES_H */
