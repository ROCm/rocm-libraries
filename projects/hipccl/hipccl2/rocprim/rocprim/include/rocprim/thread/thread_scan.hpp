/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2021, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef ROCPRIM_THREAD_THREAD_SCAN_HPP_
#define ROCPRIM_THREAD_THREAD_SCAN_HPP_


#include "../config.hpp"
#include "../functional.hpp"

BEGIN_ROCPRIM_NAMESPACE

 /**
  * \addtogroup UtilModule
  * @{
  */

 /**
  * \name Sequential prefix scan over statically-sized array types
  * @{
  */

  /// \brief Perform a sequential exclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
  /// \tparam LENGTH           - Length of \p input and \p output arrays
  /// \tparam T                - <b>[inferred]</b> The data type to be scanned.
  /// \tparam ScanOp           - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
  /// \param inclusive [in]    - Initial value for inclusive aggregate
  /// \param exclusive [in]    - Initial value for exclusive aggregate
  /// \param input [in]        - Input array
  /// \param output [out]      - Output array (may be aliased to \p input)
  /// \param scan_op [in]      - Binary scan operator
  /// \return                  - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_exclusive(
     T                   inclusive,
     T                   exclusive,
     T                   *input,                 ///< [in] Input array
     T                   *output,                ///< [out] Output array (may be aliased to \p input)
     ScanOp              scan_op,                ///< [in] Binary scan operator
     Int2Type<LENGTH>    /*length*/)
 {
     ROCPRIM_UNROLL
     for (int i = 0; i < LENGTH; ++i)
     {
         inclusive = scan_op(exclusive, input[i]);
         output[i] = exclusive;
         exclusive = inclusive;
     }

     return inclusive;
 }



 /// \brief Perform a sequential exclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
 /// \tparam LENGTH           - Length of \p input and \p output arrays
 /// \tparam T                - <b>[inferred]</b> The data type to be scanned.
 /// \tparam ScanOp           - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 /// \param input [in]        - Input array
 /// \param output [out]      - Output array (may be aliased to \p input)
 /// \param scan_op [in]      - Binary scan operator
 /// \param prefix  [in]      - Prefix to seed scan with
 /// \param apply_prefix [in] - Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
 /// \return                  - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_exclusive(
     T           *input,                 ///< [in] Input array
     T           *output,                ///< [out] Output array (may be aliased to \p input)
     ScanOp      scan_op,                ///< [in] Binary scan operator
     T           prefix,                 ///< [in] Prefix to seed scan with
     bool        apply_prefix = true)    ///< [in] Whether or not the calling thread should apply its prefix.  If not, the first output element is undefined.  (Handy for preventing thread-0 from applying a prefix.)
 {
     T inclusive = input[0];
     if (apply_prefix)
     {
         inclusive = scan_op(prefix, inclusive);
     }
     output[0] = prefix;
     T exclusive = inclusive;

     return thread_scan_exclusive(inclusive, exclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }

 /// \brief Perform a sequential exclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
 /// \tparam LENGTH           - Length of \p input and \p output arrays
 /// \tparam T                - <b>[inferred]</b> The data type to be scanned.
 /// \tparam ScanOp           - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 /// \param input [in]        - Input array
 /// \param output [out]      - Output array (may be aliased to \p input)
 /// \param scan_op [in]      - Binary scan operator
 /// \param prefix  [in]      - Prefix to seed scan with
 /// \param apply_prefix [in] - Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
 /// \return                  - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_exclusive(
     T           (&input)[LENGTH],       ///< [in] Input array
     T           (&output)[LENGTH],      ///< [out] Output array (may be aliased to \p input)
     ScanOp      scan_op,                ///< [in] Binary scan operator
     T           prefix,                 ///< [in] Prefix to seed scan with
     bool        apply_prefix = true)    ///< [in] Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
 {
     return thread_scan_exclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
 }

 /// \brief Perform a sequential exclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
 /// \tparam LENGTH           - Length of \p input and \p output arrays
 /// \tparam T                - <b>[inferred]</b> The data type to be scanned.
 /// \tparam ScanOp           - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 /// \param inclusive [in]    - Initial value for inclusive aggregate
 /// \param input [in]        - Input array
 /// \param output [out]      - Output array (may be aliased to \p input)
 /// \param scan_op [in]      - Binary scan operator
 /// \return                  - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_inclusive(
     T                   inclusive,
     T                   *input,                 ///< [in] Input array
     T                   *output,                ///< [out] Output array (may be aliased to \p input)
     ScanOp              scan_op,                ///< [in] Binary scan operator
     Int2Type<LENGTH>    /*length*/)
 {
     ROCPRIM_UNROLL
     for (int i = 0; i < LENGTH; ++i)
     {
         inclusive = scan_op(inclusive, input[i]);
         output[i] = inclusive;
     }

     return inclusive;
 }


/// \brief Perform a sequential inclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
/// \tparam LENGTH     - LengthT of \p input and \p output arrays
/// \tparam T          - <b>[inferred]</b> The data type to be scanned.
/// \tparam ScanOp     - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
/// \param input [in] - Input array
/// \param output [out] - Output array (may be aliased to \p input)
/// \param scan_op [in] - Binary scan operator
/// \return - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_inclusive(
     T           *input,
     T           *output,
     ScanOp      scan_op)
 {
     T inclusive = input[0];
     output[0] = inclusive;

     // Continue scan
     return thread_scan_inclusive(inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }


 /// \brief Perform a sequential inclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
 /// \tparam LENGTH     - LengthT of \p input and \p output arrays
 /// \tparam T          - <b>[inferred]</b> The data type to be scanned.
 /// \tparam ScanOp     - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 /// \param input [in] - Input array
 /// \param output [out] - Output array (may be aliased to \p input)
 /// \param scan_op [in] - Binary scan operator
 /// \return - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_inclusive(
     T           (&input)[LENGTH],       ///< [in] Input array
     T           (&output)[LENGTH],      ///< [out] Output array (may be aliased to \p input)
     ScanOp      scan_op)                ///< [in] Binary scan operator
 {
     return thread_scan_inclusive<LENGTH>((T*) input, (T*) output, scan_op);
 }


 /// \brief Perform a sequential inclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
 /// \tparam LENGTH           - LengthT of \p input and \p output arrays
 /// \tparam T                - <b>[inferred]</b> The data type to be scanned.
 /// \tparam ScanOp           - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 /// \param input [in]        - Input array
 /// \param output [out]      - Output array (may be aliased to \p input)
 /// \param scan_op [in]      - Binary scan operator
 /// \param prefix  [in]      - Prefix to seed scan with
 /// \param apply_prefix [in] - Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
 /// \return                  - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_inclusive(
     T           *input,                 ///< [in] Input array
     T           *output,                ///< [out] Output array (may be aliased to \p input)
     ScanOp      scan_op,                ///< [in] Binary scan operator
     T           prefix,                 ///< [in] Prefix to seed scan with
     bool        apply_prefix = true)    ///< [in] Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
 {
     T inclusive = input[0];
     if (apply_prefix)
     {
         inclusive = scan_op(prefix, inclusive);
     }
     output[0] = inclusive;

     // Continue scan
     return thread_scan_inclusive(inclusive, input + 1, output + 1, scan_op, Int2Type<LENGTH - 1>());
 }


 /// \brief Perform a sequential inclusive prefix scan over \p LENGTH elements of the \p input array.  The aggregate is returned.
 /// \tparam LENGTH           - LengthT of \p input and \p output arrays
 /// \tparam T                - <b>[inferred]</b> The data type to be scanned.
 /// \tparam ScanOp           - <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
 /// \param input [in]        - Input array
 /// \param output [out]      - Output array (may be aliased to \p input)
 /// \param scan_op [in]      - Binary scan operator
 /// \param prefix  [in]      - Prefix to seed scan with
 /// \param apply_prefix [in] - Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
 /// \return                  - Aggregate of the scan
 template <
     int         LENGTH,
     typename    T,
     typename    ScanOp>
 ROCPRIM_DEVICE inline
 T thread_scan_inclusive(
     T           (&input)[LENGTH],
     T           (&output)[LENGTH],
     ScanOp      scan_op,
     T           prefix,
     bool        apply_prefix = true)
 {
     return thread_scan_inclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix, apply_prefix);
 }


 //@}  end member group

 /** @} */       // end group UtilModule

 END_ROCPRIM_NAMESPACE

 #endif // ROCPRIM_THREAD_THREAD_SCAN_HPP_
