// Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef BWR_UTILS_HPP
#define BWR_UTILS_HPP

#include <thrust/host_vector.h>
#include <thrust/rocthrust_version.hpp>

#include <hip/hip_runtime_api.h>

#include <string>

// These macros are be used to get a stringified version of a type name (see get_typename_str below) .
// If data_type matches key_type, then we return a stringified version of key_type.
#define STRING(s) #s
#define IF_TYPE_TEST(data_type, key_type)       \
  if (std::is_same<data_type, key_type>::value) \
    return STRING(key_type);
#define ELSE_IF_TYPE_TEST(data_type, key_type) else IF_TYPE_TEST(data_type, key_type)

namespace bwr_utils
{

// The separator character used in the ROCm and rocThrust versions.
static const std::string ver_sep = std::string(".");

// These members shouldn't need to be called directly from outside this file.
namespace crc
{
/*! \brief Builds a table that's used to cache values for the CRC algorithm.
 * \note: We're assuming the system is little-endian here.
 * \param table Pointer to the buffer to use to store the table (must be of size 256).
 */
void make_table(uint32_t* table)
{
  table[0]     = 0;
  uint32_t crc = 1;
  uint8_t i    = 128;
  do
  {
    // Note: The constant below is the polynomial for the
    // CRC-32/ISO-HDLC version of the 32-bit algorithm.
    if (crc & 1)
    {
      crc = (crc >> 1) ^ 0xedb88320;
    }
    else
    {
      crc >>= 1;
    }

    for (uint32_t j = 0; j < 256; j += 2 * i)
    {
      table[i + j] = crc ^ table[j];
    }
    i >>= 1;
  } while (i > 0);
}

/*! \brief Performs a 32-bit cyclic redundancy check on the data it's passed.
 * \param data Pointer to the data to compute the check for, as a byte-array.
 * \param len Number of bytes in the data buffer.
 */
uint32_t crc(const uint8_t* data, size_t len)
{
  // Precompute values we know we'll use frequently and store them in a
  // lookup table.
  static uint32_t table[256];
  static bool table_exists = false;
  if (!table_exists)
  {
    make_table(table);
    table_exists = true;
  }

  // Start with a negated version of (unsigned) 0 to handle the
  // case where data contains zeros.
  uint32_t crc = 0xffffffff;

  for (size_t i = 0; i < len; i++)
  {
    uint32_t index = (crc ^ data[i]) & 0xff;
    crc            = (crc >> 8) ^ table[index];
  }

  // Under the negation we performed at the start.
  crc = crc ^ 0xffffffff;

  return crc;
}
} // namespace crc

/*! \brief Uses a cyclic redundancy check (CRC) to hash the contents of the buffer passed in.
 * \note The result does not include type or length information, which is necessary to uniquely
 * identify a buffer (eg. two vectors of zeros that are of different lengths will both
 * produce the same CRC hash; two vectors of zeros that are the same length but different integral
 * types may also produce the same CRC hash.)
 *
 * \tparam T Buffer element type
 * \param buffer Pointer to the buffer to hash
 * \param size Number of elements in the range to hash
 * \return String hash value as indicated above
 */
template <typename T>
std::string hash_buffer_crc(T* buffer, const size_t size)
{
  const uint8_t* bytes    = reinterpret_cast<uint8_t*>(buffer);
  const size_t num_bytes  = size * sizeof(T);
  const uint32_t hash_val = crc::crc(bytes, num_bytes);

  std::stringstream sstream;
  sstream << hash_val;

  return sstream.str();
}

/*! \brief Uses a cyclic redundancy check (CRC) to hash the contents of the buffer passed in.
 * \note Includes length, but not type information. The length information is important because
 * two zero-containing vectors of different lengths will both hash to zero.
 *
 * \param begin Iterator pointing to the begining of the range to hash
 * \param end Iterator pointing to one-past-the-end of the range to hash
 * \return String of the form "hash,length"
 */
template <typename T>
std::string hash_vector(const T begin, const T end)
{
  const size_t size = end - begin;
  return hash_buffer_crc(thrust::raw_pointer_cast(&(*begin)), size) + "," + std::to_string(size);
}

/*! \brief This function returns a string version of the template parameter typename it's passed.
 * \note Because C++ has no reflection/introspection, we must manually define strings for each
 * type we want to be able to do this for.
 *
 * Currently, only types used in tests in test/test_reproducibility.cpp are supported here.
 * Input types passed passed to TokenHelper (below) also need to be defined here.
 *
 * \tparam The type to find a string name for.
 * \return String version of the type, or empty string if not found.
 */
template <typename T>
std::string get_typename_str()
{
  IF_TYPE_TEST(T, thrust::host_vector<short>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<int>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<long long>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<unsigned short>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<unsigned int>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<unsigned long long>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<float>)
  ELSE_IF_TYPE_TEST(T, thrust::host_vector<double>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<short>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<int>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<long long>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<unsigned short>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<unsigned int>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<unsigned long long>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<float>)
  ELSE_IF_TYPE_TEST(T, thrust::device_vector<double>)
  ELSE_IF_TYPE_TEST(T, short)
  ELSE_IF_TYPE_TEST(T, int)
  ELSE_IF_TYPE_TEST(T, long long)
  ELSE_IF_TYPE_TEST(T, unsigned short)
  ELSE_IF_TYPE_TEST(T, unsigned int)
  ELSE_IF_TYPE_TEST(T, unsigned long long)
  ELSE_IF_TYPE_TEST(T, float)
  ELSE_IF_TYPE_TEST(T, double)
  else throw std::runtime_error("Unable to lookup type name in get_typename_str()");

  return "";
}

/*! \brief Builds and returns a string containing the rocThrust version number.
 *
 * \return String of the form "<major version>.<minor version>.<patch version>".
 */
std::string get_rocthrust_version()
{
  static const std::string rocthrust_ver =
    std::to_string(ROCTHRUST_VERSION_MAJOR) + ver_sep + std::to_string(ROCTHRUST_VERSION_MINOR) + ver_sep
    + std::to_string(ROCTHRUST_VERSION_PATCH);
  return rocthrust_ver;
}

/*! \brief Builds and returns a string containing the GPU architecture.
 *
 * \return String of the form "gfx<arch number>".
 */
std::string get_gpu_arch()
{
  hipDeviceProp_t device_prop;
  if (hipGetDeviceProperties(&device_prop, 0) != hipSuccess)
  {
    throw std::runtime_error("hipGetDeviceProperties failure");
  }

  static const std::string gpu_arch(device_prop.gcnArchName);

  return gpu_arch;
}

/*! \brief Builds and returns a string containing the ROCm version number.
 * \note We intentionally omit the "HIP_PATCH_VERSION" here, since we don't
 * anticipate results to change at that granularity, and storing results at
 * that granularity would require frequent database updates.
 *
 * \return String of the form "<major version>.<minor version>".
 */
std::string get_rocm_version()
{
  static const std::string runtime_ver =
    std::to_string(HIP_VERSION_MAJOR) + ver_sep + std::to_string(HIP_VERSION_MINOR);

  return runtime_ver;
}

/*! \brief Returns a "token" string the uniquely identifies a scalar value.
 *
 * \tparam T Type of the scalar
 * \param val Scalar value to hash
 * \return String of the form "scalar<typename>(value)"
 */
template <typename T>
std::string get_scalar_token(const T& val)
{
  return "scalar<" + get_typename_str<T>() + ">(" + std::to_string(val) + ")";
}

/*! \brief Returns a "token" string the uniquely identifies a vector.
 * \note This version accepts an existing hash string and type string
 * (obtained from hash_vector and get_typename_str, respectively). This
 * allows callers to make multiple calls to this function without invoking
 * hash_vector multiple times, since it may be expensive if the vector is
 * large.
 *
 * \param vec_hash Hash string for the vector, obtained from hash_vector.
 * \param data_type Data type name string, obtained from get_typename_str.
 * \return String of the form "vector<typename>(hash)".
 */
std::string get_vector_token(const std::string& vec_hash, const std::string& data_type)
{
  return "vector<" + data_type + ">(" + vec_hash + ")";
}

/*! \brief Returns a "token" string the uniquely identifies a vector iterator.
 * \note This version accepts an existing hash string and type string
 * (obtained from hash_vector and get_typename_str, respectively). This
 * allows callers to make multiple calls to this function without invoking
 * hash_vector multiple times, since it may be expensive if the vector is
 * large.
 *
 * \param vec_hash Hash string for the vector, obtained from hash_vector.
 * \param data_type Data type name string, obtained from get_typename_str.
 * \param offset The iterator's current offset from the beginning of the vector.
 * \return String of the form "iter(vector<typename>(vec_hash),offset)".
 */
std::string get_iterator_token(const std::string& vec_hash, const std::string& data_type, const size_t offset)
{
  return "iter(" + get_vector_token(vec_hash, data_type) + "," + std::to_string(offset) + ")";
}

/*! \brief Returns a "token" string that uniquely identifies a functor (callable object).
 *
 * \tparam T The data type that the functor operates on.
 * \param functor_type String representing the functor type, without the <T> datatype (eg. "thrust::plus" for functor
 * thrust::plus<T>). \return Strin gof the form "function<typename>(function_type)"
 */
template <typename T>
std::string get_functor_token(const std::string& functor_type)
{
  return "functor<" + get_typename_str<T>() + ">(" + functor_type + ")";
}

/*! \brief Builds a compound string hash value from existing string hashes.
 *
 * \param begin Iterator pointing to the beginning of the vector of string hashes to combine
 * \param end Iterator pointing to the end of the vector of string hashes to combine
 * \return String hash value of the form "(<hash 0>,<hash1>)"
 */
std::string build_compound_token(const std::vector<std::string>::const_iterator begin,
                                 const std::vector<std::string>::const_iterator end)
{
  std::string token = "(";
  if (begin != end)
  {
    for (auto it = begin; it != end; it++)
    {
      token += *it;
      if (it + 1 != end)
      {
        token += ",";
      }
    }
  }
  token += ")";

  return token;
}

/*! \brief Builds a string "token" from a function call's inputs. This token is unique to a particular function call.
 *
 * \param list Vector of strings where the first element is the function name, and the remainder are hashes representing
 * inputs to the function call. \return String hash value of the form "fcn_name(<hash 0>,<hash1>,...)"
 */
std::string build_input_token(const std::vector<std::string>& list)
{
  return *(list.begin()) + build_compound_token(list.begin() + 1, list.end());
}

/*! \brief Builds a string "token" from a function call's outputs. This token is unique to a particular return value.
 *
 * \param list Vector of string hashes representing outputs to the function call.
 * \return String hash value of the form "(<hash 0>,<hash1>,...)"
 */
std::string build_output_token(const std::vector<std::string>& list)
{
  return build_compound_token(list.begin(), list.end());
}

/*! \brief This class helps create input and output tokens representing a function call.
 * You can call build_input_token and build_output_token, passing them string and iterators,
 * and the class will combine them together into a single token.
 * Eg.
 * \code
 * // Suppose we want to record this call:
 * std::vector<int> input = {...};
 * thrust::inclusive_scan(policy, d_input.begin(), d_input.end(), d_output.begin(), thrust::plus<int>);
 *
 * TokenHelper helper;
 * token_helper.build_input_token(
 *   "thrust::inclusive_scan",
 *   d_input.begin(),
 *   d_input.end(),
 *   {bwr_utils::get_functor_token<T>("thrust::plus")}
 * );
 *
 * token_helper.build_output_token(d_output.begin(), d_output.size());
 *
 * // Can access the input/output tokens using:
 * std::string input_token = helper.get_input_token();
 * std::string output_Token = helper.get_output_token();
 * \endcode
 */
class TokenHelper
{
public:
  TokenHelper()  = default;
  ~TokenHelper() = default;

  /*! \brief Returns a "token" string that uniquely identifies a function call's inputs.
   * \note It's assumed that you'll pass in all the function call's inputs in the order they appear in the call from
   * left to right.
   *
   * \tparam Iter The input vector iterator type
   * \param fcn_name The name of the function being called
   * \param input_begin Iterator to the beginning of the input data vector
   * \param input_end Iterator to the end of the input data vector
   * \param extra_inputs A vector of (ordered) extra input tokens obtained from the bwr_utils::get_*_token functions.
   */
  template <typename Iter>
  void build_input_token(const std::string& fcn_name,
                         const Iter input_begin,
                         const Iter input_end,
                         std::vector<std::string>&& extra_inputs = {})
  {
    save_input_token(fcn_name, input_begin, input_end, {}, std::forward<std::vector<std::string>&&>(extra_inputs));
  }

  /*! \brief Returns a "token" string that uniquely identifies a function call's inputs. This version accepts both
   * values and keys. \note It's assumed that you'll pass in all the function call's inputs in the order they appear in
   * the call from left to right.
   *
   * \tparam KeyIter The key input vector iterator type
   * \tparam ValueIter The value input vector iterator type
   * \param fcn_name The name of the function being called
   * \param key_input_begin Iterator to the beginning of the key input data vector
   * \param key_input_end Iterator to the end of the key input data vector
   * \param key_input_begin Iterator to the beginning of the value input data vector
   * \param extra_inputs A vector of (ordered) extra input tokens obtained from the bwr_utils::get_*_token functions.
   */
  template <typename KeyIter, typename ValueIter>
  void build_input_token(
    const std::string& fcn_name,
    const KeyIter key_input_begin,
    const KeyIter key_input_end,
    const ValueIter value_input_begin,
    std::vector<std::string>&& extra_inputs = {})
  {
    save_input_token(
      fcn_name,
      key_input_begin,
      key_input_end,
      std::vector<ValueIter>({value_input_begin}),
      std::forward<std::vector<std::string>&&>(extra_inputs));
  }

  /*! \brief Returns a "token" string that uniquely identifies a function call's output.
   *
   * \tparam Iter The output vector iterator type
   * \param output_begin Iterator to the beginning of the output data vector
   * \param size Number of elements in the output vector.
   */
  template <typename Iter>
  void build_output_token(const Iter output_begin, const size_t size)
  {
    save_output_token(output_begin, {}, size);
  }

  /*! \brief Returns a "token" string that uniquely identifies a function call's outputs. This version accepts both
   * values and keys.
   *
   * \tparam KeyIter The key output vector iterator type
   * \tparam ValueIter The value output vector iterator type
   * \param key_output_begin Iterator to the beginning of the key output data vector
   * \param value_output_begin Iterator to the beginning of the value output data vector
   */
  template <typename KeyIter, typename ValueIter>
  void build_output_token(const KeyIter key_output_begin, const ValueIter value_output_begin, const size_t size)
  {
    save_output_token(key_output_begin, std::vector<ValueIter>({value_output_begin}), size);
  }

  /*! \brief Retrieves the input token that was generated by the last call to build_input_token().
   *
   * \return The input token as described above, or the empty string if build_input_token hasn't been called yet.
   */
  std::string get_input_token() const
  {
    return m_input_token;
  }

  /*! \brief Retrieves the output token that was generated by the last call to build_output_token().
   *
   * \return The output token as described above, or the empty string if build_output_token hasn't been called yet.
   */
  std::string get_output_token() const
  {
    return m_output_token;
  }

private:
  template <typename KeyIter, typename ValueIter = KeyIter>
  void save_input_token(
    std::string fcn_name,
    KeyIter key_input_begin,
    KeyIter key_input_end,
    std::vector<ValueIter> value_begins,
    std::vector<std::string>&& extra_inputs = {})
  {
    using KeyDataType                     = typename std::iterator_traits<KeyIter>::value_type;
    using ValueDataType                   = typename std::iterator_traits<ValueIter>::value_type;
    const std::string key_data_type       = get_typename_str<KeyDataType>();
    const size_t size                     = key_input_end - key_input_begin;
    const std::string key_input_data_hash = bwr_utils::hash_vector(key_input_begin, key_input_end);

    // Build a vector of tokens to pass to bwr_utils::build_input_token.
    // We will always have the key iterators:
    std::vector<std::string> subtokens = {
      fcn_name,
      bwr_utils::get_iterator_token(key_input_data_hash, key_data_type, 0),
      bwr_utils::get_iterator_token(key_input_data_hash, key_data_type, size)};

    // But we may or may not have a value iterator:
    for (auto value_input_begin : value_begins)
    {
      const std::string value_data_type       = get_typename_str<ValueDataType>();
      const std::string value_input_data_hash = bwr_utils::hash_vector(value_input_begin, value_input_begin + size);
      subtokens.push_back(bwr_utils::get_iterator_token(value_input_data_hash, value_data_type, 0));
    }

    subtokens.insert(
      subtokens.end(), std::make_move_iterator(extra_inputs.begin()), std::make_move_iterator(extra_inputs.end()));

    // Save the resulting compound token so that it can be retrieved later
    m_input_token = bwr_utils::build_input_token(subtokens);
  }

  template <typename KeyIter, typename ValueIter = KeyIter>
  void save_output_token(const KeyIter key_output_begin, std::vector<ValueIter> value_begins, const size_t size)
  {
    using KeyDataType   = typename std::iterator_traits<KeyIter>::value_type;
    using ValueDataType = typename std::iterator_traits<ValueIter>::value_type;

    const std::string key_output_data_hash = bwr_utils::hash_vector(key_output_begin, key_output_begin + size);
    const std::string key_data_type        = bwr_utils::get_typename_str<KeyDataType>();

    // Build a vector of tokens to pass to bwr_utils::build_output_token.
    // We will always have the key iterators:
    std::vector<std::string> subtokens = {
      bwr_utils::get_vector_token(key_output_data_hash, key_data_type),
    };

    // But we may or may not have a value iterator:
    for (auto value_output_begin : value_begins)
    {
      const std::string value_data_type        = bwr_utils::get_typename_str<ValueDataType>();
      const std::string value_output_data_hash = bwr_utils::hash_vector(value_output_begin, value_output_begin + size);
      subtokens.push_back(bwr_utils::get_vector_token(value_output_data_hash, value_data_type));
    }

    // Save the resulting compound token so that it can be retrieved later
    m_output_token = bwr_utils::build_output_token(subtokens);
  }

  std::string m_input_token;
  std::string m_output_token;
};

} // end namespace bwr_utils

#endif // BRW_UTILS_HPP
