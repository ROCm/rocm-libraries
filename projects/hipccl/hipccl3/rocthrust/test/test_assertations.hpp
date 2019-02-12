// Google Test
#include <gtest/gtest.h>

//#include <thrust/complex.h>

template <typename T1, typename T2>
testing::AssertionResult CmpHelperEQQuite(const char* lhs_expression,
                                          const char* rhs_expression,
                                          const T1& lhs,
                                          const T2& rhs) {
  if (lhs == rhs) {
    return testing::AssertionSuccess();
  }

  testing::Message msg;
  msg << "Expressions during equality check:";
  msg << "\n  " << lhs_expression;
  msg << "\n  " << rhs_expression;

  return testing::AssertionFailure() << msg;
}

#define ASSERT_EQ_QUIET(val1, val2)\
  ASSERT_PRED_FORMAT2(CmpHelperEQQuite, \
                      val1, val2)


/*template <typename T>
testing::AssertionResult ComplexNearPredFormat(const char* expr1,
                                               const char* expr2,
                                               const char* abs_error_expr,
                                               thrust::complex<T> val1,
                                               thrust::complex<T> val2,
                                               thrust::complex<double> abs_error){
  const thrust::complex<double> diff(fabs((double)val1.real() - (double)val2.real())
                                    ,fabs((double)val1.imag() - (double)val2.imag()));
  if (diff.real() <= abs_error.real() &&
      diff.imag() <= abs_error.imag())
    return testing::AssertionSuccess();

  return testing::AssertionFailure()
      << "The difference between " << expr1 << " and " << expr2
      << " is " << diff << ", which exceeds " << abs_error_expr << ", where\n"
      << expr1 << " evaluates to " << val1 << ",\n"
      << expr2 << " evaluates to " << val2 << ", and\n"
      << abs_error_expr << " evaluates to " << abs_error << ".";
}

#define ASSERT_NEAR_COMPLEX_ERROR(val1, val2, abs_error)\
  ASSERT_PRED_FORMAT3(ComplexNearPredFormat<typename decltype(val1)::value_type>, \
                      val1, val2, abs_error)


#define ASSERT_NEAR_COMPLEX(val1, val2)\
  ASSERT_NEAR_COMPLEX_ERROR(val1, val2, thrust::complex<T>(std::numeric_limits<T>::epsilon(),std::numeric_limits<T>::epsilon()))
*/
