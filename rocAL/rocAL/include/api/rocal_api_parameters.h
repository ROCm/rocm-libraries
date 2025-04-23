/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_PARAMETERS_H
#define MIVISIONX_ROCAL_API_PARAMETERS_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Parameters
 *
 * \defgroup group_rocal_parameters API: AMD rocAL - Parameter API
 * \brief The AMD rocAL Parameters.
 */

/*! \brief set seed for random number generation
 * \ingroup group_rocal_parameters
 * \param [in] seed seed for the random number generation
 */
extern "C" void ROCAL_API_CALL rocalSetSeed(unsigned seed);

/*! \brief gets the seed value
 * \ingroup group_rocal_parameters
 * \return seed value
 */
extern "C" unsigned ROCAL_API_CALL rocalGetSeed();

/*! \brief Creates a new uniform random integer parameter within a specified range.
 * \ingroup group_rocal_parameters
 * \param start start value of the integer range
 * \param end end value of the integer range
 * \return RocalIntParam representing the uniform random integer parameter.
 */
extern "C" RocalIntParam ROCAL_API_CALL rocalCreateIntUniformRand(int start, int end);

/*! \brief updates uniform random integer parameter within a specified range.
 * \ingroup group_rocal_parameters
 * \param start start value of the integer range
 * \param end start value of the integer range
 * \param input_obj  RocalIntParam to be updated.
 * \return rocal status value
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateIntUniformRand(int start, int end, RocalIntParam updating_obj);

/*! \brief gets the value of a RocalIntParam.
 * \ingroup group_rocal_parameters
 * \param [in] obj The RocalIntParam from which to retrieve the value.
 * \return integer value of the RocalIntParam.
 */
extern "C" int ROCAL_API_CALL rocalGetIntValue(RocalIntParam obj);

/*! \brief gets the value of a RocalFloatParam.
 * \ingroup group_rocal_parameters
 * \param [in] obj The RocalFloatParam from which to retrieve the value.
 * \return float value of the RocalIntParam.
 */
extern "C" float ROCAL_API_CALL rocalGetFloatValue(RocalFloatParam obj);

/*! \brief Creates a new uniform random float parameter within a specified range.
 * \ingroup group_rocal_parameters
 * \param start start value of the float range
 * \param end end value of the float range
 * \return RocalFloatParam representing the uniform random float parameter.
 */
extern "C" RocalFloatParam ROCAL_API_CALL rocalCreateFloatUniformRand(float start, float end);

/*! \brief Creates a new float parameter with a specified value.
 * \ingroup group_rocal_parameters
 * \param [in] val value to create float param
 * \return A new RocalFloatParam representing the float parameter.
 */
extern "C" RocalFloatParam ROCAL_API_CALL rocalCreateFloatParameter(float val);

/*! \brief Creates a new int parameter with a specified value.
 * \ingroup group_rocal_parameters
 * \param [in] val value to create integer param
 * \return A new RocalIntParam representing the integer parameter.
 */
extern "C" RocalIntParam ROCAL_API_CALL rocalCreateIntParameter(int val);

/*! \brief Updates a float parameter with a new value.
 * \ingroup group_rocal_parameters
 * \param[in] new_val The new value to update the float parameter.
 * \param[in] input_obj The RocalFloatParam to be updated.
 * \return RocalStatus value.
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateFloatParameter(float new_val, RocalFloatParam input_obj);

/*! \brief Updates a integer parameter with a new value.
 * \ingroup group_rocal_parameters
 * \param[in] new_val The new value to update the integer parameter.
 * \param[in] input_obj The RocalIntParam to be updated.
 * \return RocalStatus value.
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateIntParameter(int new_val, RocalIntParam input_obj);

/*! \brief updates uniform random float parameter within a specified range.
 * \ingroup group_rocal_parameters
 * \param start start value of the float range
 * \param end start value of the float range
 * \param input_obj  RocalFloatParam to be updated.
 * \return rocal status value
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateFloatUniformRand(float start, float end, RocalFloatParam updating_obj);

/*! \brief Sets the parameters for a new or existing RocalIntRandGen object
 * \ingroup group_rocal_parameters
 * \param [in] values random int values
 * \param [in] frequencies frequencies of the values
 * \param size size of the array
 * \return random int paraeter
 */
extern "C" RocalIntParam ROCAL_API_CALL rocalCreateIntRand(const int *values, const double *frequencies, unsigned size);

/*! \brief update the int random value
 * \ingroup group_rocal_parameters
 * \param [in] values random int values
 * \param [in] frequencies frequencies of the values
 * \param [in] size size of the array
 * \param [in] updating_obj Rocal int Param to update
 * \return rocal status value
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateIntRand(const int *values, const double *frequencies, unsigned size, RocalIntParam updating_obj);

/*! \brief Sets the parameters for a new or existing RocalFloatRandGen object
 * \ingroup group_rocal_parameters
 * \param [in] values random float values
 * \param [in] frequencies frequencies of the values
 * \param size size of the array
 * \return random float parameter
 */
extern "C" RocalFloatParam ROCAL_API_CALL rocalCreateFloatRand(const float *values, const double *frequencies, unsigned size);

/*! \brief update the float random value
 * \ingroup group_rocal_parameters
 * \param [in] values random float values
 * \param [in] frequencies frequencies of the values
 * \param [in] size size of the array
 * \param [in] updating_obj Rocal Float Param to update
 * \return rocal status value
 */
extern "C" RocalStatus ROCAL_API_CALL rocalUpdateFloatRand(const float *values, const double *frequencies, unsigned size, RocalFloatParam updating_obj);

#endif  // MIVISIONX_ROCAL_API_PARAMETERS_H
