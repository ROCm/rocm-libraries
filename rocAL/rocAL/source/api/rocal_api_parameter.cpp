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

#include "pipeline/commons.h"
#include "parameters/parameter_factory.h"
#include "rocal_api.h"

void ROCAL_API_CALL
rocalSetSeed(unsigned seed) {
    ParameterFactory::instance()->set_seed(seed);
}

unsigned ROCAL_API_CALL
rocalGetSeed() {
    return ParameterFactory::instance()->get_seed();
}

int ROCAL_API_CALL
rocalGetIntValue(RocalIntParam p_obj) {
    auto obj = static_cast<IntParam *>(p_obj);
    return obj->core->get();
}

float ROCAL_API_CALL
rocalGetFloatValue(RocalFloatParam p_obj) {
    auto obj = static_cast<FloatParam *>(p_obj);
    return obj->core->get();
}

RocalIntParam ROCAL_API_CALL
rocalCreateIntUniformRand(
    int start,
    int end) {
    return ParameterFactory::instance()->create_uniform_int_rand_param(start, end);
}

RocalStatus ROCAL_API_CALL
rocalUpdateIntUniformRand(
    int start,
    int end,
    RocalIntParam p_input_obj) {
    auto input_obj = static_cast<IntParam *>(p_input_obj);
    if (!validate_uniform_rand_param(input_obj)) {
        ERR("rocalUpdateIntUniformRand : not a UniformRand object!");
        return ROCAL_INVALID_PARAMETER_TYPE;
    }

    UniformRand<int> *obj;
    if ((obj = dynamic_cast<UniformRand<int> *>(input_obj->core)) == nullptr)
        return ROCAL_INVALID_PARAMETER_TYPE;

    return (obj->update(start, end) == 0) ? ROCAL_OK : ROCAL_UPDATE_PARAMETER_FAILED;
}

RocalFloatParam ROCAL_API_CALL
rocalCreateFloatUniformRand(
    float start,
    float end) {
    return ParameterFactory::instance()->create_uniform_float_rand_param(start, end);
}

RocalStatus ROCAL_API_CALL
rocalUpdateFloatUniformRand(
    float start,
    float end,
    RocalFloatParam p_input_obj) {
    auto input_obj = static_cast<FloatParam *>(p_input_obj);
    if (!validate_uniform_rand_param(input_obj)) {
        ERR("rocalUpdateFloatUniformRand : not a uniform random object!");
        return ROCAL_INVALID_PARAMETER_TYPE;
    }

    UniformRand<float> *obj;
    if ((obj = dynamic_cast<UniformRand<float> *>(input_obj->core)) == nullptr)
        return ROCAL_INVALID_PARAMETER_TYPE;

    return (obj->update(start, end) == 0) ? ROCAL_OK : ROCAL_UPDATE_PARAMETER_FAILED;
}

RocalFloatParam ROCAL_API_CALL
rocalCreateFloatRand(
    const float *values,
    const double *frequencies,
    unsigned size) {
    return ParameterFactory::instance()->create_custom_float_rand_param(values,
                                                                        frequencies,
                                                                        size);
}

RocalFloatParam ROCAL_API_CALL
rocalCreateFloatParameter(float val) {
    return ParameterFactory::instance()->create_single_value_float_param(val);
}

RocalIntParam ROCAL_API_CALL
rocalCreateIntParameter(int val) {
    return ParameterFactory::instance()->create_single_value_int_param(val);
}

RocalStatus ROCAL_API_CALL
rocalUpdateIntParameter(int new_val, RocalIntParam p_input_obj) {
    auto input_obj = static_cast<IntParam *>(p_input_obj);
    if (!validate_simple_rand_param(input_obj)) {
        ERR("rocalUpdateIntParameter : not a custom random object!");
        return ROCAL_INVALID_PARAMETER_TYPE;
    }

    SimpleParameter<int> *obj;
    if ((obj = dynamic_cast<SimpleParameter<int> *>(input_obj->core)) == nullptr)
        return ROCAL_INVALID_PARAMETER_TYPE;
    return (obj->update(new_val) == 0) ? ROCAL_OK : ROCAL_UPDATE_PARAMETER_FAILED;
}

RocalStatus ROCAL_API_CALL
rocalUpdateFloatParameter(float new_val, RocalFloatParam p_input_obj) {
    auto input_obj = static_cast<FloatParam *>(p_input_obj);
    if (!validate_simple_rand_param(input_obj)) {
        ERR("rocalUpdateFloatParameter : not a custom random object!");
        return ROCAL_INVALID_PARAMETER_TYPE;
    }

    SimpleParameter<float> *obj;
    if ((obj = dynamic_cast<SimpleParameter<float> *>(input_obj->core)) == nullptr)
        return ROCAL_INVALID_PARAMETER_TYPE;
    return (obj->update(new_val) == 0) ? ROCAL_OK : ROCAL_UPDATE_PARAMETER_FAILED;
}

RocalStatus ROCAL_API_CALL
rocalUpdateFloatRand(
    const float *values,
    const double *frequencies,
    unsigned size,
    RocalFloatParam p_updating_obj) {
    auto updating_obj = static_cast<FloatParam *>(p_updating_obj);
    if (!validate_custom_rand_param(updating_obj)) {
        ERR("rocalUpdateFloatRand : not a custom random object!");
        return ROCAL_INVALID_PARAMETER_TYPE;
    }

    CustomRand<float> *obj;
    if ((obj = dynamic_cast<CustomRand<float> *>(updating_obj->core)) == nullptr)
        return ROCAL_INVALID_PARAMETER_TYPE;

    return (obj->update(values, frequencies, size) == 0) ? ROCAL_OK : ROCAL_UPDATE_PARAMETER_FAILED;
}

RocalIntParam ROCAL_API_CALL
rocalCreateIntRand(
    const int *values,
    const double *frequencies,
    unsigned size) {
    return ParameterFactory::instance()->create_custom_int_rand_param(values,
                                                                      frequencies,
                                                                      size);
}

RocalStatus ROCAL_API_CALL
rocalUpdateIntRand(
    const int *values,
    const double *frequencies,
    unsigned size,
    RocalIntParam p_updating_obj) {
    auto updating_obj = static_cast<IntParam *>(p_updating_obj);
    if (!validate_custom_rand_param(updating_obj)) {
        ERR("rocalUpdateIntRand : not a CustomRand object!");
        return ROCAL_INVALID_PARAMETER_TYPE;
    }

    CustomRand<int> *obj;
    if ((obj = dynamic_cast<CustomRand<int> *>(updating_obj->core)) == nullptr)
        return ROCAL_INVALID_PARAMETER_TYPE;

    return (obj->update(values, frequencies, size) == 0) ? ROCAL_OK : ROCAL_UPDATE_PARAMETER_FAILED;
}
