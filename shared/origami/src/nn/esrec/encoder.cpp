// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/esrec/encoder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace origami::nn::esrec {
namespace {

float dot(const float* a, const float* b, std::size_t n) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

void apply_scaler(const detail::EncoderModel& encoder, std::vector<float>* features) {
  assert(features->size() == encoder.scaler_mean.size());
  assert(features->size() == encoder.scaler_scale.size());
  for (std::size_t i = 0; i < features->size(); ++i) {
    const float scale = encoder.scaler_scale[i] != 0.0f ? encoder.scaler_scale[i] : 1.0f;
    (*features)[i]    = ((*features)[i] - encoder.scaler_mean[i]) / scale;
  }
}

void dense_relu(const std::vector<float>& input,
                const std::vector<float>& weights,
                const std::vector<float>& bias,
                std::vector<float>* output) {
  const std::size_t input_dim  = input.size();
  const std::size_t output_dim = bias.size();
  output->assign(output_dim, 0.0f);
  for (std::size_t j = 0; j < output_dim; ++j) {
    float acc = bias[j] + dot(weights.data() + j * input_dim, input.data(), input_dim);
    (*output)[j] = acc > 0.0f ? acc : 0.0f;
  }
}

void dense_linear(const std::vector<float>& input,
                  const std::vector<float>& weights,
                  const std::vector<float>& bias,
                  std::vector<float>* output) {
  const std::size_t input_dim  = input.size();
  const std::size_t output_dim = bias.size();
  output->assign(output_dim, 0.0f);
  for (std::size_t j = 0; j < output_dim; ++j) {
    (*output)[j] = bias[j] + dot(weights.data() + j * input_dim, input.data(), input_dim);
  }
}

}  // namespace

std::vector<float> encode_query(const detail::EncoderModel& encoder,
                                const std::vector<float>& features) {
  std::vector<float> hidden = features;
  apply_scaler(encoder, &hidden);

  std::vector<float> layer_out;
  for (std::size_t i = 0; i < encoder.weights.size(); ++i) {
    dense_relu(hidden, encoder.weights[i], encoder.bias[i], &layer_out);
    hidden.swap(layer_out);
  }

  std::vector<float> embedding;
  dense_linear(hidden, encoder.proj_weights, encoder.proj_bias, &embedding);
  return embedding;
}

}  // namespace origami::nn::esrec
