
#include "header.h"


static const int32_t num_class[] = {  1, };

int32_t get_num_target(void) {
  return N_TARGET;
}
void get_num_class(int32_t* out) {
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t get_num_feature(void) {
  return 52;
}
const char* get_threshold_type(void) {
  return "float64";
}
const char* get_leaf_output_type(void) {
  return "float64";
}

void predict(union Entry* data, int pred_margin, double* result) {
  unsigned int tmp;
  predict_unit0(data, result);
  predict_unit1(data, result);
  predict_unit2(data, result);
  predict_unit3(data, result);
  
  // Apply base_scores
  result[0] += 0;
  
  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void postprocess(double* result) {
  // Do nothing
}

