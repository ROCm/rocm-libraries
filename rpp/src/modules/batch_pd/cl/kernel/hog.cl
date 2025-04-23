__kernel void hog_primary_pkd(__global unsigned char *input,
                              __global int *magnitude, __global int *phase,
                              const unsigned int height,
                              const unsigned int width,
                              const unsigned int channel) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= width || id_y >= height || id_z >= channel) {
    return;
  }

  unsigned int pixId = id_y * channel * width + id_x * channel + id_z;

  float Gx = 0.0;
  float Gy = 0.0;
  Gx = ((id_x == 0) ? 0 : input[pixId - channel]) * -1 +
       ((id_x == width - 1) ? 0 : input[pixId + channel]);
  Gy = ((id_y == 0) ? 0 : input[pixId - channel * width]) * -1 +
       ((id_y == height - 1) ? 0 : input[pixId + channel * width]);

  magnitude[pixId] = (int)sqrt((Gx * Gx) + (Gy * Gy));
  if (Gy < 0 && Gx == 0)
    phase[pixId] = 270;
  else if (Gy > 0 && Gx == 0)
    phase[pixId] = 90;
  else if (Gy == 0 && Gx == 0)
    phase[pixId] = 0;
  else {
    phase[pixId] = (int)((atan(Gy / Gx) * 180) / 3.14);
    phase[pixId] = (phase[pixId] < 0) ? 360 + phase[pixId] : phase[pixId];
  }
}

__kernel void hog_primary_pln(__global unsigned char *input,
                              __global int *magnitude, __global int *phase,
                              const unsigned int height,
                              const unsigned int width,
                              const unsigned int channel) {
  int id_x = get_global_id(0);
  int id_y = get_global_id(1);
  int id_z = get_global_id(2);

  if (id_x >= width || id_y >= height || id_z >= channel) {
    return;
  }

  unsigned int pixId = id_z * height * width + id_y * width + id_x;

  float Gx = 0.0;
  float Gy = 0.0;
  Gx = ((id_x == 0) ? 0 : input[pixId - 1]) * -1 +
       ((id_x == width - 1) ? 0 : input[pixId + 1]);
  Gy = ((id_y == 0) ? 0 : input[pixId - width]) * -1 +
       ((id_y == height - 1) ? 0 : input[pixId + width]);

  magnitude[pixId] = (int)sqrt((Gx * Gx) + (Gy * Gy));
  if (Gy < 0 && Gx == 0)
    phase[pixId] = 270;
  else if (Gy > 0 && Gx == 0)
    phase[pixId] = 90;
  else if (Gy == 0 && Gx == 0)
    phase[pixId] = 0;
  else {
    phase[pixId] = (int)((atan(Gy / Gx) * 180) / 3.14);
    phase[pixId] = (phase[pixId] < 0) ? 360 + phase[pixId] : phase[pixId];
  }
}

__kernel void hog_pkd(__global int *magnitude, __global int *phase,
                      __global unsigned int *output, const unsigned int height,
                      const unsigned int width, const unsigned int channel,
                      const unsigned int kHeight, const unsigned int kWidth,
                      const unsigned int bins, const unsigned int binsRow,
                      const unsigned int binsCol) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int c = get_global_id(2);

  if (x >= binsCol || y >= binsRow || c >= channel) {
    return;
  }

  unsigned int binId =
      y * bins * channel * binsCol + x * bins * channel + c * bins;

  int id_x, id_y, id_z;

  id_x = x * kWidth;
  id_y = y * kHeight;
  id_z = c;

  unsigned int pixId = id_y * channel * width + id_x * channel + id_z;

  for (int i = 0; i < kHeight; i++) {
    for (int j = 0; j < kWidth; j++) {
      if ((id_y + i) >= height && (id_x + j) >= width)
        continue;

      unsigned int index =
          ((id_y + i) * width * channel) + ((id_x + j) * channel) + id_z;

      float interBinIndex = ((phase[index] * (bins - 1)) / 360);

      unsigned int binIndex;
      binIndex = (unsigned int)ceil(interBinIndex);
      binIndex += binId;

      int a = magnitude[index];
      atomic_add(&output[binIndex], a);
    }
  }
}

__kernel void hog_pln(__global int *magnitude, __global int *phase,
                      __global unsigned int *output, const unsigned int height,
                      const unsigned int width, const unsigned int channel,
                      const unsigned int kHeight, const unsigned int kWidth,
                      const unsigned int bins, const unsigned int binsRow,
                      const unsigned int binsCol) {
  int x = get_global_id(0);
  int y = get_global_id(1);
  int c = get_global_id(2);

  if (x >= binsCol || y >= binsRow || c >= channel) {
    return;
  }

  unsigned int binId = y * bins * binsCol + x * bins + c * binsRow * binsCol;

  int id_x, id_y, id_z;

  id_x = x * kWidth;
  id_y = y * kHeight;
  id_z = c;

  unsigned int pixId = id_z * height * width + id_y * width + id_x;

  for (int i = 0; i < kHeight; i++) {
    for (int j = 0; j < kWidth; j++) {
      if ((id_y + i) >= height && (id_x + j) >= width)
        continue;

      unsigned int index =
          ((id_y + i) * width) + (id_x + j) + id_z * height * width;

      float interBinIndex = ((phase[index] * (bins - 1)) / 360);

      unsigned int binIndex;
      binIndex = (unsigned int)ceil(interBinIndex);
      binIndex += binId;

      int a = magnitude[index];
      atomic_add(&output[binIndex], a);
    }
  }
}