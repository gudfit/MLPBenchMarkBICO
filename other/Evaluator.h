#pragma once
#include "KernelConfig.h"
#include <cuda_runtime.h>

class Evaluator {
public:
  Evaluator(float *d_A, const float *d_B, float *d_C, int M, int N, int K);
  double evaluate(const KernelConfig &config);

private:
  float *d_A;
  const float *d_B;
  float *d_C;
  int M, N, K;
};
