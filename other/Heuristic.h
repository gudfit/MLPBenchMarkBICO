#pragma once
#include "KernelConfig.h"
#include <algorithm>
#include <vector>

double calculate_reuse_score(const KernelConfig &c) {
  if (c.TILE_M <= 0 || c.TILE_N <= 0 || c.TILE_K <= 0 || c.BLOCK_DIM_X <= 0 ||
      c.BLOCK_DIM_Y <= 0) {
    return 0.0;
  }
  double ops = 2.0 * c.TILE_M * c.TILE_N * c.TILE_K;
  double bytes = 4.0 * (c.TILE_M * c.TILE_K + c.TILE_K * c.TILE_N);
  return bytes == 0.0 ? 0.0 : ops / bytes;
}

std::vector<KernelConfig> predictive_search(std::vector<KernelConfig> space) {
  std::sort(space.begin(), space.end(),
            [](const KernelConfig &a, const KernelConfig &b) {
              return calculate_reuse_score(a) > calculate_reuse_score(b);
            });
  return space;
}
