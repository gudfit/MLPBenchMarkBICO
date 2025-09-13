#pragma once
#include <sstream>
#include <string>

struct KernelConfig {
  int TILE_M = 0;
  int TILE_N = 0;
  int TILE_K = 0;
  int BLOCK_DIM_X = 0;
  int BLOCK_DIM_Y = 0;

  std::string toString() const {
    std::stringstream ss;
    ss << "TM=" << TILE_M << ", TN=" << TILE_N << ", TK=" << TILE_K
       << ", BX=" << BLOCK_DIM_X << ", BY=" << BLOCK_DIM_Y;
    return ss.str();
  }

  bool isValid() const {
    return TILE_M > 0 && TILE_N > 0 && TILE_K > 0 && BLOCK_DIM_X > 0 &&
           BLOCK_DIM_Y > 0 && BLOCK_DIM_X * BLOCK_DIM_Y <= 1024;
  }
};
