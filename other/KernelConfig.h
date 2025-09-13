#pragma once
#include <sstream>
#include <string>
struct KernelConfig {
  int TILE_M = -1;
  int TILE_N = -1;
  int TILE_K = -1;
  int BLOCK_DIM_X = -1;
  int BLOCK_DIM_Y = -1;
  std::string toString() const {
    std::stringstream ss;
    ss << "TM=" << TILE_M << ", TN=" << TILE_N << ", TK=" << TILE_K
       << ", BX=" << BLOCK_DIM_X << ", BY=" << BLOCK_DIM_Y;
    return ss.str();
  }
};
