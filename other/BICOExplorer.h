#pragma once
#include "Evaluator.h"
#include "KernelConfig.h"
#include <vector>
class BICOExplorer {
private:
  std::vector<KernelConfig> exploration_space_;
  double best_latency_;
  KernelConfig best_config_;
  std::vector<KernelConfig> information_sink_;
  Evaluator evaluator_;

public:
  BICOExplorer(std::vector<KernelConfig> search_space, Evaluator evaluator);
  void explore(int max_budget);
};
