#pragma once
#include "Evaluator.h"
#include "KernelConfig.h"
#include <string>
#include <vector>
class BICOExplorer {
private:
  std::vector<KernelConfig> exploration_space_;
  double best_latency_;
  KernelConfig best_config_;
  std::vector<KernelConfig> information_sink_;
  Evaluator evaluator_;
  std::string explorer_name_;
  bool shuffle_;

public:
  BICOExplorer(std::string name, std::vector<KernelConfig> search_space,
               Evaluator evaluator, bool shuffle = true);
  void explore(int max_budget);
};
