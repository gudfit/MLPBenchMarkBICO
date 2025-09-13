#pragma once

#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <vector>

struct BudgetConfig {
  float smem_budget;
  float occ_budget;

  BudgetConfig(float smem = 0, float occ = 0)
      : smem_budget(smem), occ_budget(occ) {}

  // Comparison operators
  bool operator<=(const BudgetConfig &other) const {
    return smem_budget <= other.smem_budget && occ_budget <= other.occ_budget;
  }

  bool operator==(const BudgetConfig &other) const {
    return smem_budget == other.smem_budget && occ_budget == other.occ_budget;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const BudgetConfig &config) {
    os << "(" << config.smem_budget << " KB, " << config.occ_budget * 100
       << "%)";
    return os;
  }
};

struct KernelResult {
  BudgetConfig config;
  float latency_ms;
  bool valid;

  KernelResult(BudgetConfig cfg = BudgetConfig(), float lat = 0, bool v = true)
      : config(cfg), latency_ms(lat), valid(v) {}
};

using Evaluator = std::function<KernelResult(const BudgetConfig &)>;

class BICOExplorer {
private:
  std::vector<BudgetConfig> frontier;
  std::vector<KernelResult> results;
  Evaluator evaluator;

public:
  BICOExplorer(Evaluator eval) : evaluator(eval) {}
  void exploreFrontier(int max_iterations = 20);
  const std::vector<BudgetConfig> &getFrontier() const { return frontier; }
  const std::vector<KernelResult> &getResults() const { return results; }
  BudgetConfig findBestConfig() const;
};
