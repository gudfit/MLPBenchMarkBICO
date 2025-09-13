#include "bico_config.h"
#include <algorithm>
#include <cmath>

void BICOExplorer::exploreFrontier(int max_iterations) {
  results.clear();
  frontier.clear();
  std::vector<BudgetConfig> initial_configs = {
      BudgetConfig(32, 0.25f), BudgetConfig(64, 0.5f), BudgetConfig(96, 0.75f),
      BudgetConfig(48, 0.9f), BudgetConfig(128, 0.3f)};

  for (const auto &config : initial_configs) {
    KernelResult result = evaluator(config);
    results.push_back(result);

    if (result.valid)
      std::cout << "Config " << config << " -> " << result.latency_ms << " ms" << std::endl;
    else
      std::cout << "Config " << config << " -> Invalid" << std::endl;
  }

  for (size_t i = 0; i < results.size(); ++i) {
    if (!results[i].valid)
      continue;

    bool is_pareto = true;
    for (size_t j = 0; j < results.size(); ++j) {
      if (i != j && results[j].valid &&
          results[j].config <= results[i].config &&
          results[j].latency_ms <= results[i].latency_ms) {
        is_pareto = false;
        break;
      }
    }

    if (is_pareto)
      frontier.push_back(results[i].config);
  }

  std::cout << "Pareto frontier found with " << frontier.size()
            << " configurations" << std::endl;
}

BudgetConfig BICOExplorer::findBestConfig() const {
  if (results.empty())
    return BudgetConfig();

  float best_latency = INFINITY;
  BudgetConfig best_config;

  for (const auto &result : results) {
    if (result.valid && result.latency_ms < best_latency) {
      best_latency = result.latency_ms;
      best_config = result.config;
    }
  }

  return best_config;
}
