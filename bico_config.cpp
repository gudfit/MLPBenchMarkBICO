#include "bico_config.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

void BICOExplorer::exploreWithTimeBudget(double max_time_seconds) {
  results.clear();

  auto start_time = std::chrono::high_resolution_clock::now();
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.1f, 0.9f);

  std::vector<double> time_budgets = {0.1, 0.5, 1.0,
                                      2.0, 5.0, max_time_seconds};

  for (double time_budget : time_budgets) {
    SearchConfig config(time_budget);
    KernelResult result = evaluator(config);
    results.push_back(result);

    auto current_time = std::chrono::high_resolution_clock::now();
    double elapsed =
        std::chrono::duration<double>(current_time - start_time).count();

    if (elapsed >= max_time_seconds)
      break;

    if (result.valid) {
      std::cout << "Time budget " << time_budget << "s -> " << result.latency_ms
                << " ms" << std::endl;
    } else {
      std::cout << "Time budget " << time_budget << "s -> Invalid" << std::endl;
    }
  }
}

KernelResult BICOExplorer::findBestConfig() const {
  if (results.empty())
    return KernelResult();

  float best_latency = INFINITY;
  KernelResult best_result;

  for (const auto &result : results) {
    if (result.valid && result.latency_ms < best_latency) {
      best_latency = result.latency_ms;
      best_result = result;
    }
  }

  return best_result;
}
