#include "BICOExplorer.h"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
BICOExplorer::BICOExplorer(std::string name,
                           std::vector<KernelConfig> search_space,
                           Evaluator evaluator, bool shuffle)
    : exploration_space_(std::move(search_space)),
      best_latency_(std::numeric_limits<double>::max()), best_config_(),
      evaluator_(std::move(evaluator)), explorer_name_(std::move(name)),
      shuffle_(shuffle) {}
void BICOExplorer::explore(int max_budget) {
  std::cout << "Starting BICO exploration with budget: " << max_budget
            << std::endl;
  std::cout << "Exploration space size: " << exploration_space_.size()
            << std::endl;
  if (shuffle_) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(exploration_space_.begin(), exploration_space_.end(), g);
  }
  int budget =
      std::min(static_cast<int>(exploration_space_.size()), max_budget);
  bool first_run = true;
  printf("\n");
  printf(" |==================================================================="
         "===============================|\n");
  printf(" | BICO EXPLORATION LOG (%-12s) |\n", explorer_name_.c_str());
  printf(" |==================================================================="
         "===============================|\n");
  printf(" | Budget(n) | Latency (ms) | TFLOP/s | Best Latency | Sink Size | "
         "Configuration Tested |\n");
  printf(" |-----------|--------------|---------|--------------|-----------|---"
         "--------------------------------|\n");
  for (int n = 1; n <= budget; ++n) {
    KernelConfig current_config = exploration_space_[n - 1];
    double current_latency = evaluator_.evaluate(current_config);
    long long MM = 1024, NN = 12288, KK = 4096; 
    double tflops = (2.0 * MM * NN * KK) / (current_latency * 1e-3) / 1e12;
    bool is_new_best = false;
    if (current_latency < best_latency_) {
      if (!first_run)
        information_sink_.push_back(best_config_);
      best_latency_ = current_latency;
      best_config_ = current_config;
      is_new_best = true;
      first_run = false;
    } else {
      information_sink_.push_back(current_config);
    }
    printf(" | %-9d | %-12.4f | %-7.2f | %-12.4f | %-9zu | %-33s |\n", n,
           current_latency, tflops, best_latency_, information_sink_.size(),
           current_config.toString().c_str());
    if (is_new_best) {
      size_t smem =
          (static_cast<size_t>(current_config.TILE_M) * current_config.TILE_K +
           static_cast<size_t>(current_config.TILE_K) * current_config.TILE_N) *
          4;
      printf(" | \033[1;32m%-9s | ==> NEW BEST! Found a configuration with "
             "%.2f TFLOP/s and %zu bytes of SMEM. \033[0m |\n",
             "", tflops, smem);
    }
  }
  printf(" |==================================================================="
         "===============================|\n\n");

  long long MM = 1024, NN = 12288, KK = 4096;
  double best_tflops = (2.0 * MM * NN * KK) / (best_latency_ * 1e-3) / 1e12;

  std::cout << "===== Exploration Finished =====\n"
            << "Optimal configuration found: " << best_config_.toString()
            << "\n"
            << "With guaranteed latency: " << best_latency_ << " ms\n"
            << "Achieving an estimated: " << std::fixed << std::setprecision(2)
            << best_tflops << " TFLOP/s\n";
}
