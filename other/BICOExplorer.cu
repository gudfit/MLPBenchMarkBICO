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

void print_progress_bar(double current_latency, double best_latency) {
  std::cout << " Perf: [";
  int bar_width = 50;
  if (current_latency < best_latency) {
    std::cout << "\033[1;32m";
    for (int i = 0; i < bar_width; ++i)
      std::cout << "*";
    std::cout << "\033[0m] NEW BEST!";
  } else {
    std::cout << "\033[1;31m";
    bar_width = static_cast<int>(50.0 * best_latency / current_latency);
    for (int i = 0; i < bar_width; ++i)
      std::cout << "|";
    for (int i = bar_width; i < 50; ++i)
      std::cout << " ";
    std::cout << "\033[0m]";
  }
  std::cout << std::endl;
}

void BICOExplorer::explore(int max_budget) {
  const long long M = 1024, N = 12288, K = 4096;
  std::cout << "Starting " << explorer_name_
            << " exploration with budget: " << max_budget << std::endl;
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
  printf("|===================================================================="
         "=======================|\n");
  printf("| BICO EXPLORATION LOG (%-12s) |\n", explorer_name_.c_str());
  printf("|===================================================================="
         "=======================|\n");
  printf("| Budget(n) | Latency (ms) | TFLOP/s | Best Latency | Sink Size | "
         "Configuration Tested        |\n");
  printf("|-----------|--------------|---------|--------------|-----------|----"
         "------------------------|\n");
  for (int n = 1; n <= budget; ++n) {
    KernelConfig current_config = exploration_space_[n - 1];
    double current_latency = evaluator_.evaluate(current_config);
    double tflops = (2.0 * M * N * K) / (current_latency * 1e-3) / 1e12;
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
    printf("| %-9d | %-12.4f | %-7.2f | %-12.4f | %-9zu | %-26s |\n", n,
           current_latency, tflops, best_latency_, information_sink_.size(),
           current_config.toString().c_str());
    if (is_new_best) {
      size_t smem =
          (static_cast<size_t>(current_config.TILE_M) * current_config.TILE_K +
           static_cast<size_t>(current_config.TILE_K) * current_config.TILE_N) *
          4;
      printf("| %-9s | ==> NEW BEST! %.2f TFLOP/s, %zu bytes SMEM. \033[0m|\n",
             "", tflops, smem);
    }
    print_progress_bar(current_latency, best_latency_);
  }
  printf("|===================================================================="
         "=======================|\n\n");
  double best_tflops = (2.0 * M * N * K) / (best_latency_ * 1e-3) / 1e12;
  std::cout << "===== " << explorer_name_ << " Exploration Finished =====\n"
            << "Optimal configuration: " << best_config_.toString() << "\n"
            << "Latency: " << best_latency_ << " ms\n"
            << "Performance: " << std::fixed << std::setprecision(2)
            << best_tflops << " TFLOP/s\n";
}
