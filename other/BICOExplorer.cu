#include "BICOExplorer.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

BICOExplorer::BICOExplorer(std::vector<KernelConfig> search_space,
                           Evaluator evaluator)
    : exploration_space_(std::move(search_space)),
      best_latency_(std::numeric_limits<double>::max()),
      evaluator_(std::move(evaluator)) {
  best_config_ = KernelConfig();
}

void BICOExplorer::explore(int max_budget) {
  std::cout << "Starting BICO exploration with budget: " << max_budget
            << std::endl;
  std::cout << "Exploration space size: " << exploration_space_.size()
            << std::endl;
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(exploration_space_.begin(), exploration_space_.end(), g);
  int budget =
      std::min(static_cast<int>(exploration_space_.size()), max_budget);
  bool first_run = true;
  for (int n = 1; n <= budget; ++n) {
    KernelConfig current_config = exploration_space_[n - 1];
    double current_latency = evaluator_.evaluate(current_config);
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
    std::cout << "--- Budget n = " << n << " ---\n"
              << " Tested: " << current_config.toString() << " -> "
              << current_latency << " ms\n"
              << " Best Latency Guaranteed (x_n*): " << best_latency_ << " ms\n"
              << " Sink Size: " << information_sink_.size() << "\n";
    if (is_new_best) {
      std::cout << " ** New best configuration found! **\n";
    }
  }
  std::cout << "\n===== Exploration Finished =====\n"
            << "Optimal configuration found: " << best_config_.toString()
            << "\n"
            << "With guaranteed latency: " << best_latency_ << " ms\n";
}
