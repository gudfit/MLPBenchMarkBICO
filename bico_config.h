#pragma once

#include <chrono>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <vector>

struct SearchConfig {
  double search_time_budget;
  double reliability_threshold;

  SearchConfig(double time = 0, double reliability = 0.95)
      : search_time_budget(time), reliability_threshold(reliability) {}

  bool operator<=(const SearchConfig &other) const {
    return search_time_budget <= other.search_time_budget;
  }

  friend std::ostream &operator<<(std::ostream &os,
                                  const SearchConfig &config) {
    os << "(" << config.search_time_budget
       << "s, p=" << config.reliability_threshold << ")";
    return os;
  }
};

struct KernelResult {
  SearchConfig config;
  float latency_ms;
  bool valid;

  KernelResult(SearchConfig cfg = SearchConfig(), float lat = 0, bool v = true)
      : config(cfg), latency_ms(lat), valid(v) {}
};

using Evaluator = std::function<KernelResult(const SearchConfig &)>;

class BICOExplorer {
private:
  std::vector<KernelResult> results;
  Evaluator evaluator;

public:
  BICOExplorer(Evaluator eval) : evaluator(eval) {}
  void exploreWithTimeBudget(double max_time_seconds = 10.0);
  const std::vector<KernelResult> &getResults() const { return results; }
  KernelResult findBestConfig() const;
};
