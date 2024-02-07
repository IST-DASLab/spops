#pragma once

#include "utility.cuh"

#include <map>
#include <vector>
#include <chrono>
#include <string>

struct Result {
  float mean_{0.0f};
  float std_dev_{0.0f};
  float median_{0.0f};
  float min_{0.0f};
  float max_{0.0f};
  int num_{0};
};

struct Timer {
  enum class Type { GPU, CPU };
  using Time = std::chrono::time_point<std::chrono::steady_clock>;

  cudaEvent_t ce_start{}, ce_stop{};
  Time cpu_ce_start{}, cpu_ce_stop{};
  std::vector<float> measurements_;
  std::string name;
  Type type;

  Time now() { return std::chrono::steady_clock::now(); }

  void start() {
    if (type == Type::GPU) {
      start_clock(ce_start);
    } else {
      cpu_ce_start = now();
    }
  }

  float end(bool add = true) {
    if (type == Type::GPU) {
      auto timing = end_clock(ce_start, ce_stop);
      if (add)
        measurements_.push_back(timing);
      return timing;
    } else {
      cpu_ce_stop = now();
      auto run_time =
          std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_ce_stop - cpu_ce_start).count();
      if (add)
        measurements_.push_back(run_time / 1000.f);
      return run_time / 1000.f;
    }
  }

  void finalize_benchmark();

  Timer(int runs, std::string name, Type type) : name(std::move(name)), type(type) {
    measurements_.reserve(runs);
    if (type == Type::GPU) {
      HANDLE_ERROR(cudaEventCreate(&ce_start));
      HANDLE_ERROR(cudaEventCreate(&ce_stop));
    }
  }

  Timer(Timer &&timer) = delete;
  Timer(const Timer &timer) = delete;

  ~Timer() {
    if (type == Type::GPU) {
      HANDLE_ERROR(cudaEventDestroy(ce_start));
      HANDLE_ERROR(cudaEventDestroy(ce_stop));
    }
  }
  void addMeasure(Timer &measure) {
    measurements_.insert(measurements_.end(), measure.measurements_.begin(), measure.measurements_.end());
  }

  float mean();

  float median();

  float std_dev(float mean);

  Result generateResult();
};

struct Timers {
  std::map<std::string, std::vector<Timer *>> timers;

  void add_gpu_timer(const std::string &parent_name, int runs, const std::string &name);

  void add_cpu_timer(const std::string &parent_name, int runs, const std::string &name);

  void finalize_benchmark();

  std::vector<std::string> method_names();

  void plot(const std::string &bench_path);
};