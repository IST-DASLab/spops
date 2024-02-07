#include "gpu_timer.cuh"

#include <iostream>
#include <algorithm>
#include <fstream>
#include <numeric>

float Timer::mean() {
  return std::accumulate(measurements_.begin(), measurements_.end(), 0.0f) / static_cast<float>(measurements_.size());
}
float Timer::std_dev(float mean) {
  std::vector<float> stdmean_measurements(measurements_);
  for (auto &elem : stdmean_measurements)
    elem = (elem - mean) * (elem - mean);
  return sqrt(std::accumulate(stdmean_measurements.begin(), stdmean_measurements.end(), 0.0f)
                  / static_cast<float>(stdmean_measurements.size()));
}
Result Timer::generateResult() {
  if (measurements_.size() == 0)
    return Result{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0};
  float mean_val = mean();
  return Result{
      mean_val,
      std_dev(mean_val),
      median(),
      *std::min_element(measurements_.begin(), measurements_.end()),
      *std::max_element(measurements_.begin(), measurements_.end()),
      static_cast<int>(measurements_.size())
  };
}

void Timer::finalize_benchmark() {
  auto gpures = generateResult();
  std::cout << name << " required " << gpures.mean_ << "ms on average with std " << gpures.std_dev_ << "ms on the "
            << (type == Type::GPU ? "GPU" : "CPU")
            << std::endl;
}
float Timer::median() {
  std::vector<float> sorted_measurements(measurements_);
  std::sort(sorted_measurements.begin(), sorted_measurements.end());
  return sorted_measurements[sorted_measurements.size() / 2];
}

void Timers::plot(const std::string &bench_path) {
  std::ofstream bench_table(bench_path);

  std::vector<std::string> input_names;

  int num_inputs = timers.begin()->second.size();

  std::vector<std::vector<double>> all_times;
  std::vector<std::vector<std::string>> all_names;

  bench_table << "test_name";
  for (const auto &n : method_names()) {
    bench_table << " " << n;
  }

  bench_table << std::endl;

  auto timer_iter = timers.begin();

  for (int i = 0; i < num_inputs; i++) {
    std::string input_name;
    for (const auto &t : timers) {
      input_name = t.second[i]->name;
      break;
    }
    bench_table << input_name;
    for (const auto &t : timers) {
      bench_table << " " << t.second[i]->measurements_.back();
    }
    bench_table << std::endl;
    timer_iter++;
  }
}
void Timers::add_gpu_timer(const std::string &parent_name, int runs, const std::string &name) {
  timers[parent_name].push_back(new Timer(runs, name, Timer::Type::GPU));
}
void Timers::finalize_benchmark() {
  for (auto &[key, value] : timers) {
    for (auto &timer : value) {
      timer->finalize_benchmark();
    }
  }
}
std::vector<std::string> Timers::method_names() {
  std::vector<std::string> methods;
  for (const auto &timer : timers) {
    methods.push_back(timer.first);
  }
  return methods;
}
void Timers::add_cpu_timer(const std::string &parent_name, int runs, const std::string &name) {
  timers[parent_name].push_back(new Timer(runs, name, Timer::Type::CPU));
}
