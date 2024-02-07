#include "utility.cuh"

#include <string>
#include <stdexcept>

void HandleError(cudaError_t err, const char *string, const char *file, int line) {
  if (err != cudaSuccess) {
    //printf("%s\n", string);
    //printf("%s in \n\n%s at line %d\n", cudaGetErrorString(err), file, line);
    throw std::runtime_error(
        std::string("CUDA Error ") + cudaGetErrorString(err) + " " + string + " in " + file + " at line "
            + std::to_string(line));
  }
}
void HandleError(const char *file, int line) {
  auto err = cudaGetLastError();
  if (err != cudaSuccess) {
    //printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    throw std::runtime_error(
        std::string("CUDA Error ") + cudaGetErrorString(err) + " in " + file + " at line " + std::to_string(line));
  }
}
float end_clock(cudaEvent_t &start, cudaEvent_t &end) {
  float time;
  HANDLE_ERROR(cudaEventRecord(end, 0));
  HANDLE_ERROR(cudaEventSynchronize(end));
  HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));

  // Returns ms
  return time;
}

void start_clock(cudaEvent_t &start) {
  HANDLE_ERROR(cudaEventRecord(start, 0));
}
