#pragma once

// TODO: This is stupid, get rid of this and just use functions instaed of macros.
#include <type_traits>
#include <driver_types.h>

// ##############################################################################################################################################
//
void HandleError(cudaError_t err,
                 const char *string,
                 const char *file,
                 int line);

// ##############################################################################################################################################
//
void HandleError(const char *file,
                 int line);

#define HANDLE_ERROR(err) (HandleError( err, "", __FILE__, __LINE__ ))
#define HANDLE_ERROR_S(err, string) (HandleError( err, string, __FILE__, __LINE__ ))

// ##############################################################################################################################################
//
void start_clock(cudaEvent_t &start);

// ##############################################################################################################################################
//
float end_clock(cudaEvent_t &start, cudaEvent_t &end);

#define CUMALLOC(X, sz) \
  cudaMalloc(reinterpret_cast<void**>(&(X)), sizeof(std::remove_pointer_t<decltype(X)>) * (sz))

#define CUMALLOC0(X, sz) \
  CUMALLOC(X, sz); \
  cudaMemset(X, std::remove_pointer_t<decltype(X)>{}, sizeof(std::remove_pointer_t<decltype(X)>) * (sz))

#define AH2D(To, From, sz) \
  CUMALLOC(To, sz); \
  cudaMemcpy((To), (From), (sz) * sizeof(std::remove_pointer_t<decltype(To)>), cudaMemcpyHostToDevice)

#define DAH2D(From, sz, Type) \
  Type *d_##From;          \
  CUMALLOC(d_##From, sz, Type); \
  cudaMemcpy(d_##From, (From), (sz) * sizeof(Type), cudaMemcpyHostToDevice)


// Allocate device to host local declaration.
#define AD2HL(To, From, sz, Type) \
std::unique_ptr<Type[]> (To) = std::make_unique<Type[]>(sz);  \
cudaMemcpy((To).get(), (From), (sz) * sizeof(Type), cudaMemcpyDeviceToHost)

// Memcopy device to host.
#define AD2H(To, From, sz) \
(To) = std::make_unique<std::remove_pointer_t<decltype(To.get())>[]>(sz);  \
cudaMemcpy(To.get(), From, (sz) * sizeof(std::remove_pointer_t<decltype(To.get())>), cudaMemcpyDeviceToHost)

// Memcopy device to host.
#define D2H(To, From, sz) \
cudaMemcpy(To, (From), (sz) * sizeof(std::remove_pointer_t<decltype(To)>), cudaMemcpyDeviceToHost)

// Memcopy host to device.
#define H2D(To, From, sz) \
cudaMemcpy(d_##From, (From), (sz) * sizeof(std::remove_pointer_t<decltype(To)>), cudaMemcpyHostToDevice)

// Allocate device to host using the Buffer/ScopedBuffer struct.
#define SAD2H(To, From, sz) \
                            \
(To).allocate(sz);                                 \
cudaMemcpy((To).buff, (From), (sz) * sizeof(std::remove_pointer_t<decltype(From)>), cudaMemcpyDeviceToHost)





#define ALLOCATE_HALF(X, sz) cudaMalloc(reinterpret_cast<void**>(&(X)), sizeof(half) * (sz))

