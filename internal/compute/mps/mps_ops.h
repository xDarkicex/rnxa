// mps_ops.h — C ABI for the rnxa MPS backend.
//
// All entry points return int32_t status (0 = success, non-zero =
// implementation-defined error). Buffers are caller-owned; the shim
// never allocates. The Go side is in the same package as this header
// (internal/compute/mps/) and #includes it via the cgo preamble.
//
// FP32 throughout: float64 callers convert on the way in and back.
// The downcast/upcast loses precision (~7 decimal digits vs ~15 for
// float64); this matches the precision the existing Metal backend
// already incurs in metal_darwin.go.
//
// Opaque device/queue handles cross the C/Go boundary as
// intptr_t (not void*). cgo's typedef-auto-naming mangles
// `void *` return types in Objective-C++ functions and the
// generated prolog fails to type-check, so we hide the
// pointer-as-integer indirection behind an explicit cast on
// both sides. The Go side stores the handle as uintptr; the C
// side casts back to id<MTLDevice> etc. internally.

#ifndef RNXA_MPS_OPS_H
#define RNXA_MPS_OPS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device and queue lifecycle. The handle is the raw pointer value
// cast to intptr_t; both sides reinterpret via casts.
intptr_t mps_device_new(void);
void     mps_device_release(intptr_t dev);
intptr_t mps_queue_new(intptr_t dev);
void     mps_queue_release(intptr_t q);

// mps_matmul: C[M,N] = A[M,K] @ B[K,N]. Row-major. All buffers
// FP32, length M*K, K*N, M*N respectively.
int32_t mps_matmul(intptr_t dev, intptr_t q,
                  const float *A, const float *B, float *C,
                  int64_t M, int64_t N, int64_t K);

// mps_relu: Y[i] = max(0, X[i]).
int32_t mps_relu(intptr_t dev, intptr_t q,
                const float *X, float *Y, int64_t n);

// mps_sigmoid: Y[i] = 1 / (1 + exp(-X[i])).
int32_t mps_sigmoid(intptr_t dev, intptr_t q,
                   const float *X, float *Y, int64_t n);

// mps_tanh: Y[i] = tanh(X[i]).
int32_t mps_tanh(intptr_t dev, intptr_t q,
                const float *X, float *Y, int64_t n);

// mps_softmax: Y[i] = exp(X[i] - max(X)) / sum(exp(X - max(X))).
// Input is treated as a flat vector of length n.
int32_t mps_softmax(intptr_t dev, intptr_t q,
                   const float *X, float *Y, int64_t n);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RNXA_MPS_OPS_H
