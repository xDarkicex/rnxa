// cuda_ops.h — C ABI for libcuda.so (the rnxa CUDA shim).
//
// The shim is built by nvcc from cuda.cu, transitively linking
// cuBLAS and cuDNN. The Go side (cuda_linux.go) loads libcuda.so
// via purego and resolves the symbols declared here.
//
// All entry points return int32_t (0 = ok). Buffers are
// caller-owned float32 slices. The shim holds a single opaque
// intptr_t context created by rnxa_cuda_init; the context owns
// the cuBLAS handle, the cuDNN handle, and (first cut) the
// implicit device=0 binding.
//
// The float32-only contract matches the rnxa MPS backend:
// float64 callers downcast on the way in and upcast on the way
// out. The precision roundtrip is the same as for the MPS path.

#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Lifecycle.

// rnxa_cuda_init creates the cuBLAS + cuDNN handles, binds device 0,
// and returns an opaque context pointer. 0 on failure (driver
// loaded but no GPU visible, or driver/runtime version mismatch).
intptr_t rnxa_cuda_init(void);

// rnxa_cuda_shutdown releases the handles. First cut is a no-op —
// the OS reclaims on process exit. The stub exists so the
// dispatcher can call it without a nil check.
void rnxa_cuda_shutdown(intptr_t ctx);

// Enumeration.

// rnxa_cuda_device_count returns the number of NVIDIA GPUs the
// driver sees. Always >= 0.
int32_t rnxa_cuda_device_count(intptr_t ctx);

// rnxa_cuda_device_name copies the name of device idx into buf
// (NUL-terminated if buf has room). Returns the number of bytes
// written, or a negative value on error. buf must be at least
// 256 bytes — the wrapper on the Go side allocates 256.
int32_t rnxa_cuda_device_name(intptr_t ctx, int32_t idx, char *buf, int32_t len);

// rnxa_cuda_device_memory returns the total global memory of
// device idx in bytes. 0 on error.
int64_t rnxa_cuda_device_memory(intptr_t ctx, int32_t idx);

// Compute. All ops are synchronous; the shim does
// cudaStreamSynchronize(0) at the end of every kernel launch
// (matching the MPS backend's waitUntilCompleted model).

// rnxa_cuda_matmul computes C = A @ B with row-major layout.
//   A is M×K, B is K×N, C is M×N.
// dev is the device index; first cut asserts dev == 0.
int32_t rnxa_cuda_matmul(intptr_t ctx, int32_t dev,
                         const float *A, const float *B, float *C,
                         int64_t M, int64_t N, int64_t K);

// rnxa_cuda_eltwise runs an elementwise binary op:
//   op == 0: C[i] = A[i] + B[i]
//   op == 1: C[i] = A[i] - B[i]
//   op == 2: C[i] = A[i] * B[i]
// n is the element count; A, B, C must each have at least n
// elements. The shim uses a custom __global__ kernel — cuBLAS
// has no sub/mul primitives, so a single kernel keeps one ABI
// slot for all three ops.
int32_t rnxa_cuda_eltwise(intptr_t ctx, int32_t dev, int32_t op,
                          const float *A, const float *B, float *C,
                          int64_t n);

// rnxa_cuda_relu / sigmoid / tanh each apply the corresponding
// activation elementwise via cuDNN cudnnActivationForward. n is
// the element count; X and Y must each have at least n elements.
int32_t rnxa_cuda_relu(intptr_t ctx, int32_t dev,
                       const float *X, float *Y, int64_t n);
int32_t rnxa_cuda_sigmoid(intptr_t ctx, int32_t dev,
                          const float *X, float *Y, int64_t n);
int32_t rnxa_cuda_tanh(intptr_t ctx, int32_t dev,
                       const float *X, float *Y, int64_t n);

// rnxa_cuda_softmax applies softmax over the entire input. The
// shim treats the input as a 1D vector of length n, configured
// as a (1, n, 1, 1) NCHW tensor with CUDNN_SOFTMAX_MODE_INSTANCE
// so the whole-tensor softmax is the natural interpretation.
int32_t rnxa_cuda_softmax(intptr_t ctx, int32_t dev,
                          const float *X, float *Y, int64_t n);

#ifdef __cplusplus
}
#endif

#endif // CUDA_OPS_H
