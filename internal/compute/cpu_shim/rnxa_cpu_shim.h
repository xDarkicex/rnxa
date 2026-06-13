// rnxa_cpu_shim.h — C ABI for the rnxa CPU shared library.
//
// This is the public surface that the Go side (internal/compute/cpu_purego.go)
// calls into via purego. The C++ implementation in rnxa_cpu_shim.cpp
// dispatches each call to a oneDNN primitive.
//
// All entry points return int32_t status code (0 = success, non-zero =
// implementation-defined error). Output buffers are caller-owned; the
// shim never allocates.
//
// Build: see CMakeLists.txt in this directory. oneDNN is fetched via
// FetchContent so there is no system dependency.

#ifndef RNXA_CPU_SHIM_H
#define RNXA_CPU_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// rnxa_matmul_f64: C[M,N] = A[M,K] @ B[K,N]. Row-major.
int32_t rnxa_matmul_f64(const double *A, const double *B, double *C,
                        int64_t M, int64_t N, int64_t K);

// rnxa_matmul_f32: float32 variant of rnxa_matmul_f64.
int32_t rnxa_matmul_f32(const float *A, const float *B, float *C,
                        int64_t M, int64_t N, int64_t K);

// Op codes for rnxa_vector_op. Values are stable; do not renumber.
#define RNXA_OP_ADD     0
#define RNXA_OP_SUB     1
#define RNXA_OP_MUL     2
#define RNXA_OP_RELU    3
#define RNXA_OP_SIGMOID 4
#define RNXA_OP_TANH    5

// rnxa_vector_op: dispatch a binary or unary elementwise op.
// For binary ops, A != B and C is the output. For unary ops (RELU,
// SIGMOID, TANH), the caller passes A == B; the shim detects this and
// uses the unary eltwise path.
int32_t rnxa_vector_op(int32_t op,
                       const double *A, const double *B, double *C,
                       int64_t n);

// rnxa_softmax: softmax along axis. axis < 0 means "all axes" (full
// softmax over the entire tensor). Otherwise 0 <= axis < ndim.
int32_t rnxa_softmax(const double *X, double *Y, int64_t n, int64_t axis);

// rnxa_reduce_sum: sum over axis. axis < 0 means "all axes" (returns
// scalar). Otherwise 0 <= axis < ndim. Output is a tensor of rank
// ndim-1 with axis removed (matches the Go helper's behavior).
int32_t rnxa_reduce_sum(const double *X, double *Y, int64_t n, int64_t axis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RNXA_CPU_SHIM_H
