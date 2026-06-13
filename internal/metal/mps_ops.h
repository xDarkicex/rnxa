// mps_ops.h — Objective-C declarations for the MPS-backed GPU engine.
//
// mps_darwin.go (Go side) calls into these via CGO. The implementations
// live in mps_ops.mm (Objective-C++); that file links
// -framework MetalPerformanceShaders and is the only place that touches
// Apple's MPS classes directly.
//
// Build tag: darwin only. The .mm file is compiled by the standard
// clang driver when cgo is active.
//
// This file is currently a skeleton — the actual MPS bindings land
// in mps_ops.mm. The Go side declares the C signatures and binds them
// via C.CFunctionPointer the same way it does for the existing
// metal_darwin.go.

#ifndef RNXA_MPS_OPS_H
#define RNXA_MPS_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// mps_matmul: matrix multiplication via MPSMatrixMultiplication.
// All buffers are FP32 (MPS's native format); float64 callers go
// through a conversion path. A, B are M×K and K×N row-major;
// C is M×N row-major.
int32_t mps_matmul(const float *A, const float *B, float *C,
                   int64_t M, int64_t N, int64_t K);

// Elementwise op codes (mirror the cpu_shim's enum).
#define RNXA_OP_ADD     0
#define RNXA_OP_SUB     1
#define RNXA_OP_MUL     2

// mps_vector_op: elementwise binary op (add/sub/mul). Backed by an
// MPSMatrixVectorMultiplication-with-ones or, failing that, a scalar
// kernel — the mps_ops.mm picks the right path.
int32_t mps_vector_op(int32_t op,
                      const float *A, const float *B, float *C,
                      int64_t n);

// mps_relu: in-place-safe ReLU (writes to output, reads from input).
int32_t mps_relu(const float *X, float *Y, int64_t n);

// mps_unary: dispatch for Sigmoid and Tanh via scalar kernels.
// op == 0 → sigmoid, op == 1 → tanh.
int32_t mps_unary(int32_t op, const float *X, float *Y, int64_t n);

// mps_softmax: softmax along axis. axis < 0 means "all axes".
int32_t mps_softmax(const float *X, float *Y, int64_t n, int64_t axis);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RNXA_MPS_OPS_H
