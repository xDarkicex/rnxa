// cuda.cu — rnxa CUDA shim implementation.
//
// Built by nvcc into libcuda.so. The Go side (cuda_linux.go) loads
// the .so via purego and binds the C ABI symbols declared in
// cuda_ops.h. The shim is a single translation unit: a context
// struct, a few wrappers around cuBLAS / cuDNN, and a tiny
// custom elementwise kernel.
//
// Synchronous-per-call model: every kernel launch ends with
// cudaStreamSynchronize(0) so the calling goroutine blocks until
// the GPU work is done. The Go-side cudaEngine serialises calls
// through its own mutex, so the shim itself does not need to be
// thread-safe.
//
// Build (see Makefile):
//   nvcc -O3 -Xcompiler -fPIC -I. -shared cuda.cu -o build/libcuda.so \
//        -lcublas -lcudnn
//
// First cut: single GPU (device 0), float32 only, per-call cuDNN
// descriptor lifecycle. Multi-GPU and FP16 are follow-ups — the
// ABI already plumbs (int32_t dev) so adding multi-device support
// is just dropping the dev==0 assertion and binding a handle per
// device.

#include "cuda_ops.h"

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

// Internal context. Opaque to the Go side; the handle is just an
// intptr_t that we cast back to a RnxaCudaCtx* on every call.
struct RnxaCudaCtx {
    cublasHandle_t cublas;
    cudnnHandle_t  cudnn;
};

// ---------------------------------------------------------------------------
// Lifecycle

extern "C" intptr_t rnxa_cuda_init(void) {
    cudaError_t cerr;

    cerr = cudaSetDevice(0);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "[rnxa_cuda] cudaSetDevice(0) failed: %s\n",
                     cudaGetErrorString(cerr));
        return 0;
    }

    RnxaCudaCtx* ctx = new RnxaCudaCtx();
    ctx->cublas = nullptr;
    ctx->cudnn  = nullptr;

    cublasStatus_t bst = cublasCreate(&ctx->cublas);
    if (bst != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[rnxa_cuda] cublasCreate failed: %d\n", (int)bst);
        delete ctx;
        return 0;
    }

    cudnnStatus_t dst = cudnnCreate(&ctx->cudnn);
    if (dst != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "[rnxa_cuda] cudnnCreate failed: %s\n",
                     cudnnGetErrorString(dst));
        cublasDestroy(ctx->cublas);
        delete ctx;
        return 0;
    }

    return reinterpret_cast<intptr_t>(ctx);
}

extern "C" void rnxa_cuda_shutdown(intptr_t handle) {
    if (handle == 0) return;
    RnxaCudaCtx* ctx = reinterpret_cast<RnxaCudaCtx*>(handle);
    if (ctx->cudnn)  cudnnDestroy(ctx->cudnn);
    if (ctx->cublas) cublasDestroy(ctx->cublas);
    delete ctx;
}

// ---------------------------------------------------------------------------
// Enumeration

extern "C" int32_t rnxa_cuda_device_count(intptr_t /*handle*/) {
    int n = 0;
    cudaError_t cerr = cudaGetDeviceCount(&n);
    if (cerr != cudaSuccess) return 0;
    return (int32_t)n;
}

extern "C" int32_t rnxa_cuda_device_name(intptr_t /*handle*/, int32_t idx,
                                         char *buf, int32_t len) {
    if (buf == nullptr || len <= 0) return -1;
    cudaDeviceProp prop;
    cudaError_t cerr = cudaGetDeviceProperties(&prop, (int)idx);
    if (cerr != cudaSuccess) return -1;
    // cudaDeviceProp::name is 256 bytes; snprintf will NUL-terminate
    // if len < 256, so we use snprintf for safety.
    int n = std::snprintf(buf, (size_t)len, "%s", prop.name);
    if (n < 0) return -1;
    if (n >= len) n = len - 1;  // truncation marker
    return (int32_t)n;
}

extern "C" int64_t rnxa_cuda_device_memory(intptr_t /*handle*/, int32_t idx) {
    cudaDeviceProp prop;
    cudaError_t cerr = cudaGetDeviceProperties(&prop, (int)idx);
    if (cerr != cudaSuccess) return 0;
    return (int64_t)prop.totalGlobalMem;
}

// ---------------------------------------------------------------------------
// Matmul (cuBLAS)
//
// Row-major C = A @ B where A is M×K, B is K×N, C is M×N.
// cuBLAS is column-major, so we declare the matrices column-major
// with the standard "transpose the problem" trick:
//   C_col(N, M) = B_col(N, K) * A_col(K, M)
// Call:
//   cublasSgemm(handle, OP_N, OP_N, N, M, K,
//               &one, B, N, A, K, &zero, C, N)
// This produces a row-major C in the caller's storage.

extern "C" int32_t rnxa_cuda_matmul(intptr_t handle, int32_t dev,
                                    const float *A, const float *B, float *C,
                                    int64_t M, int64_t N, int64_t K) {
    if (dev != 0) return 10;
    if (M <= 0 || N <= 0 || K <= 0) return 1;
    if (handle == 0) return 2;
    RnxaCudaCtx* ctx = reinterpret_cast<RnxaCudaCtx*>(handle);

    const float one  = 1.0f;
    const float zero = 0.0f;
    cublasStatus_t st = cublasSgemm(
        ctx->cublas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &one,
        B, (int)N,
        A, (int)K,
        &zero,
        C, (int)N);
    if (st != CUBLAS_STATUS_SUCCESS) return 3;

    cudaError_t cerr = cudaStreamSynchronize(0);
    if (cerr != cudaSuccess) return 4;
    return 0;
}

// ---------------------------------------------------------------------------
// Elementwise: custom CUDA kernel
//
// cuBLAS has no sub/mul primitives (only axpy, which is
// add-and-scale). To keep one ABI entry point for all three ops
// we launch a small __global__ kernel. Block size 256, grid
// sized to cover n.

__global__ void rnxa_eltwise_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int64_t n, int32_t op) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = A[i];
    float b = B[i];
    if (op == 0)      C[i] = a + b;  // add
    else if (op == 1) C[i] = a - b;  // sub
    else              C[i] = a * b;  // mul
}

extern "C" int32_t rnxa_cuda_eltwise(intptr_t handle, int32_t dev, int32_t op,
                                     const float *A, const float *B, float *C,
                                     int64_t n) {
    if (dev != 0) return 10;
    if (n <= 0) return 1;
    if (op < 0 || op > 2) return 2;
    if (handle == 0) return 3;
    (void)handle;  // we don't need the ctx for this kernel

    int threads = 256;
    int blocks  = (int)((n + threads - 1) / threads);
    rnxa_eltwise_kernel<<<blocks, threads>>>(A, B, C, n, op);
    cudaError_t cerr = cudaGetLastError();
    if (cerr != cudaSuccess) return 4;
    cerr = cudaStreamSynchronize(0);
    if (cerr != cudaSuccess) return 5;
    return 0;
}

// ---------------------------------------------------------------------------
// Activations (cuDNN)
//
// cudnnActivationForward with CUDNN_ACTIVATION_{RELU,SIGMOID,TANH}.
// The tensor is a flat 1D n-element float32 vector described as
// (1, n, 1, 1) NCHW. Descriptors are created per call — descriptor
// creation is microseconds, and the alternative (a small
// per-shape cache) is a follow-up perf optimization.

static int32_t run_activation(intptr_t handle, int32_t dev,
                              const float *X, float *Y, int64_t n,
                              cudnnActivationMode_t mode) {
    if (dev != 0) return 10;
    if (n <= 0) return 1;
    if (handle == 0) return 2;
    RnxaCudaCtx* ctx = reinterpret_cast<RnxaCudaCtx*>(handle);

    cudnnTensorDescriptor_t xDesc = nullptr;
    cudnnActivationDescriptor_t actDesc = nullptr;
    float one = 1.0f, zero = 0.0f;

    cudnnStatus_t st;

    st = cudnnCreateTensorDescriptor(&xDesc);
    if (st != CUDNN_STATUS_SUCCESS) goto done;
    st = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT, 1, (int)n, 1, 1);
    if (st != CUDNN_STATUS_SUCCESS) goto done;

    st = cudnnCreateActivationDescriptor(&actDesc);
    if (st != CUDNN_STATUS_SUCCESS) goto done;
    st = cudnnSetActivationDescriptor(actDesc, mode, CUDNN_PROPAGATE_NAN, 0.0);
    if (st != CUDNN_STATUS_SUCCESS) goto done;

    st = cudnnActivationForward(ctx->cudnn, actDesc,
                                &one, xDesc, X,
                                &zero, xDesc, Y);
    if (st != CUDNN_STATUS_SUCCESS) goto done;

    cudaError_t cerr = cudaStreamSynchronize(0);
    if (cerr != cudaSuccess) { st = CUDNN_STATUS_INTERNAL_ERROR; goto done; }

done:
    if (xDesc)   cudnnDestroyTensorDescriptor(xDesc);
    if (actDesc) cudnnDestroyActivationDescriptor(actDesc);
    if (st != CUDNN_STATUS_SUCCESS) return 3;
    return 0;
}

extern "C" int32_t rnxa_cuda_relu(intptr_t handle, int32_t dev,
                                  const float *X, float *Y, int64_t n) {
    return run_activation(handle, dev, X, Y, n, CUDNN_ACTIVATION_RELU);
}

extern "C" int32_t rnxa_cuda_sigmoid(intptr_t handle, int32_t dev,
                                     const float *X, float *Y, int64_t n) {
    return run_activation(handle, dev, X, Y, n, CUDNN_ACTIVATION_SIGMOID);
}

extern "C" int32_t rnxa_cuda_tanh(intptr_t handle, int32_t dev,
                                  const float *X, float *Y, int64_t n) {
    return run_activation(handle, dev, X, Y, n, CUDNN_ACTIVATION_TANH);
}

// ---------------------------------------------------------------------------
// Softmax (cuDNN)
//
// Treats the input as a 1D vector of length n, configured as
// (1, n, 1, 1) NCHW with CUDNN_SOFTMAX_MODE_INSTANCE so the
// whole-tensor softmax is the natural interpretation. Per-call
// descriptor lifecycle matches the activations path.

extern "C" int32_t rnxa_cuda_softmax(intptr_t handle, int32_t dev,
                                     const float *X, float *Y, int64_t n) {
    if (dev != 0) return 10;
    if (n <= 0) return 1;
    if (handle == 0) return 2;
    RnxaCudaCtx* ctx = reinterpret_cast<RnxaCudaCtx*>(handle);

    cudnnTensorDescriptor_t xDesc = nullptr;
    float one = 1.0f, zero = 0.0f;
    cudnnStatus_t st;

    st = cudnnCreateTensorDescriptor(&xDesc);
    if (st != CUDNN_STATUS_SUCCESS) goto done;
    st = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW,
                                    CUDNN_DATA_FLOAT, 1, (int)n, 1, 1);
    if (st != CUDNN_STATUS_SUCCESS) goto done;

    st = cudnnSoftmaxForward(ctx->cudnn, CUDNN_SOFTMAX_ACCURATE,
                             CUDNN_SOFTMAX_MODE_INSTANCE,
                             &one, xDesc, X, &zero, xDesc, Y);
    if (st != CUDNN_STATUS_SUCCESS) goto done;

    {
        cudaError_t cerr = cudaStreamSynchronize(0);
        if (cerr != cudaSuccess) { st = CUDNN_STATUS_INTERNAL_ERROR; goto done; }
    }

done:
    if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
    if (st != CUDNN_STATUS_SUCCESS) return 3;
    return 0;
}
