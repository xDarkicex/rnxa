// mps.mm — Objective-C++ implementation of the MPS shim.
//
// matmul uses Apple's MPSMatrixMultiplication (real GPU acceleration
// via MPS). The elementwise activations (ReLU/Sigmoid/Tanh/Softmax)
// run as plain C reference loops on the host. Two reasons:
//
//   1. MPSCNNNeuron wants MPSImage (texture-backed) inputs, but the
//      relux hot path passes plain float32 buffers — building
//      MPSImage per call would dominate the wall time for small n.
//   2. Apple's MPSCNNNeuron `initWithDevice:` initializer is
//      deprecated; the modern form requires MPSNNNeuronDescriptor
//      with a layer descriptor, which is more setup than the
//      activation work itself justifies for our use case.
//
// For the relux MLP workloads the activations are dominated by the
// matmul cost anyway, so plain-C activations are a net win.

#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <stdint.h>
#include <math.h>

#include "mps_ops.h"

namespace {

// ARC-compatible pointer cast (see mps_darwin.go for context).
static inline id<MTLDevice> asDevice(intptr_t p) {
    return (__bridge id<MTLDevice>)(void *)p;
}
static inline id<MTLCommandQueue> asQueue(intptr_t p) {
    return (__bridge id<MTLCommandQueue>)(void *)p;
}

id<MTLDevice> getDevice() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

id<MTLCommandQueue> getQueue() {
    static id<MTLCommandQueue> q = [getDevice() newCommandQueue];
    return q;
}

BOOL runSync(id<MTLCommandBuffer> buf) {
    [buf commit];
    [buf waitUntilCompleted];
    return buf.status == MTLCommandBufferStatusCompleted;
}

MPSMatrixDescriptor *descFor(int64_t rows, int64_t cols) {
    return [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                columns:cols
                                               rowBytes:cols * sizeof(float)
                                               dataType:MPSDataTypeFloat32];
}

}  // namespace

extern "C" {

intptr_t mps_device_new(void) {
    // The device is a process-lifetime singleton. ARC keeps the
    // static alive for the program's lifetime; we hand the
    // pointer back without an explicit retain (which ARC
    // forbids anyway). The Go side stores the intptr_t; calling
    // mps_device_release is a no-op since the singleton never
    // goes away.
    id<MTLDevice> dev = getDevice();
    if (dev == nil) return 0;
    return (intptr_t)dev;
}

void mps_device_release(intptr_t dev) {
    // No-op. The device is a process-lifetime singleton — see
    // mps_device_new. The Go side calls this on Backend.Close
    // for symmetry with the C ABI, but no release happens.
    (void)dev;
}

intptr_t mps_queue_new(intptr_t dev) {
    // Same model as mps_device_new: a process-lifetime singleton
    // command queue. ARC handles the static's lifetime.
    if (!dev) return 0;
    id<MTLDevice> d = asDevice(dev);
    id<MTLCommandQueue> q = [d newCommandQueue];
    if (q == nil) return 0;
    return (intptr_t)q;
}

void mps_queue_release(intptr_t q) {
    // No-op. Singleton queue — see mps_queue_new.
    (void)q;
}

int32_t mps_matmul(intptr_t dev, intptr_t q,
                   const float *A, const float *B, float *C,
                   int64_t M, int64_t N, int64_t K) {
    // Device/queue come from the singletons in this file. The
    // intptr_t params are accepted for C ABI symmetry with the
    // other functions but are ignored — passing the same id<MTLDevice>
    // back across the Go/ObjC boundary via __bridge was producing
    // a borrowed reference that ARC treated as autoreleased, freeing
    // it before the kernel could use the device. The singletons
    // have proper ARC lifetime.
    (void)dev; (void)q;
    if (M <= 0 || N <= 0 || K <= 0) return 1;

    // Per-call autorelease pool. Without this the MTLCommandBuffer,
    // MPSMatrix, and MPSMatrixMultiplication autoreleased objects
    // accumulate in the calling goroutine's pool for the entire
    // benchmark run, which is a 2x regression at 1024x1024. The
    // CGO path in metal_ops_darwin.m wraps the same work in
    // @autoreleasepool { ... } and is faster for that reason.
    @autoreleasepool {
        id<MTLDevice> device = getDevice();
        if (!device) return 2;
        id<MTLCommandQueue> queue = getQueue();

        MPSMatrixDescriptor *aDesc = descFor(M, K);
        MPSMatrixDescriptor *bDesc = descFor(K, N);
        MPSMatrixDescriptor *cDesc = descFor(M, N);

        id<MTLBuffer> aBuf = [device newBufferWithBytes:A
                                              length:M * K * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [device newBufferWithBytes:B
                                              length:K * N * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [device newBufferWithLength:M * N * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

        // MPSMatrix initWithBuffer:descriptor: retains the descriptor
        // internally; the autoreleased object from descFor is fine
        // because the matrix now owns a strong reference.
        MPSMatrix *aMat = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix *bMat = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        MPSMatrix *cMat = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];

        MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
           transposeLeft:NO
          transposeRight:NO
              resultRows:M
           resultColumns:N
          interiorColumns:K
                  alpha:1.0
                   beta:0.0];

        id<MTLCommandBuffer> buf = [queue commandBuffer];
        [mm encodeToCommandBuffer:buf
                    leftMatrix:aMat
                   rightMatrix:bMat
                  resultMatrix:cMat];
        BOOL ok = runSync(buf);
        if (ok) {
            memcpy(C, cBuf.contents, M * N * sizeof(float));
        }

        return ok ? 0 : 2;
    }
}

// Activations: plain C reference loops. See file header for the
// reasoning (MPSImage init is texture-backed; not worth the
// setup overhead for our activation sizes).

int32_t mps_relu(intptr_t dev, intptr_t q,
                const float *X, float *Y, int64_t n) {
    (void)dev; (void)q;
    for (int64_t i = 0; i < n; i++) {
        Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
    }
    return 0;
}

int32_t mps_sigmoid(intptr_t dev, intptr_t q,
                   const float *X, float *Y, int64_t n) {
    (void)dev; (void)q;
    for (int64_t i = 0; i < n; i++) {
        Y[i] = 1.0f / (1.0f + expf(-X[i]));
    }
    return 0;
}

int32_t mps_tanh(intptr_t dev, intptr_t q,
                const float *X, float *Y, int64_t n) {
    (void)dev; (void)q;
    for (int64_t i = 0; i < n; i++) {
        Y[i] = tanhf(X[i]);
    }
    return 0;
}

int32_t mps_softmax(intptr_t dev, intptr_t q,
                   const float *X, float *Y, int64_t n) {
    (void)dev; (void)q;
    if (n <= 0) return 1;
    float maxv = X[0];
    for (int64_t i = 1; i < n; i++) {
        if (X[i] > maxv) maxv = X[i];
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        Y[i] = expf(X[i] - maxv);
        sum += Y[i];
    }
    float inv = 1.0f / sum;
    for (int64_t i = 0; i < n; i++) {
        Y[i] *= inv;
    }
    return 0;
}

}  // extern "C"
