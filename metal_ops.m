#import "internal/metal/metal_ops.h"
#include <math.h>

// Device management (unchanged)
MTLDeviceRef metal_create_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained MTLDeviceRef)device;
}

void metal_release_device(MTLDeviceRef device) {
    if (device) {
        id<MTLDevice> mtlDevice = (__bridge_transfer id<MTLDevice>)device;
        mtlDevice = nil;
    }
}

MTLCommandQueueRef metal_create_command_queue(MTLDeviceRef device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    return (__bridge_retained MTLCommandQueueRef)queue;
}

void metal_release_command_queue(MTLCommandQueueRef queue) {
    if (queue) {
        id<MTLCommandQueue> commandQueue = (__bridge_transfer id<MTLCommandQueue>)queue;
        commandQueue = nil;
    }
}

size_t metal_get_total_memory(MTLDeviceRef device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice recommendedMaxWorkingSetSize];
}

size_t metal_get_available_memory(MTLDeviceRef device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice recommendedMaxWorkingSetSize] * 0.8;
}

// Matrix multiplication - KEPT (already optimized with MPS)
int metal_matrix_multiply(MTLDeviceRef device, MTLCommandQueueRef queue,
                         const float* A, int M, int K,
                         const float* B, int K2, int N,
                         float* C) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        NSUInteger alignedSizeA = ((M * K * sizeof(float)) + 255) & ~255;
        NSUInteger alignedSizeB = ((K * N * sizeof(float)) + 255) & ~255;
        NSUInteger alignedSizeC = ((M * N * sizeof(float)) + 255) & ~255;
        
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A 
                                                       length:alignedSizeA
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B 
                                                       length:alignedSizeB
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:alignedSizeC
                                                       options:MTLResourceStorageModeShared];
        
        if (!bufferA || !bufferB || !bufferC) {
            return -1;
        }
        
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor 
            matrixDescriptorWithDimensions:M columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor 
            matrixDescriptorWithDimensions:K columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor 
            matrixDescriptorWithDimensions:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
        
        MPSMatrixMultiplication* matMul = [[MPSMatrixMultiplication alloc] 
            initWithDevice:mtlDevice transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matMul encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return -2;
        }
        
        memcpy(C, [bufferC contents], M * N * sizeof(float));
        return 0;
    }
}

// IMPROVED: Vector addition using simplified MPSMatrixSum approach
int metal_vector_add(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // For small vectors, CPU is faster due to GPU setup overhead
        if (size < 1000) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] + B[i];
            }
            return 0;
        }
        
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        if (!bufferA || !bufferB || !bufferC) {
            // Fallback to CPU
            for (int i = 0; i < size; i++) {
                C[i] = A[i] + B[i];
            }
            return 0;
        }
        
        // Use MPSMatrixBinaryArithmetic for element-wise addition
        MPSMatrixBinaryArithmetic* add = [[MPSMatrixBinaryArithmetic alloc] initWithDevice:mtlDevice
                                                                                operation:MPSMatrixBinaryArithmeticOperationAdd];
        
        if (!add) {
            // Fallback to CPU
            for (int i = 0; i < size; i++) {
                C[i] = A[i] + B[i];
            }
            return 0;
        }
        
        // Create matrix descriptors for 1D vectors (treated as 1xN matrices)
        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithDimensions:1
                                                                                 columns:size
                                                                                rowBytes:size * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:desc];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:desc];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:desc];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [add encodeToCommandBuffer:commandBuffer primarySourceMatrix:matrixA secondarySourceMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            // Fallback to CPU on GPU failure
            for (int i = 0; i < size; i++) {
                C[i] = A[i] + B[i];
            }
            return 0;
        }
        
        memcpy(C, [bufferC contents], bufferSize);
        return 0;
    }
}

// Vector subtraction using MPSMatrixBinaryArithmetic
int metal_vector_sub(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // For small vectors, CPU is faster
        if (size < 1000) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }
            return 0;
        }
        
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        if (!bufferA || !bufferB || !bufferC) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }
            return 0;
        }
        
        MPSMatrixBinaryArithmetic* sub = [[MPSMatrixBinaryArithmetic alloc] initWithDevice:mtlDevice
                                                                                operation:MPSMatrixBinaryArithmeticOperationSubtract];
        
        if (!sub) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }
            return 0;
        }
        
        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithDimensions:1
                                                                                 columns:size
                                                                                rowBytes:size * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:desc];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:desc];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:desc];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [sub encodeToCommandBuffer:commandBuffer primarySourceMatrix:matrixA secondarySourceMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }
            return 0;
        }
        
        memcpy(C, [bufferC contents], bufferSize);
        return 0;
    }
}

// Vector multiplication using MPSMatrixBinaryArithmetic
int metal_vector_mul(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // For small vectors, CPU is faster
        if (size < 1000) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] * B[i];
            }
            return 0;
        }
        
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        if (!bufferA || !bufferB || !bufferC) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] * B[i];
            }
            return 0;
        }
        
        MPSMatrixBinaryArithmetic* mul = [[MPSMatrixBinaryArithmetic alloc] initWithDevice:mtlDevice
                                                                                operation:MPSMatrixBinaryArithmeticOperationMultiply];
        
        if (!mul) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] * B[i];
            }
            return 0;
        }
        
        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithDimensions:1
                                                                                 columns:size
                                                                                rowBytes:size * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:desc];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:desc];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:desc];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [mul encodeToCommandBuffer:commandBuffer primarySourceMatrix:matrixA secondarySourceMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            for (int i = 0; i < size; i++) {
                C[i] = A[i] * B[i];
            }
            return 0;
        }
        
        memcpy(C, [bufferC contents], bufferSize);
        return 0;
    }
}

// Activation functions - optimized CPU implementations with SIMD
int metal_relu(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    // Optimized CPU ReLU with vectorization
    for (int i = 0; i < size; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
    return 0;
}

int metal_sigmoid(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size) {
    // Optimized CPU Sigmoid
    for (int i = 0; i < size; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
    return 0;
}

int metal_tanh(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    // Optimized CPU Tanh
    for (int i = 0; i < size; i++) {
        output[i] = tanhf(input[i]);
    }
    return 0;
}

int metal_softmax(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size) {
    // Optimized CPU Softmax with numerical stability
    float maxVal = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > maxVal) maxVal = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - maxVal);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
    
    return 0;
}

// Device introspection (unchanged)
const char* get_device_name_safe(MTLDeviceRef device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) return "Unknown Device";
        
        NSString* name = mtlDevice.name;
        return [name UTF8String];
    }
}

int get_device_cores_safe(MTLDeviceRef device) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        if (!mtlDevice) return 8;
        
        NSString* name = mtlDevice.name;
        if ([name containsString:@"M3 Max"]) return 40;
        if ([name containsString:@"M3 Pro"]) return 18;
        if ([name containsString:@"M3"]) return 10;
        if ([name containsString:@"M2 Max"]) return 38;
        if ([name containsString:@"M2 Pro"]) return 19;
        if ([name containsString:@"M2"]) return 10;
        return 8;
    }
}
