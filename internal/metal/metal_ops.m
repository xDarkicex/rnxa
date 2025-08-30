#import "metal_ops.h"

// Device management
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
    return [mtlDevice recommendedMaxWorkingSetSize] * 0.8; // Conservative estimate
}

// Matrix multiplication
int metal_matrix_multiply(MTLDeviceRef device, MTLCommandQueueRef queue,
                         const float* A, int M, int K,
                         const float* B, int K2, int N,
                         float* C) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // Create Metal buffers with proper alignment
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
            return -1; // Memory allocation failed
        }
        
        // Use Metal Performance Shaders for optimal performance
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor 
            matrixDescriptorWithDimensions:M
            columns:K
            rowBytes:K * sizeof(float)
            dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor 
            matrixDescriptorWithDimensions:K
            columns:N  
            rowBytes:N * sizeof(float)
            dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor 
            matrixDescriptorWithDimensions:M
            columns:N
            rowBytes:N * sizeof(float) 
            dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
        
        // Optimized matrix multiplication for MLP layers
        MPSMatrixMultiplication* matMul = [[MPSMatrixMultiplication alloc] 
            initWithDevice:mtlDevice
            transposeLeft:NO
            transposeRight:NO
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:1.0
            beta:0.0];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matMul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
            return -2; // Command execution failed
        }
        
        // Copy result back
        memcpy(C, [bufferC contents], M * N * sizeof(float));
        
        return 0; // Success
    }
}

// FIXED: Vector operations using proper MPS APIs
int metal_vector_add(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        if (!bufferA || !bufferB || !bufferC) return -1;
        
        // Use MPS vector operations
        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithDimensions:1
                                                                                 columns:size
                                                                                rowBytes:size * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:desc];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:desc];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:desc];
        
        // FIXED: Use proper method signature
        MPSMatrixSum* vectorAdd = [[MPSMatrixSum alloc] initWithDevice:mtlDevice
                                                                 count:2
                                                                  rows:1
                                                               columns:size
                                                             transpose:NO];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        NSArray<MPSMatrix*>* sourceMatrices = @[matrixA, matrixB];
        [vectorAdd encodeToCommandBuffer:commandBuffer
                           sourceMatrices:sourceMatrices
                             resultMatrix:matrixC];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) return -2;
        
        memcpy(C, [bufferC contents], bufferSize);
        return 0;
    }
}

// Simple CPU implementation for vector subtraction (can optimize later)
int metal_vector_sub(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] - B[i];
    }
    return 0;
}

int metal_vector_mul(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        NSUInteger bufferSize = size * sizeof(float);
        
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        // Simple element-wise multiplication using CPU (can optimize later with Metal compute shaders)
        for (int i = 0; i < size; i++) {
            C[i] = A[i] * B[i];
        }
        
        return 0;
    }
}

// FIXED: Activation functions using modern MPS APIs
int metal_relu(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // FIXED: Use non-deprecated initializer
        MPSCNNNeuronDescriptor* neuronDesc = [MPSCNNNeuronDescriptor cnnNeuronDescriptorWithType:MPSCNNNeuronTypeReLU
                                                                                                a:0.0];
        MPSCNNNeuron* relu = [[MPSCNNNeuron alloc] initWithDevice:mtlDevice neuronDescriptor:neuronDesc];
        
        // Simple CPU implementation for now (can optimize with proper MPS image setup later)
        for (int i = 0; i < size; i++) {
            output[i] = input[i] > 0.0f ? input[i] : 0.0f;
        }
        
        return 0;
    }
}

int metal_sigmoid(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size) {
    @autoreleasepool {
        // Simple CPU implementation for now (can optimize with MPS later)
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f / (1.0f + expf(-input[i]));
        }
        return 0;
    }
}

int metal_tanh(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    @autoreleasepool {
        // Simple CPU implementation for now (can optimize with MPS later)
        for (int i = 0; i < size; i++) {
            output[i] = tanhf(input[i]);
        }
        return 0;
    }
}

int metal_softmax(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size) {
    @autoreleasepool {
        // Simple CPU implementation for numerical stability
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
}

// Safe wrapper functions for device introspection
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
        return 8; // Default estimate
    }
}
