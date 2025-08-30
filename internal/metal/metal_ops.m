#import "metal_ops.h"

MTLDeviceRef metal_create_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained MTLDeviceRef)device;
}

void metal_release_device(MTLDeviceRef device) {
    id<MTLDevice> mtlDevice = (__bridge_transfer id<MTLDevice>)device;
    mtlDevice = nil;
}

MTLCommandQueueRef metal_create_command_queue(MTLDeviceRef device) {
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    return (__bridge_retained MTLCommandQueueRef)queue;
}

int metal_matrix_multiply(MTLDeviceRef device, MTLCommandQueueRef queue,
                         const float* A, int M, int K,
                         const float* B, int K2, int N,
                         float* C) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // Create Metal buffers
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A 
                                                       length:M * K * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B 
                                                       length:K * N * sizeof(float)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:M * N * sizeof(float)
                                                       options:MTLResourceStorageModeShared];
        
        // Use Metal Performance Shaders for optimal performance
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                                 columns:K
                                                                                rowBytes:K * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor matrixDescriptorWithDimensions:K
                                                                                 columns:N
                                                                                rowBytes:N * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor matrixDescriptorWithDimensions:M
                                                                                 columns:N
                                                                                rowBytes:N * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
        
        // Perform matrix multiplication: C = A × B
        MPSMatrixMultiplication* matMul = [[MPSMatrixMultiplication alloc] initWithDevice:mtlDevice
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
        
        // Copy result back
        memcpy(C, [bufferC contents], M * N * sizeof(float));
        
        return 0; // Success
    }
}

int metal_relu(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // Use Metal Performance Shaders ReLU
        MPSCNNNeuronReLU* relu = [[MPSCNNNeuronReLU alloc] initWithDevice:mtlDevice];
        
        // Create 1D image descriptor (treat as 1D array)
        MPSImageDescriptor* imageDesc = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                                                        width:size
                                                                                       height:1
                                                                                 featureChannels:1];
        
        id<MTLBuffer> inputBuffer = [mtlDevice newBufferWithBytes:input
                                                           length:size * sizeof(float)
                                                          options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> outputBuffer = [mtlDevice newBufferWithLength:size * sizeof(float)
                                                            options:MTLResourceStorageModeShared];
        
        MPSImage* inputImage = [[MPSImage alloc] initWithBuffer:inputBuffer
                                                     descriptor:imageDesc];
        MPSImage* outputImage = [[MPSImage alloc] initWithBuffer:outputBuffer
                                                      descriptor:imageDesc];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [relu encodeToCommandBuffer:commandBuffer sourceImage:inputImage destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(output, [outputBuffer contents], size * sizeof(float));
        
        return 0; // Success
    }
}
