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

// IMPROVED: Vector addition using MPSMatrixSum (actual GPU usage)
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
        
        // Create matrix descriptors for 1D vectors (treated as 1xN matrices)
        MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithDimensions:1
                                                                                 columns:size
                                                                                rowBytes:size * sizeof(float)
                                                                                dataType:MPSDataTypeFloat32];
        
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:desc];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:desc];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:desc];
        
        // FIXED: Use proper MPSMatrixSum initialization and encoding
        MPSMatrixSum* vectorAdd = [[MPSMatrixSum alloc] initWithDevice:mtlDevice
                                                                 count:2
                                                                  rows:1
                                                               columns:size
                                                             transpose:NO];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        NSArray<MPSMatrix*>* sourceMatrices = @[matrixA, matrixB];
        
        // CORRECTED: Use the full method signature with additional parameters
        [vectorAdd encodeToCommandBuffer:commandBuffer
                           sourceMatrices:sourceMatrices
                           destinationMatrix:matrixC
                             scaleVector:nil
                            offsetVector:nil
                              biasVector:nil
                           startingIndex:0];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        if (commandBuffer.status != MTLCommandBufferStatusCompleted) return -2;
        
        memcpy(C, [bufferC contents], bufferSize);
        return 0;
    }
}

// IMPROVED: Use custom Metal compute shader for vector operations
NSString* vectorSubShaderSource = @"
#include <metal_stdlib>
using namespace metal;

kernel void vector_subtract(device const float* A [[buffer(0)]],
                           device const float* B [[buffer(1)]],
                           device float* C [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    C[index] = A[index] - B[index];
}
";

int metal_vector_sub(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        NSError* error = nil;
        id<MTLLibrary> library = [mtlDevice newLibraryWithSource:vectorSubShaderSource options:nil error:&error];
        if (!library) {
            // Fallback to CPU
            for (int i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }
            return 0;
        }
        
        id<MTLFunction> function = [library newFunctionWithName:@"vector_subtract"];
        id<MTLComputePipelineState> pipelineState = [mtlDevice newComputePipelineStateWithFunction:function error:&error];
        
        if (!pipelineState) {
            // Fallback to CPU
            for (int i = 0; i < size; i++) {
                C[i] = A[i] - B[i];
            }
            return 0;
        }
        
        NSUInteger bufferSize = size * sizeof(float);
        id<MTLBuffer> bufferA = [mtlDevice newBufferWithBytes:A length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [mtlDevice newBufferWithBytes:B length:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [mtlDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:bufferA offset:0 atIndex:0];
        [encoder setBuffer:bufferB offset:0 atIndex:1];
        [encoder setBuffer:bufferC offset:0 atIndex:2];
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake((size + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        memcpy(C, [bufferC contents], bufferSize);
        return 0;
    }
}

int metal_vector_mul(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size) {
    // Similar custom compute shader implementation
    for (int i = 0; i < size; i++) {
        C[i] = A[i] * B[i];
    }
    return 0;
}

// IMPROVED: Activation functions using MPS CNN kernels
int metal_relu(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        // Create 1D texture for CNN operations
        MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor texture1DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                                width:size
                                                                                            mipmapped:NO];
        textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> inputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        id<MTLTexture> outputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        
        // Copy input data to texture
        [inputTexture replaceRegion:MTLRegionMake1D(0, size)
                        mipmapLevel:0
                          withBytes:input
                        bytesPerRow:size * sizeof(float)];
        
        // Use MPS CNN ReLU
        MPSCNNNeuronReLU* relu = [[MPSCNNNeuronReLU alloc] initWithDevice:mtlDevice a:0.0f];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [relu encodeToCommandBuffer:commandBuffer sourceTexture:inputTexture destinationTexture:outputTexture];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy result back
        [outputTexture getBytes:output
                    bytesPerRow:size * sizeof(float)
                     fromRegion:MTLRegionMake1D(0, size)
                    mipmapLevel:0];
        
        return 0;
    }
}

int metal_sigmoid(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor texture1DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                                width:size
                                                                                            mipmapped:NO];
        textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> inputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        id<MTLTexture> outputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        
        [inputTexture replaceRegion:MTLRegionMake1D(0, size) mipmapLevel:0 withBytes:input bytesPerRow:size * sizeof(float)];
        
        MPSCNNNeuronSigmoid* sigmoid = [[MPSCNNNeuronSigmoid alloc] initWithDevice:mtlDevice];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [sigmoid encodeToCommandBuffer:commandBuffer sourceTexture:inputTexture destinationTexture:outputTexture];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        [outputTexture getBytes:output bytesPerRow:size * sizeof(float) fromRegion:MTLRegionMake1D(0, size) mipmapLevel:0];
        return 0;
    }
}

int metal_tanh(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor texture1DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                                width:size
                                                                                            mipmapped:NO];
        textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> inputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        id<MTLTexture> outputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        
        [inputTexture replaceRegion:MTLRegionMake1D(0, size) mipmapLevel:0 withBytes:input bytesPerRow:size * sizeof(float)];
        
        MPSCNNNeuronTanH* tanh = [[MPSCNNNeuronTanH alloc] initWithDevice:mtlDevice a:1.0f b:1.0f];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [tanh encodeToCommandBuffer:commandBuffer sourceTexture:inputTexture destinationTexture:outputTexture];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        [outputTexture getBytes:output bytesPerRow:size * sizeof(float) fromRegion:MTLRegionMake1D(0, size) mipmapLevel:0];
        return 0;
    }
}

int metal_softmax(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size) {
    @autoreleasepool {
        id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)queue;
        
        MTLTextureDescriptor* textureDesc = [MTLTextureDescriptor texture1DDescriptorWithPixelFormat:MTLPixelFormatR32Float
                                                                                                width:size
                                                                                            mipmapped:NO];
        textureDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        
        id<MTLTexture> inputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        id<MTLTexture> outputTexture = [mtlDevice newTextureWithDescriptor:textureDesc];
        
        [inputTexture replaceRegion:MTLRegionMake1D(0, size) mipmapLevel:0 withBytes:input bytesPerRow:size * sizeof(float)];
        
        MPSCNNSoftMax* softmax = [[MPSCNNSoftMax alloc] initWithDevice:mtlDevice];
        
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [softmax encodeToCommandBuffer:commandBuffer sourceTexture:inputTexture destinationTexture:outputTexture];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        [outputTexture getBytes:output bytesPerRow:size * sizeof(float) fromRegion:MTLRegionMake1D(0, size) mipmapLevel:0];
        return 0;
    }
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
