// Metal Performance Shaders C interface
#ifndef METAL_OPS_H
#define METAL_OPS_H

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;

// Device management
MTLDeviceRef metal_create_device(void);
void metal_release_device(MTLDeviceRef device);
MTLCommandQueueRef metal_create_command_queue(MTLDeviceRef device);
void metal_release_command_queue(MTLCommandQueueRef queue);

// Memory info
size_t metal_get_total_memory(MTLDeviceRef device);
size_t metal_get_available_memory(MTLDeviceRef device);

// Core MLP operations
int metal_matrix_multiply(MTLDeviceRef device, MTLCommandQueueRef queue,
                         const float* A, int M, int K,
                         const float* B, int K2, int N,
                         float* C);

// Vector operations for bias addition
int metal_vector_add(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size);
int metal_vector_sub(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size);
int metal_vector_mul(MTLDeviceRef device, MTLCommandQueueRef queue,
                     const float* A, const float* B, float* C, int size);

// MLP activation functions
int metal_relu(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size);
int metal_sigmoid(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size);
int metal_tanh(MTLDeviceRef device, MTLCommandQueueRef queue,
               const float* input, float* output, int size);
int metal_softmax(MTLDeviceRef device, MTLCommandQueueRef queue,
                  const float* input, float* output, int size);

// Add these lines at the end, before #endif
// Safe device introspection functions
const char* get_device_name_safe(MTLDeviceRef device);
int get_device_cores_safe(MTLDeviceRef device);

#endif

