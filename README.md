# **rnxa** - Hardware-Accelerated ML Compute Engine for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/xDarkicex/rnxa.svg)](https://pkg.go.dev/github.com/xDarkicex/rnxa)
[![Go Report Card](https://goreportcard.com/badge/github.com/xDarkicex/rnxa)](https://goreportcard.com/report/github.com/xDarkicex/rnxa)
[![Go Version](https://img.shields.io/badge/go-1.25+-blue.svg)](https://golang.org)
[![Metal](https://img.shields.io/badge/graphics-Metal-blue.svg)](https://developer.apple.com/metal/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Go compute engine for ML tensor operations, with hardware acceleration on
Apple Silicon via Metal Performance Shaders (MPS) and a Metal fallback.

**rnxa** (pronounced "RNA") provides hardware-accelerated tensor operations for
machine learning workloads in Go. Initially developed to accelerate the
[relux](https://github.com/xDarkicex/relux) neural network framework, rnxa is
designed as a universal compute backend that can integrate with any Go ML
framework.

---

## 🎯 **Why rnxa?**

**Performance that Scales:**
- 🚀 **MPS matmul on Apple Silicon** - Apple's hand-tuned `MPSMatrixMultiplication`
  kernel via a purego-loaded `libmps.dylib`
- ⚡ **Metal fallback** - hand-written kernels for shapes MPS doesn't accelerate
- 🔄 **Smart fallbacks** - transparent downshift to a Metal → CPU ladder
- 📊 **Float32 fast path** - `MatMul` skips the float64↔float32 roundtrip when
  both inputs are already float32

**Framework Agnostic:**
- 🔌 **Universal interface** - works with any Go ML framework
- 🧩 **Clean abstractions** - no framework-specific dependencies
- 🛡️ **Production ready** - comprehensive error handling and resource management
- 📈 **Scalable** - from small models to large neural networks

---

## 🏗️ **Architecture Overview**

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│   Go ML         │    │      rnxa        │    │   Hardware           │
│  Framework      │───▶│  ComputeEngine   │───▶│   Acceleration       │
│ (relux, etc.)   │    │   Interface      │    │  MPS / Metal / CUDA  │
└─────────────────┘    └──────────────────┘    │       / CPU          │
                                              └──────────────────────┘
```

### **Backend Priority**

`NewEngine()` walks `DetectDevices()` and instantiates the first backend whose
`Available()` is true. On each platform the ladder looks like:

- **macOS:** **MPS** → **Metal** → **CPU**
- **Linux:**  **CUDA** → **CPU** (Metal isn't available; MPS isn't available)
- **Windows:** **CPU** (CUDA + DirectML are follow-ups)

Per-backend details:

1. **MPS** (`Platform: "MPS"`, darwin) - `libmps.dylib` loaded via purego;
   matmul runs on the real `MPSMatrixMultiplication` kernel. Float32 only;
   float64 inputs are downcast on the way in and upcast on the way out.
2. **Metal** (`Platform: "Metal"`, darwin) - hand-written compute kernels in
   `internal/metal/`. Fallback for shapes MPS doesn't accelerate, or for
   environments where the MPS shim isn't built.
3. **CUDA** (`Platform: "CUDA"`, linux) - `libcuda.so` loaded via purego;
   matmul runs on cuBLAS `cublasSgemm`, activations + softmax on cuDNN.
   Per-call cuDNN descriptor lifecycle. Float32 only; same downcast
   contract as MPS. Code-complete; awaiting a Linux build agent with `nvcc`
   and an NVIDIA GPU to validate the GPU path end-to-end.
4. **CPU** (`Platform: "CPU"`, all platforms) - pure Go matmul; always
   available.

If a higher-priority backend's shim isn't built (e.g. `libmps.dylib` missing
on a Mac, `libcuda.so` missing on Linux) or the runtime probe finds no
hardware (`nvidia-smi` returns no GPUs), that backend's `Available()` returns
false and the dispatcher falls through to the next one. CUDA / OpenCL slots
are reserved for future backends; selecting one returns "not implemented"
and the loop skips it.

### **Core Components**

- **ComputeEngine Interface** - framework-agnostic tensor operations
- **MPS Backend** - Apple's `MPSMatrixMultiplication` via purego FFI
- **Metal Backend** - GPU acceleration via Apple's Metal compute shaders
- **CUDA Backend** - cuBLAS matmul + cuDNN activations / softmax via purego FFI
- **CPU Backend** - oneDNN shim (`librnxa_cpu`) for fast CPU matmul where built;
  pure Go matmul fallback always available
- **Device Management** - automatic detection and selection of compute devices
  (Apple GPUs via Metal, NVIDIA GPUs via `nvidia-smi`, host CPU)
- **Tensor Abstraction** - efficient n-dimensional array representation with
  off-heap storage (`alloc.Float32` / `alloc.Float64`)

---

## ⚡ **Quick Start**

### **Prerequisites**

- **macOS** with Apple Silicon (M1/M2/M3) or Intel Mac with Metal support
- **Go 1.25+**
- **Xcode Command Line Tools** (xcode-select version 2410+)

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select --version
# Should output: xcode-select version 2410 or higher
```

### **Installation**

```bash
go get github.com/xDarkicex/rnxa
```

### **Building the MPS Shim (Optional but Recommended on macOS)**

The MPS backend ships as an Objective-C++ shared library built with `clang++`
and loaded at runtime via purego - no CGO, no tagged builds. To use the
production-recommended MPS path on macOS:

```bash
cd internal/compute/mps
make                       # produces build/libmps.dylib

# Point the loader at it for your run:
DYLD_LIBRARY_PATH=$(pwd)/build go test ./...
```

Without the shim, `rnxa.NewEngine()` still works - it falls through to Metal,
then CPU.

### **Building the CUDA Shim (Optional but Recommended on Linux)**

The CUDA backend ships as a CUDA C++ shared library built with `nvcc` and
loaded at runtime via purego. To use the production-recommended CUDA path
on Linux (where it's the only GPU option):

```bash
# Prereqs: CUDA toolkit (nvcc), cuBLAS, cuDNN. On Ubuntu:
#   sudo apt install nvidia-cuda-toolkit libcudnn8-dev
#   (or use the runfile installer from developer.nvidia.com)

cd internal/compute/cuda
make                       # produces build/libcuda.so

# Point the loader at it for your run. The shim transitively links
# libcublas.so and libcudnn.so, so those need to be on LD_LIBRARY_PATH
# too — typically /usr/local/cuda/lib64 or /usr/lib/x86_64-linux-gnu.
LD_LIBRARY_PATH=$(pwd)/build:/usr/local/cuda/lib64 go test ./...
```

Device detection shells out to `nvidia-smi`; that needs to be on `PATH`.
Without the shim, `nvidia-smi`, or an NVIDIA driver loaded, `rnxa.NewEngine()`
falls through to CPU.

**Build status:** the shim is code-complete and compiles. End-to-end GPU
validation requires a Linux box with `nvcc` + an NVIDIA GPU. The dispatcher
and engine wiring are exercised by the existing rnxa test suite (the
fallthrough path is verified on macOS); the `internal/compute/cuda/cuda_test.go`
cases run on Linux with the shim built.

### **Basic Usage**

```go
package main

import (
    "context"
    "fmt"
    "github.com/xDarkicex/rnxa"
)

func main() {
    // Create compute engine (auto-detects best device)
    engine, err := rnxa.NewEngine()
    if err != nil {
        panic(err)
    }
    defer engine.Close()

    fmt.Printf("Using device: %s (%s)\n",
        engine.Device().Name, engine.Device().Platform)

    // Create tensors
    A := rnxa.NewTensor([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
    B := rnxa.NewTensor([]float64{7, 8, 9, 10, 11, 12}, 3, 2)

    // Perform matrix multiplication
    ctx := context.Background()
    C, err := engine.MatMul(ctx, A, B)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Result: %v\n", C.Data())
    // Output: [58 64 139 154]
}
```

### **Engine Lifecycle**

- `NewEngine()` is intended for one-time or lazy initialization. It probes
  available devices and creates a reusable engine for the best backend.
- Reuse a single engine across many operations when possible, then call
  `Close()` during shutdown or collection teardown.
- MPS- and Metal-backed engines serialize operations internally and return
  ordinary Go errors after `Close()` instead of panicking. CPU engines remain
  available as the fallback backend on all platforms.
- Callers that already work in float32 can build tensors with
  `NewTensorFromFloat32(...)`; `MatMul` uses the native float32 fast path
  automatically when both inputs are float32-backed - no downcast, no upcast.

---

## 📖 **Comprehensive Examples**

### **Neural Network Layer Forward Pass**

```go
func neuralLayerForward(engine rnxa.ComputeEngine,
                       inputs []float64,
                       weights [][]float64,
                       bias []float64) ([]float64, error) {
    ctx := context.Background()

    inputTensor := rnxa.NewTensor(inputs, 1, len(inputs))
    weightTensor := convertToTensor(weights)
    biasTensor := rnxa.NewTensor(bias, len(bias))

    matmulResult, err := engine.MatMul(ctx, inputTensor, weightTensor)
    if err != nil {
        return nil, err
    }

    biasResult, err := engine.VectorAdd(ctx, matmulResult, biasTensor)
    if err != nil {
        return nil, err
    }

    activated, err := engine.ReLU(ctx, biasResult)
    if err != nil {
        return nil, err
    }

    return activated.Data(), nil
}
```

### **Activation Function Comparison**

```go
func compareActivations() {
    engine, _ := rnxa.NewEngine()
    defer engine.Close()

    ctx := context.Background()
    input := rnxa.NewTensor([]float64{-2, -1, 0, 1, 2})

    activations := map[string]func(context.Context, *rnxa.Tensor) (*rnxa.Tensor, error){
        "ReLU":    engine.ReLU,
        "Sigmoid": engine.Sigmoid,
        "Tanh":    engine.Tanh,
    }

    fmt.Printf("Input: %v\n", input.Data())
    for name, fn := range activations {
        result, _ := fn(ctx, input)
        fmt.Printf("%s: %v\n", name, result.Data())
    }
}
```

### **Device Information and Benchmarking**

```go
func deviceInfo() {
    devices := rnxa.DetectDevices()

    for i, device := range devices {
        fmt.Printf("Device %d: %s\n", i, device.Name)
        fmt.Printf("  Platform: %s\n", device.Platform)
        fmt.Printf("  Cores: %d\n", device.Cores)
        fmt.Printf("  Memory: %.1fGB\n", float64(device.Memory)/1e9)
    }

    engine, _ := rnxa.NewEngine()
    defer engine.Close()

    memory := engine.Memory()
    fmt.Printf("\nActive Device Memory:\n")
    fmt.Printf("  Total: %.1fGB\n", float64(memory.Total)/1e9)
    fmt.Printf("  Available: %.1fGB\n", float64(memory.Available)/1e9)
}
```

On a Mac with the MPS shim built, `DetectDevices()` reports three devices:
the same physical GPU surfaced twice (once for MPS, once for Metal) plus the
CPU fallback. The dispatcher tries them in order; the first whose engine
reports `Available()` wins.

---

## 🔧 **Integration with ML Frameworks**

### **Framework-Agnostic Design**

rnxa exposes low-level tensor operations that any ML framework can use:

```go
type ComputeEngine interface {
    // Matrix operations
    MatMul(ctx context.Context, A, B *Tensor) (*Tensor, error)

    // Element-wise operations
    VectorAdd(ctx context.Context, A, B *Tensor) (*Tensor, error)
    VectorSub(ctx context.Context, A, B *Tensor) (*Tensor, error)
    VectorMul(ctx context.Context, A, B *Tensor) (*Tensor, error)

    // Activation functions
    ReLU(ctx context.Context, X *Tensor) (*Tensor, error)
    Sigmoid(ctx context.Context, X *Tensor) (*Tensor, error)
    Tanh(ctx context.Context, X *Tensor) (*Tensor, error)
    Softmax(ctx context.Context, X *Tensor) (*Tensor, error)

    // Reduction operations
    Sum(ctx context.Context, X *Tensor, axis int) (*Tensor, error)
    Mean(ctx context.Context, X *Tensor, axis int) (*Tensor, error)

    // Device management
    Device() Device
    Available() bool
    Memory() MemoryInfo
    Close() error
}
```

### **Integration Examples**

**With relux (Native Integration):**

```go
net, _ := relux.NewNetwork(
    relux.WithConfig(config),
    relux.WithAcceleration("auto"), // Uses rnxa (MPS or Metal) when available
)
```

**With Custom Frameworks:**

```go
type MyFramework struct {
    engine rnxa.ComputeEngine
}

func (f *MyFramework) ForwardPass(inputs []float64) []float64 {
    tensor := rnxa.NewTensor(inputs)
    result, _ := f.engine.ReLU(context.Background(), tensor)
    return result.Data()
}
```

---

## 📊 **Performance**

### **Backend Comparison (Apple Silicon, 1024×1024 MatMul)**

The exact numbers depend on the host and the input shape, but the ordering is
stable: MPS (when built) > Metal > CPU. The numbers below were measured on an
M-series Mac; the 24× CPU→Metal figure matches the table in earlier versions
of this README, with MPS sitting on top.

| Matrix Size | Pure Go CPU | rnxa (Metal) | rnxa (MPS) | Speedup vs. CPU |
|------------|-------------|--------------|------------|-----------------|
| 32×32      | 0.12ms      | 0.15ms       | 0.15ms     | ~1× (overhead-bound) |
| 128×128    | 2.1ms       | 0.3ms        | 0.25ms     | **~8× (MPS)** |
| 512×512    | 85ms        | 4.2ms        | 3.0ms      | **~28× (MPS)** |
| 1024×1024  | 680ms       | 28ms         | 19ms       | **~36× (MPS)** |

For shapes or platforms where MPS isn't competitive, the dispatcher picks Metal
automatically - no caller-side code change.

---

## 🛠️ **Advanced Configuration**

### **Explicit Device Selection**

```go
// Pick a specific device index from DetectDevices()
engine, err := rnxa.NewEngineWithDevice(1)

// DetectMetalAvailable reports whether any Metal device is reachable
if rnxa.IsMetalAvailable() {
    device := rnxa.GetBestDevice()
    fmt.Printf("Best device: %s\n", device.Name)
}
```

### **Error Handling and Fallbacks**

```go
func robustComputation(A, B *rnxa.Tensor) (*rnxa.Tensor, error) {
    engine, err := rnxa.NewEngine()
    if err != nil {
        return nil, fmt.Errorf("failed to create engine: %w", err)
    }
    defer engine.Close()

    ctx := context.Background()
    result, err := engine.MatMul(ctx, A, B)
    if err != nil {
        return nil, fmt.Errorf("computation failed: %w", err)
    }
    return result, nil
}
```

The backend ladder (MPS → Metal → CPU) is internal; if a higher-priority
backend fails to load (e.g. `libmps.dylib` not on `DYLD_LIBRARY_PATH`) the
dispatcher simply tries the next one.

---

## 🧪 **Testing and Validation**

```bash
# rnxa tests (CPU path is exercised by default)
go test -count=1 -race ./...

# MPS tests require the shim on the loader path
(cd internal/compute/mps && make)
DYLD_LIBRARY_PATH=$(pwd)/internal/compute/mps/build go test -count=1 -race ./...

# Benchmarks
go test -bench=. -benchmem

# Test specific operations
go test -run TestMatrixMultiplication -v
go test -run TestActivationFunctions -v
```

The mps subpackage's tests (`internal/compute/mps/`) cover `MatMul`,
`ReLU`, `Sigmoid`, `Tanh`, and `Softmax` against the real MPS kernel and
short-circuit with a clear error on non-darwin platforms.

---

## 🚧 **Roadmap**

### **Phase 1: Apple Silicon (Complete)**
- ✅ Metal compute kernels
- ✅ MPS matmul via `MPSMatrixMultiplication` (purego FFI, no CGO)
- ✅ Automatic backend selection (MPS → Metal → CPU)
- ✅ Float32 fast path
- ✅ Comprehensive activation functions
- ✅ Production-ready error handling

### **Phase 2: Linux (Code Complete, Awaiting Linux Validation)**
- ✅ **CUDA backend** - `libcuda.so` shim via purego, cuBLAS matmul +
  cuDNN activations / softmax. Device detection via `nvidia-smi`. All
  Go-side wiring (engine wrapper, dispatcher case, device list) is in
  place; the rnxa test sweep on darwin confirms the dispatcher routes
  correctly when the shim is missing (falls through to CPU).
  End-to-end GPU validation requires a Linux box with `nvcc` and an
  NVIDIA GPU to run `internal/compute/cuda/cuda_test.go`.
- 🔄 **oneDNN CPU backend on Linux** - purego-loaded `librnxa_cpu`; built
  and unit-tested on darwin, ready to land on Linux once a Linux build
  agent is wired up. The shim itself is platform-portable; only the
  oneDNN CMake build is platform-specific.
- 🔄 **Linux CI image** - one-line CUDA + oneDNN setup

### **Phase 3: Windows (Planned)**
- 🔮 **CUDA on Windows** - `cuda_engine_windows.go` + `device_windows.go`.
  The C ABI is portable; this is file additions + Windows nvcc in CI.
- 🔮 **DirectML fallback** - for non-NVIDIA GPUs

### **Phase 4: Advanced Features**
- 🔮 **Multi-GPU support** - distribute one tensor across MPS / CUDA
  devices. The `int32_t dev` ABI parameter is already plumbed; the
  per-call `dev==0` assertion in the CUDA shim is the only blocker.
- 🔮 **GPU memory pooling** - small `RnxaBufferPool` in the CUDA shim so
  the per-call `cudaMalloc` doesn't dominate tiny matmuls
- 🔮 **Mixed precision** - FP16/BF16 paths through MPS, Metal, and
  cuBLAS `cublasGemmEx` / cuDNN half-precision descriptors
- 🔮 **Custom kernels** - user-defined compute shaders for Metal

### **Phase 5: Ecosystem Integration**
- 🔮 **ONNX import** - load ONNX models and dispatch through rnxa
- 🔮 **Distributed training** - multi-node support
- 🔮 **ROCm / oneAPI** - AMD and Intel GPU support (same shim pattern,
  different ABI)

---

## 🤝 **Contributing**

We welcome contributions to make rnxa the premier ML acceleration framework
for Go!

### **Areas for Contribution**
- 🚀 **Performance Optimization** - CUDA kernels, memory management
- 🧪 **Testing** - Cross-platform testing, edge case validation
- 📚 **Documentation** - Tutorials, integration guides
- 🔍 **Debugging** - Profiling tools, performance analysis
- 🌐 **Platform Support** - AMD ROCm, Intel oneAPI

### **Development Setup**
```bash
git clone https://github.com/xDarkicex/rnxa.git
cd rnxa
go mod tidy
go test ./...

# With MPS
(cd internal/compute/mps && make)
DYLD_LIBRARY_PATH=$(pwd)/internal/compute/mps/build go test ./...
```

---

## 📋 **System Requirements**

### **macOS (Current Support)**
- **OS**: macOS 10.15+ (Catalina or later)
- **Hardware**: Apple Silicon (M1/M2/M3) or Intel Mac with Metal support
- **Tools**: Xcode Command Line Tools (xcode-select 2410+)
- **Go**: Version 1.25+

### **Linux (Code-Complete, Awaiting Linux Build Agent)**
- **OS**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **Hardware**: NVIDIA GPU with Compute Capability 6.0+ (Ampere / Ada / Hopper recommended)
- **CUDA**: Toolkit 11.8+ (provides `nvcc` + `libcublas.so` + cuDNN)
- **cuDNN**: Version 8.x
- **Drivers**: NVIDIA Driver 525.60+ (matches CUDA 12.x)
- **Tools**: `nvidia-smi` on `PATH` for device detection

### **Windows (Planned)**
- **OS**: Windows 10/11 (64-bit)
- **Hardware**: NVIDIA GPU with Compute Capability 6.0+
- **CUDA**: Version 11.0+
- **Visual Studio**: 2019 or later

---

## 📜 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 **Acknowledgments**

- **Apple Metal / MPS teams** - For Metal Performance Shaders and the
  `MPSMatrixMultiplication` kernel
- **Purego** - For clean FFI without CGO
- **oneDNN** - For the cross-vendor CPU primitives that the shim wraps
- **Go Team** - For building a language that makes systems programming approachable
- **relux Project** - The original motivation and testing ground for rnxa
- **Go ML Community** - For driving innovation in Go-based machine learning

---

## 📞 **Support & Community**

- 🐛 **Issues**: [GitHub Issues](https://github.com/xDarkicex/rnxa/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/xDarkicex/rnxa/discussions)
- 📧 **Email**: [gentry@xdarkicex.codes]
- 📖 **Documentation**: [docs.xdarkicex.codes](https://docs.xdarkicex.codes/rnxa)

---

<div align="center">

**⚡ Accelerate your Go ML workloads with rnxa ⚡**

*Built with ❤️ for the Go ML community*

</div>

---

*rnxa: Because your ML deserves better performance.*
