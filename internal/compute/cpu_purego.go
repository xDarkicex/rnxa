// Package compute provides the CPU backend for rnxa via purego
// FFI into a C++ shared library that wraps oneDNN. The shared
// library (`librnxa_cpu.so` / `.dylib` / `.dll`) is built by
// `internal/compute/cpu_shim/CMakeLists.txt` from a thin C ABI over
// oneDNN primitives.
//
// The intent is end-game CPU performance without CGO: purego calls
// the shared library at runtime, so Go compilation stays fast and
// cross-platform. The CPU backend is the only CPU path — if the
// library fails to load, `NewEngine` reports the engine as
// unavailable rather than falling back to a slow pure-Go matmul.
//
// Build tags live in the per-platform files (cpu_purego_darwin.go,
// cpu_purego_linux.go, cpu_purego_windows.go); they only differ in
// the library name passed to Dlopen. The shared body is in this
// file and references platform symbols provided by the tagged
// siblings.
package compute

import (
	"context"
	"fmt"
	"math"
	"sync"

	"github.com/ebitengine/purego"

	"github.com/xDarkicex/rnxa"
)

// cpuOp enumerates the vector / elementwise operations exported by
// the C++ shim. The shim dispatches to dnnl::eltwise with the
// appropriate algorithm tag.
type cpuOp int32

const (
	cpuOpAdd cpuOp = iota
	cpuOpSub
	cpuOpMul
	cpuOpReLU
	cpuOpSigmoid
	cpuOpTanh
)

// cpuPuregoEngine is the CPU backend driven by purego + oneDNN.
type cpuPuregoEngine struct {
	device rnxa.Device

	// lib is a handle returned by purego.Dlopen. nil means the
	// library failed to load and Available() will return false.
	lib uintptr

	mu     sync.Mutex
	closed bool

	// C function bindings. Registered in newCPUPuregoEngine via
	// purego.RegisterFunc. nil if registration failed.
	fnMatmulF64 func(A, B, C *float64, M, N, K int64) int32
	fnMatmulF32 func(A, B, C *float32, M, N, K int64) int32
	fnVectorOp  func(op int32, A, B, C *float64, n int64) int32
	fnSoftmax   func(X, Y *float64, n, axis int64) int32
	fnReduceSum func(X, Y *float64, n int64) int32
}

// newCPUPuregoEngine loads librnxa_cpu via purego and binds the
// exported functions. If loading fails, the returned engine has
// lib == 0 and Available() == false; callers (notably NewEngine)
// detect this and skip the CPU path.
func newCPUPuregoEngine() *cpuPuregoEngine {
	lib, err := purego.Dlopen(cpuLibName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil || lib == 0 {
		return &cpuPuregoEngine{device: rnxa.Device{
			ID: -1, Name: "CPU (offline)", Type: rnxa.CPU, Platform: "CPU",
		}}
	}

	e := &cpuPuregoEngine{
		lib:    lib,
		device: rnxa.Device{ID: -1, Name: "CPU", Type: rnxa.CPU, Platform: "CPU"},
	}

	// RegisterFunc panics if the symbol is missing; catch via recover
	// so a partial build (some symbols compiled out) doesn't crash
	// the process. The recovered engine still has lib != 0 but its
	// fn* slots stay nil; the relevant method falls back to a clear
	// error at call time.
	bind := func(slot any, sym string) {
		defer func() { _ = recover() }()
		cfn, derr := purego.Dlsym(lib, sym)
		if derr != nil || cfn == 0 {
			return
		}
		purego.RegisterFunc(slot, cfn)
	}

	bind(&e.fnMatmulF64, "rnxa_matmul_f64")
	bind(&e.fnMatmulF32, "rnxa_matmul_f32")
	bind(&e.fnVectorOp, "rnxa_vector_op")
	bind(&e.fnSoftmax, "rnxa_softmax")
	bind(&e.fnReduceSum, "rnxa_reduce_sum")

	return e
}

func (e *cpuPuregoEngine) errUnavail(op string) error {
	return fmt.Errorf("cpu: %s unavailable (build librnxa_cpu)", op)
}

func (e *cpuPuregoEngine) MatMul(ctx context.Context, A, B *rnxa.Tensor) (*rnxa.Tensor, error) {
	if e.fnMatmulF64 == nil || e.fnMatmulF32 == nil {
		return nil, e.errUnavail("MatMul")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("cpu engine is closed")
	}
	if A.DType() == rnxa.Float32 && B.DType() == rnxa.Float32 {
		shape := A.Shape()
		M, K := int64(shape[0]), int64(shape[1])
		Bshape := B.Shape()
		N := int64(Bshape[1])
		result := rnxa.ZerosFloat32(int(M), int(N))
		if rc := e.fnMatmulF32(&A.Data32()[0], &B.Data32()[0], &result.Data32()[0], M, N, K); rc != 0 {
			return nil, fmt.Errorf("cpu matmul f32 failed: rc=%d", rc)
		}
		return result, nil
	}
	shape := A.Shape()
	M, K := int64(shape[0]), int64(shape[1])
	Bshape := B.Shape()
	N := int64(Bshape[1])
	result := rnxa.Zeros(int(M), int(N))
	if rc := e.fnMatmulF64(&A.Data()[0], &B.Data()[0], &result.Data()[0], M, N, K); rc != 0 {
		return nil, fmt.Errorf("cpu matmul f64 failed: rc=%d", rc)
	}
	return result, nil
}

func (e *cpuPuregoEngine) VectorAdd(ctx context.Context, A, B *rnxa.Tensor) (*rnxa.Tensor, error) {
	return e.eltwise(ctx, A, B, cpuOpAdd)
}

func (e *cpuPuregoEngine) VectorSub(ctx context.Context, A, B *rnxa.Tensor) (*rnxa.Tensor, error) {
	return e.eltwise(ctx, A, B, cpuOpSub)
}

func (e *cpuPuregoEngine) VectorMul(ctx context.Context, A, B *rnxa.Tensor) (*rnxa.Tensor, error) {
	return e.eltwise(ctx, A, B, cpuOpMul)
}

func (e *cpuPuregoEngine) eltwise(ctx context.Context, A, B *rnxa.Tensor, op cpuOp) (*rnxa.Tensor, error) {
	if e.fnVectorOp == nil {
		return nil, e.errUnavail("vector op")
	}
	if A.Size() != B.Size() {
		return nil, fmt.Errorf("cpu vector op: size mismatch %d != %d", A.Size(), B.Size())
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("cpu engine is closed")
	}
	n := int64(A.Size())
	result := rnxa.Zeros(A.Shape()...)
	if rc := e.fnVectorOp(int32(op), &A.Data()[0], &B.Data()[0], &result.Data()[0], n); rc != 0 {
		return nil, fmt.Errorf("cpu vector op %d failed: rc=%d", op, rc)
	}
	return result, nil
}

func (e *cpuPuregoEngine) ReLU(ctx context.Context, X *rnxa.Tensor) (*rnxa.Tensor, error) {
	return e.unary(ctx, X, cpuOpReLU)
}

func (e *cpuPuregoEngine) Sigmoid(ctx context.Context, X *rnxa.Tensor) (*rnxa.Tensor, error) {
	return e.unary(ctx, X, cpuOpSigmoid)
}

func (e *cpuPuregoEngine) Tanh(ctx context.Context, X *rnxa.Tensor) (*rnxa.Tensor, error) {
	return e.unary(ctx, X, cpuOpTanh)
}

// unary wraps a single-input elementwise op. oneDNN's eltwise has
// both unary and binary forms; the shim's rnxa_vector_op dispatches
// unary by passing A == B (the C++ side treats equal pointers as
// "use the unary path for op").
func (e *cpuPuregoEngine) unary(ctx context.Context, X *rnxa.Tensor, op cpuOp) (*rnxa.Tensor, error) {
	if e.fnVectorOp == nil {
		return nil, e.errUnavail("unary op")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("cpu engine is closed")
	}
	n := int64(X.Size())
	result := rnxa.Zeros(X.Shape()...)
	if rc := e.fnVectorOp(int32(op), &X.Data()[0], &X.Data()[0], &result.Data()[0], n); rc != 0 {
		return nil, fmt.Errorf("cpu unary op %d failed: rc=%d", op, rc)
	}
	return result, nil
}

func (e *cpuPuregoEngine) Softmax(ctx context.Context, X *rnxa.Tensor) (*rnxa.Tensor, error) {
	if e.fnSoftmax == nil {
		return nil, e.errUnavail("Softmax")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("cpu engine is closed")
	}
	// The shim takes n = total elements and a single axis=0 to
	// reduce across. Axis-aware softmax for ndim > 1 is a TODO.
	result := rnxa.Zeros(X.Shape()...)
	if rc := e.fnSoftmax(&X.Data()[0], &result.Data()[0], int64(X.Size()), 0); rc != 0 {
		return nil, fmt.Errorf("cpu softmax failed: rc=%d", rc)
	}
	return result, nil
}

func (e *cpuPuregoEngine) Sum(ctx context.Context, X *rnxa.Tensor, axis int) (*rnxa.Tensor, error) {
	// The shim's reduce-sum is currently a flat total; for axis-aware
	// reductions we delegate to the rnxa package's helper until the
	// shim exposes a per-axis reduce.
	return rnxa.Sum(X, axis)
}

func (e *cpuPuregoEngine) Mean(ctx context.Context, X *rnxa.Tensor, axis int) (*rnxa.Tensor, error) {
	return rnxa.Mean(X, axis)
}

func (e *cpuPuregoEngine) Device() rnxa.Device  { return e.device }
func (e *cpuPuregoEngine) Available() bool       { return e.lib != 0 }
func (e *cpuPuregoEngine) Memory() rnxa.MemoryInfo { return rnxa.MemoryInfo{} }

func (e *cpuPuregoEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.closed = true
	// The shared library stays loaded for the process lifetime;
	// closing the handle would invalidate the bound function pointers
	// in any concurrent goroutine. The OS reclaims it on exit.
	return nil
}

// cpuLibName is set in a per-platform build-tagged file:
//   - cpu_purego_darwin.go → "librnxa_cpu.dylib"
//   - cpu_purego_linux.go   → "librnxa_cpu.so"
//   - cpu_purego_windows.go → "rnxa_cpu.dll"
var cpuLibName string

// silence the math import for parity with future kernels that may
// compute activations in Go when the shim is unavailable.
var _ = math.Inf
