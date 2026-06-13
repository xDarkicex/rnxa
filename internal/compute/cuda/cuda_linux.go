//go:build linux
// +build linux

// cuda_linux.go — linux implementation of the CUDA backend using
// purego to load the pre-built libcuda.so (see Makefile). No cgo:
// the .cu file is compiled out-of-band, the .so is loaded at
// runtime, and the Go side declares the function signatures
// natively.
//
// Resilience: purego.RegisterFunc is wrapped in a recover() so a
// partial build (e.g. libcudnn missing on the host, so nvcc
// emitted no cuDNN symbols) leaves the relevant fn* slot nil
// rather than panicking. The corresponding Backend method then
// returns a clear "unavailable" error at call time. This is the
// same pattern cpu_purego.go uses for the oneDNN shim.

package cuda

import (
	"errors"
	"fmt"
	"sync"

	"github.com/ebitengine/purego"
)

const libName = "libcuda.so"

var (
	libOnce sync.Once
	lib     uintptr
	libErr  error
)

func loadLib() (uintptr, error) {
	libOnce.Do(func() {
		handle, err := purego.Dlopen(libName, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil || handle == 0 {
			libErr = fmt.Errorf("cuda: failed to load %s: %v (build via `make` in this directory)", libName, err)
			return
		}
		lib = handle
	})
	return lib, libErr
}

// bindCudaFns resolves every CUDA C ABI symbol exposed by the
// shim and binds it to a Go-side function pointer on b. Symbol
// resolution failures for non-critical symbols (e.g. cuDNN
// activations if the shim was built without cuDNN) leave the
// corresponding fn* slot nil; failures for critical symbols
// (init, matmul) return an error so New() can surface a clear
// "build the shim" message.
//
// Called once via sync.Once. safeBind wraps each RegisterFunc in
// recover() so a panic from purego (e.g. ABI mismatch on a
// missing symbol) does not take down the goroutine.
var bindOnce sync.Once

func bindCudaFns(b *Backend) error {
	var bindErr error
	bindOnce.Do(func() {
		handle, err := loadLib()
		if err != nil {
			bindErr = err
			return
		}

		// Critical symbols: missing → error out.
		critical := []struct {
			slot any
			sym  string
		}{
			{&b.fnDeviceCount, "rnxa_cuda_device_count"},
			{&b.fnDeviceName, "rnxa_cuda_device_name"},
			{&b.fnDeviceMem, "rnxa_cuda_device_memory"},
			{&b.fnMatmul, "rnxa_cuda_matmul"},
			{&b.fnEltwise, "rnxa_cuda_eltwise"},
			{&b.fnSoftmax, "rnxa_cuda_softmax"},
		}
		// Non-critical symbols: missing → leave nil, op returns
		// "unavailable" at call time.
		optional := []struct {
			slot any
			sym  string
		}{
			{&b.fnReLU, "rnxa_cuda_relu"},
			{&b.fnSigmoid, "rnxa_cuda_sigmoid"},
			{&b.fnTanh, "rnxa_cuda_tanh"},
		}

		// safeBind is purego.RegisterFunc with a recover() so a
		// missing or ABI-mismatched symbol panics inside the
		// recover, not the caller's goroutine.
		safeBind := func(slot any, sym string) (ok bool) {
			defer func() { _ = recover() }()
			cfn, e := purego.Dlsym(handle, sym)
			if e != nil || cfn == 0 {
				return false
			}
			purego.RegisterFunc(slot, cfn)
			return true
		}

		// The init symbol is special: it's called with no args
		// and returns the opaque context. We need it to bind
		// the ctx field; the bound DeviceCount / Matmul / etc.
		// functions take ctx as their first arg.
		var fnInit func() int64
		if !safeBind(&fnInit, "rnxa_cuda_init") {
			bindErr = fmt.Errorf("cuda: critical symbol %q not found in %s", "rnxa_cuda_init", libName)
			return
		}

		for _, s := range critical {
			if !safeBind(s.slot, s.sym) {
				bindErr = fmt.Errorf("cuda: critical symbol %q not found in %s", s.sym, libName)
				return
			}
		}
		// Optional symbols: silently leave nil if missing.
		for _, s := range optional {
			_ = safeBind(s.slot, s.sym)
		}

		// Now actually call the init symbol to get a context.
		ctx := fnInit()
		if ctx == 0 {
			bindErr = errors.New("cuda: no CUDA device (driver/runtime mismatch?)")
			return
		}
		// Verify the device count is non-zero so we can give a
		// precise error if the driver is loaded but no GPU is
		// visible (e.g. WSL with no GPU passthrough).
		if b.fnDeviceCount(ctx) == 0 {
			bindErr = errors.New("cuda: no NVIDIA GPU detected")
			return
		}
		b.ctx = ctx
	})
	return bindErr
}

func newCudaBackend() (*Backend, error) {
	b := &Backend{}
	if err := bindCudaFns(b); err != nil {
		return nil, err
	}
	return b, nil
}

func closeCudaBackend(b *Backend) error {
	if b == nil || b.ctx == 0 {
		return nil
	}
	// We don't have a separate release fn; the shim could grow
	// one but for first cut we let the OS reclaim on process
	// exit. The cuBLAS / cuDNN handles leak at most once per
	// process, which matches the cpu_purego Close semantics.
	b.ctx = 0
	return nil
}

func cudaMatmul(b *Backend, A, B []float32, M, N, K int) ([]float32, error) {
	if b.fnMatmul == nil {
		return nil, errors.New("cuda: matmul unavailable (build libcuda.so with cuBLAS)")
	}
	if len(A) != M*K {
		return nil, fmt.Errorf("cuda matmul: A length %d, expected %d", len(A), M*K)
	}
	if len(B) != K*N {
		return nil, fmt.Errorf("cuda matmul: B length %d, expected %d", len(B), K*N)
	}
	c := make([]float32, M*N)
	rc := b.fnMatmul(b.ctx, 0, &A[0], &B[0], &c[0], int64(M), int64(N), int64(K))
	if rc != 0 {
		return nil, fmt.Errorf("cuda matmul: kernel returned %d", rc)
	}
	return c, nil
}

func cudaEltwise(b *Backend, A, B []float32, n, op int) ([]float32, error) {
	if b.fnEltwise == nil {
		return nil, errors.New("cuda: eltwise unavailable (build libcuda.so)")
	}
	if len(A) < n || len(B) < n {
		return nil, fmt.Errorf("cuda eltwise: A/B length %d/%d, need %d", len(A), len(B), n)
	}
	c := make([]float32, n)
	rc := b.fnEltwise(b.ctx, 0, int32(op), &A[0], &B[0], &c[0], int64(n))
	if rc != 0 {
		return nil, fmt.Errorf("cuda eltwise(op=%d): kernel returned %d", op, rc)
	}
	return c, nil
}

func cudaReLU(b *Backend, X []float32) ([]float32, error) {
	return runUnary(b, "relu", X, b.fnReLU)
}

func cudaSigmoid(b *Backend, X []float32) ([]float32, error) {
	return runUnary(b, "sigmoid", X, b.fnSigmoid)
}

func cudaTanh(b *Backend, X []float32) ([]float32, error) {
	return runUnary(b, "tanh", X, b.fnTanh)
}

func cudaSoftmax(b *Backend, X []float32) ([]float32, error) {
	if b.fnSoftmax == nil {
		return nil, errors.New("cuda: softmax unavailable (build libcuda.so with cuDNN)")
	}
	if len(X) == 0 {
		return []float32{}, nil
	}
	xCopy := make([]float32, len(X))
	copy(xCopy, X)
	y := make([]float32, len(X))
	rc := b.fnSoftmax(b.ctx, 0, &xCopy[0], &y[0], int64(len(X)))
	if rc != 0 {
		return nil, fmt.Errorf("cuda softmax: kernel returned %d", rc)
	}
	return y, nil
}

func runUnary(b *Backend, name string, X []float32,
	fn func(int64, int32, *float32, *float32, int64) int32) ([]float32, error) {
	if fn == nil {
		return nil, fmt.Errorf("cuda: %s unavailable (build libcuda.so with cuDNN)", name)
	}
	if len(X) == 0 {
		return []float32{}, nil
	}
	xCopy := make([]float32, len(X))
	copy(xCopy, X)
	y := make([]float32, len(X))
	rc := fn(b.ctx, 0, &xCopy[0], &y[0], int64(len(X)))
	if rc != 0 {
		return nil, fmt.Errorf("cuda %s: kernel returned %d", name, rc)
	}
	return y, nil
}
